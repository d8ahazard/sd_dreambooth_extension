import os
import random
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept
from extensions.sd_dreambooth_extension.dreambooth.utils import get_images
from modules import images


class SampleData:
    def __init__(self, prompt: str, concept: Concept):
        self.prompt = prompt
        self.negative_prompt = concept.save_sample_negative_prompt
        self.seed = concept.sample_seed
        self.steps = concept.save_infer_steps
        self.scale = concept.save_guidance_scale
        self.n_samples = concept.n_save_sample


class TrainingData:
    def __init__(self, image: str, prompt, tokens):
        self.image = image
        self.prompt = prompt
        self.tokens = tokens


class ConceptData:
    def __init__(self, name: str, instance_images: [TrainingData], class_images: [TrainingData],
                 sample_prompts: List[str], concept: Concept):
        """
        A class for holding all the info on a concept without needing a bunch of different dictionaries and stuff
        @param name: Token for our subject.
        @param instance_images: A list of instance images or whatever
        @param class_images: A list of class images and prompts
        @param sample_prompts: A list of sample prompts
        """
        self.name = name
        self.instance_images = instance_images
        self.class_images = class_images
        self.sample_prompts = sample_prompts
        self.concept = concept
        self.instance_indices = []
        self.class_indices = []
        self.sample_indices = []
        self.with_prior = len(self.class_images) > 0
        self.length = len(self.instance_images)

    def has_prior(self):
        return len(self.class_images) > 0

    def get_instance_image(self) -> TrainingData:
        if len(self.instance_indices) <= 0:
            img_idx = 0
            for _ in self.instance_images:
                self.instance_indices.append(img_idx)
                img_idx += 1
            random.shuffle(self.instance_indices)
        index = self.instance_indices.pop()
        return self.instance_images[index]

    def get_class_image(self) -> TrainingData:
        if len(self.class_indices) <= 0:
            img_idx = 0
            for _ in self.class_images:
                self.class_indices.append(img_idx)
                img_idx += 1
            random.shuffle(self.class_indices)
        index = self.class_indices.pop()
        return self.class_images[index]

    def get_sample_prompt(self) -> str:
        if len(self.sample_indices) <= 0:
            img_idx = 0
            for _ in self.sample_prompts:
                self.sample_indices.append(img_idx)
                img_idx += 1
            random.shuffle(self.sample_indices)
        index = self.sample_indices.pop()
        return self.sample_prompts[index]


class SuperDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            concepts_list: [Concept],
            tokenizer,
            size=512,
            center_crop=False,
            pad_tokens=False,
            hflip=False,
            max_token_length=75,
            lifetime_steps=-1,
            shuffle_tags=False
    ):
        from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import FilenameTextGetter
        self.concepts = []
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.pad_tokens = pad_tokens
        self.max_token_length = max_token_length
        self.tokenizer_max_length = self.tokenizer.model_max_length if max_token_length == 75 else max_token_length + 2
        self.text_getter = FilenameTextGetter(shuffle_tags)
        self.lifetime_steps = lifetime_steps
        self.current_concept = 0
        self.shuffle_tags = shuffle_tags
        needs_crop = False
        total_images = 0

        for concept_dict in concepts_list:
            # Parse concept info
            concept = Concept(input_dict=concept_dict)
            concept_with_prior = concept.num_class_images > 0
            num_class_images = concept.num_class_images
            instance_prompt = concept.instance_prompt
            class_prompt = concept.class_prompt
            instance_token = concept.instance_token
            class_token = concept.class_token

            # Get a mostly pointless unique key for each concept
            if "[filewords]" in instance_prompt and instance_token != "":
                concept_key = instance_token
            else:
                concept_key = images.sanitize_filename_part(instance_prompt)

            instance_data = []
            concept_images = get_images(concept.instance_data_dir)
            for file in concept_images:
                try:
                    img = Image.open(file)
                    if img.width != size or img.height != size:
                        needs_crop = True
                except Exception as e:
                    print(f"Exception parsing instance image: {e}")
                    continue
                file_text = self.text_getter.read_text(file)
                file_prompt = self.text_getter.create_text(instance_prompt, file_text, instance_token,
                                                           class_token, False)
                prompt_tokens = self.tokenize(file_prompt)
                instance_data.append(TrainingData(file, file_prompt, prompt_tokens))
                # Create a dictionary for each concept, one with the images, and another with the max training steps
                random.shuffle(instance_data)

                class_data = []
                if concept_with_prior:
                    concept_images = get_images(concept.class_data_dir)
                    for file in concept_images:
                        try:
                            img = Image.open(file)
                            if img.width != size or img.height != size:
                                needs_crop = True
                        except Exception as e:
                            print(f"Exception parsing instance image: {e}")
                            continue

                        file_text = self.text_getter.read_text(file)
                        file_prompt = self.text_getter.create_text(class_prompt, file_text, instance_token,
                                                                   class_token, True)
                        prompt_tokens = self.tokenize(file_prompt)
                        class_data.append(TrainingData(file, file_prompt, prompt_tokens))

                    random.shuffle(class_data)
                    if len(class_data) > num_class_images:
                        class_data = class_data[:num_class_images]

                # Generate sample prompts for concept
                samples = self.generate_sample_prompts(instance_data, concept)
                print(f"Concept {concept_key} has {len(samples)} sample prompts.")
                # One pretty wrapper for the whole concept
                concept_data = ConceptData(concept_key, instance_data, class_data, samples, concept)
                self.concepts.append(concept_data)
                total_images += concept_data.length

        # We do this above when creating the dict of instance images
        self._length = total_images
        transforms_list = []

        if hflip:
            transforms_list.append(transforms.RandomHorizontalFlip(0.5 * hflip))

        if needs_crop:
            transforms_list.append(transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size))
            transforms_list.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR))

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize([0.5], [0.5]))

        # Less transforms should be more better, right?
        self.image_transforms = transforms.Compose(transforms_list)

    def generate_sample_prompts(self, instance_data: [TrainingData], concept: Concept):
        prompts = []
        # Try populating prompts from template file if specified
        if concept.save_sample_template != "":
            if os.path.exists(concept.save_sample_template):
                print(f"Loading sample strings from {concept.save_sample_template}")
                file = open(concept.save_sample_template, 'r')
                prompts = file.readlines()

        # If no prompts, get them from instance data
        if len(prompts) == 0:
            if concept.save_sample_prompt == "" or "[filewords]" in concept.save_sample_prompt:
                for data in instance_data:
                    prompts.append(data.prompt)
            else:
                prompts.append(concept.save_sample_prompt)
        else:
            to_replace = ["[filewords]", "[name]"]
            out_prompts = []
            for prompt in prompts:
                for replace in to_replace:
                    if replace in prompt:
                        prompt = prompt.replace(replace, f"{concept.instance_token} {concept.class_token}")
                if "," in prompt and self.shuffle_tags:
                    prompt_tags = prompt.split(",")
                    first_tag = prompt_tags.pop(0)
                    # Shuffle tags in sample prompt
                    if len(prompt_tags) > 1:
                        random.shuffle(prompt_tags)
                        prompt_tags.insert(0, first_tag)
                out_prompts.append(prompt)
            prompts = out_prompts
            random.shuffle(prompts)
        return prompts

    def tokenize(self, text):
        if not self.pad_tokens:
            input_ids = self.tokenizer(text, padding="max_length", truncation=True,
                                       max_length=self.tokenizer.model_max_length, return_tensors="pt").input_ids
            return input_ids

        input_ids = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.tokenizer_max_length,
                                   return_tensors="pt").input_ids
        if self.tokenizer_max_length > self.tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            # v1
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                for i in range(1, self.tokenizer_max_length - self.tokenizer.model_max_length + 2, self.tokenizer.model_max_length - 2):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i: i + self.tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2
                for i in range(1, self.tokenizer_max_length - self.tokenizer.model_max_length + 2, self.tokenizer.model_max_length - 2):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i: i + self.tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)

                    if ids_chunk[-2] != self.tokenizer.eos_token_id and ids_chunk[-2] != self.tokenizer.pad_token_id:
                        ids_chunk[-1] = self.tokenizer.eos_token_id

                    if ids_chunk[1] == self.tokenizer.pad_token_id:
                        ids_chunk[1] = self.tokenizer.eos_token_id
                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)

        return input_ids

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # Get the current concepts index and corresponding key for our dicts
        c_index = self.current_concept
        if c_index >= len(self.concepts):
            print("Invalid index specified.")
            c_index = 0

        concept_data = self.concepts[c_index]
        instance_data = concept_data.get_instance_image()

        instance_image = Image.open(instance_data.image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt"] = instance_data.prompt
        example["instance_prompt_ids"] = instance_data.tokens

        if concept_data.has_prior():
            class_data = concept_data.get_class_image()
            class_image = Image.open(class_data.image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = class_data.prompt
            example["class_prompt_ids"] = class_data.tokens

        # Rotate to the next concept
        self.current_concept += 1
        if self.current_concept >= len(self.concepts):
            self.current_concept = 0
        return example

    def get_sample_prompts(self) -> [SampleData]:
        prompts = []
        for concept in self.concepts:
            for n in range(concept.concept.n_save_sample):
                s_data = SampleData(concept.get_sample_prompt(), concept.concept)
                prompts.append(s_data)
        random.shuffle(prompts)
        return prompts
