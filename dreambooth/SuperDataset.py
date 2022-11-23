import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

from extensions.sd_dreambooth_extension.dreambooth.dreambooth import is_image, list_features
from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import FilenameTextGetter
from modules import images


class SuperDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            concepts_list,
            tokenizer,
            file_prompt_contents,
            with_prior_preservation=True,
            size=512,
            center_crop=False,
            num_class=None,
            pad_tokens=False,
            hflip=False,
            max_token_length=75,
            shuffle=False,
            lifetime_steps=-1,
            max_train_steps=0
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation
        self.pad_tokens = pad_tokens
        self.shuffle = shuffle
        self.file_prompt_contents = file_prompt_contents
        self.instance_dict = {}
        self.class_dict = {}
        self.instance_counts = {}
        self.instance_images_randomizer_dict = {}
        self.class_images_randomizer_dict = {}
        pil_features = list_features()
        self.max_token_length = max_token_length
        self.tokenizer_max_length = self.tokenizer.model_max_length if max_token_length == 75 else max_token_length + 2
        self.text_getter = FilenameTextGetter()
        self.lifetime_steps = lifetime_steps
        self.max_steps = max_train_steps
        self.num_concepts = len(concepts_list)
        self.concepts_index = 0
        total_images = 0

        for concept in concepts_list:
            print(f"Loading concept: {concept}")
            concept_with_prior = with_prior_preservation
            num_class_images = num_class
            instance_prompt = concept["instance_prompt"]
            if "[filewords]" in instance_prompt:
                concept_key = concept["instance_token"]
            else:
                concept_key = images.sanitize_filename_part(instance_prompt)
            max_steps = max_train_steps
            if "max_steps" in concept:
                max_steps = concept["max_steps"]
                # Ensure we don't over train our concept
                if lifetime_steps >= max_steps:
                    print(f"Excluding concept {instance_prompt}")
                    max_steps = 0
                else:
                    print(f"Setting max_steps for {instance_prompt} to {max_steps}")
                    max_steps -= lifetime_steps

            # Only append things to the dict if we still want to train them
            if max_steps > 0:
                if max_steps > total_images:
                    total_images = max_steps
                # We can now override num_class_images per concept
                if "num_class_images" in concept:
                    num_class_images = concept["num_class_images"]
                    if num_class_images > 0:
                        concept_with_prior = True

                inst_img_path = [(x, instance_prompt, concept["instance_token"], concept["class_token"],
                                  self.text_getter.read_text(x)) for x in
                                 Path(concept["instance_data_dir"]).iterdir() if is_image(x, pil_features)]
                # Create a dictionary for each concept, one with the images, and another with the max training steps
                random.shuffle(inst_img_path)

                self.instance_dict[concept_key] = inst_img_path
                self.instance_counts[concept_key] = max_steps

                if concept_with_prior:
                    class_img_path = [(x, concept["class_prompt"], concept["instance_token"], concept["class_token"],
                                       self.text_getter.read_text(x)) for x in
                                      Path(concept["class_data_dir"]).iterdir() if is_image(x, pil_features)]
                    out = class_img_path[:num_class_images]
                    random.shuffle(out)
                    self.class_dict[concept_key] = out

        # We do this above when creating the dict of instance images
        self._length = total_images

        self.image_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5 * hflip),
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def tokenize(self, text):
        if not self.pad_tokens:
            input_ids = self.tokenizer(text, padding="do_not_pad", truncation=True,
                                       max_length=self.tokenizer.model_max_length).input_ids
            return input_ids

        input_ids = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.tokenizer_max_length,
                                   return_tensors="pt").input_ids
        if self.tokenizer_max_length > self.tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            for i in range(1, self.tokenizer_max_length - self.tokenizer.model_max_length + 2,
                           self.tokenizer.model_max_length - 2):
                iid = (input_ids[0].unsqueeze(0),
                       input_ids[i:i + self.tokenizer.model_max_length - 2],
                       input_ids[-1].unsqueeze(0))
                iid = torch.cat(iid)
                iids_list.append(iid)
            input_ids = torch.stack(iids_list)  # 3,77

        return input_ids

    def __len__(self):
        return self._length

    def _get_random_image_index(self, concepts_key: str, is_instance: bool):
        rand_dict = self.instance_images_randomizer_dict if is_instance else self.class_images_randomizer_dict
        if concepts_key not in rand_dict:
            rand_dict[concepts_key] = []
        if len(rand_dict[concepts_key]) == 0:
            keys = self.instance_dict[concepts_key] if is_instance else self.class_dict[concepts_key]
            values = []
            for key in keys:
                values.append(key)
            rand_dict[concepts_key] = values
        random_index = random.randint(0, len(rand_dict[concepts_key]) - 1)
        result = rand_dict[concepts_key].pop(random_index)
        if is_instance:
            self.instance_images_randomizer_dict = rand_dict
        else:
            self.class_images_randomizer_dict = rand_dict
        return result

    def __getitem__(self, index):
        example = {}
        # Get the current concepts index and corresponding key for our dicts
        c_index = self.concepts_index
        concept_key = list(self.instance_dict)[c_index]
        # Get current dict of values for stuff
        instance_idx = self._get_random_image_index(concept_key, True)
        class_images = []
        if concept_key in self.class_dict:
            class_images = self.class_dict[concept_key]

        instance_path, instance_prompt, instance_token, class_token, instance_text = instance_idx
        instance_image = Image.open(instance_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt"] = self.text_getter.create_text(instance_prompt,
                                                                  instance_text, instance_token, class_token,
                                                                  self.file_prompt_contents,
                                                                  False)
        example["instance_prompt_ids"] = self.tokenize(example["instance_prompt"])

        if self.with_prior_preservation and len(class_images):
            class_idx = self._get_random_image_index(concept_key, False)
            class_path, class_prompt, instance_token, class_token, class_text = class_idx
            class_image = Image.open(class_path)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.text_getter.create_text(class_prompt, class_text, instance_token,
                                                                   class_token, self.file_prompt_contents)
            example["class_prompt_ids"] = self.tokenize(example["class_prompt"])

        # Here's where the "Super" comes in.
        concept_count = self.instance_counts[concept_key]
        concept_count -= 1
        if concept_count <= 0:
            print(f"Popping concept: {concept_key}")
            self.instance_dict.pop(concept_key)
            self.class_dict.pop(concept_key)
            self.instance_counts.pop(concept_key)
        else:
            self.instance_counts[concept_key] = concept_count

        # Rotate to the next concept
        self.concepts_index += 1
        if self.concepts_index >= len(self.instance_dict):
            self.concepts_index = 0
        return example
