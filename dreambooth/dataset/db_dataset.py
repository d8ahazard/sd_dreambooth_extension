import copy
import os.path
import random
import traceback
from typing import List, Tuple, Union

import safetensors.torch
import torch.utils.data
from torchvision.transforms import transforms
from transformers import CLIPTokenizer

from dreambooth import shared
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.shared import status
from dreambooth.utils.image_utils import make_bucket_resolutions, \
    closest_resolution, shuffle_tags, open_and_trim
from dreambooth.utils.text_utils import build_strict_tokens
from helpers.mytqdm import mytqdm


class DbDataset(torch.utils.data.Dataset):
    """
    Dataset for handling training data
    """

    def __init__(
            self,
            batch_size: int,
            instance_prompts: List[PromptData],
            class_prompts: List[PromptData],
            tokens: List[Tuple[str, str]],
            tokenizer: Union[CLIPTokenizer, None],
            resolution: int,
            hflip: bool,
            shuffle_tags: bool,
            strict_tokens: bool,
            not_pad_tokens: bool,
            debug_dataset: bool,
            model_dir: str
    ) -> None:
        super().__init__()
        self.batch_indices = []
        self.batch_samples = []
        self.cache_dir = os.path.join(model_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        print("Init dataset!")
        # A dictionary of string/latent pairs matching image paths
        self.latents_cache = {}
        # A dictionary of string/input_ids(s) pairs matching image paths
        self.caption_cache = {}
        # A dictionary of (int, int) / List[(string, string)] of resolutions and the corresponding image paths/captions
        self.train_dict = {}
        # A dictionary of (int, int) / List[(string, string)] of resolutions and the corresponding image paths/captions
        self.class_dict = {}
        # A mash-up of the class/train dicts that is perfectly fair and balanced.
        self.sample_dict = {}
        # This is where we just keep a list of everything for batching
        self.sample_cache = []
        # This is just a list of the sample names that we can use to find where in the cache an image is
        self.sample_indices = []
        # All of the available bucket resolutions
        self.resolutions = []
        # Currently active resolution
        self.active_resolution = (0, 0)
        # The currently active image index while iterating
        self.image_index = 0
        # Total len of the dataloader
        self._length = 0
        self.batch_size = batch_size
        self.batch_sampler = torch.utils.data.BatchSampler(self, batch_size, drop_last=True)
        self.train_img_data = instance_prompts
        self.class_img_data = class_prompts
        self.num_train_images = len(self.train_img_data)
        self.num_class_images = len(self.class_img_data)

        self.tokenizer = tokenizer
        self.resolution = resolution
        self.debug_dataset = debug_dataset
        self.shuffle_tags = shuffle_tags
        self.not_pad_tokens = not_pad_tokens
        self.strict_tokens = strict_tokens
        self.tokens = tokens
        self.vae = None
        self.cache_latents = False
        flip_p = 0.5 if hflip else 0.0
        if hflip:
            self.image_transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(flip_p),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            self.image_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

    def load_image(self, image_path, caption, res):
        if self.debug_dataset:
            image = os.path.splitext(image_path)
            input_ids = caption
        else:
            if self.cache_latents:
                image = self.latents_cache[image_path]
            else:
                img = open_and_trim(image_path, res, False)
                image = self.image_transforms(img)
            if self.shuffle_tags:
                caption, input_ids = self.cache_caption(image_path, caption)
            else:
                input_ids = self.caption_cache[image_path]
        return image, input_ids

    def cache_latent(self, image_path, res):
        if self.vae is not None:
            image = open_and_trim(image_path, res, False)
            img_tensor = self.image_transforms(image)
            img_tensor = img_tensor.unsqueeze(0).to(device=self.vae.device, dtype=self.vae.dtype)
            latents = self.vae.encode(img_tensor).latent_dist.sample().squeeze(0).to("cpu")
            self.latents_cache[image_path] = latents

    def cache_caption(self, image_path, caption):
        input_ids = None
        auto_add_special_tokens = False if self.strict_tokens else True
        if self.tokenizer is not None and (image_path not in self.caption_cache or self.debug_dataset):
            if self.shuffle_tags:
                caption = shuffle_tags(caption)
            if self.strict_tokens:
                caption = build_strict_tokens(caption, self.tokenizer.bos_token, self.tokenizer.eos_token)
            if self.not_pad_tokens:
                input_ids = self.tokenizer(caption, padding=True, truncation=True,
                                           add_special_tokens=auto_add_special_tokens,
                                           return_tensors="pt").input_ids
            else:
                input_ids = self.tokenizer(caption, padding='max_length', truncation=True,
                                           add_special_tokens=auto_add_special_tokens,
                                           return_tensors='pt').input_ids
            if not self.shuffle_tags:
                self.caption_cache[image_path] = input_ids
        return caption, input_ids

    def make_buckets_with_caching(self, vae):
        self.vae = vae
        self.cache_latents = vae is not None
        state = f"Preparing Dataset ({'With Caching' if self.cache_latents else 'Without Caching'})"
        print(state)
        status.textinfo = state

        # Create a list of resolutions
        bucket_resos = make_bucket_resolutions(self.resolution)
        self.train_dict = {}

        def sort_images(img_data: List[PromptData], resos, target_dict, is_class_img):
            for prompt_data in img_data:
                path = prompt_data.src_image
                image_width, image_height = prompt_data.resolution
                cap = prompt_data.prompt
                reso = closest_resolution(image_width, image_height, resos)
                concept_idx = prompt_data.concept_index
                # Append the concept index to the resolution, and boom, we got ourselves split concepts.
                di = (*reso, concept_idx)
                target_dict.setdefault(di, []).append((path, cap, is_class_img))

        sort_images(self.train_img_data, bucket_resos, self.train_dict, False)
        sort_images(self.class_img_data, bucket_resos, self.class_dict, True)

        def cache_images(images, reso, p_bar):
            for img_path, cap, is_prior in images:
                try:
                    # If the image is not in the "precache",cache it
                    if img_path not in latents_cache:
                        if self.cache_latents and not self.debug_dataset:
                            self.cache_latent(img_path, reso)
                    # Otherwise, load it from existing cache
                    else:
                        self.latents_cache[img_path] = latents_cache[img_path]
                    if not self.shuffle_tags:
                        self.cache_caption(img_path, cap)
                    self.sample_indices.append(img_path)
                    self.sample_cache.append((img_path, cap, is_prior))
                    p_bar.update()
                except Exception as e:
                    traceback.print_exc()
                    print(f"Exception caching: {img_path}: {e}")
                    if img_path in self.caption_cache:
                        del self.caption_cache[img_path]
                    if (img_path, cap, is_prior) in self.sample_cache:
                        del self.sample_cache[(img_path, cap, is_prior)]
                    if img_path in self.sample_indices:
                        del self.sample_indices[img_path]
                    if img_path in self.latents_cache:
                        del self.latents_cache[img_path]
            self.latents_cache.update(latents_cache)

        bucket_idx = 0
        total_len = 0
        bucket_len = {}
        max_idx_chars = len(str(len(self.train_dict.keys())))
        p_len = self.num_class_images + self.num_train_images
        nc = self.num_class_images
        ni = self.num_train_images
        ti = nc + ni
        shared.status.job_count = p_len
        shared.status.job_no = 0
        total_instances = 0
        total_classes = 0
        pbar = mytqdm(range(p_len), desc="Caching latents..." if self.cache_latents else "Processing images...", position=0)
        image_cache_file = os.path.join(self.cache_dir, f"image_cache_{self.resolution}.safetensors")
        latents_cache = {}
        if os.path.exists(image_cache_file):
            print("Loading cached latents...")
            latents_cache = safetensors.torch.load_file(image_cache_file)
        for dict_idx, train_images in self.train_dict.items():
            if not train_images:
                continue
            # Separate the resolution from the index where we need it
            res = (dict_idx[0], dict_idx[1])
            # This should really be the index, because we want the bucket sampler to shuffle them all
            self.resolutions.append(dict_idx)
            # Cache with the actual res, because it's used to crop
            cache_images(train_images, res, pbar)
            inst_count = len(train_images)
            class_count = 0
            if dict_idx in self.class_dict:
                # Use dict index to find class images
                class_images = self.class_dict[dict_idx]
                # Use actual res here as well
                cache_images(class_images, res, pbar)
                class_count = len(class_images)
            total_instances += inst_count
            total_classes += class_count
            example_len = inst_count if class_count == 0 else inst_count * 2
            # Use index here, not res
            bucket_len[dict_idx] = example_len
            total_len += example_len
            bucket_str = str(bucket_idx).rjust(max_idx_chars, " ")
            inst_str = str(len(train_images)).rjust(len(str(ni)), " ")
            class_str = str(class_count).rjust(len(str(nc)), " ")
            ex_str = str(example_len).rjust(len(str(ti * 2)), " ")
            # Log both here
            pbar.write(
                f"Bucket {bucket_str} {dict_idx} - Instance Images: {inst_str} | Class Images: {class_str} | Max Examples/batch: {ex_str}")
            bucket_idx += 1
        try:
            if set(self.latents_cache.keys()) != set(latents_cache.keys()):
                print("Saving cache!")
                del latents_cache
                if os.path.exists(image_cache_file):
                    os.remove(image_cache_file)
                safetensors.torch.save_file(copy.deepcopy(self.latents_cache), image_cache_file)
        except:
            pass
        bucket_str = str(bucket_idx).rjust(max_idx_chars, " ")
        inst_str = str(total_instances).rjust(len(str(ni)), " ")
        class_str = str(total_classes).rjust(len(str(nc)), " ")
        tot_str = str(total_len).rjust(len(str(ti)), " ")
        pbar.write(
            f"Total Buckets {bucket_str} - Instance Images: {inst_str} | Class Images: {class_str} | Max Examples/batch: {tot_str}")
        self._length = total_len
        print(f"\nTotal images / batch: {self._length}, total examples: {total_len}")

    def shuffle_buckets(self):
        sample_dict = {}
        batch_indices = []
        batch_samples = []
        keys = list(self.train_dict.keys())
        if not self.debug_dataset:
            random.shuffle(keys)
        for key in keys:
            sample_list = []
            if not self.debug_dataset:
                random.shuffle(self.train_dict[key])
            for entry in self.train_dict[key]:
                sample_list.append(entry)
                batch_indices.append(entry[0])
                batch_samples.append(entry)
                if key in self.class_dict:
                    class_entries = self.class_dict[key]
                    selection = random.choice(class_entries)
                    batch_indices.append(selection[0])
                    batch_samples.append(selection)
                    sample_list.append(selection)
            sample_dict[key] = sample_list
        self.sample_dict = sample_dict
        self.batch_indices = batch_indices
        self.batch_samples = batch_samples

    def __len__(self):
        return self._length

    def get_example(self, res):
        # Select the current bucket of image paths
        bucket = self.sample_dict[res]

        # Set start position from last iteration
        img_index = self.image_index

        # Reset image index (double-check)
        if img_index >= len(bucket):
            img_index = 0

        repeats = 0
        # Grab instance image data
        image_path, caption, is_class_image = bucket[img_index]
        image_index = self.sample_indices.index(image_path)

        img_index += 1

        # Reset image index
        if img_index >= len(bucket):
            img_index = 0
            repeats += 1

        self.image_index = img_index

        return image_index, repeats

    def __getitem__(self, index):
        image_path, caption, is_class_image = self.sample_cache[index]
        if not self.debug_dataset:
            image_data, input_ids = self.load_image(image_path, caption, self.active_resolution)
        else:
            image_data = image_path
            # print(f"Recoding: {caption}")
            caption, cap_tokens = self.cache_caption(image_path, caption)
            rebuilt = self.tokenizer.decode(cap_tokens.tolist()[0])
            input_ids = (caption, rebuilt)
        # If we have reached the end of our bucket, increment to the next, update the count, reset image index.
        example = {
            "image": image_data,
            "input_ids": input_ids,
            "res": self.active_resolution,
            "is_class": is_class_image
        }
        return example
