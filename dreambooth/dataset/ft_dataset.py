import copy
import logging
import os.path
import random
import re
import traceback
from pathlib import Path
from typing import List, Union, Dict

import safetensors.torch
import torch.utils.data
from PIL import Image
from PIL.ExifTags import TAGS
from diffusers import AutoencoderKL
from torchvision.transforms import transforms
from transformers import CLIPTokenizer

from dreambooth import shared
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.shared import status
from dreambooth.utils.image_utils import make_bucket_resolutions, \
    closest_resolution, shuffle_tags, open_and_trim, is_image
from dreambooth.utils.text_utils import build_strict_tokens
from helpers.mytqdm import mytqdm

logger = logging.getLogger(__name__)


class FtDataset(torch.utils.data.Dataset):
    """
    Dataset for handling training data
    """

    def __init__(
            self,
            instance_data_dir: str,
            resolution: int,
            batch_size: int,
            hflip: bool,
            shuffle_tags: bool,
            strict_tokens: bool,
            dynamic_img_norm: bool,
            not_pad_tokens: bool,
            model_dir: str,
            cache_latents: bool,
            user: str,
            tokenizer: Union[CLIPTokenizer, None],
            vae: Union[AutoencoderKL, None],
            debug_dataset: bool = False,
            use_dir_tags: bool = False

    ) -> None:
        super().__init__()
        self.batch_indices = []
        self.batch_samples = []
        self.instance_data_dir = instance_data_dir
        self.cache_dir = os.path.join(model_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.cache_latents = cache_latents
        # A dictionary of string/latent pairs matching image paths
        self.latents_cache = {}
        # A dictionary of string/input_ids(s) pairs matching image paths
        self.caption_cache = {}
        # A dictionary of (int, int) / List[(string, string)] of resolutions and the corresponding image paths/captions
        self.train_dict = {}
        # A mash-up of the class/train dicts that is perfectly fair and balanced.
        self.sample_dict = {}
        # This is where we just keep a list of everything for batching
        self.sample_cache = []
        # This is just a list of the sample names that we can use to find where in the cache an image is
        self.sample_indices = []
        # All the available bucket resolutions
        self.resolutions = []
        # Currently active resolution
        self.active_resolution = (0, 0)
        # The currently active image index while iterating
        self.image_index = 0
        # Total len of the dataloader
        self._length = 0
        self.batch_size = batch_size
        self.batch_sampler = torch.utils.data.BatchSampler(self, batch_size, drop_last=True)
        self.train_img_data = self.enumerate_inputs(use_dir_tags)
        self.num_train_images = len(self.train_img_data)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.debug_dataset = debug_dataset
        self.shuffle_tags = shuffle_tags
        self.not_pad_tokens = not_pad_tokens
        self.strict_tokens = strict_tokens
        self.dynamic_img_norm = dynamic_img_norm
        self.vae = vae
        self.cache_latents = cache_latents
        flip_p = 0.5 if hflip else 0.0
        self.image_transforms = self.build_compose(hflip, flip_p)
        print(f"Initializing finetune dataset with {self.num_train_images} images")
        self.pbar = mytqdm(range(self.num_train_images),
                           desc="Caching latents..." if self.cache_latents else "Processing images...", position=0,
                           user=user, target="dreamProgress")
        self.pbar.status_index = 1

        self.make_buckets_with_caching()

    def enumerate_inputs(self, use_dir_tags: bool = False):
        data = []
        logger.debug(f"Enumerating inputs in {self.instance_data_dir}")
        for root, dirs, files in os.walk(self.instance_data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if is_image(file_path):
                    txt_file = file_path.rsplit('.', 1)[0] + '.txt'
                    if os.path.isfile(txt_file):
                        with open(txt_file, 'r') as f:
                            caption = f.read().strip()
                    else:
                        caption = re.sub('[\W_]+', ' ', Path(file_path).stem)
                        caption = re.sub('^\d+|\d+$', '', caption)
                    if caption != "":
                        if use_dir_tags:
                            directory = os.path.basename(os.path.dirname(file_path))
                            dir_tag = directory
                            dir_replace = None
                            logger.debug(f"Dir tag: {dir_tag}")
                            if "-" in directory:
                                dir_tag = directory.split("-")[0]
                                dir_replace = directory.split("-")[1]
                                logger.debug(f"Dir replace: {dir_replace}")
                            cap_parts = caption.split(",")
                            if dir_replace is not None:
                                to_replace = [f"a {dir_replace}", f"the {dir_replace}", f"an {dir_replace}", dir_replace]
                                fixed_parts = []
                                for part in cap_parts:
                                    part = part.strip()
                                    for replace in to_replace:
                                        if replace in part:
                                            part = part.replace(replace, dir_tag)
                                            logger.debug(f"Replaced: {replace} with {dir_tag}")
                                    fixed_parts.append(part)
                                cap_parts = fixed_parts
                            else:
                                cap_parts.append(dir_tag)
                            # Get unique parts
                            cap_parts = list(set(cap_parts))
                            # Shuffle order
                            caption = ", ".join(cap_parts)
                            caption = caption.replace("_", " ")
                        data.append({'image': file_path, 'text': caption})
                    else:
                        logger.warning(f"Skipping {file_path} due to empty caption")
        return data

    def build_compose(self, hflip, flip_p):
        img_augmentation = [transforms.ToPILImage(), transforms.RandomHorizontalFlip(flip_p)]
        to_tensor = [transforms.ToTensor()]

        image_transforms = (
            to_tensor if not hflip else img_augmentation + to_tensor
        )
        return transforms.Compose(image_transforms)

    def get_img_std(self, img):
        if self.dynamic_img_norm:
            return img.mean(), img.std()
        else:
            return [0.5], [0.5]

    def image_transform(self, img):
        img = self.image_transforms(img)
        mean, std = self.get_img_std(img)
        norm = transforms.Normalize(mean, std)
        return norm(img)

    def load_image(self, image_path, caption, res):
        if self.debug_dataset:
            image = os.path.splitext(image_path)
            input_ids = caption
        else:
            if self.cache_latents:
                image = self.latents_cache[image_path]
            else:
                img = open_and_trim(image_path, res, False)
                image = self.image_transform(img)
            if self.shuffle_tags:
                caption, input_ids = self.cache_caption(image_path, caption)
            else:
                input_ids = self.caption_cache[image_path]
        return image, input_ids

    def cache_latent(self, image_path, res):
        if self.vae is not None:
            image = open_and_trim(image_path, res, False)
            img_tensor = self.image_transform(image)
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

    def make_buckets_with_caching(self):
        state = f"Preparing Dataset ({'With Caching' if self.cache_latents else 'Without Caching'})"
        print(state)
        if self.pbar is not None:
            self.pbar.set_description(state)
        status.textinfo = state

        # Create a list of resolutions
        bucket_resos = make_bucket_resolutions(self.resolution)
        logger.debug(f"Bucket Resolutions: {bucket_resos}")
        self.train_dict = {}

        def sort_images(img_data: List[Dict], resos):
            for prompt_data in img_data:
                path = prompt_data["image"]
                cap = prompt_data["text"]

                # Open the image and get the original resolution
                image = Image.open(path)
                image_width, image_height = image.size
                if image_width > self.resolution or image_height > self.resolution:
                    # Calculate the scaling factor for each dimension
                    width_scale = self.resolution / image_width
                    height_scale = self.resolution / image_height
                    scale_factor = min(width_scale, height_scale)
                    # Scale the image dimensions using the calculated scaling factor
                    image_width = int(image_width * scale_factor)
                    image_height = int(image_height * scale_factor)

                # Check if the image has EXIF data
                exif = image._getexif()
                if exif is not None:
                    # Retrieve the rotation information from the EXIF data
                    for tag, value in exif.items():
                        if TAGS.get(tag) == 'Orientation':
                            if value == 6 or value == 8:
                                # Swap width and height for 90° or 270° rotation
                                image_width, image_height = image_height, image_width
                            break

                # Get the closest resolution from the provided options
                reso = closest_resolution(image_width, image_height, resos)

                # Set the detected resolution as the key in self.train_dict
                self.train_dict.setdefault(reso, []).append((path, cap))

        sort_images(self.train_img_data, bucket_resos)

        def cache_images(images, reso, p_bar: mytqdm):
            for img_path, cap in images:
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
                    self.sample_cache.append((img_path, cap))
                    p_bar.update()
                except Exception as e:
                    traceback.print_exc()
                    print(f"Exception caching: {img_path}: {e}")
                    if img_path in self.caption_cache:
                        del self.caption_cache[img_path]
                    if (img_path, cap) in self.sample_cache:
                        del self.sample_cache[(img_path, cap)]
                    if img_path in self.sample_indices:
                        del self.sample_indices[img_path]
                    if img_path in self.latents_cache:
                        del self.latents_cache[img_path]
            self.latents_cache.update(latents_cache)

        bucket_idx = 0
        total_len = 0
        bucket_len = {}
        max_idx_chars = len(str(len(self.train_dict.keys())))
        p_len = self.num_train_images
        ni = self.num_train_images
        ti = ni
        shared.status.job_count = p_len
        shared.status.job_no = 0
        total_instances = 0
        image_cache_file = os.path.join(self.cache_dir, f"image_cache_finetune_{self.resolution}.safetensors")
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
            cache_images(train_images, res, self.pbar)
            inst_count = len(train_images)
            total_instances += inst_count
            example_len = inst_count
            # Use index here, not res
            bucket_len[dict_idx] = example_len
            total_len += example_len
            bucket_str = str(bucket_idx).rjust(max_idx_chars, " ")
            inst_str = str(len(train_images)).rjust(len(str(ni)), " ")
            ex_str = str(example_len).rjust(len(str(ti * 2)), " ")
            # Log both here
            self.pbar.write(
                f"Bucket {bucket_str} {dict_idx} - Instance Images: {inst_str} | Max Examples/batch: {ex_str}")
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
        tot_str = str(total_len).rjust(len(str(ti)), " ")
        self.pbar.write(
            f"Total Buckets {bucket_str} - Instance Images: {inst_str} | Max Examples/batch: {tot_str}")
        self._length = total_len
        print(f"\nTotal images / batch: {self._length}, total examples: {total_len}")
        self.pbar.reset(0)

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
        image_path, caption = bucket[img_index]
        image_index = self.sample_indices.index(image_path)

        img_index += 1

        # Reset image index
        if img_index >= len(bucket):
            img_index = 0
            repeats += 1

        self.image_index = img_index

        return image_index, repeats

    def __getitem__(self, index):
        image_path, caption = self.sample_cache[index]
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
            "res": self.active_resolution
        }
        return example
