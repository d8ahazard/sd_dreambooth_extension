import os.path
import random
from typing import List

import cv2
import numpy as np
import torch.utils.data
from PIL import Image
from torchvision.transforms import transforms

from extensions.sd_dreambooth_extension.dreambooth import db_shared
from extensions.sd_dreambooth_extension.dreambooth.db_shared import status
from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import closest_resolution, make_bucket_resolutions, \
    mytqdm, PromptData


class DbDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size, instance_prompts, class_prompts, tokens, tokenizer,
                 resolution, prior_loss_weight, hflip, random_crop, shuffle_tokens, not_pad_tokens, debug_dataset) -> None:
        super().__init__()
        self.batch_indices = []
        self.batch_samples = []
        print("Init dataset!")
        # A dictionary of string/latent pairs matching image paths
        self.latents_cache = {}
        # A dictionary of string/input_id(s) pairs matching image paths
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
        self.prior_loss_weight = prior_loss_weight
        self.random_crop = random_crop
        self.debug_dataset = debug_dataset
        self.shuffle_tokens = shuffle_tokens
        self.not_pad_tokens = not_pad_tokens
        self.tokens = tokens
        self.vae = None
        self.cache_latents = False
        flip_p = 0.5 if hflip else 0.0
        self.image_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(flip_p),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )



    @staticmethod
    def open_and_trim(image_path, reso):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image, np.uint8)
        image_height, image_width = image.shape[0:2]
        # Don't resize and junk if the image is already properly sized
        if image_width == reso[0] and image_height == reso[1]:
            return image
        ar_img = image_width / image_height
        ar_reso = reso[0] / reso[1]
        if ar_img > ar_reso:
            scale = reso[1] / image_height
        else:
            scale = reso[0] / image_width
        resized_size = (int(image_width * scale + .5), int(image_height * scale + .5))

        image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)
        if resized_size[0] > reso[0]:
            trim_size = resized_size[0] - reso[0]
            image = image[:, trim_size // 2:trim_size // 2 + reso[0]]
        elif resized_size[1] > reso[1]:
            trim_size = resized_size[1] - reso[1]
            image = image[trim_size // 2:trim_size // 2 + reso[1]]
        assert image.shape[0] == reso[1] and image.shape[1] == reso[0], \
            f"internal error, illegal trimmed size: {image.shape}, {reso}"
        return image

    def load_image(self, image_path, caption, res):
        if self.debug_dataset:
            image = os.path.splitext(image_path)
            input_ids = caption
        else:
            if self.cache_latents:
                image = self.latents_cache[image_path]
            else:
                img = self.open_and_trim(image_path, res)
                image = self.image_transforms(img)
            input_ids = self.caption_cache[image_path]
        return image, input_ids

    def cache_latent(self, image_path, res):
        latents = None
        if self.vae is not None and image_path not in self.latents_cache:
            image = self.open_and_trim(image_path, res)
            img_tensor = self.image_transforms(image)
            img_tensor = img_tensor.unsqueeze(0).to(device=self.vae.device, dtype=self.vae.dtype)
            latents = self.vae.encode(img_tensor).latent_dist.sample().squeeze(0).to("cpu")
        self.latents_cache[image_path] = latents

    def cache_caption(self, image_path, caption):
        input_ids = None
        if self.tokenizer is not None and image_path not in self.caption_cache:
            if self.not_pad_tokens:
                input_ids = self.tokenizer(caption, padding=True, truncation=True,
                                           return_tensors="pt").input_ids
            else:
                input_ids = self.tokenizer(caption, padding='max_length', truncation=True,
                                           return_tensors='pt').input_ids
        self.caption_cache[image_path] = input_ids

    def make_buckets_with_caching(self, vae, min_size):
        self.vae = vae
        self.cache_latents = vae is not None
        state = f"Preparing Dataset ({'With Caching' if self.cache_latents else 'Without Caching'})"
        print(state)
        status.textinfo = state

        # Create a list of resolutions
        bucket_resos = make_bucket_resolutions(self.resolution, min_size)
        self.train_dict = {}


        def sort_images(img_data: List[PromptData], resos, target_dict, is_class_img):
            for prompt_data in img_data:
                path = prompt_data.src_image
                image_width, image_height = prompt_data.resolution
                cap = prompt_data.prompt
                reso = closest_resolution(image_width, image_height, resos)
                target_dict.setdefault(reso, []).append((path, cap, is_class_img))

        sort_images(self.train_img_data, bucket_resos, self.train_dict, False)
        sort_images(self.class_img_data, bucket_resos, self.class_dict, True)

        # Enumerate by resolution, cache as needed
        def cache_images(images, reso, p_bar):
            for img_path, cap, is_prior in images:
                if self.cache_latents and not self.debug_dataset:
                    self.cache_latent(img_path, reso)
                self.cache_caption(img_path, cap)
                self.sample_indices.append(img_path)
                self.sample_cache.append((img_path, cap, is_prior))
                pbar.update()

        bucket_idx = 0
        total_len = 0
        bucket_len = {}
        max_idx_chars = len(str(len(self.train_dict.keys())))
        p_len = self.num_class_images + self.num_train_images
        nc = self.num_class_images
        ni = self.num_train_images
        ti = nc + ni
        db_shared.status.job_count = p_len
        db_shared.status.job_no = 0
        total_instances = 0
        total_classes = 0
        pbar = mytqdm(range(p_len), desc="Caching latents..." if self.cache_latents else "Processing images...")
        for res, train_images in self.train_dict.items():
            if not train_images:
                continue
            self.resolutions.append(res)
            cache_images(train_images, res, pbar)
            inst_count = len(train_images)
            class_count = 0
            if res in self.class_dict:
                class_images = self.class_dict[res]
                cache_images(class_images, res, pbar)
                class_count = len(class_images)
            total_instances += inst_count
            total_classes += class_count
            example_len = inst_count if class_count == 0 else inst_count * 2
            bucket_len[res] = example_len
            total_len += example_len
            bucket_str = str(bucket_idx).rjust(max_idx_chars, " ")
            inst_str = str(len(train_images)).rjust(len(str(ni)), " ")
            class_str = str(class_count).rjust(len(str(nc)), " ")
            ex_str = str(example_len).rjust(len(str(ti * 2)), " ")
            pbar.write(f"Bucket {bucket_str} {res} - Instance Images: {inst_str} | Class Images: {class_str} | Examples/batch: {ex_str}")
            bucket_idx += 1
        bucket_str = str(bucket_idx).rjust(max_idx_chars, " ")
        inst_str = str(total_instances).rjust(len(str(ni)), " ")
        class_str = str(total_classes).rjust(len(str(nc)), " ")
        tot_str = str(total_len).rjust(len(str(ti)), " ")
        pbar.write(f"Totals: Buckets {bucket_str} | Instance Images: {inst_str} | Class Images: {class_str} | Examples/batch: {tot_str}")
        self._length = len(self.train_dict.items()) * self.batch_size
        self._length = total_len
        print(f"Total images / batch: {self._length}, total examples: {total_len}")

    def shuffle_buckets(self):
        sample_dict = {}
        batch_indices = []
        batch_samples = []
        keys = list(self.train_dict.keys())
        random.shuffle(keys)
        for key in keys:
            sample_list = []
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
        if self.debug_dataset:
            print(f"Shuffled bucket keys: {sample_dict.keys()}")
        self.sample_dict = sample_dict
        self.batch_indices = batch_indices
        self.batch_samples = batch_samples

    def check_shuffle_tokens(self, caption):
        if self.shuffle_tokens:
            tags = caption.split(',')
            if len(tags) > 2:
                first_tag = tags.pop(0)
                random.shuffle(tags)
                tags.insert(0, first_tag)
            caption = ','.join(tags)
        return caption

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
            image_data, input_id = self.load_image(image_path, caption, self.active_resolution)
        else:
            image_data = image_path
            input_id = caption
        loss_weight = self.prior_loss_weight if is_class_image else 1.0
        # If we have reached the end of our bucket, increment to the next, update the count, reset image index.
        example = {"image": image_data, "input_id": input_id, "loss_weight": loss_weight, "res": self.active_resolution}
        return example


