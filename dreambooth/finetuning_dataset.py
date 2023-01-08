import math
import random

import PIL.Image
import albumentations as albu
import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

from extensions.sd_dreambooth_extension.dreambooth.db_shared import status
from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import closest_resolution, make_bucket_resolutions, \
    compare_prompts, get_dim


class DreamBoothOrFineTuningDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size, train_img_path_captions, reg_img_path_captions, tokens, tokenizer, resolution,
                 prior_loss_weight, hflip, random_crop, shuffle_tokens, not_pad_tokens, debug_dataset) -> None:
        super().__init__()

        self.train_buckets = []
        self.reg_buckets = []
        self.train_buckets_indices = []
        self.reg_buckets_indices = []
        self._length = 0
        self.size_lat_cache = {}
        self.batch_size = batch_size
        self.train_img_path_captions = train_img_path_captions
        self.reg_img_path_captions = reg_img_path_captions
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.prior_loss_weight = prior_loss_weight
        self.random_crop = random_crop
        self.debug_dataset = debug_dataset
        self.shuffle_tokens = shuffle_tokens
        self.not_pad_tokens = not_pad_tokens
        self.latents_cache = None
        self.tokens = tokens
        # augmentation
        flip_p = 0.5 if hflip else 0.0
        if hflip:
            self.aug = albu.Compose([
                albu.HorizontalFlip(p=flip_p)
            ], p=1.)
        else:
            self.aug = None

        self.num_train_images = len(self.train_img_path_captions)
        self.num_reg_images = len(self.reg_img_path_captions)
        self.enable_reg_images = self.num_reg_images > 0

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def make_buckets_with_caching(self, vae, min_size):

        cache_latents = vae is not None
        state = f"Preparing Dataset ({'With Caching' if cache_latents else 'Without Caching'})"
        print(state)
        status.textinfo = state

        bucket_resos, bucket_aspect_ratios = make_bucket_resolutions(self.resolution, min_size)
        status.job_count = len(self.train_img_path_captions) + len(self.reg_img_path_captions)
        status.job_no = 0
        for image_path, _ in tqdm(self.train_img_path_captions + self.reg_img_path_captions):
            if image_path in self.size_lat_cache:
                continue
            status.job_no += 1
            image_width, image_height = get_dim(image_path, self.resolution)
            reso = closest_resolution(image_width, image_height, bucket_resos)

            if cache_latents and not self.debug_dataset:
                image = self.open_and_trim(image_path, reso)
                img_tensor = self.image_transforms(image)
                img_tensor = img_tensor.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
                latents = vae.encode(img_tensor).latent_dist.sample().squeeze(0).to("cpu")
            else:
                latents = None

            self.size_lat_cache[image_path] = (reso, latents)

        self.train_buckets = [[] for _ in range(len(bucket_resos))]
        self.reg_buckets = [[] for _ in range(len(bucket_resos))]
        reso_to_index = {}
        for i, reso in enumerate(bucket_resos):
            reso_to_index[reso] = i

        def split_to_buckets(buckets, img_path_captions):
            for path, caption in img_path_captions:
                img_reso, _ = self.size_lat_cache[path]
                bidx = reso_to_index[img_reso]
                buckets[bidx].append((path, caption))

        split_to_buckets(self.train_buckets, self.train_img_path_captions)

        if self.enable_reg_images:
            caps = []
            caps += self.reg_img_path_captions
            split_to_buckets(self.reg_buckets, caps)

        bi = 0
        for i, (reso, images) in enumerate(zip(bucket_resos, self.train_buckets)):
            if len(images) > 0:
                bi += 1
                print(f"Train Bucket {bi}: Resolution {reso}, Count: {len(images)}")

        bi = 0
        for i, (reso, images) in enumerate(zip(bucket_resos, self.reg_buckets)):
            if len(images) > 0:
                bi += 1
                print(f"Reg Bucket {bi}: Resolution {reso}, Count: {len(images)}")

        for bucket_index, bucket in enumerate(self.train_buckets):
            batch_count = int(math.ceil(len(bucket) / self.batch_size))
            for batch_index in range(batch_count):
                self.train_buckets_indices.append((bucket_index, batch_index))

        for bucket_index, bucket in enumerate(self.reg_buckets):
            batch_count = int(math.ceil(len(bucket) / self.batch_size))
            for batch_index in range(batch_count):
                self.reg_buckets_indices.append((bucket_index, batch_index))

        if self.debug_dataset:
            for i, (reso, images) in enumerate(zip(bucket_resos, self.train_buckets)):
                reg_images = self.reg_buckets[i]
                missing_prompts = []
                for path, caption in images:
                    class_found = False
                    for reg_path, reg_caption in reg_images:
                        if compare_prompts(caption, reg_caption, self.tokens):
                            class_found = True
                            break
                    if not class_found:
                        missing_prompts.append(caption)

        self.shuffle_buckets()
        self._length = len(self.train_buckets_indices)

        print(f"Total images: {self._length}")


    @staticmethod
    def open_and_trim(image_path, reso):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image, np.uint8)
        image_height, image_width = image.shape[0:2]
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

    def shuffle_buckets(self):
        random.shuffle(self.train_buckets_indices)
        random.shuffle(self.reg_buckets_indices)
        for bucket in self.train_buckets:
            random.shuffle(bucket)
        for bucket in self.reg_buckets:
            random.shuffle(bucket)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # Shuffle the buckets at the start of each epoch
        if index == 0:
            self.shuffle_buckets()

        # Select the current bucket and image index
        bucket = self.train_buckets[self.train_buckets_indices[index][0]]
        image_index = self.train_buckets_indices[index][1] * self.batch_size

        # Initialize lists to store the latents, images, captions, and loss weights for the batch
        latents_list = []
        images = []
        captions = []

        # Process instance images
        for image_path, caption in bucket[image_index:image_index + self.batch_size]:

            reso, latents = self.size_lat_cache[image_path]

            if self.debug_dataset:
                image = Image.open(image_path)
            else:
                if latents is None:
                    img = self.open_and_trim(image_path, reso)

                    if self.aug is not None:
                        img = self.aug(image=img)['image']

                    image = self.image_transforms(img)
                else:
                    image = None

            images.append(image)
            latents_list.append(latents)
            if self.shuffle_tokens:
                tags = caption.split(',')
                if len(tags) > 2:
                    first_tag = tags.pop(0)
                    random.shuffle(tags)
                    tags.insert(0, first_tag)
                    caption = ','.join(tags)
                captions.append(caption)
            else:
                captions.append(caption)

            # Select the bucket with reg images
            reg_bucket = self.reg_buckets[self.train_buckets_indices[index][0]]
            # Randomize the order in which we pick a reg image
            b_indices = list(range(len(reg_bucket)))
            random.shuffle(b_indices)
            has_class = False

            for reg_idx in b_indices:
                if has_class:
                    break
                image_path, reg_caption = reg_bucket[reg_idx]
                if compare_prompts(caption, reg_caption, self.tokens):
                    res, latents = self.size_lat_cache[image_path]
                    if reso == res:
                        if self.debug_dataset:
                            image = Image.open(image_path)
                        else:
                            img = self.open_and_trim(image_path, res)
                            if self.aug is not None:
                                img = self.aug(image=img)['image']

                            image = self.image_transforms(img)
                        images.append(image)
                        latents_list.append(latents)
                        if self.shuffle_tokens:
                            tags = reg_caption.split(',')
                            if len(tags) > 2:
                                first_tag = tags.pop(0)
                                random.shuffle(tags)
                                tags.insert(0, first_tag)
                                reg_caption = ','.join(tags)
                            captions.append(reg_caption)
                        else:
                            captions.append(reg_caption)
                        has_class = True

            if self.tokenizer is not None:
                if self.not_pad_tokens:
                    input_ids = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").input_ids
                else:
                    input_ids = self.tokenizer(captions, padding='max_length', truncation=True, return_tensors='pt').input_ids
            else:
                input_ids = self.tokenizer
            # Create the example to be returned
            example = {'input_ids': input_ids, "with_prior": has_class}
            if self.debug_dataset:
                example["captions"] = captions
                example["images"] = images
                example["res"] = reso
            if images[0] is not None and not self.debug_dataset:
                images = torch.stack(images)
                images = images.to(memory_format=torch.contiguous_format)
            else:
                images = None
            example['pixel_values'] = images
            example['latents'] = torch.stack(latents_list) if latents_list[0] is not None else None
            return example
