import os.path
import random
from typing import Tuple

import albumentations as albu
import cv2
import numpy as np
import torch.utils.data
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

from extensions.sd_dreambooth_extension.dreambooth import db_shared
from extensions.sd_dreambooth_extension.dreambooth.db_shared import status
from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import closest_resolution, make_bucket_resolutions, \
    compare_prompts, get_dim, mytqdm


class BucketCounter:
    def __init__(self, starting_keys=None):
        self.counts = {}
        print("Initializing bucket counter!")
        if starting_keys is not None:
            for key in starting_keys:
                self.counts[key] = 0

    def count(self, key: Tuple[int, int]):
        if key in self.counts:
            self.counts[key] += 1
        else:
            self.counts[key] = 1

    def min(self):
        return min(self.counts.values()) if len(self.counts) else 0

    def max(self):
        return max(self.counts.values()) if len(self.counts) else 0

    def get(self, key: Tuple[int, int]):
        return self.counts[key] if key in self.counts else 0

    def check_reset(self):
        if self.max() == self.min():
            for key in list(self.counts.keys()):
                self.counts[key] = 0

    def missing(self):
        out = {}
        max = self.max()
        for key in list(self.counts.keys()):
            if self.counts[key] < max:
                out[key] = max - self.counts[key]
        return out

    def print(self):
        print(f"Bucket counts: {self.counts}")

class DbDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size, counter, train_img_path_captions, class_img_path_captions, tokens, tokenizer,
                 resolution, prior_loss_weight, hflip, random_crop, shuffle_tokens, not_pad_tokens, debug_dataset) -> None:
        super().__init__()
        print("Init dataset!")
        # A dictionary of string/latent pairs matching image paths
        self.latents_cache = {}
        # A dictionary of string/input_id(s) pairs matching image paths
        self.caption_cache = {}
        # The max len of all caption input_ids
        self.max_seq_len = 0
        # A dictionary of (int, int) / List[(string, string)] of resolutions and the corresponding image paths/captions
        self.train_dict = {}
        # A dictionary of string/list[(string, string)], where the string is the instance image path, the path/captions
        # are matching reg images
        self.class_dict = {}
        # A dictionary containing the number of times each bucket has been enumerated over
        self.counter = counter
        # A list of the currently "active" resolutions to enumerate
        self.active_resos = []
        # All of the available bucket resolutions
        self.resolutions = []
        # The currently active index in our dict of buckets
        self.bucket_index = 0
        # The currently active image index while iterating
        self.image_index = 0
        # Total len of the dataloader
        self._length = 0

        self.batch_size = batch_size
        self.train_img_path_captions = train_img_path_captions
        self.class_img_path_captions = class_img_path_captions
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

        # augmentation
        flip_p = 0.5 if hflip else 0.0
        if hflip:
            self.aug = albu.Compose([
                albu.HorizontalFlip(p=flip_p)
            ], p=1.)
        else:
            self.aug = None

        self.num_train_images = len(self.train_img_path_captions)
        self.num_class_images = len(self.class_img_path_captions)


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

    def load_image(self, image_path, caption, res):
        if self.debug_dataset:
            image = os.path.splitext(image_path)
            input_ids = caption
        else:
            if self.cache_latents:
                image = self.latents_cache[image_path]
            else:
                img = self.open_and_trim(image_path, res)
                if self.aug is not None:
                    img = self.aug(image=img)['image']
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize(size=(res[1], res[0])),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                )
                image = image_transform(img)
            input_ids = self.caption_cache[image_path]
        return image, input_ids

    def cache_latent(self, image_path, res):
        latents = None
        if self.vae is not None:
            image = self.open_and_trim(image_path, res)
            if self.aug is not None:
                image = self.aug(image=image)['image']

            image_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=(res[1], res[0])),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            img_tensor = image_transform(image)
            img_tensor = img_tensor.unsqueeze(0).to(device=self.vae.device, dtype=self.vae.dtype)
            latents = self.vae.encode(img_tensor).latent_dist.sample().squeeze(0).to("cpu")
        self.latents_cache[image_path] = latents

    def cache_caption(self, image_path, caption):
        input_ids = None
        if self.tokenizer is not None:
            if self.not_pad_tokens:
                input_ids = self.tokenizer(caption, padding=True, truncation=True,
                                           return_tensors="pt").input_ids
            else:
                input_ids = self.tokenizer(caption, padding='max_length', truncation=True,
                                           return_tensors='pt').input_ids
        self.max_seq_len = max(self.max_seq_len, input_ids.size()[1])
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
        class_dict = {}

        def sort_images(img_path_captions, resos, target_dict):
            for path, cap in img_path_captions:
                image_width, image_height = get_dim(path, self.resolution)
                reso = closest_resolution(image_width, image_height, resos)
                target_dict.setdefault(reso, []).append((path, cap))

        sort_images(self.train_img_path_captions, bucket_resos, self.train_dict)
        sort_images(self.class_img_path_captions, bucket_resos, class_dict)
        # Enumerate by resolution, cache as needed
        def cache_images(images, reso):
            for img_path, cap in images:
                if self.cache_latents and not self.debug_dataset:
                    self.cache_latent(img_path, reso)
                self.cache_caption(img_path, cap)

        bucket_idx = 0
        class_count = 0
        total_len = 0
        bucket_len = {}
        max_idx_chars = len(str(len(self.train_dict.keys())))
        p_len = self.num_class_images + self.num_train_images
        pbar = mytqdm(total=p_len, desc="Processing images")
        db_shared.status.job_count = p_len
        db_shared.status.job_no = 0
        for res, train_images in self.train_dict.items():
            self.resolutions.append(res)
            class_len = 0
            if not train_images:
                continue
            class_list = class_dict.get(res, [])
            cache_images(train_images, res)
            for image_path, caption in train_images:
                instance_classes = []
                for class_path, class_caption in class_list:
                    if compare_prompts(caption, class_caption, self.tokens):
                        instance_classes.append((class_path, class_caption))
                        pbar.update()
                class_count += len(instance_classes)
                cache_images(instance_classes, res)
                self.class_dict[image_path] = instance_classes
                pbar.update()
                class_len += 1 + (1 if class_count > 0 else 0)
            bucket_len[res] = class_len
            total_len += class_len
            bucket_str = str(bucket_idx).rjust(max_idx_chars, " ")
            inst_str = str(len(train_images)).rjust(5, " ")
            class_str = str(class_count).rjust(5, " ")
            ex_str = str(class_len).rjust(5, " ")
            tqdm.write(f"Bucket {bucket_str} {res} - Instance Images: {inst_str} | Class Images: {class_str} | Examples: {ex_str}")
            bucket_idx += 1

        self._length = total_len // self.batch_size
        print(f"Total images / batch: {self._length}, total examples: {total_len}")

    def check_shuffle_tokens(self, caption):
        if self.shuffle_tokens:
            tags = caption.split(',')
            if len(tags) > 2:
                first_tag = tags.pop(0)
                random.shuffle(tags)
                tags.insert(0, first_tag)
            caption = ','.join(tags)
        return caption


    def set_buckets(self):
        # Initialize list of bucket counts if not set
        all_resos = self.resolutions

        # Enumerate each bucket, adding those which need to be leveled
        pop_index = 0
        resos_to_use = []
        am = self.counter.missing()
        missing = am.copy()
        pop_index = 0
        #self.counter.check_reset()
        if len(missing):
            while len(resos_to_use) < len(all_resos):
                if len(missing):
                    for res, count in am.items():
                        resos_to_use.append(res)
                        missing[res] -= 1
                        if missing[res] == 0:
                            del missing[res]
                        am = missing.copy()
                else:
                    resos_to_use.append(all_resos[pop_index])
                    pop_index += 1
                    if pop_index >= len(all_resos):
                        pop_index = 0
        else:
            resos_to_use = all_resos.copy()
        random.shuffle(resos_to_use)
        self.active_resos = resos_to_use
        self.bucket_index = 0
        self.image_index = 0

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # Reset bucket counts if we're starting from the start of the dataloader
        if index == 0:
            self.set_buckets()
        # Select the current bucket of image paths
        if self.bucket_index >= len(self.active_resos):
            self.bucket_index = 0
        bucket_res = self.active_resos[self.bucket_index]
        bucket = self.train_dict[bucket_res]
        # Initialize outputs
        image_paths = []
        images = [None] * self.batch_size
        ids = [None] * self.batch_size
        loss_weights = torch.empty(self.batch_size, dtype=torch.float32)

        # Set start position from last iteration
        img_index = self.image_index

        count = 0
        # Loop the current bucket until we fill our batch
        bucket_emptied = False
        repeats = 0
        while count < self.batch_size:
            # Grab instance image data
            image_path, caption = bucket[img_index]
            image_data, input_ids = self.load_image(image_path, caption, bucket_res)
            if self.debug_dataset:
                image_paths.append(image_path)
            images[count] = image_data
            ids[count] = input_ids
            loss_weights[count] = 1.0
            count += 1
            if count < self.batch_size:
                class_image_paths = self.class_dict[image_path]
                # Select a random class image from our instance image's available class image.
                if len(class_image_paths):
                    class_image_path, class_caption = random.choice(class_image_paths)
                    if class_image_path is not None:
                        # Grab class image data
                        image_data, input_ids = self.load_image(class_image_path, class_caption, bucket_res)
                        images[count] = image_data
                        ids[count] = input_ids
                        loss_weights[count] = self.prior_loss_weight
                        count += 1

            img_index += 1
            if img_index >= len(bucket):
                bucket_emptied = True
                img_index = 0
                repeats += 1
                self.counter.count(bucket_res)


        self.image_index = img_index
        # If we have reached the end of our bucket, increment to the next, update the count, reset image index.
        if bucket_emptied:
            # print(f"Incrementing bucket {self.bucket_index}-{bucket_res}, {self.counter.get(bucket_res)}")
            self.bucket_index += 1
            self.image_index = 0

        # Stack and return outputs
        if not self.debug_dataset:
            images = torch.stack(images)
            ids = torch.cat(ids, dim=0)
            if not self.cache_latents:
                images = images.to(memory_format=torch.contiguous_format)

        example = {"images": images, "input_ids": ids, "loss_weight": loss_weights, "res": bucket_res}
        return example


