import math
import os
import random

import albumentations as albu
import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm


def make_bucket_resolutions(
        max_reso, min_size=256, max_size=1024, divisible=64
):
    max_width, max_height = max_reso
    max_area = (max_width // divisible) * (max_height // divisible)

    resos = set()

    size = int(math.sqrt(max_area)) * divisible
    resos.add((size, size))

    size = min_size
    while size <= max_size:
        width = size
        height = min(max_size, (max_area // (width // divisible)) * divisible)
        resos.add((width, height))
        resos.add((height, width))

        # # make additional resos
        # if width >= height and width - divisible >= min_size:
        #   resos.add((width - divisible, height))
        #   resos.add((height, width - divisible))
        # if height >= width and height - divisible >= min_size:
        #   resos.add((width, height - divisible))
        #   resos.add((height - divisible, width))

        size += divisible

    resos = list(resos)
    resos.sort()

    aspect_ratios = [w / h for w, h in resos]
    return resos, aspect_ratios


class DreamBoothOrFineTuningDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size, fine_tuning, train_img_path_captions, reg_img_path_captions, tokenizer, resolution,
                 prior_loss_weight, flip_aug, color_aug, face_crop_aug_range, random_crop, shuffle_caption,
                 disable_padding, debug_dataset, dtype) -> None:
        super().__init__()

        self.buckets = []
        self.buckets_indices = []
        self._length = 0
        self.size_lat_cache = {}
        self.batch_size = batch_size
        self.fine_tuning = fine_tuning
        self.train_img_path_captions = train_img_path_captions
        self.reg_img_path_captions = reg_img_path_captions
        self.tokenizer = tokenizer
        self.width, self.height = resolution
        self.size = min(self.width, self.height)  # 短いほう
        self.prior_loss_weight = prior_loss_weight
        self.face_crop_aug_range = face_crop_aug_range
        self.random_crop = random_crop
        self.debug_dataset = debug_dataset
        self.shuffle_caption = shuffle_caption
        self.disable_padding = disable_padding
        self.latents_cache = None
        self.enable_bucket = False
        self.dtype = dtype
        
        # augmentation
        flip_p = 0.5 if flip_aug else 0.0
        if color_aug:
            self.aug = albu.Compose([
                albu.OneOf([
                    albu.HueSaturationValue(5, 8, 0, p=.2),
                    albu.RandomGamma((95, 105), p=.5),
                ], p=.33),
                albu.HorizontalFlip(p=flip_p)
            ], p=1.)
        elif flip_aug:
            self.aug = albu.Compose([
                albu.HorizontalFlip(p=flip_p)
            ], p=1.)
        else:
            self.aug = None

        self.num_train_images = len(self.train_img_path_captions)
        self.num_reg_images = len(self.reg_img_path_captions)
        self.enable_reg_images = self.num_reg_images > 0

        if self.enable_reg_images and self.num_train_images < self.num_reg_images:
            print("some of reg images are not used / 正則化画像の数が多いので、一部使用されない正則化画像があります")

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    # bucketingを行わない場合も呼び出し必須（ひとつだけbucketを作る）
    def make_buckets_with_caching(self, enable_bucket, vae, min_size, max_size):
        self.enable_bucket = enable_bucket

        cache_latents = vae is not None
        if cache_latents:
            if enable_bucket:
                print("cache latents with bucketing")
            else:
                print("cache latents")
        else:
            if enable_bucket:
                print("make buckets")
            else:
                print("prepare dataset")

        if enable_bucket:
            bucket_resos, bucket_aspect_ratios = make_bucket_resolutions((self.width, self.height), min_size, max_size)
        else:
            bucket_resos = [(self.width, self.height)]
            bucket_aspect_ratios = [self.width / self.height]
        bucket_aspect_ratios = np.array(bucket_aspect_ratios)

        img_ar_errors = []
        for image_path, _ in tqdm(self.train_img_path_captions + self.reg_img_path_captions):
            if image_path in self.size_lat_cache:
                continue

            image = self.load_image(image_path)[0]
            image_height, image_width = image.shape[0:2]

            if not enable_bucket:
                reso = (self.width, self.height)
            else:
                aspect_ratio = image_width / image_height
                ar_errors = bucket_aspect_ratios - aspect_ratio
                bucket_id = np.abs(ar_errors).argmin()
                reso = bucket_resos[bucket_id]
                ar_error = ar_errors[bucket_id]
                img_ar_errors.append(ar_error)

                if cache_latents:
                    image = self.resize_and_trim(image, reso)

            if cache_latents:
                img_tensor = self.image_transforms(image)
                img_tensor = img_tensor.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
                latents = vae.encode(img_tensor).latent_dist.sample().squeeze(0).to("cpu")
            else:
                latents = None

            self.size_lat_cache[image_path] = (reso, latents)

        self.buckets = [[] for _ in range(len(bucket_resos))]
        reso_to_index = {}
        for i, reso in enumerate(bucket_resos):
            reso_to_index[reso] = i

        def split_to_buckets(is_reg, img_path_captions):
            for path, caption in img_path_captions:
                img_reso, _ = self.size_lat_cache[path]
                bi = reso_to_index[img_reso]
                self.buckets[bi].append((is_reg, path, caption))

        split_to_buckets(False, self.train_img_path_captions)

        if self.enable_reg_images:
            caps = []
            while len(caps) < len(self.train_img_path_captions):
                caps += self.reg_img_path_captions
            caps = caps[:len(self.train_img_path_captions)]
            split_to_buckets(True, caps)

        if enable_bucket:
            print("number of images with repeats / 繰り返し回数込みの各bucketの画像枚数")
            for i, (reso, images) in enumerate(zip(bucket_resos, self.buckets)):
                print(f"bucket {i}: resolution {reso}, count: {len(images)}")
            img_ar_errors = np.array(img_ar_errors)
            print(f"mean ar error: {np.mean(np.abs(img_ar_errors))}")

        for bucket_index, bucket in enumerate(self.buckets):
            batch_count = int(math.ceil(len(bucket) / self.batch_size))
            for batch_index in range(batch_count):
                self.buckets_indices.append((bucket_index, batch_index))

        self.shuffle_buckets()
        self._length = len(self.buckets_indices)

    def resize_and_trim(self, image, reso):
        image_height, image_width = image.shape[0:2]
        ar_img = image_width / image_height
        ar_reso = reso[0] / reso[1]
        if ar_img > ar_reso:  # 横が長い→縦を合わせる
            scale = reso[1] / image_height
        else:
            scale = reso[0] / image_width
        resized_size = (int(image_width * scale + .5), int(image_height * scale + .5))

        image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)  # INTER_AREAでやりたいのでcv2でリサイズ
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
        random.shuffle(self.buckets_indices)
        for bucket in self.buckets:
            random.shuffle(bucket)

    def load_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        img = np.array(image, np.uint8)

        face_cx = face_cy = face_w = face_h = 0
        if self.face_crop_aug_range is not None:
            tokens = os.path.splitext(os.path.basename(image_path))[0].split('_')
            if len(tokens) >= 5:
                face_cx = int(tokens[-4])
                face_cy = int(tokens[-3])
                face_w = int(tokens[-2])
                face_h = int(tokens[-1])

        return img, face_cx, face_cy, face_w, face_h

    def crop_target(self, image, face_cx, face_cy, face_w, face_h):
        height, width = image.shape[0:2]
        if height == self.height and width == self.width:
            return image

        face_size = max(face_w, face_h)
        min_scale = max(self.height / height, self.width / width)
        min_scale = min(1.0, max(min_scale, self.size / (face_size * self.face_crop_aug_range[1])))
        max_scale = min(1.0, max(min_scale, self.size / (face_size * self.face_crop_aug_range[0])))
        if min_scale >= max_scale:
            scale = min_scale
        else:
            scale = random.uniform(min_scale, max_scale)

        nh = int(height * scale + .5)
        nw = int(width * scale + .5)
        assert nh >= self.height and nw >= self.width, f"internal error. small scale {scale}, {width}*{height}"
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        face_cx = int(face_cx * scale + .5)
        face_cy = int(face_cy * scale + .5)
        height, width = nh, nw

        for axis, (target_size, length, face_p) in enumerate(
                zip((self.height, self.width), (height, width), (face_cy, face_cx))):
            p1 = face_p - target_size // 2

            if self.random_crop:
                c_range = max(length - face_p, face_p)
                p1 = p1 + (random.randint(0, c_range) + random.randint(0, c_range)) - c_range
            else:
                if self.face_crop_aug_range[0] != self.face_crop_aug_range[1]:
                    if face_size > self.size // 10 and face_size >= 40:
                        p1 = p1 + random.randint(-face_size // 20, +face_size // 20)

            p1 = max(0, min(p1, length - target_size))

            if axis == 0:
                image = image[p1:p1 + target_size, :]
            else:
                image = image[:, p1:p1 + target_size]

        return image

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index == 0:
            self.shuffle_buckets()

        bucket = self.buckets[self.buckets_indices[index][0]]
        image_index = self.buckets_indices[index][1] * self.batch_size

        latents_list = []
        images = []
        captions = []
        loss_weights = []

        for is_reg, image_path, caption in bucket[image_index:image_index + self.batch_size]:
            loss_weights.append(self.prior_loss_weight if is_reg else 1.0)

            reso, latents = self.size_lat_cache[image_path]

            if latents is None:
                img, face_cx, face_cy, face_w, face_h = self.load_image(image_path)
                im_h, im_w = img.shape[0:2]

                if self.enable_bucket:
                    img = self.resize_and_trim(img, reso)
                else:
                    if face_cx > 0:
                        img = self.crop_target(img, face_cx, face_cy, face_w, face_h)
                    elif im_h > self.height or im_w > self.width:
                        assert self.random_crop, f"Image too large, face_crop_aug_range and random_crop are disabled."
                        if im_h > self.height:
                            p = random.randint(0, im_h - self.height)
                            img = img[p:p + self.height]
                        if im_w > self.width:
                            p = random.randint(0, im_w - self.width)
                            img = img[:, p:p + self.width]

                    im_h, im_w = img.shape[0:2]
                    assert im_h == self.height and im_w == self.width, f"Image size is small: {image_path}"

                if self.aug is not None:
                    img = self.aug(image=img)['image']

                image = self.image_transforms(img)
            else:
                image = None

            images.append(image)
            latents_list.append(latents)

            if self.shuffle_caption:
                tokens = caption.strip().split(",")
                random.shuffle(tokens)
                caption = ",".join(tokens).strip()
            captions.append(caption)

        if self.disable_padding:
            input_ids = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").input_ids
        else:
            input_ids = self.tokenizer(captions, padding='max_length', truncation=True, return_tensors='pt').input_ids

        example = {'loss_weights': torch.FloatTensor(loss_weights), 'input_ids': input_ids}
        if images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format)
        else:
            images = None
        example['pixel_values'] = images
        example['latents'] = torch.stack(latents_list) if latents_list[0] is not None else None
        if self.debug_dataset:
            example['image_paths'] = [image_path for _, image_path, _ in
                                      bucket[image_index:image_index + self.batch_size]]
            example['captions'] = captions
        return example
