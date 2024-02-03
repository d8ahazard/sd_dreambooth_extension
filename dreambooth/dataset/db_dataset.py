import json
import logging
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

logger = logging.getLogger(__name__)


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
            tokenizer: Union[CLIPTokenizer, List[CLIPTokenizer], None],
            text_encoder,
            accelerator,
            resolution: int,
            hflip: bool,
            do_shuffle_tags: bool,
            strict_tokens: bool,
            dynamic_img_norm: bool,
            not_pad_tokens: bool,
            max_token_length: int,
            debug_dataset: bool,
            model_dir: str,
            pbar: mytqdm = None
    ) -> None:
        super().__init__()
        self.batch_indices = []
        self.batch_samples = []
        self.class_count = 0
        self.max_token_length = max_token_length
        self.model_dir = model_dir
        self.cache_dir = os.path.join(model_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        print("Init dataset!")
        # A dictionary of string/latent pairs matching image paths
        self.data_cache = {"captions": {}, "latents": {}, "sdxl": {}}
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
        self.train_img_data = instance_prompts
        self.class_img_data = class_prompts
        self.num_train_images = len(self.train_img_data)
        self.num_class_images = len(self.class_img_data)

        self.tokenizers = []
        if isinstance(tokenizer, CLIPTokenizer):
            self.tokenizers = [tokenizer]
        elif isinstance(tokenizer, list):
            self.tokenizers = tokenizer
        self.text_encoders = text_encoder
        self.accelerator = accelerator
        self.resolution = resolution
        self.debug_dataset = debug_dataset
        self.shuffle_tags = do_shuffle_tags
        self.not_pad_tokens = not_pad_tokens
        self.strict_tokens = strict_tokens
        self.dynamic_img_norm = dynamic_img_norm
        self.tokens = tokens
        self.vae = None
        self.pbar = pbar
        self.cache_latents = False
        flip_p = 0.5 if hflip else 0.0
        self.image_transforms = self.build_compose(hflip, flip_p)

    @staticmethod
    def load_cache_file(cache_dir, resolution):
        cache_file = os.path.join(cache_dir, f"cache_{resolution}.safetensors")
        latents_cache = {}
        if os.path.exists(cache_file):
            print("Loading latent cache...")
            latents_cache = safetensors.torch.load_file(cache_file)

        data_cache = {}
        for key, value in latents_cache.items():
            sub_keys = key.split("||")
            parent_key = sub_keys[0]
            element_key = sub_keys[1]

            if parent_key not in data_cache:
                data_cache[parent_key] = {}

            if parent_key == "sdxl":
                if len(sub_keys) != 3:
                    logger.warning(f"Skipping invalid key: {key}")
                    continue
                main_key = element_key
                subkey_type = sub_keys[2]
                if main_key not in data_cache[parent_key]:
                    data_cache[parent_key][main_key] = [None, {"text_embeds": None, "time_ids": None}]

                if subkey_type == "prompt_embeds":
                    data_cache[parent_key][main_key][0] = value
                elif subkey_type == "text_embeds":
                    data_cache[parent_key][main_key][1]["text_embeds"] = value
                elif subkey_type == "time_ids":
                    data_cache[parent_key][main_key][1]["time_ids"] = value
            else:
                data_cache[parent_key][element_key] = value

        for key_name in ["latents", "sdxl", "captions"]:
            if key_name not in data_cache:
                data_cache[key_name] = {}

        return data_cache

    def save_cache_file(self, data_cache):
        cache_file = os.path.join(self.cache_dir, f"cache_{self.resolution}.safetensors")

        try:
            keys_to_check = ["latents", "sdxl", "captions"]
            do_save = False
            for key_check in keys_to_check:
                if key_check not in self.data_cache:
                    self.data_cache[key_check] = {}
                if key_check not in data_cache:
                    data_cache[key_check] = {}
                if set(key for key, value in self.data_cache[key_check].items()) != set(
                        key for key, value in data_cache[key_check].items()):
                    do_save = True
                    break

            if do_save:
                print("Saving cache!")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                full_dict = {}
                for key in keys_to_check:
                    if key in self.data_cache:
                        for key2, value in self.data_cache[key].items():
                            if key == "sdxl":
                                embeds, extras = value
                                full_dict[f"{key}||{key2}||prompt_embeds"] = embeds
                                full_dict[f"{key}||{key2}||text_embeds"] = extras["text_embeds"]
                                full_dict[f"{key}||{key2}||time_ids"] = extras["time_ids"]
                            else:
                                combined_key = f"{key}||{key2}"
                                full_dict[combined_key] = value

                safetensors.torch.save_file(full_dict, cache_file)
        except:
            logger.error("Error saving cache!")
            traceback.print_exc()

    @staticmethod
    def build_compose(hflip, flip_p):
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

    def encode_prompt(self, prompt):
        prompt_embeds_list = []
        pooled_prompt_embeds = None  # default declaration
        bs_embed = None  # default declaration

        auto_add_special_tokens = False if self.strict_tokens else True
        if self.shuffle_tags:
            prompt = shuffle_tags(prompt)
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            if self.strict_tokens:
                prompt = build_strict_tokens(prompt, tokenizer.bos_token, tokenizer.eos_token)

            b_size = 1  # as we are working with a single prompt
            n_size = 1 if self.max_token_length is None else self.max_token_length // 75

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                add_special_tokens=auto_add_special_tokens,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.view(-1,
                                                        tokenizer.model_max_length)  # reshape to handle different token lengths

            enc_out = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
                return_dict=True
            )

            # get hidden states and handle reshaping
            prompt_embeds = enc_out["hidden_states"][-2]  # penultimate layer
            prompt_embeds = prompt_embeds.reshape(
                (b_size, -1, prompt_embeds.shape[-1]))  # reshape to handle different token lengths

            # handle varying max token lengths
            if self.max_token_length is not None:
                states_list = [prompt_embeds[:, 0].unsqueeze(1)]
                for i in range(1, self.max_token_length, tokenizer.model_max_length):
                    states_list.append(prompt_embeds[:, i: i + tokenizer.model_max_length - 2])
                states_list.append(prompt_embeds[:, -1].unsqueeze(1))
                prompt_embeds = torch.cat(states_list, dim=1)

            if "text_embeds" not in enc_out:
                # Thanks autopilot!
                pooled_prompt_embeds = enc_out["pooler_output"]
            else:
                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = enc_out["text_embeds"]
            if self.max_token_length is not None:
                pooled_prompt_embeds = pooled_prompt_embeds[::n_size]

            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    def compute_embeddings(self, reso, prompt):
        original_size = reso
        target_size = reso
        crops_coords_top_left = (0, 0)
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt)
            add_text_embeds = pooled_prompt_embeds

            # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])

            prompt_embeds = prompt_embeds.to(self.accelerator.device)
            add_text_embeds = add_text_embeds.to(self.accelerator.device)
            add_time_ids = add_time_ids.to(self.accelerator.device, dtype=prompt_embeds.dtype)
            unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        return prompt_embeds, unet_added_cond_kwargs

    def load_image(self, image_path, caption, res):
        if self.debug_dataset:
            image = os.path.splitext(image_path)
            input_ids = caption
        else:
            if self.cache_latents:
                image = self.data_cache["latents"][image_path]
            else:
                img = open_and_trim(image_path, res, False)
                image = self.image_transform(img)
            if self.shuffle_tags:
                caption, input_ids = self.cache_caption(image_path, caption)
            else:
                input_ids = self.data_cache["captions"][image_path]
        return image, input_ids

    def cache_latent(self, image_path, res):
        if self.vae is not None:
            image = open_and_trim(image_path, res, False)
            img_tensor = self.image_transform(image)
            img_tensor = img_tensor.unsqueeze(0).to(device=self.vae.device, dtype=self.vae.dtype)
            latents = self.vae.encode(img_tensor).latent_dist.sample().squeeze(0).to("cpu")
            self.data_cache["latents"][image_path] = latents

    def cache_caption(self, image_path, caption):
        input_ids = None
        auto_add_special_tokens = False if self.strict_tokens else True
        if len(self.tokenizers) > 0 and (image_path not in self.data_cache["captions"] or self.debug_dataset):
            if self.shuffle_tags:
                caption = shuffle_tags(caption)
            if self.strict_tokens:
                caption = build_strict_tokens(caption, self.tokenizers[0].bos_token, self.tokenizers[0].eos_token)
            if self.not_pad_tokens:
                input_ids = self.tokenizers[0](caption, padding=True, truncation=True,
                                               add_special_tokens=auto_add_special_tokens,
                                               return_tensors="pt").input_ids
            else:
                input_ids = self.tokenizers[0](caption, padding='max_length', truncation=True,
                                               add_special_tokens=auto_add_special_tokens,
                                               return_tensors='pt').input_ids
            if not self.shuffle_tags:
                self.data_cache["captions"][image_path] = input_ids

        return caption, input_ids

    def make_buckets_with_caching(self, vae, data_cache = None):
        self.vae = vae
        self.cache_latents = vae is not None
        state = f"Preparing Dataset ({'With Caching' if self.cache_latents else 'Without Caching'})"
        print(state)
        if self.pbar is not None:
            self.pbar.set_description(state)
        status.textinfo = state

        # Create a list of resolutions
        bucket_resolutions = make_bucket_resolutions(self.resolution)
        self.train_dict = {}

        def sort_images(img_data: List[PromptData], resolutions, target_dict, is_class_img):
            for prompt_data in img_data:
                path = prompt_data.src_image
                image_width, image_height = prompt_data.resolution
                cap = prompt_data.prompt
                reso = closest_resolution(image_width, image_height, resolutions)
                concept_idx = prompt_data.concept_index
                # Append the concept index to the resolution, and boom, we got ourselves split concepts.
                di = (*reso, concept_idx)
                target_dict.setdefault(di, []).append((path, cap, is_class_img))

        sort_images(self.train_img_data, bucket_resolutions, self.train_dict, False)
        sort_images(self.class_img_data, bucket_resolutions, self.class_dict, True)
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
        if data_cache == None:
            data_cache = DbDataset.load_cache_file(self.cache_dir, self.resolution) if self.cache_latents else {"captions": {}, "latents": {}, "sdxl": {}}
        has_cache = len(data_cache) > 0
        if self.cache_latents:
            if has_cache:
                bar_description = "Loading cached latents..."
            else:
                bar_description = "Caching latents..."
        else:
            bar_description = "Processing images..."
        if self.pbar is None:
            self.pbar = mytqdm(range(p_len), desc=bar_description, position=0)
        else:
            self.pbar.reset(total=p_len)
            self.pbar.set_description(bar_description)
        self.pbar.status_index = 1

        def cache_images(images, reso, p_bar: mytqdm):
            if "captions" not in data_cache:
                data_cache["captions"] = {}
            if "latents" not in data_cache:
                data_cache["latents"] = {}
            for img_path, cap, is_prior in images:
                try:
                    # If the image is not in the "precache",cache it
                    if self.cache_latents:
                        if img_path not in data_cache["latents"] and not self.debug_dataset:
                            self.cache_latent(img_path, reso)
                        else:
                            self.data_cache["latents"][img_path] = data_cache["latents"][img_path]

                    if not self.shuffle_tags:
                        if img_path not in data_cache["captions"] and not self.debug_dataset:
                            self.cache_caption(img_path, cap)
                        else:
                            self.data_cache["captions"][img_path] = data_cache["captions"][img_path]

                    # This likely needs to happen regardless of cache_latents?
                    if len(self.tokenizers) == 2:
                        if img_path not in data_cache["sdxl"]:
                            embeds, extras = self.compute_embeddings(reso, cap)
                            self.data_cache["sdxl"][img_path] = (embeds, extras)
                        else:
                            self.data_cache["sdxl"][img_path] = data_cache["sdxl"][img_path]

                    self.sample_indices.append(img_path)
                    self.sample_cache.append((img_path, cap, is_prior))
                    p_bar.update()
                except Exception as e:
                    traceback.print_exc()
                    print(f"Exception caching: {img_path}: {e}")
                    if img_path in self.data_cache["captions"]:
                        del self.data_cache["captions"][img_path]
                    if img_path in self.data_cache["latents"]:
                        del self.data_cache["latents"][img_path]
                    if img_path in self.data_cache["sdxl"]:
                        del self.data_cache["sdxl"][img_path]
                    if (img_path, cap, is_prior) in self.sample_cache:
                        self.sample_cache.remove((img_path, cap, is_prior))
                    if img_path in self.sample_indices:
                        del self.sample_indices[img_path]

        bucket_dict = {}

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
            class_count = 0
            if dict_idx in self.class_dict:
                # Use dict index to find class images
                class_images = self.class_dict[dict_idx]
                # Use actual res here as well
                cache_images(class_images, res, self.pbar)
                class_count = len(class_images)
            total_instances += inst_count
            total_classes += class_count
            example_len = inst_count if class_count == 0 else inst_count * 2
            # Use index here, not res
            bucket_len[dict_idx] = example_len
            total_len += example_len
            bucket_dict[f"{dict_idx}"] = {
                "resolution": [dict_idx[0], dict_idx[1]],
                "count": inst_count + class_count
            }
            bucket_str = str(bucket_idx).rjust(max_idx_chars, " ")
            inst_str = str(len(train_images)).rjust(len(str(ni)), " ")
            class_str = str(class_count).rjust(len(str(nc)), " ")
            ex_str = str(example_len).rjust(len(str(ti * 2)), " ")
            # Log both here
            self.pbar.write(
                f"Bucket {bucket_str} {dict_idx} - Instance Images: {inst_str} | Class Images: {class_str} | Max Examples/batch: {ex_str}")
            bucket_idx += 1
        bucket_array = {"buckets": bucket_dict}
        bucket_json_file = os.path.join(self.model_dir, "bucket_counts.json")
        with open(bucket_json_file, "w") as f:
            f.write(json.dumps(bucket_array, indent=4))
        self.save_cache_file(data_cache)
        del data_cache
        bucket_str = str(bucket_idx).rjust(max_idx_chars, " ")
        inst_str = str(total_instances).rjust(len(str(ni)), " ")
        class_str = str(total_classes).rjust(len(str(nc)), " ")
        tot_str = str(total_len).rjust(len(str(ti)), " ")
        self.class_count = total_classes
        self.pbar.write(
            f"Total Buckets {bucket_str} - Instance Images: {inst_str} | Class Images: {class_str} | Max Examples/batch: {tot_str}")
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
        example = {}
        image_path, caption, is_class_image = self.sample_cache[index]
        if not self.debug_dataset:
            image_data, input_ids, = self.load_image(image_path, caption, self.active_resolution)
            if len(self.tokenizers) > 1:
                if image_path in self.data_cache["sdxl"]:
                    input_ids, added_conditions = self.data_cache["sdxl"][image_path]
                else:
                    input_ids, added_conditions = self.compute_embeddings(self.active_resolution, caption)
                    self.data_cache["sdxl"][image_path] = (input_ids, added_conditions)
                example["instance_added_cond_kwargs"] = added_conditions
        else:
            image_data = image_path
            caption, cap_tokens = self.cache_caption(image_path, caption)
            rebuilt = self.tokenizers[0].decode(cap_tokens.tolist()[0])
            input_ids = (caption, rebuilt)

        example["input_ids"] = input_ids
        example["image"] = image_data
        example["res"] = self.active_resolution
        example["is_class"] = is_class_image

        return example
