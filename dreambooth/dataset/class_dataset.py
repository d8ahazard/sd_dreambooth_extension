import os
import random

from torch.utils.data import Dataset

from dreambooth import shared
from dreambooth.dataclasses.db_concept import Concept
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.shared import status
from dreambooth.utils.image_utils import FilenameTextGetter, \
    make_bucket_resolutions, \
    sort_prompts, get_images
from helpers.mytqdm import mytqdm


class ClassDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, concepts: [Concept], model_dir: str, max_width: int, shuffle: bool):
        # Existing training image data
        self.instance_prompts = []
        # Existing class image data
        self.class_prompts = []
        # Data for new prompts to generate
        self.new_prompts = {}
        self.required_prompts = 0

        # Thingy to build prompts
        text_getter = FilenameTextGetter(shuffle)

        # Create available resolutions
        bucket_resos = make_bucket_resolutions(max_width)
        class_images = {}
        instance_images = {}
        total_images = 0
        for concept_idx, concept in enumerate(concepts):
            if not concept.is_valid:
                continue

            instance_dir = concept.instance_data_dir
            class_dir = concept.class_data_dir
            # Filter empty class dir and set/create if necessary
            if class_dir == "" or class_dir is None or class_dir == shared.script_path:
                class_dir = os.path.join(model_dir, f"classifiers_{concept_idx}")
            os.makedirs(class_dir, exist_ok=True)
            instance_images[concept_idx] = get_images(instance_dir)
            class_images[concept_idx] = get_images(class_dir)
            total_images += len(instance_images[concept_idx])
            total_images += len(class_images[concept_idx])

        status.textinfo = "Sorting images..."
        pbar = mytqdm(desc="Pre-processing images.", position=0)
        pbar.reset(total_images)

        for concept_idx, concept in enumerate(concepts):
            if not concept.is_valid:
                continue

            class_dir = concept.class_data_dir
            # Filter empty class dir and set/create if necessary
            if class_dir == "" or class_dir is None or class_dir == shared.script_path:
                class_dir = os.path.join(model_dir, f"classifiers_{concept_idx}")

            # ===== Instance =====
            instance_prompt_buckets = sort_prompts(concept, text_getter, instance_dir, instance_images[concept_idx], bucket_resos, concept_idx, False, pbar)
            for _, instance_prompt_datas in instance_prompt_buckets.items():
                # Extend instance prompts by the instance data
                self.instance_prompts.extend(instance_prompt_datas)

            # ===== Class =====
            if concept.num_class_images_per <= 0 or not class_dir:
                continue

            required_prompt_buckets = sort_prompts(concept, text_getter, class_dir, instance_images[concept_idx], bucket_resos, concept_idx, True, pbar)
            existing_prompt_buckets = sort_prompts(concept, text_getter, class_dir, class_images[concept_idx], bucket_resos, concept_idx, True, pbar, True)

            # Iterate over each resolution of images, per concept
            for res, required_prompt_datas in required_prompt_buckets.items():
                classes_per_bucket = len(required_prompt_datas) * concept.num_class_images_per
                if classes_per_bucket == 0:
                    continue

                new_prompts_datas = []
                existing_prompt_datas = existing_prompt_buckets[res] if res in existing_prompt_buckets.keys() else []

                # If we have enough or more classes already, randomly select the required amount
                if len(existing_prompt_datas) >= classes_per_bucket:
                    existing_prompt_datas = random.sample(existing_prompt_datas, classes_per_bucket)

                # Otherwise, generate and append new class images
                else:
                    existing_prompts = [img.prompt for img in existing_prompt_datas]
                    required_prompts = [img.prompt for img in required_prompt_datas]

                    for prompt in required_prompts:
                        num_to_gen = concept.num_class_images_per * required_prompts.count(prompt) - existing_prompts.count(prompt)
                        for _ in range(num_to_gen):
                            pd = PromptData(
                                prompt=prompt,
                                negative_prompt=concept.class_negative_prompt,
                                instance_token=concept.instance_token,
                                class_token=concept.class_token,
                                steps=concept.class_infer_steps,
                                scale=concept.class_guidance_scale,
                                out_dir=class_dir,
                                concept_index=concept_idx,
                                resolution=res)
                            existing_prompts.append(prompt)
                            new_prompts_datas.append(pd)

                # Extend class prompts by the proper amount
                self.class_prompts.extend(existing_prompt_datas)

                if len(new_prompts_datas):
                    self.required_prompts += len(new_prompts_datas)
                    if res in self.new_prompts:
                        self.new_prompts[res].extend(new_prompts_datas)
                    else:
                        self.new_prompts[res] = new_prompts_datas

        pbar.reset(0)
        if self.required_prompts > 0:
            print(f"We need a total of {self.required_prompts} class images.")

    def __len__(self) -> int:
        return self.required_prompts

    def __getitem__(self, index) -> PromptData:
        res_index = 0
        for res, prompt_datas in self.new_prompts.items():
            for p in range(len(prompt_datas)):
                if res_index == index:
                    return prompt_datas[p]
                res_index += 1
        print(f"Invalid index: {index}/{self.required_prompts}")
        return None
