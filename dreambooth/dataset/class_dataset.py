import os
import random
from pathlib import Path

from torch.utils.data import Dataset

from extensions.sd_dreambooth_extension.dreambooth import shared
from extensions.sd_dreambooth_extension.dreambooth.dataclasses.db_concept import Concept
from extensions.sd_dreambooth_extension.dreambooth.shared import status
from extensions.sd_dreambooth_extension.dreambooth.utils.image_utils import FilenameTextGetter, make_bucket_resolutions, \
    sort_prompts
from extensions.sd_dreambooth_extension.helpers.mytqdm import mytqdm
from extensions.sd_dreambooth_extension.dreambooth.dataclasses.prompt_data import PromptData


class ClassDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, concepts: [Concept], model_dir: str, max_width:int, shuffle: bool):
        # Existing training image data
        self.instance_prompts = []
        # Existing class image data
        self.class_prompts = []
        # Data for new prompts to generate
        self.new_prompts = {}
        self.required_prompts = 0
        # Calculate minimum width
        min_width = (int(max_width * 0.28125) // 64) * 64

        # Thingy to build prompts
        text_getter = FilenameTextGetter(shuffle)

        # Create available resolutions
        bucket_resos = make_bucket_resolutions(max_width, min_width)
        c_idx = 0

        for concept in concepts:
            instance_dir = concept.instance_data_dir
            if not concept.is_valid:
                continue
            class_dir = concept.class_data_dir
            # Filter empty class dir and set/create if necessary
            if class_dir == "" or class_dir is None or class_dir == shared.script_path:
                class_dir = os.path.join(model_dir, f"classifiers_{c_idx}")
            class_dir = Path(class_dir)
            class_dir.mkdir(parents=True, exist_ok=True)

            status.textinfo = "Sorting images..."
            # Sort existing prompts
            class_prompt_datas = {}
            instance_prompt_datas = sort_prompts(concept, text_getter, instance_dir, bucket_resos, c_idx, False)
            if concept.num_class_images_per > 0 and class_dir:
                class_prompt_datas = sort_prompts(concept, text_getter, str(class_dir), bucket_resos, c_idx, True)

            print(f"Concept requires {concept.num_class_images_per} class images per instance image.")

            # Iterate over each resolution of images, per concept
            for res, i_prompt_datas in mytqdm(instance_prompt_datas.items(), desc="Sorting instance images"):
                # Extend instance prompts by the instance data
                self.instance_prompts.extend(i_prompt_datas)

                classes_per_bucket = len(i_prompt_datas) * concept.num_class_images_per
                # Don't do anything else if we don't need class images

                if concept.num_class_images_per == 0 or classes_per_bucket == 0:
                    continue

                # Get class prompt list, if it exists
                c_prompt_datas = class_prompt_datas[res] if res in class_prompt_datas.keys() else []

                # We may not need this, so initialize it here
                new_prompts = []

                # If we have enough or more classes already, randomly select the required amount
                if len(c_prompt_datas) >= classes_per_bucket:
                    c_prompt_datas = random.sample(c_prompt_datas, classes_per_bucket)

                # Otherwise, generate and append new class images
                else:
                    class_prompts = [img.prompt for img in c_prompt_datas]
                    instance_prompts = [img.prompt for img in i_prompt_datas]

                    for prompt in instance_prompts:
                        sample_prompt = text_getter.create_text(
                            concept.class_prompt, prompt, concept.instance_token, concept.class_token, True)
                        num_to_gen = concept.num_class_images_per - class_prompts.count(sample_prompt)
                        for _ in range(num_to_gen):
                            pd = PromptData(
                                prompt=sample_prompt,
                                negative_prompt=concept.class_negative_prompt,
                                instance_token=concept.instance_token,
                                class_token=concept.class_token,
                                steps=concept.class_infer_steps,
                                scale=concept.class_guidance_scale,
                                out_dir=class_dir,
                                seed=-1,
                                concept_index=c_idx,
                                resolution=res)
                            new_prompts.append(pd)
                            # BAD BAD BAD. Need to append this after generating, so we have the output path
                            # c_prompt_datas.append(pd)

                # Extend class prompts by the proper amount
                self.class_prompts.extend(c_prompt_datas)

                if len(new_prompts):
                    self.required_prompts += len(new_prompts)
                    if res in self.new_prompts:
                        self.new_prompts[res].extend(new_prompts)
                    else:
                        self.new_prompts[res] = new_prompts
            c_idx += 1

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
