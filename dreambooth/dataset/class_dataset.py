import os
import random

from torch.utils.data import Dataset

try:
    from extensions.sd_dreambooth_extension.dreambooth import shared
    from extensions.sd_dreambooth_extension.dreambooth.dataclasses.db_concept import Concept
    from extensions.sd_dreambooth_extension.dreambooth.dataclasses.prompt_data import PromptData
    from extensions.sd_dreambooth_extension.dreambooth.shared import status
    from extensions.sd_dreambooth_extension.dreambooth.utils.image_utils import FilenameTextGetter, \
        make_bucket_resolutions, \
        sort_prompts, get_images
    from extensions.sd_dreambooth_extension.helpers.mytqdm import mytqdm
except:
    from dreambooth.dreambooth import shared  # noqa
    from dreambooth.dreambooth.dataclasses.db_concept import Concept  # noqa
    from dreambooth.dreambooth.dataclasses.prompt_data import PromptData  # noqa
    from dreambooth.dreambooth.shared import status  # noqa
    from dreambooth.dreambooth.utils.image_utils import FilenameTextGetter, make_bucket_resolutions, sort_prompts, get_images  # noqa
    from dreambooth.helpers.mytqdm import mytqdm  # noqa


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
        # Calculate minimum width
        min_width = (int(max_width * 0.28125) // 64) * 64

        # Thingy to build prompts
        text_getter = FilenameTextGetter(shuffle)

        # Create available resolutions
        bucket_resos = make_bucket_resolutions(max_width, min_width)
        c_idx = 0
        c_images = {}
        i_images = {}
        total_images = 0
        for concept in concepts:
            if not concept.is_valid:
                continue
            instance_dir = concept.instance_data_dir
            class_dir = concept.class_data_dir
            # Filter empty class dir and set/create if necessary
            if class_dir == "" or class_dir is None or class_dir == shared.script_path:
                class_dir = os.path.join(model_dir, f"classifiers_{c_idx}")
            os.makedirs(class_dir, exist_ok=True)
            i_images[c_idx] = get_images(instance_dir)
            c_images[c_idx] = get_images(class_dir)
            total_images += len(i_images[c_idx])
            total_images += len(c_images[c_idx])
            c_idx += 1

        c_idx = 0
        pbar = mytqdm(desc="Pre-processing images.")
        pbar.reset(total_images)
        for concept in concepts:
            instance_dir = concept.instance_data_dir
            if not concept.is_valid:
                continue
            class_dir = concept.class_data_dir
            # Filter empty class dir and set/create if necessary
            if class_dir == "" or class_dir is None or class_dir == shared.script_path:
                class_dir = os.path.join(model_dir, f"classifiers_{c_idx}")

            status.textinfo = "Sorting images..."
            # Sort existing prompts
            class_prompt_datas = {}
            instance_prompt_datas = sort_prompts(concept, text_getter, instance_dir, i_images[c_idx], bucket_resos,
                                                 c_idx, False, pbar)
            if concept.num_class_images_per > 0 and class_dir:
                class_prompt_datas = sort_prompts(concept, text_getter, class_dir, c_images[c_idx], bucket_resos, c_idx,
                                                  True, pbar)

            # Iterate over each resolution of images, per concept
            for res, i_prompt_datas in instance_prompt_datas.items():
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

                    if "[filewords]" in concept.class_prompt:
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
                    else:
                        sample_prompt = text_getter.create_text(
                            concept.class_prompt, "", concept.instance_token, concept.class_token, True)
                        num_to_gen = concept.num_class_images_per * len(i_prompt_datas) - class_prompts.count(
                            sample_prompt)
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

                # Extend class prompts by the proper amount
                self.class_prompts.extend(c_prompt_datas)

                if len(new_prompts):
                    self.required_prompts += len(new_prompts)
                    if res in self.new_prompts:
                        self.new_prompts[res].extend(new_prompts)
                    else:
                        self.new_prompts[res] = new_prompts
            c_idx += 1
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
