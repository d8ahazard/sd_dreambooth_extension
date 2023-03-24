import os.path
import random

from PIL import Image

from dreambooth.dataclasses.db_config import DreamboothConfig
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.utils.image_utils import get_images, FilenameTextGetter, \
    closest_resolution, make_bucket_resolutions


class SampleDataset:
    """
    Dataset for generating prompts
    """

    def __init__(self, config: DreamboothConfig):
        concepts = config.concepts()
        shuffle_tags = config.shuffle_tags
        self.prompts = []
        c_idx = 0
        bucket_resos = make_bucket_resolutions(config.resolution)
        out_dir = os.path.join(config.model_dir, "samples")
        for concept in concepts:
            required = concept.n_save_sample
            if concept.instance_data_dir == "" or concept.instance_data_dir is None or required == 0:
                continue

            sample_prompt = concept.save_sample_prompt
            sample_file = concept.save_sample_template
            seed = concept.sample_seed
            neg = concept.save_sample_negative_prompt
            # If no sample file, look for filewords
            if sample_file and os.path.exists(sample_file):
                with open(sample_file, "r") as samples:
                    lines = samples.readlines()
                    prompts = [(line, (config.resolution, config.resolution)) for line in lines if line.strip()]
            elif "[filewords]" in sample_prompt:
                prompts = []
                images = get_images(concept.instance_data_dir)
                getter = FilenameTextGetter(shuffle_tags)
                for image in images:
                    file_text = getter.read_text(image)
                    prompt = getter.create_text(sample_prompt, file_text, concept, False)
                    img = Image.open(image)
                    res = img.size
                    closest = closest_resolution(res[0], res[1], bucket_resos)
                    prompts.append((prompt, closest))
            else:
                prompts = [(sample_prompt, (config.resolution, config.resolution))]
            for i in range(required):
                pi = random.choice(prompts)
                pd = PromptData(
                    prompt=pi[0],
                    negative_prompt=neg,
                    steps=concept.save_infer_steps,
                    scale=concept.save_guidance_scale,
                    seed=seed,
                    out_dir=out_dir,
                    concept_index=c_idx,
                    resolution=pi[1]
                )
                self.prompts.append(pd)
            c_idx += 1
