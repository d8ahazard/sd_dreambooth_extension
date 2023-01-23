import os.path
import random
from typing import List

from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig
from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import FilenameTextGetter
from extensions.sd_dreambooth_extension.dreambooth.prompt_data import PromptData
from extensions.sd_dreambooth_extension.dreambooth.utils import get_images


class SampleDataset:
    def __init__(self, config: DreamboothConfig):
        concepts = config.concepts()
        shuffle_tags = config.shuffle_tags
        valid_concepts = []
        prompts = {}
        c_idx = 0
        for concept in concepts:
            concept_prompts = []
            concept_seed = concept.sample_concept_seed
            if concept.instance_data_dir == "" or concept.instance_data_dir is None:
                continue
            if concept.n_save_sample == 0:
                continue
            valid_concepts.append(concept)
            sample_base = concept.save_sample_prompt
            sample_file = concept.save_sample_template
            # If no sample file, look for filewords
            if sample_file != "" and sample_file is not None and os.path.exists(sample_file):
                with open(sample_file, "r") as samples:
                    lines = samples.readlines()
                    for line in lines:
                        if concept_seed is None or concept_seed == '' or concept_seed == -1:
                            seed = int(random.randrange(21474836147))
                        else:
                            seed = concept_seed
                        if line.strip() != "":
                            pd = PromptData()
                            pd.concept_seed = seed
                            pd.seed = seed
                            pd.prompt = line.strip()
                            pd.negative_prompt = concept.save_sample_negative_prompt
                            concept_prompts.append(pd)
            else:
                if "[filewords]" in sample_base:
                    images = get_images(concept.instance_data_dir)
                    getter = FilenameTextGetter(shuffle_tags)
                    for image in images:
                        base = getter.read_text(image)
                        prompt = getter.create_text(sample_base,
                                                    base,
                                                    concept.instance_token,
                                                    concept.class_token,
                                                    False
                                                    )
                        pd = PromptData()
                        if concept_seed is None or concept_seed == '' or concept_seed == -1:
                            seed = int(random.randrange(21474836147))
                        else:
                            seed = concept_seed
                        pd.concept_seed = seed
                        pd.seed = seed
                        pd.prompt = prompt
                        pd.negative_prompt = concept.save_sample_negative_prompt
                        concept_prompts.append(pd)
                else:
                    if concept_seed is None or concept_seed == '' or concept_seed == -1:
                        seed = int(random.randrange(21474836147))
                    else:
                        seed = concept_seed
                    pd = PromptData()
                    pd.concept_seed = seed
                    pd.seed = seed
                    pd.prompt = sample_base
                    pd.negative_prompt = concept.save_sample_negative_prompt
                    concept_prompts.append(pd)
            random.shuffle(concept_prompts)
            prompts[c_idx] = concept_prompts
            c_idx += 1
        self.prompts = prompts
        self.concepts = valid_concepts

    def get_prompts(self) -> List[PromptData]:
        concept_index = 0
        output = []
        for concept in self.concepts:
            num_images = concept.n_save_sample
            for idx in range(num_images):
                concept_prompt = random.choice(self.prompts[concept_index])
                output.append(concept_prompt)
            concept_index += 1
        return output




