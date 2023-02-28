import json
import random
from dataclasses import dataclass, asdict
from typing import Tuple


@dataclass
class PromptData:
    def __init__(self,
                 prompt: str = "",
                 negative_prompt: str = "",
                 instance_token: str = "",
                 class_token: str = "",
                 src_image: str = "",
                 steps: int = 40,
                 scale: float = 7.5,
                 out_dir: str = "",
                 seed: int = -1,
                 resolution: Tuple[int, int] = (512, 512),
                 concept_index: int = 0
                 ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.instance_token = instance_token
        self.class_token = class_token
        self.src_image = src_image
        self.steps = steps
        self.scale = scale
        self.out_dir = out_dir
        if seed == -1:
            seed = int(random.randrange(21474836147))
        self.seed = seed
        self.resolution = resolution
        self.concept_index = concept_index

    @property
    def __dict__(self):
        """
        get a python dictionary
        """
        return asdict(self)

    @property
    def json(self):
        """
        get the json formated string
        """
        return json.dumps(self.__dict__)
