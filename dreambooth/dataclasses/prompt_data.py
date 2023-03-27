import json
import random
from dataclasses import dataclass, asdict
from typing import Tuple


@dataclass
class PromptData:
    prompt: str = ""
    negative_prompt: str = ""
    instance_token: str = ""
    class_token: str = ""
    src_image: str = ""
    steps: int = 40
    scale: float = 7.5
    out_dir: str = ""
    seed: int = -1
    resolution: Tuple[int, int] = (512, 512)
    concept_index: int = 0

    def __post_init__(self):
        if self.seed == -1:
            self.seed = int(random.randrange(0, 21474836147))

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
