import json
import os
from typing import Dict, Union

from pydantic import BaseModel

from extensions.sd_dreambooth_extension.dreambooth.utils import get_images


class Concept(BaseModel):
    class_data_dir: str = ""
    class_guidance_scale: float = 7.5
    class_infer_steps: int = 60
    class_negative_prompt: str = ""
    class_prompt: str = ""
    class_token: str = ""
    instance_data_dir: str = ""
    instance_prompt: str = ""
    instance_token: str = ""
    is_valid: bool = False
    n_save_sample: int = 1
    num_class_images: int = 0
    num_class_images_per: int = 0
    sample_seed: int = -1
    save_guidance_scale: float = 7.5
    save_infer_steps: int = 60
    save_sample_negative_prompt: str = ""
    save_sample_prompt: str = ""
    save_sample_template: str = ""

    def __init__(self, input_dict: Union[Dict, None] = None, **kwargs):
        super().__init__(**kwargs)
        if input_dict is not None:
            self.load_params(input_dict)
        if self.is_valid and self.num_class_images != 0:
            if self.num_class_images_per == 0:
                images = get_images(self.instance_data_dir)
                if len(images) < self.num_class_images * 2:
                    self.num_class_images_per = 1
                else:
                    self.num_class_images_per = self.num_class_images // len(images)
                self.num_class_images = 0

    def to_dict(self):
        return self.dict()

    def to_json(self):
        return json.dumps(self.to_dict())

    def load_params(self, params_dict):
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if self.instance_data_dir:
                self.is_valid = os.path.isdir(self.instance_data_dir)
            else:
                self.is_valid = False

