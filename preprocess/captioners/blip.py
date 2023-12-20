from typing import Dict

import torch
from PIL.Image import Image

from preprocess.captioners.base import BaseCaptioner

model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class BlipCaptioner(BaseCaptioner):
    def __init__(self):
        super().__init__(None)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from lavis.models import load_model_and_preprocess
        self.model, self.processor, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco",
                                                                       is_eval=True, device=device)

    def caption(self, image: Image, params: Dict = None, unload: bool = False) -> str:
        self._to_gpu()
        raw_image = image.resize((512, 512))
        blip_image = self.processor["eval"](raw_image).unsqueeze(0).to(self.device)
        if unload:
            self._to_cpu()
        return self.model.generate({"image": blip_image})
