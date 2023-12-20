import gc
import logging
from typing import Dict

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from preprocess.captioners.base import BaseCaptioner

logger = logging.getLogger(__name__)


class BlipLargeCaptioner(BaseCaptioner):
    model = None
    processor = None

    def __init__(self):
        super().__init__(None)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    def caption(self, image: Image, params: Dict = None, unload: bool = True):
        self._to_gpu()
        raw_image = image.convert('RGB')
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)

        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        logger.debug(f"Caption: {caption}")
        cap_words = caption.split(" ")
        new_words = []
        for word in cap_words:
            if word.startswith("arafe") or word.startswith("araff"):
                continue
            new_words.append(word)
        caption = " ".join(new_words)
        if "there is " in caption:
            caption = caption.replace("there is ", "")
        if unload:
            self._to_cpu()

        return caption

    def unload(self):
        self._to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
