import gc
import logging
import os
from typing import Dict

import torch
from PIL import Image
from transformers import AutoTokenizer

from preprocess.captioners.base import BaseCaptioner
from preprocess.captioners.mplug_owl import MplugOwlForConditionalGeneration, MplugOwlImageProcessor, MplugOwlProcessor

logger = logging.getLogger(__name__)


class LLMCaptioner(BaseCaptioner):
    model = None
    processor = None

    def __init__(self):
        super().__init__(None)
        logger.debug("Initializing LLM model...")
        pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            pretrained_ckpt,
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )
        self.image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self._to_cpu()
        logger.debug("Initialized LLM model.")

    def caption(self, image: Image, params: Dict=None, unload: bool = False):
        self._to_gpu()
        raw_image = image.convert('RGB')
        character = params.get("character", "nsfw")
        char_file = os.path.join(os.path.dirname(__file__), "mplug_owl", "characters", f"{character}.txt")
        with open(char_file, "r") as f:
            character = f.read().strip()
        
        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 77
        }

        images = [raw_image]
        prompts = [character]
        
        logger.debug("Processing inputs...")
        inputs = self.processor(text=prompts, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        logger.debug("Generating response...")
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        caption = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        if "," in caption:
            parts = caption.split(",")
            caption = " ".join([part.strip() for part in parts if part.strip() != ""])
        logger.debug(f"Caption: {caption}")
        if unload:
            self._to_cpu()

        return caption

    def _to_cpu(self):
        self.model.to('cpu')

    def _to_gpu(self):
        self.model.to(self.device)

    def unload(self):
        self._to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
