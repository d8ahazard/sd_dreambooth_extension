import gc
import logging
from typing import Dict

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from preprocess.captioners.base import BaseCaptioner


class Blip2Captioner(BaseCaptioner):
    model = None
    processor = None

    def __init__(self):
        super().__init__(None)

        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True,
                                                                   device_map="auto")
        self._to_cpu()

    def caption(self, image: Image, params: Dict = None, unload: bool = False) -> str:
        self._to_gpu()
        question = params["question"] if "question" in params else None
        if question:
            if "?" in question:
                question = f"Question: {question} Answer: "
            logging.getLogger(__name__).debug(f"Asking: {question}")
            inputs = self.processor(image, text=question, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=20)
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        else:
            logging.getLogger(__name__).debug(f"Describing image")
            inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=20)
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if "," in response:
            fixed_res = []
            responses = response.split(",")
            for res in responses:
                if res.strip() != "" and res.strip() not in fixed_res:
                    fixed_res.append(res.strip())
            response = ", ".join(fixed_res)
        else:
            fixed_res = []
            responses = response.split(" ")
            for res in responses:
                if res.strip() != "" and res.strip() != fixed_res[-1]:
                    fixed_res.append(res.strip())
            response = " ".join(fixed_res)
        logging.getLogger(__name__).debug(f"Response: {response}")
        if unload:
            self._to_cpu()
        return response

    def unload(self):
        self._to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
