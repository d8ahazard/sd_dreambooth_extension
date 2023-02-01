import os
import random
from typing import List

import torch
from PIL import Image
from accelerate import Accelerator
from diffusers import DiffusionPipeline, AutoencoderKL

from extensions.sd_dreambooth_extension.dreambooth import shared
from extensions.sd_dreambooth_extension.dreambooth.shared import disable_safe_unpickle
from extensions.sd_dreambooth_extension.dreambooth.utils.image_utils import process_txt2img
from modules import shared as auto_shared
from extensions.sd_dreambooth_extension.dreambooth.dataclasses.db_config import DreamboothConfig
from extensions.sd_dreambooth_extension.dreambooth.dataclasses.prompt_data import PromptData
from extensions.sd_dreambooth_extension.dreambooth.utils.model_utils import get_checkpoint_match, reload_system_models, \
    enable_safe_unpickle
from extensions.sd_dreambooth_extension.helpers.mytqdm import mytqdm
from extensions.sd_dreambooth_extension.lora_diffusion.lora import _text_lora_path_ui, patch_pipe, tune_lora_scale, \
get_target_module
from modules import sd_models
from modules.processing import StableDiffusionProcessingTxt2Img


class ImageBuilder:
    def __init__(
            self, config: DreamboothConfig, 
            use_txt2img: bool, 
            lora_model: str = None, 
            batch_size: int = 1, 
            accelerator: Accelerator = None,
            lora_unet_rank: int = 4,
            lora_txt_rank: int = 4
        ):
        self.image_pipe = None
        self.txt_pipe = None
        self.resolution = config.resolution
        self.last_model = None
        self.batch_size = batch_size
        self.exception_count = 0
        config_src = config.src
        if not os.path.exists(config_src):
            alt_src = os.path.join(shared.dreambooth_models_path, config_src)
            if os.path.exists(alt_src):
                config_src = alt_src
        if not os.path.exists(config_src) and use_txt2img:
            print("Source model is from hub, can't use txt2image.")
            use_txt2img = False

        self.use_txt2img = use_txt2img
        self.del_accelerator = False

        if not self.use_txt2img:
            self.accelerator = accelerator
            if accelerator is None:
                try:
                    accelerator = Accelerator(
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        mixed_precision=config.mixed_precision,
                        log_with="tensorboard",
                        logging_dir=os.path.join(config.model_dir, "logging")
                    )
                    self.accelerator = accelerator
                    self.del_accelerator = True
                except Exception as e:
                    if "AcceleratorState" in str(e):
                        msg = "Change in precision detected, please restart the webUI entirely to use new precision."
                    else:
                        msg = f"Exception initializing accelerator: {e}"
                    print(msg)
            torch_dtype = torch.float16 if shared.device.type == "cuda" else torch.float32

            self.image_pipe = DiffusionPipeline.from_pretrained(
                config.pretrained_model_name_or_path,
                vae=AutoencoderKL.from_pretrained(
                    config.pretrained_vae_name_or_path or config.pretrained_model_name_or_path,
                    subfolder=None if config.pretrained_vae_name_or_path else "vae",
                    revision=config.revision,
                    torch_dtype=torch_dtype
                ),
                torch_dtype=torch_dtype,
                requires_safety_checker=False,
                safety_checker=None,
                revision=config.revision
            )
            self.image_pipe.enable_xformers_memory_efficient_attention()
            self.image_pipe.to(accelerator.device)
            new_hotness = os.path.join(config.model_dir, "checkpoints", f"checkpoint-{config.revision}")
            if os.path.exists(new_hotness):
                accelerator.print(f"Resuming from checkpoint {new_hotness}")
                disable_safe_unpickle()
                accelerator.load_state(new_hotness)
                enable_safe_unpickle()
            if config.use_lora and lora_model is not None and lora_model != "":

                lora_model_path = os.path.join(shared.models_path, "lora", lora_model)
                lora_txt_path = _text_lora_path_ui(lora_model)

                patch_pipe(
                    pipe=self.image_pipe,
                    unet_path=lora_model_path,
                    unet_target_replace_module=get_target_module("module", config.use_lora_extended),
                    token="None",
                    r=config.lora_unet_rank,
                    r_txt=config.lora_txt_rank
                )

                tune_lora_scale(self.image_pipe.unet, config.lora_weight)
                if os.path.exists(lora_txt_path):
                    tune_lora_scale(self.image_pipe.text_encoder, config.lora_txt_weight)

        else:
            current_model = sd_models.select_checkpoint()
            new_model_info = get_checkpoint_match(config.src)
            if new_model_info is not None and current_model is not None:

                if isinstance(new_model_info, sd_models.CheckpointInfo) and new_model_info.sha256 != current_model.sha256:
                    self.last_model = current_model
                    print(f"Loading model: {new_model_info.model_name}")
                    sd_models.load_model(new_model_info)

            if new_model_info is not None and current_model is None:
                sd_models.load_model(new_model_info)
            reload_system_models()


    def generate_images(self, prompt_data: List[PromptData], pbar: mytqdm) -> [Image]:
        positive_prompts = []
        negative_prompts = []
        seed = -1
        scale = 7.5
        steps = 60
        width = self.resolution
        height = self.resolution
        for prompt in prompt_data:
            positive_prompts.append(prompt.prompt)
            negative_prompts.append(prompt.negative_prompt)
            scale = prompt.scale
            steps = prompt.steps
            seed = prompt.seed
            width, height = prompt.resolution

        if self.use_txt2img:
            p = StableDiffusionProcessingTxt2Img(
                sampler_name='DPM++ 2S a Karras',
                sd_model=auto_shared.sd_model,
                prompt=positive_prompts,
                negative_prompt=negative_prompts,
                batch_size=self.batch_size,
                steps=steps,
                cfg_scale=scale,
                width=width,
                height=height,
                do_not_save_grid=True,
                do_not_save_samples=True,
                do_not_reload_embeddings=True
            )

            auto_tqdm = auto_shared.total_tqdm
            shared.total_tqdm = pbar
            pbar.reset(steps)
            processed = process_txt2img(p)
            p.close()
            auto_shared.total_tqdm = auto_tqdm
            output = processed
        else:
            with self.accelerator.autocast(), torch.inference_mode():
                if seed is None or seed == '' or seed == -1:
                    seed = int(random.randrange(21474836147))
                g_cuda = torch.Generator(device=self.accelerator.device).manual_seed(seed)
                try:
                    output = self.image_pipe(
                        positive_prompts,
                        num_inference_steps=steps,
                        guidance_scale=scale,
                        height=height,
                        width=width,
                        generator=g_cuda,
                        negative_prompt=negative_prompts).images
                    self.exception_count = 0
                except Exception as e:
                    print(f"Exception generating images: {e}")
                    self.exception_count += 1
                    if self.exception_count > 10:
                        raise
                    output = []
                    pass

        return output

    def unload(self, is_ui):
        # If we have an image pipe, delete it
        if self.image_pipe is not None:
            del self.image_pipe
        if self.del_accelerator:
            del self.accelerator
        # If there was a model loaded already, reload it
        if self.last_model is not None and not is_ui:
            sd_models.load_model(self.last_model)
