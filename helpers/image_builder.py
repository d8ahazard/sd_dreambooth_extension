import os
import random
import traceback
from typing import List, Union

import torch
from PIL import Image
from accelerate import Accelerator
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel

from dreambooth import shared
from dreambooth.dataclasses.db_config import DreamboothConfig
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.shared import disable_safe_unpickle
from dreambooth.utils import image_utils
from dreambooth.utils.image_utils import process_txt2img, get_scheduler_class
from dreambooth.utils.model_utils import get_checkpoint_match, \
    reload_system_models, \
    enable_safe_unpickle, disable_safe_unpickle, unload_system_models, xformerify
from helpers.mytqdm import mytqdm
from lora_diffusion.lora import _text_lora_path_ui, patch_pipe, tune_lora_scale, \
    get_target_module


class ImageBuilder:
    def __init__(
            self, config: DreamboothConfig,
            class_gen_method: str = "Native Diffusers",
            lora_model: str = None,
            batch_size: int = 1,
            accelerator: Accelerator = None,
            source_checkpoint: str = None,
            lora_unet_rank: int = 4,
            lora_txt_rank: int = 4,
            scheduler: Union[str, None] = None
    ):
        self.image_pipe = None
        self.txt_pipe = None
        self.resolution = config.resolution
        self.last_model = None
        self.batch_size = batch_size
        self.exception_count = 0
        use_txt2img = class_gen_method == "A1111 txt2img (Euler a)"

        if not image_utils.txt2img_available and use_txt2img:
            print("No txt2img available.")
            use_txt2img = False

        if (source_checkpoint is None or not os.path.isfile(source_checkpoint)) and use_txt2img:
            print("Unable to find source model, can't use txt2img.")
            use_txt2img = False

        self.use_txt2img = use_txt2img
        self.del_accelerator = False

        if not self.use_txt2img:
            unload_system_models()
            self.accelerator = accelerator
            if accelerator is None:
                try:
                    accelerator = Accelerator(
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        mixed_precision=config.mixed_precision,
                        log_with="tensorboard",
                        project_dir=os.path.join(config.model_dir, "logging")
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
            disable_safe_unpickle()
            unet_path = os.path.join(config.pretrained_model_name_or_path, "unet")
            if config.infer_ema:
                ema_path = os.path.join(config.pretrained_model_name_or_path, "ema_unet",
                                        "diffusion_pytorch_model.safetensors")
                if os.path.isfile(ema_path):
                    unet_path = os.path.join(config.pretrained_model_name_or_path, "ema_unet")

            self.image_pipe = DiffusionPipeline.from_pretrained(
                config.pretrained_model_name_or_path,
                vae=AutoencoderKL.from_pretrained(
                    config.pretrained_vae_name_or_path or config.pretrained_model_name_or_path,
                    subfolder=None if config.pretrained_vae_name_or_path else "vae",
                    revision=config.revision,
                    torch_dtype=torch_dtype
                ),
                unet=UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch_dtype),
                torch_dtype=torch_dtype,
                requires_safety_checker=False,
                safety_checker=None,
                revision=config.revision
            )
            self.image_pipe.enable_attention_slicing()

            xformerify(self.image_pipe)

            self.image_pipe.progress_bar = self.progress_bar

            if scheduler is None:
                scheduler = config.scheduler

            print(f"Using scheduler: {scheduler}")
            scheduler_class = get_scheduler_class(scheduler)

            self.image_pipe.scheduler = scheduler_class.from_config(self.image_pipe.scheduler.config)

            if "UniPC" in scheduler:
                self.image_pipe.scheduler.config.solver_type = "bh2"

            self.image_pipe.to(accelerator.device)
            new_hotness = os.path.join(config.model_dir, "checkpoints", f"checkpoint-{config.revision}")
            if os.path.exists(new_hotness):
                accelerator.print(f"Resuming from checkpoint {new_hotness}")
                disable_safe_unpickle()
                accelerator.load_state(new_hotness)
                enable_safe_unpickle()

            if config.use_lora and lora_model:
                lora_model_path = shared.ui_lora_models_path
                if os.path.exists(lora_model_path):
                    patch_pipe(
                        pipe=self.image_pipe,
                        maybe_unet_path=lora_model_path,
                        unet_target_replace_module=get_target_module("module", config.use_lora_extended),
                        token=None,
                        r=lora_unet_rank,
                        r_txt=lora_txt_rank
                    )
                    tune_lora_scale(self.image_pipe.unet, config.lora_weight)

                    lora_txt_path = _text_lora_path_ui(lora_model_path)
                    if os.path.exists(lora_txt_path):
                        tune_lora_scale(self.image_pipe.text_encoder, config.lora_txt_weight)

        else:
            try:
                from modules import sd_models
                current_model = sd_models.select_checkpoint()
                print(f"Source checkpoint: {source_checkpoint}")
                new_model_info = get_checkpoint_match(source_checkpoint)
                print(f"Model info: {new_model_info.filename}")
                self.last_model = current_model
                if new_model_info is not None:
                    print(f"Loading model: {new_model_info.model_name}")
                    shared.sd_model = sd_models.load_model(new_model_info)
                    reload_system_models()
            except:
                pass

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return mytqdm(iterable, **self._progress_bar_config, position=0)
        elif total is not None:
            return mytqdm(total=total, **self._progress_bar_config, position=0)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def generate_images(self, prompt_data: List[PromptData], pbar: mytqdm) -> [Image]:
        positive_prompts = []
        negative_prompts = []
        seed = -1
        scale = 7.5
        steps = 60
        width = self.resolution
        height = self.resolution
        output = []
        for prompt in prompt_data:
            positive_prompts.append(prompt.prompt)
            negative_prompts.append(prompt.negative_prompt)
            scale = prompt.scale
            steps = prompt.steps
            seed = prompt.seed
            width, height = prompt.resolution

        if self.use_txt2img:
            try:
                from modules.processing import StableDiffusionProcessingTxt2Img
                from modules import shared as auto_shared

                p = StableDiffusionProcessingTxt2Img(
                    sampler_name='Euler a',
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
            except:
                print("No txt2img.")
                self.use_txt2img = False
        else:
            with self.accelerator.autocast(), torch.inference_mode():
                if seed is None or seed == '' or seed == -1:
                    seed = int(random.randrange(0, 21474836147))

                generator = torch.manual_seed(seed)
                try:
                    output = self.image_pipe(
                        positive_prompts,
                        num_inference_steps=steps,
                        guidance_scale=scale,
                        height=height,
                        width=width,
                        generator=generator,
                        negative_prompt=negative_prompts).images
                    self.exception_count = 0
                except Exception as e:
                    print(f"Exception generating images: {e}")
                    traceback.print_exc()
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
            try:
                from modules import sd_models
                shared.sd_model = sd_models.load_model(self.last_model)
            except:
                pass

        if not is_ui:
            reload_system_models()
