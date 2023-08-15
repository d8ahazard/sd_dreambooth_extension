#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import contextlib
import itertools
import json
import logging
import math
import os
import random
import time
import traceback
from decimal import Decimal

import accelerate
import datasets
import diffusers
import numpy as np
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import transformers.utils.logging
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import (
    DiffusionPipeline,
    UNet2DConditionModel, UniPCMultistepScheduler, EMAModel, StableDiffusionControlNetPipeline,
    ControlNetModel, StableDiffusionPipeline
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import LoRAAttnProcessor2_0, LoRAAttnProcessor
from diffusers.utils import logging as dl, is_xformers_available, randn_tensor
from packaging import version
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

from core.handlers.images import ImageHandler
from core.pipelines import StableDiffusionXLPipeline
from dreambooth import shared
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.dataclasses.train_result import TrainResult
from dreambooth.dataclasses.training_config import TrainingConfig
from dreambooth.dataset.bucket_sampler import BucketSampler
from dreambooth.dataset.controlnet_dataset import ControlDataset
from dreambooth.dataset.ft_dataset import FtDataset
from dreambooth.oft_utils import MHE_OFT
from dreambooth.optimization import UniversalScheduler, get_optimizer, get_noise_scheduler
from dreambooth.shared import status
from dreambooth.training_utils import current_prior_loss, set_seed, deepspeed_zero_init_disabled_context_manager, \
    save_lora, load_lora, compute_snr, apply_oft, create_vae
from dreambooth.utils.gen_utils import generate_classifiers, generate_dataset
from dreambooth.utils.model_utils import (
    unload_system_models,
    import_model_class_from_model_name_or_path,
    disable_safe_unpickle,
    xformerify,
    torch2ify, unet_attn_processors_state_dict,
)
from dreambooth.utils.text_utils import encode_hidden_state
from dreambooth.utils.utils import cleanup, printm
from helpers.log_parser import LogParser
from helpers.mytqdm import mytqdm
from lora_diffusion.extra_networks import apply_lora
from lora_diffusion.lora import get_target_module, TEXT_ENCODER_DEFAULT_TARGET_REPLACE, set_lora_requires_grad

logger = logging.getLogger(__name__)
# define a Handler which writes DEBUG messages or higher to the sys.stderr
dl.set_verbosity_error()


def main(args: TrainingConfig, user: str = None) -> TrainResult:
    """
    @param args: TrainingConfig - I don't know why we removed this, but please leave it.
    @param user: User to send training updates to (for new UI)
    @return: TrainResult
    """
    cleanup()
    status_handler = None
    last_samples = []

    logging_dir = os.path.join(args.model_dir, "logging")
    try:
        from core.handlers.status import StatusHandler
        from core.handlers.config import ConfigHandler
        from core.handlers.models import ModelHandler

        mh = ModelHandler(user_name=user)
        status_handler = StatusHandler(user_name=user, target="dreamProgress")
        export_diffusers = True
        user_model_dir = mh.user_path
        logger.debug(f"Export diffusers: {export_diffusers}, diffusers dir: {user_model_dir}")
        shared.status_handler = status_handler
    except:
        pass

    log_parser = LogParser()
    if status_handler is not None:
        status_handler.start()

    def update_status(data: dict):
        if status_handler is not None:
            if "iterations_per_second" in data:
                data = {"status": json.dumps(data)}
            status_handler.update(items=data)

    result = TrainResult()
    result.config = args

    def log_validation():
        logger.info("Running validation... ")
        update_status({"status": "Generating samples..."})
        pipeline_args = {
            "require_safety_checker": False,
            "tokenizer": tokenizer,
            "revision": args.revision,
            "torch_dtype": weight_dtype
        }

        if train_lora:
            tgt_module = get_target_module("module", True)

            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_tenc = accelerator.unwrap_model(text_encoder)

            modelmap = {"unet": (unwrapped_unet, tgt_module)}

            # save text_encoder
            if stop_text_percentage:
                modelmap["text_encoder"] = (unwrapped_tenc, TEXT_ENCODER_DEFAULT_TARGET_REPLACE)
            # TODO: Load LORA if training.
            if args.train_mode == "SDXL":
                validation_pipeline = StableDiffusionXLPipeline.from_pretrained(args.pretrained_model_name_or_path)
            else:
                validation_pipeline = StableDiffusionPipeline.from_pretrained(
                    args.src
                )
                validation_pipeline = apply_lora(validation_pipeline, model_map=modelmap)

        else:
            if vae is not None:
                pipeline_args["vae"] = vae

            if text_encoder is not None:
                pipeline_args["text_encoder"] = accelerator.unwrap_model(text_encoder)

            if controlnet is not None:
                pipeline_args["controlnet"] = controlnet

            # create pipeline (note: unet and vae are loaded again in float32)
            if args.train_mode == "controlnet":
                validation_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet)
                         ** pipeline_args,
                )
            else:
                if args.train_mode == "SDXL":
                    validation_pipeline = StableDiffusionXLPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet)
                             ** pipeline_args,
                    )
                else:
                    # If src is a checkpoint, load from that
                    validation_pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet)
                             ** pipeline_args,
                    )

        scheduler_args = {}

        if "variance_type" in validation_pipeline.scheduler.config:
            variance_type = validation_pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        validation_pipeline.scheduler = UniPCMultistepScheduler.from_config(validation_pipeline.scheduler.config,
                                                                            **scheduler_args)
        validation_pipeline = validation_pipeline.to(accelerator.device)
        validation_pipeline.set_progress_bar_config(disable=True)
        if args.attention == "xformers":
            validation_pipeline.enable_xformers_memory_efficient_attention()

        if args.seed is None or args.seed == -1:
            seed = int(random.randrange(21474836147))
        else:
            seed = args.seed
        generator = torch.Generator(device=accelerator.device).manual_seed(seed)

        images = []
        # Get random items from the dataset
        all_prompts = []
        all_train_data = train_dataset.train_img_data
        for item in all_train_data:
            all_prompts.append(item['text'])

        prompts = []

        # TODO: Make a wrapper or this for auto1111
        image_handler = ImageHandler(user_name=user)
        random_indices = random.sample(range(len(all_prompts)), min(len(all_prompts), args.num_save_samples))
        out_dir = os.path.join(args.model_dir, "samples")
        for i in random_indices:
            prompts.append(all_prompts[i])
        for i in range(len(prompts)):
            with torch.autocast("cuda"):
                image = validation_pipeline(prompts[i], num_inference_steps=30, generator=generator).images[0]

            prompt_dats = PromptData()
            prompt_dats.prompt = prompts[i]
            prompt_dats.steps = 30
            prompt_dats.seed = seed
            img_path = image_handler.save_image(image, out_dir, prompt_dats)
            images.append(img_path)
        update_status({"images": images, "prompts": prompts})
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            else:
                logger.warning(f"image logging not implemented for {tracker.name}")
                msg = f"Exception initializing accelerator: {e}"
            logger.warning(msg)
            result.msg = msg
            result.config = args
            stop_profiler(profiler)
            return result

        # This is the secondary status bar
        pbar2 = mytqdm(
            disable=not accelerator.is_local_main_process,
            position=1,
            user=user,
            target="dreamProgress",
            index=1
        )
        # Currently, it's not possible to do gradient accumulation when training two models with
        # accelerate.accumulate This will be enabled soon in accelerate. For now, we don't allow gradient
        # accumulation when training two models.
        # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
        if (
                stop_text_percentage != 0
                and gradient_accumulation_steps > 1
                and accelerator.num_processes > 1
        ):
            msg = (
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future. Text "
                "encoder training will be disabled."
            )
            logger.warning(msg)
            status.textinfo = msg
            update_status({"status": msg})
            stop_text_percentage = 0
        pretrained_path = args.get_pretrained_model_name_or_path()
        logger.debug(f"Pretrained path: {pretrained_path}")

        count, instance_prompts, class_prompts = generate_classifiers(
            args, class_gen_method=class_gen_method, accelerator=accelerator, ui=False, pbar=pbar2
        )
        pbar2.reset()
        if status.interrupted:
            result.msg = "Training interrupted."
            stop_profiler(profiler)
            return result

        if class_gen_method == "Native Diffusers" and count > 0:
            unload_system_models()

        def create_vae():
            vae_path = (
                args.pretrained_vae_name_or_path
                if args.pretrained_vae_name_or_path
                else args.get_pretrained_model_name_or_path()
            )
            disable_safe_unpickle()
            new_vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder=None if args.pretrained_vae_name_or_path else "vae",
                revision=args.revision,
            )
            enable_safe_unpickle()
            new_vae.requires_grad_(False)
            new_vae.to(accelerator.device, dtype=weight_dtype)
            return new_vae

        disable_safe_unpickle()
        # Load the tokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(pretrained_path, "tokenizer"),
            revision=args.revision,
            use_fast=False,
        )

        del validation_pipeline
        cleanup()

    text_encoder = None

    stop_text_percentage = args.stop_text_encoder
    train_unet = args.train_unet
    train_lora = args.train_lora
    train_ema = args.train_ema
    train_oft = args.train_oft

    if args.train_mode == "finetune" or args.train_mode == "SDXL" or train_oft:
        train_unet = True
        stop_text_percentage = 0

    if args.train_mode == "SDXL":
        train_ema = False

    if not train_unet:
        stop_text_percentage = 1

    if args.train_mode == "controlnet":
        stop_text_percentage = 0
        train_unet = False
        train_lora = False
        train_oft = False
        train_ema = False

    args.max_token_length = int(args.max_token_length)
    if not args.pad_tokens and args.max_token_length > 75:
        args.pad_tokens = True

    precision = args.mixed_precision if not args.cpu_only else "no"

    weight_dtype = torch.float32
    if precision == "fp16":
        weight_dtype = torch.float16
    elif precision == "bf16":
        weight_dtype = torch.bfloat16

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit,
                                                      project_dir=args.pretrained_model_name_or_path,
                                                      logging_dir=logging_dir)

    try:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="tensorboard",
            project_config=accelerator_project_config,
            cpu=args.cpu_only
        )

        run_name = "dreambooth.events"
        max_log_size = 250 * 1024  # specify the maximum log size

    except Exception as e:
        if "AcceleratorState" in str(e):
            msg = "Change in precision detected, please restart the webUI entirely to use new precision."
        else:
            msg = f"Exception initializing accelerator: {e}"
        logger.warning(msg)
        result.msg = msg
        result.config = args
        return result

    # This is the secondary status bar
    pbar2 = mytqdm(
        disable=not accelerator.is_local_main_process,
        position=1,
        user=user,
        target="dreamProgress",
        index=1
    )

    # Currently, it's not possible to do gradient accumulation when training two models with
    # accelerate.accumulate This will be enabled soon in accelerate. For now, we don't allow gradient
    # accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
            stop_text_percentage != 0
            and args.gradient_accumulation_steps > 1
            and accelerator.num_processes > 1
    ):
        msg = (
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future. Text "
            "encoder training will be disabled."
        )

        logger.warning(msg)
        status.textinfo = msg
        update_status({"status": msg})
        stop_text_percentage = 0

    pbar2.reset()
    if status.interrupted:
        result.msg = "Training interrupted."
        return result

    unload_system_models()
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # TODO: Use "normal" db gen method here

    disable_safe_unpickle()

    tokenizer_path = os.path.join(args.get_pretrained_model_name_or_path(), "tokenizer")
    logger.debug(f"Loading tokenizer from {tokenizer_path}")
    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.get_pretrained_model_name_or_path(), args.revision
    )

    ema_unet = None
    unet_lora_params = None
    text_encoder_lora_params = None
    controlnet = None
    oft_params = {}

    # Load scheduler and models
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            torch_dtype=torch.float32,
        )
    printm("Created tenc")
    vae = create_vae(args, accelerator.device, weight_dtype)
    vae_factor = vae.config.scaling_factor

    printm("Created vae")

    unet = UNet2DConditionModel.from_pretrained(
        args.get_pretrained_model_name_or_path(),
        subfolder="unet",
        revision=args.revision,
        torch_dtype=torch.float32,
    )

    # Disable info log message
    unet_logger = logging.getLogger('diffusers.models.unet_2d_condition')
    unet_logger.setLevel(logging.WARNING)

    if args.train_mode == "controlnet":
        controlnet_dir = os.path.join(args.pretrained_model_name_or_path, "controlnet")
        if os.path.exists(controlnet_dir):
            logger.info("Loading existing controlnet weights")
            controlnet = ControlNetModel.from_pretrained(controlnet_dir)
        else:
            logger.info("Initializing controlnet weights from unet")
            controlnet = ControlNetModel.from_unet(unet)

    if train_lora:
        if args.train_mode == "SDXL":
            unet.requires_grad_(False)
            unet.to(accelerator.device, dtype=weight_dtype)
        else:
            unet_lora_params, text_encoder_lora_params, text_encoder = load_lora(args, stop_text_percentage, unet,
                                                                                 text_encoder)
    elif train_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "unet"))
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    elif train_oft:
        unet, accelerator, oft_params = apply_oft(unet, accelerator, args.oft_eps, args.oft_rank, args.oft_coft)
        stop_text_percentage = 0
        # TODO: Move these elsewhere
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

    tokenizer_two = None
    text_encoder_two = None

    if args.train_mode == "SDXL":
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
        )

        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
        )

        text_encoder_two = text_encoder_cls_two.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
        )

        text_encoder_two.requires_grad_(False)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        if args.train_lora:
            unet_lora_attn_procs = {}
            unet_lora_params = []
            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                hidden_size = None
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                if hidden_size is not None:
                    lora_attn_processor_class = (
                        LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                    )
                    module = lora_attn_processor_class(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                    unet_lora_attn_procs[name] = module
                    unet_lora_params.extend(module.parameters())

            unet.set_attn_processor(unet_lora_attn_procs)

    if args.attention == "xformers" and not args.cpu_only:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
        xformerify(unet, False)
        xformerify(vae, False)

    unet = torch2ify(unet)
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        # TODO: Add lora saving hooks here.
        def save_model_hook(models, weights, output_dir):
            if train_oft:
                oft_unet = accelerator.unwrap_model(unet)
                oft_unet.save_attn_procs(os.path.join(output_dir, "unet_oft"))

            elif train_lora and args.train_mode != "SDXL":
                save_lora(args, stop_text_percentage, accelerator, unet, text_encoder, pbar2,
                          user_model_dir=user_model_dir)
            elif train_lora and args.train_mode == "SDXL":
                logger.debug("Saving lora weights...")
                update_status({"status": "Saving lora weights..."})
                unet_lora_layers_to_save = None

                for model in models:
                    unet_lora_layers_to_save = unet_attn_processors_state_dict(model)

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

                LoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=None,
                    safe_serialization=True
                )
            else:
                if train_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for model in models:
                    sub_dir = None
                    if isinstance(model, type(accelerator.unwrap_model(unet))):
                        sub_dir = "unet"
                    if isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                        sub_dir = "text_encoder"
                    if controlnet is not None:
                        if isinstance(model, type(controlnet)):
                            sub_dir = "controlnet"
                    if sub_dir:
                        model.save_pretrained(os.path.join(output_dir, sub_dir))
                    else:
                        logger.debug("Not saving: {}".format(type(model)))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.train_mode == "SDXL" and args.train_lora:
                unet_ = None
                text_encoder_ = None

                while len(models) > 0:
                    model = models.pop()

                    if isinstance(model, type(accelerator.unwrap_model(unet))):
                        unet_ = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                lora_state_dict, network_alpha = LoraLoaderMixin.lora_state_dict(input_dir)
                LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alpha=network_alpha, unet=unet_)
                LoraLoaderMixin.load_lora_into_text_encoder(
                    lora_state_dict, network_alpha=network_alpha, text_encoder=text_encoder_
                )
            else:
                if train_oft and os.path.exists(os.path.join(input_dir, "unet_oft")):
                    oft_unet = accelerator.unwrap_model(unet)
                    oft_unet.load_attn_procs(os.path.join(input_dir, "unet_oft"))
                if train_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                    ema_unet.load_state_dict(load_model.state_dict())
                    ema_unet.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()
                    if isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                        # load transformers style into model
                        load_model = text_encoder_cls.from_pretrained(os.path.join(input_dir, "text_encoder"))
                        model.config = load_model.config
                        model.load_state_dict(load_model.state_dict())
                    elif isinstance(model, type(accelerator.unwrap_model(unet))):
                        # load diffusers style into model
                        load_model = UNet2DConditionModel.from_pretrained(os.path.join(input_dir, "unet"))
                        model.register_to_config(**load_model.config)
                        model.load_state_dict(load_model.state_dict())

                    elif controlnet is not None and isinstance(model, type(controlnet)):
                        load_model = ControlNetModel.from_pretrained(os.path.join(input_dir, "controlnet"))
                        model.load_state_dict(load_model.state_dict())

                    del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # control requires_grad attribute based on conditions
    text_encoder.requires_grad_(not stop_text_percentage == 0)
    unet.requires_grad_(not (train_lora or not train_unet))

    # control gradient_checkpointing based on args.gradient_checkpointing flag
    if args.gradient_checkpointing:
        if controlnet is not None:
            controlnet.enable_gradient_checkpointing()
        if train_unet and not train_lora:
            unet.enable_gradient_checkpointing()
        if stop_text_percentage != 0:
            text_encoder.gradient_checkpointing_enable()

    # move text_encoder to the appropriate device
    if stop_text_percentage == 0:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        if text_encoder_two is not None:
            text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32 and not train_lora and args.train_mode != "SDXL":
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if stop_text_percentage != 0 and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 8:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Create shared unet/tenc learning rate variables
    learning_rate = args.learning_rate if not train_lora else args.learning_rate_lora
    txt_learning_rate = args.learning_rate_txt if not train_lora else args.learning_rate_txt

    if train_lora:
        if args.train_mode == "SDXL":
            logger.debug("Loading SDXL Lora...")
            # now we will add new LoRA weights to the attention layers
            # Set correct lora layers
            unet_lora_attn_procs = {}
            unet_lora_params = []
            unet.requires_grad_(False)
            hidden_size = None
            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]

                if hidden_size is not None:
                    lora_attn_processor_class = (
                        LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                    )
                    module = lora_attn_processor_class(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                    unet_lora_attn_procs[name] = module
                    unet_lora_params.extend(module.parameters())
            unet.set_attn_processor(unet_lora_attn_procs)
            params_to_optimize = unet_lora_params
            logger.debug(f"We have {len(unet_lora_params)} LoRA parameters")
        else:
            logger.info(
                "Training LoRA. Learning rate for unet: {}, text encoder: {}".format(learning_rate, txt_learning_rate))
            params_to_optimize = itertools.chain(*unet_lora_params) if stop_text_percentage == 0 else [
                {"params": itertools.chain(*unet_lora_params), "lr": learning_rate},
                {"params": itertools.chain(*text_encoder_lora_params), "lr": txt_learning_rate},
            ]
    elif train_oft:
        params_to_optimize = oft_params
    elif stop_text_percentage != 0:
        params_to_optimize = [
            {"params": itertools.chain(unet.parameters()), "lr": learning_rate},
            {"params": itertools.chain(text_encoder.parameters()), "lr": txt_learning_rate},
        ] if train_unet else [
            {"params": itertools.chain(text_encoder.parameters()), "lr": txt_learning_rate},
        ]
    else:
        params_to_optimize = [
            {"params": itertools.chain(unet.parameters()), "lr": learning_rate},
        ]

    if args.train_mode == "controlnet":
        params_to_optimize = [
            {"params": itertools.chain(controlnet.parameters()), "lr": learning_rate},
        ]
    logger.debug(f"Getting da optimizer: {args.optimizer} {learning_rate}")

    optimizer = get_optimizer(
        params_to_optimize,
        learning_rate=learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        optimizer=args.optimizer
    )

    if len(optimizer.param_groups) > 1:
        try:
            optimizer.param_groups[1]["weight_decay"] = args.tenc_weight_decay
            optimizer.param_groups[1]["grad_clip_norm"] = args.tenc_grad_clip_norm
        except:
            logger.warning("Exception setting tenc weight decay")
            traceback.print_exc()

    noise_scheduler = get_noise_scheduler(args)

    def cleanup_memory():
        try:
            if controlnet:
                del controlnet
            if unet:
                del unet
            if ema_unet:
                del ema_unet
            if text_encoder:
                del text_encoder
            if text_encoder_two:
                del text_encoder_two
            if tokenizer:
                del tokenizer
            if tokenizer_two:
                del tokenizer_two
            if optimizer:
                del optimizer
            if train_dataloader:
                del train_dataloader
            if train_dataset:
                del train_dataset
            if lr_scheduler:
                del lr_scheduler
            if vae:
                del vae
        except:
            pass
        cleanup(True)

    if args.train_mode == "SDXL":
        logger.debug("Training SDXL, enabling latent caching.")
        args.cache_latents = True

    if args.cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()

    if status.interrupted:
        result.msg = "Training interrupted."
        return result

    printm("Loading dataset...")
    pbar2.reset()
    pbar2.set_description("Loading dataset")
    logger.debug(f"Dataset dir is {args.model_dir}")

    train_dataset = None
    with_prior_preservation = False

    if args.train_mode == "default" or args.train_mode == "SDXL":
        count, instance_prompts, class_prompts = generate_classifiers(args, accelerator=accelerator, ui=False,
                                                                      pbar=pbar2)
        tokenizers = [tokenizer] if tokenizer_two is None else [tokenizer, tokenizer_two]
        text_encoders = [text_encoder] if text_encoder_two is None else [text_encoder, text_encoder_two]
        logger.debug(f"Creating DB/SDXL Dataset: {len(text_encoders)}")
        train_dataset = generate_dataset(
            instance_prompts=instance_prompts,
            class_prompts=class_prompts,
            batch_size=args.train_batch_size,
            tokenizer=tokenizers,
            text_encoder=text_encoders,
            accelerator=accelerator,
            vae=vae if args.cache_latents else None,
            debug=False,
            model_dir=args.model_dir,
            max_token_length=args.max_token_length,
            pbar=pbar2
        )
        if train_dataset.class_count > 0:
            with_prior_preservation = True

    elif args.train_mode == "finetune":
        train_dataset = FtDataset(
            instance_data_dir=args.train_data_dir,
            resolution=args.resolution,
            batch_size=args.train_batch_size,
            hflip=args.hflip,
            shuffle_tags=args.shuffle_tags,
            strict_tokens=args.strict_tokens,
            dynamic_img_norm=args.dynamic_img_norm,
            not_pad_tokens=not args.pad_tokens,
            model_dir=args.model_dir,
            tokenizer=tokenizer,
            vae=vae,
            cache_latents=args.cache_latents,
            user=user,
            use_dir_tags=args.use_dir_tags,
        )
    elif args.train_mode == "controlnet":
        train_dataset = ControlDataset(
            instance_data_dir=args.train_data_dir,
            resolution=args.resolution,
            batch_size=args.train_batch_size,
            hflip=args.hflip,
            shuffle_tags=args.shuffle_tags,
            strict_tokens=args.strict_tokens,
            dynamic_img_norm=args.dynamic_img_norm,
            not_pad_tokens=not args.pad_tokens,
            model_dir=args.model_dir,
            tokenizer=tokenizer,
            vae=vae,
            cache_latents=args.cache_latents,
            user=user,
            use_dir_tags=args.use_dir_tags,
        )

    pbar2.reset()
    if train_dataset:
        printm("Dataset loaded.")
    else:
        logger.warning("Dataset not loaded.")
        result.msg = "Dataset not loaded."
        cleanup_memory()
        return result

    if args.cache_latents:
        printm("Unloading vae.")
        del vae
        # Preserve reference to vae for later checks
        vae = None

    if status.interrupted:
        result.msg = "Training interrupted."
        return result

    if train_dataset.__len__ == 0:
        msg = "Please provide a directory with actual images in it."
        logger.warning(msg)
        status.textinfo = msg
        update_status({"status": status})
        cleanup_memory()
        result.msg = msg
        result.config = args
        return result

    cleanup()

    def collate_fn_db(examples):
        input_ids = [example["input_ids"] for example in examples]
        pixel_values = [example["image"] for example in examples]
        types = [example["is_class"] for example in examples]
        weights = [
            current_prior_loss_weight if example["is_class"] else 1.0
            for example in examples
        ]
        loss_avg = 0
        for weight in weights:
            loss_avg += weight
        loss_avg /= len(weights)
        pixel_values = torch.stack(pixel_values)
        if not args.cache_latents:
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()
        input_ids = torch.cat(input_ids, dim=0)

        batch_data = {
            "input_ids": input_ids,
            "images": pixel_values,
            "types": types,
            "loss_avg": loss_avg,
        }
        if "input_ids2" in examples[0]:
            input_ids_2 = [example["input_ids2"] for example in examples]
            input_ids_2 = torch.stack(input_ids_2)

            batch_data["input_ids2"] = input_ids_2
            batch_data["original_sizes_hw"] = torch.stack([torch.LongTensor(x["original_sizes_hw"]) for x in examples])
            batch_data["crop_top_lefts"] = torch.stack([torch.LongTensor(x["crop_top_lefts"]) for x in examples])
            batch_data["target_sizes_hw"] = torch.stack([torch.LongTensor(x["target_sizes_hw"]) for x in examples])
        return batch_data

    def collate_fn_sdxl(examples):
        has_attention_mask = "instance_attention_mask" in examples[0]

        input_ids = [example["input_ids"] for example in examples if not example["is_class"]]
        pixel_values = [example["image"] for example in examples if not example["is_class"]]
        add_text_embeds = [example["instance_added_cond_kwargs"]["text_embeds"] for example in examples if
                           not example["is_class"]]
        add_time_ids = [example["instance_added_cond_kwargs"]["time_ids"] for example in examples if
                        not example["is_class"]]
        # if has_attention_mask:
        #     attention_mask = [example["instance_attention_mask"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if with_prior_preservation:
            input_ids += [example["input_ids"] for example in examples if example["is_class"]]
            pixel_values += [example["images"] for example in examples if example["is_class"]]
            add_text_embeds += [example["class_added_cond_kwargs"]["text_embeds"] for example in examples if
                                example["is_class"]]
            add_time_ids += [example["class_added_cond_kwargs"]["time_ids"] for example in examples if
                             example["is_class"]]

            # if has_attention_mask:
            #     attention_mask += [example["class_attention_mask"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.cat(input_ids, dim=0)
        add_text_embeds = torch.cat(add_text_embeds, dim=0)
        add_time_ids = torch.cat(add_time_ids, dim=0)

        batch = {
            "input_ids": input_ids,
            "images": pixel_values,
            "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
        }

        # if has_attention_mask:
        #     batch["attention_mask"] = attention_mask

        return batch

    def collate_fn_finetune(examples):
        input_ids = [example["input_ids"] for example in examples]
        pixel_values = [example["image"] for example in examples]
        pixel_values = torch.stack(pixel_values)
        if not args.cache_latents:
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()
        input_ids = torch.cat(input_ids, dim=0)

        batch_data = {
            "input_ids": input_ids,
            "images": pixel_values
        }
        return batch_data

    def collate_fn_controlnet(examples):
        input_ids = [example["input_ids"] for example in examples]
        input_ids_2 = None
        if "input_ids_2" in examples[0]:
            input_ids_2 = [example["input_ids_2"] for example in examples]

        pixel_values = [example["image"] for example in examples]
        pixel_values = torch.stack(pixel_values)

        control_image_values = [example["control_image"] for example in examples]
        control_image_values = torch.stack(control_image_values)

        if not args.cache_latents:
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()
            control_image_values = control_image_values.to(
                memory_format=torch.contiguous_format
            ).float()

        input_ids = torch.cat(input_ids, dim=0)

        batch_data = {
            "input_ids": input_ids,
            "images": pixel_values,
            "control_images": control_image_values
        }
        if input_ids_2 is not None:
            input_ids_2 = torch.cat(input_ids_2, dim=0)
            batch_data["input_ids_2"] = input_ids_2

        return batch_data

    sampler = BucketSampler(train_dataset, args.train_batch_size)

    collate_fn = collate_fn_db
    if args.train_mode == "SDXL":
        collate_fn = collate_fn_sdxl
    elif args.train_mode == "finetune":
        collate_fn = collate_fn_finetune
    elif args.train_mode == "controlnet":
        collate_fn = collate_fn_controlnet

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # This is separate, because optimizer.step is only called once per "step" in training, so it's not
    # affected by batch size
    num_update_steps_per_epoch = len(train_dataset)

    sched_train_steps = args.num_train_epochs * train_dataset.num_train_images

    lr_scale_pos = args.lr_scale_pos
    if with_prior_preservation:
        lr_scale_pos *= 2

    logger.debug(f"Setting learning rate to {learning_rate} and tenc to {txt_learning_rate}")
    lr_scheduler = UniversalScheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        total_training_steps=sched_train_steps,
        min_lr=args.learning_rate_min,
        total_epochs=args.num_train_epochs,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        factor=args.lr_factor,
        scale_pos=lr_scale_pos,
        unet_lr=learning_rate,
        tenc_lr=txt_learning_rate,
    )

    # create ema, fix OOM
    # Always prepare these components
    if train_oft:
        oft_params, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            oft_params, optimizer, train_dataloader, lr_scheduler
        )
    elif args.train_mode == "controlnet":
        unet.to(accelerator.device, dtype=weight_dtype)

        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )
    elif train_ema:
        ema_unet.model, unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ema_unet.model, unet, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # Conditionally prepare the text_encoder if stop_text_percentage is not zero
    if stop_text_percentage != 0:
        text_encoder = accelerator.prepare(text_encoder)
        if text_encoder_two is not None:
            text_encoder_two = accelerator.prepare(text_encoder_two)
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        if text_encoder_two is not None:
            text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if not args.cache_latents and vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = len(train_dataset)

    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        train_string = args.train_mode
        if train_oft:
            train_string += "_oft"
        if train_ema:
            train_string += "_ema"
        if train_lora:
            train_string += "_lora"
        accelerator.init_trackers(train_string)

    # Calc steps
    def calc_max_steps(n_pics: int, n_batch: int):
        steps_per_epoch = math.ceil(n_pics / n_batch) * n_batch
        max_steps = args.num_train_epochs * steps_per_epoch
        return max_steps

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    max_train_epochs = args.num_train_epochs
    max_train_steps = calc_max_steps(len(train_dataset), total_batch_size)

    # we calculate our number of tenc training epochs
    text_encoder_epochs = round(max_train_epochs * stop_text_percentage)
    global_step = 0
    global_epoch = 0
    first_epoch = 0
    resume_step = 0
    grad_steps = 0
    epoch_steps = 0
    resume_from_checkpoint = False
    # Potentially load in the weights and states from a previous save
    if args.checkpoint != "" and args.checkpoint is not None:
        if args.checkpoint != "latest":
            path = os.path.join(args.model_dir, "checkpoints", args.checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(os.path.join(args.model_dir, "checkpoints"))
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = False
        else:
            resume_from_checkpoint = True
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.model_dir, "checkpoints", path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    # Train!

    logger.info("***** Running training *****")
    if args.cpu_only:
        logger.info(f"  TRAINING WITH CPU ONLY")
    logger.info(f"  Training mode: {args.train_mode}")
    logger.info(f"  Num batches each epoch = {len(train_dataset) // args.train_batch_size}")
    logger.info(f"  Num Epochs = {max_train_epochs}")
    logger.info(f"  Batch Size Per Device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Precision = {args.mixed_precision}")
    logger.info(f"  Device = {accelerator.device}")
    logger.info(f"  Optimizer = {args.optimizer}")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Gradient Checkpointing: {args.gradient_checkpointing}")
    logger.info(f"  Initial learning rate = {args.learning_rate}")
    logger.info(f"  Initial txt learning rate = {txt_learning_rate}")
    logger.info(f"  Learning rate scheduler = {args.scheduler}")
    logger.info(f"  Text encoder epochs = {text_encoder_epochs}")
    logger.info(f"  Train UNET = {train_unet}")
    logger.info(f"  EMA: {train_ema}")
    logger.info(f"  LoRA: {train_lora}")
    logger.info(f"  OFT: {train_oft}")
    os.environ.__setattr__("CUDA_LAUNCH_BLOCKING", 1)

    # Only show the progress bar once on each machine.
    progress_bar = mytqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
        position=0
    )
    progress_bar.set_description(f"Epoch(0), Steps")
    progress_bar.set_postfix(refresh=True)
    args.revision = (
        args.revision if isinstance(args.revision, int) else
        int(args.revision) if str(args.revision).strip() else
        0
    )
    lifetime_step = args.revision
    status.job_count = max_train_steps
    status.job_no = global_step
    update_status({"progress_1_total": max_train_steps, "progress_1_job_current": global_step})
    training_complete = False
    msg = ""

    last_tenc = 0 < text_encoder_epochs
    if stop_text_percentage == 0:
        last_tenc = False
        if args.train_mode == "SDXL" and not args.train_lora:
            del text_encoders
            del tokenizers
            if text_encoder is not None:
                del text_encoder
            if text_encoder_two is not None:
                del text_encoder_two
            if tokenizer is not None:
                del tokenizer
            if tokenizer_two is not None:
                del tokenizer_two
            text_encoder = None
            text_encoder_two = None
            tokenizer = None
            tokenizer_two = None
            printm("Deleted text encoders.")

    cleanup()
    stats = {
        "loss": 0.0,
        "prior_loss": 0.0,
        "instance_loss": 0.0,
        "unet_lr": args.learning_rate,
        "session_epoch": 0,
        "lifetime_epoch": args.epoch,
        "total_session_epoch": args.num_train_epochs,
        "total_lifetime_epoch": args.epoch + args.num_train_epochs,
        "lifetime_step": args.revision,
        "session_step": 0,
        "total_session_step": max_train_steps,
        "total_lifetime_step": args.revision + max_train_steps,
        "steps_per_epoch": len(train_dataset),
        "iterations_per_second": 0.0,
        "vram": round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
    }

    if train_oft:
        # calculate the hyperspherical energy fine-tuning
        mhe = MHE_OFT(unet, eps=args.oft_eps, rank=args.oft_rank)
        mhe_loss = mhe.calculate_mhe()
        accelerator.log({"mhe_loss": mhe_loss}, step=0)
        accelerator.log({"eps": args.oft_eps}, step=0)
        accelerator.log({"rank": args.oft_rank}, step=0)
        accelerator.log({"COFT": 1 if args.oft_coft else 0}, step=0)
    if stop_text_percentage != 0:
        stats["tenc_lr"] = txt_learning_rate
    save_weights = False
    try:
        for epoch in range(first_epoch, max_train_epochs):
            if training_complete:
                logger.debug("Training complete, breaking epoch.")
                break

            if train_unet:
                unet.train()
            elif train_lora:
                set_lora_requires_grad(unet, False)

            progress_bar.set_description(f"Epoch({epoch}), Steps")
            progress_bar.set_postfix(refresh=True)

            train_tenc = epoch < text_encoder_epochs
            if stop_text_percentage == 0:
                train_tenc = False

            if text_encoder is not None:
                if args.freeze_clip_normalization:
                    text_encoder.eval()
                    if text_encoder_two is not None:
                        text_encoder_two.eval()
                else:
                    text_encoder.train(train_tenc)
                    if text_encoder_two is not None:
                        text_encoder_two.train(train_tenc)

            if train_lora and text_encoder is not None:
                set_lora_requires_grad(text_encoder, train_tenc)
                # We need to enable gradients on an input for gradient checkpointing to work
                # This will not be optimized because it is not a param to optimizer
                text_encoder.text_model.embeddings.position_embedding.requires_grad_(train_tenc)
                if text_encoder_two is not None:
                    text_encoder_two.text_model.embeddings.position_embedding.requires_grad_(train_tenc)
            else:
                if text_encoder is not None:
                    text_encoder.requires_grad_(train_tenc)
                if text_encoder_two is not None:
                    text_encoder_two.requires_grad_(train_tenc)

            if last_tenc != train_tenc:
                last_tenc = train_tenc
                cleanup()

            loss_total = 0

            current_prior_loss_weight = current_prior_loss(
                args, current_epoch=global_epoch
            )
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if status_handler is not None:
                    if status_handler.status.canceled:
                        logger.info("Training canceled, returning.")
                        if epoch > 0:
                            save_weights = True
                        break
                if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(args.train_batch_size)
                        progress_bar.reset()
                        status.job_count = max_train_steps
                        status.job_no += args.train_batch_size
                        stats["session_step"] += args.train_batch_size
                        stats["lifetime_step"] += args.train_batch_size
                        update_status(stats)
                        continue
                if args.simulate_training:
                    time.sleep(0.1)
                    continue

                # List to hold models
                accumulate_models = []

                # Check conditions and add models to the list
                if args.train_mode == "controlnet":
                    accumulate_models.append(controlnet)
                elif train_unet:
                    accumulate_models.append(unet)
                if stop_text_percentage > 0 and text_encoder is not None:
                    accumulate_models.append(text_encoder)
                    if text_encoder_two is not None:
                        accumulate_models.append(text_encoder_two)

                # Use a context manager to accumulate the models
                with contextlib.ExitStack() as stack:
                    for model in accumulate_models:
                        stack.enter_context(accelerator.accumulate(model))
                        # Convert images to latent space
                    with torch.no_grad():
                        if args.cache_latents:
                            latents = batch["images"].to(accelerator.device)
                        else:
                            latents = vae.encode(
                                batch["images"].to(dtype=weight_dtype)
                            ).latent_dist.sample()
                        latents = latents * vae_factor

                    # Sample noise that we'll add to the model input
                    noise = torch.randn_like(latents, device=latents.device)
                    if args.offset_noise != 0:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.offset_noise * torch.randn(
                            (latents.shape[0],
                             latents.shape[1],
                             1,
                             1),
                            device=latents.device
                        )

                    if args.input_pertubation:
                        new_noise = noise + args.input_pertubation * torch.randn_like(noise)
                    b_size, channels, height, width = latents.shape
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (b_size,),
                        device=latents.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if args.input_pertubation:
                        noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                    else:
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # TODO: Figure out if we need to do something different here for SDXL
                    pad_tokens = args.pad_tokens if train_tenc else False

                    input_ids = batch["input_ids"]
                    encoder_hidden_states = None
                    if args.train_mode != "SDXL" and text_encoder is not None:
                        encoder_hidden_states = encode_hidden_state(
                            text_encoder,
                            batch["input_ids"],
                            pad_tokens,
                            b_size,
                            args.max_token_length,
                            tokenizer.model_max_length,
                            args.clip_skip,
                        )

                    if unet.config.in_channels > channels:
                        needed_additional_channels = unet.config.in_channels - channels
                        additional_latents = randn_tensor(
                            (b_size, needed_additional_channels, height, width),
                            device=noisy_latents.device,
                            dtype=noisy_latents.dtype,
                        )
                        noisy_latents = torch.cat([additional_latents, noisy_latents], dim=1)
                    if controlnet is not None:
                        if args.cache_latents:
                            controlnet_image = batch["control_images"].to(accelerator.device)
                        else:
                            controlnet_image = vae.encode(
                                batch["control_images"].to(dtype=weight_dtype)).latent_dist.sample()

                        down_block_res_samples, mid_block_res_sample = controlnet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            controlnet_cond=controlnet_image,
                            return_dict=False,
                        )

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if controlnet is not None:
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                                          down_block_additional_residuals=[
                                              sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                                          ],
                                          mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                                          ).sample
                    else:
                        if args.train_mode == "SDXL":
                            with accelerator.autocast():
                                model_pred = unet(
                                    noisy_latents, timesteps, batch["input_ids"],
                                    added_cond_kwargs=batch["unet_added_conditions"]
                                ).sample
                        else:
                            # Predict the noise residual and compute loss
                            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if args.train_mode == "default":
                        # TODO: set a prior preservation flag and use that to ensure this ony happens in dreambooth
                        if not args.split_loss and not with_prior_preservation:
                            loss = instance_loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(),
                                                                                reduction="mean")
                            loss *= batch["loss_avg"]
                        else:
                            # Predict the noise residual
                            if model_pred.shape[1] == 6:
                                model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                            if with_prior_preservation:
                                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                                target, target_prior = torch.chunk(target, 2, dim=0)

                                # Compute instance loss
                                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                                # Compute prior loss
                                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(),
                                                        reduction="mean")
                            else:
                                # Compute loss
                                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    elif args.train_mode == "SDXL":
                        if with_prior_preservation:
                            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                            target, target_prior = torch.chunk(target, 2, dim=0)

                            # Compute instance loss
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                            # Compute prior loss
                            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                            # Add the prior loss to the instance loss.
                            loss = loss + args.prior_loss_weight * prior_loss
                        else:
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        if args.snr_gamma == 0:
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        else:
                            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                            # This is discussed in Section 4.2 of the same paper.
                            snr = compute_snr(timesteps, noise_scheduler)
                            mse_loss_weights = (
                                    torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[
                                        0] / snr
                            )
                            # We first calculate the original loss. Then we mean over the non-batch dimensions and
                            # re-balance the sample-wise losses with their respective loss weights.
                            # Finally, we take the mean of the rebalanced loss.
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                            loss = loss.mean()

                    accelerator.backward(loss)

                    if accelerator.sync_gradients and not train_lora:
                        if train_oft:
                            params_to_clip = oft_params.parameters()
                        elif args.train_mode == "controlnet":
                            params_to_clip = (controlnet.parameters())
                        else:
                            if stop_text_percentage != 0:
                                if text_encoder_two is not None:
                                    params_to_clip = itertools.chain(unet.parameters(), text_encoder.parameters(),
                                                                     text_encoder_two.parameters())
                                else:
                                    params_to_clip = itertools.chain(unet.parameters(),
                                                                     text_encoder.parameters())
                            else:
                                params_to_clip = unet.parameters()

                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                        if train_ema:
                            ema_unet.step(unet.parameters())
                        progress_bar.update(1)
                        grad_steps += 1
                        global_step += 1
                        accelerator.log({"loss": loss}, step=global_step)
                        train_loss = 0.0

                    optimizer.step()
                    lr_scheduler.step(args.train_batch_size)
                    if train_ema and ema_unet is not None:
                        ema_unet.step(unet)

                    optimizer.zero_grad(set_to_none=args.gradient_set_to_none)

                allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
                cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
                lr_data = lr_scheduler.get_last_lr()
                last_lr = lr_data[0]
                last_tenc_lr = 0
                stats["lr_data"] = lr_data
                try:
                    if len(optimizer.param_groups) > 1:
                        last_tenc_lr = optimizer.param_groups[1]["lr"] if train_tenc else 0
                except:
                    logger.debug("Exception getting tenc lr")
                    pass

                if 'adapt' in args.optimizer:
                    last_lr = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
                    if len(optimizer.param_groups) > 1:
                        try:
                            last_tenc_lr = optimizer.param_groups[1]["d"] * optimizer.param_groups[1]["lr"]
                        except:
                            logger.warning("Exception setting tenc weight decay")
                            traceback.print_exc()

                update_status(stats)
                global_step += args.train_batch_size
                args.revision += args.train_batch_size
                status.job_no += args.train_batch_size
                loss_step = loss.detach().item()
                if not train_oft:
                    loss_total += loss_step
                else:
                    mhe = MHE_OFT(unet, eps=args.oft_eps, rank=args.oft_rank)
                    loss_step = mhe.calculate_mhe()
                    loss_total += loss_step

                stats["session_step"] += args.train_batch_size
                stats["lifetime_step"] += args.train_batch_size
                stats["loss"] = loss_step

                logs = {
                    "lr": float(last_lr),
                    "loss": float(loss_step),
                    "vram": float(cached),
                }

                stats["vram"] = logs["vram"]
                stats["unet_lr"] = '{:.2E}'.format(Decimal(last_lr))
                if stop_text_percentage != 0:
                    stats["tenc_lr"] = '{:.2E}'.format(Decimal(last_tenc_lr))

                if args.split_loss and with_prior_preservation:
                    logs["inst_loss"] = float(instance_loss.detach().item())
                    logs["prior_loss"] = float(prior_loss.detach().item())
                    stats["instance_loss"] = logs["inst_loss"]
                    stats["prior_loss"] = logs["prior_loss"]

                if 'adapt' in args.optimizer:
                    status.textinfo2 = (
                        f"Loss: {'%.2f' % loss_step}, UNET DLR: {'{:.2E}'.format(Decimal(last_lr))}, TENC DLR: {'{:.2E}'.format(Decimal(last_tenc_lr))}, "
                        f"VRAM: {allocated}/{cached} GB"
                    )
                else:
                    status.textinfo2 = (
                        f"Loss: {'%.2f' % loss_step}, LR: {'{:.2E}'.format(Decimal(last_lr))}, "
                        f"VRAM: {allocated}/{cached} GB"
                    )

                progress_bar.update(args.train_batch_size)
                rate = progress_bar.format_dict["rate"] if "rate" in progress_bar.format_dict else None
                if rate is None:
                    rate_string = ""
                else:
                    if rate > 1:
                        rate_string = f"{rate:.2f} it/s"
                    else:
                        rate_string = f"{1 / rate:.2f} s/it" if rate != 0 else "N/A"
                stats["iterations_per_second"] = rate_string
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=args.revision)

                logs = {"epoch_loss": loss_total / len(train_dataloader)}
                accelerator.log(logs, step=global_step)
                stats["epoch_loss"] = '%.2f' % (loss_total / len(train_dataloader))

                status.job_count = max_train_steps
                status.job_no = global_step
                stats["lifetime_step"] = args.revision
                stats["session_step"] = global_step
                # status0 = f"Steps: {global_step}/{max_train_steps} (Current), {rate_string}"
                # status1 = f"{args.revision}/{lifetime_step + max_train_steps} (Lifetime), Epoch: {global_epoch}"
                status.textinfo = (
                    f"Steps: {global_step}/{max_train_steps} (Current), {rate_string}"
                    f" {args.revision}/{lifetime_step + max_train_steps} (Lifetime), Epoch: {global_epoch}"
                )
                update_status(stats)

                if math.isnan(loss_step):
                    logger.warning("Loss is NaN, your model is dead. Cancelling training.")
                    status.interrupted = True
                    if status_handler:
                        status_handler.end("Training interrrupted due to NaN loss.")

                # Log completion message
                if training_complete or status.interrupted:
                    shared.in_progress = False
                    shared.in_progress_step = 0
                    shared.in_progress_epoch = 0
                    logger.debug("  Training complete (step check).")
                    if status.interrupted:
                        state = "canceled"
                    else:
                        state = "complete"

                    status.textinfo = (
                        f"Training {state} {global_step}/{max_train_steps}, {args.revision}"
                        f" total."
                    )
                    if status_handler:
                        status_handler.end(status.textinfo)
                    break

            # Create the pipeline using the trained modules and save it.
            if accelerator.is_main_process:
                accelerator.wait_for_everyone()
                args.epoch += 1
                stats["session_epoch"] = epoch
                if args.save_embedding_every != 0 and epoch % args.save_embedding_every == 0 and epoch != 0:
                    if accelerator.is_main_process:
                        if args.checkpoint is not None and args.checkpoint != "":
                            save_path = os.path.join(args.model_dir, "checkpoints", f"checkpoint-{global_step}")
                        else:
                            save_path = args.pretrained_model_name_or_path
                        args.save()
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if args.save_preview_every != 0 and epoch % args.save_preview_every == 0 and epoch != 0:
                    if train_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    try:
                        log_validation()
                    except:
                        logger.warning("Validation failed.")
                        traceback.print_exc()
                        cleanup()
                    if train_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())
    except Exception as e:
        logger.warning(f"Exception during training: {e}")
        traceback.print_exc()
        cleanup_memory()
        if status_handler is not None:
            status_handler.end(f"Training failed: {e}")

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and save_weights:
        if controlnet is not None:
            controlnet = accelerator.unwrap_model(controlnet)
            controlnet.save_pretrained(os.path.join(args.pretrained_model_name_or_path, "controlnet"))
        else:
            unet = accelerator.unwrap_model(unet)
            if train_ema:
                ema_unet.copy_to(unet.parameters())
            elif train_lora:
                if args.train_mode == "SDXL" and train_lora:
                    unet = unet.to(torch.float32)
                    unet_lora_layers = unet_attn_processors_state_dict(unet)

                    LoraLoaderMixin.save_lora_weights(
                        save_directory=args.output_dir,
                        unet_lora_layers=unet_lora_layers,
                        text_encoder_lora_layers=None,
                    )
                else:
                    save_lora(args, stop_text_percentage, accelerator, unet, text_encoder, pbar2, True)
            else:
                update_status({"status": f"Training complete, saving pipeline."})
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    unet=unet,
                    revision=args.revision,
                )
                pipeline.save_pretrained(args.pretrained_model_name_or_path)
                del pipeline
    accelerator.end_training()
    cleanup_memory()
    if status_handler is not None:
        status_handler.end("Training complete.")
    result.msg = msg
    result.config = args
    result.samples = last_samples
    return result
