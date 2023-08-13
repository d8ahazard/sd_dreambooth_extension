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
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
import time
import traceback

import accelerate
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionControlNetPipeline, UniPCMultistepScheduler,
    ControlNetModel, EMAModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.optimization import get_scheduler
from diffusers.utils import randn_tensor
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import upload_folder
from packaging import version
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

from core.handlers.images import ImageHandler
from core.pipelines import StableDiffusionXLPipeline
from dreambooth import shared
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.dataclasses.train_result import TrainResult
from dreambooth.dataclasses.training_config import TrainingConfig
from dreambooth.oft_utils import MHE_OFT
from dreambooth.optimization import get_optimizer, get_noise_scheduler, UniversalScheduler
from dreambooth.shared import status
from dreambooth.training_utils import cleanup, load_lora, apply_oft, deepspeed_zero_init_disabled_context_manager, \
    create_vae, save_lora, encode_prompt, current_prior_loss
from dreambooth.utils.model_utils import import_model_class_from_model_name_or_path, unload_system_models, \
    disable_safe_unpickle, xformerify, unet_attn_processors_state_dict, torch2ify
from dreambooth.utils.text_utils import encode_hidden_state
from dreambooth.utils.utils import printm
from helpers.log_parser import LogParser
from helpers.mytqdm import mytqdm
from lora_diffusion.extra_networks import apply_lora
from lora_diffusion.lora import get_target_module, TEXT_ENCODER_DEFAULT_TARGET_REPLACE, set_lora_requires_grad

logger = logging.getLogger(__name__)
# define a Handler which writes DEBUG messages or higher to the sys.stderr
dl.set_verbosity_error()


# parser.add_argument(
#     "--crops_coords_top_left_h",
#     type=int,
#     default=0,
#     help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
# )
# parser.add_argument(
#     "--crops_coords_top_left_w",
#     type=int,
#     default=0,
#     help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
# )


def main(args: TrainingConfig, user: str = None) -> TrainResult:
    """
    @param args: TrainingConfig - I don't know why we removed this, but please leave it.
    @param user: User to send training updates to (for new UI)
    @return: TrainResult
    """
    cleanup()
    status_handler = None
    last_samples = []
    last_prompts = []

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
                validation_pipeline = StableDiffusionXLPipeline.from_pretrained(args.src)
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
        train_lora = False
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

    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.get_pretrained_model_name_or_path(), "tokenizer"),
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
            unet_lora_parameters = []
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
                    unet_lora_parameters.extend(module.parameters())

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

            elif train_lora:
                save_lora(args, stop_text_percentage, accelerator, unet, text_encoder, pbar2,
                          user_model_dir=user_model_dir)
            elif args.train_mode == "SDXL" and args.train_lora:
                unet_lora_layers_to_save = None

                for model in models:
                    unet_lora_layers_to_save = unet_attn_processors_state_dict(model)

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

                LoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=None,
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
    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
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
    logger.debug(f"Params to optimize: {params_to_optimize}")
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
            if tokenizer:
                del tokenizer
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

    # We ALWAYS pre-compute the additional condition embeddings needed for SDXL
    # UNet as the model is already big and it uses two text encoders.
    # TODO: when we add support for text encoder training, will reivist.
    tokenizers = [tokenizer, tokenizer_two]
    text_encoders = [text_encoder, text_encoder_two]

    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(prompt, text_encoders, tokenizers):
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
            add_text_embeds = pooled_prompt_embeds

            # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])

            prompt_embeds = prompt_embeds.to(accelerator.device)
            add_text_embeds = add_text_embeds.to(accelerator.device)
            add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
            unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return prompt_embeds, unet_added_cond_kwargs

    instance_prompt_hidden_states, instance_unet_added_conditions = compute_embeddings(
        args.instance_prompt, text_encoders, tokenizers
    )

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

    if args.train_mode == "default":
        count, instance_prompts, class_prompts = generate_classifiers(args, accelerator=accelerator, ui=False,
                                                                      pbar=pbar2)

        train_dataset = generate_dataset(
            instance_prompts=instance_prompts,
            class_prompts=class_prompts,
            batch_size=args.train_batch_size,
            tokenizer=tokenizer,
            vae=vae if args.cache_latents else None,
            debug=False,
            model_dir=args.model_dir,
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


    gc.collect()
    torch.cuda.empty_cache()

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        size=args.resolution,
        center_crop=args.center_crop,
        instance_prompt_hidden_states=instance_prompt_hidden_states,
        class_prompt_hidden_states=class_prompt_hidden_states,
        instance_unet_added_conditions=instance_unet_added_conditions,
        class_unet_added_conditions=class_unet_added_conditions,
    )

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
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if not args.cache_latents and vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = len(train_dataset)

    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
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
    if args.checkpoint != "":
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

            if args.freeze_clip_normalization:
                text_encoder.eval()
            else:
                text_encoder.train(train_tenc)

            if train_lora:
                set_lora_requires_grad(text_encoder, train_tenc)
                # We need to enable gradients on an input for gradient checkpointing to work
                # This will not be optimized because it is not a param to optimizer
                text_encoder.text_model.embeddings.position_embedding.requires_grad_(train_tenc)
            else:
                text_encoder.requires_grad_(train_tenc)

            if last_tenc != train_tenc:
                last_tenc = train_tenc
                cleanup()

            loss_total = 0

            current_prior_loss_weight = current_prior_loss(
                args, current_epoch=global_epoch
            )
            for step, batch in enumerate(train_dataloader):
                current_step.value = global_step
                # with accelerator.accumulate(training_models[0]):  # 複数モデルに対応していない模様だがとりあえずこうしておく
                if True:
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                    else:
                        with torch.no_grad():
                            # latentに変換
                            latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)

                            # NaNが含まれていれば警告を表示し0に置き換える
                            if torch.any(torch.isnan(latents)):
                                accelerator.print("NaN found in latents, replacing with zeros")
                                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                    latents = latents * sdxl_model_util.VAE_SCALE_FACTOR
                    b_size = latents.shape[0]

                    input_ids1 = batch["input_ids"]
                    input_ids2 = batch["input_ids2"]
                    if not args.cache_text_encoder_outputs:
                        with torch.set_grad_enabled(args.train_text_encoder):
                            input_ids1 = input_ids1.to(accelerator.device)
                            input_ids2 = input_ids2.to(accelerator.device)
                            encoder_hidden_states1, encoder_hidden_states2, pool2 = sdxl_train_util.get_hidden_states(
                                args,
                                input_ids1,
                                input_ids2,
                                tokenizer1,
                                tokenizer2,
                                text_encoder1,
                                text_encoder2,
                                None if not args.full_fp16 else weight_dtype,
                            )
                    else:
                        encoder_hidden_states1 = []
                        encoder_hidden_states2 = []
                        pool2 = []
                        for input_id1, input_id2 in zip(input_ids1, input_ids2):
                            input_id1_cache_key = tuple(input_id1.squeeze(0).flatten().tolist())
                            input_id2_cache_key = tuple(input_id2.squeeze(0).flatten().tolist())
                            encoder_hidden_states1.append(text_encoder1_cache[input_id1_cache_key])
                            hidden_states2, p2 = text_encoder2_cache[input_id2_cache_key]
                            encoder_hidden_states2.append(hidden_states2)
                            pool2.append(p2)
                        encoder_hidden_states1 = torch.stack(encoder_hidden_states1).to(accelerator.device).to(
                            weight_dtype)
                        encoder_hidden_states2 = torch.stack(encoder_hidden_states2).to(accelerator.device).to(
                            weight_dtype)
                        pool2 = torch.stack(pool2).to(accelerator.device).to(weight_dtype)

                    # get size embeddings
                    orig_size = batch["original_sizes_hw"]
                    crop_size = batch["crop_top_lefts"]
                    target_size = batch["target_sizes_hw"]
                    embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size,
                                                               accelerator.device).to(weight_dtype)

                    # concat embeddings
                    vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                    text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                    # Sample noise, sample a random timestep for each image, and add noise to the latents,
                    # with noise offset and/or multires noise if specified
                    noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args,
                                                                                                       noise_scheduler,
                                                                                                       latents)

                    noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

                    # Predict the noise residual
                    with accelerator.autocast():
                        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)

                    target = noise

                    if args.min_snr_gamma:
                        # do not mean over batch dimension for snr weight or scale v-pred loss
                        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                        loss = loss.mean([1, 2, 3])

                        if args.min_snr_gamma:
                            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)

                        loss = loss.mean()  # mean over batch dimension
                    else:
                        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in training_models:
                            params_to_clip.extend(m.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if accelerator.is_main_process:
                if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                    )

                    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                    scheduler_args = {}

                    if "variance_type" in pipeline.scheduler.config:
                        variance_type = pipeline.scheduler.config.variance_type

                        if variance_type in ["learned", "learned_range"]:
                            variance_type = "fixed_small"

                        scheduler_args["variance_type"] = variance_type

                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipeline.scheduler.config, **scheduler_args
                    )

                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
                    pipeline_args = {"prompt": args.validation_prompt}

                    images = [
                        pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)
                    ]

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    del pipeline
                    torch.cuda.empty_cache()
    except:
        pass

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = unet_attn_processors_state_dict(unet)

        LoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=None,
        )

        # Final inference
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype
        )

        # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.load_lora_weights(args.output_dir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
            images = [
                pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                for _ in range(args.num_validation_images)
            ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                prompt=args.instance_prompt,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
