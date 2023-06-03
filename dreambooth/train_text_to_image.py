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
import gc
import json
import math
import os
import random
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
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from pydantic.types import Decimal
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

from core.handlers.images import ImageHandler
from dreambooth.dataclasses.finetune_config import FinetuneConfig
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.dataset.bucket_sampler import BucketSampler
from dreambooth.dataset.ft_dataset import FtDataset
from dreambooth.utils.image_utils import get_scheduler_class
from dreambooth.utils.model_utils import xformerify, torch2ify
from helpers.mytqdm import mytqdm

if is_wandb_available():
    pass

logger = get_logger(__name__)

status_handler = None
app_user = None


def cleanup(do_print: bool = False):
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except:
        print("cleanup exception")
    if do_print:
        print("Cleanup completed.")


def main(args: FinetuneConfig, user: str):
    cleanup()
    global status_handler
    status_handler = None
    global app_user
    app_user = user
    logging_dir = os.path.join(args.model_dir, "logging")
    try:
        from core.handlers.status import StatusHandler
        status_handler = StatusHandler(user_name=user)
    except:
        pass

    if status_handler is not None:
        status_handler.start()

    def update_status(data: dict):
        if status_handler is not None:
            if "iterations_per_second" in data:
                data = {"status": json.dumps(data)}
            status_handler.update(items=data)

    def log_validation():
        logger.info("Running validation... ")
        update_status({"status": "Generating samples..."})
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            safety_checker=None,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
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
        all_train_data = fine_dataset.train_img_data
        for item in all_train_data:
            all_prompts.append(item['text'])

        prompts = []
        image_handler = ImageHandler(user_name=app_user)
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
                logger.warn(f"image logging not implemented for {tracker.name}")

        del validation_pipeline
        cleanup()

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    logger.info(accelerator.state, main_process_only=False)
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

    # Load scheduler, tokenizer and models.
    logger.debug(f"Loading scheduler, tokenizer and models from {args.pretrained_model_name_or_path}")
    ptp = args.pretrained_model_name_or_path
    noise_scheduler = DDPMScheduler.from_pretrained(os.path.join(ptp, "scheduler"))
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(ptp, "tokenizer"))

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(os.path.join(ptp, "text_encoder"))
        vae = AutoencoderKL.from_pretrained(os.path.join(ptp, "vae"))

    unet = UNet2DConditionModel.from_pretrained(os.path.join(ptp, "unet"))

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(os.path.join(ptp, "unet"))
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
    else:
        ema_unet = None

    if args.attention == "xformers":
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        xformerify(unet, False)
        xformerify(vae, False)

    unet = torch2ify(unet)

    # TODO: Add the other attention modules here

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 8:
        torch.backends.cuda.matmul.allow_tf32 = True


    # Initialize the optimizer
    if args.optimizer == "8bit AdamW":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    def cleanup_memory():
        try:
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
            if fine_dataset:
                del fine_dataset
            if lr_scheduler:
                del lr_scheduler
            if vae:
                del vae
        except:
            pass
        cleanup(True)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()

    fine_dataset = FtDataset(
        instance_data_dir=args.train_data_dir,
        resolution=args.resolution,
        batch_size=args.train_batch_size,
        hflip=False,
        shuffle_tags=True,
        strict_tokens=True,
        dynamic_img_norm=False,
        not_pad_tokens=False,
        model_dir=args.model_dir,
        tokenizer=tokenizer,
        vae=vae,
        cache_latents=args.cache_latents,
        user=user,
        use_dir_tags=args.use_dir_tags,
    )

    if args.cache_latents:
        logger.debug("Unloading vae.")
        del vae
        # Preserve reference to vae for later checks
        vae = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def collate_fn(examples):
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

    sampler = BucketSampler(fine_dataset, args.train_batch_size)

    train_dataloader = torch.utils.data.DataLoader(
        fine_dataset,
        batch_size=1,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = len(fine_dataset)

    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if not args.cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = len(fine_dataset)

    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.model_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(fine_dataset)}")
    logger.info(f"  Num batches each epoch = {len(fine_dataset) // args.train_batch_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Precision = {args.mixed_precision}")
    logger.info(f"  Device = {accelerator.device}")
    logger.info(f"  Optimizer = {args.optimizer}")
    logger.info(f"  Resuming from checkpoint: {args.snapshot}")
    logger.info(f"  Gradient Checkpointing: {args.gradient_checkpointing}")
    logger.info(f"  EMA: {args.use_ema}")
    global_step = 0
    first_epoch = 0
    resume_step = 0

    resume_from_checkpoint = False
    # Potentially load in the weights and states from a previous save
    if args.snapshot != "":
        if args.snapshot != "latest":
            path = os.path.join(args.model_dir, "checkpoints", args.snapshot)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(os.path.join(args.model_dir, "checkpoints"))
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.snapshot}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = False
        else:
            resume_from_checkpoint = True
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.model_dir, "snapshots", path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = mytqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch(0), Steps")
    progress_bar.set_postfix(refresh=True)
    update_status({"progress_1_total": max_train_steps, "progress_1_job_current": global_step})
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
        "steps_per_epoch": len(fine_dataset),
        "iterations_per_second": 0.0,
        "vram": round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
    }

    canceled = False
    save_weights = False
    try:
        for epoch in range(first_epoch, args.num_train_epochs):
            epoch_steps = 0
            grad_steps = 0
            progress_bar.set_description(f"Epoch({epoch}), Steps")
            progress_bar.set_postfix(refresh=True)
            if status_handler is not None:
                if status_handler.status.canceled:
                    logger.debug("Training canceled, returning.")
                    canceled = True
                    break
            unet.train()
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if status_handler is not None:
                    if status_handler.status.canceled:
                        logger.debug("Training canceled, returning.")
                        canceled = True
                        if epoch > 0:
                            save_weights = True
                        break
                if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    global_step += 1
                    stats["session_step"] += args.train_batch_size
                    stats["lifetime_step"] += args.train_batch_size
                    update_status(stats)
                    continue

                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    if args.cache_latents:
                        latents = batch["images"].to(accelerator.device)
                    else:
                        latents = vae.encode(
                            batch["images"].to(dtype=weight_dtype)
                        ).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
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
                    b_size = latents.shape[0]
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

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    # Predict the noise residual and compute loss
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if args.snr_gamma == 0.0:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(timesteps)
                        mse_loss_weights = (
                                torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[
                                    0] / snr
                        )
                        # We first calculate the original loss. Then we mean over the non-batch dimensions and
                        # rebalance the sample-wise losses with their respective loss weights.
                        # Finally, we take the mean of the rebalanced loss.
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.gradient_set_to_none)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_unet.step(unet.parameters())
                    progress_bar.update(1)
                    grad_steps += 1
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0
                epoch_steps += 1
                last_lr = lr_scheduler.get_last_lr()[0]
                step_loss = loss.detach().item()
                logs = {"step_loss": step_loss, "lr": last_lr}
                cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
                stats["lr_data"] = last_lr
                stats["session_step"] += 1
                stats["lifetime_step"] += 1
                stats["loss"] = step_loss

                stats["vram"] = cached
                stats["unet_lr"] = '{:.2E}'.format(Decimal(last_lr))

                rate = progress_bar.format_dict.get("rate", 0.0)
                # Parse the rate as a float if it's a string
                if isinstance(rate, str):
                    rate = float(rate)
                rate = rate * args.train_batch_size if rate is not None else 0.0
                if rate is None:
                    rate_string = ""
                else:
                    if rate > 1:
                        rate_string = f"{rate:.2f} it/s"
                    else:
                        rate_string = f"{1 / rate:.2f} s/it" if rate != 0 else "N/A"
                stats["iterations_per_second"] = rate_string
                update_status(stats)
                progress_bar.set_postfix(**logs)
                args.revision += args.train_batch_size
                if global_step >= max_train_steps:
                    break

            if accelerator.is_main_process:
                args.epoch += 1
                logger.debug(f"Epoch steps: {epoch_steps}, grad steps: {grad_steps}")
                stats["session_epoch"] = epoch
                if epoch % args.save_embedding_every == 0 and epoch != 0:
                    if accelerator.is_main_process:
                        if args.save_ckpt_during:
                            save_path = os.path.join(args.model_dir, "checkpoints", f"checkpoint-{global_step}")
                        else:
                            save_path = args.pretrained_model_name_or_path
                        args.save()
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if args.save_preview_every != 0 and epoch % args.save_preview_every == 0 and epoch != 0:
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    try:
                        log_validation()
                    except:
                        logger.warning("Validation failed.")
                        traceback.print_exc()
                        cleanup()
                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())
    except Exception as e:
        logger.warning(f"Exception during training: {e}")
        cleanup_memory()
        if status_handler is not None:
            status_handler.end(f"Training failed: {e}")

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and save_weights:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        update_status({"status": f"Training complete, saving pipeline."})
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.pretrained_model_name_or_path)
        del pipeline
    accelerator.end_training()
    cleanup_memory()
    if status_handler is not None:
        status_handler.end("Training complete.")
