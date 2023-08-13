# Borrowed heavily from https://github.com/bmaltais/kohya_ss/blob/master/train_db.py and
# https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
# With some custom bits sprinkled in and some stuff from OG diffusers as well.

import itertools
import json
import logging
import math
import os
import shutil
import tempfile
import time
import traceback
from decimal import Decimal
from pathlib import Path

import tomesd
import torch
import torch.backends.cuda
import torch.backends.cudnn
from accelerate import Accelerator
from accelerate.utils.random import set_seed as set_seed2
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    DEISMultistepScheduler,
    UniPCMultistepScheduler
)
from diffusers.utils import logging as dl, is_xformers_available
from packaging import version
from torch.cuda.profiler import profile
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from dreambooth import shared
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.dataclasses.train_result import TrainResult
from dreambooth.dataset.bucket_sampler import BucketSampler
from dreambooth.dataset.sample_dataset import SampleDataset
from dreambooth.deis_velocity import get_velocity
from dreambooth.diff_to_sd import compile_checkpoint, copy_diffusion_model
from dreambooth.memory import find_executable_batch_size
from dreambooth.optimization import UniversalScheduler, get_optimizer, get_noise_scheduler
from dreambooth.shared import status
from dreambooth.utils.gen_utils import generate_classifiers, generate_dataset
from dreambooth.utils.image_utils import db_save_image, get_scheduler_class
from dreambooth.utils.model_utils import (
    unload_system_models,
    import_model_class_from_model_name_or_path,
    disable_safe_unpickle,
    enable_safe_unpickle,
    xformerify,
    torch2ify,
)
from dreambooth.utils.text_utils import encode_hidden_state
from dreambooth.utils.utils import cleanup, printm, verify_locon_installed
from dreambooth.webhook import send_training_update
from dreambooth.xattention import optim_to
from helpers.ema_model import EMAModel
from helpers.log_parser import LogParser
from helpers.mytqdm import mytqdm
from lora_diffusion.extra_networks import save_extra_networks
from lora_diffusion.lora import (
    save_lora_weight,
    TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
    get_target_module,
    set_lora_requires_grad,
)

logger = logging.getLogger(__name__)
# define a Handler which writes DEBUG messages or higher to the sys.stderr
dl.set_verbosity_error()

last_samples = []
last_prompts = []


def check_and_patch_scheduler(scheduler_class):
    if not hasattr(scheduler_class, 'get_velocity'):
        logger.debug(f"Adding 'get_velocity' method to {scheduler_class.__name__}...")
        scheduler_class.get_velocity = get_velocity


try:
    check_and_patch_scheduler(DEISMultistepScheduler)
    check_and_patch_scheduler(UniPCMultistepScheduler)
except:
    logger.warning("Exception while adding 'get_velocity' method to the schedulers.")

export_diffusers = False
user_model_dir = ""


def set_seed(deterministic: bool):
    if deterministic:
        torch.backends.cudnn.deterministic = True
        seed = 0
        set_seed2(seed)
    else:
        torch.backends.cudnn.deterministic = False


def current_prior_loss(args, current_epoch):
    if not args.prior_loss_scale:
        return args.prior_loss_weight
    if not args.prior_loss_target:
        args.prior_loss_target = 150
    if not args.prior_loss_weight_min:
        args.prior_loss_weight_min = 0.1
    if current_epoch >= args.prior_loss_target:
        return args.prior_loss_weight_min
    percentage_completed = current_epoch / args.prior_loss_target
    prior = (
            args.prior_loss_weight * (1 - percentage_completed)
            + args.prior_loss_weight_min * percentage_completed
    )
    printm(f"Prior: {prior}")
    return prior


def stop_profiler(profiler):
    if profiler is not None:
        try:
            logger.debug("Stopping profiler.")
            profiler.stop()
        except:
            pass


def main(class_gen_method: str = "Native Diffusers", user: str = None) -> TrainResult:
    """
    @param class_gen_method: Image Generation Library.
    @param user: User to send training updates to (for new UI)
    @return: TrainResult
    """
    args = shared.db_model_config
    status_handler = None
    logging_dir = Path(args.model_dir, "logging")
    global export_diffusers, user_model_dir
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
        logger.debug(f"Loaded config: {args.__dict__}")
    except:
        pass
    log_parser = LogParser()

    def update_status(data: dict):
        if status_handler is not None:
            if "iterations_per_second" in data:
                data = {"status": json.dumps(data)}
            status_handler.update(items=data)

    result = TrainResult
    result.config = args

    set_seed(args.deterministic)

    @find_executable_batch_size(
        starting_batch_size=args.train_batch_size,
        starting_grad_size=args.gradient_accumulation_steps,
        logging_dir=logging_dir,
    )
    def inner_loop(train_batch_size: int, gradient_accumulation_steps: int, profiler: profile):

        text_encoder = None
        global last_samples
        global last_prompts
        stop_text_percentage = args.stop_text_encoder
        if not args.train_unet:
            stop_text_percentage = 1

        n_workers = 0
        args.max_token_length = int(args.max_token_length)
        if not args.pad_tokens and args.max_token_length > 75:
            logger.warning("Cannot raise token length limit above 75 when pad_tokens=False")

        verify_locon_installed(args)

        precision = args.mixed_precision if not shared.force_cpu else "no"

        weight_dtype = torch.float32
        if precision == "fp16":
            weight_dtype = torch.float16
        elif precision == "bf16":
            weight_dtype = torch.bfloat16

        try:
            accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=precision,
                log_with="tensorboard",
                project_dir=logging_dir,
                cpu=shared.force_cpu,
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

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            args.get_pretrained_model_name_or_path(), args.revision
        )

        # Load models and create wrapper for stable diffusion
        text_encoder = text_encoder_cls.from_pretrained(
            args.get_pretrained_model_name_or_path(),
            subfolder="text_encoder",
            revision=args.revision,
            torch_dtype=torch.float32,
        )
        printm("Created tenc")
        vae = create_vae()
        printm("Created vae")

        unet = UNet2DConditionModel.from_pretrained(
            args.get_pretrained_model_name_or_path(),
            subfolder="unet",
            revision=args.revision,
            torch_dtype=torch.float32,
        )

        if args.attention == "xformers" and not shared.force_cpu:
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

        # Check that all trainable models are in full precision
        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - "
            "even if doing mixed precision training. copy of the weights should still be float32."
        )

        if accelerator.unwrap_model(unet).dtype != torch.float32:
            logger.warning(
                f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
            )

        if (
                args.stop_text_encoder != 0
                and accelerator.unwrap_model(text_encoder).dtype != torch.float32
        ):
            logger.warning(
                f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        if args.gradient_checkpointing:
            if args.train_unet:
                unet.enable_gradient_checkpointing()
            if stop_text_percentage != 0:
                text_encoder.gradient_checkpointing_enable()
                if args.use_lora:
                    # We need to enable gradients on an input for gradient checkpointing to work
                    # This will not be optimized because it is not a param to optimizer
                    text_encoder.text_model.embeddings.position_embedding.requires_grad_(True)
            else:
                text_encoder.to(accelerator.device, dtype=weight_dtype)

        ema_model = None
        if args.use_ema:
            if os.path.exists(
                    os.path.join(
                        args.get_pretrained_model_name_or_path(),
                        "ema_unet",
                        "diffusion_pytorch_model.safetensors",
                    )
            ):
                ema_unet = UNet2DConditionModel.from_pretrained(
                    args.get_pretrained_model_name_or_path(),
                    subfolder="ema_unet",
                    revision=args.revision,
                    torch_dtype=torch.float32,
                )
                if args.attention == "xformers" and not shared.force_cpu:
                    xformerify(ema_unet)

                ema_model = EMAModel(
                    ema_unet, device=accelerator.device, dtype=weight_dtype
                )
                del ema_unet
            else:
                ema_model = EMAModel(
                    unet, device=accelerator.device, dtype=weight_dtype
                )

        # Create shared unet/tenc learning rate variables

        learning_rate = args.learning_rate
        txt_learning_rate = args.txt_learning_rate
        if args.use_lora:
            learning_rate = args.lora_learning_rate
            txt_learning_rate = args.lora_txt_learning_rate

        if args.use_lora or not args.train_unet:
            unet.requires_grad_(False)

        unet_lora_params = None
        text_encoder_lora_params = None
        lora_path = None
        lora_txt = None

        if args.use_lora:
            if args.lora_model_name:
                lora_path = os.path.join(args.model_dir, "loras", args.lora_model_name)
                lora_txt = lora_path.replace(".pt", "_txt.pt")

                if not os.path.exists(lora_path) or not os.path.isfile(lora_path):
                    lora_path = None
                    lora_txt = None

            injectable_lora = get_target_module("injection", args.use_lora_extended)
            target_module = get_target_module("module", args.use_lora_extended)

            unet_lora_params, _ = injectable_lora(
                unet,
                r=args.lora_unet_rank,
                loras=lora_path,
                target_replace_module=target_module,
            )

            if stop_text_percentage != 0:
                text_encoder.requires_grad_(False)
                inject_trainable_txt_lora = get_target_module("injection", False)
                text_encoder_lora_params, _ = inject_trainable_txt_lora(
                    text_encoder,
                    target_replace_module=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
                    r=args.lora_txt_rank,
                    loras=lora_txt,
                )
            printm("Lora loaded")
            cleanup()
            printm("Cleaned")

            if stop_text_percentage != 0:
                params_to_optimize = [
                    {
                        "params": itertools.chain(*unet_lora_params),
                        "lr": learning_rate,
                    },
                    {
                        "params": itertools.chain(*text_encoder_lora_params),
                        "lr": txt_learning_rate,
                    },
                ]
            else:
                params_to_optimize = itertools.chain(*unet_lora_params)

        elif stop_text_percentage != 0:
            if args.train_unet:
                params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())
            else:
                params_to_optimize = itertools.chain(text_encoder.parameters())
        else:
            params_to_optimize = unet.parameters()

        optimizer = get_optimizer(args.optimizer, learning_rate, args.weight_decay, params_to_optimize)
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
                if unet:
                    del unet
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
                if unet_lora_params:
                    del unet_lora_params
            except:
                pass
            cleanup(True)

        if args.cache_latents:
            vae.to(accelerator.device, dtype=weight_dtype)
            vae.requires_grad_(False)
            vae.eval()

        if status.interrupted:
            result.msg = "Training interrupted."
            stop_profiler(profiler)
            return result

        printm("Loading dataset...")
        pbar2.reset()
        pbar2.set_description("Loading dataset")

        train_dataset = generate_dataset(
            model_name=args.model_name,
            instance_prompts=instance_prompts,
            class_prompts=class_prompts,
            batch_size=train_batch_size,
            tokenizer=tokenizer,
            vae=vae if args.cache_latents else None,
            debug=False,
            model_dir=args.model_dir,
            pbar=pbar2
        )
        pbar2.reset()
        printm("Dataset loaded.")

        if args.cache_latents:
            printm("Unloading vae.")
            del vae
            # Preserve reference to vae for later checks
            vae = None

        if status.interrupted:
            result.msg = "Training interrupted."
            stop_profiler(profiler)
            return result

        if train_dataset.__len__ == 0:
            msg = "Please provide a directory with actual images in it."
            logger.warning(msg)
            status.textinfo = msg
            update_status({"status": status})
            cleanup_memory()
            result.msg = msg
            result.config = args
            stop_profiler(profiler)
            return result

        def collate_fn(examples):
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
            return batch_data

        sampler = BucketSampler(train_dataset, train_batch_size)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=n_workers,
        )

        max_train_steps = args.num_train_epochs * len(train_dataset)

        # This is separate, because optimizer.step is only called once per "step" in training, so it's not
        # affected by batch size
        sched_train_steps = args.num_train_epochs * train_dataset.num_train_images

        lr_scale_pos = args.lr_scale_pos
        if class_prompts:
            lr_scale_pos *= 2

        lr_scheduler = UniversalScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            total_training_steps=sched_train_steps,
            min_lr=args.learning_rate_min,
            total_epochs=args.num_train_epochs,
            num_cycles=args.lr_cycles,
            power=args.lr_power,
            factor=args.lr_factor,
            scale_pos=lr_scale_pos,
            unet_lr=learning_rate,
            tenc_lr=txt_learning_rate,
        )

        # create ema, fix OOM
        if args.use_ema:
            if stop_text_percentage != 0:
                (
                    ema_model.model,
                    unet,
                    text_encoder,
                    optimizer,
                    train_dataloader,
                    lr_scheduler,
                ) = accelerator.prepare(
                    ema_model.model,
                    unet,
                    text_encoder,
                    optimizer,
                    train_dataloader,
                    lr_scheduler,
                )
            else:
                (
                    ema_model.model,
                    unet,
                    optimizer,
                    train_dataloader,
                    lr_scheduler,
                ) = accelerator.prepare(
                    ema_model.model, unet, optimizer, train_dataloader, lr_scheduler
                )
        else:
            if stop_text_percentage != 0:
                (
                    unet,
                    text_encoder,
                    optimizer,
                    train_dataloader,
                    lr_scheduler,
                ) = accelerator.prepare(
                    unet, text_encoder, optimizer, train_dataloader, lr_scheduler
                )
            else:
                unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    unet, optimizer, train_dataloader, lr_scheduler
                )

        if not args.cache_latents and vae is not None:
            vae.to(accelerator.device, dtype=weight_dtype)

        if stop_text_percentage == 0:
            text_encoder.to(accelerator.device, dtype=weight_dtype)
        # Afterwards we recalculate our number of training epochs
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers will initialize automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("dreambooth")

        # Train!
        total_batch_size = (
                train_batch_size * accelerator.num_processes * gradient_accumulation_steps
        )
        max_train_epochs = args.num_train_epochs
        # we calculate our number of tenc training epochs
        text_encoder_epochs = round(max_train_epochs * stop_text_percentage)
        global_step = 0
        global_epoch = 0
        session_epoch = 0
        first_epoch = 0
        resume_step = 0
        last_model_save = 0
        last_image_save = 0
        resume_from_checkpoint = False
        new_hotness = os.path.join(
            args.model_dir, "checkpoints", f"checkpoint-{args.snapshot}"
        )
        if os.path.exists(new_hotness):
            logger.debug(f"Resuming from checkpoint {new_hotness}")

            try:
                import modules.shared
                no_safe = modules.shared.cmd_opts.disable_safe_unpickle
                modules.shared.cmd_opts.disable_safe_unpickle = True
            except:
                no_safe = False

            try:
                import modules.shared
                accelerator.load_state(new_hotness)
                modules.shared.cmd_opts.disable_safe_unpickle = no_safe
                global_step = resume_step = args.revision
                resume_from_checkpoint = True
                first_epoch = args.epoch
                global_epoch = first_epoch
            except Exception as lex:
                logger.warning(f"Exception loading checkpoint: {lex}")

        # if shared.in_progress:
        #    logger.debug("  ***** OOM detected. Resuming from last step *****")
        #    max_train_steps = max_train_steps - shared.in_progress_step
        #    max_train_epochs = max_train_epochs - shared.in_progress_epoch
        #    session_epoch = shared.in_progress_epoch
        #    text_encoder_epochs = (shared.in_progress_epoch/max_train_epochs)*text_encoder_epochs
        # else:
        #    shared.in_progress = True

        logger.debug("  ***** Running training *****")
        if shared.force_cpu:
            logger.debug(f"  TRAINING WITH CPU ONLY")
        logger.debug(f"  Num batches each epoch = {len(train_dataset) // train_batch_size}")
        logger.debug(f"  Num Epochs = {max_train_epochs}")
        logger.debug(f"  Batch Size Per Device = {train_batch_size}")
        logger.debug(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.debug(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.debug(f"  Text Encoder Epochs: {text_encoder_epochs}")
        logger.debug(f"  Total optimization steps = {sched_train_steps}")
        logger.debug(f"  Total training steps = {max_train_steps}")
        logger.debug(f"  Resuming from checkpoint: {resume_from_checkpoint}")
        logger.debug(f"  First resume epoch: {first_epoch}")
        logger.debug(f"  First resume step: {resume_step}")
        logger.debug(f"  Lora: {args.use_lora}, Optimizer: {args.optimizer}, Prec: {precision}")
        logger.debug(f"  Gradient Checkpointing: {args.gradient_checkpointing}")
        logger.debug(f"  EMA: {args.use_ema}")
        logger.debug(f"  UNET: {args.train_unet}")
        logger.debug(f"  Freeze CLIP Normalization Layers: {args.freeze_clip_normalization}")
        logger.debug(f"  LR{' (Lora)' if args.use_lora else ''}: {learning_rate}")
        if stop_text_percentage > 0:
            logger.debug(f"  Tenc LR{' (Lora)' if args.use_lora else ''}: {txt_learning_rate}")
        logger.debug(f"  LoRA Extended: {args.use_lora_extended}")
        logger.debug(f"  V2: {args.v2}")

        os.environ.__setattr__("CUDA_LAUNCH_BLOCKING", 1)

        def check_save(is_epoch_check=False):
            nonlocal last_model_save
            nonlocal last_image_save
            save_model_interval = args.save_embedding_every
            save_image_interval = args.save_preview_every
            save_completed = session_epoch >= max_train_epochs
            save_canceled = status.interrupted
            save_image = False
            save_model = False
            if not save_canceled and not save_completed:
                # Check to see if the number of epochs since last save is gt the interval
                if 0 < save_model_interval <= session_epoch - last_model_save:
                    save_model = True
                    last_model_save = session_epoch

                # Repeat for sample images
                if 0 < save_image_interval <= session_epoch - last_image_save:
                    save_image = True
                    last_image_save = session_epoch

            else:
                logger.debug("\nSave completed/canceled.")
                if global_step > 0:
                    save_image = True
                    save_model = True

            save_snapshot = False
            save_lora = False
            save_checkpoint = False

            if is_epoch_check:
                if shared.status.do_save_samples:
                    save_image = True
                    shared.status.do_save_samples = False

                if shared.status.do_save_model:
                    save_model = True
                    shared.status.do_save_model = False

            if save_model:
                if save_canceled:
                    if global_step > 0:
                        logger.debug("Canceled, enabling saves.")
                        save_lora = args.save_lora_cancel
                        save_snapshot = args.save_state_cancel
                        save_checkpoint = args.save_ckpt_cancel
                elif save_completed:
                    if global_step > 0:
                        logger.debug("Completed, enabling saves.")
                        save_lora = args.save_lora_after
                        save_snapshot = args.save_state_after
                        save_checkpoint = args.save_ckpt_after
                else:
                    save_lora = args.save_lora_during
                    save_snapshot = args.save_state_during
                    save_checkpoint = args.save_ckpt_during

            if (
                    save_checkpoint
                    or save_snapshot
                    or save_lora
                    or save_image
                    or save_model
            ):
                save_weights(
                    save_image,
                    save_model,
                    save_snapshot,
                    save_checkpoint,
                    save_lora,
                )

            return save_model

        def save_weights(
                save_image, save_model, save_snapshot, save_checkpoint, save_lora
        ):
            global last_samples
            global last_prompts
            nonlocal vae
            nonlocal pbar2

            printm(" Saving weights.")
            pbar2.reset()
            pbar2.set_description("Saving weights/samples...")
            pbar2.set_postfix(refresh=True)

            # Create the pipeline using the trained modules and save it.
            if accelerator.is_main_process:
                printm("Pre-cleanup.")

                # Save random states so sample generation doesn't impact training.
                if shared.device.type == 'cuda':
                    torch_rng_state = torch.get_rng_state()
                    cuda_gpu_rng_state = torch.cuda.get_rng_state(device="cuda")
                    cuda_cpu_rng_state = torch.cuda.get_rng_state(device="cpu")

                optim_to(profiler, optimizer)

                if profiler is None:
                    cleanup()

                if vae is None:
                    printm("Loading vae.")
                    vae = create_vae()

                printm("Creating pipeline.")

                s_pipeline = DiffusionPipeline.from_pretrained(
                    args.get_pretrained_model_name_or_path(),
                    unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=True),
                    text_encoder=accelerator.unwrap_model(
                        text_encoder, keep_fp32_wrapper=True
                    ),
                    vae=vae,
                    torch_dtype=weight_dtype,
                    revision=args.revision,
                    safety_checker=None,
                    requires_safety_checker=None,
                )

                # Is inference_mode() needed here to prevent issues when saving?
                with accelerator.autocast(), torch.inference_mode():
                    if save_model:
                        # We are saving weights, we need to ensure revision is saved
                        args.save()
                        try:
                            out_file = None
                            # Loras resume from pt
                            if not args.use_lora:
                                if save_snapshot:
                                    pbar2.reset(1)
                                    pbar2.set_description("Saving Snapshot")
                                    status.textinfo = (
                                        f"Saving snapshot at step {args.revision}..."
                                    )
                                    update_status({"status": status.textinfo})
                                    accelerator.save_state(
                                        os.path.join(
                                            args.model_dir,
                                            "checkpoints",
                                            f"checkpoint-{args.revision}",
                                        )
                                    )
                                    pbar2.update()

                                # We should save this regardless, because it's our fallback if no snapshot exists.
                                status.textinfo = (
                                    f"Saving diffusion model at step {args.revision}..."
                                )
                                update_status({"status": status.textinfo})
                                pbar2.reset(1)
                                pbar2.set_description("Saving diffusion model")
                                s_pipeline.save_pretrained(
                                    os.path.join(args.model_dir, "working"),
                                    safe_serialization=True,
                                )
                                if ema_model is not None:
                                    ema_model.save_pretrained(
                                        os.path.join(
                                            args.get_pretrained_model_name_or_path(),
                                            "ema_unet",
                                        ),
                                        safe_serialization=True,
                                    )
                                pbar2.update()

                            elif save_lora:
                                pbar2.reset(1)
                                pbar2.set_description("Saving Lora Weights...")
                                # setup directory
                                if user_model_dir != "":
                                    loras_dir = os.path.join(user_model_dir, "loras")
                                else:
                                    loras_dir = os.path.join(args.model_dir, "loras")
                                os.makedirs(loras_dir, exist_ok=True)
                                # setup pt path
                                if args.custom_model_name == "":
                                    lora_model_name = args.model_name
                                else:
                                    lora_model_name = args.custom_model_name
                                lora_file_prefix = f"{lora_model_name}_{args.revision}"
                                out_file = os.path.join(
                                    loras_dir, f"{lora_file_prefix}.pt"
                                )
                                # create pt
                                tgt_module = get_target_module(
                                    "module", args.use_lora_extended
                                )
                                save_lora_weight(s_pipeline.unet, out_file, tgt_module)

                                modelmap = {"unet": (s_pipeline.unet, tgt_module)}
                                # save text_encoder
                                if stop_text_percentage != 0:
                                    out_txt = out_file.replace(".pt", "_txt.pt")
                                    modelmap["text_encoder"] = (
                                        s_pipeline.text_encoder,
                                        TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
                                    )
                                    save_lora_weight(
                                        s_pipeline.text_encoder,
                                        out_txt,
                                        target_replace_module=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
                                    )
                                    pbar2.update()
                                # save extra_net
                                if args.save_lora_for_extra_net:
                                    pbar2.reset(1)
                                    pbar2.set_description("Saving Extra Networks")
                                    os.makedirs(
                                        shared.ui_lora_models_path, exist_ok=True
                                    )
                                    out_safe = os.path.join(
                                        shared.ui_lora_models_path,
                                        f"{lora_file_prefix}.safetensors",
                                    )
                                    save_extra_networks(modelmap, out_safe)
                                    pbar2.update(0)
                            # package pt into checkpoint
                            if save_checkpoint:
                                pbar2.reset(1)
                                pbar2.set_description("Compiling Checkpoint")
                                snap_rev = str(args.revision) if save_snapshot else ""
                                if export_diffusers:
                                    copy_diffusion_model(args.model_name, os.path.join(user_model_dir, "diffusers"))
                                else:
                                    compile_checkpoint(args.model_name, reload_models=False, lora_file_name=out_file,
                                                       log=False, snap_rev=snap_rev, pbar=pbar2)
                                printm("Restored, moved to acc.device.")
                                pbar2.update()
                        except Exception as ex:
                            logger.warning(f"Exception saving checkpoint/model: {ex}")
                            traceback.print_exc()
                            pass
                    save_dir = args.model_dir

                if save_image:
                    logger.debug("Saving images...")
                    # Get the path to a temporary directory
                    tmp_dir = tempfile.mkdtemp()
                    s_pipeline.save_pretrained(tmp_dir, safe_serialization=True)
                    del s_pipeline
                    cleanup()
                    s_pipeline = DiffusionPipeline.from_pretrained(
                        tmp_dir,
                        vae=vae,
                        torch_dtype=weight_dtype,
                        low_cpu_mem_usage=False,
                        device_map=None
                    )

                    if args.tomesd:
                        tomesd.apply_patch(s_pipeline, ratio=args.tomesd, use_rand=False)

                    s_pipeline.enable_vae_tiling()
                    s_pipeline.enable_vae_slicing()
                    try:
                        s_pipeline.enable_xformers_memory_efficient_attention()
                    except:
                        pass
                    s_pipeline.enable_sequential_cpu_offload()

                    s_pipeline.scheduler = get_scheduler_class("UniPCMultistep").from_config(s_pipeline.scheduler.config)
                    s_pipeline.scheduler.config.solver_type = "bh2"
                    samples = []
                    sample_prompts = []
                    last_samples = []
                    last_prompts = []
                    status.textinfo = (
                        f"Saving preview image(s) at step {args.revision}..."
                    )
                    update_status({"status": status.textinfo})
                    try:
                        s_pipeline.set_progress_bar_config(disable=True)
                        sample_dir = os.path.join(save_dir, "samples")
                        os.makedirs(sample_dir, exist_ok=True)

                        sd = SampleDataset(args)
                        prompts = sd.prompts
                        logger.debug(f"Generating {len(prompts)} samples...")

                        concepts = args.concepts()
                        if args.sanity_prompt:
                            epd = PromptData(
                                prompt=args.sanity_prompt,
                                seed=args.sanity_seed,
                                negative_prompt=concepts[
                                    0
                                ].save_sample_negative_prompt,
                                resolution=(args.resolution, args.resolution),
                            )
                            prompts.append(epd)

                        prompt_lengths = len(prompts)
                        if args.disable_logging:
                            pbar2.reset(prompt_lengths)
                        else:
                            pbar2.reset(prompt_lengths + 2)
                        pbar2.set_description("Generating Samples")
                        ci = 0
                        for c in prompts:
                            c.out_dir = os.path.join(args.model_dir, "samples")
                            generator = torch.manual_seed(int(c.seed))
                            s_image = s_pipeline(
                                c.prompt,
                                num_inference_steps=c.steps,
                                guidance_scale=c.scale,
                                negative_prompt=c.negative_prompt,
                                height=c.resolution[1],
                                width=c.resolution[0],
                                generator=generator,
                            ).images[0]
                            sample_prompts.append(c.prompt)
                            image_name = db_save_image(
                                s_image,
                                c,
                                custom_name=f"sample_{args.revision}-{ci}",
                            )
                            shared.status.current_image = image_name
                            shared.status.sample_prompts = [c.prompt]
                            update_status({"images": [image_name], "prompts": [c.prompt]})
                            samples.append(image_name)
                            pbar2.update()
                            ci += 1
                        for sample in samples:
                            last_samples.append(sample)
                        for prompt in sample_prompts:
                            last_prompts.append(prompt)
                        del samples
                        del prompts
                    except:
                        logger.warning(f"Exception saving sample.")
                        traceback.print_exc()
                        pass

                    del s_pipeline
                    printm("Starting cleanup.")

                    if os.path.isdir(tmp_dir):
                        shutil.rmtree(tmp_dir)

                if save_image:
                    if "generator" in locals():
                        del generator

                    if not args.disable_logging:
                        try:
                            printm("Parse logs.")
                            log_images, log_names = log_parser.parse_logs(
                                model_name=args.model_name
                            )
                            pbar2.update()
                            for log_image in log_images:
                                last_samples.append(log_image)
                            for log_name in log_names:
                                last_prompts.append(log_name)

                            del log_images
                            del log_names
                        except Exception as l:
                            traceback.print_exc()
                            logger.warning(f"Exception parsing logz: {l}")
                            pass

                    send_training_update(
                        last_samples,
                        args.model_name,
                        last_prompts,
                        global_step,
                        args.revision
                    )

                    status.sample_prompts = last_prompts
                    status.current_image = last_samples
                    update_status({"images": last_samples, "prompts": last_prompts})
                    pbar2.update()

                if args.cache_latents:
                    printm("Unloading vae.")
                    del vae
                    # Preserve the reference again
                    vae = None

                status.current_image = last_samples
                update_status({"images": last_samples})

                cleanup()
                printm("Cleanup.")

                optim_to(profiler, optimizer, accelerator.device)

                # Restore all random states to avoid having sampling impact training.
                if shared.device.type == 'cuda':
                    torch.set_rng_state(torch_rng_state)
                    torch.cuda.set_rng_state(cuda_cpu_rng_state, device="cpu")
                    torch.cuda.set_rng_state(cuda_gpu_rng_state, device="cuda")

                cleanup()
                printm("Completed saving weights.")
                pbar2.reset()

        # Only show the progress bar once on each machine, and do not send statuses to the new UI.
        progress_bar = mytqdm(
            range(global_step, max_train_steps),
            disable=not accelerator.is_local_main_process,
            position=0
        )
        progress_bar.set_description("Steps")
        progress_bar.set_postfix(refresh=True)
        args.revision = (
            args.revision if isinstance(args.revision, int) else
            int(args.revision) if str(args.revision).strip() else
            0
        )
        lifetime_step = args.revision
        lifetime_epoch = args.epoch
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
            "unet_lr": learning_rate,
            "tenc_lr": txt_learning_rate,
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
        for epoch in range(first_epoch, max_train_epochs):
            if training_complete:
                logger.debug("Training complete, breaking epoch.")
                break

            if args.train_unet:
                unet.train()
            elif args.use_lora and not args.lora_use_buggy_requires_grad:
                set_lora_requires_grad(unet, False)

            train_tenc = epoch < text_encoder_epochs
            if stop_text_percentage == 0:
                train_tenc = False

            if args.freeze_clip_normalization:
                text_encoder.eval()
            else:
                text_encoder.train(train_tenc)

            if args.use_lora:
                if not args.lora_use_buggy_requires_grad:
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
                # Skip steps until we reach the resumed step
                if (
                        resume_from_checkpoint
                        and epoch == first_epoch
                        and step < resume_step
                ):
                    progress_bar.update(train_batch_size)
                    progress_bar.reset()
                    status.job_count = max_train_steps
                    status.job_no += train_batch_size
                    stats["session_step"] += train_batch_size
                    stats["lifetime_step"] += train_batch_size
                    update_status(stats)
                    continue
                with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    with torch.no_grad():
                        if args.cache_latents:
                            latents = batch["images"].to(accelerator.device)
                        else:
                            latents = vae.encode(
                                batch["images"].to(dtype=weight_dtype)
                            ).latent_dist.sample()
                        latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    if args.offset_noise < 0:
                        noise = torch.randn_like(latents, device=latents.device)
                    else:
                        noise = torch.randn_like(
                            latents, device=latents.device
                        ) + args.offset_noise * torch.randn(
                            latents.shape[0],
                            latents.shape[1],
                            1,
                            1,
                            device=latents.device,
                        )
                    b_size = latents.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (b_size,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    pad_tokens = args.pad_tokens if train_tenc else False
                    encoder_hidden_states = encode_hidden_state(
                        text_encoder,
                        batch["input_ids"],
                        pad_tokens,
                        b_size,
                        args.max_token_length,
                        tokenizer.model_max_length,
                        args.clip_skip,
                    )

                    # Predict the noise residual
                    if args.use_ema and args.ema_predict:
                        noise_pred = ema_model(
                            noisy_latents, timesteps, encoder_hidden_states
                        ).sample
                    else:
                        noise_pred = unet(
                            noisy_latents, timesteps, encoder_hidden_states
                        ).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    if not args.split_loss:
                        loss = instance_loss = torch.nn.functional.mse_loss(
                            noise_pred.float(), target.float(), reduction="mean"
                        )
                        loss *= batch["loss_avg"]

                    else:
                        model_pred_chunks = torch.split(noise_pred, 1, dim=0)
                        target_pred_chunks = torch.split(target, 1, dim=0)
                        instance_chunks = []
                        prior_chunks = []
                        instance_pred_chunks = []
                        prior_pred_chunks = []

                        # Iterate over the list of boolean values in batch["types"]
                        for i, is_prior in enumerate(batch["types"]):
                            # If is_prior is False, append the corresponding chunk to instance_chunks
                            if not is_prior:
                                instance_chunks.append(model_pred_chunks[i])
                                instance_pred_chunks.append(target_pred_chunks[i])
                            # If is_prior is True, append the corresponding chunk to prior_chunks
                            else:
                                prior_chunks.append(model_pred_chunks[i])
                                prior_pred_chunks.append(target_pred_chunks[i])

                        # initialize with 0 in case we are having batch = 1
                        instance_loss = torch.tensor(0)
                        prior_loss = torch.tensor(0)

                        # Concatenate the chunks in instance_chunks to form the model_pred_instance tensor
                        if len(instance_chunks):
                            model_pred = torch.stack(instance_chunks, dim=0)
                            target = torch.stack(instance_pred_chunks, dim=0)
                            instance_loss = torch.nn.functional.mse_loss(
                                model_pred.float(), target.float(), reduction="mean"
                            )

                        if len(prior_pred_chunks):
                            model_pred_prior = torch.stack(prior_chunks, dim=0)
                            target_prior = torch.stack(prior_pred_chunks, dim=0)
                            prior_loss = torch.nn.functional.mse_loss(
                                model_pred_prior.float(),
                                target_prior.float(),
                                reduction="mean",
                            )

                        if len(instance_chunks) and len(prior_chunks):
                            # Add the prior loss to the instance loss.
                            loss = instance_loss + current_prior_loss_weight * prior_loss
                        elif len(instance_chunks):
                            loss = instance_loss
                        else:
                            loss = prior_loss * current_prior_loss_weight

                    accelerator.backward(loss)

                    if accelerator.sync_gradients and not args.use_lora:
                        if train_tenc:
                            params_to_clip = itertools.chain(unet.parameters(), text_encoder.parameters())
                        else:
                            params_to_clip = unet.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, 1)

                    optimizer.step()
                    lr_scheduler.step(train_batch_size)
                    if args.use_ema and ema_model is not None:
                        ema_model.step(unet)
                    if profiler is not None:
                        profiler.step()

                    optimizer.zero_grad(set_to_none=args.gradient_set_to_none)

                    # Track current step and epoch for OOM resume
                    # shared.in_progress_epoch = global_epoch
                    # shared.in_progress_steps = global_step

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
                del noise_pred
                del latents
                del encoder_hidden_states
                del noise
                del timesteps
                del noisy_latents
                del target

                global_step += train_batch_size
                args.revision += train_batch_size
                status.job_no += train_batch_size
                loss_step = loss.detach().item()
                loss_total += loss_step

                stats["session_step"] += train_batch_size
                stats["lifetime_step"] += train_batch_size
                stats["loss"] = loss_step

                logs = {
                    "lr": float(last_lr),
                    "loss": float(loss_step),
                    "vram": float(cached),
                }

                stats["vram"] = logs["vram"]
                stats["unet_lr"] = '{:.2E}'.format(Decimal(last_lr))
                stats["tenc_lr"] = '{:.2E}'.format(Decimal(last_tenc_lr))

                if args.split_loss:
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

                progress_bar.update(train_batch_size)
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

            accelerator.wait_for_everyone()

            args.epoch += 1
            global_epoch += 1
            lifetime_epoch += 1
            session_epoch += 1
            stats["session_epoch"] += 1
            stats["lifetime_epoch"] += 1
            lr_scheduler.step(is_epoch=True)
            status.job_count = max_train_steps
            status.job_no = global_step
            update_status(stats)
            check_save(True)

            if args.num_train_epochs > 1:
                training_complete = session_epoch >= max_train_epochs

            if training_complete or status.interrupted:
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

            # Do this at the very END of the epoch, only after we're sure we're not done
            if args.epoch_pause_frequency > 0 and args.epoch_pause_time > 0:
                if not session_epoch % args.epoch_pause_frequency:
                    logger.debug(
                        f"Giving the GPU a break for {args.epoch_pause_time} seconds."
                    )
                    for i in range(args.epoch_pause_time):
                        if status.interrupted:
                            training_complete = True
                            logger.debug("Training complete, interrupted.")
                            shared.in_progress = False
                            shared.in_progress_step = 0
                            shared.in_progress_epoch = 0
                            if status_handler:
                                status_handler.end("Training interrrupted.")
                            break
                        time.sleep(1)

        cleanup_memory()
        accelerator.end_training()
        result.msg = msg
        result.config = args
        result.samples = last_samples
        stop_profiler(profiler)
        return result

    return inner_loop()
