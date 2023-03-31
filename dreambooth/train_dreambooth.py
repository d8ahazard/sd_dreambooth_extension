# Borrowed heavily from https://github.com/bmaltais/kohya_ss/blob/master/train_db.py and
# https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
# With some custom bits sprinkled in and some stuff from OG diffusers as well.

import itertools
import logging
import math
import os
import time
import traceback
from decimal import Decimal
from pathlib import Path

import importlib_metadata
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.utils.checkpoint
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
from tensorflow.python.framework.random_seed import set_seed as set_seed1
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
)

logger = logging.getLogger(__name__)
# define a Handler which writes DEBUG messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)
dl.set_verbosity_error()

last_samples = []
last_prompts = []

try:
    diff_version = importlib_metadata.version("diffusers")
    version_string = diff_version.split(".")
    major_version = int(version_string[0])
    minor_version = int(version_string[1])
    patch_version = int(version_string[2])
    if minor_version < 14 or (minor_version == 14 and patch_version <= 0):
        print(
            "The version of diffusers is less than or equal to 0.14.0. Performing monkey-patch..."
        )
        DEISMultistepScheduler.get_velocity = get_velocity
        UniPCMultistepScheduler.get_velocity = get_velocity
    else:
        print(
            "The version of diffusers is greater than 0.14.0, hopefully they merged the PR by now"
        )
except:
    print("Exception monkey-patching DEIS scheduler.")

export_diffusers = False
diffusers_dir = ""
try:
    from core.handlers.config import ConfigHandler
    from core.handlers.models import ModelHandler
    ch = ConfigHandler()
    mh = ModelHandler()
    export_diffusers = ch.get_item("export_diffusers", "dreambooth", True)
    diffusers_dir = os.path.join(mh.models_path, "diffusers")
except:
    pass


def set_seed(deterministic: bool):
    if deterministic:
        torch.backends.cudnn.deterministic = True
        seed = 0
        set_seed1(seed)
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
            print("Stopping profiler.")
            profiler.stop()
        except:
            pass


def main(class_gen_method: str = "Native Diffusers") -> TrainResult:
    """
    @param class_gen_method: Image Generation Library.
    @return: TrainResult
    """
    args = shared.db_model_config
    logging_dir = Path(args.model_dir, "logging")
    log_parser = LogParser()

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
            print("Cannot raise token length limit above 75 when pad_tokens=False")

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
            print(msg)
            result.msg = msg
            result.config = args
            stop_profiler(profiler)
            return result
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
            print(msg)
            status.textinfo = msg
            stop_text_percentage = 0
        count, instance_prompts, class_prompts = generate_classifiers(
            args, class_gen_method=class_gen_method, accelerator=accelerator, ui=False
        )
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
                else args.pretrained_model_name_or_path
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
            os.path.join(args.pretrained_model_name_or_path, "tokenizer"),
            revision=args.revision,
            use_fast=False,
        )

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision
        )

        # Load models and create wrapper for stable diffusion
        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            torch_dtype=torch.float32,
        )
        printm("Created tenc")
        vae = create_vae()
        printm("Created vae")

        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
            torch_dtype=torch.float32,
        )
        unet = torch2ify(unet)

        # Check that all trainable models are in full precision
        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
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
            xformerify(unet)
            xformerify(vae)

        if accelerator.unwrap_model(unet).dtype != torch.float32:
            print(
                f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
            )

        if (
                args.stop_text_encoder != 0
                and accelerator.unwrap_model(text_encoder).dtype != torch.float32
        ):
            print(
                f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        try:
            # Apparently, some versions of torch don't have a cuda_version flag? IDK, but it breaks my runpod.
            if (
                    torch.cuda.is_available()
                    and float(torch.cuda_version) >= 11.0
                    and args.tf32_enable
            ):
                print("Attempting to enable TF32.")
                torch.backends.cuda.matmul.allow_tf32 = True
        except:
            pass

        if args.gradient_checkpointing:
            if args.train_unet:
                unet.enable_gradient_checkpointing()
            if stop_text_percentage != 0:
                text_encoder.gradient_checkpointing_enable()
                if args.use_lora:
                    text_encoder.text_model.embeddings.requires_grad_(True)
            else:
                text_encoder.to(accelerator.device, dtype=weight_dtype)

        ema_model = None
        if args.use_ema:
            if os.path.exists(
                    os.path.join(
                        args.pretrained_model_name_or_path,
                        "ema_unet",
                        "diffusion_pytorch_model.safetensors",
                    )
            ):
                ema_unet = UNet2DConditionModel.from_pretrained(
                    args.pretrained_model_name_or_path,
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

            args.learning_rate = args.lora_learning_rate
            if stop_text_percentage != 0:
                params_to_optimize = [
                    {
                        "params": itertools.chain(*unet_lora_params),
                        "lr": args.lora_learning_rate,
                    },
                    {
                        "params": itertools.chain(*text_encoder_lora_params),
                        "lr": args.lora_txt_learning_rate,
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

        optimizer = get_optimizer(args, params_to_optimize)

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
        train_dataset = generate_dataset(
            model_name=args.model_name,
            instance_prompts=instance_prompts,
            class_prompts=class_prompts,
            batch_size=train_batch_size,
            tokenizer=tokenizer,
            vae=vae if args.cache_latents else None,
            debug=False,
            model_dir=args.model_dir,
        )

        printm("Dataset loaded.")

        if args.cache_latents:
            printm("Unloading vae.")
            del vae
            # Preserve reference to vae for later checks
            vae = None
        cleanup()
        if status.interrupted:
            result.msg = "Training interrupted."
            stop_profiler(profiler)
            return result

        if train_dataset.__len__ == 0:
            msg = "Please provide a directory with actual images in it."
            print(msg)
            status.textinfo = msg
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
            accelerator.print(f"Resuming from checkpoint {new_hotness}")

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
                print(f"Exception loading checkpoint: {lex}")

        print("  ***** Running training *****")
        if shared.force_cpu:
            print(f"  TRAINING WITH CPU ONLY")
        print(f"  Num batches each epoch = {len(train_dataset) // train_batch_size}")
        print(f"  Num Epochs = {max_train_epochs}")
        print(f"  Batch Size Per Device = {train_batch_size}")
        print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Text Encoder Epochs: {text_encoder_epochs}")
        print(f"  Total optimization steps = {sched_train_steps}")
        print(f"  Total training steps = {max_train_steps}")
        print(f"  Resuming from checkpoint: {resume_from_checkpoint}")
        print(f"  First resume epoch: {first_epoch}")
        print(f"  First resume step: {resume_step}")
        print(f"  Lora: {args.use_lora}, Optimizer: {args.optimizer}, Prec: {precision}")
        print(f"  Gradient Checkpointing: {args.gradient_checkpointing}")
        print(f"  EMA: {args.use_ema}")
        print(f"  UNET: {args.train_unet}")
        print(f"  Freeze CLIP Normalization Layers: {args.freeze_clip_normalization}")
        print(f"  LR: {args.learning_rate}")
        if args.use_lora_extended:
            print(f"  LoRA Extended: {args.use_lora_extended}")
        if args.use_lora and stop_text_percentage > 0:
            print(f"  LoRA Text Encoder LR: {args.lora_txt_learning_rate}")
        print(f"  V2: {args.v2}")

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
                print("\nSave completed/canceled.")
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
                        print("Canceled, enabling saves.")
                        save_lora = args.save_lora_cancel
                        save_snapshot = args.save_state_cancel
                        save_checkpoint = args.save_ckpt_cancel
                elif save_completed:
                    if global_step > 0:
                        print("Completed, enabling saves.")
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

            printm(" Saving weights.")
            pbar = mytqdm(
                range(4),
                desc="Saving weights",
                disable=not accelerator.is_local_main_process,
                position=1
            )
            pbar.set_postfix(refresh=True)

            # Create the pipeline using the trained modules and save it.
            if accelerator.is_main_process:
                printm("Pre-cleanup.")
                
                # Save random states so sample generation doesn't impact training.
                if shared.device.type == 'cuda':
                    torch_rng_state = torch.get_rng_state()
                    cuda_gpu_rng_state = torch.cuda.get_rng_state(device="cuda")
                    cuda_cpu_rng_state = torch.cuda.get_rng_state(device="cpu")

                optim_to(profiler, optimizer)
                
                if profiler is not None:
                    cleanup()

                if vae is None:
                    printm("Loading vae.")
                    vae = create_vae()

                printm("Creating pipeline.")

                s_pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
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

                scheduler_class = get_scheduler_class(args.scheduler)
                if args.attention == "xformers" and not shared.force_cpu:
                    xformerify(s_pipeline)

                s_pipeline.scheduler = scheduler_class.from_config(
                    s_pipeline.scheduler.config
                )
                if "UniPC" in args.scheduler:
                    s_pipeline.scheduler.config.solver_type = "bh2"

                s_pipeline = s_pipeline.to(accelerator.device)

                with accelerator.autocast(), torch.inference_mode():
                    if save_model:
                        # We are saving weights, we need to ensure revision is saved
                        args.save()
                        try:
                            out_file = None
                            # Loras resume from pt
                            if not args.use_lora:
                                if save_snapshot:
                                    pbar.set_description("Saving Snapshot")
                                    status.textinfo = (
                                        f"Saving snapshot at step {args.revision}..."
                                    )
                                    accelerator.save_state(
                                        os.path.join(
                                            args.model_dir,
                                            "checkpoints",
                                            f"checkpoint-{args.revision}",
                                        )
                                    )
                                    pbar.update()

                                # We should save this regardless, because it's our fallback if no snapshot exists.
                                status.textinfo = (
                                    f"Saving diffusion model at step {args.revision}..."
                                )
                                pbar.set_description("Saving diffusion model")
                                s_pipeline.save_pretrained(
                                    os.path.join(args.model_dir, "working"),
                                    safe_serialization=True,
                                )
                                if ema_model is not None:
                                    ema_model.save_pretrained(
                                        os.path.join(
                                            args.pretrained_model_name_or_path,
                                            "ema_unet",
                                        ),
                                        safe_serialization=True,
                                    )
                                pbar.update()

                            elif save_lora:
                                pbar.set_description("Saving Lora Weights...")
                                # setup directory
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
                                    pbar.update()
                                # save extra_net
                                if args.save_lora_for_extra_net:
                                    os.makedirs(
                                        shared.ui_lora_models_path, exist_ok=True
                                    )
                                    out_safe = os.path.join(
                                        shared.ui_lora_models_path,
                                        f"{lora_file_prefix}.safetensors",
                                    )
                                    save_extra_networks(modelmap, out_safe)
                            # package pt into checkpoint
                            if save_checkpoint:
                                pbar.set_description("Compiling Checkpoint")
                                snap_rev = str(args.revision) if save_snapshot else ""
                                if export_diffusers:
                                    copy_diffusion_model(args.model_name, diffusers_dir)
                                else:
                                    compile_checkpoint(args.model_name, reload_models=False, lora_file_name=out_file,
                                                       log=False, snap_rev=snap_rev, pbar=pbar)
                                printm("Restored, moved to acc.device.")
                        except Exception as ex:
                            print(f"Exception saving checkpoint/model: {ex}")
                            traceback.print_exc()
                            pass

                    save_dir = args.model_dir
                    if save_image:
                        samples = []
                        sample_prompts = []
                        last_samples = []
                        last_prompts = []
                        status.textinfo = (
                            f"Saving preview image(s) at step {args.revision}..."
                        )
                        try:
                            s_pipeline.set_progress_bar_config(disable=True)
                            sample_dir = os.path.join(save_dir, "samples")
                            os.makedirs(sample_dir, exist_ok=True)
                            with accelerator.autocast(), torch.inference_mode():
                                sd = SampleDataset(args)
                                prompts = sd.prompts
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
                                pbar.set_description("Generating Samples")
                                pbar.reset(len(prompts) + 2)
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
                                    samples.append(image_name)
                                    pbar.update()
                                    ci += 1
                                for sample in samples:
                                    last_samples.append(sample)
                                for prompt in sample_prompts:
                                    last_prompts.append(prompt)
                                del samples
                                del prompts

                        except Exception as em:
                            print(f"Exception saving sample: {em}")
                            traceback.print_exc()
                            pass
                printm("Starting cleanup.")
                del s_pipeline
                if save_image:
                    if "generator" in locals():
                        del generator
                    try:
                        printm("Parse logs.")
                        log_images, log_names = log_parser.parse_logs(
                            model_name=args.model_name
                        )
                        pbar.update()
                        for log_image in log_images:
                            last_samples.append(log_image)
                        for log_name in log_names:
                            last_prompts.append(log_name)
                        send_training_update(
                            last_samples,
                            args.model_name,
                            last_prompts,
                            global_step,
                            args.revision,
                        )

                        del log_images
                        del log_names
                    except Exception as l:
                        traceback.print_exc()
                        print(f"Exception parsing logz: {l}")
                        pass
                    status.sample_prompts = last_prompts
                    status.current_image = last_samples
                    pbar.update()

                if args.cache_latents:
                    printm("Unloading vae.")
                    del vae
                    # Preserve the reference again
                    vae = None

                status.current_image = last_samples
                printm("Cleanup.")

                optim_to(profiler, optimizer, accelerator.device)

                # Restore all random states to avoid having sampling impact training.
                if shared.device.type == 'cuda':
                    torch.set_rng_state(torch_rng_state)
                    torch.cuda.set_rng_state(cuda_cpu_rng_state, device="cpu")
                    torch.cuda.set_rng_state(cuda_gpu_rng_state, device="cuda")

                cleanup()
                printm("Completed saving weights.")

        # Only show the progress bar once on each machine.
        progress_bar = mytqdm(
            range(global_step, max_train_steps),
            disable=not accelerator.is_local_main_process,
            position=0,
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
        training_complete = False
        msg = ""

        last_tenc = 0 < text_encoder_epochs
        if stop_text_percentage == 0:
            last_tenc = False

        for epoch in range(first_epoch, max_train_epochs):
            if training_complete:
                print("Training complete, breaking epoch.")
                break

            if args.train_unet:
                unet.train()

            train_tenc = epoch < text_encoder_epochs
            if stop_text_percentage == 0:
                train_tenc = False

            if args.freeze_clip_normalization:
                text_encoder.eval()
            else:
                text_encoder.train(train_tenc)

            if not args.use_lora:
                text_encoder.requires_grad_(train_tenc)
            elif train_tenc:
                text_encoder.text_model.embeddings.requires_grad_(True)

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

                allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
                cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
                last_lr = lr_scheduler.get_last_lr()[0]

                global_step += train_batch_size
                args.revision += train_batch_size
                status.job_no += train_batch_size

                del noise_pred
                del latents
                del encoder_hidden_states
                del noise
                del timesteps
                del noisy_latents
                del target

                loss_step = loss.detach().item()
                loss_total += loss_step
                if args.split_loss:
                    logs = {
                        "lr": float(last_lr),
                        "loss": float(loss_step),
                        "inst_loss": float(instance_loss.detach().item()),
                        "prior_loss": float(prior_loss.detach().item()),
                        "vram": float(cached),
                    }
                else:
                    logs = {
                        "lr": float(last_lr),
                        "loss": float(loss_step),
                        "vram": float(cached),
                    }

                status.textinfo2 = (
                    f"Loss: {'%.2f' % loss_step}, LR: {'{:.2E}'.format(Decimal(last_lr))}, "
                    f"VRAM: {allocated}/{cached} GB"
                )
                progress_bar.update(train_batch_size)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=args.revision)

                logs = {"epoch_loss": loss_total / len(train_dataloader)}
                accelerator.log(logs, step=global_step)

                status.job_count = max_train_steps
                status.job_no = global_step
                status.textinfo = (
                    f"Steps: {global_step}/{max_train_steps} (Current),"
                    f" {args.revision}/{lifetime_step + max_train_steps} (Lifetime), Epoch: {global_epoch}"
                )

                if math.isnan(loss_step):
                    print("Loss is NaN, your model is dead. Cancelling training.")
                    status.interrupted = True

                # Log completion message
                if training_complete or status.interrupted:
                    print("  Training complete (step check).")
                    if status.interrupted:
                        state = "cancelled"
                    else:
                        state = "complete"

                    status.textinfo = (
                        f"Training {state} {global_step}/{max_train_steps}, {args.revision}"
                        f" total."
                    )

                    break

            accelerator.wait_for_everyone()

            args.epoch += 1
            global_epoch += 1
            lifetime_epoch += 1
            session_epoch += 1
            lr_scheduler.step(is_epoch=True)
            status.job_count = max_train_steps
            status.job_no = global_step

            check_save(True)

            if args.num_train_epochs > 1:
                training_complete = session_epoch >= max_train_epochs

            if training_complete or status.interrupted:
                print("  Training complete (step check).")
                if status.interrupted:
                    state = "cancelled"
                else:
                    state = "complete"

                status.textinfo = (
                    f"Training {state} {global_step}/{max_train_steps}, {args.revision}"
                    f" total."
                )

                break

            # Do this at the very END of the epoch, only after we're sure we're not done
            if args.epoch_pause_frequency > 0 and args.epoch_pause_time > 0:
                if not session_epoch % args.epoch_pause_frequency:
                    print(
                        f"Giving the GPU a break for {args.epoch_pause_time} seconds."
                    )
                    for i in range(args.epoch_pause_time):
                        if status.interrupted:
                            training_complete = True
                            print("Training complete, interrupted.")
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
