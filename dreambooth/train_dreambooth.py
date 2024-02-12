# Borrowed heavily from https://github.com/bmaltais/kohya_ss/blob/master/train_db.py and
# https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
# With some custom bits sprinkled in and some stuff from OG diffusers as well.

import itertools
import json
import logging
import math
import os
import shutil
import time
import traceback
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path

import safetensors.torch
import tomesd
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils.random import set_seed as set_seed2
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    DEISMultistepScheduler,
    UniPCMultistepScheduler, StableDiffusionXLPipeline, StableDiffusionPipeline
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import LoRAAttnProcessor2_0, LoRAAttnProcessor
from diffusers.training_utils import unet_lora_state_dict
from diffusers.utils import logging as dl
from diffusers.utils.torch_utils import randn_tensor
from torch.cuda.profiler import profile
from torch.nn.utils.parametrizations import _SpectralNorm
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from dreambooth import shared
from dreambooth.dataclasses.db_config import from_file
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.dataclasses.train_result import TrainResult
from dreambooth.dataset.bucket_sampler import BucketSampler
from dreambooth.dataset.db_dataset import DbDataset
from dreambooth.dataset.sample_dataset import SampleDataset
from dreambooth.deis_velocity import get_velocity
from dreambooth.diff_lora_to_sd_lora import convert_diffusers_to_kohya_lora
from dreambooth.diff_to_sd import compile_checkpoint, copy_diffusion_model
from dreambooth.diff_to_sdxl import compile_checkpoint as compile_checkpoint_xl
from dreambooth.memory import find_executable_batch_size
from dreambooth.optimization import UniversalScheduler, get_optimizer, get_noise_scheduler
from dreambooth.shared import status
from dreambooth.utils.gen_utils import generate_classifiers, generate_dataset
from dreambooth.utils.image_utils import db_save_image, get_scheduler_class
from dreambooth.utils.model_utils import (
    unload_system_models,
    import_model_class_from_model_name_or_path,
    safe_unpickle_disabled,
    xformerify,
    torch2ify
)
from dreambooth.utils.text_utils import encode_hidden_state, save_token_counts
from dreambooth.utils.utils import (cleanup, printm, verify_locon_installed,
                                    patch_accelerator_for_fp16_training)
from dreambooth.webhook import send_training_update
from dreambooth.xattention import optim_to
from helpers.ema_model import EMAModel
from helpers.log_parser import LogParser
from helpers.mytqdm import mytqdm
from lora_diffusion.lora import (
    set_lora_requires_grad,
)

try:
    import wandb

    # Disable annoying wandb popup?
    wandb.config.auto_init = False
except:
    pass

logger = logging.getLogger(__name__)
# define a Handler which writes DEBUG messages or higher to the sys.stderr
dl.set_verbosity_error()

last_samples = []
last_prompts = []


class ConditionalAccumulator:
    def __init__(self, accelerator, *encoders):
        self.accelerator = accelerator
        self.encoders = encoders
        self.stack = ExitStack()

    def __enter__(self):
        for encoder in self.encoders:
            if encoder is not None:
                self.stack.enter_context(self.accelerator.accumulate(encoder))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.__exit__(exc_type, exc_value, traceback)


# This implements spectral norm reparametrization. Unlike the pytorch
# built-in version, it computes the current spectral norm of the parameter
# when added and normalizes so that the norm remains constant.
class FrozenSpectralNorm(_SpectralNorm):
    @torch.autograd.no_grad()
    def __init__(
            self,
            weight: torch.Tensor,
            n_power_iterations: int = 1,
            dim: int = 0,
            eps: float = 1e-12
    ) -> None:
        super().__init__(weight, n_power_iterations, dim, eps)

        if weight.ndim == 1:
            sigma = F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            sigma = torch.dot(self._u, torch.mv(weight_mat, self._v))
        self.register_buffer('_sigma', sigma)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            return self._sigma * F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            sigma = torch.dot(u, torch.mv(weight_mat, v))
            return weight * (self._sigma / sigma)


def text_encoder_lora_state_dict(text_encoder):
    state_dict = {}

    def text_encoder_attn_modules(text_encoder):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        attn_modules = []

        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))

        return attn_modules

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict


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


to_delete = []


def clean_global_state():
    for check in to_delete:
        if check:
            try:
                obj_name = check.__name__
                del check
                # Log the name of the thing deleted
                logger.debug(f"Deleted {obj_name}")
            except:
                pass


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
        cleanup_function=clean_global_state()
    )
    def inner_loop(train_batch_size: int, gradient_accumulation_steps: int, profiler: profile):

        text_encoder = None
        text_encoder_two = None
        global last_samples
        global last_prompts
        stop_text_percentage = args.stop_text_encoder
        if not args.train_unet:
            stop_text_percentage = 1

        n_workers = 0
        args.max_token_length = int(args.max_token_length)
        if not args.pad_tokens and args.max_token_length > 75:
            logger.warning("Cannot raise token length limit above 75 when pad_tokens=False")

        if args.use_lora and args.freeze_spectral_norm:
            logger.warning("freeze_spectral_norm is not compatible with LORA")
            args.freeze_spectral_norm = False

        verify_locon_installed(args)

        precision = args.mixed_precision if not shared.force_cpu else "no"

        weight_dtype = torch.float32
        if precision == "fp16":
            weight_dtype = torch.float16
        elif precision == "bf16":
            weight_dtype = torch.bfloat16

        try:
            accelerator_logger = "tensorboard"
            # Check if Wandb API key is set
            if "WANDB_API_KEY" in os.environ:
                accelerator_logger = "wandb"
            else:
                logger.warning(
                    "Wandb API key not set. Please set WANDB_API_KEY environment variable to use wandb."
                )
            accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=precision,
                log_with=accelerator_logger,
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

        dataset_args = from_file(args.model_name)
        data_cache = DbDataset.load_cache_file(os.path.join(args.model_dir, "cache"),
                                               dataset_args.resolution) if args.cache_latents else None
        if data_cache != None:
            print(f"{len(data_cache['latents'])} cached latents")

        count, instance_prompts, class_prompts = generate_classifiers(
            args, class_gen_method=class_gen_method, accelerator=accelerator, ui=False, pbar=pbar2,
            data_cache=data_cache
        )

        save_token_counts(args, instance_prompts, 10)

        if status.interrupted:
            result.msg = "Training interrupted."
            stop_profiler(profiler)
            return result

        num_components = 5
        if args.model_type == "SDXL":
            num_components = 7
        pbar2.reset(num_components)
        pbar2.set_description("Loading model components...")

        pbar2.set_postfix(refresh=True)
        if class_gen_method == "Native Diffusers" and count > 0:
            unload_system_models()

        def create_vae():
            vae_path = (
                args.pretrained_vae_name_or_path
                if args.pretrained_vae_name_or_path
                else args.get_pretrained_model_name_or_path()
            )
            with safe_unpickle_disabled():
                new_vae = AutoencoderKL.from_pretrained(
                    vae_path,
                    subfolder=None if args.pretrained_vae_name_or_path else "vae",
                    revision=args.revision,
                )
            new_vae.requires_grad_(False)
            new_vae.to(accelerator.device, dtype=weight_dtype)
            return new_vae

        with safe_unpickle_disabled():
            # Load the tokenizer
            pbar2.set_description("Loading tokenizer...")
            pbar2.update()
            pbar2.set_postfix(refresh=True)
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(pretrained_path, "tokenizer"),
                revision=args.revision,
                use_fast=False,
            )

            tokenizer_two = None
            if args.model_type == "SDXL":
                pbar2.set_description("Loading tokenizer 2...")
                pbar2.update()
                pbar2.set_postfix(refresh=True)
                tokenizer_two = AutoTokenizer.from_pretrained(
                    os.path.join(pretrained_path, "tokenizer_2"),
                    revision=args.revision,
                    use_fast=False,
                )

            # import correct text encoder class
            text_encoder_cls = import_model_class_from_model_name_or_path(
                args.get_pretrained_model_name_or_path(), args.revision
            )

            pbar2.set_description("Loading text encoder...")
            pbar2.update()
            pbar2.set_postfix(refresh=True)
            # Load models and create wrapper for stable diffusion
            text_encoder = text_encoder_cls.from_pretrained(
                args.get_pretrained_model_name_or_path(),
                subfolder="text_encoder",
                revision=args.revision,
                torch_dtype=torch.float32,
            )

            if args.model_type == "SDXL":
                # import correct text encoder class
                text_encoder_cls_two = import_model_class_from_model_name_or_path(
                    args.get_pretrained_model_name_or_path(), args.revision, subfolder="text_encoder_2"
                )

                pbar2.set_description("Loading text encoder 2...")
                pbar2.update()
                pbar2.set_postfix(refresh=True)
                # Load models and create wrapper for stable diffusion
                text_encoder_two = text_encoder_cls_two.from_pretrained(
                    args.get_pretrained_model_name_or_path(),
                    subfolder="text_encoder_2",
                    revision=args.revision,
                    torch_dtype=torch.float32,
                )

            printm("Created tenc")
            pbar2.set_description("Loading VAE...")
            pbar2.update()
            vae = create_vae()
            printm("Created vae")

            pbar2.set_description("Loading unet...")
            pbar2.update()
            unet = UNet2DConditionModel.from_pretrained(
                args.get_pretrained_model_name_or_path(),
                subfolder="unet",
                revision=args.revision,
                torch_dtype=torch.float32,
            )

            if args.attention == "xformers" and not shared.force_cpu:
                xformerify(unet, use_lora=args.use_lora)
                xformerify(vae, use_lora=args.use_lora)

            unet = torch2ify(unet)

            if args.full_mixed_precision:
                if args.mixed_precision == "fp16":
                    patch_accelerator_for_fp16_training(accelerator)
                unet.to(accelerator.device, dtype=weight_dtype)
            else:
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

                if (
                        args.stop_text_encoder != 0
                        and accelerator.unwrap_model(text_encoder_two).dtype != torch.float32
                ):
                    logger.warning(
                        f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder_two).dtype}."
                        f" {low_precision_error_string}"
                    )

            if args.gradient_checkpointing:
                if args.train_unet:
                    unet.enable_gradient_checkpointing()
                if stop_text_percentage != 0:
                    text_encoder.gradient_checkpointing_enable()
                    if args.model_type == "SDXL":
                        text_encoder_two.gradient_checkpointing_enable()
                    if args.use_lora:
                        # We need to enable gradients on an input for gradient checkpointing to work
                        # This will not be optimized because it is not a param to optimizer
                        text_encoder.text_model.embeddings.position_embedding.requires_grad_(True)
                        if args.model_type == "SDXL":
                            text_encoder_two.text_model.embeddings.position_embedding.requires_grad_(True)
                else:
                    text_encoder.to(accelerator.device, dtype=weight_dtype)
                    if args.model_type == "SDXL":
                        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

            ema_model = None
            if args.use_ema:
                if os.path.exists(
                        os.path.join(
                            args.get_pretrained_model_name_or_path(),
                            "ema_unet",
                            "diffusion_pytorch_model.safetensors",
                        )
                ):
                    # EMA weights must be kept in fp32 even during mixed-precision training, or floating
                    # point rounding will force (almost) all updates to 0.
                    ema_unet = UNet2DConditionModel.from_pretrained(
                        args.get_pretrained_model_name_or_path(),
                        subfolder="ema_unet",
                        revision=args.revision,
                        torch_dtype=torch.float32,
                    )
                    if args.attention == "xformers" and not shared.force_cpu:
                        xformerify(ema_unet, use_lora=args.use_lora)

                    ema_model = EMAModel(
                        ema_unet, device=accelerator.device, dtype=torch.float32
                    )
                    del ema_unet
                else:
                    ema_model = EMAModel(
                        unet, device=accelerator.device, dtype=torch.float32
                    )

            def add_spectral_reparametrization(unet):
                for module in unet.modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                        weight = getattr(module, "weight", None)
                        register_parametrization(module, "weight", FrozenSpectralNorm(weight))

            def remove_spectral_reparametrization(unet):
                # Remove the spectral reparametrization and set all parameters to their adjusted versions
                for module in unet.modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                        remove_parametrizations(module, "weight", leave_parametrized=True)

            # Create shared unet/tenc learning rate variables

            learning_rate = args.learning_rate
            txt_learning_rate = args.txt_learning_rate
            if args.use_lora:
                learning_rate = args.lora_learning_rate
                txt_learning_rate = args.lora_txt_learning_rate

            if args.use_lora or not args.train_unet:
                unet.requires_grad_(False)

            unet_lora_params = None

            if args.use_lora:
                pbar2.reset(1)
                pbar2.set_description("Loading LoRA...")
                # now we will add new LoRA weights to the attention layers
                # Set correct lora layers
                unet_lora_attn_procs = {}
                unet_lora_params = []
                rank = args.lora_unet_rank

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

                    lora_attn_processor_class = (
                        LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                    )
                    if hidden_size is None:
                        logger.warning(f"Could not find hidden size for {name}. Skipping...")
                        continue
                    module = lora_attn_processor_class(
                        hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
                    )
                    unet_lora_attn_procs[name] = module
                    unet_lora_params.extend(module.parameters())

                unet.set_attn_processor(unet_lora_attn_procs)

                # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
                # So, instead, we monkey-patch the forward calls of its attention-blocks.
                if stop_text_percentage != 0:
                    # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
                    text_encoder_lora_params = LoraLoaderMixin._modify_text_encoder(
                        text_encoder, dtype=torch.float32, rank=args.lora_txt_rank
                    )

                    if args.model_type == "SDXL":
                        text_encoder_lora_params_two = LoraLoaderMixin._modify_text_encoder(
                            text_encoder_two, dtype=torch.float32, rank=args.lora_txt_rank
                        )
                        params_to_optimize = (
                            itertools.chain(unet_lora_params, text_encoder_lora_params, text_encoder_lora_params_two))
                    else:
                        params_to_optimize = (itertools.chain(unet_lora_params, text_encoder_lora_params))

                else:
                    params_to_optimize = unet_lora_params

                # Load LoRA weights if specified
                if args.lora_model_name is not None and args.lora_model_name != "":
                    logger.debug(f"Load lora from {args.lora_model_name}")
                    lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(args.lora_model_name)
                    LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet)

                    LoraLoaderMixin.load_lora_into_text_encoder(
                        lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder)
                    if text_encoder_two is not None:
                        LoraLoaderMixin.load_lora_into_text_encoder(
                            lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_two)


            elif stop_text_percentage != 0:
                if args.train_unet:
                    if args.model_type == "SDXL":
                        params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters(),
                                                             text_encoder_two.parameters())
                    else:
                        params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())
                else:
                    if args.model_type == "SDXL":
                        params_to_optimize = itertools.chain(text_encoder.parameters(), text_encoder_two.parameters())
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

            if len(optimizer.param_groups) > 2:
                try:
                    optimizer.param_groups[2]["weight_decay"] = args.tenc_weight_decay
                    optimizer.param_groups[2]["grad_clip_norm"] = args.tenc_grad_clip_norm
                except:
                    logger.warning("Exception setting tenc weight decay")
                    traceback.print_exc()

            noise_scheduler = get_noise_scheduler(args)
            global to_delete
            to_delete = [unet, text_encoder, text_encoder_two, tokenizer, tokenizer_two, optimizer, vae]

            def cleanup_memory():
                try:
                    if unet:
                        del unet
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

            with_prior_preservation = False
            tokenizers = [tokenizer] if tokenizer_two is None else [tokenizer, tokenizer_two]
            text_encoders = [text_encoder] if text_encoder_two is None else [text_encoder, text_encoder_two]
            train_dataset = generate_dataset(
                model_name=args.model_name,
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
                pbar=pbar2,
                data_cache=data_cache,
            )
            if train_dataset.class_count > 0:
                with_prior_preservation = True
            pbar2.reset()
            printm("Dataset loaded.")
            tokenizer_max_length = tokenizer.model_max_length
            if args.cache_latents:
                printm("Unloading vae.")
                del vae
                # Preserve reference to vae for later checks
                vae = None
                # TODO: Try unloading tokenizers here?
                del tokenizer
                if tokenizer_two is not None:
                    del tokenizer_two
                tokenizer = None
                tokenizer2 = None

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
                    batch_data["original_sizes_hw"] = torch.stack(
                        [torch.LongTensor(x["original_sizes_hw"]) for x in examples])
                    batch_data["crop_top_lefts"] = torch.stack(
                        [torch.LongTensor(x["crop_top_lefts"]) for x in examples])
                    batch_data["target_sizes_hw"] = torch.stack(
                        [torch.LongTensor(x["target_sizes_hw"]) for x in examples])
                return batch_data

            def collate_fn_sdxl(examples):
                input_ids = [example["input_ids"] for example in examples if not example["is_class"]]
                pixel_values = [example["image"] for example in examples if not example["is_class"]]
                add_text_embeds = [example["instance_added_cond_kwargs"]["text_embeds"] for example in examples if
                                   not example["is_class"]]
                add_time_ids = [example["instance_added_cond_kwargs"]["time_ids"] for example in examples if
                                not example["is_class"]]

                # Concat class and instance examples for prior preservation.
                # We do this to avoid doing two forward passes.
                if with_prior_preservation:
                    input_ids += [example["input_ids"] for example in examples if example["is_class"]]
                    pixel_values += [example["image"] for example in examples if example["is_class"]]
                    add_text_embeds += [example["instance_added_cond_kwargs"]["text_embeds"] for example in examples if
                                        example["is_class"]]
                    add_time_ids += [example["instance_added_cond_kwargs"]["time_ids"] for example in examples if
                                     example["is_class"]]

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

                return batch

            sampler = BucketSampler(train_dataset, train_batch_size)

            collate_fn = collate_fn_db
            if args.model_type == "SDXL":
                collate_fn = collate_fn_sdxl
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
                    first_epoch = args.lifetime_epoch
                    global_epoch = args.lifetime_epoch
                except Exception as lex:
                    logger.warning(f"Exception loading checkpoint: {lex}")

            # Add spectral norm reparametrization. See https://arxiv.org/abs/2303.06296
            # This needs to be done after the saved checkpoint is loaded (if any), because
            # saved checkpoints have normal parametrization.
            if args.freeze_spectral_norm:
                add_spectral_reparametrization(unet)

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
                save_lora = False

                if save_canceled or save_completed:
                    logger.debug("\nSave completed/canceled.")
                    if global_step > 0:
                        save_image = True
                        save_model = True
                        if args.use_lora:
                            save_lora = True
                elif is_epoch_check:
                    # Check to see if the number of epochs since last save is gt the interval
                    if 0 < save_model_interval <= session_epoch - last_model_save:
                        save_model = True
                        if args.use_lora:
                            save_lora = True
                        last_model_save = session_epoch

                    # Repeat for sample images
                    if 0 < save_image_interval <= session_epoch - last_image_save:
                        save_image = True
                        last_image_save = session_epoch

                save_snapshot = False

                if shared.status.do_save_samples:
                    save_image = True
                    shared.status.do_save_samples = False

                if shared.status.do_save_model:
                    if args.use_lora:
                        save_lora = True
                    save_model = True
                    shared.status.do_save_model = False

                save_checkpoint = False
                if save_model:
                    if save_canceled:
                        if global_step > 0:
                            logger.debug("Canceled, enabling saves.")
                            save_snapshot = args.save_state_cancel
                            save_checkpoint = args.save_ckpt_cancel
                    elif save_completed:
                        if global_step > 0:
                            logger.debug("Completed, enabling saves.")
                            save_snapshot = args.save_state_after
                            save_checkpoint = args.save_ckpt_after
                    else:
                        save_snapshot = args.save_state_during
                        save_checkpoint = args.save_ckpt_during
                    if save_checkpoint and args.use_lora:
                        save_checkpoint = False
                        save_lora = True
                if not args.use_lora:
                    save_lora = False

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
                        save_lora
                    )

                return save_model, save_image

            def save_weights(
                    save_image, save_diffusers, save_snapshot, save_checkpoint, save_lora
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
                    torch_rng_state = None
                    cuda_gpu_rng_state = None
                    cuda_cpu_rng_state = None
                    # Save random states so sample generation doesn't impact training.
                    if shared.device.type == 'cuda':
                        torch_rng_state = torch.get_rng_state()
                        cuda_gpu_rng_state = torch.cuda.get_rng_state(device="cuda")
                        cuda_cpu_rng_state = torch.cuda.get_rng_state(device="cpu")

                    if args.freeze_spectral_norm:
                        remove_spectral_reparametrization(unet)

                    optim_to(profiler, optimizer)

                    if profiler is None:
                        cleanup()

                    if vae is None:
                        printm("Loading vae.")
                        vae = create_vae()

                    printm("Creating pipeline.")
                    if args.model_type == "SDXL":
                        s_pipeline = StableDiffusionXLPipeline.from_pretrained(
                            args.get_pretrained_model_name_or_path(),
                            unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=True),
                            text_encoder=accelerator.unwrap_model(
                                text_encoder, keep_fp32_wrapper=True
                            ),
                            text_encoder_2=accelerator.unwrap_model(
                                text_encoder_two, keep_fp32_wrapper=True
                            ),
                            vae=vae.to(accelerator.device),
                            torch_dtype=weight_dtype,
                            revision=args.revision,
                            safety_checker=None,
                            requires_safety_checker=False,
                        )
                        xformerify(s_pipeline.unet, use_lora=args.use_lora)
                    else:
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
                            requires_safety_checker=False,
                        )
                        xformerify(s_pipeline.unet, use_lora=args.use_lora)
                        xformerify(s_pipeline.vae, use_lora=args.use_lora)

                    weights_dir = args.get_pretrained_model_name_or_path()

                    if user_model_dir != "":
                        loras_dir = os.path.join(user_model_dir, "Lora")
                    else:
                        model_dir = shared.models_path
                        loras_dir = os.path.join(model_dir, "Lora")
                    delete_tmp_lora = False
                    # Update the temp path if we just need to save an image
                    if save_image:
                        logger.debug("Save image is set.")
                        if args.use_lora:
                            if not save_lora:
                                logger.debug("Saving lora weights instead of checkpoint, using temp dir.")
                                save_lora = True
                                delete_tmp_lora = True
                                save_checkpoint = False
                                save_diffusers = False
                                os.makedirs(loras_dir, exist_ok=True)
                        elif not save_diffusers:
                            logger.debug("Saving checkpoint, using temp dir.")
                            save_diffusers = True
                            weights_dir = f"{weights_dir}_temp"
                            os.makedirs(weights_dir, exist_ok=True)
                        else:
                            save_lora = False
                            logger.debug(f"Save checkpoint: {save_checkpoint} save lora {save_lora}.")
                    # Is inference_mode() needed here to prevent issues when saving?
                    logger.debug(f"Loras dir: {loras_dir}")

                    # setup pt path
                    if args.custom_model_name == "":
                        lora_model_name = args.model_name
                    else:
                        lora_model_name = args.custom_model_name

                    lora_save_file = os.path.join(loras_dir, f"{lora_model_name}_{args.revision}.safetensors")

                    with accelerator.autocast(), torch.inference_mode():

                        def lora_save_function(weights, filename):
                            metadata = args.export_ss_metadata()
                            logger.debug(f"Saving lora to {filename}")
                            safetensors.torch.save_file(weights, filename, metadata=metadata)

                        if save_lora:
                            # TODO: Add a version for the lora model?
                            pbar2.reset(1)
                            pbar2.set_description("Saving Lora Weights...")
                            # setup directory
                            logger.debug(f"Saving lora to {lora_save_file}")
                            unet_lora_layers_to_save = unet_lora_state_dict(unet)
                            text_encoder_one_lora_layers_to_save = None
                            text_encoder_two_lora_layers_to_save = None
                            if args.stop_text_encoder != 0:
                                text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(text_encoder)
                            if args.model_type == "SDXL":
                                if args.stop_text_encoder != 0:
                                    text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(
                                        text_encoder_two)
                                StableDiffusionXLPipeline.save_lora_weights(
                                    loras_dir,
                                    unet_lora_layers=unet_lora_layers_to_save,
                                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                                    text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
                                    weight_name=lora_save_file,
                                    safe_serialization=True,
                                    save_function=lora_save_function
                                )
                                scheduler_args = {}

                                if "variance_type" in s_pipeline.scheduler.config:
                                    variance_type = s_pipeline.scheduler.config.variance_type

                                    if variance_type in ["learned", "learned_range"]:
                                        variance_type = "fixed_small"

                                    scheduler_args["variance_type"] = variance_type

                                s_pipeline.scheduler = UniPCMultistepScheduler.from_config(s_pipeline.scheduler.config,
                                                                                           **scheduler_args)
                                save_lora = False
                                save_model = False
                            else:
                                StableDiffusionPipeline.save_lora_weights(
                                    loras_dir,
                                    unet_lora_layers=unet_lora_layers_to_save,
                                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                                    weight_name=lora_save_file,
                                    safe_serialization=True
                                )
                                s_pipeline.scheduler = get_scheduler_class("UniPCMultistep").from_config(
                                    s_pipeline.scheduler.config)
                            s_pipeline.scheduler.config.solver_type = "bh2"
                            save_lora = False
                            save_model = False

                        elif save_diffusers:
                            # We are saving weights, we need to ensure revision is saved
                            if "_tmp" not in weights_dir:
                                args.save()
                            try:
                                out_file = None
                                status.textinfo = (
                                    f"Saving diffusion model at step {args.revision}..."
                                )
                                update_status({"status": status.textinfo})
                                pbar2.reset(1)

                                pbar2.set_description("Saving diffusion model")
                                s_pipeline.save_pretrained(
                                    weights_dir,
                                    safe_serialization=False,
                                )
                                if ema_model is not None:
                                    ema_model.save_pretrained(
                                        os.path.join(
                                            weights_dir,
                                            "ema_unet",
                                        ),
                                        safe_serialization=True,
                                    )
                                pbar2.update()

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

                                # package pt into checkpoint
                                if save_checkpoint:
                                    pbar2.reset(1)
                                    pbar2.set_description("Compiling Checkpoint")
                                    snap_rev = str(args.revision) if save_snapshot else ""
                                    if export_diffusers:
                                        copy_diffusion_model(args.model_name, os.path.join(user_model_dir, "diffusers"))
                                    else:
                                        if args.model_type == "SDXL":
                                            compile_checkpoint_xl(args.model_name, reload_models=False,
                                                                  lora_file_name=out_file,
                                                                  log=False, snap_rev=snap_rev, pbar=pbar2)
                                        else:
                                            compile_checkpoint(args.model_name, reload_models=False,
                                                               lora_file_name=out_file,
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
                        del s_pipeline
                        logger.debug(f"Loading image pipeline from {weights_dir}...")
                        if args.model_type == "SDXL":
                            s_pipeline = StableDiffusionXLPipeline.from_pretrained(
                                weights_dir, vae=vae, revision=args.revision,
                                torch_dtype=weight_dtype
                            )
                        else:
                            s_pipeline = StableDiffusionPipeline.from_pretrained(
                                weights_dir, vae=vae, revision=args.revision,
                                torch_dtype=weight_dtype
                            )
                            if args.tomesd:
                                tomesd.apply_patch(s_pipeline, ratio=args.tomesd, use_rand=False)
                        if args.use_lora:
                            s_pipeline.load_lora_weights(lora_save_file)

                        try:
                            s_pipeline.enable_vae_tiling()
                            s_pipeline.enable_vae_slicing()
                            s_pipeline.enable_sequential_cpu_offload()
                            s_pipeline.enable_xformers_memory_efficient_attention()
                        except:
                            pass

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

                        if os.path.isdir(loras_dir) and "_tmp" in loras_dir:
                            shutil.rmtree(loras_dir)

                        if os.path.isdir(weights_dir) and "_tmp" in weights_dir:
                            shutil.rmtree(weights_dir)

                        if "generator" in locals():
                            del generator

                        if not args.disable_logging:
                            try:
                                printm("Parse logs.")
                                log_images, log_names = log_parser.parse_logs(model_name=args.model_name)
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

                    # Save the lora weights if we are saving the model
                    if os.path.isfile(lora_save_file) and not delete_tmp_lora:
                        meta = args.export_ss_metadata()
                        convert_diffusers_to_kohya_lora(lora_save_file, meta, args.lora_weight)
                    else:
                        if os.path.isfile(lora_save_file):
                            os.remove(lora_save_file)

                    if args.freeze_spectral_norm:
                        add_spectral_reparametrization(unet)

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
                    if args.model_type == "SDXL":
                        text_encoder_two.eval()
                else:
                    text_encoder.train(train_tenc)
                    if args.model_type == "SDXL":
                        text_encoder_two.train(train_tenc)

                if args.use_lora:
                    if not args.lora_use_buggy_requires_grad:
                        set_lora_requires_grad(text_encoder, train_tenc)
                        # We need to enable gradients on an input for gradient checkpointing to work
                        # This will not be optimized because it is not a param to optimizer
                        text_encoder.text_model.embeddings.position_embedding.requires_grad_(train_tenc)
                        if args.model_type == "SDXL":
                            set_lora_requires_grad(text_encoder_two, train_tenc)
                            text_encoder_two.text_model.embeddings.position_embedding.requires_grad_(train_tenc)
                else:
                    text_encoder.requires_grad_(train_tenc)
                    if args.model_type == "SDXL":
                        text_encoder_two.requires_grad_(train_tenc)

                if last_tenc != train_tenc:
                    last_tenc = train_tenc
                    cleanup()

                loss_total = 0

                current_prior_loss_weight = current_prior_loss(
                    args, current_epoch=global_epoch
                )

                instance_loss = None
                prior_loss = None

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

                    with ConditionalAccumulator(accelerator, unet, text_encoder, text_encoder_two):
                        # Convert images to latent space
                        with torch.no_grad():
                            if args.cache_latents:
                                latents = batch["images"].to(accelerator.device)
                            else:
                                latents = vae.encode(
                                    batch["images"].to(dtype=weight_dtype)
                                ).latent_dist.sample()
                            latents = latents * 0.18215

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
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                        pad_tokens = args.pad_tokens if train_tenc else False
                        input_ids = batch["input_ids"]
                        encoder_hidden_states = None
                        if args.model_type != "SDXL" and text_encoder is not None:
                            encoder_hidden_states = encode_hidden_state(
                                text_encoder,
                                batch["input_ids"],
                                pad_tokens,
                                b_size,
                                args.max_token_length,
                                tokenizer_max_length,
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
                        # Get the target for loss depending on the prediction type
                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        # See http://arxiv.org/abs/2312.00210 (DREAM) algorithm 3
                        if args.use_dream and unet.config.in_channels == channels:
                            with torch.no_grad():
                                alpha_prod = noise_scheduler.alphas_cumprod.to(timesteps.device)[
                                    timesteps, None, None, None]
                                sqrt_alpha_prod = alpha_prod ** 0.5
                                sqrt_one_minus_alpha_prod = (1 - alpha_prod) ** 0.5

                                # The paper uses lambda = sqrt(1 - alpha) ** p, with p = 1 in their experiments.
                                dream_lambda = (1 - alpha_prod) ** args.dream_detail_preservation

                                if args.model_type == "SDXL":
                                    with accelerator.autocast():
                                        model_pred = unet(
                                            noisy_latents, timesteps, batch["input_ids"],
                                            added_cond_kwargs=batch["unet_added_conditions"]
                                        ).sample
                                else:
                                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                                delta_pred = (target - model_pred).detach()
                                delta_pred.mul_(dream_lambda)
                                if noise_scheduler.config.prediction_type == "epsilon":
                                    latents.add_(sqrt_one_minus_alpha_prod * delta_pred)
                                    target.add_(delta_pred)
                                elif noise_scheduler.config.prediction_type == "v_prediction":
                                    latents.add_(sqrt_one_minus_alpha_prod * delta_pred)
                                    target.add_(sqrt_alpha_prod * delta_pred)
                                else:
                                    raise ValueError(
                                        f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                                del alpha_prod, sqrt_alpha_prod, sqrt_one_minus_alpha_prod, dream_lambda, model_pred, delta_pred

                        if args.model_type == "SDXL":
                            with accelerator.autocast():
                                model_pred = unet(
                                    noisy_latents, timesteps, batch["input_ids"],
                                    added_cond_kwargs=batch["unet_added_conditions"]
                                ).sample
                        else:
                            # Predict the noise residual and compute loss
                            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                        if args.model_type != "SDXL":
                            # TODO: set a prior preservation flag and use that to ensure this ony happens in dreambooth
                            if not args.split_loss and not with_prior_preservation:
                                loss = instance_loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(),
                                                                                    reduction="mean")
                                loss *= batch["loss_avg"]
                            else:
                                # Predict the noise residual
                                if model_pred.shape[1] == 6:
                                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                                if model_pred.shape[0] > 1 and with_prior_preservation:
                                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                                    print("model shape:")
                                    print(model_pred.shape)
                                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                                    target, target_prior = torch.chunk(target, 2, dim=0)

                                    # Compute instance loss
                                    loss = instance_loss = F.mse_loss(model_pred.float(), target.float(),
                                                                      reduction="mean")

                                    # Compute prior loss
                                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(),
                                                            reduction="mean")
                                else:
                                    # Compute loss
                                    loss = instance_loss = F.mse_loss(model_pred.float(), target.float(),
                                                                      reduction="mean")
                        else:
                            if with_prior_preservation:
                                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                                if args.model_type == "SDXL":
                                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=1)
                                    target, target_prior = torch.chunk(target, 2, dim=1)
                                else:
                                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                                    target, target_prior = torch.chunk(target, 2, dim=0)

                                # Compute instance loss
                                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                                # Compute prior loss
                                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(),
                                                        reduction="mean")

                                # Add the prior loss to the instance loss.
                                loss = loss + args.prior_loss_weight * prior_loss
                            else:
                                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        accelerator.backward(loss)

                        if accelerator.sync_gradients and not args.use_lora:
                            if train_tenc:
                                if args.model_type == "SDXL":
                                    params_to_clip = itertools.chain(unet.parameters(), text_encoder.parameters(),
                                                                     text_encoder_two.parameters())
                                else:
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

                    if args.split_loss and with_prior_preservation and args.model_type != "SDXL":
                        logs["inst_loss"] = float(instance_loss.detach().item())

                        if prior_loss is not None:
                            logs["prior_loss"] = float(prior_loss.detach().item())
                        else:
                            logs["prior_loss"] = None  # or some other default value
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

                    if status.do_save_model or status.do_save_samples:
                        check_save(False)

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
