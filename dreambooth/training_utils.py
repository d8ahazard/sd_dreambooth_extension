import gc
import logging
import os

import accelerate
import torch
from accelerate.state import AcceleratorState
from accelerate.utils.random import set_seed as set_seed2
from diffusers import AutoencoderKL
from huggingface_hub import model_info

from lora_diffusion.extra_networks import save_extra_networks
from lora_diffusion.lora import get_target_module, TEXT_ENCODER_DEFAULT_TARGET_REPLACE

logger = logging.getLogger(__name__)


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
    logger.debug(f"Prior: {prior}")
    return prior


def cleanup(do_print: bool = False):
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except:
        logger.warning("cleanup exception")
    if do_print:
        print("Cleanup completed.")


def model_has_vae(args):
    config_file_name = os.path.join("vae", AutoencoderKL.config_name)
    if os.path.isdir(args.pretrained_model_name_or_path):
        config_file_name = os.path.join(args.pretrained_model_name_or_path, config_file_name)
        return os.path.isfile(config_file_name)
    else:
        files_in_repo = model_info(args.pretrained_model_name_or_path, revision=args.revision).siblings
        return any(file.rfilename == config_file_name for file in files_in_repo)


def set_seed(deterministic: bool, user_seed: int = None):
    if deterministic:
        torch.backends.cudnn.deterministic = True
        seed = 0 if user_seed is None else user_seed
        set_seed2(seed)
    else:
        torch.backends.cudnn.deterministic = False


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


def compute_snr(timesteps, noise_scheduler):
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


from dreambooth import shared


def save_lora(args, stop_text_percentage, accelerator, unet, text_encoder, pbar2, user_model_dir=None):
    pbar2.reset(1)
    pbar2.set_description("Saving Lora Weights...")

    # setup directory
    loras_dir = os.path.join(user_model_dir, "loras") if user_model_dir else shared.ui_lora_models_path

    os.makedirs(loras_dir, exist_ok=True)

    # setup pt path
    lora_model_name = args.model_name
    lora_file_prefix = f"{lora_model_name}_{args.revision}"

    tgt_module = get_target_module("module", True)

    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_tenc = accelerator.unwrap_model(text_encoder)

    modelmap = {"unet": (unwrapped_unet, tgt_module)}

    # save text_encoder
    if stop_text_percentage:
        modelmap["text_encoder"] = (unwrapped_tenc, TEXT_ENCODER_DEFAULT_TARGET_REPLACE)

    pbar2.reset(1)
    pbar2.set_description("Saving Extra Networks")

    out_safe = os.path.join(shared.ui_lora_models_path, f"{lora_file_prefix}.safetensors")
    save_extra_networks(modelmap, out_safe)

    pbar2.update(0)


def load_lora(args, stop_text_percentage, unet, text_encoder):
    if not os.path.exists(args.lora_model_name):
        lora_path = os.path.join(args.model_dir, "loras", args.lora_model_name)
    else:
        lora_path = args.lora_model_name
    lora_txt = lora_path.replace(".pt", "_txt.pt")

    # Handle invalid paths
    if not os.path.exists(lora_path) or not os.path.isfile(lora_path):
        lora_path, lora_txt = None, None

    injectable_lora = get_target_module("injection", args.use_lora_extended)
    target_module = get_target_module("module", args.use_lora_extended)
    unet_lora_params, _ = injectable_lora(
        unet,
        r=args.lora_unet_rank,
        loras=lora_path,
        target_replace_module=target_module,
    )
    text_encoder_lora_params = None
    if stop_text_percentage != 0:
        text_encoder.requires_grad_(False)
        inject_trainable_txt_lora = get_target_module("injection", False)
        text_encoder_lora_params, _ = inject_trainable_txt_lora(
            text_encoder,
            target_replace_module=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
            r=args.lora_txt_rank,
            loras=lora_txt,
        )
    return unet_lora_params, text_encoder_lora_params, text_encoder
