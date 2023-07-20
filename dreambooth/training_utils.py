import gc
import logging
import math
import os

import accelerate
import torch
from accelerate.state import AcceleratorState
from accelerate.utils.random import set_seed as set_seed2
from diffusers import AutoencoderKL, DEISMultistepScheduler, UniPCMultistepScheduler
from diffusers.loaders import AttnProcsLayers
from huggingface_hub import model_info

from dreambooth.deis_velocity import get_velocity
from dreambooth.oft_utils.attention_processor import OFTAttnProcessor
from dreambooth.utils.model_utils import disable_safe_unpickle, enable_safe_unpickle
from lora_diffusion.extra_networks import save_extra_networks
from lora_diffusion.lora import get_target_module, TEXT_ENCODER_DEFAULT_TARGET_REPLACE

logger = logging.getLogger(__name__)


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt):
    prompt_embeds_list = []

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


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


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_timestep_embedding(x, outdim):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


def get_size_embeddings(orig_size, crop_size, target_size, device):
    emb1 = get_timestep_embedding(orig_size, 256)
    emb2 = get_timestep_embedding(crop_size, 256)
    emb3 = get_timestep_embedding(target_size, 256)
    vector = torch.cat([emb1, emb2, emb3], dim=1).to(device)
    return vector


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
    return prior


def stop_profiler(profiler):
    if profiler is not None:
        try:
            logger.debug("Stopping profiler.")
            profiler.stop()
        except:
            pass


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

    injectable_lora = get_target_module("injection", True)
    target_module = get_target_module("module", True)
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


def create_vae(args, device, weight_dtype):
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
    new_vae.to(device, dtype=weight_dtype)
    return new_vae


def apply_oft(unet, accelerator, eps, rank, coft):
    # now we will add new COT weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct oft layers
    oft_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            logger.error(f"Unknown attention processor name: {name}")
            continue

        oft_attn_procs[name] = OFTAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                eps=eps, rank=rank, is_coft=coft)

    unet.set_attn_processor(oft_attn_procs)
    oft_layers = AttnProcsLayers(unet.attn_processors)

    accelerator.register_for_checkpointing(oft_layers)
    return unet, accelerator, oft_layers
