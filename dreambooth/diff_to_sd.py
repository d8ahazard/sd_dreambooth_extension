# Script for converting Diffusers saved pipeline to a Stable Diffusion checkpoint.
# *Only* converts the UNet, VAE, and Text Encoder.
# Does not convert optimizer state or any other thing.

import os
import os.path as osp
import re
import shutil
import traceback
from typing import Dict

import safetensors.torch
from torch import Tensor
from tqdm.auto import tqdm
import torch
from diffusers import DiffusionPipeline

from extensions.sd_dreambooth_extension.dreambooth import db_shared as shared
from extensions.sd_dreambooth_extension.dreambooth.db_config import from_file
from extensions.sd_dreambooth_extension.dreambooth.db_shared import status
from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import mytqdm
from extensions.sd_dreambooth_extension.dreambooth.utils import printi, unload_system_models, \
    reload_system_models
from extensions.sd_dreambooth_extension.lora_diffusion.lora import weight_apply_lora

unet_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
]

unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(3):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

        if i > 0:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3 * i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2 * j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))


def convert_unet_state_dict(unet_state_dict):
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]

for i in range(4):
    # down_blocks have two resnets
    for j in range(2):
        hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
        sd_down_prefix = f"encoder.down.{i}.block.{j}."
        vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
        sd_downsample_prefix = f"down.{i}.downsample."
        vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"up.{3 - i}.upsample."
        vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

    # up_blocks have three resnets
    # also, up blocks in hf are numbered in reverse from sd
    for j in range(3):
        hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
        sd_up_prefix = f"decoder.up.{3 - i}.block.{j}."
        vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

# this part accounts for mid-blocks in both the encoder and the decoder
for i in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{i}."
    sd_mid_res_prefix = f"mid.block_{i + 1}."
    vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))

vae_conversion_map_attn = [
    # (stable-diffusion, HF Diffusers)
    ("norm.", "group_norm."),
    ("q.", "query."),
    ("k.", "key."),
    ("v.", "value."),
    ("proj_out.", "proj_attn."),
]


def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    return w.reshape(*w.shape, 1, 1)


def convert_vae_state_dict(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                new_state_dict[k] = reshape_weight_for_sd(v)
    return new_state_dict


# =========================#
# Text Encoder Conversion #
# =========================#


textenc_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ('resblocks.', 'text_model.encoder.layers.'),
    ('ln_1', 'layer_norm1'),
    ('ln_2', 'layer_norm2'),
    ('.c_fc.', '.fc1.'),
    ('.c_proj.', '.fc2.'),
    ('.attn', '.self_attn'),
    ('ln_final.', 'transformer.text_model.final_layer_norm.'),
    ('token_embedding.weight', 'transformer.text_model.embeddings.token_embedding.weight'),
    ('positional_embedding', 'transformer.text_model.embeddings.position_embedding.weight')
]
protected = {re.escape(x[1]): x[0] for x in textenc_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))

# Ordering is from https://github.com/pytorch/pytorch/blob/master/test/cpp/api/modules.cpp
code2idx = {'q': 0, 'k': 1, 'v': 2}

def conv_fp16(t: Tensor):
    return t.half()


def conv_bf16(t: Tensor):
    return t.bfloat16()


def conv_full(t):
    return t


_g_precision_func = {
    "no": conv_full,
    "fp16": conv_fp16,
    "bf16": conv_bf16,
}

def check_weight_type(k: str) -> str:
    if k.startswith("model.diffusion_model"):
        return "unet"
    elif k.startswith("first_stage_model"):
        return "vae"
    elif k.startswith("cond_stage_model"):
        return "clip"
    return "other"

def split_dict(state_dict):
    ok = {}
    json_dict = {}

    def _hf(wk: str, t: Tensor):
        if isinstance(t, Tensor) or str(type(t)) == "tensor":
            ok[wk] = t
        else:
            if isinstance(t, int) or isinstance(t, float):
                json_dict[wk] = str(t)
            if isinstance(t, str):
                json_dict[wk] = t
            if isinstance(t, Dict):
                moar_ok, moar_json = split_dict(t)
                ok.update(moar_ok)
                json_dict.update(moar_json)

    print("Converting model...")


    for k, v in mytqdm(state_dict.items()):
        _hf(k, v)

    return ok, json_dict

def convert_text_enc_state_dict_v20(text_enc_dict: Dict[str, torch.Tensor]):
    new_state_dict = {}
    capture_qkv_weight = {}
    capture_qkv_bias = {}
    for k, v in text_enc_dict.items():
        if k.endswith('.self_attn.q_proj.weight') or k.endswith('.self_attn.k_proj.weight') or k.endswith(
                '.self_attn.v_proj.weight'):
            k_pre = k[:-len('.q_proj.weight')]
            k_code = k[-len('q_proj.weight')]
            if k_pre not in capture_qkv_weight:
                capture_qkv_weight[k_pre] = [None, None, None]
            capture_qkv_weight[k_pre][code2idx[k_code]] = v
            continue

        if k.endswith('.self_attn.q_proj.bias') or k.endswith('.self_attn.k_proj.bias') or k.endswith(
                '.self_attn.v_proj.bias'):
            k_pre = k[:-len('.q_proj.bias')]
            k_code = k[-len('q_proj.bias')]
            if k_pre not in capture_qkv_bias:
                capture_qkv_bias[k_pre] = [None, None, None]
            capture_qkv_bias[k_pre][code2idx[k_code]] = v
            continue

        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k)

        new_state_dict[relabelled_key] = v

    for k_pre, tensors in capture_qkv_weight.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_weight"] = torch.cat(tensors)

    for k_pre, tensors in capture_qkv_bias.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_bias"] = torch.cat(tensors)

    return new_state_dict


def convert_text_enc_state_dict(text_enc_dict: Dict[str, torch.Tensor]):
    return text_enc_dict


def compile_checkpoint(model_name: str, lora_path: str=None, reload_models: bool = True, log:bool =True, snap_rev: str=""):
    """

    @param model_name: The model name to compile
    @param reload_models: Whether to reload the system list of checkpoints.
    @param lora_path: The path to a lora pt file to merge with the unet. Auto set during training.
    @param log: Whether to print messages to console/UI.
    @param snap_rev: The revision of snapshot to load from
    @return: status: What happened, path: Checkpoint path
    """
    unload_system_models()
    status.textinfo = "Compiling checkpoint."
    status.job_no = 0
    status.job_count = 7

    config = from_file(model_name)
    if lora_path is None and config.lora_model_name:
        lora_path = config.lora_model_name
    save_model_name = model_name if config.custom_model_name == "" else config.custom_model_name
    if config.custom_model_name == "":
        printi(f"Compiling checkpoint for {model_name}...", log=log)
    else:
        printi(f"Compiling checkpoint for {model_name} with a custom name {config.custom_model_name}", log=log)

    if not model_name:
        msg = "Select a model to compile."
        print(msg)
        return msg

    ckpt_dir = shared.ckpt_dir
    models_path = os.path.join(shared.models_path, "Stable-diffusion")
    if ckpt_dir is not None:
        models_path = ckpt_dir

    save_safetensors = config.save_safetensors
    lora_diffusers = ""
    v2 = config.v2
    total_steps = config.revision
    if config.use_subdir:
        os.makedirs(os.path.join(models_path, save_model_name), exist_ok=True)
        models_path = os.path.join(models_path, save_model_name)

    checkpoint_path = os.path.join(models_path, f"{save_model_name}_{total_steps}.ckpt")

    model_path = config.pretrained_model_name_or_path
    unet_path = None
    text_enc_path = None
    if snap_rev != "":
        new_hotness = os.path.join(config.model_dir, "checkpoints", f"checkpoint-{snap_rev}")
        if os.path.exists(new_hotness):
            tqdm.write(f"Loading snapshot paths from {new_hotness}")
            u_path = osp.join(new_hotness, "pytorch_model.bin")
            text_path = osp.join(new_hotness, "pytorch_model_1.bin")
            if os.path.exists(u_path):
                unet_path = u_path
            if os.path.exists(text_path):
                text_enc_path = text_path

    if unet_path is None:
        unet_path = osp.join(model_path, "unet", "diffusion_pytorch_model.bin")

    if text_enc_path is None:
        text_enc_path = osp.join(model_path, "text_encoder", "pytorch_model.bin")

    vae_path = osp.join(model_path, "vae", "diffusion_pytorch_model.bin")

    try:
        printi("Converting unet...", log=log)

        if lora_path is not None and lora_path != "":
            loaded_pipeline = DiffusionPipeline.from_pretrained(model_path).to("cpu")
            lora_diffusers = config.pretrained_model_name_or_path + "_lora"
            os.makedirs(lora_diffusers, exist_ok=True)
            if not os.path.exists(lora_path):
                try:
                    cmd_lora_models_path = shared.lora_models_path
                except:
                    cmd_lora_models_path = None
                model_dir = os.path.dirname(cmd_lora_models_path) if cmd_lora_models_path else shared.models_path
                lora_path = os.path.join(model_dir, "lora", lora_path)
            printi(f"Loading lora from {lora_path}", log=log)
            if os.path.exists(lora_path):
                checkpoint_path = checkpoint_path.replace(".ckpt", "_lora.ckpt")
                printi(f"Applying lora weight of alpha: {config.lora_weight} to unet...", log=log)
                weight_apply_lora(loaded_pipeline.unet, torch.load(lora_path), alpha=config.lora_weight)
                printi("Saving lora unet...", log=log)
                loaded_pipeline.unet.save_pretrained(os.path.join(config.pretrained_model_name_or_path, "unet_lora"))
                unet_path = osp.join(config.pretrained_model_name_or_path, "unet_lora", "diffusion_pytorch_model.bin")
            lora_txt = lora_path.replace(".pt", "_txt.pt")
            if os.path.exists(lora_txt):
                printi(f"Applying lora weight of alpha: {config.lora_txt_weight} to text encoder...", log=log)
                weight_apply_lora(loaded_pipeline.text_encoder, torch.load(lora_txt), target_replace_module=["CLIPAttention"], alpha=config.lora_weight)
                printi("Saving lora text encoder...", log=log)
                loaded_pipeline.text_encoder.save_pretrained(
                    os.path.join(config.pretrained_model_name_or_path, "text_encoder_lora"))
                text_enc_path = osp.join(config.pretrained_model_name_or_path, "text_encoder_lora", "pytorch_model.bin")
            del loaded_pipeline

        # Convert the UNet model
        unet_state_dict = torch.load(unet_path, map_location="cpu")
        unet_state_dict = convert_unet_state_dict(unet_state_dict)
        # unet_state_dict = convert_unet_state_dict_to_sd(v2, unet_state_dict)
        unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}
        printi("Converting vae...", log=log)
        # Convert the VAE model
        vae_state_dict = torch.load(vae_path, map_location="cpu")
        vae_state_dict = convert_vae_state_dict(vae_state_dict)
        vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}
        printi("Converting text encoder...", log=log)
        # Convert the text encoder model
        text_enc_dict = torch.load(text_enc_path, map_location="cpu")

        if v2:
            printi("Converting text enc dict for V2 model.", log=log)
            # Need to add the tag 'transformer' in advance, so we can knock it out from the final layer-norm
            text_enc_dict = {"transformer." + k: v for k, v in text_enc_dict.items()}
            text_enc_dict = convert_text_enc_state_dict_v20(text_enc_dict)
            text_enc_dict = {"cond_stage_model.model." + k: v for k, v in text_enc_dict.items()}
            
        else:
            printi("Converting text enc dict for V1 model.", log=log)
            text_enc_dict = convert_text_enc_state_dict(text_enc_dict)
            text_enc_dict = {"cond_stage_model.transformer." + k: v for k, v in text_enc_dict.items()}
            

        # Put together new checkpoint
        state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict}
        if config.half_model:
            state_dict = {k: v.half() for k, v in state_dict.items()}

        state_dict = {"db_global_step": config.revision, "db_epoch": config.epoch, "state_dict": state_dict}
        printi(f"Saving checkpoint to {checkpoint_path}...", log=log)
        if save_safetensors:
            print("Safe tensors already saved?")
            checkpoint_path = checkpoint_path.replace(".ckpt", ".safetensors")
            safe_dict, json_dict = split_dict(state_dict)
            safetensors.torch.save_file(safe_dict, checkpoint_path, json_dict)
        else:
            torch.save(state_dict, checkpoint_path)
        cfg_file = None
        new_name = os.path.join(config.model_dir, f"{config.model_name}.yaml")
        if os.path.exists(new_name):
            cfg_file = new_name

        if v2 and cfg_file is None:
            cfg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "configs", "v2-inference-v.yaml")

        if cfg_file is not None:
            cfg_dest = checkpoint_path.replace(".ckpt", ".yaml")
            cfg_dest = checkpoint_path.replace(".safetensors", ".yaml")
            printi(f"Copying config file from {cfg_dest} to {cfg_dest}", log=log)
            shutil.copyfile(cfg_file, cfg_dest)

    except Exception as e:
        msg = f"Exception compiling checkpoint: {e}"
        print(msg)
        traceback.print_exc()
        return msg

    try:
        del unet_state_dict
        del vae_state_dict
        del text_enc_path
        del state_dict
        if os.path.exists(lora_diffusers):
            shutil.rmtree(lora_diffusers, True)
    except:
        pass
    # cleanup()
    if reload_models:
        reload_system_models()
    msg = f"Checkpoint compiled successfully: {checkpoint_path}"
    printi(msg, log=log)
    return msg
