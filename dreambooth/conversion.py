# From https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py
# AND https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_stable_diffusion.py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
# limitations under the License.
""" Conversion script for the LDM checkpoints. """
import gc
import os
import shutil
import traceback

import gradio as gr
import torch

import modules.sd_models
from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig, from_file
from extensions.sd_dreambooth_extension.dreambooth.dreambooth import get_db_models, printm, reload_system_models, \
    unload_system_models, cleanup, sanitize_name
from modules import paths, shared

try:
    cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
except:
    cmd_dreambooth_models_path = None

try:
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError(
        "OmegaConf is required to convert the LDM checkpoints. Please install it with `pip install OmegaConf`."
    )

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig

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
UNET_PARAMS_MODEL_CHANNELS = 320
UNET_PARAMS_CHANNEL_MULT = [1, 2, 4, 4]
UNET_PARAMS_ATTENTION_RESOLUTIONS = [4, 2, 1]
UNET_PARAMS_IMAGE_SIZE = 32  # unused
UNET_PARAMS_IN_CHANNELS = 4
UNET_PARAMS_OUT_CHANNELS = 4
UNET_PARAMS_NUM_RES_BLOCKS = 2
UNET_PARAMS_CONTEXT_DIM = 768
UNET_PARAMS_NUM_HEADS = 8

unet_params = {
    "model_channels": 320,
    "channel_mult": [1, 2, 4, 4],
    "attention_resolutions": [4, 2, 1],
    "image_size": 32,  # unused
    "in_channels": 4,
    "out_channels": 4,
    "num_res_blocks": 2,
    "context_dim": 768,
    "num_heads": 8
}

unet_v2_params = unet_params.copy()
unet_v2_params["num_heads"] = [5, 10, 20, 20]
unet_v2_params["context_dim"] = 1024

VAE_PARAMS_Z_CHANNELS = 4
VAE_PARAMS_RESOLUTION = 256
VAE_PARAMS_IN_CHANNELS = 3
VAE_PARAMS_OUT_CH = 3
VAE_PARAMS_CH = 128
VAE_PARAMS_CH_MULT = [1, 2, 4, 4]
VAE_PARAMS_NUM_RES_BLOCKS = 2

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


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item
        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "query.weight")
        new_item = new_item.replace("q.bias", "query.bias")

        new_item = new_item.replace("k.weight", "key.weight")
        new_item = new_item.replace("k.bias", "key.bias")

        new_item = new_item.replace("v.weight", "value.weight")
        new_item = new_item.replace("v.bias", "value.bias")

        new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
        new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
        paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def linear_transformer_to_conv(checkpoint):
    keys = list(checkpoint.keys())
    tf_keys = ["proj_in.weight", "proj_out.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in tf_keys:
            if checkpoint[key].ndim == 2:
                checkpoint[key] = checkpoint[key].unsqueeze(2).unsqueeze(2)


def create_unet_diffusers_config(unet_params):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """

    block_out_channels = [unet_params["model_channels"] * mult for mult in unet_params["channel_mult"]]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params["attention_resolutions"] else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params["attention_resolutions"] else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    config = dict(
        sample_size=unet_params["image_size"],
        in_channels=unet_params["in_channels"],
        out_channels=unet_params["out_channels"],
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=unet_params["num_res_blocks"],
        cross_attention_dim=unet_params["context_dim"],
        attention_head_dim=unet_params["num_heads"],
    )

    return config


def create_vae_diffusers_config():
    """
  Creates a config for the diffusers based on the config of the LDM model.
  """
    # vae_params = original_config.model.params.first_stage_config.params.ddconfig
    # _ = original_config.model.params.first_stage_config.params.embed_dim
    block_out_channels = [VAE_PARAMS_CH * mult for mult in VAE_PARAMS_CH_MULT]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = dict(
        sample_size=VAE_PARAMS_RESOLUTION,
        in_channels=VAE_PARAMS_IN_CHANNELS,
        out_channels=VAE_PARAMS_OUT_CH,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        latent_channels=VAE_PARAMS_Z_CHANNELS,
        layers_per_block=VAE_PARAMS_NUM_RES_BLOCKS,
    )
    return config


def create_scheduler(scheduler_type):
    num_train_timesteps = 1000
    beta_start = 0.00085
    beta_end = 0.0120
    if scheduler_type == "pndm":
        scheduler = PNDMScheduler(
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            skip_prk_steps=True,
        )
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end,
                                         beta_schedule="scaled_linear")
    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler(beta_start=beta_start, beta_end=beta_end,
                                           beta_schedule="scaled_linear")
    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear"
        )
    elif scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler(
            beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear"
        )
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
    else:
        print(f"Scheduler of type {scheduler_type} doesn't exist!")
        scheduler = None
    return scheduler


def convert_ldm_unet_checkpoint(v2, checkpoint, config, extract_ema=False):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    has_ema = False
    # extract state_dict for UNet
    unet_state_dict = {}
    keys = list(checkpoint.keys())
    unet_key = "model.diffusion_model."
    if sum(k.startswith("model_ema") for k in keys) > 100:
        print(f"Checkpoint has both EMA and non-EMA weights.")
        if extract_ema:
            has_ema = True
            print(
                "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
                " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
            )
            for key in keys:
                if key.startswith("model.diffusion_model"):
                    flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)
        else:
            print(
                "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
            )
    for key in keys:
        if key.startswith(unet_key):
            unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {"time_embedding.linear_1.weight": unet_state_dict["time_embed.0.weight"],
                      "time_embedding.linear_1.bias": unet_state_dict["time_embed.0.bias"],
                      "time_embedding.linear_2.weight": unet_state_dict["time_embed.2.weight"],
                      "time_embedding.linear_2.bias": unet_state_dict["time_embed.2.bias"],
                      "conv_in.weight": unet_state_dict["input_blocks.0.0.weight"],
                      "conv_in.bias": unet_state_dict["input_blocks.0.0.bias"],
                      "conv_norm_out.weight": unet_state_dict["out.0.weight"],
                      "conv_norm_out.bias": unet_state_dict["out.0.bias"],
                      "conv_out.weight": unet_state_dict["out.2.weight"],
                      "conv_out.bias": unet_state_dict["out.2.bias"]}

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            res_paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(
                res_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            if ["conv.weight", "conv.bias"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                res_paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(
                    res_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    if v2:
        linear_transformer_to_conv(new_checkpoint)

    return new_checkpoint, has_ema


def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    vae_state_dict = {}
    vae_key = "first_stage_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {"encoder.conv_in.weight": vae_state_dict["encoder.conv_in.weight"],
                      "encoder.conv_in.bias": vae_state_dict["encoder.conv_in.bias"],
                      "encoder.conv_out.weight": vae_state_dict["encoder.conv_out.weight"],
                      "encoder.conv_out.bias": vae_state_dict["encoder.conv_out.bias"],
                      "encoder.conv_norm_out.weight": vae_state_dict["encoder.norm_out.weight"],
                      "encoder.conv_norm_out.bias": vae_state_dict["encoder.norm_out.bias"],
                      "decoder.conv_in.weight": vae_state_dict["decoder.conv_in.weight"],
                      "decoder.conv_in.bias": vae_state_dict["decoder.conv_in.bias"],
                      "decoder.conv_out.weight": vae_state_dict["decoder.conv_out.weight"],
                      "decoder.conv_out.bias": vae_state_dict["decoder.conv_out.bias"],
                      "decoder.conv_norm_out.weight": vae_state_dict["decoder.norm_out.weight"],
                      "decoder.conv_norm_out.bias": vae_state_dict["decoder.norm_out.bias"],
                      "quant_conv.weight": vae_state_dict["quant_conv.weight"],
                      "quant_conv.bias": vae_state_dict["quant_conv.bias"],
                      "post_quant_conv.weight": vae_state_dict["post_quant_conv.weight"],
                      "post_quant_conv.bias": vae_state_dict["post_quant_conv.bias"]}

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


def convert_ldm_clip_checkpoint_v1(checkpoint):
    keys = list(checkpoint.keys())
    text_model_dict = {}
    for key in keys:
        if key.startswith("cond_stage_model.transformer"):
            text_model_dict[key[len("cond_stage_model.transformer."):]] = checkpoint[key]
    return text_model_dict


def convert_ldm_clip_checkpoint_v2(checkpoint, max_length):
    def convert_key(c_key):
        if not c_key.startswith("cond_stage_model"):
            return None

        # common conversion
        c_key = c_key.replace("cond_stage_model.model.transformer.", "text_model.encoder.")
        c_key = c_key.replace("cond_stage_model.model.", "text_model.")

        if "resblocks" in c_key:
            # resblocks conversion
            c_key = c_key.replace(".resblocks.", ".layers.")
            if ".ln_" in c_key:
                c_key = c_key.replace(".ln_", ".layer_norm")
            elif ".mlp." in c_key:
                c_key = c_key.replace(".c_fc.", ".fc1.")
                c_key = c_key.replace(".c_proj.", ".fc2.")
            elif '.attn.out_proj' in c_key:
                c_key = c_key.replace(".attn.out_proj.", ".self_attn.out_proj.")
            elif '.attn.in_proj' in c_key:
                c_key = None  # ???????????
            else:
                raise ValueError(f"unexpected key in SD: {c_key}")
        elif '.positional_embedding' in c_key:
            c_key = c_key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
        elif '.text_projection' in c_key:
            c_key = None  # ????????
        elif '.logit_scale' in c_key:
            c_key = None  # ????????
        elif '.token_embedding' in c_key:
            c_key = c_key.replace(".token_embedding.weight", ".embeddings.token_embedding.weight")
        elif '.ln_final' in c_key:
            c_key = c_key.replace(".ln_final", ".final_layer_norm")
        return c_key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        # remove resblocks 23
        if '.resblocks.23.' in key:
            continue
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attn???
    for key in keys:
        if '.resblocks.23.' in key:
            continue
        if '.resblocks' in key and '.attn.in_proj_' in key:
            # ?????
            values = torch.chunk(checkpoint[key], 3)

            key_suffix = ".weight" if "weight" in key else ".bias"
            key_pfx = key.replace("cond_stage_model.model.transformer.resblocks.", "text_model.encoder.layers.")
            key_pfx = key_pfx.replace("_weight", "")
            key_pfx = key_pfx.replace("_bias", "")
            key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
            new_sd[key_pfx + "q_proj" + key_suffix] = values[0]
            new_sd[key_pfx + "k_proj" + key_suffix] = values[1]
            new_sd[key_pfx + "v_proj" + key_suffix] = values[2]

    # position_ids???
    new_sd["text_model.embeddings.position_ids"] = torch.Tensor([list(range(max_length))]).to(torch.int64)
    return new_sd


def conv_transformer_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    tf_keys = ["proj_in.weight", "proj_out.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in tf_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]


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

# this part accounts for mid blocks in both the encoder and the decoder
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


def convert_unet_state_dict_to_sd(v2, unet_state_dict):
    conversion_map_layer = []
    for q in range(4):
        for r in range(2):
            hfd_res_prefix = f"down_blocks.{q}.resnets.{r}."
            sdd_res_prefix = f"input_blocks.{3 * q + r + 1}.0."
            conversion_map_layer.append((sdd_res_prefix, hfd_res_prefix))
            if q < 3:
                hfd_atn_prefix = f"down_blocks.{q}.attentions.{r}."
                sdd_atn_prefix = f"input_blocks.{3 * q + r + 1}.1."
                conversion_map_layer.append((sdd_atn_prefix, hfd_atn_prefix))
        for r in range(3):
            hfu_res_prefix = f"up_blocks.{q}.resnets.{r}."
            sdu_res_prefix = f"output_blocks.{3 * q + r}.0."
            conversion_map_layer.append((sdu_res_prefix, hfu_res_prefix))
            if q > 0:
                hfu_atn_prefix = f"up_blocks.{q}.attentions.{r}."
                sdu_atn_prefix = f"output_blocks.{3 * q + r}.1."
                conversion_map_layer.append((sdu_atn_prefix, hfu_atn_prefix))
        if q < 3:
            # no downsample in down_blocks.3
            hfd_prefix = f"down_blocks.{q}.downsamplers.0.conv."
            sdd_prefix = f"input_blocks.{3 * (q + 1)}.0.op."
            conversion_map_layer.append((sdd_prefix, hfd_prefix))

            # no upsample in up_blocks.3
            hfu_prefix = f"up_blocks.{q}.upsamplers.0."
            sdu_prefix = f"output_blocks.{3 * q + 2}.{1 if q == 0 else 2}."
            conversion_map_layer.append((sdu_prefix, hfu_prefix))

    hfm_atn_prefix = "mid_block.attentions.0."
    sdm_atn_prefix = "middle_block.1."
    conversion_map_layer.append((sdm_atn_prefix, hfm_atn_prefix))

    for r in range(2):
        hfm_res_prefix = f"mid_block.resnets.{r}."
        sdm_res_prefix = f"middle_block.{2 * r}."
        conversion_map_layer.append((sdm_res_prefix, hfm_res_prefix))

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
        for sd_part, hf_part in conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}

    if v2:
        conv_transformer_to_linear(new_state_dict)

    return new_state_dict


def load_checkpoint_with_text_encoder_conversion(ckpt_path):
    text_encoder_key_replacements = [
        ('cond_stage_model.transformer.embeddings.', 'cond_stage_model.transformer.text_model.embeddings.'),
        ('cond_stage_model.transformer.encoder.', 'cond_stage_model.transformer.text_model.encoder.'),
        ('cond_stage_model.transformer.final_layer_norm.', 'cond_stage_model.transformer.text_model.final_layer_norm.')
    ]

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    key_reps = []
    for rep_from, rep_to in text_encoder_key_replacements:
        for key in state_dict.keys():
            if key.startswith(rep_from):
                new_key = rep_to + key[len(rep_from):]
                key_reps.append((key, new_key))

    for key, new_key in key_reps:
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    return checkpoint


def printi(msg, params=None):
    shared.state.textinfo = msg
    if shared.state.job_count > shared.state.job_no:
        shared.state.job_no += 1
    if params:
        print(msg, params)
    else:
        print(msg)


def extract_checkpoint(new_model_name: str, ckpt_path: str, scheduler_type="ddim", new_model_url="", new_model_token="",
                       extract_ema=False, v2=False):
    printi("Extracting checkpoint...")
    shared.state.job_count = 8
    unload_system_models()
    map_location = shared.device
    # Set up our base directory for the model and sanitize our file name
    new_model_name = sanitize_name(new_model_name)
    if new_model_url == "":
        config = DreamboothConfig(new_model_name, scheduler=scheduler_type, src=ckpt_path, v2=v2)
    else:
        config = DreamboothConfig(new_model_name, scheduler_type=scheduler_type, src=new_model_url)
    config.save()
    revision = config.revision
    model_scheduler = scheduler_type
    status = ""
    has_ema = False
    src = config.src
    shared.state.job_no = 0
    if shared.cmd_opts.ckptfix or shared.cmd_opts.medvram or shared.cmd_opts.lowvram:
        printm(f"Using CPU for extraction.")
        map_location = torch.device('cpu')

    scheduler = create_scheduler(scheduler_type)
    if new_model_url and new_model_token:
        print(f"Trying to create {new_model_name} from hugginface.co/{new_model_url}")
        printi("Loading model from hub.")
        pipe = StableDiffusionPipeline.from_pretrained(new_model_url, use_auth_token=new_model_token,
                                                       scheduler=scheduler)
        printi("Model loaded.")
        shared.state.job_no = 7
    else:
        try:
            checkpoint_info = modules.sd_models.get_closet_checkpoint_match(ckpt_path)

            if checkpoint_info is None:
                print("Unable to find checkpoint file!")
                shared.state.job_no = 8
                return "", "", "", "", "", "", "", "Unable to find base checkpoint.", ""

            if not os.path.exists(checkpoint_info.filename):
                print("Unable to find checkpoint file!")
                shared.state.job_no = 8
                return "", "", "", "", "", "", "", "Unable to find base checkpoint.", ""

            ckpt_path = checkpoint_info[0]
            printi("Extracting checkpoint...")
            checkpoint = load_checkpoint_with_text_encoder_conversion(ckpt_path)
            if "global_step" in checkpoint:
                config.revision = checkpoint["global_step"]
                config.save()
            state_dict = checkpoint["state_dict"]
            unet_config = create_unet_diffusers_config(unet_v2_params if v2 else unet_params)
            converted_unet_checkpoint, has_ema = convert_ldm_unet_checkpoint(v2, state_dict, unet_config, extract_ema)

            unet = UNet2DConditionModel(**unet_config)
            info = unet.load_state_dict(converted_unet_checkpoint)
            printi("Loading Unet:", info)
            # Convert the VAE model.
            vae_config = create_vae_diffusers_config()
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)

            vae = AutoencoderKL(**vae_config)
            info = vae.load_state_dict(converted_vae_checkpoint)
            printi("Loading VAE:", info)
            shared.state.job_no = 3
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

            # convert text_model
            if v2:
                converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v2(state_dict, 77)
                cfg = CLIPTextConfig(
                    vocab_size=49408,
                    hidden_size=1024,
                    intermediate_size=4096,
                    num_hidden_layers=23,
                    num_attention_heads=16,
                    max_position_embeddings=77,
                    hidden_act="gelu",
                    layer_norm_eps=1e-05,
                    dropout=0.0,
                    attention_dropout=0.0,
                    initializer_range=0.02,
                    initializer_factor=1.0,
                    pad_token_id=1,
                    bos_token_id=0,
                    eos_token_id=2,
                    model_type="clip_text_model",
                    projection_dim=512,
                    torch_dtype="float32",
                    transformers_version="4.25.0.dev0",
                )
                text_model = CLIPTextModel(cfg)
                info = text_model.load_state_dict(converted_text_encoder_checkpoint)
            else:
                converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v1(state_dict)
                text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
                info = text_model.load_state_dict(converted_text_encoder_checkpoint)
            printi("Loading text encoder:", info)
            pipe = StableDiffusionPipeline(
                unet=unet,
                text_encoder=text_model,
                vae=vae,
                scheduler=scheduler,
                tokenizer=tokenizer,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=None,
            )
        except Exception as e:
            print(f"Exception with the conversion: {e}")
            traceback.print_exc()
            del pipe
            pipe = None

    if pipe is not None:
        pipe = pipe.to(map_location)
        printi("Saving diffusion weights.")
        pipe.save_pretrained(config.pretrained_model_name_or_path)
        del pipe
    try:
        del vae
        del text_model
        del tokenizer
        del unet
        del scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except:
        pass
    extraction_successful = True
    required_dirs = ["scheduler", "text_encoder", "tokenizer", "unet", "vae"]
    for r_dir in required_dirs:
        full_path = os.path.join(config.pretrained_model_name_or_path, r_dir)
        if not os.path.exists(full_path):
            print(f"Unable to find working dir, extraction likely failed: {r_dir}")
            extraction_successful = False

    if not extraction_successful:
        print("Extraction failed, removing model directory.")
        status = "Extraction failed."
        shutil.rmtree(config.model_dir, ignore_errors=True)
    else:
        status = "Checkpoint successfully extracted."

    reload_system_models()
    printm("Extraction completed.", True)
    printi("Extraction completed.")
    dirs = get_db_models()
    return gr.Dropdown.update(choices=sorted(dirs), value=new_model_name), config.model_dir, revision, model_scheduler, \
           src, "True" if has_ema else "False", v2, status


def convert_text_encoder_state_dict_to_sd_v2(checkpoint):
    def convert_key(conv_key):
        if ".position_ids" in conv_key:
            return None

        # common
        conv_key = conv_key.replace("text_model.encoder.", "transformer.")
        conv_key = conv_key.replace("text_model.", "")
        if "layers" in conv_key:
            # resblocks conversion
            conv_key = conv_key.replace(".layers.", ".resblocks.")
            if ".layer_norm" in conv_key:
                conv_key = conv_key.replace(".layer_norm", ".ln_")
            elif ".mlp." in conv_key:
                conv_key = conv_key.replace(".fc1.", ".c_fc.")
                conv_key = conv_key.replace(".fc2.", ".c_proj.")
            elif '.self_attn.out_proj' in conv_key:
                conv_key = conv_key.replace(".self_attn.out_proj.", ".attn.out_proj.")
            elif '.self_attn.' in conv_key:
                conv_key = None
            else:
                raise ValueError(f"unexpected key in DiffUsers model: {conv_key}")
        elif '.position_embedding' in conv_key:
            conv_key = conv_key.replace("embeddings.position_embedding.weight", "positional_embedding")
        elif '.token_embedding' in conv_key:
            conv_key = conv_key.replace("embeddings.token_embedding.weight", "token_embedding.weight")
        elif 'final_layer_norm' in conv_key:
            conv_key = conv_key.replace("final_layer_norm", "ln_final")
        return conv_key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    for key in keys:
        if 'layers' in key and 'q_proj' in key:
            key_q = key
            key_k = key.replace("q_proj", "k_proj")
            key_v = key.replace("q_proj", "v_proj")

            value_q = checkpoint[key_q]
            value_k = checkpoint[key_k]
            value_v = checkpoint[key_v]
            value = torch.cat([value_q, value_k, value_v])

            new_key = key.replace("text_model.encoder.layers.", "transformer.resblocks.")
            new_key = new_key.replace(".self_attn.q_proj.", ".attn.in_proj_")
            new_sd[new_key] = value

    return new_sd


def compile_checkpoint(model_name):
    unload_system_models()
    shared.state.textinfo = "Compiling checkpoint."
    shared.state.job_no = 0
    shared.state.job_count = 2
    print(f"Compiling checkpoint: {model_name}")
    if not model_name:
        return "Select a model to compile.", "No model selected."

    ckpt_dir = shared.cmd_opts.ckpt_dir
    vae_path = None
    models_path = os.path.join(paths.models_path, "Stable-diffusion")
    if ckpt_dir is not None:
        models_path = ckpt_dir

    config = from_file(model_name)
    half = config.half_model
    total_steps = config.revision
    if total_steps == 0:
        shared.state.textinfo = "You should probably train the model first...compiling..."

    shared.state.job_no = 1
    out_file = os.path.join(models_path, f"{model_name}_{total_steps}.ckpt")

    model_path = config.pretrained_model_name_or_path
    state_dict = {}

    def assign_new_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            state_dict[key] = v

    unet_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.bin")
    if vae_path == "" or vae_path is None:
        vae_path = os.path.join(model_path, "vae", "diffusion_pytorch_model.bin")
    else:
        vae_path = os.path.join(vae_path, "diffusion_pytorch_model.bin")
    text_enc_path = os.path.join(model_path, "text_encoder", "pytorch_model.bin")
    # Convert the UNet model
    unet_state_dict = torch.load(unet_path, map_location="cpu")
    vae_state_dict = torch.load(vae_path, map_location="cpu")
    text_encoder = torch.load(text_enc_path, map_location="cpu")
    v2 = config.v2
    vae_state_dict = convert_vae_state_dict(vae_state_dict)
    assign_new_sd("first_stage_model.", vae_state_dict)

    # Convert the UNet model
    unet_state_dict = convert_unet_state_dict_to_sd(v2, unet_state_dict)
    assign_new_sd("model.diffusion_model.", unet_state_dict)

    # Convert the text encoder model
    if v2:
        text_enc_dict = convert_text_encoder_state_dict_to_sd_v2(text_encoder)
        assign_new_sd("cond_stage_model.model.", text_enc_dict)
    else:
        text_enc_dict = text_encoder
        assign_new_sd("cond_stage_model.transformer.", text_enc_dict)

    if half:
        state_dict = {k: v.half() for k, v in state_dict.items()}
    new_ckpt = {'state_dict': state_dict, 'global_step': config.revision}

    torch.save(new_ckpt, out_file)
    if v2:
        cfg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", "v2-inference-v.yaml")
        cfg_dest = os.path.join(models_path, f"{model_name}_{total_steps}.yaml")
        shutil.copyfile(cfg_file, cfg_dest)
    try:
        del unet_state_dict
        del vae_state_dict
        del text_enc_path
        del state_dict
    except:
        pass
    cleanup()
    return "Checkpoint compiled successfully.", "Checkpoint compiled successfully."
