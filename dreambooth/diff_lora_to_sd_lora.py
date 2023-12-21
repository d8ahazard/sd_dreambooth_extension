import json
import os

import safetensors.torch
import torch


unet_conversion_map_layer_reverse = {}

for i in range(3):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}_{j}_"
        sd_down_res_prefix = f"input_blocks_{3 * i + j + 1}_0_"
        unet_conversion_map_layer_reverse[sd_down_res_prefix] = hf_down_res_prefix

        if i > 0:
            hf_down_atn_prefix = f"down_blocks.{i}_{j}_"
            sd_down_atn_prefix = f"input_blocks_{3 * i + j + 1}_1_"
            unet_conversion_map_layer_reverse[sd_down_atn_prefix] = hf_down_atn_prefix

    for j in range(4):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}_{j}_"
        sd_up_res_prefix = f"output_blocks_{3 * i + j}_0_"
        unet_conversion_map_layer_reverse[sd_up_res_prefix] = hf_up_res_prefix

        if i < 2:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}_{j}_"
            sd_up_atn_prefix = f"output_blocks_{3 * i + j}_1_"
            unet_conversion_map_layer_reverse[sd_up_atn_prefix] = hf_up_atn_prefix

# This is somewhat hacky, and I should probably feel bad...but I don't.
to_replace = {
    "text_encoder.text_model.encoder.": "lora_te1_text_model_encoder_",
    "text_encoder_2.text_model.encoder.": "lora_te2_text_model_encoder_",
    "to_q_": "to_q.",
    "to_k_": "to_k.",
    "to_v_": "to_v.",
    "to_out_": "to_out_0.",
    "lora.down": "lora_down",
    "lora.up": "lora_up",
    "self_attn.q": "self_attn_q",
    "self_attn.k": "self_attn_k",
    "self_attn.v": "self_attn_v",
    "self_attn.out": "self_attn_out",
    ".lora_linear_layer.down": ".lora_down",
    ".lora_linear_layer.up": ".lora_up",
    "mid_block_0": "middle_block_1",
    "unet.": "lora_unet_",
}

# This is excessive...but...
for i in range(20):
    to_replace[f".attentions.{i}.transformer"] = f"_{i}_transformer"

for i in range(100):
    to_replace[f"layers.{i}."] = f"layers_{i}_"

# As is this
for i in range(20):
    for j in range(1,3):
        to_replace[f"blocks.{i}.attn{j}.processor."] = f"blocks_{i}_attn{j}_"
for (kk, vv) in unet_conversion_map_layer_reverse.items():
    to_replace[vv] = kk

# Check for missing alpha keys, set to 0.8 if not found
base_attn_keys = [
    "attn1_to_out",
    "attn1_to_k",
    "attn1_to_q",
    "attn1_to_v",
    "attn2_to_out",
    "attn2_to_k",
    "attn2_to_q",
    "attn2_to_v",
    "self_attn_k_proj",
    "self_attn_q_proj",
    "self_attn_v_proj",
    "self_attn_out_proj"
]

secondary_keys = [
    "lora_down.weight",
    "lora_up.weight"
]


def convert_diffusers_to_kohya_lora(path, metadata, alpha=0.8):
    model_dict = safetensors.torch.load_file(path)
    new_model_dict = {}
    alpha_keys = []
    # Replace the things
    for (key, v) in model_dict.items():
        for (kc,vc) in to_replace.items():
            key = key.replace(kc, vc)
        # We really shouldn't have to do this, but I don't want to change other things and break it, so...
        key = key.replace("mid_block_0", "middle_block_1")
        akey = key

        # Check for missing alpha keys
        for k in base_attn_keys:
            if k in akey:
                for rep in secondary_keys:
                    akey = akey.replace(rep, "alpha")
                if akey not in alpha_keys:
                    alpha_keys.append(akey)
        new_model_dict[key] = v

    # Add missing alpha keys
    for k in alpha_keys:
        if k not in new_model_dict:
            new_model_dict[k] = torch.tensor(alpha)
    conv_path = path.replace(".safetensors", "_auto.safetensors")
    safetensors.torch.save_file(new_model_dict, conv_path, metadata=metadata)
    # Delete the file at path, move the new file to path
    os.remove(path)
    os.rename(conv_path, path)
