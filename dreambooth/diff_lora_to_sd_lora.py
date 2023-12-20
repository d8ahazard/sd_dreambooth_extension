import os

import safetensors.torch
import torch

# This map reverses the transformations done in _convert_kohya_lora_to_diffusers
unet_conversion_map_layer_reverse = {}

# Handling UNet structure and layer naming
for i in range(3):
    # Loop over downblocks/upblocks
    for j in range(2):
        # Loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.{j}."
        sd_down_res_prefix = f"lora_unet_down_blocks_{i}_{j}_"
        unet_conversion_map_layer_reverse[hf_down_res_prefix] = sd_down_res_prefix

        # Similar handling for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.{j}."
        sd_up_res_prefix = f"lora_unet_up_blocks_{i}_{j}_"
        unet_conversion_map_layer_reverse[hf_up_res_prefix] = sd_up_res_prefix

# Handling text encoder layers
for prefix in ["lora_te_", "lora_te1_", "lora_te2_"]:
    for i in range(20):
        hf_te_prefix = f"text_encoder.{i}."
        sd_te_prefix = prefix + f"text_model_encoder_{i}_"
        unet_conversion_map_layer_reverse[hf_te_prefix] = sd_te_prefix

# General replacement rules
to_replace = {
    "down_blocks": "lora_unet_down_blocks",
    "up_blocks": "lora_unet_up_blocks",
    "text_encoder": "lora_te_text_model_encoder",
    "text_encoder_2": "lora_te2_text_model_encoder",
    "to_q_lora": "to_q.lora",
    "to_k_lora": "to_k.lora",
    "to_v_lora": "to_v.lora",
    "to_out_lora": "to_out_0.lora",
    "lora_down.weight": "lora.down.weight",
    "lora_up.weight": "lora.up.weight",
    "self_attn_q": "self_attn.q",
    "self_attn_k": "self_attn.k",
    "self_attn_v": "self_attn.v",
    "self_attn_out": "self_attn.out",
    "lora_linear_layer_down": "lora.down",
    "lora_linear_layer_up": "lora.up",
    "mid_block": "middle_block",
    "lora_unet_": "unet.",
}

# Adjust the layer name reversals according to the Kohya to Diffusers conversion
for (hf, kohya) in unet_conversion_map_layer_reverse.items():
    to_replace[hf] = kohya


# Function to convert diffusers to kohya lora
def convert_diffusers_to_kohya_lora(path, metadata, alpha=0.8):
    model_dict = safetensors.torch.load_file(path)
    new_model_dict = {}
    alpha_keys = []

    # Replace the things
    for (key, v) in model_dict.items():
        for (kc, vc) in to_replace.items():
            key = key.replace(kc, vc)
        new_model_dict[key] = v

    # Add missing alpha keys
    for k in alpha_keys:
        if k not in new_model_dict:
            new_model_dict[k] = torch.tensor(alpha)

    # Save the converted model
    conv_path = path.replace(".safetensors", "_auto.safetensors")
    safetensors.torch.save_file(new_model_dict, conv_path, metadata=metadata)

    # Replace the original file with the converted one
    os.remove(path)
    os.rename(conv_path, path)
