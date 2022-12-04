# Script for converting a HF Diffusers saved pipeline to a Stable Diffusion checkpoint.
# *Only* converts the UNet, VAE, and Text Encoder.
# Does not convert optimizer state or any other thing.

import os
import os.path as osp
import shutil

import torch

from extensions.sd_dreambooth_extension.dreambooth.db_config import from_file
from extensions.sd_dreambooth_extension.dreambooth.utils import cleanup, printi, unload_system_models, \
    reload_system_models
from modules import shared

# =================#
# UNet Conversion #
# =================#

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
unet_v2_params["attention_head_dim"] = [5, 10, 20, 20]
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


def conv_transformer_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    tf_keys = ["proj_in.weight", "proj_out.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in tf_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]


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
                print(f"Reshaping {k} for SD format")
                new_state_dict[k] = reshape_weight_for_sd(v)
    return new_state_dict


# =========================#
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


# Text Encoder Conversion #
# =========================#
# pretty much a no-op


def convert_text_enc_state_dict(text_enc_dict):
    return text_enc_dict


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


def compile_checkpoint(model_name: str, half: bool, use_subdir: bool = False, reload_models=False):
    """

    @param model_name: The model name to compile
    @param half: Use FP16 when compiling the model
    @param use_subdir: The model will be saved to a subdirectory of the checkpoints folder
    @return: status: What happened, path: Checkpoint path
    """
    unload_system_models()
    shared.state.textinfo = "Compiling checkpoint."
    shared.state.job_no = 0
    shared.state.job_count = 6
    printi(f"Compiling checkpoint for {model_name}...")
    if not model_name:
        return "Select a model to compile.", "No model selected."

    ckpt_dir = shared.cmd_opts.ckpt_dir
    models_path = os.path.join(shared.models_path, "Stable-diffusion")
    if ckpt_dir is not None:
        models_path = ckpt_dir

    config = from_file(model_name)
    try:
        if "use_subdir" in config.__dict__:
            use_subdir = config["use_subdir"]
    except:
        print("Yeah, we can't use dict to find config values.")

    v2 = config.v2
    total_steps = config.revision

    if use_subdir:
        os.makedirs(os.path.join(models_path, model_name))
        out_file = os.path.join(models_path, model_name, f"{model_name}_{total_steps}.ckpt")
    else:
        out_file = os.path.join(models_path, f"{model_name}_{total_steps}.ckpt")

    model_path = config.pretrained_model_name_or_path
    unet_path = osp.join(model_path, "unet", "diffusion_pytorch_model.bin")
    vae_path = osp.join(model_path, "vae", "diffusion_pytorch_model.bin")
    text_enc_path = osp.join(model_path, "text_encoder", "pytorch_model.bin")
    printi("Converting unet...")
    # Convert the UNet model
    unet_state_dict = torch.load(unet_path, map_location="cpu")
    unet_state_dict = convert_unet_state_dict(unet_state_dict)
    #unet_state_dict = convert_unet_state_dict_to_sd(v2, unet_state_dict)
    unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}
    printi("Converting vae...")
    # Convert the VAE model
    vae_state_dict = torch.load(vae_path, map_location="cpu")
    vae_state_dict = convert_vae_state_dict(vae_state_dict)
    vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}
    printi("Converting text encoder...")
    # Convert the text encoder model
    text_enc_dict = torch.load(text_enc_path, map_location="cpu")
    text_enc_dict = convert_text_enc_state_dict(text_enc_dict)
    #text_enc_dict = convert_text_enc_state_dict(text_enc_dict) if not v2 else convert_text_encoder_state_dict_to_sd_v2(text_enc_dict)
    text_enc_dict = {"cond_stage_model.transformer." + k: v for k, v in text_enc_dict.items()}
    printi("Compiling new state dict...")
    # Put together new checkpoint
    state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict}
    if half:
        state_dict = {k: v.half() for k, v in state_dict.items()}

    state_dict = {"state_dict": state_dict}
    new_ckpt = {'global_step': config.revision, 'state_dict': state_dict}
    printi(f"Saving checkpoint to {out_file}...")
    torch.save(new_ckpt, out_file)
    if v2:
        cfg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "configs", "v2-inference-v.yaml")
        cfg_dest = out_file.replace(".ckpt", ".yaml")
        print(f"Copying config file to {cfg_dest}")
        shutil.copyfile(cfg_file, cfg_dest)
    try:
        del unet_state_dict
        del vae_state_dict
        del text_enc_path
        del state_dict
    except:
        pass
    cleanup()
    if reload_models:
        reload_system_models()
    return "Checkpoint compiled successfully.", "Checkpoint compiled successfully."
