import gc
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import torch
import torch.utils.checkpoint
from PIL import features
from accelerate.logging import get_logger
from huggingface_hub import HfFolder, whoami
from six import StringIO

from extensions.sd_dreambooth_extension.dreambooth import conversion
from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig
from modules import paths, shared, devices, sd_models
from diffusers import StableDiffusionPipeline
from extensions.sd_dreambooth_extension.dreambooth.xattention import save_pretrained

try:
    cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
except:
    cmd_dreambooth_models_path = None

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

mem_record = {}

StableDiffusionPipeline.save_pretrained = save_pretrained


def performance_wizard():
    num_class_images = 0
    train_batch_size = 1
    sample_batch_size = 1
    not_cache_latents = True
    gradient_checkpointing = True
    use_ema = False
    train_text_encoder = False
    mixed_precision = 'fp16'
    use_cpu = False
    use_8bit_adam = True
    try:
        t = torch.cuda.get_device_properties(0).total_memory
        gb = math.ceil(t / 1073741824)
        print(f"Total VRAM: {gb}")
        if gb >= 24:
            train_batch_size = 2
            sample_batch_size = 4
            not_cache_latents = False
            train_text_encoder = True
            use_ema = True
            use_8bit_adam = False
            gradient_checkpointing = False
        if 24 > gb >= 12:
            train_text_encoder = True
            use_ema = False
            gradient_checkpointing = True
            not_cache_latents = True
        if gb < 12:
            use_cpu = True
            use_8bit_adam = False
            mixed_precision = 'no'
    except:
        pass

    return num_class_images, train_batch_size, sample_batch_size, not_cache_latents, gradient_checkpointing, use_ema, \
           train_text_encoder, mixed_precision, use_cpu, use_8bit_adam


def printm(msg, reset=False):
    global mem_record
    if reset:
        mem_record = {}
    allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
    cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
    mem_record[msg] = f"{allocated}/{cached}GB"
    print(f' {msg} \n Allocated: {allocated}GB \n Reserved: {cached}GB \n')


def dumb_safety(images, clip_input):
    return images, False


def isset(val: str):
    return val is not None and val != "" and val != "*"


def list_features():
    # Create buffer for pilinfo() to write into rather than stdout
    buffer = StringIO()
    features.pilinfo(out=buffer)
    global pil_features
    pil_features = []
    # Parse and analyse lines
    for line in buffer.getvalue().splitlines():
        if "Extensions:" in line:
            ext_list = line.split(": ")[1]
            extensions = ext_list.split(", ")
            for extension in extensions:
                if not extension in pil_features:
                    pil_features.append(extension)
    return pil_features


def is_image(path: Path, feats=None):
    if feats is None:
        feats = []
    if not len(feats):
        feats = list_features()
    is_img = path.is_file() and path.suffix.lower() in feats
    return is_img


def load_params(model_dir, *args):
    data = DreamboothConfig().from_file(model_dir)

    target_values = ["pretrained_vae_name_or_path",
                     "instance_data_dir",
                     "class_data_dir",
                     "instance_prompt",
                     "class_prompt",
                     "save_sample_prompt",
                     "save_sample_negative_prompt",
                     "n_save_sample",
                     "seed",
                     "save_guidance_scale",
                     "save_infer_steps",
                     "num_class_images",
                     "resolution",
                     "center_crop",
                     "train_text_encoder",
                     "train_batch_size",
                     "sample_batch_size",
                     "num_train_epochs",
                     "max_train_steps",
                     "gradient_accumulation_steps",
                     "gradient_checkpointing",
                     "learning_rate",
                     "scale_lr",
                     "lr_scheduler",
                     "lr_warmup_steps",
                     "use_8bit_adam",
                     "adam_beta1",
                     "adam_beta2",
                     "adam_weight_decay",
                     "adam_epsilon",
                     "max_grad_norm",
                     "save_preview_every",
                     "save_embedding_every",
                     "mixed_precision",
                     "not_cache_latents",
                     "concepts_list",
                     "use_cpu",
                     "pad_tokens",
                     "max_token_length",
                     "hflip",
                     "use_ema",
                     "class_negative_prompt",
                     "class_guidance_scale",
                     "class_infer_steps"
                     ]

    values = []
    for target in target_values:
        if target in data:
            values.append(data[target])
        else:
            values.append(None)
    values.append(f"Loaded params from {model_dir}.")
    return values


def get_db_models():
    model_dir = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
    out_dir = os.path.join(model_dir, "dreambooth")
    output = []
    if os.path.exists(out_dir):
        dirs = os.listdir(out_dir)
        for found in dirs:
            if os.path.isdir(os.path.join(out_dir, found)):
                output.append(found)
    return output


def start_training(model_dir,
                   pretrained_vae_name_or_path,
                   instance_data_dir,
                   class_data_dir,
                   instance_prompt,
                   class_prompt,
                   save_sample_prompt,
                   save_sample_negative_prompt,
                   n_save_sample,
                   seed,
                   save_guidance_scale,
                   save_infer_steps,
                   num_class_images,
                   resolution,
                   center_crop,
                   train_text_encoder,
                   train_batch_size,
                   sample_batch_size,
                   num_train_epochs,
                   max_train_steps,
                   gradient_accumulation_steps,
                   gradient_checkpointing,
                   learning_rate,
                   scale_lr,
                   lr_scheduler,
                   lr_warmup_steps,
                   use_8bit_adam,
                   adam_beta1,
                   adam_beta2,
                   adam_weight_decay,
                   adam_epsilon,
                   max_grad_norm,
                   save_preview_every,  # Replaces save_interval, save_min_steps
                   save_embedding_every,
                   mixed_precision,
                   not_cache_latents,
                   concepts_list,
                   use_cpu,
                   pad_tokens,
                   max_token_length,
                   hflip,
                   use_ema,
                   class_negative_prompt,
                   class_guidance_scale,
                   class_infer_steps
                   ):
    global mem_record
    if model_dir == "" or model_dir is None:
        print("Invalid model name.")
        return "Create or select a model first.", ""

    config = DreamboothConfig().from_ui(model_dir,
                                        pretrained_vae_name_or_path,
                                        instance_data_dir,
                                        class_data_dir,
                                        instance_prompt,
                                        class_prompt,
                                        save_sample_prompt,
                                        save_sample_negative_prompt,
                                        n_save_sample,
                                        seed,
                                        save_guidance_scale,
                                        save_infer_steps,
                                        num_class_images,
                                        resolution,
                                        center_crop,
                                        train_text_encoder,
                                        train_batch_size,
                                        sample_batch_size,
                                        num_train_epochs,
                                        max_train_steps,
                                        gradient_accumulation_steps,
                                        gradient_checkpointing,
                                        learning_rate,
                                        scale_lr,
                                        lr_scheduler,
                                        lr_warmup_steps,
                                        use_8bit_adam,
                                        adam_beta1,
                                        adam_beta2,
                                        adam_weight_decay,
                                        adam_epsilon,
                                        max_grad_norm,
                                        save_preview_every,  # Replaces save_interval, save_min_steps
                                        save_embedding_every,
                                        mixed_precision,
                                        not_cache_latents,
                                        concepts_list,
                                        use_cpu,
                                        pad_tokens,
                                        max_token_length,
                                        hflip,
                                        use_ema,
                                        class_negative_prompt,
                                        class_guidance_scale,
                                        class_infer_steps
                                        )

    # Parse/sanitize concepts list
    concepts_loaded = False

    if config.concepts_list is not None and config.concepts_list != "":
        is_json = False
        try:
            alist = str(config.concepts_list)
            if "'" in alist:
                alist = alist.replace("'", '"')
            print(f"Trying to parse: {alist}")
            config.concepts_list = json.loads(alist)
            is_json = True
            concepts_loaded = True
        except Exception as e:
            print(f"Unable to load concepts as JSON, trying as file: {e}")
            pass
        if not is_json:
            try:
                if os.path.exists(config.concepts_list):
                    with open(config.concepts_list, "r") as f:
                        config.concepts_list = json.load(f)
                    concepts_loaded = True
                print(f"Loaded concepts from {config.concepts_list}")
            except:
                print("Unable to load concepts from file either, this is bad.")
                pass
    first_class = None
    if not concepts_loaded:
        config.concepts_list = None
    else:
        first_concepts = config.concepts_list[0]
        first_class = first_concepts["class_prompt"]

    # Set class dir if not specified
    if config.class_data_dir is None or config.class_data_dir == "":
        config.class_data_dir = os.path.join(config.model_dir, "classifiers")

    # Disable prior preservation if no class prompt and no sample images
    if (config.class_prompt is None or config.class_prompt == "") and first_class is None:
        if config.num_class_images == 0:
            config.with_prior_preservation = False

    # Clear pretrained VAE Name if applicable
    if "pretrained_vae_name_or_path" in config.__dict__:
        if config.pretrained_vae_name_or_path == "":
            config.pretrained_vae_name_or_path = None
    else:
        config.pretrained_vae_name_or_path = None

    config.save()
    msg = None

    if not isset(instance_data_dir) and not isset(concepts_list):
        msg = "No instance data specified."
    if not isset(instance_prompt) and not isset(concepts_list):
        msg = "No instance prompt specified."
    if not os.path.exists(config.pretrained_model_name_or_path):
        msg = "Invalid training data directory."
    if isset(pretrained_vae_name_or_path) and not os.path.exists(pretrained_vae_name_or_path):
        msg = "Invalid Pretrained VAE Path."
    if resolution <= 0:
        msg = "Invalid resolution."
    if isset(concepts_list):
        concepts_loaded = False
        try:
            alist = str(config.concepts_list)
            if "'" in alist:
                alist = alist.replace("'", '"')
            print(f"Trying to parse: {alist}")
            concepts_list = json.loads(alist)
            concepts_loaded = True
        except:
            pass

        if not concepts_loaded:
            try:
                if os.path.exists(concepts_list):
                    with open(concepts_list, "r") as f:
                        concepts_list = json.load(f)
                    concepts_loaded = True
            except:
                print("Unable to load concepts from file either, this is bad.")
                pass
        if not concepts_loaded:
            msg = "Unable to parse concepts list."

    if msg:
        shared.state.textinfo = msg
        print(msg)
        return msg, ""

    # Clear memory and do "stuff" only after we've ensured all the things are right
    print("Starting Dreambooth training...")
    shared.sd_model.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()
    printm("VRAM cleared.", True)
    total_steps = config.revision
    shared.state.textinfo = "Initializing dreambooth training..."
    from dreambooth.train_dreambooth import main
    config, mem_record = main(config, mem_record)
    if config.revision != total_steps:
        config.save()
    total_steps = config.revision
    devices.torch_gc()
    gc.collect()
    printm("Training completed, reloading SD Model.")
    print(f'Memory output: {mem_record}')
    shared.sd_model.to(shared.device)
    print("Re-applying optimizations...")
    res = f"Training {'interrupted' if shared.state.interrupted else 'finished'}. " \
          f"Total lifetime steps: {total_steps} \n"
    print(f"Returning result: {res}")
    return res, ""


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def printm(msg, reset=False):
    global mem_record
    if reset:
        mem_record = {}
    allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
    cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
    mem_record[msg] = f"{allocated}/{cached}GB"
    print(f' {msg} \n Allocated: {allocated}GB \n Reserved: {cached}GB \n')


def dumb_safety(images, clip_input):
    return images, False


def save_checkpoint(model_name: str, vae_path: str, total_steps: int, use_half: bool = False):
    print(f"Successfully trained model for a total of {total_steps} steps, converting to ckpt.")
    ckpt_dir = shared.cmd_opts.ckpt_dir
    models_path = os.path.join(paths.models_path, "Stable-diffusion")
    if ckpt_dir is not None:
        models_path = ckpt_dir
    src_path = os.path.join(
        os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path, "dreambooth",
        model_name, "working")
    out_file = os.path.join(models_path, f"{model_name}_{total_steps}.ckpt")
    conversion.diff_to_sd(src_path, vae_path, out_file, use_half)
    sd_models.list_models()
