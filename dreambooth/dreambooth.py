import gc
import json
import logging
import math
import os
import random
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.utils.checkpoint
from PIL import features
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, whoami
from six import StringIO

from extensions.sd_dreambooth_extension.dreambooth import conversion
from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig
from extensions.sd_dreambooth_extension.dreambooth.xattention import save_pretrained
from modules import paths, shared, devices, sd_models, generation_parameters_copypaste
from modules.images import sanitize_filename_part
from modules.processing import create_infotext

try:
    cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
except:
    cmd_dreambooth_models_path = None

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

mem_record = {}

StableDiffusionPipeline.save_pretrained = save_pretrained


def log_memory():
    mem = printm("", True)
    return f"Current memory usage: {mem}"


def generate_sample_img(model_dir: str, save_sample_prompt: str, save_sample_negative_prompt: str, sample_seed: int,
                        save_guidance_scale: float, save_infer_steps: int, save_sample_count: int):
    print("Gensample?")
    unload_system_models()
    models_path = shared.models_path
    db_model_path = os.path.join(models_path, "dreambooth")
    if shared.cmd_opts.dreambooth_models_path:
        db_model_path = shared.cmd_opts.dreambooth_models_path
    model_path = os.path.join(db_model_path, model_dir, "working")
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' doesn't exist.")
        return f"Can't find diffusers model at {model_path}."
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None)
        pipeline = pipeline.to(shared.device)
        with devices.autocast(), torch.inference_mode():
            save_dir = os.path.join(shared.sd_path, "outputs", "dreambooth")
            if save_sample_prompt is None:
                msg = "Please provide a sample prompt."
                print(msg)
                return msg
            shared.state.textinfo = f"Generating preview image for model {db_model_path}..."
            seed = sample_seed
            # I feel like this might not actually be necessary...but what the heck.
            if seed is None or seed == '' or seed == -1:
                seed = int(random.randrange(21474836147))
            g_cuda = torch.Generator(device=shared.device).manual_seed(seed)
            sample_dir = os.path.join(save_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            file_count = 0
            for x in Path(sample_dir).iterdir():
                if is_image(x, pil_features):
                    file_count += 1
            shared.state.job_count = save_sample_count
            for n in range(save_sample_count):
                file_count += 1
                shared.state.job_no = n
                image = pipeline(save_sample_prompt, num_inference_steps=save_infer_steps,
                                 guidance_scale=save_guidance_scale,
                                 scheduler=EulerAncestralDiscreteScheduler(beta_start=0.00085,
                                                                           beta_end=0.012),
                                 negative_prompt=save_sample_negative_prompt,
                                 generator=g_cuda).images[0]

                if shared.opts.enable_pnginfo:
                    params = {
                        "Steps": save_infer_steps,
                        "Sampler": "Euler A",
                        "CFG scale": save_guidance_scale,
                        "Seed": sample_seed
                    }
                    generation_params_text = ", ".join(
                        [k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in
                         params.items() if v is not None])

                    negative_prompt_text = "\nNegative prompt: " + save_sample_negative_prompt

                    data = f"{save_sample_prompt}{negative_prompt_text}\n{generation_params_text}".strip()
                    image.info["parameters"] = data

                shared.state.current_image = image
                shared.state.textinfo = save_sample_prompt
                sanitized_prompt = sanitize_filename_part(save_sample_prompt, replace_spaces=False)
                image.save(os.path.join(sample_dir, f"{str(file_count).zfill(3)}-{sanitized_prompt}.png"))
    except:
        print("Exception generating sample!")
        traceback.print_exc()
    reload_system_models()
    return "Sample generated."


def cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except:
        pass
    printm("Cleanup completed.")


def unload_system_models():
    if shared.sd_model is not None:
        shared.sd_model.to("cpu")
    for former in shared.face_restorers:
        try:
            former.to("cpu")
        except:
            pass
    cleanup()
    printm("", True)


def reload_system_models():
    if shared.sd_model is not None:
        shared.sd_model.to(shared.device)
    printm("Restored system models.")


# Borrowed from https://wandb.ai/psuraj/dreambooth/reports/Training-Stable-Diffusion-with-Dreambooth
# --VmlldzoyNzk0NDc3#tl,dr; and https://www.reddit.com/r/StableDiffusion/comments/ybxv7h/good_dreambooth_formula/
def training_wizard_person(
        model_dir,
        use_concepts,
        concepts_list,
        instance_data_dir,
        class_data_dir,
        learning_rate
):
    return training_wizard(
        model_dir,
        use_concepts,
        concepts_list,
        instance_data_dir,
        class_data_dir,
        learning_rate,
        is_person=True)


def training_wizard(
        model_dir,
        use_concepts,
        concepts_list,
        instance_data_dir,
        class_data_dir,
        learning_rate,
        is_person=False
):
    # Load config, get total steps
    config = DreamboothConfig().from_file(model_dir)
    total_steps = config.revision
    config.use_concepts = use_concepts
    config.concepts_list = concepts_list
    config.instance_data_dir = instance_data_dir
    config.class_data_dir = class_data_dir
    config.instance_prompt = "foo"
    config.class_prompt = "foo"
    config.save_sample_prompt = ""
    config.instance_token = ""
    config.class_token = ""
    # Build concepts list using current settings
    concepts = config.concepts_list
    pil_feats = list_features()

    if concepts is None:
        print("Error loading params.")
        return "Unable to load concepts.", 1000, 100, False, 0, "constant"

    # Count the total number of images in all datasets
    total_images = 0
    for concept in concepts:
        if not os.path.exists(concept["instance_data_dir"]):
            print("Nonexistent instance directory.")
        else:
            for x in Path(concept["instance_data_dir"]).iterdir():
                if is_image(x, pil_feats):
                    total_images += 1

    if total_images == 0:
        print("No training images found, can't do math.")
        return "No training images found, can't do math.", 1000, 100, False, 0, "constant"

    # Set "base" value
    magick_number = 58139534.88372093
    required_steps = round(total_images * magick_number * learning_rate, -2)
    if is_person:
        num_class_images = round(total_images * 12, -1)
    else:
        num_class_images = 0
        required_steps = round(required_steps * 1.5, -2)

    # Ensure we don't over-train?
    if total_steps >= required_steps:
        required_steps = 0
    else:
        required_steps = required_steps - total_steps
    msg = f"Wizard completed, using {required_steps} lifetime steps and {num_class_images} class images."
    return msg, required_steps, num_class_images


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
    gb = 0
    try:
        t = torch.cuda.get_device_properties(0).total_memory
        gb = math.ceil(t / 1073741824)
        print(f"Total VRAM: {gb}")
        if gb >= 24:
            train_batch_size = 2
            sample_batch_size = 4
            train_text_encoder = True
            use_ema = True
            use_8bit_adam = False
            gradient_checkpointing = False
        if 24 > gb >= 10:
            train_text_encoder = True
            use_ema = False
            gradient_checkpointing = True
            not_cache_latents = True
        if gb < 10:
            use_cpu = True
            use_8bit_adam = False
            mixed_precision = 'no'

    except:
        pass
    msg = f"Calculated training params based on {gb}GB of VRAM detected."

    has_xformers = False
    try:
        if (shared.cmd_opts.xformers or shared.cmd_opts.force_enable_xformers) and is_xformers_available():
            import xformers
            import xformers.ops
            has_xformers = shared.cmd_opts.xformers or shared.cmd_opts.force_enable_xformers
    except:
        pass
    if has_xformers:
        use_8bit_adam = True
        mixed_precision = "fp16"
        msg += "<br>Xformers detected, enabling 8Bit Adam and setting mixed precision to 'fp16'"
        print()

    if use_cpu:
        msg += "<br>Detected less than 10GB of VRAM, setting CPU training to true."
    return msg, num_class_images, train_batch_size, sample_batch_size, not_cache_latents, gradient_checkpointing, \
           use_ema, train_text_encoder, mixed_precision, use_cpu, use_8bit_adam


def printm(msg="", reset=False):
    global mem_record
    try:
        allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
        reserved = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
        if not mem_record:
            mem_record = {}
        if reset:
            max_allocated = round(torch.cuda.max_memory_allocated(0) / 1024 ** 3, 1)
            max_reserved = round(torch.cuda.max_memory_reserved(0) / 1024 ** 3, 1)
            output = f" Allocated {allocated}/{max_allocated}GB \n Reserved: {reserved}/{max_reserved}GB \n"
            torch.cuda.reset_peak_memory_stats()
            print(output)
            mem_record = {}
        else:
            mem_record[msg] = f"{allocated}/{reserved}GB"
            output = f' {msg} \n Allocated: {allocated}GB \n Reserved: {reserved}GB \n'
            print(output)
    except:
        output = "Error parsing memory stats. Do you have a NVIDIA GPU?"
    return output


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


def load_params(model_dir):
    data = DreamboothConfig().from_file(model_dir)

    target_values = ["half_model",
                     "use_concepts",
                     "pretrained_vae_name_or_path",
                     "instance_data_dir",
                     "class_data_dir",
                     "instance_prompt",
                     "class_prompt",
                     "file_prompt_contents",
                     "instance_token",
                     "class_token",
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
                     "attention",
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
                     "class_infer_steps",
                     "shuffle_after_epoch"
                     ]

    values = []
    for target in target_values:
        if target in data.__dict__:
            value = data.__dict__[target]
            if target == "max_token_length":
                value = str(value)
            values.append(value)
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
                   half_model,
                   use_concepts,
                   pretrained_vae_name_or_path,
                   instance_data_dir,
                   class_data_dir,
                   instance_prompt,
                   class_prompt,
                   file_prompt_contents,
                   instance_token,
                   class_token,
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
                   attention,
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
                   class_infer_steps,
                   shuffle_after_epoch
                   ):
    global mem_record
    if model_dir == "" or model_dir is None:
        print("Invalid model name.")
        return "Create or select a model first.", ""
    config = DreamboothConfig(model_dir)
    config.from_ui(model_dir,
                   half_model,
                   use_concepts,
                   pretrained_vae_name_or_path,
                   instance_data_dir,
                   class_data_dir,
                   instance_prompt,
                   class_prompt,
                   file_prompt_contents,
                   instance_token,
                   class_token,
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
                   attention,
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
                   class_infer_steps,
                   shuffle_after_epoch
                   )

    concepts, msg = build_concepts(config)

    if concepts is not None:
        config.concepts_list = concepts
    else:
        print(f"Unable to build concepts: {msg}")
        return config, "Unable to load concepts."

    # Disable prior preservation if no class prompt and no sample images
    if config.num_class_images == 0:
        config.with_prior_preservation = False

    # Ensure we have a max token length set
    if config.max_token_length is None or config.max_token_length == 0:
        config.max_token_length = 75

    # Clear pretrained VAE Name if applicable
    if config.pretrained_vae_name_or_path == "":
        config.pretrained_vae_name_or_path = None

    config.save()
    msg = None
    if attention == "xformers":
        if mixed_precision == "no":
            msg = "Using xformers, please set mixed precision to 'fp16' to continue."
        if not shared.cmd_opts.xformers and not shared.cmd_opts.force_enable_xformers:
            msg = "Xformers is not enabled, please relaunch using the --xformers command-line argument to continue."
    if use_cpu:
        if use_8bit_adam or mixed_precision != "no":
            msg = "CPU Training detected, please disable 8Bit Adam and set mixed precision to 'no' to continue."
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

    if msg:
        shared.state.textinfo = msg
        print(msg)
        return msg, msg

    # Clear memory and do "stuff" only after we've ensured all the things are right
    print("Starting Dreambooth training...")
    unload_system_models()
    total_steps = config.revision
    shared.state.textinfo = "Initializing dreambooth training..."
    from extensions.sd_dreambooth_extension.dreambooth.train_dreambooth import main
    config, mem_record = main(config, mem_record)
    if config.revision != total_steps:
        config.save()
    total_steps = config.revision
    devices.torch_gc()
    gc.collect()
    printm("Training completed, reloading SD Model.")
    print(f'Memory output: {mem_record}')
    reload_system_models()
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
