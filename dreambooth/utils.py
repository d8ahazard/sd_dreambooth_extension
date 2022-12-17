import gc
import os
import random
import traceback
from io import StringIO
from pathlib import Path
from typing import Optional

import torch
from PIL import features
from diffusers import StableDiffusionPipeline
from huggingface_hub import HfFolder, whoami
from transformers import CLIPTextModel

from extensions.sd_dreambooth_extension.dreambooth import dream_state
from extensions.sd_dreambooth_extension.dreambooth.db_config import from_file
from modules import shared, paths, sd_models
from modules.shared import opts

try:
    cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
except:
    cmd_dreambooth_models_path = None

try:
    cmd_lora_models_path = shared.cmd_opts.lora_models_path
except:
    cmd_lora_models_path = None


def printi(msg, params=None, log=True):
    if log:
        dream_state.status.textinfo = msg
        if dream_state.status.job_count > dream_state.status.job_no:
            dream_state.status.job_no += 1
        if params:
            print(msg, params)
        else:
            print(msg)


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


def get_lora_models():
    model_dir = os.path.dirname(cmd_lora_models_path) if cmd_lora_models_path else paths.models_path
    out_dir = os.path.join(model_dir, "lora")
    output = [""]
    if os.path.exists(out_dir):
        dirs = os.listdir(out_dir)
        for found in dirs:
            if os.path.isfile(os.path.join(out_dir, found)):
                if not "_txt.pt" in found and ".pt" in found:
                    output.append(found)
    return output


def get_images(image_path):
    pil_features = list_features()
    output = []
    if isinstance(image_path, str):
        image_path = Path(image_path)
    if image_path.exists():
        for file in image_path.iterdir():
            if is_image(file, pil_features):
                output.append(file)
            if file.is_dir():
                sub_images = get_images(file)
                for image in sub_images:
                    output.append(image)
    return output


def sanitize_tags(name):
    tags = name.split(",")
    name = ""
    for tag in tags:
        tag = tag.replace(" ", "_").strip()
        name = "".join(x for x in tag if (x.isalnum() or x in "._-"))
    name = name.replace(" ", "_")
    return "".join(x for x in name if (x.isalnum() or x in "._-,"))


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in "._-"))


mem_record = {}


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


def log_memory():
    mem = printm("", True)
    return f"Current memory usage: {mem}"


def cleanup(do_print: bool = False):
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except:
        pass
    if do_print:
        printm("Cleanup completed.")


def unload_system_models():
    if shared.sd_model is not None:
        shared.sd_model.to("cpu")
    for former in shared.face_restorers:
        try:
            former.send_model_to("cpu")
        except:
            pass
    cleanup()
    printm("", True)


def list_attention():
    has_xformers = False
    try:
        import xformers
        import xformers.ops
        has_xformers = True
    except:
        pass
    pass

    if has_xformers:
        return ["default", "xformers", "flash_attention"]
    else:
        return ["default", "flash_attention"]


def list_floats():
    has_bf16 = False
    try:
        has_bf16 = torch.cuda.is_bf16_supported()
    except:
        pass
    if has_bf16:
        return ["no", "fp16", "bf16"]
    else:
        return ["no", "fp16"]


def reload_system_models():
    if shared.sd_model is not None:
        shared.sd_model.to(shared.device)
    printm("Restored system models.")


def generate_sample_img(model_dir: str, save_sample_prompt: str, negative_prompt: str, seed: int, num_samples: int):
    dream_state.status.job_count = (num_samples * 60) + 2
    print("Generate samples definitely clicked.")
    if model_dir is None or model_dir == "":
        return "Please select a model."
    config = from_file(model_dir)
    unload_system_models()
    model_path = config.pretrained_model_name_or_path
    if not os.path.exists(config.pretrained_model_name_or_path):
        print(f"Model path '{config.pretrained_model_name_or_path}' doesn't exist.")
        return None, f"Can't find diffusers model at {config.pretrained_model_name_or_path}."
    try:
        print(f"Loading model from {model_path}.")
        dream_state.status.job_no = 1
        dream_state.status.textinfo = "Loading diffusion model..."
        dream_state.status.job_count = num_samples
        dream_state.status.current_image_sampling_step = 0
        text_enc_model = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path,
                                                       subfolder="text_encoder", revision=config.revision)
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            text_encoder=text_enc_model,
            torch_dtype=torch.float16,
            revision=config.revision,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        pipeline = pipeline.to(shared.device)

        def update_latent(step: int, timestep: int, latents: torch.FloatTensor):
            dream_state.status.job_no += opts.show_progress_every_n_steps
            decoded = pipeline.decode_latents(latents)
            dream_state.status.current_latent = pipeline.numpy_to_pil(decoded)
            print(f"latent callback fired: {step} {timestep}")

        db_model_path = config.model_dir
        if save_sample_prompt is None:
            msg = "Please provide a sample prompt."
            print(msg)
            return None, msg
        dream_state.status.textinfo = f"Generating preview image for model {db_model_path}..."
        # I feel like this might not actually be necessary...but what the heck.
        if seed is None or seed == '' or seed == -1:
            seed = int(random.randrange(21474836147))
        g_cuda = torch.Generator(device=shared.device).manual_seed(seed)
        dream_state.status.job_count = 1
        with torch.autocast("cuda"), torch.inference_mode():
            images = pipeline([save_sample_prompt] * num_samples,
                              negative_prompt=[negative_prompt] * num_samples,
                              num_inference_steps=60,
                              guidance_scale=7.5,
                              width=config.resolution,
                              height=config.resolution,
                              callback_steps=opts.show_progress_every_n_steps,
                              callback=update_latent,
                              generator=g_cuda).images


    except:
        print("Exception generating sample!")
        traceback.print_exc()
    reload_system_models()
    return images, "Sample generated."


def isset(val: str):
    return val is not None and val != "" and val != "*"


def list_features():
    # Create buffer for pilinfo() to write into rather than stdout
    buffer = StringIO()
    features.pilinfo(out=buffer)
    pil_features = []
    # Parse and analyse lines
    for line in buffer.getvalue().splitlines():
        if "Extensions:" in line:
            ext_list = line.split(": ")[1]
            extensions = ext_list.split(", ")
            for extension in extensions:
                if extension not in pil_features:
                    pil_features.append(extension)
    return pil_features


def is_image(path: Path, feats=None):
    if feats is None:
        feats = []
    if not len(feats):
        feats = list_features()
    is_img = path.is_file() and path.suffix.lower() in feats
    return is_img


def get_checkpoint_match(searchString):
    for info in sd_models.checkpoints_list.values():
        if searchString in info.title or searchString in info.model_name or searchString in info.filename:
            return info
    return None


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"
