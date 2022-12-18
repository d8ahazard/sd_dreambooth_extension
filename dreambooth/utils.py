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
