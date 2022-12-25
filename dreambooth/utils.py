from __future__ import annotations

import gc
import html
import os
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Optional, Union, Tuple

import pandas as pd
import torch
from PIL import features, Image
from huggingface_hub import HfFolder, whoami
from pandas import DataFrame
from tensorflow.python.summary.summary_iterator import summary_iterator

from extensions.sd_dreambooth_extension.dreambooth.db_shared import status
from modules import shared, paths, sd_models

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
        status.textinfo = msg
        if status.job_count > status.job_no:
            status.job_no += 1
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
                if "_txt.pt" not in found and ".pt" in found:
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


def wrap_gpu_call(func, extra_outputs=None):
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        try:
            status.begin()
            res = func(*args, **kwargs)
            status.end()

        except Exception as e:
            # When printing out our debug argument list, do not print out more than a MB of text
            max_debug_str_len = 131072  # (1024*1024)/8

            print("Error completing request", file=sys.stderr)
            arg_str = f"Arguments: {str(args)} {str(kwargs)}"
            print(arg_str[:max_debug_str_len], file=sys.stderr)
            if len(arg_str) > max_debug_str_len:
                print(f"(Argument list truncated at {max_debug_str_len}/{len(arg_str)} characters)", file=sys.stderr)

            print(traceback.format_exc(), file=sys.stderr)

            status.job = ""
            status.job_count = 0

            if extra_outputs_array is None:
                extra_outputs_array = [None, '']

            res = extra_outputs_array + [f"<div class='error'>{html.escape(type(e).__name__ + ': ' + str(e))}</div>"]

        status.skipped = False
        status.interrupted = False
        status.job_count = 0

        return res

    return f


def isset(val: Union[str | None]):
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


def get_checkpoint_match(search_string):
    for info in sd_models.checkpoints_list.values():
        if search_string in info.title or search_string in info.model_name or search_string in info.filename:
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


def parse_logs(model_name: str, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """
    def convert_tfevent(filepath) -> Tuple[DataFrame, DataFrame, DataFrame]:
        loss_events = []
        lr_events = []
        ram_events = []
        for e in summary_iterator(filepath):
            if len(e.summary.value):
                parsed = parse_tfevent(e)
                if parsed["Name"] == "lr":
                    lr_events.append(parsed)
                elif parsed["Name"] == "loss":
                    loss_events.append(parsed)
                elif parsed["Name"] == "vram_usage":
                    ram_events.append(parsed)
        return pd.DataFrame(loss_events), pd.DataFrame(lr_events), pd.DataFrame(ram_events)

    def parse_tfevent(tfevent):
        return {
            "Wall_time": tfevent.wall_time,
            "Name": tfevent.summary.value[0].tag,
            "Step": tfevent.step,
            "Value": float(tfevent.summary.value[0].simple_value),
        }

    from extensions.sd_dreambooth_extension.dreambooth.db_config import from_file
    model_config = from_file(model_name)
    if model_config is None:
        print("Unable to load model config!")
        return None
    root_dir = os.path.join(model_config.model_dir, "logging", "dreambooth")
    columns_order = ['Wall_time', 'Name', 'Step', 'Value']

    out_loss = []
    out_lr = []
    out_ram = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            converted_loss, converted_lr, converted_ram = convert_tfevent(file_full_path)
            out_loss.append(converted_loss)
            out_lr.append(converted_lr)
            out_ram.append(converted_ram)

    # Concatenate (and sort) all partial individual dataframes
    all_df_loss = pd.concat(out_loss)[columns_order]
    all_df_loss = all_df_loss.sort_values("Wall_time")
    all_df_loss = all_df_loss.reset_index(drop=True)

    all_df_lr = pd.concat(out_lr)[columns_order]
    all_df_lr = all_df_lr.sort_values("Wall_time")
    all_df_lr = all_df_lr.reset_index(drop=True)

    plotted_loss = all_df_loss.plot(x="Step", y="Value", title="Loss Averages")
    plotted_lr = all_df_lr.plot(x="Step", y="Value", title="Learning Rate")

    loss_img = os.path.join(model_config.model_dir, "logging", f"loss_plot_{model_config.revision}.png")
    lr_img = os.path.join(model_config.model_dir, "logging", f"lr_plot_{model_config.revision}.png")

    plotted_loss.figure.savefig(loss_img)
    plotted_lr.figure.savefig(lr_img)

    log_pil = Image.open(loss_img)
    log_lr = Image.open(lr_img)

    out_images = [log_pil, log_lr]

    try:
        all_df_ram = pd.concat(out_ram)[columns_order]
        all_df_ram = all_df_ram.sort_values("Wall_time")
        all_df_ram = all_df_ram.reset_index(drop=True)

        plotted_ram = all_df_ram.plot(x="Step", y="Value", title="VRAM Usage")

        ram_img = os.path.join(model_config.model_dir, "logging", f"ram_plot_{model_config.revision}.png")
        plotted_ram.figure.savefig(ram_img)
        log_ram = Image.open(ram_img)
        out_images.append(log_ram)
    except:
       pass
    return out_images
