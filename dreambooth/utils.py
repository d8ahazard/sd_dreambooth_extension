from __future__ import annotations

import gc
import html
import os
import sys
import traceback
from tqdm.auto import tqdm
from io import StringIO
from pathlib import Path
from typing import Optional, Union, Tuple, List

import matplotlib
import pandas as pd
from pandas.plotting._matplotlib.style import get_standard_colors
from transformers import PretrainedConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
import torch
from PIL import features, Image
from huggingface_hub import HfFolder, whoami
from pandas import DataFrame
from tensorboard.compat.proto import event_pb2

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
            tqdm.write(msg, params)
        else:
            tqdm.write(msg)


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


def printm(msg=""):
    from extensions.sd_dreambooth_extension.dreambooth import db_shared
    if db_shared.debug:
        allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
        cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
        print(f"{msg}({allocated}/{cached})")


def cleanup(do_print: bool = False):
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except:
        pass
    if do_print:
        print("Cleanup completed.")


def unload_system_models():
    if shared.sd_model is not None:
        shared.sd_model.to("cpu")
    for former in shared.face_restorers:
        try:
            former.send_model_to("cpu")
        except:
            pass
    cleanup()


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
        # return ["default", "xformers", "sub_quad", "flash_attention"]
        return ["default", "xformers", "flash_attention"]
    else:
        # return ["default", "sub_quad", "flash_attention"]
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
    print("Restored system models.")


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


def plot_multi(
        data: pd.DataFrame,
        x: Union[str, None] = None,
        y: Union[List[str], None] = None,
        spacing: float = 0.1,
        **kwargs
) -> matplotlib.axes.Axes:
    """Plot multiple Y axes on the same chart with same x axis.

    Args:
        data: dataframe which contains x and y columns
        x: column to use as x axis. If None, use index.
        y: list of columns to use as Y axes. If None, all columns are used
            except x column.
        spacing: spacing between the plots
        **kwargs: keyword arguments to pass to data.plot()

    Returns:
        a matplotlib.axes.Axes object returned from data.plot()

    Example:

    See Also:
        This code is mentioned in https://stackoverflow.com/q/11640243/2593810
    """

    # Get default color style from pandas - can be changed to any other color list
    if y is None:
        y = data.columns

    # remove x_col from y_cols
    if x:
        y = [col for col in y if col != x]

    if len(y) == 0:
        return
    colors = get_standard_colors(num_colors=len(y))

    if "legend" not in kwargs:
        kwargs["legend"] = False  # prevent multiple legends

    # First axis
    ax = data.plot(x=x, y=y[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=y[0])
    lines, labels = ax.get_legend_handles_labels()

    for i in range(1, len(y)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines["right"].set_position(("axes", 1 + spacing * (i - 1)))
        data.plot(
            ax=ax_new, x=x, y=y[i], color=colors[i % len(colors)], **kwargs
        )
        ax_new.set_ylabel(ylabel=y[i])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)

    return ax


def parse_logs(model_name: str, for_ui: bool = False):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        model_name: (str) path to db model config/dir.
        for_ui: (bool) Generate UI-formatted text outputs.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """
    matplotlib.use("Agg")
    def convert_tfevent(filepath) -> Tuple[DataFrame, DataFrame, DataFrame, bool]:
        loss_events = []
        lr_events = []
        ram_events = []
        serialized_examples = tensorflow.data.TFRecordDataset(filepath)
        for serialized_example in serialized_examples:
            e = event_pb2.Event.FromString(serialized_example.numpy())
            if len(e.summary.value):
                parsed = parse_tfevent(e)
                if parsed["Name"] == "lr":
                    lr_events.append(parsed)
                elif parsed["Name"] == "loss":
                    loss_events.append(parsed)
                elif parsed["Name"] == "vram_usage":
                    ram_events.append(parsed)

        merged_events = []

        has_all = True
        for le in loss_events:
            lr = next((item for item in lr_events if item["Step"] == le["Step"]), None)
            if lr is not None:
                le["LR"] = lr["Value"]
                le["Loss"] = le["Value"]
                merged_events.append(le)
            else:
                has_all = False
        if has_all:
            loss_events = merged_events

        return pd.DataFrame(loss_events), pd.DataFrame(lr_events), pd.DataFrame(ram_events), has_all

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
    smoothing_window = int(model_config.graph_smoothing)
    root_dir = os.path.join(model_config.model_dir, "logging", "dreambooth")
    columns_order = ['Wall_time', 'Name', 'Step', 'Value']

    out_loss = []
    out_lr = []
    out_ram = []
    has_all_lr = True
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            converted_loss, converted_lr, converted_ram, merged = convert_tfevent(file_full_path)
            out_loss.append(converted_loss)
            out_lr.append(converted_lr)
            out_ram.append(converted_ram)
            if not merged:
                has_all_lr = False

    loss_columns = columns_order
    if has_all_lr:
        loss_columns = ['Wall_time', 'Name', 'Step', 'Loss', "LR"]
    # Concatenate (and sort) all partial individual dataframes
    all_df_loss = pd.concat(out_loss)[loss_columns]
    all_df_loss = all_df_loss.sort_values("Wall_time")
    all_df_loss = all_df_loss.reset_index(drop=True)
    all_df_loss = all_df_loss.rolling(smoothing_window).mean()
    out_images = []
    out_names = []
    if has_all_lr:
        plotted_loss = plot_multi(all_df_loss, "Step", ["Loss", "LR"],
                                  title=f"Loss Average/Learning Rate ({model_config.lr_scheduler})")
        loss_name = "Loss Average/Learning Rate"
    else:
        plotted_loss = all_df_loss.plot(x="Step", y="Value", title="Loss Averages")
        loss_name = "Loss Averages"
        all_df_lr = pd.concat(out_lr)[columns_order]
        all_df_lr = all_df_lr.sort_values("Wall_time")
        all_df_lr = all_df_lr.reset_index(drop=True)
        all_df_lr = all_df_lr.rolling(smoothing_window).mean()
        plotted_lr = all_df_lr.plot(x="Step", y="Value", title="Learning Rate")
        lr_img = os.path.join(model_config.model_dir, "logging", f"lr_plot_{model_config.revision}.png")
        plotted_lr.figure.savefig(lr_img)
        log_lr = Image.open(lr_img)
        out_images.append(log_lr)
        out_names.append("Learning Rate")

    loss_img = os.path.join(model_config.model_dir, "logging", f"loss_plot_{model_config.revision}.png")
    plotted_loss.figure.savefig(loss_img)

    log_pil = Image.open(loss_img)
    out_images.append(log_pil)
    out_names.append(loss_name)
    try:
        all_df_ram = pd.concat(out_ram)[columns_order]
        all_df_ram = all_df_ram.sort_values("Wall_time")
        all_df_ram = all_df_ram.reset_index(drop=True)
        all_df_ram = all_df_ram.rolling(smoothing_window).mean(numeric_only=True)
        plotted_ram = all_df_ram.plot(x="Step", y="Value", title="VRAM Usage")

        ram_img = os.path.join(model_config.model_dir, "logging", f"ram_plot_{model_config.revision}.png")
        plotted_ram.figure.savefig(ram_img)
        log_ram = Image.open(ram_img)
        out_images.append(log_ram)
        out_names.append("VRAM Usage")
        if for_ui:
            out_names = "<br>".join(out_names)
    except:
        pass

    return out_images, out_names


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
