from __future__ import annotations

import gc
import html
import importlib.util
import os
import sys
import traceback
from typing import Optional, Union, Tuple, List

import importlib_metadata
import matplotlib
import pandas as pd
from packaging import version
from pandas.plotting._matplotlib.style import get_standard_colors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from PIL import Image
from huggingface_hub import HfFolder, whoami
from pandas import DataFrame
from tensorboard.compat.proto import event_pb2

try:
    from extensions.sd_dreambooth_extension.helpers.mytqdm import mytqdm
    from extensions.sd_dreambooth_extension.dreambooth.shared import status
except:
    from dreambooth.helpers.mytqdm import mytqdm  # noqa
    from dreambooth.dreambooth.shared import status  # noqa


def printi(msg, params=None, log=True):
    if log:
        status.textinfo = msg
        if status.job_count > status.job_no:
            status.job_no += 1
        if params:
            mytqdm.write(msg, params)
        else:
            mytqdm.write(msg)


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


def printm(msg=""):
    from extensions.sd_dreambooth_extension.dreambooth import shared
    if shared.debug:
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
        print('cleanup exception')
    if do_print:
        print("Cleanup completed.")


def xformers_check():
    ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
    ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
    if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
        _torch_available = importlib.util.find_spec("torch") is not None

        if _torch_available:
            try:
                _torch_version = importlib_metadata.version("torch")
            except importlib_metadata.PackageNotFoundError:
                print("No metadatapackage")
                _torch_available = False
    else:
        _torch_available = False

    try:
        _xformers_version = importlib_metadata.version("xformers")
        if _torch_available:
            import torch
            if version.Version(torch.__version__) < version.Version("1.12"):
                raise ValueError("PyTorch should be >= 1.12")
        has_xformers = True
    except Exception as e:
        print(f"Exception importing xformers: {e}")
        has_xformers = False
    return has_xformers


def list_optimizer():
    try:
        from lion_pytorch import Lion
        return ["8Bit Adam", "Lion"]
    except:
        return ["8Bit Adam"]


def list_attention():
    has_xformers = xformers_check()
    import diffusers.utils
    diffusers.utils.is_xformers_available = xformers_check
    if has_xformers:
        return ["default", "xformers"]
    else:
        return ["default"]


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

        return res

    return f


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


from dataclasses import dataclass


@dataclass
class YAxis:
    name: str
    columns: List[str]


@dataclass
class PlotDefinition:
    title: str
    x_axis: str
    y_axis: List[YAxis]


def plot_multi_alt(
        data: pd.DataFrame,
        plot_definition: PlotDefinition,
        spacing: float = 0.1,
):
    styles = ["-", ":", "--", "-."]
    colors = get_standard_colors(num_colors=7)
    loss_color = colors[0]
    avg_colors = colors[1:]
    for i, yi in enumerate(plot_definition.y_axis):
        if len(yi.columns) > len(styles):
            raise ValueError(
                f"Maximum {len(styles)} traces per yaxis allowed. If we want to allow this we need to add some logic.")
        if i > len(colors):
            raise ValueError(
                f"Maximum {len(colors)} yaxis axis allowed. If we want to allow this we need to add some logic.")

        if i == 0:
            ax = data.plot(
                x=plot_definition.x_axis,
                y=yi.columns,
                title=plot_definition.title,
                color=[loss_color] * len(yi.columns)
            )
            ax.set_ylabel(ylabel=yi.name)

        else:
            # Multiple y-axes
            ax_new = ax.twinx()
            ax_new.spines["right"].set_position(("axes", 1 + spacing * (i - 1)))
            data.plot(
                ax=ax_new,
                x=plot_definition.x_axis,
                y=yi.columns,
                color=[avg_colors[yl] for yl in range(len(yi.columns))]
            )
            ax_new.set_ylabel(ylabel=yi.name)

    ax.legend(loc=0)

    return ax


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
    if for_ui:
        status.textinfo = "Generating graphs"

    def convert_tfevent(filepath) -> Tuple[DataFrame, DataFrame, DataFrame, bool]:
        loss_events = []
        lr_events = []
        ram_events = []
        instance_loss_events = []
        prior_loss_events = []
        has_all = False
        try:
            import tensorflow
        except:
            print("Unable to import tensorflow")
            return pd.DataFrame(loss_events), pd.DataFrame(lr_events), pd.DataFrame(ram_events), has_all

        serialized_examples = tensorflow.data.TFRecordDataset(filepath)

        for serialized_example in serialized_examples:
            e = event_pb2.Event.FromString(serialized_example.numpy())
            if len(e.summary.value):
                parsed = parse_tfevent(e)
                if parsed["Name"] == "lr":
                    lr_events.append(parsed)
                elif parsed["Name"] == "loss":
                    loss_events.append(parsed)
                elif parsed["Name"] == "vram_usage" or parsed["Name"] == "vram":
                    ram_events.append(parsed)
                elif parsed["Name"] == "instance_loss" or parsed["Name"] == "inst_loss":
                    instance_loss_events.append(parsed)
                elif parsed["Name"] == "prior_loss":
                    prior_loss_events.append(parsed)

        merged_events = []

        has_all = True
        for le in loss_events:
            lr = next((item for item in lr_events if item["Step"] == le["Step"]), None)
            instance_loss = next((item for item in instance_loss_events if item["Step"] == le["Step"]), None)
            prior_loss = next((item for item in prior_loss_events if item["Step"] == le["Step"]), None)
            if lr is not None and instance_loss is not None and prior_loss is not None:
                le["LR"] = lr["Value"]
                le["Loss"] = le["Value"]
                le["Instance_Loss"] = instance_loss["Value"]
                le["Prior_Loss"] = prior_loss["Value"]
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

    from extensions.sd_dreambooth_extension.dreambooth.dataclasses.db_config import from_file
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
        loss_columns = ['Wall_time', 'Name', 'Step', 'Loss', "LR", "Instance_Loss", "Prior_Loss"]
    # Concatenate (and sort) all partial individual dataframes
    all_df_loss = pd.concat(out_loss)[loss_columns]
    all_df_loss = all_df_loss.fillna(
        method="ffill")  # since we do not use the standard dreambooth algorithm it's possible to have NaN for instance or prior loss -> forward fill
    all_df_loss = all_df_loss.sort_values("Wall_time")
    all_df_loss = all_df_loss.reset_index(drop=True)
    sw = int(smoothing_window if smoothing_window < len(all_df_loss) / 3 else len(all_df_loss) / 3)
    all_df_loss = all_df_loss.rolling(sw).mean(numeric_only=True)

    out_images = []
    out_names = []
    status.job_count = 2
    status.job_no = 1
    status.textinfo = "Plotting data..."
    if has_all_lr:
        plotted_loss = plot_multi_alt(
            all_df_loss,
            plot_definition=PlotDefinition(
                title=f"Loss Average/Learning Rate ({model_config.lr_scheduler})",
                x_axis="Step",
                y_axis=[
                    YAxis(name="LR", columns=["LR"]),
                    YAxis(name="Loss", columns=["Instance_Loss", "Prior_Loss", "Loss"]),

                ]
            )
        )
        loss_name = "Loss Average/Learning Rate"
    else:
        plotted_loss = all_df_loss.plot(x="Step", y="Value", title="Loss Averages")
        loss_name = "Loss Averages"
        all_df_lr = pd.concat(out_lr)[columns_order]
        all_df_lr = all_df_lr.sort_values("Wall_time")
        all_df_lr = all_df_lr.reset_index(drop=True)
        all_df_lr = all_df_lr.rolling(smoothing_window).mean(numeric_only=True)
        plotted_lr = all_df_lr.plot(x="Step", y="Value", title="Learning Rate")
        lr_img = os.path.join(model_config.model_dir, "logging", f"lr_plot_{model_config.revision}.png")
        plotted_lr.figure.savefig(lr_img)
        log_lr = Image.open(lr_img)
        out_images.append(log_lr)
        out_names.append("Learning Rate")

    status.job_no = 2
    status.textinfo = "Saving graph data..."
    loss_img = os.path.join(model_config.model_dir, "logging", f"loss_plot_{model_config.revision}.png")
    printm(f"Saving {loss_img}")
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
        printm(f"Saving {ram_img}")
        plotted_ram.figure.savefig(ram_img)
        out_images.append(ram_img)
        out_names.append("VRAM Usage")
        if for_ui:
            out_names = "<br>".join(out_names)
    except:
        pass

    del out_loss
    del out_lr
    del out_ram
    try:
        matplotlib.pyplot.close()
    except:
        pass
    printm("Cleanup log parse.")
    return out_images, out_names
