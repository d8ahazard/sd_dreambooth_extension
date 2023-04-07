from __future__ import annotations

import gc
import html
import importlib.util
import os
import sys
import traceback
from typing import Optional

import importlib_metadata
from packaging import version

from dreambooth import shared

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
from huggingface_hub import HfFolder, whoami

from helpers.mytqdm import mytqdm
from dreambooth.shared import status


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
    from dreambooth import shared

    if shared.debug:
        allocated = round(torch.cuda.memory_allocated(0) / 1024**3, 1)
        cached = round(torch.cuda.memory_reserved(0) / 1024**3, 1)
        print(f"{msg}({allocated}/{cached})")


def cleanup(do_print: bool = False):
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except:
        print("cleanup exception")
    if do_print:
        print("Cleanup completed.")


def xformers_check():
    env_vars_true_values = {"1", "ON", "YES", "TRUE"}
    env_vars_true_and_auto_values = env_vars_true_values.union({"AUTO"})

    use_tf = os.environ.get("USE_TF", "AUTO").upper()
    use_torch = os.environ.get("USE_TORCH", "AUTO").upper()
    if (
        use_torch in env_vars_true_and_auto_values
        and use_tf not in env_vars_true_values
    ):
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
                raise ValueError("PyTorch version must be >= 1.12")
            if version.Version(_xformers_version) < version.Version("0.0.17.dev"):
                raise ValueError("Xformers version must be >= 0.0.17.dev")
        has_xformers = True
    except Exception as e:
        # print(f"Exception importing xformers: {e}")
        has_xformers = False

    return has_xformers


def list_optimizer():
    optimizer_list = ["Torch AdamW"]

    try:
        if shared.device.type != "mps":
            from bitsandbytes.optim import AdamW8bit
            optimizer_list.append("8bit AdamW")
    except:
        pass

    try:
        from lion_pytorch import Lion
        optimizer_list.append("Lion")
    except:
        pass

    # try:
    #     from dadaptation import DAdaptSGD
    #     optimizer_list.append("SGD Dadaptation")
    # except:
    #     pass
    #
    # try:
    #     from dadaptation import DAdaptAdaGrad
    #     optimizer_list.append("AdaGrad Dadaptation")
    # except:
    #     pass
    #
    # try:
    #     from dadaptation import DAdaptAdam
    #     optimizer_list.append("AdamW Dadaptation")
    # except:
    #     pass
    #
    # try:
    #     from dreambooth.dadapt_adan import DAdaptAdan
    #     optimizer_list.append("Adan Dadaptation")
    # except:
    #     pass

    return optimizer_list


def list_attention():
    has_xformers = xformers_check()
    import diffusers.utils

    diffusers.utils.is_xformers_available = xformers_check
    if has_xformers:
        return ["default", "xformers"]
    else:
        return ["default"]


def list_precisions():
    precisions = ["no", "fp16"]
    try:
        if torch.cuda.is_bf16_supported():
            precisions.append("bf16")
    except:
        pass

    return precisions


def list_schedulers():
    return [
        "linear",
        "linear_with_warmup",
        "cosine",
        "cosine_annealing",
        "cosine_annealing_with_restarts",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ]


def wrap_gpu_call(func, extra_outputs=None):
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        try:
            status.begin()
            res = func(*args, **kwargs)
            status.end()

        except Exception as e:
            status.end()
            # When printing out our debug argument list, do not print out more than a MB of text
            max_debug_str_len = 131072  # (1024*1024)/8

            print("Error completing request", file=sys.stderr)
            arg_str = f"Arguments: {str(args)} {str(kwargs)}"
            print(arg_str[:max_debug_str_len], file=sys.stderr)
            if len(arg_str) > max_debug_str_len:
                print(
                    f"(Argument list truncated at {max_debug_str_len}/{len(arg_str)} characters)",
                    file=sys.stderr,
                )

            print(traceback.format_exc(), file=sys.stderr)

            status.job = ""
            status.job_count = 0

            if extra_outputs_array is None:
                extra_outputs_array = [None, ""]

            res = extra_outputs_array + [
                f"<div class='error'>{html.escape(type(e).__name__ + ': ' + str(e))}</div>"
            ]

        return res

    return f


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def verify_locon_installed(args):
    is_locon_installed = os.path.exists(
        os.path.join(
            shared.script_path,
            "extensions",
            "a1111-sd-webui-locon",
        )
    )
    if args.save_lora_for_extra_net and args.use_lora_extended and not is_locon_installed:
        raise Exception(
            r"a1111-sd-webui-locon extension is required to save "
            r"extra net for extended lora. Please install "
            r"https://github.com/KohakuBlueleaf/a1111-sd-webui-locon"
        )
