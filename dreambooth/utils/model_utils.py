from __future__ import annotations

import os
from typing import Union

import gradio
from transformers import PretrainedConfig

from extensions.sd_dreambooth_extension.dreambooth import shared
from extensions.sd_dreambooth_extension.dreambooth.utils.utils import cleanup
from modules.sd_models import CheckpointInfo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_db_models():
    model_dir = shared.models_path
    out_dir = os.path.join(model_dir, "dreambooth")
    output = []
    if os.path.exists(out_dir):
        dirs = os.listdir(out_dir)
        for found in dirs:
            if os.path.isdir(os.path.join(out_dir, found)):
                output.append(found)
    return output

def get_lora_models():
    model_dir = shared.lora_models_path
    out_dir = os.path.join(model_dir, "lora")
    output = [""]
    if os.path.exists(out_dir):
        dirs = os.listdir(out_dir)
        for found in dirs:
            if os.path.isfile(os.path.join(out_dir, found)):
                if "_txt.pt" not in found and ".pt" in found:
                    output.append(found)
    return output

def get_model_snapshots(model_name: str):
    from extensions.sd_dreambooth_extension.dreambooth.dataclasses.db_config import from_file
    result = gradio.update(visible=True)
    if model_name == "" or model_name is None:
        return result
    config = from_file(model_name)
    snaps_path = os.path.join(config.model_dir, "snapshots")
    snaps = []
    if os.path.exists(snaps_path):
        for directory in os.listdir(snaps_path):
            if "checkpoint_" in directory:
                fullpath = os.path.join(snaps_path, directory)
                snaps.append(fullpath)
    return snaps

def unload_system_models():
    try:
        import modules.shared
        if modules.shared.sd_model is not None:
            modules.shared.sd_model.to("cpu")
        for former in modules.shared.face_restorers:
            try:
                former.send_model_to("cpu")
            except:
                pass
        cleanup()
    except:
        pass

def reload_system_models():
    try:
        import modules.shared
        if modules.shared.sd_model is not None:
            modules.shared.sd_model.to(shared.device)
        print("Restored system models.")
    except:
        pass

def get_checkpoint_match(search_string) -> Union[CheckpointInfo, None]:
    try:
        from modules import sd_models
        for info in sd_models.checkpoints_list.values():
            if search_string in info.title or search_string in info.model_name or search_string in info.filename:
                return info
    except:
        pass
    return None


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

def disable_safe_unpickle():
    try:
        from modules import shared as auto_shared
        auto_shared.cmd_opts.disable_safe_unpickle = True
    except:
        pass


def enable_safe_unpickle():
    try:
        from modules import shared as auto_shared
        auto_shared.cmd_opts.disable_safe_unpickle = False
    except:
        pass

