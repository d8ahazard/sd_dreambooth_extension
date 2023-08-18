# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conversion script for the LDM checkpoints. """

import importlib
import json
import logging
import os
import shutil
import traceback
from typing import Union

import torch
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

from dreambooth import shared
from dreambooth.dataclasses.db_config import DreamboothConfig
from dreambooth.utils.model_utils import enable_safe_unpickle, disable_safe_unpickle


def copy_config_file(original_config_file, dest_dir, model_name):
    if original_config_file is not None and os.path.exists(original_config_file):
        shutil.copy(original_config_file, dest_dir)
        basename = os.path.basename(original_config_file)
        if basename == f"{model_name}.yaml":
            return
        new_ex_path = os.path.join(dest_dir, basename)
        new_name = os.path.join(dest_dir, f"{model_name}.yaml")
        if os.path.exists(new_name):
            os.remove(new_name)
        os.rename(new_ex_path, new_name)


def get_config_path(
        model_version: str = "v1",
        train_type: str = "default",
        config_base_name: str = "training",
        prediction_type: str = "epsilon"
):
    train_type = f"{train_type}" if not prediction_type == "v_prediction" else f"{train_type}-v"

    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "configs",
        f"{model_version}-{config_base_name}-{train_type}.yaml"
    )


def get_config_file(train_unfrozen=False, v2=False, prediction_type="epsilon"):
    config_base_name = "training"

    model_versions = {
        "v1": "v1",
        "v2": "v2"
    }
    train_types = {
        "default": "default",
        "unfrozen": "unfrozen",
    }

    model_train_type = train_types["default"]
    model_version_name = f"{model_versions['v1'] if not v2 else model_versions['v2']}"

    if train_unfrozen:
        model_train_type = train_types["unfrozen"]

    return get_config_path(model_version_name, model_train_type, config_base_name, prediction_type)


def extract_checkpoint(
    new_model_name: str,
    checkpoint_file: str,
    extract_ema: bool = False,
    train_unfrozen: bool = False,
    is_512: bool = True,
    model_type="v1x",
    original_config_file: str = None,
    num_in_channels: int = None,
    scheduler_type: str = "pndm",
    pipeline_type: str = None,
    image_size: int = None,
    prediction_type: str = None,
    upcast_attention: bool = False,
    device: str = None,
    stable_unclip: str = None,
    stable_unclip_prior: str = None,
    clip_stats_path: str = None,
    controlnet: bool = False,
    half: bool = False,
    vae_path: str = None,
    pipeline_class_name: str = None
) -> Union[None, DreamboothConfig]:
    """
    Extract a checkpoint from a given path and convert it.

    Parameters:
    - checkpoint_file (str): Path to the checkpoint to convert.
    - dump_path (str): Path to the output model.
    - original_config_file (str): The YAML config file corresponding to the original architecture.
    - ... [other parameters matching those from the command line arguments]

    Returns:
    None
    """
    # sh = None
    # try:
    #     from core.modules.status import StatusHandler
    #     sh = StatusHandler()
    # except:
    #     pass
    #
    # def update_status(status):
    #     if sh is not None:
    #         sh.update_status(status)
    #     else:
    #         modules.shared.status.update(status)
    disable_safe_unpickle()
    if image_size is None:
        image_size = 512 if is_512 else 768

    to_safetensors = True
    if pipeline_class_name is not None:
        library = importlib.import_module("diffusers")
        class_obj = getattr(library, pipeline_class_name)
        pipeline_class = class_obj
    else:
        pipeline_class = None

    if original_config_file is None:
        original_config_file = get_config_file(train_unfrozen, v2=is_512 == False, prediction_type=prediction_type)
    print(f"Extracting config from {original_config_file}")
    checkpoint_file = os.path.join(shared.models_path, checkpoint_file)
    print(f"Extracting checkpoint from {checkpoint_file}")

    from_safetensors = False
    if ".safetensors" in checkpoint_file:
        from_safetensors = True
    required_elements = ["unet", "vae", "text_encoder", "scheduler", "tokenizer"]
    db_config = DreamboothConfig(model_name=new_model_name, src=checkpoint_file)
    db_config.model_type = model_type
    db_config.resolution = image_size
    db_config.save()
    try:
        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path=checkpoint_file,
            original_config_file=original_config_file,
            image_size=image_size,
            prediction_type=prediction_type,
            model_type=pipeline_type,
            extract_ema=extract_ema,
            scheduler_type=scheduler_type,
            num_in_channels=num_in_channels,
            upcast_attention=upcast_attention,
            from_safetensors=from_safetensors,
            device=device,
            stable_unclip=stable_unclip,
            stable_unclip_prior=stable_unclip_prior,
            clip_stats_path=clip_stats_path,
            controlnet=controlnet,
            vae_path=vae_path,
            pipeline_class=pipeline_class,
        )

        if half:
            pipe.to(torch_dtype=torch.float16)

        dump_path = db_config.get_pretrained_model_name_or_path()
        if controlnet:
            # only save the controlnet model
            pipe.controlnet.save_pretrained(dump_path, safe_serialization=to_safetensors)
        else:
            try:
                tmp_path = f"{dump_path}_tmp"
                pipe.save_pretrained(dump_path, safe_serialization=True)
            except:
                print("Couldn't save the pipe")
                traceback.print_exc()
                return

    except:
        print("Something went wrong, removing model directory")
        traceback.print_exc()
        pass


    copy_config_file(original_config_file, db_config.model_dir, db_config.model_name)
    success = True
    for req_dir in required_elements:
        full_path = os.path.join(db_config.get_pretrained_model_name_or_path(), req_dir)
        if not os.path.exists(full_path):
            shutil.rmtree(db_config.model_dir, ignore_errors=False, onerror=None)
            success = False
            print(f"Couldn't find {full_path}")
            break
    remove_dirs = ["logging", "samples"]
    enable_safe_unpickle()
    if success:
        for rd in remove_dirs:
            rem_dir = os.path.join(db_config.model_dir, rd)
            if os.path.exists(rem_dir):
                shutil.rmtree(rem_dir, True)
                if not os.path.exists(rem_dir):
                    logging.getLogger(__name__).info(f"Making rd {rem_dir}")
                    os.makedirs(rem_dir)

        return db_config
    return None