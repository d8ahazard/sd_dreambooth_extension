from __future__ import annotations

import collections
import contextlib
import json
import logging
import os
import re
import sys
from typing import Dict

import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import PretrainedConfig

from dreambooth import shared  # noqa
from dreambooth.dataclasses.db_config import DreamboothConfig  # noqa
from dreambooth.utils.utils import cleanup  # noqa
from modules import hashes
from modules.safe import unsafe_torch_load, load

logger = logging.getLogger(__name__)
checkpoints_list = {}
checkpoint_alisases = {}
checkpoints_loaded = collections.OrderedDict()

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(shared.models_path, model_dir))

LORA_SHARED_SRC_CREATE = " <create new>"


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


class CheckpointInfo:
    def __init__(self, filename):
        self.filename = filename
        abspath = os.path.abspath(filename)

        if shared.cmd_opts.ckpt_dir is not None and abspath.startswith(shared.cmd_opts.ckpt_dir):
            name = abspath.replace(shared.cmd_opts.ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        if name.startswith(os.sep) or name.startswith(os.sep):
            name = name[1:]

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace(os.sep, "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, "checkpoint/" + name)
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, name, f'{name} [{self.hash}]'] + (
            [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]'] if self.shorthash else [])

    def register(self):
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_alisases[id] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, "checkpoint/" + self.name)
        if self.sha256 is None:
            return

        self.shorthash = self.sha256[0:10]

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]']

        checkpoints_list.pop(self.title)
        self.title = f'{self.name} [{self.shorthash}]'
        self.register()

        return self.shorthash


def list_models():
    checkpoints_list.clear()
    checkpoint_alisases.clear()
    model_list = modelloader.load_models(model_path=model_path, command_path=shared.cmd_opts.ckpt_dir,
                                         ext_filter=[".ckpt", ".safetensors"], ext_blacklist=[".vae.safetensors"])

    cmd_ckpt = shared.cmd_opts.ckpt
    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()

        shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        logger.debug(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}",
                     file=sys.stderr)

    for filename in model_list:
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_db_models():
    output = [""]
    out_dir = shared.dreambooth_models_path
    if os.path.exists(out_dir):
        for item in os.listdir(out_dir):
            if os.path.isdir(os.path.join(out_dir, item)):
                output.append(item)
    return output


def get_shared_models():
    output = ["", LORA_SHARED_SRC_CREATE]
    out_dir = os.path.join(shared.models_path, "diffusers")
    if os.path.exists(out_dir):
        for item in os.listdir(out_dir):
            if os.path.isdir(os.path.join(out_dir, item)):
                output.append(item)
    return output


def get_lora_models(config: DreamboothConfig = None):
    output = [""]
    if config is None:
        config = shared.db_model_config
    if config is not None:
        lora_dir = os.path.join(shared.models_path, "Lora")
        if os.path.exists(lora_dir):
            files = os.listdir(lora_dir)
            for file in files:
                if os.path.isfile(os.path.join(lora_dir, file)):
                    if ".safetensors" in file or ".pt" in file or ".ckpt" in file:
                        output.append(file)
    return output


def get_sorted_lora_models(config: DreamboothConfig = None):
    models = get_lora_models(config)

    def get_iteration(name: str):
        regex = re.compile(r'.*_(\d+)\.pt$')
        match = regex.search(name)
        return int(match.group(1)) if match else 0

    sorted_models = sorted(models, key=lambda x: get_iteration(x))
    # Insert empty string at the beginning
    sorted_models.insert(0, "")
    return sorted_models


def get_model_snapshots(config: DreamboothConfig = None):
    snaps = [""]
    if config is None:
        config = shared.db_model_config
    if config is not None:
        snaps_dir = os.path.join(config.model_dir, "checkpoints")
        if os.path.exists(snaps_dir):
            for file in os.listdir(snaps_dir):
                if os.path.isdir(os.path.join(snaps_dir, file)):
                    rev_parts = file.split("-")
                    if rev_parts[0] == "checkpoint" and len(rev_parts) == 2:
                        snaps.append(rev_parts[1])
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
        logger.debug("Restored system models.")
    except:
        pass


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision,
                                               subfolder: str = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    """
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict


def get_checkpoint_match(search_string):
    try:
        from modules import sd_models
        for info in sd_models.checkpoints_list.values():
            if search_string in info.title or search_string in info.model_name or search_string in info.filename:
                return info
    except:
        pass
    return None


disable_safe_unpickle_count = 0


def disable_safe_unpickle():
    global disable_safe_unpickle_count
    try:
        from modules import shared as auto_shared
        if not auto_shared.cmd_opts.disable_safe_unpickle:
            auto_shared.cmd_opts.disable_safe_unpickle = True
            torch.load = unsafe_torch_load
        disable_safe_unpickle_count += 1
    except:
        pass


def enable_safe_unpickle():
    global disable_safe_unpickle_count
    try:
        from modules import shared as auto_shared
        if disable_safe_unpickle_count > 0:
            disable_safe_unpickle_count -= 1
            if disable_safe_unpickle_count == 0 and auto_shared.cmd_opts.disable_safe_unpickle:
                auto_shared.cmd_opts.disable_safe_unpickle = False
                torch.load = load
    except:
        pass


@contextlib.contextmanager
def safe_unpickle_disabled():
    disable_safe_unpickle()
    try:
        yield
    finally:
        enable_safe_unpickle()


def xformerify(obj, use_lora):
    try:
        import xformers
        obj.enable_xformers_memory_efficient_attention
        logger.debug("Enabled XFormers for " + obj.__class__.__name__)

    except ImportError:
        obj.set_attn_processor(AttnProcessor2_0())
        logger.debug("Enabled AttnProcessor2_0 for " + obj.__class__.__name__)


def torch2ify(unet):
    if hasattr(torch, 'compile'):
        try:
            unet = torch.compile(unet, mode="max-autotune", fullgraph=False)
            logger.debug("Enabled Torch2 compilation for unet.")
        except:
            pass
    return unet


def is_xformers_available():
    pass


def read_metadata_from_safetensors(filename):
    with open(filename, mode="rb") as file:
        # Read metadata length
        metadata_len = int.from_bytes(file.read(8), "little")

        # Read the metadata based on its length
        json_data = file.read(metadata_len).decode('utf-8')

        res = {}

        # Check if it's a valid JSON string
        try:
            json_obj = json.loads(json_data)
        except json.JSONDecodeError:
            return res

        # Extract metadata
        metadata = json_obj.get("__metadata__", {})
        if not isinstance(metadata, dict):
            return res

        # Process the metadata to handle nested JSON strings
        for k, v in metadata.items():
            # if not isinstance(v, str):
            #     raise ValueError("All values in __metadata__ must be strings")

            # If the string value looks like a JSON string, attempt to parse it
            if v.startswith('{'):
                try:
                    res[k] = json.loads(v)
                except Exception:
                    res[k] = v
            else:
                res[k] = v

        return res
