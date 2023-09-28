import datetime
import hashlib
from io import BytesIO
from typing import Optional, Union, Tuple

import safetensors.torch

BASE_METADATA = {
    # === Must ===
    "modelspec.sai_model_spec": "1.0.0",  # Required version ID for the spec
    "modelspec.architecture": None,
    "modelspec.implementation": None,
    "modelspec.title": None,
    "modelspec.resolution": None,
    # === Should ===
    "modelspec.description": None,
    "modelspec.author": None,
    "modelspec.date": None,
    # === Can ===
    "modelspec.license": None,
    "modelspec.tags": None,
    "modelspec.merged_from": None,
    "modelspec.prediction_type": None,
    "modelspec.timestep_range": None,
    "modelspec.encoder_layer": None,
}

# 別に使うやつだけ定義
MODELSPEC_TITLE = "modelspec.title"

ARCH_SD_V1 = "stable-diffusion-v1"
ARCH_SD_V2_512 = "stable-diffusion-v2-512"
ARCH_SD_V2_768_V = "stable-diffusion-v2-768-v"
ARCH_SD_XL_V1_BASE = "stable-diffusion-xl-v1-base"

ADAPTER_LORA = "lora"
ADAPTER_TEXTUAL_INVERSION = "textual-inversion"

IMPL_STABILITY_AI = "https://github.com/Stability-AI/generative-models"
IMPL_DIFFUSERS = "diffusers"

PRED_TYPE_EPSILON = "epsilon"
PRED_TYPE_V = "v"


def load_bytes_in_safetensors(tensors):
    bytes = safetensors.torch.save(tensors)
    b = BytesIO(bytes)

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)

    return b.read()


def precalculate_safetensors_hashes(state_dict):
    # calculate each tensor one by one to reduce memory usage
    hash_sha256 = hashlib.sha256()
    for tensor in state_dict.values():
        single_tensor_sd = {"tensor": tensor}
        bytes_for_tensor = load_bytes_in_safetensors(single_tensor_sd)
        hash_sha256.update(bytes_for_tensor)

    return f"0x{hash_sha256.hexdigest()}"


def update_hash_sha256(metadata: dict, state_dict: dict):
    raise NotImplementedError


def build_metadata(
        state_dict: Optional[dict],
        v2: bool,
        v_parameterization: bool,
        sdxl: bool,
        lora: bool,
        textual_inversion: bool,
        timestamp: float,
        title: Optional[str] = None,
        reso: Optional[Union[int, Tuple[int, int]]] = None,
        is_stable_diffusion_ckpt: Optional[bool] = None,
        author: Optional[str] = None,
        description: Optional[str] = None,
        license: Optional[str] = None,
        tags: Optional[str] = None,
        buckets: Optional[dict] = None,
        merged_from: Optional[str] = None,
        timesteps: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
):
    # if state_dict is None, hash is not calculated

    metadata = {}
    metadata.update(BASE_METADATA)
    model_str = ""
    if sdxl:
        arch = ARCH_SD_XL_V1_BASE
        model_str = "sdxl_base_v1-0"
    elif v2:
        model_str = "ss_v2"
        if v_parameterization:
            arch = ARCH_SD_V2_768_V
        else:
            arch = ARCH_SD_V2_512
    else:
        model_str = "ss_v1"
        arch = ARCH_SD_V1

    metadata["ss_base_model_version"] = model_str
    if lora:
        arch += f"/{ADAPTER_LORA}"
    elif textual_inversion:
        arch += f"/{ADAPTER_TEXTUAL_INVERSION}"

    metadata["modelspec.architecture"] = arch

    if not lora and not textual_inversion and is_stable_diffusion_ckpt is None:
        is_stable_diffusion_ckpt = True  # default is stable diffusion ckpt if not lora and not textual_inversion

    if (lora and sdxl) or textual_inversion or is_stable_diffusion_ckpt:
        # Stable Diffusion ckpt, TI, SDXL LoRA
        impl = IMPL_STABILITY_AI
    else:
        # v1/v2 LoRA or Diffusers
        impl = IMPL_DIFFUSERS
    metadata["modelspec.implementation"] = impl


    if title is None:
        if lora:
            title = "LoRA"
        elif textual_inversion:
            title = "TextualInversion"
        else:
            title = "Checkpoint"
        title += f"@{timestamp}"
    metadata[MODELSPEC_TITLE] = title

    if author is not None:
        metadata["modelspec.author"] = author
    else:
        del metadata["modelspec.author"]

    if description is not None:
        metadata["modelspec.description"] = description
    else:
        del metadata["modelspec.description"]

    if merged_from is not None:
        metadata["modelspec.merged_from"] = merged_from
    else:
        del metadata["modelspec.merged_from"]

    if license is not None:
        metadata["modelspec.license"] = license
    else:
        del metadata["modelspec.license"]

    if tags is not None:
        metadata["modelspec.tags"] = tags
        metadata["ss_tag_frequency"] = tags
    else:
        del metadata["modelspec.tags"]

    if buckets is not None:
        metadata["ss_bucket_info"] = buckets

    # remove microsecond from time
    int_ts = int(timestamp)

    # time to iso-8601 compliant date
    date = datetime.datetime.fromtimestamp(int_ts).isoformat()
    metadata["modelspec.date"] = date

    if reso is not None:
        # comma separated to tuple
        if isinstance(reso, str):
            reso = tuple(map(int, reso.split(",")))
        if len(reso) == 1:
            reso = (reso[0], reso[0])
    else:
        # resolution is defined in dataset, so use default
        if sdxl:
            reso = 1024
        elif v2 and v_parameterization:
            reso = 768
        else:
            reso = 512
    if isinstance(reso, int):
        reso = (reso, reso)

    metadata["modelspec.resolution"] = f"{reso[0]}x{reso[1]}"

    if v_parameterization:
        metadata["modelspec.prediction_type"] = PRED_TYPE_V
    else:
        metadata["modelspec.prediction_type"] = PRED_TYPE_EPSILON

    if timesteps is not None:
        if isinstance(timesteps, str) or isinstance(timesteps, int):
            timesteps = (timesteps, timesteps)
        if len(timesteps) == 1:
            timesteps = (timesteps[0], timesteps[0])
        metadata["modelspec.timestep_range"] = f"{timesteps[0]},{timesteps[1]}"
    else:
        del metadata["modelspec.timestep_range"]

    if clip_skip is not None:
        metadata["modelspec.encoder_layer"] = f"{clip_skip}"
    else:
        del metadata["modelspec.encoder_layer"]

    # # assert all values are filled
    # assert all([v is not None for v in metadata.values()]), metadata
    if not all([v is not None for v in metadata.values()]):
        print(f"Internal error: some metadata values are None: {metadata}")

    return metadata

