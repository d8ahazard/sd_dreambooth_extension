import json
import logging
import traceback
from typing import Optional, Set

import torch
import torch.nn as nn
from safetensors.torch import load_file
from safetensors.torch import save_file as safe_save

from lora_diffusion.lora import DEFAULT_TARGET_REPLACE, LoraInjectedLinear, LoraInjectedConv2d

logger = logging.getLogger(__name__)


def _find_modules_with_ancestor(
        model,
        ancestor_class: Optional[Set[str]] = None,
        search_class=None,
        exclude_children_of=None,
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).
    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by as the full ansestor name.
    This is a copy of _find_modules_v2 from lora.py. It was copied instead of
    refactored to keep the implementation in lora.py in sync with cloneofsimo.
    """
    # Get the targets we should replace all linears under
    if exclude_children_of is None:
        exclude_children_of = [
            LoraInjectedLinear,
            LoraInjectedConv2d,
        ]
    if search_class is None:
        search_class = [nn.Linear]
    if ancestor_class is not None:
        ancestors = (
            (name, module)
            for name, module in model.named_modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [(name, module) for name, module in model.named_modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for anc_name, ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                        [isinstance(parent, _class)
                         for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module, fullname, anc_name


def get_extra_networks_diffsuers_key(name, child_module, full_child_name):
    """
    Computed the diffusers key that is compatible with the extra networks feature in the webui.
    """
    if child_module.__class__.__name__ == "LoraInjectedLinear" or child_module.__class__.__name__ == "LoraInjectedConv2d":
        lora_name = f"{name}.{full_child_name}"
        lora_name = lora_name.replace('.', '_')
        return lora_name
    else:
        print(f"Unsupported module type {child_module.__class__.__name__}")
        return None


def get_extra_networks_ups_down(model, target_replace_module=None):
    """
    Get a list of loras and keys for saving to extra networks.
    """
    if target_replace_module is None:
        target_replace_module = DEFAULT_TARGET_REPLACE
    loras = []

    for _m, child_name, _child_module, fullname, ancestor_name in _find_modules_with_ancestor(
            model,
            target_replace_module,
            search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):
        key = get_extra_networks_diffsuers_key(
            ancestor_name, _child_module, fullname)
        if key is None:
            print(f"{ancestor_name}, {child_name} did not converst to key.")
        loras.append((key, _child_module.lora_up, _child_module.lora_down))

    if len(loras) == 0:
        raise ValueError("No lora injected.")

    return loras


def save_extra_networks(model_map=None, out_path="./lora.safetensors"):
    """
    Saves the Lora from multiple modules in a single safetensor file that is compatible with extra_networks.
    modelmap is a dictionary of {
        "module name": (module, target_replace_module)
    }

    metadata contains a mapping from the keys to the normal lora keys.
    """

    if model_map is None:
        model_map = {}
    weights = {}
    metadata = {"lora_key_encoding": "extra_network_diffusers"}
    # set some metadata
    for name, (model, target_replace_module) in model_map.items():
        metadata[name] = json.dumps(list(target_replace_module))
        prefix = "lora_unet" if name == "unet" else "lora_te"
        rank = None
        for i, (_key, _up, _down) in enumerate(
                get_extra_networks_ups_down(model, target_replace_module)
        ):
            try:
                rank = getattr(_down, "out_features")
            except:
                rank = getattr(_down, "out_channels")
            weights[f"{prefix}_{_key}.lora_up.weight"] = _up.weight
            weights[f"{prefix}_{_key}.lora_down.weight"] = _down.weight
        if rank:
            metadata[f"{prefix}_rank"] = f"{rank}"

    print(f"Saving weights to {out_path}")
    safe_save(weights, out_path, metadata)


def apply_lora(pipeline, checkpoint_path=None, alpha=0.75, model_map=None):
    lora_prefix_unet = "lora_unet"
    lora_prefix_text_encoder = "lora_te"
    if model_map is None and checkpoint_path is None:
        raise ValueError("Must provide either model_map or checkpoint_path")
    # load LoRA weight from .safetensors or use modelmap
    if model_map is not None:
        state_dict = model_map
    else:
        state_dict = load_file(checkpoint_path)

    visited = []
    errors = 0
    total = 0
    bad_keys = []

    # directly update weight in diffusers model
    for key in state_dict:
        try:
            if ".alpha" in key or key in visited:
                continue
            total += 1

            if "text" in key:
                layer_infos = key.split(".")[0].split(lora_prefix_text_encoder + "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
            else:
                layer_infos = key.split(".")[0].split(lora_prefix_unet + "_")[-1].split("_")
                curr_layer = pipeline.unet

            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            pair_keys = []
            if "lora_down" in key:
                pair_keys.append(key.replace("lora_down", "lora_up"))
                pair_keys.append(key)
            else:
                pair_keys.append(key)
                pair_keys.append(key.replace("lora_up", "lora_down"))

            # update weight
            if len(state_dict[pair_keys[0]].shape) == 4:
                weight_up = state_dict[pair_keys[0]].to(torch.float32).reshape(state_dict[pair_keys[0]].shape[0], -1)
                weight_down = state_dict[pair_keys[1]].to(torch.float32).transpose(-1, -2).reshape(
                    state_dict[pair_keys[1]].shape[0], -1)

                # Calculate the reshaping dimensions for the output tensor
                out_channels, in_channels, kernel_height, kernel_width = curr_layer.weight.shape

                updated_weight = torch.matmul(weight_up, weight_down).reshape(out_channels, in_channels, kernel_height,
                                                                              kernel_width)
                curr_layer.weight.data += alpha * updated_weight
            else:
                weight_up = state_dict[pair_keys[0]].to(torch.float32)
                weight_down = state_dict[pair_keys[1]].to(torch.float32)
                curr_layer.weight.data += alpha * torch.matmul(weight_up, weight_down)

            # update visited list
            for item in pair_keys:
                visited.append(item)

        except Exception as e:
            errors += 1
            logger.debug(f"Exception loading LoRA key {key}: {e} {traceback.format_exc()}")
            bad_keys.append(key)

    logger.debug(f"LoRA loaded {total - errors} / {total} keys")
    logger.debug(f"BadKeys: {bad_keys}")
    return pipeline
