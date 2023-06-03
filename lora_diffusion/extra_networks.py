import json
from typing import Optional, Set

import torch.nn as nn
from safetensors.torch import save_file as safe_save

from lora_diffusion.lora import DEFAULT_TARGET_REPLACE, LoraInjectedLinear, LoraInjectedConv2d


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


def save_extra_networks(modelmap={}, outpath="./lora.safetensors"):
    """
    Saves the Lora from multiple modules in a single safetensor file that is compatible with extra_networks.
    modelmap is a dictionary of {
        "module name": (module, target_replace_module)
    }

    metadata contains a mapping from the keys to the normal lora keys.
    """

    weights = {}
    metadata = {}
    # set some metadata
    metadata["lora_key_encoding"] = "extra_network_diffusers"
    for name, (model, target_replace_module) in modelmap.items():
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

    print(f"Saving weights to {outpath}")
    safe_save(weights, outpath, metadata)
