import os
from itertools import groupby
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from extensions.sd_dreambooth_extension.dreambooth import db_shared
from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig
from extensions.sd_dreambooth_extension.dreambooth.db_shared import start_safe_unpickle, stop_safe_unpickle


class LoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )

        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = 1.0

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return self.linear(input) + self.lora_up(self.lora_down(input)) * self.scale


UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}
TEXT_ENCODER_DEFAULT_TARGET_REPLACE = {"CLIPAttention"}

DEFAULT_TARGET_REPLACE = UNET_DEFAULT_TARGET_REPLACE


def _find_children(
    model,
        search_class=None,
):
    """
    Find all modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """
    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    if search_class is None:
        search_class = [nn.Linear]
    for parent in model.modules():
        for name, module in parent.named_children():
            if any([isinstance(module, _class) for _class in search_class]):
                yield parent, name, module


def _find_modules_v2(
    model,
        ancestor_class=None,
        search_class=None,
        exclude_children_of=None,
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    if exclude_children_of is None:
        exclude_children_of = [LoraInjectedLinear]
    if search_class is None:
        search_class = [nn.Linear]
    if ancestor_class is None:
        ancestor_class = DEFAULT_TARGET_REPLACE
    ancestors = (
        module
        for module in model.modules()
        if module.__class__.__name__ in ancestor_class
    )

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module


def _find_modules_old(
    model,
        ancestor_class=None,
        search_class=None,
):
    if search_class is None:
        search_class = [nn.Linear]
    if ancestor_class is None:
        ancestor_class = DEFAULT_TARGET_REPLACE
    ret = []
    for _module in model.modules():
        if _module.__class__.__name__ in ancestor_class:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__ in search_class:
                    ret.append((_module, name, _child_module))
    print(ret)
    return ret


_find_modules = _find_modules_v2


def inject_trainable_lora(
    model: nn.Module,
        target_replace_module=None,
    r: int = 4,
    loras=None,
    device = None# path to lora .pt
):
    """
    inject lora into model, and returns lora parameter groups.
    """
    stop_safe_unpickle()
    if target_replace_module is None:
        target_replace_module = DEFAULT_TARGET_REPLACE
    require_grad_params = []
    names = []

    if loras is not None and os.path.exists(loras) and os.path.isfile(loras):
        loras = torch.load(loras)
    else:
        loras = None

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        _tmp = LoraInjectedLinear(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].lora_up.parameters())
        require_grad_params.append(_module._modules[name].lora_down.parameters())

        if loras != None:
            _module._modules[name].lora_up.weight = loras.pop(0)
            _module._modules[name].lora_down.weight = loras.pop(0)

        _module._modules[name].lora_up.weight.requires_grad = True
        _module._modules[name].lora_down.weight.requires_grad = True
        names.append(name)
    start_safe_unpickle()
    return require_grad_params, names

def apply_lora_weights_sample(target_unet, target_text_encoder, lora_path, lora_txt, lora_alpha=1, lora_txt_alpha=1):
    
    print("Applying lora weights for sampling...")
    
    lora_dicts = [
        {"path": lora_path, "model": target_unet, "alpha": lora_alpha, "is_text": False},
        {"path": lora_txt, "model": target_text_encoder, "alpha": lora_txt_alpha, "is_text": True}
    ]
    
    for loras in lora_dicts:
        if os.path.exists(loras["path"]) and os.path.isfile(loras["path"]):
            monkeypatch_lora(
                loras["model"],
                torch.load(loras["path"]), 
                target_replace_module=None if loras["is_text"] == False else ["CLIPAttention"]
            )
            tune_lora_scale(loras["model"], loras["alpha"]) 

def apply_lora_weights(target_unet, target_text_encoder, config: DreamboothConfig, device=None, is_ui=False):
    if device is None:
        device = db_shared.device
    target_unet.requires_grad_(False)
    lora_path = os.path.join(db_shared.models_path, "lora", config.lora_model_name)
    lora_txt = lora_path.replace(".pt", "_txt.pt")

    if is_ui:
        apply_lora_weights_sample(
            target_unet,
            target_text_encoder,
            lora_path, 
            lora_txt, 
            lora_alpha, 
            lora_txt_alpha
        )
        return

    print("Injecting trainable lora...")
    unet_lora_params, _ = inject_trainable_lora(target_unet, None, config.lora_rank, lora_path, device)
    text_encoder_lora_params = None

    if target_text_encoder is not None:
        target_text_encoder.requires_grad_(False)
        text_encoder_lora_params, _ = inject_trainable_lora(target_text_encoder,
                                                            ["CLIPAttention"], config.lora_rank, lora_txt, device)
    return unet_lora_params, text_encoder_lora_params

def extract_lora_ups_down(model, target_replace_module=None):

    if target_replace_module is None:
        target_replace_module = DEFAULT_TARGET_REPLACE
    loras = []

    for _m, _n, _child_module in _find_modules(
        model, target_replace_module, search_class=[LoraInjectedLinear]
    ):
        loras.append((_child_module.lora_up, _child_module.lora_down))

    if len(loras) == 0:
        raise ValueError("No lora injected.")

    return loras


def save_lora_weight(
    model,
    path="./lora.pt",
        target_replace_module=None,
):
    if target_replace_module is None:
        target_replace_module = DEFAULT_TARGET_REPLACE
    weights = []
    for _up, _down in extract_lora_ups_down(
        model, target_replace_module=target_replace_module
    ):
        weights.append(_up.weight)
        weights.append(_down.weight)

    torch.save(weights, path)


def save_lora_as_json(model, path="./lora.json"):
    weights = []
    for _up, _down in extract_lora_ups_down(model):
        weights.append(_up.weight.detach().cpu().numpy().tolist())
        weights.append(_down.weight.detach().cpu().numpy().tolist())

    import json

    with open(path, "w") as f:
        json.dump(weights, f)


def save_safeloras(
        modelmap=None,
    outpath="./lora.safetensors",
):
    """
    Saves the Lora from multiple modules in a single safetensor file.

    modelmap is a dictionary of {
        "module name": (module, target_replace_module)
    }
    """
    if modelmap is None:
        modelmap = {}
    weights = {}
    metadata = {}

    import json

    from safetensors.torch import save_file

    for name, (model, target_replace_module) in modelmap.items():
        metadata[name] = json.dumps(list(target_replace_module))

        for i, (_up, _down) in enumerate(
            extract_lora_ups_down(model, target_replace_module)
        ):
            metadata[f"{name}:{i}:rank"] = str(_down.out_features)
            weights[f"{name}:{i}:up"] = _up.weight
            weights[f"{name}:{i}:down"] = _down.weight

    print(f"Saving weights to {outpath} with metadata", metadata)
    save_file(weights, outpath, metadata)


def convert_loras_to_safeloras(
        modelmap=None,
    outpath="./lora.safetensors",
):
    """
    Converts the Lora from multiple pytorch .pt files into a single safetensor file.

    modelmap is a dictionary of {
        "module name": (pytorch_model_path, target_replace_module, rank)
    }
    """

    if modelmap is None:
        modelmap = {}
    weights = {}
    metadata = {}

    import json

    from safetensors.torch import save_file

    for name, (path, target_replace_module, r) in modelmap.items():
        metadata[name] = json.dumps(list(target_replace_module))

        lora = torch.load(path)
        for i, weight in enumerate(lora):
            is_up = i % 2 == 0
            i = i // 2

            if is_up:
                metadata[f"{name}:{i}:rank"] = str(r)
                weights[f"{name}:{i}:up"] = weight
            else:
                weights[f"{name}:{i}:down"] = weight

    print(f"Saving weights to {outpath} with metadata", metadata)
    save_file(weights, outpath, metadata)


def parse_safeloras(
    safeloras,
) -> Dict[str, Tuple[List[nn.parameter.Parameter], List[int], List[str]]]:
    """
    Converts a loaded safetensor file that contains a set of module Loras
    into Parameters and other information

    Output is a dictionary of {
        "module name": (
            [list of weights],
            [list of ranks],
            target_replacement_modules
        )
    }
    """
    loras = {}

    import json

    metadata = safeloras.metadata()

    get_name = lambda k: k.split(":")[0]

    keys = list(safeloras.keys())
    keys.sort(key=get_name)

    for name, module_keys in groupby(keys, get_name):
        # Extract the targets
        target = json.loads(metadata[name])

        # Build the result lists - Python needs us to preallocate lists to insert into them
        module_keys = list(module_keys)
        ranks = [None] * (len(module_keys) // 2)
        weights = [None] * len(module_keys)

        for key in module_keys:
            # Split the model name and index out of the key
            _, idx, direction = key.split(":")
            idx = int(idx)

            # Add the rank
            ranks[idx] = json.loads(metadata[f"{name}:{idx}:rank"])

            # Insert the weight into the list
            idx = idx * 2 + (1 if direction == "down" else 0)
            weights[idx] = nn.parameter.Parameter(safeloras.get_tensor(key))

        loras[name] = (weights, ranks, target)

    return loras


def load_safeloras(path, device="cpu"):

    from safetensors.torch import safe_open

    safeloras = safe_open(path, framework="pt", device=device)
    return parse_safeloras(safeloras)


def weight_apply_lora(
    model, loras, target_replace_module=None, alpha=1.0
):

    if target_replace_module is None:
        target_replace_module = DEFAULT_TARGET_REPLACE
    for _m, _n, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight

        up_weight = loras.pop(0).detach().to(weight.device)
        down_weight = loras.pop(0).detach().to(weight.device)

        # W <- W + U * D
        weight = weight + alpha * (up_weight @ down_weight).type(weight.dtype)
        _child_module.weight = nn.Parameter(weight)


def monkeypatch_lora(
    model, loras, target_replace_module=None, r: int = 4
):
    if target_replace_module is None:
        target_replace_module = DEFAULT_TARGET_REPLACE
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        _tmp = LoraInjectedLinear(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=r,
        )
        _tmp.linear.weight = weight

        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _module._modules[name] = _tmp

        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype)
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype)
        )

        _module._modules[name].to(weight.device)


def monkeypatch_replace_lora(
    model, loras, target_replace_module=None, r: int = 4
):
    if target_replace_module is None:
        target_replace_module = DEFAULT_TARGET_REPLACE
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[LoraInjectedLinear]
    ):
        weight = _child_module.linear.weight
        bias = _child_module.linear.bias
        _tmp = LoraInjectedLinear(
            _child_module.linear.in_features,
            _child_module.linear.out_features,
            _child_module.linear.bias is not None,
            r=r,
        )
        _tmp.linear.weight = weight

        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _module._modules[name] = _tmp

        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype)
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype)
        )

        _module._modules[name].to(weight.device)


def monkeypatch_or_replace_lora(
    model,
    loras,
        target_replace_module=None,
    r: Union[int, List[int]] = 4,
):
    if target_replace_module is None:
        target_replace_module = DEFAULT_TARGET_REPLACE
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear, LoraInjectedLinear]
    ):
        _source = (
            _child_module.linear
            if isinstance(_child_module, LoraInjectedLinear)
            else _child_module
        )

        weight = _source.weight
        bias = _source.bias
        _tmp = LoraInjectedLinear(
            _source.in_features,
            _source.out_features,
            _source.bias is not None,
            r=r.pop(0) if isinstance(r, list) else r,
        )
        _tmp.linear.weight = weight

        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _module._modules[name] = _tmp

        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype)
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype)
        )

        _module._modules[name].to(weight.device)


def monkeypatch_or_replace_safeloras(models, safeloras):
    loras = parse_safeloras(safeloras)

    for name, (lora, ranks, target) in loras.items():
        model = getattr(models, name, None)

        if not model:
            print(f"No model provided for {name}, contained in Lora")
            continue

        monkeypatch_or_replace_lora(model, lora, target, ranks)


def monkeypatch_remove_lora(model):
    for _module, name, _child_module in _find_children(
        model, search_class=[LoraInjectedLinear]
    ):
        _source = _child_module.linear
        weight, bias = _source.weight, _source.bias

        _tmp = nn.Linear(_source.in_features, _source.out_features, bias is not None)

        _tmp.weight = weight
        if bias is not None:
            _tmp.bias = bias

        _module._modules[name] = _tmp


def monkeypatch_add_lora(
    model,
    loras,
        target_replace_module=None,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    if target_replace_module is None:
        target_replace_module = DEFAULT_TARGET_REPLACE
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[LoraInjectedLinear]
    ):
        weight = _child_module.linear.weight

        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype).to(weight.device) * alpha
            + _module._modules[name].lora_up.weight.to(weight.device) * beta
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype).to(weight.device) * alpha
            + _module._modules[name].lora_down.weight.to(weight.device) * beta
        )

        _module._modules[name].to(weight.device)


def tune_lora_scale(model, alpha: float = 1.0):
    for _module in model.modules():
        if _module.__class__.__name__ == "LoraInjectedLinear":
            _module.scale = alpha


def _text_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["text_encoder", "pt"])


def _ti_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["ti", "pt"])


def load_learned_embed_in_clip(
    learned_embeds_path, text_encoder, tokenizer, token=None, idempotent=False
):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    i = 1
    if not idempotent:
        while num_added_tokens == 0:
            print(f"The tokenizer already contains the token {token}.")
            token = f"{token[:-1]}-{i}>"
            print(f"Attempting to add the token {token}.")
            num_added_tokens = tokenizer.add_tokens(token)
            i += 1
    elif num_added_tokens == 0 and idempotent:
        print(f"The tokenizer already contains the token {token}.")
        print(f"Replacing {token} embedding.")

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token


def patch_pipe(
    pipe,
    unet_path,
    token: str,
    r: int = 4,
    patch_unet=True,
    patch_text=False,
    patch_ti=False,
    idempotent_token=True,
        unet_target_replace_module=None,
        text_target_replace_module=None,
):
    if text_target_replace_module is None:
        text_target_replace_module = TEXT_ENCODER_DEFAULT_TARGET_REPLACE
    if unet_target_replace_module is None:
        unet_target_replace_module = DEFAULT_TARGET_REPLACE
    assert (
        len(token) > 0
    ), "Token cannot be empty. Input token non-empty token like <s>."

    ti_path = _ti_lora_path(unet_path)
    text_path = _text_lora_path(unet_path)

    if patch_unet:
        print("LoRA : Patching Unet")
        monkeypatch_or_replace_lora(
            pipe.unet,
            torch.load(unet_path),
            r=r,
            target_replace_module=unet_target_replace_module,
        )

    if patch_text:
        print("LoRA : Patching text encoder")
        monkeypatch_or_replace_lora(
            pipe.text_encoder,
            torch.load(text_path),
            target_replace_module=text_target_replace_module,
            r=r,
        )
    if patch_ti:
        print("LoRA : Patching token input")
        token = load_learned_embed_in_clip(
            ti_path,
            pipe.text_encoder,
            pipe.tokenizer,
            token,
            idempotent=idempotent_token,
        )


@torch.no_grad()
def inspect_lora(model):
    moved = {}

    for name, _module in model.named_modules():
        if _module.__class__.__name__ == "LoraInjectedLinear":
            ups = _module.lora_up.weight.data.clone()
            downs = _module.lora_down.weight.data.clone()

            wght: torch.Tensor = ups @ downs

            dist = wght.flatten().abs().mean().item()
            if name in moved:
                moved[name].append(dist)
            else:
                moved[name] = [dist]

    return moved


def save_all(
    unet,
    text_encoder,
    placeholder_token_id,
    placeholder_token,
    save_path,
    save_lora=True,
        target_replace_module_text=None,
        target_replace_module_unet=None,
):

    # save ti
    if target_replace_module_unet is None:
        target_replace_module_unet = DEFAULT_TARGET_REPLACE
    if target_replace_module_text is None:
        target_replace_module_text = TEXT_ENCODER_DEFAULT_TARGET_REPLACE
    ti_path = _ti_lora_path(save_path)
    learned_embeds = text_encoder.get_input_embeddings().weight[placeholder_token_id]
    print("Current Learned Embeddings: ", learned_embeds[:4])

    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, ti_path)
    print("Ti saved to ", ti_path)

    if save_lora:
        save_lora_weight(
            unet, save_path, target_replace_module=target_replace_module_unet
        )
        print("Unet saved to ", save_path)

        save_lora_weight(
            text_encoder,
            _text_lora_path(save_path),
            target_replace_module=target_replace_module_text,
        )
        print("Text Encoder saved to ", _text_lora_path(save_path))
