import os
from typing import List

import torch
import torch.nn as nn

from modules import shared, paths


class LoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4):
        super().__init__()

        if r >= min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less than {min(in_features, out_features)}"
            )

        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = 1.0

        nn.init.normal_(self.lora_down.weight, std=1 / r ** 2)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return self.linear(input) + self.lora_up(self.lora_down(input)) * self.scale


def inject_trainable_lora(
        model: nn.Module,
        target_replace_module: List[str] = ["CrossAttention", "Attention"],
        r: int = 4,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear":

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
                    _module._modules[name] = _tmp

                    require_grad_params.append(
                        _module._modules[name].lora_up.parameters()
                    )
                    require_grad_params.append(
                        _module._modules[name].lora_down.parameters()
                    )

                    _module._modules[name].lora_up.weight.requires_grad = True
                    _module._modules[name].lora_down.weight.requires_grad = True
                    names.append(name)

    return require_grad_params, names


def extract_lora_ups_down(model, target_replace_module=None):
    no_injection = True
    if target_replace_module is None:
        target_replace_module = ["CrossAttention", "Attention"]

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            for _child_module in _module.modules():
                if _child_module.__class__.__name__ == "LoraInjectedLinear":
                    no_injection = False
                    yield _child_module.lora_up, _child_module.lora_down
    if no_injection:
        raise ValueError("No lora injected.")


def save_lora_weight(
        model, path="./lora.pt", target_replace_module=["CrossAttention", "Attention"]
):
    weights = []
    for _up, _down in extract_lora_ups_down(
            model, target_replace_module=target_replace_module
    ):
        weights.append(_up.weight)
        weights.append(_down.weight)

    torch.save(weights, path)


def get_lora_weight(model):
    weights = []
    for _up, _down in extract_lora_ups_down(model):
        weights.append(_up.weight)
        weights.append(_down.weight)

    return weights


def save_lora_as_json(model, path="./lora.json"):
    weights = []
    for _up, _down in extract_lora_ups_down(model):
        weights.append(_up.weight.detach().cpu().numpy().tolist())
        weights.append(_down.weight.detach().cpu().numpy().tolist())

    import json

    with open(path, "w") as f:
        json.dump(weights, f)


def weight_apply_lora(
        model, loras, target_replace_module=["CrossAttention", "Attention"], alpha=1.0
):
    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            for _child_module in _module.modules():
                if _child_module.__class__.__name__ == "Linear":
                    weight = _child_module.weight

                    up_weight = loras.pop(0).detach().to(weight.device)
                    down_weight = loras.pop(0).detach().to(weight.device)

                    # W <- W + U * D
                    weight = weight + alpha * (up_weight @ down_weight).type(
                        weight.dtype
                    )
                    _child_module.weight = nn.Parameter(weight)


def apply_lora_weights(lora_model, target_unet, target_text_encoder, lora_alpha=1, lora_txt_alpha=1, device=None):
    if device is None:
        device = shared.device
    if not lora_model:
        lora_model = ""
    target_unet.requires_grad_(False)
    lora_path = os.path.join(paths.models_path, "lora", lora_model)
    lora_txt = lora_path.replace(".pt", "_txt.pt")
    if os.path.exists(lora_path) and os.path.isfile(lora_path):
        print("Applying lora unet weights before training...")
        loras = torch.load(lora_path, map_location=device)
        weight_apply_lora(target_unet, loras, alpha=lora_alpha)
    print("Injecting trainable lora...")
    unet_lora_params, _ = inject_trainable_lora(target_unet)
    text_encoder_lora_params = None

    if target_text_encoder is not None:
        target_text_encoder.requires_grad_(False)
        if os.path.exists(lora_txt) and os.path.isfile(lora_txt):
            print("Applying lora text_encoder weights before training...")
            loras = torch.load(lora_txt, map_location=device)
            weight_apply_lora(target_text_encoder, loras, target_replace_module=["CLIPAttention"], alpha=lora_txt_alpha)
        text_encoder_lora_params, _ = inject_trainable_lora(target_text_encoder,
                                                            target_replace_module=["CLIPAttention"])

    return unet_lora_params, text_encoder_lora_params


def monkeypatch_lora(
        model, loras, target_replace_module=None
):
    if target_replace_module is None:
        target_replace_module = ["CrossAttention", "Attention"]
    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear":

                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = LoraInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
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


def tune_lora_scale(model, alpha: float = 1.0):
    for _module in model.modules():
        if _module.__class__.__name__ == "LoraInjectedLinear":
            _module.scale = alpha
