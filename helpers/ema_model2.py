import copy
import json
import os
from typing import Union, Optional, Tuple

import safetensors.torch
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from torch import device, no_grad


class EMAModel(UNet2DConditionModel):
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, model, update_after_step=0, inv_gamma=1.0, power=2 / 3, min_value=0.0, max_value=0.9999,
                 **params):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        super().__init__()
        self.unet = copy.deepcopy(model).eval()
        self.unet.requires_grad_(False)
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.decay = 0.0
        self.optimization_step = 0

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to(self, device=None) -> None:
        self.unet = self.unet.to(device=device)
        return self

    @property
    def dtype(self) -> torch.dtype:
        return self.unet.dtype

    @property
    def device(self) -> device:
        return self.unet.device

    def __call__(self, noisy_latents, timesteps, encoder_hidden_states):
        with torch.no_grad, torch.autocast:
            return self.unet(noisy_latents, timesteps, encoder_hidden_states)

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        return self.unet.forward(sample, timestep, encoder_hidden_states, class_labels, return_dict)

    def set_use_memory_efficient_attention_xformers(self, valid: bool) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(self.unet)

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        ema_params = self.unet.state_dict()
        self.decay = self.get_decay(self.optimization_step)

        for name, param in new_model.named_parameters():
            if param.requires_grad:
                ema_param = ema_params[name]
                ema_param.mul_(self.decay)
                ema_param.add_(param.data, alpha=1 - self.decay)

        for name, buffer in new_model.named_buffers():
            ema_params[name] = buffer

        self.unet.load_state_dict(ema_params, strict=False)
        self.optimization_step += 1

    def parameters(self, recurse: bool = True):
        return self.unet.parameters()


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, **kwargs)
        ema_params_path = os.path.join(pretrained_model_name_or_path, "ema_params.json")
        if os.path.exists(ema_params_path):
            with open(ema_params_path, "r") as f:
                params = json.load(f)
            return cls(model, **params)
        return cls(model)


    def save_pretrained(self, save_directory: str, safe_serialization: bool = False, **kwargs):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        params = {
            "update_after_step": self.update_after_step,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "min_value": self.min_value,
            "max_value": self.max_value
        }

        with open(os.path.join(save_directory, "ema_params.json"), "w") as f:
            json.dump(params, f)

        if safe_serialization:
            safetensors.torch.save_file(self.unet.state_dict(),
                                        os.path.join(save_directory, "diffusion_pytorch_model.safetensors"))
        else:
            torch.save(self.unet.state_dict(), os.path.join(save_directory, "diffusion_pytorch_model.bin"))
