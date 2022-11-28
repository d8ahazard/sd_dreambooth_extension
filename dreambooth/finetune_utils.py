import gc
import os
import random
import re
from typing import Iterable

import torch
import torch.utils.checkpoint
from transformers import CLIPTextModel

from modules import shared, devices


class FilenameTextGetter:
    """Adapted from modules.textual_inversion.dataset.PersonalizedBase to get caption for image."""

    re_numbers_at_start = re.compile(r"^[-\d]+\s*")

    def __init__(self):
        self.re_word = re.compile(shared.opts.dataset_filename_word_regex) if len(
            shared.opts.dataset_filename_word_regex) > 0 else None

    def read_text(self, img_path):
        text_filename = os.path.splitext(img_path)[0] + ".txt"
        filename = os.path.basename(img_path)

        if os.path.exists(text_filename):
            with open(text_filename, "r", encoding="utf8") as file:
                filename_text = file.read()
        else:
            filename_text = os.path.splitext(filename)[0]
            filename_text = re.sub(self.re_numbers_at_start, '', filename_text)
            if self.re_word:
                tokens = self.re_word.findall(filename_text)
                filename_text = (shared.opts.dataset_filename_join_string or "").join(tokens)

        filename_text = filename_text.replace("\\", "")  # work with \(franchies\)
        return filename_text

    def create_text(self, text_template, filename_text, instance_token, class_token, file_prompt_contents,
                    is_class=True):
        # If we are creating text for a class image and it has our instance token in it, remove/replace it
        class_tokens = [f"a {class_token}", f"the {class_token}", f"an {class_token}", class_token]
        if instance_token != "" and class_token != "":
            if is_class and "Instance" in file_prompt_contents:
                if "Class" in file_prompt_contents:
                    filename_text = filename_text.replace(instance_token, "")
                    filename_text = filename_text.replace("  ", " ")
                else:
                    filename_text = filename_text.replace(instance_token, class_token)

            if not is_class:
                # If the instance prompt is not in the prompt, add it
                if "Instance" not in file_prompt_contents:
                    for token in class_tokens:
                        if token in filename_text:
                            filename_text = filename_text.replace(token, f"{instance_token} {class_token}")
                else:
                    # Append the class as well
                    if "Class" not in file_prompt_contents:
                        filename_text = filename_text.replace(instance_token, f"{instance_token} {class_token}")

        tags = filename_text.split(',')
        output = text_template.replace("[filewords]", ','.join(tags))
        return output


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0
        self.collected_params = []

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        devices.torch_gc()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    # From CompVis LitEMA implementation
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

        del self.collected_params
        gc.collect()

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
            dtype: Floating point-type for the stuff.
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]


# Implementation from https://github.com/bmaltais/kohya_ss
def encode_hidden_state(text_encoder: CLIPTextModel, input_ids, pad_tokens, b_size, max_token_length,
                        tokenizer_max_length):
    if pad_tokens:
        input_ids = input_ids.reshape((-1, tokenizer_max_length))  # batch_size*3, 77

    clip_skip = shared.opts.CLIP_stop_at_last_layers
    if clip_skip <= 1:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out['hidden_states'][-clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    if not pad_tokens:
        return encoder_hidden_states

    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if max_token_length > 75:
        sts_list = [encoder_hidden_states[:, 0].unsqueeze(1)]
        for i in range(1, max_token_length, tokenizer_max_length):
            sts_list.append(encoder_hidden_states[:, i:i + tokenizer_max_length - 2])
        sts_list.append(encoder_hidden_states[:, -1].unsqueeze(1))
        encoder_hidden_states = torch.cat(sts_list, dim=1)

    return encoder_hidden_states
