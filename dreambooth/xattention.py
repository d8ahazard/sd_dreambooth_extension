# Borrowed from Shivam's repo so we don't have to completely clone a different diffusers version
import importlib
import os
from typing import Optional, Union

import torch
from attr import dataclass
from diffusers.models.attention import xformers
from diffusers.pipeline_utils import LOADABLE_CLASSES
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from torch import nn

from modules import shared


@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            Hidden states conditioned on `encoder_hidden_states` input. If discrete, returns probability distributions
            for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


if (shared.cmd_opts.xformers or shared.cmd_opts.force_enable_xformers) and is_xformers_available():
    import xformers
    import xformers.ops

    xformers._is_functorch_available = True
else:
    xformers = None
    print("[!] Not using xformers memory efficient attention.")


class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
            self,
            query_dim: int,
            cross_attention_dim: Optional[int] = None,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
            bias=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = xformers is not None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value)
        elif self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def _attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        if query.device.type == "mps":
            # Better performance on mps (~20-25%)
            attention_scores = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
        else:
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output

        if query.device.type == "mps":
            hidden_states = torch.einsum("b i j, b j d -> b i d", attention_probs, value)
        else:
            hidden_states = torch.matmul(attention_probs, value)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            if query.device.type == "mps":
                # Better performance on mps (~20-25%)
                attn_slice = (
                        torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx])
                        * self.scale
                )
            else:
                attn_slice = (
                        torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
                )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            if query.device.type == "mps":
                attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])
            else:
                attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        return hidden_states


def save_pretrained(self, save_directory: Union[str, os.PathLike]):
    """
    Save all variables of the pipeline that can be saved and loaded as well as the pipelines configuration file to
    a directory. A pipeline variable can be saved and loaded if its class implements both a save and loading
    method. The pipeline can easily be re-loaded using the `[`~DiffusionPipeline.from_pretrained`]` class method.

    Arguments:
        save_directory (`str` or `os.PathLike`):
            Directory to which to save. Will be created if it doesn't exist.
    """
    self.save_config(save_directory)

    model_index_dict = dict(self.config)
    model_index_dict.pop("_class_name")
    model_index_dict.pop("_diffusers_version")
    model_index_dict.pop("_module", None)

    for pipeline_component_name in model_index_dict.keys():
        sub_model = getattr(self, pipeline_component_name)
        if sub_model is None:
            # edge case for saving a pipeline with safety_checker=None
            continue

        model_cls = sub_model.__class__

        save_method_name = None
        # search for the model's base class in LOADABLE_CLASSES
        for library_name, library_classes in LOADABLE_CLASSES.items():
            library = importlib.import_module(library_name)
            for base_class, save_load_methods in library_classes.items():
                class_candidate = getattr(library, base_class)
                if issubclass(model_cls, class_candidate):
                    # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                    save_method_name = save_load_methods[0]
                    break
            if save_method_name is not None:
                break

        if save_method_name is not None:
            save_method = getattr(sub_model, save_method_name)
            save_method(os.path.join(save_directory, pipeline_component_name))