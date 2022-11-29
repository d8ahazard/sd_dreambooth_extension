# Borrowed from Shivam's repo so we don't have to completely clone a different diffusers version
import importlib
import math
import os
from typing import Any, Dict, Union

import diffusers
import torch
import transformers
from diffusers.pipeline_utils import LOADABLE_CLASSES
from einops import rearrange
from torch import einsum


def replace_unet_cross_attn_to_default():
    print("Replace CrossAttention.forward to use default")

    def forward_default(self, hidden_states, context=None, mask=None):
        # diffusers.models.attention.CrossAttention.forward
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
        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(
                query, key, value, sequence_length, dim)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    diffusers.models.attention.CrossAttention.forward = forward_default


# FlashAttention based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main
# /memory_efficient_attention_pytorch/flash_attention.py LICENSE MIT
# https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE constants
EPSILON = 1e-6


# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# flash attention forwards and backwards
# https://arxiv.org/abs/2205.14135


class FlashAttentionFunction(torch.autograd.function.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """ Algorithm 2 in the paper """

        device = q.device
        dtype = q.dtype
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros(
            (*q.shape[:-1], 1), dtype=dtype, device=device)
        all_row_maxes = torch.full(
            (*q.shape[:-1], 1), max_neg_value, dtype=dtype, device=device)

        scale = (q.shape[-1] ** -0.5)

        if not exists(mask):
            mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
        else:
            mask = rearrange(mask, 'b n -> b 1 1 n')
            mask = mask.split(q_bucket_size, dim=-1)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            mask,
            all_row_sums.split(q_bucket_size, dim=-2),
            all_row_maxes.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum(
                    '... i d, ... j d -> ... i j', qc, kc) * scale

                if exists(row_mask):
                    attn_weights.masked_fill_(~row_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                             device=device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
                attn_weights -= block_row_maxes
                exp_weights = torch.exp(attn_weights)

                if exists(row_mask):
                    exp_weights.masked_fill_(~row_mask, 0.)

                block_row_sums = exp_weights.sum(
                    dim=-1, keepdims=True).clamp(min=EPSILON)

                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_values = einsum(
                    '... i j, ... j d -> ... i d', exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
                exp_block_row_max_diff = torch.exp(
                    block_row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + \
                               exp_block_row_max_diff * block_row_sums

                oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_(
                    (exp_block_row_max_diff / new_row_sums) * exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, all_row_sums, all_row_maxes)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """ Algorithm 4 in the paper """

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, l, m = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            do.split(q_bucket_size, dim=-2),
            mask,
            l.split(q_bucket_size, dim=-2),
            m.split(q_bucket_size, dim=-2),
            dq.split(q_bucket_size, dim=-2)
        )

        for ind, (qc, oc, doc, row_mask, lc, mc, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
                dk.split(k_bucket_size, dim=-2),
                dv.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum(
                    '... i d, ... j d -> ... i j', qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                             device=device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                exp_attn_weights = torch.exp(attn_weights - mc)

                if exists(row_mask):
                    exp_attn_weights.masked_fill_(~row_mask, 0.)

                p = exp_attn_weights / lc

                dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                D = (doc * oc).sum(dim=-1, keepdims=True)
                ds = p * scale * (dp - D)

                dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None


def replace_unet_cross_attn_to_flash_attention():
    print("Replace CrossAttention.forward to use FlashAttention")

    def forward_flash_attn(self, x, context=None, mask=None):
        q_bucket_size = 512
        k_bucket_size = 1024

        h = self.heads
        q = self.to_q(x)

        context = context if context is not None else x
        context = context.to(x.dtype)

        if hasattr(self, 'hypernetwork') and self.hypernetwork is not None:
            context_k, context_v = self.hypernetwork.forward(x, context)
            context_k = context_k.to(x.dtype)
            context_v = context_v.to(x.dtype)
        else:
            context_k = context
            context_v = context

        k = self.to_k(context_k)
        v = self.to_v(context_v)
        del context, x

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        out = FlashAttentionFunction.apply(q, k, v, mask, False,
                                           q_bucket_size, k_bucket_size)

        out = rearrange(out, 'b h n d -> b n (h d)')

        # diffusers 0.6.0
        if type(self.to_out) is torch.nn.Sequential:
            return self.to_out(out)

        # diffusers 0.7.0
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    diffusers.models.attention.CrossAttention.forward = forward_flash_attn


def replace_unet_cross_attn_to_xformers():
    print("Replace CrossAttention.forward to use xformers")
    try:
        import xformers.ops
    except ImportError:
        raise ImportError(
            "xformers not installed. Re-launch webui with --xformers.")

    def forward_xformers(self, x, context=None, mask=None):
        h = self.heads
        q_in = self.to_q(x)

        context = default(context, x)
        context = context.to(x.dtype)

        if hasattr(self, 'hypernetwork') and self.hypernetwork is not None:
            context_k, context_v = self.hypernetwork.forward(x, context)
            context_k = context_k.to(x.dtype)
            context_v = context_v.to(x.dtype)
        else:
            context_k = context
            context_v = context

        k_in = self.to_k(context_k)
        v_in = self.to_v(context_v)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b n h d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None)

        out = rearrange(out, 'b n h d -> b n (h d)', h=h)

        # diffusers 0.6.0
        if type(self.to_out) is torch.nn.Sequential:
            return self.to_out(out)

        # diffusers 0.7.0~
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    diffusers.models.attention.CrossAttention.forward = forward_xformers


def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
    pass


trans_ver = transformers.__version__
if int(trans_ver.split(".")[1]) > 19:
    print("Patching transformers to fix kwargs errors.")

    transformers.generation_utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs
