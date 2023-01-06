# From https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6055/files

# original source:
#   https://github.com/AminRezaei0x443/memory-efficient-attention/blob/1bc0d9e6ac5f82ea43a375135c4e1d3896ee1694/memory_efficient_attention/attention_torch.py
# license:
#   unspecified
# credit:
#   Amin Rezaei (original author)
#   Alex Birch (optimized algorithm for 3D tensors, at the expense of removing bias, masking and callbacks)
# implementation of:
#   Self-attention Does Not Need O(n2) Memory":
#   https://arxiv.org/abs/2112.05682v2

from functools import partial

import psutil
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, NamedTuple, Protocol, List

from extensions.sd_dreambooth_extension.dreambooth import db_shared



def narrow_trunc(input, dim, start, length):
    return torch.narrow(input, dim, start, length if input.shape[dim] >= start + length else input.shape[dim] - start)

# Based on Birch-san's modified implementation of sub-quadratic attention from https://github.com/Birch-san/diffusers/pull/1
def sub_quad_attention_forward(self, x, context=None, mask=None):
    assert mask is None, "attention-mask not currently implemented for SubQuadraticCrossAttnProcessor."

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
    del context, context_k, context_v, x

    q = q.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)
    k = k.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)
    v = v.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)

    x = sub_quad_attention(q, k, v, q_chunk_size=db_shared.sub_quad_q_chunk_size, kv_chunk_size=db_shared.sub_quad_kv_chunk_size, chunk_threshold_bytes=db_shared.sub_quad_chunk_threshold, use_checkpoint=self.training)

    x = x.unflatten(0, (-1, h)).transpose(1,2).flatten(start_dim=2)

    out_proj, dropout = self.to_out
    x = out_proj(x)
    x = dropout(x)

    return x

def get_available_vram():
    if db_shared.device.type == 'cuda':
        stats = torch.cuda.memory_stats(db_shared.device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch
        return mem_free_total
    else:
        return psutil.virtual_memory().available

def sub_quad_attention(q, k, v, q_chunk_size=1024, kv_chunk_size=None, kv_chunk_size_min=None, chunk_threshold_bytes=None, use_checkpoint=True):
    bytes_per_token = torch.finfo(q.dtype).bits//8
    batch_x_heads, q_tokens, _ = q.shape
    _, k_tokens, _ = k.shape
    qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens

    available_vram = int(get_available_vram() * 0.9) if q.device.type == 'mps' else int(get_available_vram() * 0.7)

    if chunk_threshold_bytes is None:
        chunk_threshold_bytes = available_vram
    elif chunk_threshold_bytes == 0:
        chunk_threshold_bytes = None

    if kv_chunk_size_min is None:
        kv_chunk_size_min = chunk_threshold_bytes // (batch_x_heads * bytes_per_token * (k.shape[2] + v.shape[2]))
    elif kv_chunk_size_min == 0:
        kv_chunk_size_min = None

    if chunk_threshold_bytes is not None and qk_matmul_size_bytes <= chunk_threshold_bytes:
        # the big matmul fits into our memory limit; do everything in 1 chunk,
        # i.e. send it down the unchunked fast-path
        query_chunk_size = q_tokens
        kv_chunk_size = k_tokens

    return efficient_dot_product_attention(
        q,
        k,
        v,
        query_chunk_size=q_chunk_size,
        kv_chunk_size=kv_chunk_size,
        kv_chunk_size_min = kv_chunk_size_min,
        use_checkpoint=use_checkpoint,
    )


def sub_quad_attnblock_forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = sub_quad_attention(q, k, v, q_chunk_size=db_shared.sub_quad_q_chunk_size, kv_chunk_size=db_shared.sub_quad_kv_chunk_size, chunk_threshold_bytes=db_shared.sub_quad_chunk_threshold, use_checkpoint=self.training)
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    out = self.proj_out(out)
    return x + out

class AttnChunk(NamedTuple):
    exp_values: Tensor
    exp_weights_sum: Tensor
    max_score: Tensor

class SummarizeChunk(Protocol):
    @staticmethod
    def __call__(
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> AttnChunk: ...

class ComputeQueryChunkAttn(Protocol):
    @staticmethod
    def __call__(
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor: ...

def _summarize_chunk(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
) -> AttnChunk:
    attn_weights = torch.baddbmm(
        torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
        query,
        key.transpose(1,2),
        alpha=scale,
        beta=0,
    )
    max_score, _ = torch.max(attn_weights, -1, keepdim=True)
    max_score = max_score.detach()
    exp_weights = torch.exp(attn_weights - max_score)
    exp_values = torch.bmm(exp_weights, value)
    max_score = max_score.squeeze(-1)
    return AttnChunk(exp_values, exp_weights.sum(dim=-1), max_score)

def _query_chunk_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    summarize_chunk: SummarizeChunk,
    kv_chunk_size: int,
) -> Tensor:
    batch_x_heads, k_tokens, k_channels_per_head = key.shape
    _, _, v_channels_per_head = value.shape

    def chunk_scanner(chunk_idx: int) -> AttnChunk:
        key_chunk = narrow_trunc(
            key,
            1,
            chunk_idx,
            kv_chunk_size
        )
        value_chunk = narrow_trunc(
            value,
            1,
            chunk_idx,
            kv_chunk_size
        )
        return summarize_chunk(query, key_chunk, value_chunk)

    chunks: List[AttnChunk] = [
        chunk_scanner(chunk) for chunk in torch.arange(0, k_tokens, kv_chunk_size)
    ]
    acc_chunk = AttnChunk(*map(torch.stack, zip(*chunks)))
    chunk_values, chunk_weights, chunk_max = acc_chunk

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)
    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    return all_values / all_weights

# TODO: refactor CrossAttention#get_attention_scores to share code with this
def _get_attention_scores_no_kv_chunking(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
) -> Tensor:
    attn_scores = torch.baddbmm(
        torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
        query,
        key.transpose(1,2),
        alpha=scale,
        beta=0,
    )
    attn_probs = attn_scores.softmax(dim=-1)
    del attn_scores
    hidden_states_slice = torch.bmm(attn_probs, value)
    return hidden_states_slice

class ScannedChunk(NamedTuple):
    chunk_idx: int
    attn_chunk: AttnChunk

def efficient_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_chunk_size=1024,
    kv_chunk_size: Optional[int] = None,
    kv_chunk_size_min: Optional[int] = None,
    use_checkpoint=True,
):
    """Computes efficient dot-product attention given query, key, and value.
      This is efficient version of attention presented in
      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
      Args:
        query: queries for calculating attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        key: keys for calculating attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        value: values to be used in attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        query_chunk_size: int: query chunks size
        kv_chunk_size: Optional[int]: key/value chunks size. if None: defaults to sqrt(key_tokens)
        kv_chunk_size_min: Optional[int]: key/value minimum chunk size. only considered when kv_chunk_size is None. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).
        use_checkpoint: bool: whether to use checkpointing (recommended True for training, False for inference)
      Returns:
        Output of shape `[batch * num_heads, query_tokens, channels_per_head]`.
      """
    batch_x_heads, q_tokens, q_channels_per_head = query.shape
    _, k_tokens, _ = key.shape
    scale = q_channels_per_head ** -0.5

    kv_chunk_size = min(kv_chunk_size or int(math.sqrt(k_tokens)), k_tokens)
    if kv_chunk_size_min is not None:
        kv_chunk_size = max(kv_chunk_size, kv_chunk_size_min)

    def get_query_chunk(chunk_idx: int) -> Tensor:
        return narrow_trunc(
            query,
            1,
            chunk_idx,
            min(query_chunk_size, q_tokens)
        )

    summarize_chunk: SummarizeChunk = partial(_summarize_chunk, scale=scale)
    summarize_chunk: SummarizeChunk = partial(checkpoint, summarize_chunk) if use_checkpoint else summarize_chunk
    compute_query_chunk_attn: ComputeQueryChunkAttn = partial(
        _get_attention_scores_no_kv_chunking,
        scale=scale
    ) if k_tokens <= kv_chunk_size else (
        # fast-path for when there's just 1 key-value chunk per query chunk (this is just sliced attention btw)
        partial(
            _query_chunk_attention,
            kv_chunk_size=kv_chunk_size,
            summarize_chunk=summarize_chunk,
        )
    )

    if q_tokens <= query_chunk_size:
        # fast-path for when there's just 1 query chunk
        return compute_query_chunk_attn(
            query=query,
            key=key,
            value=value,
        )

    # TODO: maybe we should use torch.empty_like(query) to allocate storage in-advance,
    # and pass slices to be mutated, instead of torch.cat()ing the returned slices
    res = torch.cat([
        compute_query_chunk_attn(
            query=get_query_chunk(chunk),
            key=key,
            value=value,
        ) for chunk in torch.arange(0, q_tokens, query_chunk_size)
    ], dim=1)
    return res