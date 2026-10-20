"""Triton kernel for one-token (decode) attention over a fixed-size KV cache with grouped query heads.

The query heads that share a KV head attend together: one program takes one KV head and one ``CHUNK``-key slice of
the cache, computes the ``[G, CHUNK]`` scores of its ``G`` query heads with one ``tl.dot`` (``G`` padded to 16 rows),
and writes an online-softmax partial (running max, sum of exponentials, unnormalized output) for that slice; the
partials are merged in PyTorch (one small fused kernel under torch.compile). Keys past ``position`` (the unfilled tail
of the buffer) are masked, and slices that lie entirely past it skip their loads.

On Qwen3.5-27B (24 query / 4 KV heads, head_dim 256, 1 216-slot cache, A100) the plain-PyTorch form of the same math
(fp32 broadcast-multiply-and-sum, which inductor turns into three reduction kernels) took 39 us per layer; this kernel
takes 7.8 us and the merge ~7 us more (two small kernels, one of them shared with the output gate).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

__all__ = ["decode_attention"]

CHUNK = 64


@triton.jit
def _decode_attention_kernel(q_ptr, k_ptr, v_ptr, pos_ptr, m_ptr, l_ptr, o_ptr, L, scale,
                             G: tl.constexpr, G_PAD: tl.constexpr, D: tl.constexpr, CHUNK: tl.constexpr):
    head = tl.program_id(0)  # batch * KV heads
    split = tl.program_id(1)
    n_split = tl.num_programs(1)
    g = tl.arange(0, G_PAD)
    d = tl.arange(0, D)
    keys = split * CHUNK + tl.arange(0, CHUNK)
    position = tl.load(pos_ptr)
    part = (head * n_split + split) * G + g  # row of this (head, split, query head) in the partial buffers
    g_ok = g < G
    if split * CHUNK > position:  # nothing filled in this slice
        tl.store(m_ptr + part, tl.full((G_PAD,), float("-inf"), tl.float32), mask=g_ok)
        tl.store(l_ptr + part, tl.zeros((G_PAD,), tl.float32), mask=g_ok)
        tl.store(o_ptr + part[:, None] * D + d[None, :], tl.zeros((G_PAD, D), tl.float32), mask=g_ok[:, None])
        return
    q = tl.load(q_ptr + (head * G + g)[:, None] * D + d[None, :], mask=g_ok[:, None], other=0.0)
    kv_ok = (keys < L)[:, None]
    kv_off = (head.to(tl.int64) * L + keys)[:, None] * D + d[None, :]
    k = tl.load(k_ptr + kv_off, mask=kv_ok, other=0.0)
    scores = tl.dot(q, tl.trans(k)) * scale  # [G_PAD, CHUNK], fp32
    scores = tl.where((keys <= position)[None, :], scores, float("-inf"))
    m = tl.max(scores, axis=1)
    p = tl.exp(scores - m[:, None])
    v = tl.load(v_ptr + kv_off, mask=kv_ok, other=0.0)
    o = tl.dot(p.to(v.dtype), v)  # [G_PAD, D], fp32
    tl.store(m_ptr + part, m, mask=g_ok)
    tl.store(l_ptr + part, tl.sum(p, axis=1), mask=g_ok)
    tl.store(o_ptr + part[:, None] * D + d[None, :], o, mask=g_ok[:, None])


@triton_op("infer_opt::decode_attention_partials", mutates_args=())
def _decode_attention_partials(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, position: torch.Tensor,
                               scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, kv_heads, groups, dim = (int(s) for s in query.shape)
    length = int(keys.shape[2])
    n_split = triton.cdiv(length, CHUNK)
    shape = (batch, kv_heads, n_split, groups)
    m = torch.empty(shape, dtype=torch.float32, device=query.device)
    l = torch.empty(shape, dtype=torch.float32, device=query.device)
    o = torch.empty(*shape, dim, dtype=torch.float32, device=query.device)
    wrap_triton(_decode_attention_kernel)[(batch * kv_heads, n_split)](
        query.contiguous(), keys, values, position, m, l, o, length, scale,
        G=groups, G_PAD=max(16, triton.next_power_of_2(groups)), D=dim, CHUNK=CHUNK, num_warps=4)
    return m, l, o


def decode_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, position: torch.Tensor,
                     scale: float) -> torch.Tensor:
    """Softmax attention of one query token per head over a ``[B, KV heads, L, D]`` cache, keys ``0..position``.

    ``query`` is ``[B, KV heads, G, D]`` (the ``G`` query heads of each KV head), ``position`` a one-element integer
    tensor on the device. Returns ``[B, KV heads, G, D]`` in ``query.dtype``. ``D`` must be a power of two.
    """
    m, l, o = _decode_attention_partials(query, keys, values, position, scale)
    weight = torch.exp(m - m.amax(dim=2, keepdim=True))  # [B, KV heads, splits, G]; 0 for empty splits
    out = (weight.unsqueeze(-1) * o).sum(2) / (weight * l).sum(2).unsqueeze(-1)
    return out.to(query.dtype)
