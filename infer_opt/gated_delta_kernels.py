"""Triton kernel for verifying drafted tokens through a GatedDeltaNet layer: ``T`` decode steps of the gated delta rule
in one launch, keeping the state after every token.

Sequential decoding runs one step per token (FLA's ``fused_recurrent_gated_delta_rule``). Verifying ``T`` drafts the
same way meant ``T`` launches per layer, each reading and writing the whole fp32 state, plus a copy of every step's
state into the cache's spec buffer for :meth:`~infer_opt.qwen3_5.Qwen3_5Cache.commit` (on the int8 Qwen3.5-27B, k=3:
5.2 ms of a 33.9 ms round against 2.0 ms of a step). Here each program loads its block of the state once, runs the
``T`` steps in registers and stores the state after each one straight into ``states[t]``: 1 read and ``T`` writes of
the state.

The math, the launch shape and the order of operations are FLA's (``fla/ops/gated_delta_rule/fused_recurrent.py``,
fla-core 0.6.0): per program a ``[K, BV]`` block of the ``[K, V]`` state of one value head, ``BV = 8``, one warp; the
query/key head of value head ``h`` is ``h // (HV // H)`` (no ``repeat_interleave``); q and k are l2-normalized as
``x / sqrt(sum(x * x) + 1e-6)`` and q is scaled by ``K ** -0.5``. Per token:
``S *= exp(g)``, ``u = beta * (v - k^T S)``, ``S += k u^T``, ``o = q^T S``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

__all__ = ["gated_delta_verify"]

BLOCK_V = 8
NUM_WARPS = 1


@triton.jit
def _gated_delta_verify_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, h0_ptr, states_ptr, o_ptr, scale, NH,
                               T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                               V: tl.constexpr, BV: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = V // BV
    i_v, i_nh = pid % NV, pid // NV  # i_nh: batch * HV + value head
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    block = o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(h0_ptr + i_nh * K * V + block).to(tl.float32)
    for t in tl.static_range(T):
        row = i_n * T + t  # token t of sequence i_n in the [B, T, ...] inputs
        b_q = tl.load(q_ptr + (row * H + i_h) * K + o_k).to(tl.float32)
        b_k = tl.load(k_ptr + (row * H + i_h) * K + o_k).to(tl.float32)
        b_v = tl.load(v_ptr + (row * HV + i_hv) * V + o_v).to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_beta = tl.load(beta_ptr + row * HV + i_hv).to(tl.float32)
        b_h *= tl.exp(tl.load(g_ptr + row * HV + i_hv).to(tl.float32))
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(o_ptr + (row * HV + i_hv) * V + o_v, b_o.to(o_ptr.dtype.element_ty))
        tl.store(states_ptr + (t * NH + i_nh) * K * V + block, b_h)  # states: [T, B, HV, K, V]


@triton_op("infer_opt::gated_delta_verify", mutates_args={"states"})
def gated_delta_verify(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor,
                       beta: torch.Tensor, initial_state: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """``T`` gated-delta-rule steps from ``initial_state`` (``[B, HV, K, V]`` fp32, read only).

    ``query``/``key`` are ``[B, T, H, K]`` (``H`` divides the ``HV`` value heads), ``value`` ``[B, T, HV, V]``,
    ``g`` (log decay) and ``beta`` ``[B, T, HV]``. The state after token ``t`` is written to ``states[t]``
    (``[T, B, HV, K, V]`` fp32). Returns the outputs ``[B, T, HV, V]`` in ``value.dtype``. ``K`` and ``V`` must be
    powers of two, ``V`` a multiple of 8.
    """
    batch, tokens, heads, dim_k = (int(s) for s in query.shape)
    value_heads, dim_v = int(value.shape[2]), int(value.shape[3])
    out = torch.empty(value.shape, dtype=value.dtype, device=value.device)
    nh = batch * value_heads
    wrap_triton(_gated_delta_verify_kernel)[(nh * (dim_v // BLOCK_V),)](
        query.contiguous(), key.contiguous(), value.contiguous(), g.contiguous(), beta.contiguous(), initial_state,
        states, out, dim_k ** -0.5, nh, T=tokens, H=heads, HV=value_heads, K=dim_k, V=dim_v, BV=BLOCK_V,
        num_warps=NUM_WARPS, num_stages=3)
    return out
