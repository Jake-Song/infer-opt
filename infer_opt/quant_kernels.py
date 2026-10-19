"""Triton kernel for the one-token group-wise W8A16 matmul (a GEMV: weight matrix times one activation vector).

``y[n] = sum_k x[k] * q[n, k] * scale[n, k // group_size]`` with int8 ``q`` and fp32 accumulation. Batch-1 decode
reads every weight byte once, so the kernel only has to stream ``q`` at full HBM bandwidth: each program owns
``BLOCK_N`` output rows and walks ``K`` one group at a time (``UNROLL`` groups per loop iteration), accumulating a
``[BLOCK_N, group_size]`` tile that is reduced once at the end. Each step loads the group's scale once per row and
multiplies it into ``x``. Reading the scale per element (``scale[n, k // group_size]`` for every ``k``) was L1 hits
but cost enough instructions to cap an A100 at ~580 GB/s, about the bf16 time; this layout reaches ~1.1 TB/s.

Registered with ``torch.library.triton_op`` so torch.compile keeps it as one kernel (nothing is fused into it) and
CUDA graphs can capture it. The launch parameters are chosen from the shape (:func:`launch_config`), not by
``triton.autotune``: autotuned inside the compiled Qwen3.5-9B decode step it picked configurations that made decode
slower than bf16 (25 ms/token against 15), while these fixed ones give 10.2 ms/token.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

__all__ = ["w8a16_group_gemv"]


# (BLOCK_N, UNROLL, num_warps, num_stages) by (N, K, group); overrides launch_config, e.g. for a tuning sweep
LAUNCH_CONFIGS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}


def launch_config(n: int, k: int, group: int) -> tuple[int, int, int, int]:
    """``(BLOCK_N, UNROLL, num_warps, num_stages)`` for an ``[n, k]`` weight.

    From a sweep over every Linear shape of Qwen3.5-9B on an A100 (scripts/tune_qwen3_5_group_gemv.py): 8 warps
    always, stages irrelevant, and the best BLOCK_N keeps the grid at roughly 256-512 programs (a few per SM). The
    rule gives the measured best for five of the six shapes, each within about 5% of the per-channel INT8 time
    (measured separately).
    """
    if (n, k, group) in LAUNCH_CONFIGS:
        return LAUNCH_CONFIGS[n, k, group]
    block_n = min(max(triton.next_power_of_2(triton.cdiv(n, 512)), 4), 32)
    return block_n, (4 if block_n >= 16 else 8), 8, 3


@triton.jit
def _w8a16_group_gemv_kernel(x_ptr, q_ptr, s_ptr, y_ptr, N, K, GROUP: tl.constexpr,
                             BLOCK_N: tl.constexpr, UNROLL: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < N
    cols = tl.arange(0, GROUP)
    groups = K // GROUP
    q_row = q_ptr + rows[:, None].to(tl.int64) * K
    s_row = s_ptr + rows * groups
    acc = tl.zeros((BLOCK_N, GROUP), dtype=tl.float32)
    for g0 in range(0, groups, UNROLL):
        for u in tl.static_range(UNROLL):
            g = g0 + u
            ok = g < groups
            k = g * GROUP + cols
            x = tl.load(x_ptr + k, mask=ok, other=0.0).to(tl.float32)
            q = tl.load(q_row + k[None, :], mask=row_ok[:, None] & ok, other=0,
                        eviction_policy="evict_first").to(tl.float32)
            s = tl.load(s_row + g, mask=row_ok & ok, other=0.0).to(tl.float32)
            acc += q * (x[None, :] * s[:, None])
    tl.store(y_ptr + rows, tl.sum(acc, axis=1).to(y_ptr.dtype.element_ty), mask=row_ok)


@triton_op("infer_opt::w8a16_group_gemv", mutates_args=())
def w8a16_group_gemv(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x @ (q * scale)^T`` for one token: ``x`` has ``K`` elements (any leading shape of size 1), ``weight`` is int8
    ``[N, K]`` and ``scale`` ``[N, K // group_size]``. Returns ``x``'s leading shape with ``N`` outputs, in ``x.dtype``."""
    n, k = (int(d) for d in weight.shape)  # specializes the shapes: the launch configuration depends on them
    group = k // int(scale.shape[1])
    y = torch.empty(*x.shape[:-1], n, dtype=x.dtype, device=x.device)
    block_n, unroll, warps, stages = launch_config(n, k, group)
    wrap_triton(_w8a16_group_gemv_kernel)[(triton.cdiv(n, block_n),)](
        x.contiguous(), weight, scale, y, n, k, group, BLOCK_N=block_n, UNROLL=unroll, num_warps=warps, num_stages=stages)
    return y
