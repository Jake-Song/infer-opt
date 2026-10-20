"""Triton kernel for the one-token group-wise W8A16 matmul (a GEMV: weight matrix times one activation vector).

``y[n] = sum_k x[k] * q[n, k] * scale[n, k // group_size]`` with int8 ``q`` and fp32 accumulation. Batch-1 decode
reads every weight byte once, so the kernel only has to stream ``q`` at full HBM bandwidth. Each program owns
``BLOCK_N`` output rows and walks ``K`` in ``BLOCK_K`` steps over a ``[BLOCK_N, BLOCK_K // VEC, VEC]`` tile: every
thread loads ``VEC`` (16) contiguous int8 of one group in one 16-byte load, sums ``q * x`` over them in registers and
multiplies that partial sum by the group's scale, so the scale costs one load and one multiply per 16 weights.
``x`` and the scales are loaded through pointers of the same 3-D shape as ``q`` (stride 0 where they repeat), which
gives the three loads one register layout; with 2-D ``x``/scale loads Triton moved the partial sums through shared
memory every step. On an A100 this runs at the speed of a kernel that only reads ``q`` (within 2-8% on every
Qwen3.5-27B shape) and as fast as a per-channel-scale kernel. The previous form, ``acc += q * (x * scale)`` on a
``[BLOCK_N, group_size]`` tile, multiplied by the scale per element and was 10-25% slower. Building the fp32 values
with bit tricks instead of int-to-float conversions was slower in both forms.

Registered with ``torch.library.triton_op`` so torch.compile keeps it as one kernel (nothing is fused into it) and
CUDA graphs can capture it. The launch parameters are chosen from the shape (:func:`launch_config`), not by
``triton.autotune``: autotuned inside the compiled Qwen3.5-9B decode step an earlier kernel picked configurations that
made decode slower than bf16 (25 ms/token against 15).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

__all__ = ["w8a16_group_gemv"]


# (BLOCK_N, BLOCK_K, num_warps) by (N, K, group); overrides launch_config, e.g. for a tuning sweep
LAUNCH_CONFIGS: dict[tuple[int, int, int], tuple[int, int, int]] = {}


def _pow2_divisor(n: int, cap: int) -> int:
    """The largest power of two that divides ``n``, at most ``cap``."""
    return min(n & -n, cap)


def launch_config(n: int, k: int, group: int) -> tuple[int, int, int]:
    """``(BLOCK_N, BLOCK_K, num_warps)`` for an ``[n, k]`` weight.

    A sweep over every Linear shape of Qwen3.5-27B on an A100 80GB (fused projections included: 1024-34816 rows,
    K 5120-17408; BLOCK_N 1-16, BLOCK_K 128-1024, 2-8 warps) put (1, 1024, 2) first for all of them but the 1024-row
    k/v projections (4 warps there, 7.3 against 7.9 us). ``BLOCK_K`` falls back to the largest power of two dividing
    ``k`` when 1024 does not.
    """
    if (n, k, group) in LAUNCH_CONFIGS:
        return LAUNCH_CONFIGS[n, k, group]
    return 1, _pow2_divisor(k, 1024), 2


@triton.jit
def _w8a16_group_gemv_kernel(x_ptr, q_ptr, s_ptr, y_ptr, N, K, GROUP: tl.constexpr,
                             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, VEC: tl.constexpr):
    NV: tl.constexpr = BLOCK_K // VEC
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < N
    jv = tl.arange(0, NV)
    cols = jv[None, :, None] * VEC + tl.arange(0, VEC)[None, None, :]  # [1, NV, VEC]
    groups = K // GROUP
    q_ptrs = q_ptr + rows[:, None, None].to(tl.int64) * K + cols
    x_ptrs = x_ptr + rows[:, None, None] * 0 + cols  # q's shape, so x gets q's layout
    s_row = s_ptr + rows[:, None, None] * groups  # [BLOCK_N, 1, 1]
    s_step = (jv * VEC // GROUP)[None, :, None]  # [1, NV, 1]: one scale per VEC weights, when GROUP divides BLOCK_K
    acc = tl.zeros((BLOCK_N, NV), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs + k0).to(tl.float32)
        q = tl.load(q_ptrs + k0, mask=row_ok[:, None, None], other=0, eviction_policy="evict_first").to(tl.float32)
        if BLOCK_K % GROUP == 0:  # hoisted offsets: recomputing them every step was 13% slower on a 9B shape
            s_ptrs = s_row + k0 // GROUP + s_step
        else:
            s_ptrs = s_row + ((k0 + jv * VEC) // GROUP)[None, :, None]
        s = tl.sum(tl.load(s_ptrs, mask=row_ok[:, None, None], other=0.0).to(tl.float32), axis=2)
        acc += tl.sum(q * x, axis=2) * s
    tl.store(y_ptr + rows, tl.sum(acc, axis=1).to(y_ptr.dtype.element_ty), mask=row_ok)


@triton_op("infer_opt::w8a16_group_gemv", mutates_args=())
def w8a16_group_gemv(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x @ (q * scale)^T`` for one token: ``x`` has ``K`` elements (any leading shape of size 1), ``weight`` is int8
    ``[N, K]`` and ``scale`` ``[N, K // group_size]``. Returns ``x``'s leading shape with ``N`` outputs, in ``x.dtype``."""
    n, k = (int(d) for d in weight.shape)  # specializes the shapes: the launch configuration depends on them
    group = k // int(scale.shape[1])
    y = torch.empty(*x.shape[:-1], n, dtype=x.dtype, device=x.device)
    block_n, block_k, warps = launch_config(n, k, group)
    vec = _pow2_divisor(group, 16)  # one thread's contiguous weights must lie in one group
    wrap_triton(_w8a16_group_gemv_kernel)[(triton.cdiv(n, block_n),)](
        x.contiguous(), weight, scale, y, n, k, group, BLOCK_N=block_n, BLOCK_K=max(block_k, vec), VEC=vec,
        num_warps=warps, num_stages=2)
    return y
