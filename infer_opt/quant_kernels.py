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

:func:`w8a16_group_gemv_rows` takes a few activation rows at once (verifying 2-8 drafted tokens) and still reads the
weight once, in one of two forms (:func:`rows_launch_config`). The tensor-core form: the int8 values are exact in
fp16/bf16, so one ``tl.dot`` per scale group of columns gives the group's partial sums in fp32, and the scale multiplies
those. The multiply-add form is the one-row kernel with a leading dimension of rows; with one output row per program
every program reads all the rows of ``x`` (2 bytes each per weight byte), which made 2 and 4 rows 1.6x and 2.4x slower
than one on the RTX 2080 SUPER, so it takes several output rows per program.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

__all__ = ["w8a16_group_gemv", "w8a16_group_gemv_rows", "MAX_ROWS"]

MAX_ROWS = 8  # w8a16_group_gemv_rows takes up to this many activation rows


# (BLOCK_N, BLOCK_K, num_warps) by (N, K, group); overrides launch_config, e.g. for a tuning sweep
LAUNCH_CONFIGS: dict[tuple[int, int, int], tuple[int, int, int]] = {}
# w8a16_group_gemv_rows by (rows, N, K, group): ("dot", BLOCK_N, num_warps, num_stages, split-K) for the tensor-core
# form or ("fma", BLOCK_N, BLOCK_K, num_warps) for the fp32 multiply-add form; overrides rows_launch_config
ROWS_LAUNCH_CONFIGS: dict[tuple[int, int, int, int], tuple] = {}


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



@triton.jit
def _w8a16_group_gemv_rows_kernel(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, GROUP: tl.constexpr, BLOCK_M: tl.constexpr,
                                  BLOCK_N: tl.constexpr, SPLIT: tl.constexpr):
    # one group of columns per step: [BLOCK_M, GROUP] x [GROUP, BLOCK_N] on the tensor cores, then the group's scale.
    # With SPLIT > 1 program (pid, s) takes the s-th of SPLIT equal runs of groups and writes fp32 partials [SPLIT, M, N].
    pid = tl.program_id(0)
    split = tl.program_id(1)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < N
    m = tl.arange(0, BLOCK_M)
    m_ok = m < M
    cols = tl.arange(0, GROUP)
    groups = K // GROUP
    x_ptrs = x_ptr + m[:, None] * K + cols[None, :]
    q_ptrs = q_ptr + rows[:, None].to(tl.int64) * K + cols[None, :]
    s_ptrs = s_ptr + rows * groups
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    per_split = groups // SPLIT
    for g in range(split * per_split, (split + 1) * per_split):
        x = tl.load(x_ptrs + g * GROUP, mask=m_ok[:, None], other=0.0)
        q = tl.load(q_ptrs + g * GROUP, mask=row_ok[:, None], other=0, eviction_policy="evict_first")
        s = tl.load(s_ptrs + g, mask=row_ok, other=0.0).to(tl.float32)
        acc += tl.dot(x, tl.trans(q.to(x.dtype))) * s[None, :]  # int8 values are exact in fp16/bf16
    y_ptrs = y_ptr + split * M * N + m[:, None] * N + rows[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=m_ok[:, None] & row_ok[None, :])


@triton.jit
def _w8a16_group_gemv_rows_fma_kernel(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, GROUP: tl.constexpr,
                                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                      VEC: tl.constexpr):
    # the one-row kernel with BLOCK_M activation rows on x, the accumulator and y: each q tile is multiplied with
    # every row in fp32. BLOCK_N > 1 output rows per program share the x loads (M rows of x per weight row otherwise).
    NV: tl.constexpr = BLOCK_K // VEC
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < N
    m = tl.arange(0, BLOCK_M)
    jv = tl.arange(0, NV)
    cols = jv[None, :, None] * VEC + tl.arange(0, VEC)[None, None, :]  # [1, NV, VEC]
    groups = K // GROUP
    q_ptrs = q_ptr + rows[:, None, None].to(tl.int64) * K + cols
    x_ptrs = x_ptr + m[:, None, None, None] * K + rows[None, :, None, None] * 0 + cols[None]  # [BLOCK_M, *q's shape]
    s_row = s_ptr + rows[:, None, None] * groups
    acc = tl.zeros((BLOCK_M, BLOCK_N, NV), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs + k0, mask=(m < M)[:, None, None, None], other=0.0).to(tl.float32)
        q = tl.load(q_ptrs + k0, mask=row_ok[:, None, None], other=0, eviction_policy="evict_first").to(tl.float32)
        s_ptrs = s_row + ((k0 + jv * VEC) // GROUP)[None, :, None]
        s = tl.sum(tl.load(s_ptrs, mask=row_ok[:, None, None], other=0.0).to(tl.float32), axis=2)
        acc += tl.sum(q[None] * x, axis=3) * s[None]
    y_ptrs = y_ptr + m[:, None] * N + rows[None, :]
    tl.store(y_ptrs, tl.sum(acc, axis=2).to(y_ptr.dtype.element_ty), mask=(m < M)[:, None] & row_ok[None, :])

def rows_launch_config(m: int, n: int, k: int, group: int, device: torch.device) -> tuple:
    """Kernel form and launch parameters of :func:`w8a16_group_gemv_rows` (see ``ROWS_LAUNCH_CONFIGS``).

    From sm80 on the tensor-core form. A sweep over the int8 Qwen3.5-27B shapes (fused, MTP head and lm_head
    included; 2-4 rows, A100 80GB) put it at 1.06-1.16x the one-row kernel on the shapes with many output rows, without
    splitting K. The 5120-row shapes (down_proj, o_proj, the MTP fc) have too few programs for HBM: 1.50-1.56x unsplit,
    1.13-1.26x with K split in 4 (64 rows per program, 8 warps). Before sm80 Triton runs ``tl.dot`` without the tensor
    cores (25x slower than one row on the RTX 2080 SUPER) and the multiply-add form with 4 output rows per program came
    within 1.0-1.2x of one row (2-4 rows); on the A100 that form was 2-7x slower than the tensor-core one."""
    if (m, n, k, group) in ROWS_LAUNCH_CONFIGS:
        return ROWS_LAUNCH_CONFIGS[m, n, k, group]
    if torch.cuda.get_device_capability(device)[0] >= 8:
        if n <= 8192 and (k // group) % 4 == 0:
            return "dot", 64, 8, 3, 4
        return "dot", 32, 4, 4, 1
    return "fma", 4, _pow2_divisor(k, 256), 4


def rows_supported(group: int) -> bool:
    """The multi-row kernel takes one group per ``tl.dot``: a power of two of at least 16 columns."""
    return group >= 16 and group & (group - 1) == 0


@triton_op("infer_opt::w8a16_group_gemv_rows", mutates_args=())
def w8a16_group_gemv_rows(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x @ (q * scale)^T`` for ``x`` with 2..``MAX_ROWS`` rows of ``K`` elements (any leading shape), group size a
    power of two >= 16. The rows are padded to 16 for ``tl.dot``; the weight is still read once."""
    n, k = (int(d) for d in weight.shape)
    group = k // int(scale.shape[1])
    m = int(x.numel()) // k
    shape = (*x.shape[:-1], n)
    config = rows_launch_config(m, n, k, group, x.device)
    form, block_n, a, b = config[:4]
    if form == "fma":
        y = torch.empty(shape, dtype=x.dtype, device=x.device)
        vec = _pow2_divisor(group, 16)
        wrap_triton(_w8a16_group_gemv_rows_fma_kernel)[(triton.cdiv(n, block_n),)](
            x.contiguous(), weight, scale, y, m, n, k, group, BLOCK_M=triton.next_power_of_2(m), BLOCK_N=block_n,
            BLOCK_K=max(a, vec), VEC=vec, num_warps=b, num_stages=2)
        return y
    split = config[4] if len(config) > 4 else 1
    if split == 1:
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
    else:  # fp32 partial sums, one slice per run of groups
        out = torch.empty(split, *shape, dtype=torch.float32, device=x.device)
    wrap_triton(_w8a16_group_gemv_rows_kernel)[(triton.cdiv(n, block_n), split)](
        x.contiguous(), weight, scale, out, m, n, k, group, BLOCK_M=16, BLOCK_N=block_n, SPLIT=split, num_warps=a,
        num_stages=b)
    return out if split == 1 else out.sum(0).to(x.dtype)
