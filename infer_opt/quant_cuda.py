"""CUDA C++ form of the one-token group-wise W8A16 GEMV, to compare with the Triton kernel in :mod:`infer_opt.quant_kernels`.

The kernel is ``csrc/w8a16_gemv.cu`` (see its header comment): ``ROWS`` output rows per block of ``THREADS`` threads,
one streaming 16-byte weight load per thread and chunk, ``UNROLL`` chunks' loads issued before any arithmetic, one
scale multiply per 16 weights, warp-shuffle reduction. It is built on first use with
``torch.utils.cpp_extension.load`` (nvcc and ninja, the CUDA major version of torch; ``CUDA_HOME`` picks the toolkit)
and registered as the custom op ``infer_opt::w8a16_group_gemv_cuda``, which torch.compile keeps as one opaque call and
CUDA graphs can capture.

``int8_linear`` runs it instead of the Triton kernel when ``infer_opt.quant.INT8_GEMV_BACKEND`` is ``"cuda"`` (or the
environment variable ``INFER_OPT_INT8_GEMV=cuda``) and :func:`supported` holds for the shape.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

__all__ = ["w8a16_group_gemv_cuda", "cuda_launch_config", "supported", "CUDA_LAUNCH_CONFIGS", "CONFIGS"]

# the (ROWS, THREADS, UNROLL) instantiations compiled into csrc/w8a16_gemv.cu
CONFIGS = [(r, t, u) for r in (1, 2, 4) for t in (32, 64, 128, 256) for u in (1, 2, 4, 8)]
# (ROWS, THREADS, UNROLL) by (N, K, group); overrides cuda_launch_config, e.g. for a tuning sweep
CUDA_LAUNCH_CONFIGS: dict[tuple[int, int, int], tuple[int, int, int]] = {}


@functools.cache
def _extension():
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("csrc") / "w8a16_gemv.cu"
    return load(name="infer_opt_w8a16_gemv", sources=[str(source)], extra_cuda_cflags=["-O3", "-lineinfo"])


def supported(n: int, k: int, group: int) -> bool:
    """The kernel takes 16 weights of one group per load: ``K`` and the group size multiples of 16."""
    return k % 16 == 0 and group % 16 == 0 and k % group == 0


def cuda_launch_config(n: int, k: int, group: int) -> tuple[int, int, int]:
    """``(ROWS, THREADS, UNROLL)`` for an ``[n, k]`` weight.

    A sweep of the 48 compiled configurations over the fused Qwen3.5-27B shapes on an A100 80GB (lm_head included;
    ``bench_int8_gemv_cuda.py``) put (2, 32, 1) within 4% of the best on every shape and 1% above the per-shape best
    in total: one warp per block, two rows per warp, two 16-byte weight loads in flight per thread. Deeper unrolling or
    more threads per block was slower everywhere (the first guess, (1, 64, 4), took 1.32x as long). The rule is the
    A100's: on the RTX 2080 SUPER (2, 32, 1) takes 1.4x the per-shape best there."""
    if (n, k, group) in CUDA_LAUNCH_CONFIGS:
        return CUDA_LAUNCH_CONFIGS[n, k, group]
    return 2, 32, 1


@torch.library.custom_op("infer_opt::w8a16_group_gemv_cuda", mutates_args=())
def w8a16_group_gemv_cuda(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x @ (q * scale)^T`` for one token, as :func:`infer_opt.quant_kernels.w8a16_group_gemv`: ``x`` has ``K``
    elements (any leading shape of size 1), ``weight`` is int8 ``[N, K]`` and ``scale`` ``[N, K // group_size]`` in
    ``x.dtype``. Returns ``x``'s leading shape with ``N`` outputs."""
    n, k = (int(d) for d in weight.shape)
    rows, threads, unroll = cuda_launch_config(n, k, k // int(scale.shape[1]))
    return _extension().w8a16_group_gemv(x.contiguous(), weight, scale, rows, threads, unroll)


@w8a16_group_gemv_cuda.register_fake
def _(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], weight.shape[0])
