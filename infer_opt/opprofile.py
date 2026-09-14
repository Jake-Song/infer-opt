"""Per-operation breakdown of a forward step.

Chapter 1 costed a step as a single number. This module splits that number by
the operation that spent it, which is what turns "decode is memory-bound" into
"decode spends 59% of its device time in attention".

The same *measured* versus *analytic* split as :mod:`infer_opt.profiling`
applies, and matters more here: :func:`profile_stages` observes where time
went, while :func:`stage_flops` and :func:`stage_bytes` count what each stage
was obliged to do. A stage whose time share dwarfs its FLOP share is not
compute-bound, and that comparison is the entire diagnostic.

Both halves are keyed by :data:`infer_opt.regions.REGION_NAMES`, so a measured
share and an analytic share line up in one table.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from .config import ModelConfig
from .profiling import kv_cache_bytes, timeit, transformer_flops
from .regions import REGION_NAMES, profiled_regions, region_of

__all__ = [
    "StageBreakdown",
    "profile_stages",
    "warmup_device",
    "stage_flops",
    "stage_bytes",
]


@dataclass(frozen=True)
class StageBreakdown:
    """Time attributed to each named region of one step, in milliseconds.

    ``stages`` sums to ``device_ms``, not to ``wall_ms``. The difference is
    :attr:`unattributed_ms` -- time the device spent idle while the host was
    still issuing work. At batch 1 that gap dominates, which is why a
    per-operation ratio measured there describes Python, not the GPU.
    """

    stages: dict[str, float]
    device_ms: float
    wall_ms: float
    steps: int

    @property
    def shares(self) -> dict[str, float]:
        """Each stage's fraction of attributed device time."""
        if self.device_ms <= 0:
            return {name: 0.0 for name in self.stages}
        return {name: ms / self.device_ms for name, ms in self.stages.items()}

    @property
    def unattributed_ms(self) -> float:
        """Wall time no region accounts for: launch gap and host overhead."""
        return max(0.0, self.wall_ms - self.device_ms)

    @property
    def busiest(self) -> str:
        """Name of the stage holding the largest share of device time."""
        return max(self.stages, key=lambda name: self.stages[name])


def _attribute(profiled: profile, *, device: torch.device) -> dict[str, float]:
    """Sum each region's span durations out of a finished profile.

    ``key_averages`` reports every annotation twice -- once as the host-side
    scope that opened it and once as a device-side range -- and the two carry
    different totals, so adding both double-counts. The host-side row is the
    span itself and the one to trust; its ``device_time_total`` covers the
    kernels launched inside it.
    """
    on_cuda = device.type == "cuda"
    totals = {name: 0.0 for name in REGION_NAMES}
    for entry in profiled.key_averages():
        name = region_of(entry.key)
        if name is None or entry.device_type != DeviceType.CPU:
            continue
        if name not in totals:
            raise ValueError(f"profile contains an unknown region {name!r}")
        micros = entry.device_time_total if on_cuda else entry.cpu_time_total
        totals[name] += micros / 1000
    return totals


@contextlib.contextmanager
def _quiet_stderr(enabled: bool) -> Iterator[None]:
    """Silence the profiler's USDT chatter on the real stderr file descriptor.

    Kineto announces every start and stop from C++, so redirecting
    ``sys.stderr`` is not enough and the notes land in the middle of a
    notebook's output. Only the profiled region is covered, and the descriptor
    is restored even if the body raises.
    """
    if not enabled:
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def warmup_device(device: torch.device, *, seconds: float = 2.0, size: int = 2048) -> None:
    """Spin the GPU until its clocks leave the idle state.

    A card sitting at idle runs its SMs at a fraction of their boost clock and
    takes a moment to ramp. Measure in that window and every early number comes
    out several times too slow -- not because the code is slow, but because the
    hardware had not woken up. Cheap insurance before the first measurement of
    a session; a no-op on CPU.
    """
    if device.type != "cuda":
        return
    left = torch.randn(size, size, device=device, dtype=torch.float16)
    right = torch.randn_like(left)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(10):
            left @ right
        torch.cuda.synchronize(device)
    del left, right
    torch.cuda.empty_cache()


def profile_stages(
    step: Callable[[], object],
    *,
    device: torch.device,
    warmup: int = 3,
    iters: int = 8,
    quiet: bool = True,
) -> StageBreakdown:
    """Attribute one step's time to the :func:`~infer_opt.regions.region` spans it runs.

    ``step`` must be callable repeatedly and is timed twice: once unprofiled
    for honest wall time, once under the profiler for the breakdown. Profiling
    adds host overhead, so mixing the two would inflate the wall clock.

    Times are reported per step, not per batch of ``iters``.
    """
    if iters <= 0 or warmup < 0:
        raise ValueError("iters must be positive and warmup cannot be negative")
    wall = timeit(step, device=device, warmup=warmup, iters=iters)

    with profiled_regions():
        for _ in range(warmup):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with _quiet_stderr(quiet), profile(activities=activities) as profiled:
            for _ in range(iters):
                step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    totals = {name: ms / iters for name, ms in _attribute(profiled, device=device).items()}
    return StageBreakdown(
        stages=totals,
        device_ms=sum(totals.values()),
        wall_ms=wall.ms,
        steps=iters,
    )


def stage_flops(
    config: ModelConfig, *, batch: int, new_tokens: int, context_len: int
) -> dict[str, int]:
    """FLOPs each region is obliged to perform, summing to ``transformer_flops``.

    Embedding, norms, RoPE and the cache moves count **zero** here, and that is
    not an oversight: a gather, a normalisation and a rotation do a negligible
    amount of arithmetic. When such a stage nonetheless owns a large slice of
    :func:`profile_stages`, its cost is memory traffic or kernel launches --
    which is precisely what the side-by-side comparison is meant to expose.
    """
    coarse = transformer_flops(
        config, batch=batch, new_tokens=new_tokens, context_len=context_len
    )
    hidden, kv_width = config.hidden_size, config.n_kv_heads * config.head_dim
    tokens = batch * new_tokens
    per_layer = config.n_layers * tokens
    qkv = per_layer * 2 * hidden * (hidden + 2 * kv_width)
    o_proj = per_layer * 2 * hidden * hidden
    mlp = per_layer * 6 * hidden * config.intermediate_size
    if qkv + o_proj + mlp != coarse["linear"]:  # pragma: no cover - invariant guard
        raise AssertionError("stage split does not reproduce transformer_flops['linear']")
    return {
        "embedding": 0,
        "norm": 0,
        "qkv_proj": qkv,
        "rope": 0,
        "kv_write": 0,
        "kv_read": 0,
        "attention": coarse["attention"],
        "o_proj": o_proj,
        "mlp": mlp,
        "lm_head": coarse["lm_head"],
    }


def stage_bytes(
    config: ModelConfig,
    *,
    batch: int,
    new_tokens: int,
    context_len: int,
    dtype: torch.dtype,
    include_lm_head_output: bool = True,
) -> dict[str, int]:
    """Bytes each region moves, summing to ``transformer_bytes``.

    Weights are charged to the stage that reads them, the cache moves to
    ``kv_read``/``kv_write``, and every intermediate tensor to the stage that
    writes it. The activation split follows the same write-once-read-once
    convention as :func:`~infer_opt.profiling.transformer_bytes`; the MLP's
    final projection output is folded into the next layer's norm input there
    and so is not charged twice here.
    """
    element = torch.empty((), dtype=dtype).element_size()
    hidden, kv_width = config.hidden_size, config.n_kv_heads * config.head_dim
    layers, tokens = config.n_layers, batch * new_tokens
    inter = config.intermediate_size

    weights = {
        "embedding": config.vocab_size * hidden,
        "norm": layers * 2 * hidden + hidden,
        "qkv_proj": layers * (hidden * hidden + 2 * hidden * kv_width),
        "o_proj": layers * hidden * hidden,
        "mlp": layers * 3 * hidden * inter,
        "lm_head": config.vocab_size * hidden,
    }
    # Intermediates each layer materialises per token, written once and read once.
    activations = {
        "norm": 2 * hidden,
        "qkv_proj": hidden + 2 * kv_width,
        "rope": hidden + kv_width,
        "attention": hidden,
        "o_proj": hidden,
        "mlp": 3 * inter,
    }
    result = {name: 0 for name in REGION_NAMES}
    for name, count in weights.items():
        result[name] += count * element
    for name, per_token in activations.items():
        result[name] += 2 * layers * tokens * per_token * element
    result["kv_read"] += kv_cache_bytes(
        config, batch=batch, context_len=context_len, dtype=dtype
    )
    result["kv_write"] += kv_cache_bytes(
        config, batch=batch, context_len=new_tokens, dtype=dtype
    )
    if include_lm_head_output:
        result["lm_head"] += tokens * config.vocab_size * element
    return result
