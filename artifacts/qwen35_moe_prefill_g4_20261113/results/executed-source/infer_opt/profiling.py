"""Measurement and analytic-cost helpers shared by the curriculum notebooks.

Two kinds of tool live here:

* *measured* quantities (:func:`timeit`, :func:`measure_peak_flops`,
  :func:`measure_peak_bandwidth`) which observe the hardware, and
* *analytic* quantities (:func:`transformer_flops`, :func:`transformer_bytes`)
  which count the work a forward pass must do regardless of implementation.

Comparing the two is the whole point: the analytic counts place a workload on
the roofline, the measured ones say how close the implementation gets to it.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .config import ModelConfig
from .runner import synchronize

__all__ = [
    "Timing",
    "Workload",
    "timeit",
    "device_summary",
    "measure_peak_flops",
    "measure_peak_bandwidth",
    "ridge_point",
    "roofline_ceiling",
    "parameter_count",
    "parameter_bytes",
    "kv_cache_bytes",
    "transformer_flops",
    "transformer_bytes",
    "step_workload",
    "capture_graph",
]


@dataclass(frozen=True)
class Timing:
    """Wall-clock result of a repeated measurement, in milliseconds."""

    ms: float
    ms_std: float
    ms_min: float
    iters: int

    @property
    def seconds(self) -> float:
        return self.ms / 1000

    def tflops(self, flops: float) -> float:
        """Achieved compute rate for a region that performs ``flops`` FLOPs."""
        return flops / self.seconds / 1e12

    def gbps(self, num_bytes: float) -> float:
        """Achieved memory rate for a region that moves ``num_bytes`` bytes."""
        return num_bytes / self.seconds / 1e9


@dataclass(frozen=True)
class Workload:
    """Analytic cost of a region: how much math, how much memory traffic."""

    flops: int
    bytes: int

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs performed per byte moved -- the x axis of the roofline."""
        return self.flops / self.bytes


def timeit(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int = 5,
    iters: int = 20,
) -> Timing:
    """Time ``fn`` properly: warm up first, then measure each iteration alone.

    On CUDA the kernel launch is asynchronous, so a bare ``perf_counter`` around
    the call times the *launch*, not the work. CUDA events are recorded into the
    same stream as the kernels and therefore measure device time.
    """
    if warmup < 0 or iters <= 0:
        raise ValueError("warmup cannot be negative and iters must be positive")
    for _ in range(warmup):
        fn()
    synchronize(device)

    samples: list[float] = []
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    else:
        for _ in range(iters):
            started = time.perf_counter()
            fn()
            samples.append(1000 * (time.perf_counter() - started))
    return Timing(
        ms=statistics.median(samples),
        ms_std=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        ms_min=min(samples),
        iters=iters,
    )


def device_summary(device: torch.device) -> dict[str, object]:
    """Facts about the device that every later number has to be read against."""
    summary: dict[str, object] = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
    }
    if device.type != "cuda":
        summary["name"] = "cpu"
        summary["bf16_supported"] = True
        return summary
    properties = torch.cuda.get_device_properties(device)
    summary.update(
        {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "sm_count": properties.multi_processor_count,
            "total_memory_gb": properties.total_memory / 1e9,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    )
    return summary


def _release(device: torch.device) -> None:
    """Hand the probe buffers back to the driver.

    These helpers allocate hundreds of megabytes. Left in the caching
    allocator they crowd out whatever is measured next, which on a small card
    shows up as the *next* benchmark mysteriously running several times slower.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.inference_mode()
def measure_peak_flops(
    device: torch.device, dtype: torch.dtype, *, size: int = 8192, iters: int = 10
) -> float:
    """Largest matmul rate the device actually reaches, in TFLOP/s.

    A big square GEMM is the friendliest possible shape: every byte loaded is
    reused ``size`` times, so nothing but the multipliers can be the limit.
    """
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    timing = timeit(lambda: torch.mm(a, b), device=device, warmup=3, iters=iters)
    del a, b
    _release(device)
    return timing.tflops(2 * size**3)


@torch.inference_mode()
def measure_peak_bandwidth(
    device: torch.device, dtype: torch.dtype, *, num_bytes: int = 256 << 20, iters: int = 20
) -> float:
    """Largest copy rate the device actually reaches, in GB/s.

    A straight copy has no reuse at all: ``num_bytes`` are read and the same
    amount written, so the only limit left is the memory system.
    """
    elements = num_bytes // torch.empty((), dtype=dtype).element_size()
    source = torch.empty(elements, device=device, dtype=dtype)
    destination = torch.empty_like(source)
    timing = timeit(lambda: destination.copy_(source), device=device, warmup=3, iters=iters)
    moved = 2 * source.numel() * source.element_size()
    del source, destination
    _release(device)
    return timing.gbps(moved)


def ridge_point(peak_tflops: float, peak_gbps: float) -> float:
    """Arithmetic intensity where the roofline turns, in FLOPs per byte.

    Below it a kernel is memory-bound, above it compute-bound.
    """
    return peak_tflops * 1e12 / (peak_gbps * 1e9)


def roofline_ceiling(intensity: float, peak_tflops: float, peak_gbps: float) -> float:
    """Best achievable TFLOP/s at a given arithmetic intensity."""
    return min(peak_tflops, peak_gbps * intensity / 1000)


def parameter_count(config: ModelConfig) -> int:
    """Weights of :class:`~infer_opt.model.DecoderOnlyTransformer` for ``config``."""
    hidden, kv_width = config.hidden_size, config.n_kv_heads * config.head_dim
    attention = 2 * hidden * hidden + 2 * hidden * kv_width  # q, o + k, v
    mlp = 3 * hidden * config.intermediate_size  # gate, up, down
    norms = 2 * hidden  # attn_norm, mlp_norm
    embeddings = 2 * config.vocab_size * hidden  # embedding + untied lm_head
    return config.n_layers * (attention + mlp + norms) + embeddings + hidden


def parameter_bytes(config: ModelConfig, dtype: torch.dtype) -> int:
    """Bytes a forward pass must stream in just to read the weights once."""
    return parameter_count(config) * torch.empty((), dtype=dtype).element_size()


def kv_cache_bytes(config: ModelConfig, *, batch: int, context_len: int, dtype: torch.dtype) -> int:
    """Bytes held by the K and V entries for ``context_len`` cached positions."""
    per_entry = config.n_kv_heads * config.head_dim
    element = torch.empty((), dtype=dtype).element_size()
    return 2 * config.n_layers * batch * per_entry * context_len * element


def _causal_pairs(new_tokens: int, context_len: int) -> int:
    """Query-key pairs a causal mask actually needs for one attention head.

    Each query at position ``p`` attends to ``p + 1`` keys, so a step that adds
    ``new_tokens`` queries ending at ``context_len`` needs the difference of two
    triangular numbers. Decode (``new_tokens == 1``) collapses to ``context_len``.
    """
    start = context_len - new_tokens
    return (context_len * (context_len + 1) - start * (start + 1)) // 2


def _validate_step(new_tokens: int, context_len: int, batch: int) -> None:
    if batch <= 0 or new_tokens <= 0:
        raise ValueError("batch and new_tokens must be positive")
    if new_tokens > context_len:
        raise ValueError("context_len must include the new tokens")


def transformer_flops(
    config: ModelConfig, *, batch: int, new_tokens: int, context_len: int
) -> dict[str, int]:
    """FLOPs for one forward step, split into the parts that scale differently.

    One function covers both phases, which is the point: prefill is
    ``new_tokens == context_len == T`` and decode is ``new_tokens == 1`` with a
    growing ``context_len``. Only the arguments change, not the model.

    Counts the multiply-add as 2 FLOPs and counts causally-required attention
    work only -- a dense masked kernel computes the masked half too and throws
    it away, which is exactly the gap a FlashAttention-style kernel closes.
    """
    _validate_step(new_tokens, context_len, batch)
    hidden, kv_width = config.hidden_size, config.n_kv_heads * config.head_dim
    tokens = batch * new_tokens

    per_token_linear = 2 * hidden * (2 * hidden + 2 * kv_width + 3 * config.intermediate_size)
    linear = config.n_layers * tokens * per_token_linear
    lm_head = 2 * tokens * hidden * config.vocab_size
    attention = (
        4
        * config.n_layers
        * batch
        * config.n_heads
        * config.head_dim
        * _causal_pairs(new_tokens, context_len)
    )
    return {
        "linear": linear,
        "attention": attention,
        "lm_head": lm_head,
        "total": linear + attention + lm_head,
    }


def transformer_bytes(
    config: ModelConfig,
    *,
    batch: int,
    new_tokens: int,
    context_len: int,
    dtype: torch.dtype,
    include_lm_head_output: bool = True,
) -> dict[str, int]:
    """Bytes one forward step moves, split by what is being moved.

    ``weights`` is paid once per step no matter how few tokens are in flight --
    the reason decode is memory-bound. ``kv_read`` grows with context length and
    ``activations`` with the token count, so the three terms dominate in
    different regimes.
    """
    _validate_step(new_tokens, context_len, batch)
    element = torch.empty((), dtype=dtype).element_size()
    hidden, kv_width = config.hidden_size, config.n_kv_heads * config.head_dim
    tokens = batch * new_tokens

    # Tensors each layer materialises per token, written once and read once:
    # norm outputs (2h), q/k/v and their RoPE copies, attention and o_proj
    # outputs, the two MLP branches plus their product, and the down output.
    per_token_activation = 2 * (6 * hidden + 3 * kv_width + 3 * config.intermediate_size)
    activations = config.n_layers * tokens * per_token_activation * element
    logits = tokens * config.vocab_size * element if include_lm_head_output else 0
    return {
        "weights": parameter_bytes(config, dtype),
        "kv_read": kv_cache_bytes(config, batch=batch, context_len=context_len, dtype=dtype),
        "kv_write": kv_cache_bytes(config, batch=batch, context_len=new_tokens, dtype=dtype),
        "activations": activations + logits,
        "total": (
            parameter_bytes(config, dtype)
            + kv_cache_bytes(config, batch=batch, context_len=context_len, dtype=dtype)
            + kv_cache_bytes(config, batch=batch, context_len=new_tokens, dtype=dtype)
            + activations
            + logits
        ),
    }


def step_workload(
    config: ModelConfig,
    *,
    batch: int,
    new_tokens: int,
    context_len: int,
    dtype: torch.dtype,
) -> Workload:
    """The analytic cost of one forward step as a single roofline point."""
    flops = transformer_flops(
        config, batch=batch, new_tokens=new_tokens, context_len=context_len
    )
    moved = transformer_bytes(
        config, batch=batch, new_tokens=new_tokens, context_len=context_len, dtype=dtype
    )
    return Workload(flops=flops["total"], bytes=moved["total"])


def capture_graph(
    step: Callable[[], object], *, device: torch.device, warmup: int = 3
) -> Callable[[], None]:
    """Record one fixed-shape step into a CUDA graph and return its replay.

    A decode step issues on the order of a thousand tiny kernels, and on a slow
    host the launch cost alone can exceed the device work -- the measurement
    then reports how fast Python is, not how fast the GPU is. A captured graph
    replays the whole step as one submission, which is the only way to see the
    device cost of a launch-bound workload.

    ``step`` must read and write fixed tensors at fixed shapes; replay reuses
    the very same memory, and any Python-side bookkeeping inside ``step`` (a
    cache position, for instance) is recorded once and *not* re-executed, so the
    caller has to restore it around each replay. Returns ``step`` unchanged on
    CPU so notebook code can stay device-agnostic.
    """
    if device.type != "cuda":
        return lambda: step()  # noqa: PLW0108 - normalise the return type to None
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph.replay
