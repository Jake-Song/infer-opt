from __future__ import annotations

import time
import statistics
from collections.abc import Callable

import torch

from .config import ModelConfig, RunConfig
from .model import DecoderOnlyTransformer, make_random_model


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def make_inputs(config: ModelConfig, run: RunConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(run.seed + 1)
    tokens = torch.randint(config.vocab_size, (run.batch_size, run.prompt_len + run.decode_len), generator=generator)
    return tokens[:, : run.prompt_len].to(device), tokens[:, run.prompt_len :].to(device)


def prepare(config: ModelConfig, run: RunConfig) -> tuple[DecoderOnlyTransformer, torch.device, torch.dtype, torch.Tensor, torch.Tensor]:
    device, dtype = resolve_device(run.device), resolve_dtype(run.dtype)
    model = make_random_model(config, run.strategy, run.seed).to(device=device, dtype=dtype).eval()
    prompt, continuation = make_inputs(config, run, device)
    return model, device, dtype, prompt, continuation


@torch.inference_mode()
def check_cached_decode(config: ModelConfig, run: RunConfig) -> dict[str, float | bool | int | str]:
    model, device, dtype, prompt, continuation = prepare(config, run)
    all_tokens = torch.cat((prompt, continuation), dim=1)
    full_logits = model(all_tokens)
    cache = make_cache(model, run, device, dtype)
    pieces = [model.forward_prefill(prompt, cache)]
    for index in range(run.decode_len):
        pieces.append(model.forward_decode(continuation[:, index : index + 1], cache))
    cached_logits = torch.cat(pieces, dim=1)
    max_abs_error = (full_logits - cached_logits).abs().max().item()
    tolerance = 2e-4 if dtype == torch.float32 else 2e-2
    return {
        "passed": bool(torch.allclose(full_logits, cached_logits, rtol=tolerance, atol=tolerance)),
        "max_abs_error": max_abs_error,
        "tolerance": tolerance,
        "device": str(device),
        "dtype": run.dtype,
        "tokens_checked": all_tokens.numel(),
    }


def make_cache(model: DecoderOnlyTransformer, run: RunConfig, device: torch.device, dtype: torch.dtype):
    """Build whichever KV cache ``run`` asks for; the model treats them alike."""
    if run.cache == "paged":
        return model.new_paged_cache(run.batch_size, device, dtype, block_size=run.block_size)
    return model.new_cache(run.batch_size, device, dtype)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timing_summary(samples: list[float]) -> dict[str, float]:
    """p90 uses linear interpolation over repetitions, not over decode tokens."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * 0.9
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return {
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "p90": ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower),
        "min": ordered[0],
        "max": ordered[-1],
    }


def measure_phase(
    setup: Callable[[], None],
    forward: Callable[[], object],
    device: torch.device,
    run: RunConfig,
) -> dict[str, object]:
    """Time forwards only; setup and its pending GPU work finish before timing.

    Wall time includes event recording and the final completion synchronization.
    CUDA event time spans the stream interval, including any host launch gaps;
    it is not the sum of kernel execution times.
    """
    for _ in range(run.warmup):
        setup()
        forward()
    synchronize(device)
    events = None
    stream = None
    if device.type == "cuda":
        stream = torch.cuda.current_stream(device)
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        # Initialize the lazy CUDA events outside measured regions.
        for event in events:
            event.record(stream)
        synchronize(device)
    samples = []
    for index in range(run.iterations):
        setup()
        synchronize(device)
        started = time.perf_counter()
        if events:
            events[0].record(stream)
        output = forward()
        if events:
            events[1].record(stream)
        synchronize(device)
        wall_ms = 1000 * (time.perf_counter() - started)
        cuda_ms = events[0].elapsed_time(events[1]) if events else None
        samples.append({"iteration": index, "wall_ms": wall_ms, "cuda_ms": cuda_ms})
        del output
    return {
        "samples": samples,
        "wall_ms": timing_summary([sample["wall_ms"] for sample in samples]),
        "cuda_ms": timing_summary([sample["cuda_ms"] for sample in samples]) if events else None,
    }


@torch.inference_mode()
def benchmark_prepared(
    model: DecoderOnlyTransformer,
    run: RunConfig,
    prompt: torch.Tensor,
    continuation: torch.Tensor,
) -> dict[str, object]:
    config = model.config
    if run.prompt_len + run.decode_len > config.max_seq_len:
        raise ValueError("prompt_len + decode_len must not exceed max_seq_len")
    device, dtype = prompt.device, next(model.parameters()).dtype
    cache = make_cache(model, run, device, dtype)
    # Even --warmup 0 must not time initial table construction or input slicing.
    model.rope.slice(0, config.max_seq_len, device)
    decode_inputs = tuple(continuation[:, i : i + 1] for i in range(run.decode_len))

    def prefill_once():
        return model.forward_prefill(prompt, cache)

    def prepare_decode() -> None:
        cache.reset()
        prefill_once()

    def decode_once():
        output = None
        for token in decode_inputs:
            output = model.forward_decode(token, cache)
        return output

    prefill = measure_phase(cache.reset, prefill_once, device, run)
    decode = measure_phase(prepare_decode, decode_once, device, run)
    prefill_seconds = sum(sample["wall_ms"] for sample in prefill["samples"]) / 1000
    decode_seconds = sum(sample["wall_ms"] for sample in decode["samples"]) / 1000
    prefill.update({
        "total_seconds": prefill_seconds,
        "latency_ms": prefill["wall_ms"]["mean"],
        "tokens_per_second": run.iterations * run.batch_size * run.prompt_len / prefill_seconds,
    })
    decode.update({
        "total_seconds": decode_seconds,
        "latency_ms_per_sequence": decode["wall_ms"]["mean"],
        "latency_ms_per_token": decode["wall_ms"]["mean"] / run.decode_len,
        "tokens_per_second": run.iterations * run.batch_size * run.decode_len / decode_seconds,
        "initial_context_len": run.prompt_len,
        "final_context_len": run.prompt_len + run.decode_len,
    })
    return {
        "schema_version": 2,
        "model": config.to_dict(),
        "run": run.to_dict(),
        "device": str(device),
        "dtype": run.dtype,
        "timing_contract": {
            "primary": "synchronized_wall",
            "prefill": "forward_prefill including KV writes and all-position vocabulary logits",
            "decode": "fixed-token sequential forward_decode calls; growing KV cache",
            "excluded": ["model_setup", "input_transfer", "cache_allocation", "cache_reset",
                         "rope_initialization", "warmup", "validation", "decode_prefill"],
            "wall_includes": ["Python_dispatch", "CUDA_event_recording", "completion_synchronization"],
            "cuda_event_note": "stream elapsed time includes host launch gaps; not kernel-only time",
            "percentiles": "over repetitions; decode per-token statistics divide each sequence by decode_len",
        },
        "prefill": prefill,
        "decode": decode,
    }


@torch.inference_mode()
def benchmark(config: ModelConfig, run: RunConfig) -> dict[str, object]:
    model, _, _, prompt, continuation = prepare(config, run)
    return benchmark_prepared(model, run, prompt, continuation)
