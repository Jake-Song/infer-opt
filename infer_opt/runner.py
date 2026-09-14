from __future__ import annotations

import time

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


@torch.inference_mode()
def benchmark(config: ModelConfig, run: RunConfig) -> dict[str, object]:
    model, device, dtype, prompt, continuation = prepare(config, run)

    def prefill_once() -> None:
        cache = make_cache(model, run, device, dtype)
        model.forward_prefill(prompt, cache)

    def decode_once() -> None:
        cache = make_cache(model, run, device, dtype)
        model.forward_prefill(prompt, cache)
        for index in range(run.decode_len):
            model.forward_decode(continuation[:, index : index + 1], cache)

    for _ in range(run.warmup):
        prefill_once()
        decode_once()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(run.iterations):
        prefill_once()
    synchronize(device)
    prefill_seconds = time.perf_counter() - started

    synchronize(device)
    started = time.perf_counter()
    for _ in range(run.iterations):
        decode_once()
    synchronize(device)
    decode_seconds = time.perf_counter() - started
    prefill_tokens = run.iterations * run.batch_size * run.prompt_len
    decode_tokens = run.iterations * run.batch_size * run.decode_len
    return {
        "model": config.to_dict(),
        "run": run.to_dict(),
        "device": str(device),
        "dtype": run.dtype,
        "prefill": {
            "total_seconds": prefill_seconds,
            "latency_ms": 1000 * prefill_seconds / run.iterations,
            "tokens_per_second": prefill_tokens / prefill_seconds,
        },
        "decode": {
            "total_seconds": decode_seconds,
            "latency_ms_per_sequence": 1000 * decode_seconds / run.iterations,
            "latency_ms_per_token": 1000 * decode_seconds / (run.iterations * run.decode_len),
            "tokens_per_second": decode_tokens / decode_seconds,
        },
    }
