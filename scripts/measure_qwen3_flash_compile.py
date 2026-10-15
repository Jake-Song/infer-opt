#!/usr/bin/env python3
"""Measure native Qwen3 StaticCache + torch.compile, with explicit SDPA policy.

Run a tiny-model compatibility gate before downloading the 14B weights. A failed
gate produces diagnostic artifacts, never timings for a substituted backend.
Transformers 4.57.3 is intentional; keep the runtime's CUDA PyTorch unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
import traceback
import warnings
from pathlib import Path

TARGET = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def compile_forward(model, *, emulate_precision_casts=False):
    import torch
    # A compiled graph is not a position-specific manually captured CUDA graph.
    return torch.compile(model.forward, fullgraph=True, dynamic=False,
                         options={"triton.cudagraphs": False,
                                  "emulate_precision_casts": emulate_precision_casts})


def backend_context(flash):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.FLASH_ATTENTION) if flash else contextlib.nullcontext()


def compatibility(output, *, emulate_precision_casts=False):
    import torch
    from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM, StaticCache
    config = Qwen3Config(vocab_size=128, hidden_size=256, intermediate_size=512,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=64,
                        max_position_embeddings=256, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    prompt = torch.randint(3, 128, (1, 16), device="cuda")
    position = torch.arange(16, device="cuda")
    cases = []
    # These are capability checks, not baseline performance comparisons.
    for cache_type, flash, compiled in (
        ("dynamic", True, False), ("static", False, False),
        ("static", True, False), ("static", True, True),
        ("static", False, True),
    ):
        torch._dynamo.reset()
        case = {"cache": cache_type, "flash_only": flash, "compile": compiled,
                "stages": {}}
        captured_warnings = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                cache = (StaticCache(config=config, max_cache_len=32) if cache_type == "static"
                         else DynamicCache(config=config))
                forward = compile_forward(model, emulate_precision_casts=emulate_precision_casts) if compiled else model.forward
                with torch.inference_mode(), backend_context(flash):
                    for stage, ids, pos in (
                        ("prefill", prompt, position),
                        ("decode", prompt[:, -1:], position[-1:] + 1),
                    ):
                        case["active_stage"] = stage
                        logits = forward(input_ids=ids, past_key_values=cache,
                                         cache_position=pos, position_ids=pos.unsqueeze(0),
                                         use_cache=True, logits_to_keep=1, return_dict=True).logits
                        torch.cuda.synchronize()
                        if not torch.isfinite(logits).all().item():
                            raise RuntimeError("nonfinite smoke-test logits")
                        case["stages"][stage] = "passed"
                case["status"] = "passed"
            except Exception as exc:
                case["status"] = "failed"
                case["error"] = f"{type(exc).__name__}: {exc}"
                case["traceback"] = traceback.format_exc()
            captured_warnings = [str(w.message) for w in caught]
        case["warnings"] = captured_warnings
        cases.append(case)
        write_json(output / "compatibility.json", cases)
        print(json.dumps({k: v for k, v in case.items() if k != "traceback"}), flush=True)
    del model
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return cases


def counters():
    from torch._dynamo.utils import counters as dynamo_counters
    return {str(group): {str(k): int(v) for k, v in counts.items()}
            for group, counts in dynamo_counters.items() if counts}


def compare(actual, expected, *, atol=0.02, rtol=0.02):
    """Check every element and report FP32 error reductions outside timing."""
    import torch
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    a, b = actual.float(), expected.float()
    error = (a - b).abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    return {"passed": bool((finite & (error <= atol + rtol * b.abs())).all().item()),
            "elements": a.numel(),
            "mismatches": int(((~finite) | (error > atol + rtol * b.abs())).sum().item()),
            "max_abs_error": float(error.max().item()),
            "relative_l2": float((torch.linalg.vector_norm(a-b) /
                                  torch.linalg.vector_norm(b).clamp_min(1e-12)).item())}


def validate(model, compiled_call, cache, ids, positions, count, select):
    import torch
    from transformers import DynamicCache
    reference_cache = DynamicCache(config=model.config)
    cache.reset()
    records, kv_records, reference_tokens, compiled_tokens = [], [], [], []
    forced = ids
    prompt_len = ids.shape[1]
    for step in range(count):
        pos = positions[:prompt_len] if step == 0 else positions[prompt_len + step-1:prompt_len + step]
        reference = model(input_ids=forced, past_key_values=reference_cache, cache_position=pos,
                          position_ids=pos.unsqueeze(0), use_cache=True,
                          logits_to_keep=1, return_dict=True).logits
        actual = compiled_call(forced, pos)
        records.append({"step": step, **compare(actual, reference)})
        ref_token, opt_token = select(reference), select(actual)
        reference_tokens.append(ref_token.item())
        compiled_tokens.append(opt_token.item())
        if step in {0, 1, 4, 32, 64, count-1}:
            length = prompt_len + step
            for index, (ref_layer, opt_layer) in enumerate(zip(reference_cache.layers, cache.layers)):
                for kind in ("keys", "values"):
                    kv_records.append({"step": step, "layer": index, "kind": kind,
                                       **compare(getattr(opt_layer, kind)[..., :length, :],
                                                 getattr(ref_layer, kind))})
        forced = ref_token
    length = prompt_len + count - 1
    current_length = cache.get_seq_length()
    cache_length = int(current_length.item() if hasattr(current_length, "item") else current_length)
    result = {"passed": all(r["passed"] for r in [*records, *kv_records]) and cache_length == length,
              "reference": "native eager forward, DynamicCache, same BF16 weights; no reference timing",
              "atol": 0.02, "rtol": 0.02,
              "logits": records, "kv": kv_records,
              "expected_cache_length": length, "actual_cache_length": cache_length,
              "teacher_forced_argmax_equal": reference_tokens == compiled_tokens,
              "reference_tokens": reference_tokens}
    del reference_cache
    return result


def attention_events(prof):
    return sorted({event.name for event in prof.events()
                   if any(part in event.name.lower() for part in ("attention", "flash", "fmha", "cutlass"))})


def run_measurements(output, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
    from torch._dynamo.utils import counters as dynamo_counters
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        TARGET, revision=REVISION, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(TARGET, revision=REVISION)
    write_json(output / "model.json", {"model": TARGET, "revision": REVISION,
               "load_and_download_seconds": time.perf_counter() - started,
               "parameter_dtype": str(next(model.parameters()).dtype)})
    text = ("한국의 GPU 추론 성능을 설명하시오. Explain how attention and KV caching work. "
            "def add(a, b): return a + b\n") * 2048
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else (eos if isinstance(eos, list) else [eos])
    eos_indices = torch.tensor(eos, dtype=torch.long, device="cuda")
    rows = []
    for prompt_len in args.prompt_lengths:
        write_json(output / "progress.json", {"stage": "compiling_and_validating", "prompt_len": prompt_len})
        torch._dynamo.reset()
        dynamo_counters.clear()
        ids = torch.tensor([token_ids[:prompt_len]], device="cuda", dtype=torch.long)
        if ids.shape[1] != prompt_len:
            raise ValueError("fixture too short")
        capacity = prompt_len + args.output_tokens
        cache = StaticCache(config=model.config, max_cache_len=capacity)
        positions = torch.arange(capacity, device="cuda")
        forward = compile_forward(model, emulate_precision_casts=args.emulate_precision_casts)

        def call(tokens, pos, fn=forward):
            return fn(input_ids=tokens, past_key_values=cache, cache_position=pos,
                      position_ids=pos.unsqueeze(0), use_cache=True,
                      logits_to_keep=1, return_dict=True).logits

        def select(logits):
            if eos:
                logits.index_fill_(-1, eos_indices, -torch.inf)
            return logits.argmax(dim=-1)

        def request(streaming=False):
            torch.cuda.synchronize()
            request_start = time.perf_counter()
            cache.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = call(ids, positions[:prompt_len])
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - start) * 1000
            token = select(logits)
            first_cpu = token.item()
            arrived = time.perf_counter()
            ttft_ms = (arrived - request_start) * 1000
            tokens = [first_cpu]
            generated = [token]
            itls = []
            decode_start = time.perf_counter()
            for step in range(args.output_tokens - 1):
                pos = positions[prompt_len + step:prompt_len + step + 1]
                token = select(call(token, pos))
                generated.append(token)
                if streaming:
                    tokens.append(token.item())
                    now = time.perf_counter()
                    itls.append((now - arrived) * 1000)
                    arrived = now
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - decode_start) * 1000
            # Read output after timing for the bulk decode measurement.
            if not streaming:
                tokens = torch.cat(generated, dim=-1).cpu().tolist()[0]
            return {"prefill_ms": prefill_ms, "ttft_ms": ttft_ms,
                    "decode_ms": decode_ms,
                    "decode_ms_per_token": decode_ms / (args.output_tokens - 1),
                    "itl_ms": itls, "tokens": tokens}

        row = {"prompt_len": prompt_len, "cache_capacity": capacity,
               "prompt_ids": ids.cpu().tolist()[0], "samples": []}
        rows.append(row)
        with torch.inference_mode(), backend_context(args.attention == "flash"):
            # Materialize cache once; allocation is setup, reset is included in TTFT.
            call(ids, positions[:prompt_len], model.forward)
            cache.reset()
            start = time.perf_counter()
            row["validation"] = validate(model, call, cache, ids, positions, args.output_tokens, select)
            torch.cuda.synchronize()
            row["compile_and_validation_seconds"] = time.perf_counter() - start
            write_json(output / "measurements.json", rows)
            if not row["validation"]["passed"]:
                # Isolate cache/backend changes from compilation before stopping.
                # This uses the same fixed-token full-logit/every-KV gate and
                # does not collect any reference latency.
                eager_call = lambda tokens, pos: call(tokens, pos, model.forward)
                row["eager_static_diagnostic"] = validate(
                    model, eager_call, cache, ids, positions, args.output_tokens, select)
                row["status"] = "validation_failed"
                write_json(output / "measurements.json", rows)
                raise RuntimeError(f"logits/KV validation failed at prompt length {prompt_len}; timing blocked")
            start = time.perf_counter()
            for _ in range(args.warmup):
                warm = request()
                if warm["tokens"] != row["validation"]["reference_tokens"]:
                    raise RuntimeError("free-generation greedy tokens differ from eager reference")
            row["warmup_and_decode_compile_seconds"] = time.perf_counter() - start
            row["compile_counters_before_measurement"] = counters()
            if row["compile_counters_before_measurement"].get("stats", {}).get("unique_graphs", 0) < 2:
                raise RuntimeError("Both prefill and decode compiled graphs were not observed")
            write_json(output / "progress.json", {"stage": "measuring", "prompt_len": prompt_len})
            for repeat in range(args.repeats):
                bulk = request()
                streaming = request(streaming=True)
                if bulk["tokens"] != streaming["tokens"]:
                    raise RuntimeError("bulk/streaming output tokens differ")
                row["samples"].append({"repeat": repeat, "bulk": bulk, "streaming": streaming})
                write_json(output / "measurements.json", rows)
            row["compile_counters_after_measurement"] = counters()
            if row["compile_counters_before_measurement"] != row["compile_counters_after_measurement"]:
                row["recompiled_during_measurement"] = True
            else:
                row["recompiled_during_measurement"] = False
            # Profile a separate fresh request, not any latency sample.
            cache.reset()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                call(ids, positions[:prompt_len])
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(output / f"prefill-{prompt_len}.json"))
            row["prefill_attention_events"] = attention_events(prof)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                call(ids[:, -1:], positions[prompt_len:prompt_len + 1])
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(output / f"decode-{prompt_len}.json"))
            row["decode_attention_events"] = attention_events(prof)
        itls = [value for sample in row["samples"] for value in sample["streaming"]["itl_ms"]]
        row["summary"] = {
            **{k: statistics.median(s["bulk"][k] for s in row["samples"])
               for k in ("prefill_ms", "ttft_ms", "decode_ms", "decode_ms_per_token")},
            "itl_p50_ms": percentile(itls, 0.5), "itl_p95_ms": percentile(itls, 0.95)}
        row["status"] = "passed"
        write_json(output / "measurements.json", rows)
        print(json.dumps({"prompt_len": prompt_len, **row["summary"]}), flush=True)
        del cache, forward
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention", choices=["flash", "auto"], default="flash")
    parser.add_argument("--gpu-memory-gb", type=int, choices=[40, 80], default=80)
    parser.add_argument("--emulate-precision-casts", action="store_true",
                        help="Preserve eager BF16 cast boundaries during Inductor fusion")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output_tokens < 2 or args.warmup < 1 or args.repeats < 1:
        parser.error("output-tokens >= 2; warmup and repeats >= 1 required")
    args.output.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "requested_gpu": f"A100 {args.gpu_memory_gb}GB",
              "model": TARGET, "revision": REVISION,
              "cache": "static", "attention": args.attention,
              "compile": {"fullgraph": True, "dynamic": False, "triton.cudagraphs": False,
                          "emulate_precision_casts": args.emulate_precision_casts}}
    write_json(args.output / "execution.json", status)
    try:
        import torch
        import transformers
        hardware = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
                    "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
                    "bf16_supported": torch.cuda.is_bf16_supported()}
        write_json(args.output / "hardware.json", hardware)
        print(json.dumps(hardware), flush=True)
        if "A100" not in hardware["gpu"] or hardware["gpu_memory_gib"] < (35 if args.gpu_memory_gb == 40 else 75):
            raise RuntimeError(f"Expected A100 {args.gpu_memory_gb}GB")
        if args.gpu_memory_gb == 40 and hardware["gpu_memory_gib"] >= 48:
            raise RuntimeError("Expected the 40GB memory class")
        if transformers.__version__ != "4.57.3" or not hardware["bf16_supported"]:
            raise RuntimeError("Requires Transformers 4.57.3 and BF16")
        write_json(args.output / "progress.json", {"stage": "compatibility"})
        cases = compatibility(args.output, emulate_precision_casts=args.emulate_precision_casts)
        relevant = [c for c in cases if c["flash_only"] == (args.attention == "flash") and c["cache"] == "static"]
        if not relevant or not all(c["status"] == "passed" for c in relevant):
            status["status"] = "incompatible"
            status["error"] = f"Native StaticCache + {args.attention} SDPA failed; no 14B weights downloaded and no latency substituted."
            return 2
        write_json(args.output / "progress.json", {"stage": "model_download_and_load"})
        run_measurements(args.output, args)
        status["status"] = "passed"
        return 0
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        (args.output / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        write_json(args.output / "execution.json", status)


if __name__ == "__main__":
    raise SystemExit(main())
