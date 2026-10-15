#!/usr/bin/env python3
"""Isolate Qwen3 BF16 cache/backend/Inductor numerics using saved token replay."""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--attention", choices=["auto"], default="auto")
    parser.add_argument("--emulate-precision-casts", action="store_true")
    parser.add_argument("--gpu-memory-gb", type=int, choices=[40, 80], default=80)
    parser.add_argument("--measure-after-diagnosis", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("qwen_measurement_helper", args.helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, DynamicCache, StaticCache
    from torch.nn.attention import SDPBackend, sdpa_kernel

    status = {"status": "running", "task": "root_cause_diagnosis",
              "model": helper.TARGET, "revision": helper.REVISION}
    result = {"cases": {}, "comparisons": {}, "microtests": {}}
    helper.write_json(args.output / "execution.json", status)

    def progress(stage, **kwargs):
        helper.write_json(args.output / "progress.json", {"stage": stage, **kwargs})
        print(json.dumps({"stage": stage, **kwargs}), flush=True)

    def save():
        helper.write_json(args.output / "diagnosis.json", result)

    def context(backend):
        return contextlib.nullcontext() if backend == "auto" else sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)

    def aggregate(records):
        return {"passed": all(r["passed"] for r in records), "checks": len(records),
                "failed_checks": sum(not r["passed"] for r in records),
                "mismatched_elements": sum(r["mismatches"] for r in records),
                "elements": sum(r["elements"] for r in records),
                "max_abs_error": max((r["max_abs_error"] for r in records), default=0),
                "max_relative_l2": max((r["relative_l2"] for r in records), default=0)}

    try:
        hardware = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
                    "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
                    "bf16_supported": torch.cuda.is_bf16_supported()}
        helper.write_json(args.output / "hardware.json", hardware)
        if "A100" not in hardware["gpu"] or hardware["gpu_memory_gib"] < (35 if args.gpu_memory_gb == 40 else 75):
            raise RuntimeError(f"Expected A100 {args.gpu_memory_gb}GB")
        if args.gpu_memory_gb == 40 and hardware["gpu_memory_gib"] >= 48:
            raise RuntimeError("Expected the 40GB memory class")
        if transformers.__version__ != "4.57.3":
            raise RuntimeError("Expected Transformers 4.57.3")
        fixture = json.loads(args.fixture.read_text())[0]
        replay = fixture["validation"]["reference_tokens"]
        prompt_len, count = len(fixture["prompt_ids"]), len(replay)
        ids = torch.tensor([fixture["prompt_ids"]], device="cuda", dtype=torch.long)
        continuation = torch.tensor([replay], device="cuda", dtype=torch.long)
        positions = torch.arange(prompt_len + count, device="cuda")
        result["workload"] = {"prompt_len": prompt_len, "output_tokens": count,
                              "prompt_ids": fixture["prompt_ids"], "replay_tokens": replay,
                              "atol": 0.02, "rtol": 0.02}
        progress("model_download_and_load")
        model = AutoModelForCausalLM.from_pretrained(
            helper.TARGET, revision=helper.REVISION, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map="cuda").eval().requires_grad_(False)
        eos = model.generation_config.eos_token_id
        eos = [] if eos is None else (eos if isinstance(eos, list) else [eos])
        eos_indices = torch.tensor(eos, dtype=torch.long, device="cuda")
        checkpoints = {0, 1, 4, 32, 64, count-1}
        states = {}

        @torch.inference_mode()
        def run_case(name, cache_type, compiled=False, preserve=False, backend="auto"):
            progress("diagnostic_case", case=name)
            torch._dynamo.reset()
            cache = (StaticCache(config=model.config, max_cache_len=prompt_len + count)
                     if cache_type == "static" else DynamicCache(config=model.config))
            record = {"cache": cache_type, "compiled": compiled,
                      "emulate_precision_casts": preserve, "backend_policy": backend,
                      "status": "running"}
            result["cases"][name] = record
            save()
            captured = {"logits": [], "kv": {}}
            started = time.perf_counter()
            with context(backend):
                if cache_type == "static":
                    model(input_ids=ids, past_key_values=cache, cache_position=positions[:prompt_len],
                          position_ids=positions[:prompt_len].unsqueeze(0), use_cache=True,
                          logits_to_keep=1, return_dict=True)
                    cache.reset()
                forward = helper.compile_forward(model, emulate_precision_casts=preserve) if compiled else model.forward
                for step in range(count):
                    token = ids if step == 0 else continuation[:, step-1:step]
                    pos = positions[:prompt_len] if step == 0 else positions[prompt_len + step-1:prompt_len + step]
                    logits = forward(input_ids=token, past_key_values=cache, cache_position=pos,
                                     position_ids=pos.unsqueeze(0), use_cache=True,
                                     logits_to_keep=1, return_dict=True).logits
                    captured["logits"].append(logits.detach().cpu().clone())
                    if step in checkpoints:
                        length = prompt_len + step
                        captured["kv"][step] = [(layer.keys[..., :length, :].detach().cpu().clone(),
                                                layer.values[..., :length, :].detach().cpu().clone())
                                               for layer in cache.layers]
                record["cache_length"] = int(cache.get_seq_length())
                # Separate profiling; it is not performance data.
                if cache_type == "static":
                    cache.reset()
                else:
                    cache = DynamicCache(config=model.config)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    forward(input_ids=ids, past_key_values=cache, cache_position=positions[:prompt_len],
                            position_ids=positions[:prompt_len].unsqueeze(0), use_cache=True,
                            logits_to_keep=1, return_dict=True)
                    torch.cuda.synchronize()
                record["prefill_attention_events"] = helper.attention_events(prof)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    forward(input_ids=continuation[:, :1], past_key_values=cache,
                            cache_position=positions[prompt_len:prompt_len+1],
                            position_ids=positions[prompt_len:prompt_len+1].unsqueeze(0),
                            use_cache=True, logits_to_keep=1, return_dict=True)
                    torch.cuda.synchronize()
                record["decode_attention_events"] = helper.attention_events(prof)
            record["elapsed_seconds_including_setup_validation_and_profile"] = time.perf_counter() - started
            record["status"] = "passed_execution"
            states[name] = captured
            # Keep a small independently inspectable tensor artifact, not model weights.
            torch.save({"prefill_logits": captured["logits"][0],
                        "first_layer_prefill_key": captured["kv"][0][0][0]}, args.output / f"{name}-sample.pt")
            del cache, forward
            gc.collect()
            torch.cuda.empty_cache()
            save()

        def compare_cases(actual_name, reference_name):
            actual, reference = states[actual_name], states[reference_name]
            logit_records, kv_records = [], []
            argmax_equal = []
            for step, (a, b) in enumerate(zip(actual["logits"], reference["logits"])):
                logit_records.append({"step": step, **helper.compare(a, b)})
                a, b = a.clone(), b.clone()
                if eos:
                    a[..., eos] = -torch.inf
                    b[..., eos] = -torch.inf
                argmax_equal.append(bool(torch.equal(a.argmax(-1), b.argmax(-1))))
            for step in sorted(checkpoints):
                for layer, (a, b) in enumerate(zip(actual["kv"][step], reference["kv"][step])):
                    for kind, aa, bb in zip(("keys", "values"), a, b):
                        kv_records.append({"step": step, "layer": layer, "kind": kind,
                                           **helper.compare(aa, bb)})
            name = actual_name + "__vs__" + reference_name
            comparison = {"logits": aggregate(logit_records), "kv": aggregate(kv_records),
                          "argmax_equal_all_steps": all(argmax_equal),
                          "logit_checks": logit_records, "kv_checks": kv_records}
            comparison["passed"] = comparison["logits"]["passed"] and comparison["kv"]["passed"]
            result["comparisons"][name] = comparison
            save()
            print(json.dumps({"comparison": name, "passed": comparison["passed"],
                              "logits": comparison["logits"], "kv": comparison["kv"]}), flush=True)
            return comparison

        run_case("eager_dynamic_auto", "dynamic")
        run_case("eager_static_auto", "static")
        compare_cases("eager_static_auto", "eager_dynamic_auto")
        run_case("compiled_static_default", "static", compiled=True)
        compare_cases("compiled_static_default", "eager_static_auto")
        compare_cases("compiled_static_default", "eager_dynamic_auto")
        run_case("compiled_static_preserve_casts", "static", compiled=True, preserve=True)
        compare_cases("compiled_static_preserve_casts", "eager_static_auto")
        fixed = compare_cases("compiled_static_preserve_casts", "eager_dynamic_auto")

        # Match attention backends to remove dispatch as a confounding variable.
        run_case("eager_dynamic_efficient", "dynamic", backend="efficient")
        run_case("eager_static_efficient", "static", backend="efficient")
        compare_cases("eager_dynamic_efficient", "eager_dynamic_auto")
        compare_cases("eager_static_efficient", "eager_dynamic_efficient")
        compare_cases("eager_static_efficient", "eager_static_auto")

        progress("isolating_first_layer_operations")
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        from torch._inductor.utils import run_and_get_code
        with torch.inference_mode():
            layer = model.model.layers[0]
            attention = layer.self_attn
            hidden = model.model.embed_tokens(ids)
            normalized = layer.input_layernorm(hidden)
            projected = attention.k_proj(normalized).view(1, prompt_len, model.config.num_key_value_heads, attention.head_dim)
            key_norm = attention.k_norm(projected).transpose(1, 2)
            cos, sin = model.model.rotary_emb(hidden, positions[:prompt_len].unsqueeze(0))

            def rope(key):
                return apply_rotary_pos_emb(key, key, cos, sin)[1]

            def key_path(value):
                value = layer.input_layernorm(value)
                value = attention.k_proj(value).view(1, prompt_len, model.config.num_key_value_heads, attention.head_dim)
                value = attention.k_norm(value).transpose(1, 2)
                return apply_rotary_pos_emb(value, value, cos, sin)[1]

            operations = [("input_rmsnorm", layer.input_layernorm, hidden),
                          ("key_rmsnorm", attention.k_norm, projected),
                          ("rope", rope, key_norm), ("first_layer_key_path", key_path, hidden)]
            result["key_path_matches_native_eager_key"] = helper.compare(
                key_path(hidden).cpu(), states["eager_dynamic_auto"]["kv"][0][0][0])
            for name, function, value in operations:
                reference = function(value)
                record = {}
                for preserve in (False, True):
                    torch._dynamo.reset()
                    compiled = torch.compile(function, fullgraph=True, dynamic=False,
                                             options={"triton.cudagraphs": False,
                                                      "emulate_precision_casts": preserve})
                    actual, generated = run_and_get_code(compiled, value)
                    label = "preserve_casts" if preserve else "default"
                    record[label] = helper.compare(actual, reference)
                    (args.output / f"micro-{name}-{label}.txt").write_text("\n\n".join(generated))
                result["microtests"][name] = record
                save()
                print(json.dumps({"microtest": name, **record}), flush=True)

        result["preserve_casts_passes_original_gate"] = fixed["passed"]
        save()
        status["status"] = "diagnosed"
        # If the original accuracy gate passes, finish the previously requested
        # metrics in the same allocation using the explicitly recorded fix.
        if args.measure_after_diagnosis and fixed["passed"] and fixed["argmax_equal_all_steps"]:
            progress("running_fixed_measurement")
            del model, states, hidden, normalized, projected, key_norm, cos, sin, layer, attention
            gc.collect()
            torch._dynamo.reset()
            torch.cuda.empty_cache()
            measure_args = argparse.Namespace(prompt_lengths=[128, 512, 2048, 8192], output_tokens=128,
                                              warmup=2, repeats=3, attention="auto", emulate_precision_casts=True)
            helper.run_measurements(args.output, measure_args)
            status["status"] = "passed"
            status["measurement_fix"] = "emulate_precision_casts=True"
        helper.write_json(args.output / "execution.json", status)
        return 0
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        (args.output / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        save()
        return 1
    finally:
        helper.write_json(args.output / "execution.json", status)


if __name__ == "__main__":
    raise SystemExit(main())
