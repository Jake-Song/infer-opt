"""Operation-level empirical roofline analysis of the unchanged eager model.

Logical tensor traffic is NOT measured DRAM traffic. Device times are exclusive
kernel durations from a separate profiler pass, not end-to-end latency.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function

from .baseline import BASELINE_MODEL, environment, source_identity, validate_prepared, _write_json
from .config import ModelConfig, RunConfig
from .model import RMSNorm, make_random_model
from .regions import profiled_regions, region_of
from .runner import make_inputs, measure_phase

PREFIX = "roofline::"
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")
OPERATIONS = ("embedding", "norm", "q_proj", "k_proj", "v_proj", "rope", "kv_write", "kv_read",
              "attention", "layout", "o_proj", "gate_proj", "up_proj", "swiglu_residual",
              "down_proj", "lm_head", "unattributed")


def operation_costs(config: ModelConfig, *, tokens: int, context: int, batch: int = 1,
                    element_bytes: int = 2) -> dict[str, dict]:
    """Per-forward FLOPs and logical bytes, summed across all layers.

    GEMMs count one read per input/weight and one output write. Attention counts
    the baseline's explicit GQA expansion and bool mask, but not score spills or
    repeated tile loads. Elementwise counts are approximate; no Tensor Core roof
    is applied to them. A KV view performs no device copy.
    """
    if min(tokens, context, batch, element_bytes) <= 0 or tokens > context:
        raise ValueError("positive sizes and tokens <= context required")
    h, k, f, v, layers = (config.hidden_size, config.n_kv_heads * config.head_dim,
                           config.intermediate_size, config.vocab_size, config.n_layers)
    m, e = batch * tokens, element_bytes
    costs = {}

    def add(name, flops, moved, kind, shape, note):
        costs[name] = dict(flops=flops, logical_bytes=moved, kind=kind, shape=shape, accounting=note)

    for name, inputs, outputs, count in (
        ("q_proj", h, h, layers), ("k_proj", h, k, layers), ("v_proj", h, k, layers),
        ("o_proj", h, h, layers), ("gate_proj", h, f, layers), ("up_proj", h, f, layers),
        ("down_proj", f, h, layers), ("lm_head", h, v, 1),
    ):
        add(name, count * 2 * m * inputs * outputs,
            count * (m * inputs + inputs * outputs + m * outputs) * e, "gemm",
            f"{count} x ({m},{inputs}) @ ({inputs},{outputs})",
            "2MNK FLOPs; (MK+KN+MN)*element_bytes; no measured HBM counters")
    pairs = (context * (context + 1) - (context - tokens) * (context - tokens + 1)) // 2
    expansion = (2 * batch * context * (k + h)) if config.n_heads != config.n_kv_heads else 0
    attn_bytes = layers * (e * (2 * m * h + 2 * batch * context * h + expansion) + 2 * tokens * context)
    add("attention", 4 * layers * batch * config.n_heads * config.head_dim * pairs, attn_bytes,
        "attention", f"{layers} x Q[{batch},{config.n_heads},{tokens},{config.head_dim}], S={context}",
        "Causally useful QK+AV FLOPs only; includes GQA expansion and mask logical traffic. "
        "Softmax, masked-out work, workspace, repeated tile loads are not modeled.")
    add("embedding", 0, 2 * m * h * e + m * 8, "gather", f"{m} selected rows x {h}",
        "Read selected rows and write output; does NOT read the entire embedding table")
    add("norm", (2 * layers + 1) * 5 * m * h, (2 * layers + 1) * (2 * m * h + h) * e,
        "elementwise", f"{2 * layers + 1} x [{m},{h}]",
        "Approximate arithmetic; minimum input/weight/output traffic omits FP32 intermediates")
    add("rope", 3 * layers * m * (h + k), 2 * layers * m * (h + k) * e,
        "elementwise", f"Q+K: {layers} x {m} x {h+k}",
        "Minimum Q/K input/output traffic; table casts and intermediate tensors omitted")
    add("kv_write", 0, 4 * layers * m * k * e, "copy", f"K,V: {layers} x {m} x {k}",
        "Read incoming K/V and write cached K/V")
    add("kv_read", 0, 0, "view", "strided K/V views", "No copy; attention is charged for K/V consumption")
    add("layout", 0, 2 * layers * m * h * e if tokens > 1 else 0, "copy", f"{layers} x [{m},{h}]",
        "Potential attention-output contiguous copy; may be elided depending on strides/backend")
    add("swiglu_residual", layers * (6 * m * f + m * h), layers * (5 * m * f + 3 * m * h) * e,
        "elementwise", f"{layers} x SwiGLU[{m},{f}] + residual[{m},{h}]",
        "SiLU and multiply plus FFN residual; transcendental count approximate, not Tensor Core work")
    add("unattributed", 0, 0, "unknown", "residual/add and any unscoped kernels",
        "No guessed traffic or roofline verdict for unscoped kernels")
    return {name: costs[name] for name in OPERATIONS}


def assess(cost: dict, gpu_ms: float, compute_tflops: float, copy_gbps: float) -> dict:
    """Keep a model-side prediction separate from measured evidence.

    Efficiency thresholds are explanatory heuristics, not hardware counters.
    They must never turn a distant point into a claim of bandwidth saturation.
    """
    if min(compute_tflops, copy_gbps) <= 0:
        raise ValueError("calibration rates must be positive")
    f, q = cost["flops"], cost["logical_bytes"]
    ai = f / q if q else None
    ridge = compute_tflops * 1000 / copy_gbps
    result = dict(arithmetic_intensity=ai, effective_gbps=q / gpu_ms / 1e6 if gpu_ms > 0 else None,
                  achieved_tflops=f / gpu_ms / 1e9 if gpu_ms > 0 else None,
                  ridge_flops_per_byte=ridge, roof_tflops=None, roof_fraction=None)
    if cost["kind"] == "view":
        result.update(predicted_bound="view: 판정 제외", evidence="GPU 메모리 복사 없음; 소비 비용은 attention")
    elif cost["kind"] not in ("gemm", "attention"):
        result.update(predicted_bound="memory/launch 후보" if q else "판정 제외",
                      evidence="비 Tensor Core 연산: 이 roof로 compute/memory 확정 불가")
    else:
        ceiling = min(compute_tflops, ai * copy_gbps / 1000)
        efficiency = result["achieved_tflops"] / ceiling if gpu_ms > 0 else None
        bound = "memory-bound (모형)" if ai < ridge else "compute-bound (모형)"
        if 0.8 <= ai / ridge <= 1.25:
            bound = "경계 영역 (모형)"
        if efficiency is None:
            evidence = "GPU 시간 없음: 실측 판정 불가"
        elif efficiency > 1.1:
            evidence = "경험적 roof 초과: 캐시/트래픽 모형/보정값 확인"
        elif efficiency < 0.5:
            evidence = "roof에서 멂: launch, occupancy, 연산 형태, 캐시 추가 확인"
        else:
            evidence = "예상 roof에 비교적 가까움; 실제 HBM 포화 여부는 미확인"
        if cost["kind"] == "attention":
            evidence += "; SDPA 유효 FLOPs 기반 참고값"
        result.update(predicted_bound=bound, evidence=evidence, roof_tflops=ceiling, roof_fraction=efficiency)
    return result


@contextlib.contextmanager
def operation_annotations(model):
    """Temporary module hooks only during profiling; do not change the model."""
    handles, active = [], defaultdict(list)

    def pre(module, args):
        scope = record_function(PREFIX + labels[id(module)])
        scope.__enter__()
        active[id(module)].append(scope)

    def post(module, args, output):
        if active[id(module)]:
            active[id(module)].pop().__exit__(None, None, None)

    labels = {}
    try:
        for name, module in model.named_modules():
            label = name.rsplit(".", 1)[-1]
            if isinstance(module, torch.nn.Linear) and label in LINEARS:
                labels[id(module)] = label
            elif isinstance(module, RMSNorm):
                labels[id(module)] = "norm"
            elif isinstance(module, torch.nn.Embedding):
                labels[id(module)] = "embedding"
            else:
                continue
            handles.append(module.register_forward_pre_hook(pre))
            handles.append(module.register_forward_hook(post, always_call=True))
        with profiled_regions():
            yield
    finally:
        for handle in handles:
            handle.remove()
        for scopes in active.values():
            while scopes:
                scopes.pop().__exit__(None, None, None)


def event_owner(event) -> str:
    """Choose nearest nested annotation, so parent+child are never added twice."""
    current = event
    while current is not None:
        name = current.name
        if name.startswith(PREFIX) and name[len(PREFIX):] in OPERATIONS:
            return name[len(PREFIX):]
        coarse = region_of(name)
        if coarse is not None:
            return {"mlp": "swiglu_residual", "o_proj": "layout", "qkv_proj": "unattributed"}.get(
                coarse, coarse if coarse in OPERATIONS else "unattributed")
        current = current.cpu_parent
    return "unattributed"


def attribute_events(events, *, steps: int) -> dict:
    """Sum only directly linked CUDA kernels on CPU events; never ancestor totals."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    totals = {name: 0.0 for name in OPERATIONS}
    cpu = {name: 0.0 for name in OPERATIONS}
    counts = {name: 0 for name in OPERATIONS}
    kernel_totals = defaultdict(lambda: [0, 0.0])
    raw_gpu_us = 0.0
    for event in events:
        if event.device_type == DeviceType.CUDA:
            if not getattr(event, "is_user_annotation", False):
                raw_gpu_us += event.time_range.elapsed_us()
            continue
        if event.device_type != DeviceType.CPU:
            continue
        owner = event_owner(event)
        cpu[owner] += event.self_cpu_time_total / 1000 / steps
        for kernel in event.kernels:
            ms = kernel.duration / 1000 / steps
            totals[owner] += ms
            counts[owner] += 1
            item = kernel_totals[(owner, kernel.name)]
            item[0] += 1
            item[1] += ms
    total = sum(totals.values())
    raw_ms = raw_gpu_us / 1000 / steps
    delta = abs(total - raw_ms) / max(raw_ms, 1e-12)
    return dict(gpu_ms=totals, cpu_self_ms=cpu, kernel_counts=counts, gpu_sum_ms=total,
                raw_gpu_sum_ms=raw_ms, accounting_relative_error=delta,
                accounting_passed=raw_ms > 0 and delta < 0.01,
                kernels=[dict(operation=o, kernel=k, calls=count, gpu_ms=ms)
                         for (o, k), (count, ms) in sorted(kernel_totals.items(), key=lambda x: -x[1][1])])


@torch.inference_mode()
def profile_step(model, setup, forward, *, device, iters: int = 3, trace_path: Path | None = None) -> dict:
    if iters <= 0:
        raise ValueError("iters must be positive")
    for _ in range(2):
        setup()
        forward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with operation_annotations(model), profile(activities=activities, record_shapes=False) as prof:
        for _ in range(iters):
            setup()  # Only Python cache position/reset here; prerequisite prefill is outside.
            with record_function("roofline_step"):
                forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    result = attribute_events(prof.events(), steps=iters)
    result["steps"] = iters
    if trace_path is not None:
        plain = trace_path.with_suffix("") if trace_path.suffix == ".gz" else trace_path
        prof.export_chrome_trace(str(plain))
        if trace_path.suffix == ".gz":
            with plain.open("rb") as inp, gzip.open(trace_path, "wb") as out:
                import shutil
                shutil.copyfileobj(inp, out)
            plain.unlink()
        result["trace"] = trace_path.name
    return result


@torch.inference_mode()
def calibrate(device: torch.device, *, sizes=(2048, 4096), copy_bytes=256 << 20, iters=10) -> dict:
    """Same-runtime BF16 GEMM/copy reference ceilings, not advertised peaks."""
    if device.type != "cuda":
        raise ValueError("roofline calibration requires a CUDA GPU")
    run = RunConfig(device="cuda", dtype="bfloat16", warmup=5, iterations=iters)
    gemms = []
    for size in sizes:
        a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
        b = torch.randn_like(a)
        out = torch.empty_like(a)
        measured = measure_phase(lambda: None, lambda: torch.mm(a, b, out=out), device, run)
        ms = measured["cuda_ms"]["median"]
        gemms.append(dict(size=size, tflops=2 * size**3 / ms / 1e9, timing=measured))
        del a, b, out
    src = torch.empty(copy_bytes // 2, dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    dst = torch.empty_like(src)
    copied = measure_phase(lambda: None, lambda: dst.copy_(src), device, run)
    moved = 2 * src.numel() * src.element_size()
    bandwidth = moved / copied["cuda_ms"]["median"] / 1e6
    del src, dst
    torch.cuda.empty_cache()
    return dict(compute_tflops=max(item["tflops"] for item in gemms), copy_gbps=bandwidth,
                gemms=gemms, copy=dict(array_bytes=copy_bytes, logical_bytes=moved, timing=copied),
                method="BF16 preallocated square GEMM and large initialized copy; CUDA-event medians; "
                       "empirical reference ceilings, not specifications or measured DRAM counters")


def write_report(output: Path, result: dict) -> None:
    _write_json(output / "roofline_results.json", result)
    rows = [row for condition in result["conditions"] for row in condition.get("operations", [])]
    columns = ["prompt_len", "phase", "operation", "shape", "gpu_ms", "gpu_share_pct", "flops",
               "logical_bytes", "arithmetic_intensity", "achieved_tflops", "effective_gbps",
               "predicted_bound", "roof_fraction", "evidence"]
    with (output / "operations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# A100 연산별 roofline 분석", "", f"상태: **{result['status']}**", "",
             "GPU 시간은 별도 profiler 실행에서 각 CUDA kernel을 한 번만 합산한 값이다.",
             "프로파일러를 끈 wall time과 GPU kernel 합은 서로 다른 실행이며, 차이를 정확한 CPU overhead로 단정하지 않는다.",
             "FLOPs와 bytes는 알고리즘/텐서 기반 추정이다. Effective GB/s는 실제 HBM 측정값이 아니다.",
             "표의 bound는 경험적 roof와 논리적 bytes로 예측한 한계이다. 실제 포화 판정에는 Nsight Compute counter가 필요하다.", ""]
    for condition in result["conditions"]:
        lines += [f"## Prompt {condition['prompt_len']} / {condition['phase']}", "",
                  f"상태: {condition['status']}", ""]
        if "timing" in condition:
            lines += [f"Profiler OFF wall 중앙값: {condition['timing']['wall_ms']['median']:.4f} ms", ""]
        if "operations" not in condition:
            continue
        lines += ["| 연산 | GPU ms | GPU 비중 % | FLOP/byte | Roof 예상 | 실측 해석 |",
                  "|---|---:|---:|---:|---|---|"]
        for row in sorted(condition["operations"], key=lambda r: -r["gpu_ms"]):
            ai = row["arithmetic_intensity"]
            lines.append(f"| {row['operation']} | {row['gpu_ms']:.4f} | {row['gpu_share_pct']:.1f} | "
                         f"{ai:.3f} | {row['predicted_bound']} | {row['evidence']} |" if ai is not None else
                         f"| {row['operation']} | {row['gpu_ms']:.4f} | {row['gpu_share_pct']:.1f} | — | "
                         f"{row['predicted_bound']} | {row['evidence']} |")
        lines.append("")
    if result.get("error"):
        lines += ["실행 오류: " + result["error"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def plot_results(result: dict, output: Path) -> list[Path]:
    """Export standalone figures; zero/unknown FLOPs never get fake log points."""
    import numpy as np
    import matplotlib.pyplot as plt

    conditions = [c for c in result["conditions"] if c["status"] == "passed" and "operations" in c]
    if not conditions:
        return []
    calibration = result["calibration"]
    compute, bandwidth = calibration["compute_tflops"], calibration["copy_gbps"]
    outputs = []
    colors = dict(zip((*LINEARS, "attention"), plt.get_cmap("tab10").colors))
    for length in sorted({c["prompt_len"] for c in conditions}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        times, time_axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        for phase, ax, tax in zip(("prefill", "decode"), axes, time_axes):
            condition = next((c for c in conditions if c["prompt_len"] == length and c["phase"] == phase), None)
            if condition is None:
                ax.set_title(f"{phase}: no validated measurement")
                tax.set_visible(False)
                continue
            rows = condition["operations"]
            plot_rows = [r for r in rows if r["kind"] in ("gemm", "attention") and r["gpu_ms"] > 0]
            xs = np.geomspace(0.1, max([1e4, *(r['arithmetic_intensity'] * 2 for r in plot_rows)]), 300)
            ax.loglog(xs, np.minimum(compute, xs*bandwidth/1000), color="black", label="empirical GEMM/copy roof")
            ax.axvline(compute*1000/bandwidth, color="gray", linestyle=":", label="ridge")
            for row in plot_rows:
                ax.scatter(row["arithmetic_intensity"], row["achieved_tflops"], color=colors[row['operation']],
                           marker="^" if row["kind"] == "attention" else "o", s=50, label=row['operation'])
            ax.set(xlabel="Logical arithmetic intensity (FLOP/byte)", ylabel="Useful FLOPs / profiler kernel time (TFLOP/s)",
                   title=f"{phase}, prefix={length}, T={condition['tokens']}, S={condition['context']}")
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=7, loc="best")
            ordered = sorted([r for r in rows if r['gpu_ms'] > 0], key=lambda r: r['gpu_ms'])
            tax.barh([r['operation'] for r in ordered], [r['gpu_ms'] for r in ordered], color="#28699b")
            tax.set(xlabel="Exclusive GPU kernel time per forward (ms)", title=f"{phase}: all {result['model']['n_layers']} layers")
            for index, row in enumerate(ordered):
                tax.text(row['gpu_ms'], index, f" {row['gpu_share_pct']:.1f}%", va="center", fontsize=8)
            tax.margins(x=0.2)
        fig.suptitle("Logical traffic model; attention uses useful causal FLOPs; not an HBM-counter roofline")
        times.suptitle("Separate profiler pass; includes pointwise/copy stages omitted from Tensor Core roof")
        for figure, name in ((fig, f"roofline-{length}.png"), (times, f"operation-times-{length}.png")):
            target = output / name
            figure.savefig(target, dpi=160)
            plt.close(figure)
            outputs.append(target)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for phase, ax in zip(("prefill", "decode"), axes):
        rows = sorted([c for c in conditions if c['phase'] == phase], key=lambda c: c['prompt_len'])
        x = [c['prompt_len'] for c in rows]
        ax.plot(x, [c['timing']['wall_ms']['median'] for c in rows], 'o-', label='Profiler OFF wall median')
        ax.plot(x, [c['profile']['gpu_sum_ms'] for c in rows], 's--', label='Profiler ON GPU kernel sum')
        ax.set(xlabel="Prefix tokens", ylabel="ms per forward", title=phase)
        ax.set_xscale('log', base=2)
        ax.set_xticks(x, [str(n) for n in x])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Different runs: their difference is not an exact measurement of CPU overhead")
    target = output / 'wall-vs-kernels.png'
    fig.savefig(target, dpi=160)
    plt.close(fig)
    outputs.append(target)
    return outputs


@torch.inference_mode()
def run_roofline(output: Path, *, prompt_lengths=(128, 512, 2048, 8192), iterations=10,
                 warmup=3, profile_iters=3, seed=0, validation_decode_len=128,
                 budget_minutes=110, config=BASELINE_MODEL) -> dict:
    if (not prompt_lengths or len(set(prompt_lengths)) != len(prompt_lengths)
            or min(*prompt_lengths, iterations, profile_iters, validation_decode_len, budget_minutes) <= 0
            or warmup < 0 or max(prompt_lengths) + validation_decode_len > config.max_seq_len):
        raise ValueError("invalid sizes, repetitions, or budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "roofline_results.json").exists():
        raise FileExistsError("choose a fresh result directory")
    started = time.monotonic()
    result = dict(status="running", schema_version=1, conditions=[], model=config.to_dict(),
                  settings=dict(prompt_lengths=list(prompt_lengths), iterations=iterations, warmup=warmup,
                                profile_iters=profile_iters, seed=seed, validation_decode_len=validation_decode_len),
                  decode_scope="one next token at fixed prefix length; reset position between repetitions; "
                               "not the prior baseline's 128-token sequence average")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; no CPU fallback")
        device = torch.device("cuda")
        result["environment"] = environment(device)
        if "A100" not in result["environment"]["gpu_name"] or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A100 with BF16 support required")
        result["source"] = source_identity()
        result["calibration"] = calibration = calibrate(device)
        model = make_random_model(config, "eager_sdpa", seed).to(device=device, dtype=torch.bfloat16).eval()
        result["parameter_count"] = sum(p.numel() for p in model.parameters())
        write_report(output, result)
        for length in prompt_lengths:
            if time.monotonic() - started >= budget_minutes * 60:
                raise TimeoutError("roofline budget exhausted")
            run = RunConfig(batch_size=1, prompt_len=length, decode_len=validation_decode_len,
                            device="cuda", dtype="bfloat16", iterations=iterations, warmup=warmup, seed=seed)
            prompt, continuation = make_inputs(config, run, device)
            print(f"prompt={length}: full logits / every KV validation", flush=True)
            validation = validate_prepared(model, prompt, continuation)
            for phase in ("prefill", "decode"):
                condition = dict(prompt_len=length, phase=phase, validation=validation, status="validating")
                result["conditions"].append(condition)
                write_report(output, result)
                if not validation["passed"]:
                    condition["status"] = "validation_failed"
                    write_report(output, result)
                    continue
                cache = model.new_cache(1, device, torch.bfloat16)
                model.rope.slice(0, config.max_seq_len, device)
                if phase == "prefill":
                    setup = cache.reset
                    forward = lambda: model.forward_prefill(prompt, cache)
                    tokens, context = length, length
                else:
                    model.forward_prefill(prompt, cache)
                    token = continuation[:, :1]
                    setup = lambda: setattr(cache, "position", length)
                    forward = lambda: model.forward_decode(token, cache)
                    tokens, context = 1, length + 1
                print(f"prompt={length}/{phase}: profiler OFF latency, then profiler ON attribution", flush=True)
                condition["timing"] = measure_phase(setup, forward, device, run)
                attribution = profile_step(model, setup, forward, device=device, iters=profile_iters,
                                           trace_path=output / f"trace-{phase}-{length}.json.gz")
                condition["profile"] = attribution
                condition["tokens"], condition["context"] = tokens, context
                if not attribution["accounting_passed"]:
                    condition["status"] = "profiling_failed"
                    raise RuntimeError("CUDA event accounting failed or CUPTI returned no GPU kernels; see trace")
                costs = operation_costs(config, tokens=tokens, context=context)
                condition["operations"] = []
                for name, cost in costs.items():
                    ms = attribution["gpu_ms"][name]
                    condition["operations"].append(dict(prompt_len=length, phase=phase, operation=name,
                        **cost, gpu_ms=ms, gpu_share_pct=100 * ms / attribution["gpu_sum_ms"],
                        **assess(cost, ms, calibration["compute_tflops"], calibration["copy_gbps"])))
                condition["status"] = "passed"
                write_report(output, result)
                del cache
        result["status"] = "passed" if all(c["status"] == "passed" for c in result["conditions"]) else "validation_failed"
    except Exception as exc:
        result.update(status="error", error=type(exc).__name__ + ": " + str(exc))
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_report(output, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-minutes", type=float, default=110)
    result = run_roofline(**vars(parser.parse_args(argv)))
    print("roofline:", result["status"], result.get("error", ""), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
