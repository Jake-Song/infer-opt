"""Reproducible, correctness-gated baseline sweeps for the repository model."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import ModelConfig, RunConfig
from .model import DecoderOnlyTransformer, make_random_model
from .runner import benchmark_prepared, make_inputs, resolve_device, resolve_dtype


BASELINE_MODEL = ModelConfig(
    vocab_size=32768, hidden_size=2048, n_layers=8, n_heads=16,
    n_kv_heads=4, intermediate_size=5632, max_seq_len=16384,
)
DEFAULT_PROMPT_LENGTHS = (128, 512, 2048, 8192)


def _command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "infer_opt").glob("*.py"))
    paths += sorted((root / "scripts").glob("*.py"))
    paths += [root / name for name in ("pyproject.toml", "uv.lock") if (root / name).exists()]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    identity = {
        "sha256": digest, "files": hashes,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"], root),
        "git_status": _command_output(["git", "status", "--porcelain"], root),
    }
    manifest = root / "source_manifest.json"
    if manifest.exists():
        original = json.loads(manifest.read_text())
        if original["sha256"] != digest:
            raise ValueError("uploaded source does not match source_manifest.json")
        identity["git_commit"] = original["git_commit"]
        identity["git_status"] = original["git_status"]
    return identity


def environment(device: torch.device) -> dict[str, object]:
    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
        "device": str(device), "cpu_threads": torch.get_num_threads(),
        "execution": "eager", "sdpa_policy": "automatic PyTorch dispatch",
        "sdpa_enabled": {name: getattr(torch.backends.cuda, name + "_sdp_enabled")()
                         for name in ("flash", "mem_efficient", "math")},
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update({
            "gpu_name": properties.name, "gpu_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "sm_count": properties.multi_processor_count,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "nvidia_smi": _command_output([
                "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,pstate,clocks.sm,clocks.mem",
                "--format=csv,noheader",
            ]),
        })
    return result


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> dict[str, object]:
    """Check every element, converting small sequence chunks to FP32 only here."""
    if actual.shape != expected.shape:
        raise ValueError(f"validation shape mismatch: {actual.shape} != {expected.shape}")
    max_error, squared_error, squared_reference, mismatches = 0.0, 0.0, 0.0, 0
    for start in range(0, actual.shape[-2], 128):
        a = actual[..., start : start + 128, :].float()
        b = expected[..., start : start + 128, :].float()
        finite = torch.isfinite(a) & torch.isfinite(b)
        error = (a - b).abs()
        valid_error = torch.where(finite, error, torch.zeros_like(error))
        max_error = max(max_error, valid_error.max().item())
        mismatches += ((~finite) | (error > tolerance + tolerance * b.abs())).sum().item()
        squared_error += valid_error.square().sum().item()
        squared_reference += torch.where(finite, b, torch.zeros_like(b)).square().sum().item()
    return {
        "passed": mismatches == 0, "elements": actual.numel(), "mismatches": mismatches,
        "max_abs_error": max_error,
        "relative_l2": (squared_error / max(squared_reference, 1e-24)) ** 0.5,
    }


@torch.inference_mode()
def validate_prepared(
    model: DecoderOnlyTransformer, prompt: torch.Tensor, continuation: torch.Tensor,
) -> dict[str, object]:
    """Full causal forward reference, all logits, every layer K/V, prefix integrity."""
    device, dtype = prompt.device, next(model.parameters()).dtype
    tolerance = 2e-4 if dtype == torch.float32 else 2e-2
    tokens = torch.cat((prompt, continuation), dim=1)
    reference = model.new_cache(prompt.shape[0], device, dtype)
    cache = model.new_cache(prompt.shape[0], device, dtype)
    # A single full causal forward also captures the reference K/V at each layer.
    full_logits = model.forward_prefill(tokens, reference)
    prompt_len, total_len = prompt.shape[1], tokens.shape[1]
    prefill_logits = model.forward_prefill(prompt, cache)
    prefill = compare_tensors(prefill_logits, full_logits[:, :prompt_len], tolerance)
    del prefill_logits
    prefill_kv = []
    for layer in range(model.config.n_layers):
        for name in ("keys", "values"):
            prefill_kv.append({"layer": layer, "kind": name, **compare_tensors(
                getattr(cache, name)[layer, :, :, :prompt_len],
                getattr(reference, name)[layer, :, :, :prompt_len], tolerance,
            )})
    prefix = (cache.keys[..., :prompt_len, :].clone(), cache.values[..., :prompt_len, :].clone())
    decode = []
    for index in range(continuation.shape[1]):
        logits = model.forward_decode(continuation[:, index : index + 1], cache)
        decode.append({"step": index + 1, **compare_tensors(
            logits, full_logits[:, prompt_len + index : prompt_len + index + 1], tolerance,
        )})
    final_kv = []
    for layer in range(model.config.n_layers):
        for name in ("keys", "values"):
            final_kv.append({"layer": layer, "kind": name, **compare_tensors(
                getattr(cache, name)[layer, :, :, :total_len],
                getattr(reference, name)[layer, :, :, :total_len], tolerance,
            )})
    prefix_preserved = torch.equal(prefix[0], cache.keys[..., :prompt_len, :]) and torch.equal(
        prefix[1], cache.values[..., :prompt_len, :]
    )
    position_correct = cache.position == total_len
    return {
        "passed": all(item["passed"] for item in [prefill, *decode, *prefill_kv, *final_kv])
                  and prefix_preserved and position_correct,
        "reference": "single full causal forward with KV capture",
        "atol": tolerance, "rtol": tolerance, "comparison_dtype": "float32",
        "prefill_logits": prefill, "decode_logits": decode,
        "prefill_kv": prefill_kv, "final_kv": final_kv,
        "prefix_preserved": prefix_preserved, "position_correct": position_correct,
        "input_tokens_sha256": hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_artifacts(output: Path, result: dict[str, object]) -> None:
    _write_json(output / "results.json", result)
    columns = ["prompt_len", "decode_len", "status", "prefill_wall_median_ms", "prefill_wall_p90_ms",
               "prefill_tokens_per_second", "decode_wall_median_ms", "decode_wall_median_ms_per_token",
               "decode_wall_p90_ms_per_token", "decode_tokens_per_second",
               "prefill_cuda_median_ms", "decode_cuda_median_ms_per_token"]
    rows = []
    for condition in result["conditions"]:
        row = {"prompt_len": condition["prompt_len"], "decode_len": result["run"]["decode_len"],
               "status": condition["status"]}
        if "benchmark" in condition:
            measured = condition["benchmark"]
            p, d = measured["prefill"], measured["decode"]
            length = measured["run"]["decode_len"]
            row.update({
                "prefill_wall_median_ms": p["wall_ms"]["median"],
                "prefill_wall_p90_ms": p["wall_ms"]["p90"],
                "prefill_tokens_per_second": 1000 * condition["prompt_len"] / p["wall_ms"]["median"],
                "decode_wall_median_ms": d["wall_ms"]["median"],
                "decode_wall_median_ms_per_token": d["wall_ms"]["median"] / length,
                "decode_wall_p90_ms_per_token": d["wall_ms"]["p90"] / length,
                "decode_tokens_per_second": 1000 * length / d["wall_ms"]["median"],
                "prefill_cuda_median_ms": p["cuda_ms"]["median"] if p["cuda_ms"] else None,
                "decode_cuda_median_ms_per_token": d["cuda_ms"]["median"] / length if d["cuda_ms"] else None,
            })
        rows.append(row)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    env = result.get("environment", {})
    lines = ["# Prefill / decode 베이스라인", "", f"상태: **{result['status']}**", "",
             f"환경: {env.get('gpu_name', env.get('device', '확인 전'))}, "
             f"PyTorch {env.get('torch', '?')}, CUDA {env.get('cuda_build', '?')}", "",
             f"모델: {result.get('parameter_count', '?'):,} parameters" if "parameter_count" in result else "모델 준비 전",
             f"설정: `{json.dumps(result['model'], sort_keys=True)}`", "",
             f"B=1, {result['run']['dtype']}, eager_sdpa, seed={result['run']['seed']}, "
             f"warmup={result['run']['warmup']}, 측정={result['run']['iterations']}회, "
             f"고정 입력 decode={result['run']['decode_len']}단계.", "",
             "## 측정 결과", "",
             "| Prompt | 상태 | Prefill ms | Prefill tokens/s | Decode ms/token | Decode tokens/s |",
             "|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        metrics = [f"{row[key]:.3f}" if key in row else "—" for key in (
            "prefill_wall_median_ms", "prefill_tokens_per_second", "decode_wall_median_ms_per_token", "decode_tokens_per_second")]
        lines.append(f"| {row['prompt_len']} | {row['status']} | " + " | ".join(metrics) + " |")
    lines += ["", "## 해석과 측정 범위", "",
              "표는 동기화한 wall time의 중앙값이다. Decode ms/token은 각 반복의 전체 decode 시간을 단계 수로 나눈 값이다.",
              "CSV의 p90은 반복별 시간 분포이며, 개별 토큰의 지연시간 p90이 아니다. 처리량은 중앙값 시간으로 계산했다.",
              "Prefill은 prompt의 모든 위치에 대한 vocabulary logits와 KV 쓰기를 포함한다. Decode는 실제로 KV cache를 늘린다.",
              "가중치는 랜덤이며 입력은 고정 토큰 replay다. 토큰 선택, tokenizer, HTTP, 모델 로딩, 캐시 할당·초기화, "
              "warmup, 검증과 decode 준비용 prefill은 측정에서 제외했다.",
              "Wall time에는 Python dispatch, CUDA event 기록과 마지막 동기화 비용이 포함된다. CUDA event 경과 시간도 "
              "host launch 사이의 빈 시간을 포함할 수 있어 kernel 실행시간의 합으로 해석하지 않는다.",
              "정확성 검증은 전체 logits, 모든 레이어의 유효 K/V, cache prefix 보존과 최종 위치를 확인한다. "
              "검증 실패 조건에는 성능 결과를 싣지 않는다.", "",
              "## 재현", "", "```bash", result["reproduce_command"], "```", "",
              f"소스 SHA-256: `{result.get('source', {}).get('sha256', 'unknown')}`", "",
              "전체 환경, 정확성 검사와 반복별 wall/CUDA 시간은 `results.json`, 요약값은 `summary.csv`에 있다."]
    if "error" in result:
        lines += ["", "실행 오류: " + result["error"]]
    for condition in result["conditions"]:
        if "error" in condition:
            lines += ["", f"Prompt {condition['prompt_len']} 오류: {condition['error']}"]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run_baseline(
    output: Path, *, prompt_lengths: tuple[int, ...] = DEFAULT_PROMPT_LENGTHS,
    decode_len: int = 128, warmup: int = 3, iterations: int = 10, seed: int = 0,
    device: str = "cuda", dtype: str = "bfloat16", expected_gpu: str | None = None,
    budget_minutes: float = 120, model_config: ModelConfig = BASELINE_MODEL,
) -> dict[str, object]:
    if not prompt_lengths or len(set(prompt_lengths)) != len(prompt_lengths):
        raise ValueError("prompt lengths must be nonempty and unique")
    if budget_minutes <= 0:
        raise ValueError("budget_minutes must be positive")
    runs = [RunConfig(prompt_len=length, decode_len=decode_len, warmup=warmup,
                      iterations=iterations, seed=seed, device=device, dtype=dtype) for length in prompt_lengths]
    if max(prompt_lengths) + decode_len > model_config.max_seq_len:
        raise ValueError("prompt_len + decode_len must not exceed max_seq_len")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("results.json", "summary.csv", "report.md")):
        raise FileExistsError(f"baseline results already exist: {output}")
    command = ["python", "-m", "infer_opt", "baseline", "--output", str(output), "--device", device,
               "--dtype", dtype, "--prompt-lengths", *map(str, prompt_lengths), "--decode-len", str(decode_len),
               "--warmup", str(warmup), "--iterations", str(iterations), "--seed", str(seed),
               "--budget-minutes", str(budget_minutes)]
    if expected_gpu:
        command += ["--expected-gpu", expected_gpu]
    result = {
        "schema_version": 2, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "model": model_config.to_dict(), "run": runs[0].to_dict(),
        "prompt_lengths": list(prompt_lengths), "conditions": [], "reproduce_command": shlex.join(command),
    }
    started = time.monotonic()
    try:
        resolved = resolve_device(device)
        result["environment"] = environment(resolved)
        result["source"] = source_identity()
        if expected_gpu and expected_gpu.lower() not in result["environment"].get("gpu_name", "").lower():
            raise RuntimeError(f"expected {expected_gpu}, got {result['environment'].get('gpu_name', resolved)}")
        if resolved.type == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support BF16")
        model = make_random_model(model_config, "eager_sdpa", seed).to(device=resolved, dtype=resolve_dtype(dtype)).eval()
        result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        write_artifacts(output, result)
        for run in runs:
            if time.monotonic() - started >= budget_minutes * 60:
                raise TimeoutError("baseline time budget exceeded before next condition")
            condition = {"prompt_len": run.prompt_len, "status": "validating"}
            result["conditions"].append(condition)
            write_artifacts(output, result)
            print(f"prompt={run.prompt_len}: validating", flush=True)
            try:
                prompt, continuation = make_inputs(model_config, run, resolved)
                validation = validate_prepared(model, prompt, continuation)
                condition["validation"] = validation
                if not validation["passed"]:
                    condition["status"] = "validation_failed"
                else:
                    print(f"prompt={run.prompt_len}: measuring", flush=True)
                    condition["benchmark"] = benchmark_prepared(model, run, prompt, continuation)
                    condition["status"] = "passed"
            except Exception as exc:
                condition["status"] = "error"
                condition["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                write_artifacts(output, result)
            print(f"prompt={run.prompt_len}: {condition['status']}", flush=True)
        result["status"] = "passed" if all(c["status"] == "passed" for c in result["conditions"]) else "validation_failed"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_artifacts(output, result)
    return result
