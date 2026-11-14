import json
from dataclasses import replace

import pytest
import torch

from infer_opt import baseline, runner
from infer_opt.config import ModelConfig, RunConfig


CONFIG = ModelConfig(vocab_size=32, hidden_size=16, n_layers=2, n_heads=2,
                     n_kv_heads=1, intermediate_size=32, max_seq_len=16)
RUN = RunConfig(prompt_len=4, decode_len=3, iterations=2, warmup=1, device="cpu")


def test_benchmark_excludes_setup_and_restarts_growing_cache(monkeypatch):
    model, device, _, prompt, continuation = runner.prepare(CONFIG, RUN)
    clock = [0.0]
    positions = []
    allocations = []
    original_cache = runner.make_cache
    original_prefill, original_decode = model.forward_prefill, model.forward_decode

    def new_cache(*args):
        clock[0] += 100
        allocations.append(True)
        return original_cache(*args)

    def prefill(tokens, cache):
        assert cache.position == 0
        clock[0] += 10
        return original_prefill(tokens, cache)

    def decode(tokens, cache):
        positions.append(cache.position)
        clock[0] += 1
        return original_decode(tokens, cache)

    monkeypatch.setattr(runner, "make_cache", new_cache)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(model, "forward_prefill", prefill)
    monkeypatch.setattr(model, "forward_decode", decode)
    result = runner.benchmark_prepared(model, RUN, prompt, continuation)
    assert len(allocations) == 1
    assert positions == [4, 5, 6] * (RUN.warmup + RUN.iterations)
    assert [s["wall_ms"] for s in result["prefill"]["samples"]] == [10000, 10000]
    assert [s["wall_ms"] for s in result["decode"]["samples"]] == [3000, 3000]
    assert result["decode"]["latency_ms_per_token"] == 1000
    assert result["decode"]["cuda_ms"] is None
    assert result["schema_version"] == 2


def test_validation_checks_all_logits_and_kv():
    model, _, _, prompt, continuation = runner.prepare(CONFIG, RUN)
    result = baseline.validate_prepared(model, prompt, continuation)
    assert result["passed"], result
    assert result["prefill_logits"]["elements"] == RUN.prompt_len * CONFIG.vocab_size
    assert len(result["decode_logits"]) == RUN.decode_len
    assert len(result["prefill_kv"]) == len(result["final_kv"]) == 2 * CONFIG.n_layers
    assert all(v["elements"] == (RUN.prompt_len + RUN.decode_len) * CONFIG.head_dim
               for v in result["final_kv"])


@pytest.mark.parametrize("corruption", ["logits", "cache"])
def test_validation_detects_intermediate_logits_or_cache_corruption(monkeypatch, corruption):
    model, _, _, prompt, continuation = runner.prepare(CONFIG, RUN)
    original = model.forward_decode

    def corrupt(tokens, cache):
        logits = original(tokens, cache)
        if cache.position == RUN.prompt_len + 1:
            if corruption == "logits":
                logits[..., 0] += 1
            else:
                cache.keys[0, 0, 0, 0, 0] += 1
        return logits

    monkeypatch.setattr(model, "forward_decode", corrupt)
    result = baseline.validate_prepared(model, prompt, continuation)
    assert not result["passed"]
    if corruption == "logits":
        assert not result["decode_logits"][0]["passed"]
    else:
        assert not result["prefix_preserved"]
        assert not result["final_kv"][0]["passed"]


def test_nonfinite_values_fail_validation():
    values = torch.zeros(1, 2, 4)
    values[0, 0, 0] = float("nan")
    result = baseline.compare_tensors(values, values, 0.02)
    assert not result["passed"]
    assert result["mismatches"] == 1
    json.dumps(result, allow_nan=False)


def test_sweep_writes_raw_samples_and_does_not_overwrite(tmp_path):
    result = baseline.run_baseline(tmp_path, prompt_lengths=(2, 4), decode_len=2,
                                   warmup=0, iterations=2, device="cpu", dtype="float32",
                                   model_config=CONFIG)
    assert result["status"] == "passed", result
    saved = json.loads((tmp_path / "results.json").read_text())
    assert [c["prompt_len"] for c in saved["conditions"]] == [2, 4]
    for condition in saved["conditions"]:
        assert len(condition["benchmark"]["prefill"]["samples"]) == 2
        assert len(condition["benchmark"]["decode"]["samples"]) == 2
    assert len((tmp_path / "summary.csv").read_text().splitlines()) == 3
    assert "Decode ms/token" in (tmp_path / "report.md").read_text()
    with pytest.raises(FileExistsError):
        baseline.run_baseline(tmp_path, prompt_lengths=(2,), decode_len=2, model_config=CONFIG)


def test_failed_validation_blocks_timing_and_saves_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(baseline, "validate_prepared", lambda *args: {"passed": False, "reason": "test"})

    def forbidden(*args):
        pytest.fail("a failed validation must block performance measurement")

    monkeypatch.setattr(baseline, "benchmark_prepared", forbidden)
    result = baseline.run_baseline(tmp_path, prompt_lengths=(2,), decode_len=2, device="cpu",
                                   dtype="float32", model_config=CONFIG)
    assert result["status"] == "validation_failed"
    assert "benchmark" not in result["conditions"][0]
    assert json.loads((tmp_path / "results.json").read_text())["conditions"][0]["validation"]["reason"] == "test"


def test_capacity_failure_precedes_model_allocation(monkeypatch, tmp_path):
    monkeypatch.setattr(baseline, "make_random_model", lambda *a: pytest.fail("must reject before allocation"))
    with pytest.raises(ValueError, match="max_seq_len"):
        baseline.run_baseline(tmp_path, prompt_lengths=(16,), decode_len=1, model_config=CONFIG)


def test_wrong_hardware_fails_with_artifact(tmp_path):
    result = baseline.run_baseline(tmp_path, prompt_lengths=(2,), decode_len=1,
                                   device="cpu", expected_gpu="A100", model_config=CONFIG)
    assert result["status"] == "error"
    assert "expected A100" in result["error"]
    assert result["conditions"] == []


def test_statistics_interpolate_p90():
    assert runner.timing_summary([10, 20])["p90"] == 19
    assert runner.timing_summary([10])["p90"] == 10


def test_cli_failure_returns_nonzero_and_saves_diagnostic(tmp_path):
    from infer_opt.cli import main

    assert main(["baseline", "--output", str(tmp_path), "--device", "cpu", "--expected-gpu", "A100"]) == 1
    assert json.loads((tmp_path / "results.json").read_text())["status"] == "error"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_validation_and_timing():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unavailable")
    run = replace(RUN, device="cuda", dtype="bfloat16")
    model, _, _, prompt, continuation = runner.prepare(CONFIG, run)
    assert baseline.validate_prepared(model, prompt, continuation)["passed"]
    result = runner.benchmark_prepared(model, run, prompt, continuation)
    assert result["prefill"]["cuda_ms"]["median"] > 0
    assert result["decode"]["cuda_ms"]["median"] > 0
