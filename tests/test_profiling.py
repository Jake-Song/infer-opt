import pytest

torch = pytest.importorskip("torch")

from infer_opt.config import ModelConfig
from infer_opt.model import make_random_model
from infer_opt.profiling import (
    Workload,
    kv_cache_bytes,
    parameter_count,
    ridge_point,
    roofline_ceiling,
    step_workload,
    timeit,
    transformer_bytes,
    transformer_flops,
)

CONFIG = ModelConfig(
    vocab_size=64, hidden_size=32, n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=64, max_seq_len=32
)


def test_parameter_count_matches_built_model() -> None:
    model = make_random_model(CONFIG, "eager_sdpa", seed=0)
    assert parameter_count(CONFIG) == sum(p.numel() for p in model.parameters())


def test_kv_cache_bytes_matches_allocated_cache() -> None:
    model = make_random_model(CONFIG, "eager_sdpa", seed=0)
    cache = model.new_cache(3, torch.device("cpu"), torch.float32)
    allocated = cache.keys.numel() * cache.keys.element_size() + cache.values.numel() * cache.values.element_size()
    assert kv_cache_bytes(CONFIG, batch=3, context_len=CONFIG.max_seq_len, dtype=torch.float32) == allocated


def test_linear_flops_follow_hand_count() -> None:
    hidden, kv_width = CONFIG.hidden_size, CONFIG.n_kv_heads * CONFIG.head_dim
    per_token = 2 * hidden * (2 * hidden + 2 * kv_width + 3 * CONFIG.intermediate_size)
    flops = transformer_flops(CONFIG, batch=2, new_tokens=8, context_len=8)
    assert flops["linear"] == CONFIG.n_layers * 2 * 8 * per_token
    assert flops["lm_head"] == 2 * (2 * 8) * hidden * CONFIG.vocab_size
    assert flops["total"] == flops["linear"] + flops["attention"] + flops["lm_head"]


def test_decode_attention_costs_one_row_of_prefill() -> None:
    prefill = transformer_flops(CONFIG, batch=1, new_tokens=8, context_len=8)
    decode_rows = [transformer_flops(CONFIG, batch=1, new_tokens=1, context_len=s) for s in range(1, 9)]
    assert prefill["attention"] == sum(step["attention"] for step in decode_rows)
    assert prefill["linear"] == 8 * decode_rows[0]["linear"]


def test_decode_is_weight_dominated_while_prefill_is_not() -> None:
    decode = transformer_bytes(CONFIG, batch=1, new_tokens=1, context_len=16, dtype=torch.float16)
    prefill = transformer_bytes(CONFIG, batch=1, new_tokens=16, context_len=16, dtype=torch.float16)
    assert decode["weights"] / decode["total"] > prefill["weights"] / prefill["total"]
    assert decode["kv_write"] == decode["kv_read"] / 16


def test_decode_intensity_rises_with_batch_and_stays_below_prefill() -> None:
    intensities = [
        step_workload(CONFIG, batch=b, new_tokens=1, context_len=16, dtype=torch.float16).arithmetic_intensity
        for b in (1, 2, 4, 8)
    ]
    assert intensities == sorted(intensities)
    prefill = step_workload(CONFIG, batch=1, new_tokens=16, context_len=16, dtype=torch.float16)
    assert prefill.arithmetic_intensity > intensities[0]


def test_workload_intensity_is_flops_per_byte() -> None:
    assert Workload(flops=200, bytes=50).arithmetic_intensity == 4.0


@pytest.mark.parametrize(
    "batch, new_tokens, context_len",
    [(0, 1, 4), (1, 0, 4), (1, 8, 4)],
)
def test_invalid_steps_are_rejected(batch: int, new_tokens: int, context_len: int) -> None:
    with pytest.raises(ValueError):
        transformer_flops(CONFIG, batch=batch, new_tokens=new_tokens, context_len=context_len)


def test_roofline_switches_limit_at_the_ridge_point() -> None:
    peak_tflops, peak_gbps = 10.0, 500.0
    ridge = ridge_point(peak_tflops, peak_gbps)
    assert ridge == pytest.approx(20.0)
    assert roofline_ceiling(ridge / 2, peak_tflops, peak_gbps) == pytest.approx(peak_tflops / 2)
    assert roofline_ceiling(ridge * 2, peak_tflops, peak_gbps) == pytest.approx(peak_tflops)


def test_timeit_runs_every_iteration_on_cpu() -> None:
    calls = 0

    def work() -> None:
        nonlocal calls
        calls += 1

    timing = timeit(work, device=torch.device("cpu"), warmup=2, iters=5)
    assert calls == 7
    assert timing.iters == 5
    assert timing.ms >= timing.ms_min >= 0
    assert timing.tflops(1e12) == pytest.approx(1000 / timing.ms, rel=1e-6)


def test_capture_graph_is_a_passthrough_on_cpu() -> None:
    from infer_opt.profiling import capture_graph

    calls = 0

    def step() -> None:
        nonlocal calls
        calls += 1

    replay = capture_graph(step, device=torch.device("cpu"))
    replay()
    assert calls == 1
