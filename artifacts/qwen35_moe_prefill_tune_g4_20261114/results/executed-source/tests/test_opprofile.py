import pytest

torch = pytest.importorskip("torch")

from infer_opt.config import ModelConfig, RunConfig
from infer_opt.opprofile import StageBreakdown, profile_stages, stage_bytes, stage_flops
from infer_opt.profiling import transformer_bytes, transformer_flops
from infer_opt.regions import REGION_NAMES, profiled_regions, region, regions_enabled
from infer_opt.runner import prepare

CONFIG = ModelConfig(
    vocab_size=64, hidden_size=32, n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=64, max_seq_len=32
)
STEPS = [
    {"batch": 2, "new_tokens": 8, "context_len": 8},
    {"batch": 1, "new_tokens": 1, "context_len": 16},
    {"batch": 3, "new_tokens": 4, "context_len": 12},
]


@pytest.mark.parametrize("step", STEPS)
def test_stage_flops_sum_to_the_whole_step(step: dict[str, int]) -> None:
    stages = stage_flops(CONFIG, **step)
    assert set(stages) == set(REGION_NAMES)
    assert sum(stages.values()) == transformer_flops(CONFIG, **step)["total"]


@pytest.mark.parametrize("step", STEPS)
def test_stage_bytes_sum_to_the_whole_step(step: dict[str, int]) -> None:
    stages = stage_bytes(CONFIG, dtype=torch.float16, **step)
    assert set(stages) == set(REGION_NAMES)
    assert sum(stages.values()) == transformer_bytes(CONFIG, dtype=torch.float16, **step)["total"]


def test_parameter_free_stages_count_no_flops() -> None:
    """Norms, RoPE and the cache moves do no arithmetic worth counting.

    The chapter leans on this: any time they take is traffic or launch cost.
    """
    stages = stage_flops(CONFIG, batch=1, new_tokens=1, context_len=8)
    assert [stages[name] for name in ("embedding", "norm", "rope", "kv_read", "kv_write")] == [0] * 5
    assert stages["attention"] > 0 and stages["mlp"] > 0


def test_stage_bytes_charge_the_kv_cache_to_the_cache_stages() -> None:
    short = stage_bytes(CONFIG, batch=1, new_tokens=1, context_len=4, dtype=torch.float16)
    long = stage_bytes(CONFIG, batch=1, new_tokens=1, context_len=16, dtype=torch.float16)
    assert long["kv_read"] > short["kv_read"]
    assert long["kv_write"] == short["kv_write"]
    assert long["mlp"] == short["mlp"]


def test_regions_are_disabled_until_profiling() -> None:
    assert not regions_enabled()
    with profiled_regions():
        assert regions_enabled()
        with region("mlp"):
            pass
    assert not regions_enabled()


def test_profile_stages_attributes_every_region_it_runs_on_cpu() -> None:
    model, device, dtype, prompt, _ = prepare(
        CONFIG, RunConfig(batch_size=2, prompt_len=8, decode_len=2, device="cpu")
    )
    cache = model.new_cache(2, device, dtype)

    def step() -> None:
        cache.reset()
        model.forward_prefill(prompt, cache)

    with torch.inference_mode():
        breakdown = profile_stages(step, device=device, warmup=1, iters=2)

    assert isinstance(breakdown, StageBreakdown)
    assert set(breakdown.stages) == set(REGION_NAMES)
    assert breakdown.device_ms == pytest.approx(sum(breakdown.stages.values()))
    assert breakdown.busiest in REGION_NAMES
    assert pytest.approx(1.0) == sum(breakdown.shares.values())
    for name in ("qkv_proj", "attention", "o_proj", "mlp", "lm_head"):
        assert breakdown.stages[name] > 0, f"{name} was never attributed any time"


def test_profile_stages_rejects_impossible_iteration_counts() -> None:
    with pytest.raises(ValueError, match="iters must be positive"):
        profile_stages(lambda: None, device=torch.device("cpu"), iters=0)
