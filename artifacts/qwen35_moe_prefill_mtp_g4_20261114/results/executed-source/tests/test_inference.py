import json

import pytest

torch = pytest.importorskip("torch")

from infer_opt.cache import KVCache
from infer_opt.cli import main
from infer_opt.config import ModelConfig, RunConfig
from infer_opt.runner import check_cached_decode


def test_cached_decode_matches_full_forward_cpu() -> None:
    result = check_cached_decode(
        ModelConfig(hidden_size=32, n_heads=4, n_kv_heads=2, intermediate_size=64, n_layers=2, max_seq_len=16),
        RunConfig(batch_size=2, prompt_len=5, decode_len=3, device="cpu"),
    )
    assert result["passed"], result


def test_cache_rejects_capacity_overflow() -> None:
    cache = KVCache(
        n_layers=1, batch_size=1, n_kv_heads=1, max_seq_len=2, head_dim=4, device=torch.device("cpu"), dtype=torch.float32
    )
    cache.advance(2)
    with pytest.raises(ValueError, match="capacity exceeded"):
        cache.advance(1)
    cache.reset()
    key = torch.zeros((1, 1, 1, 4))
    cache.write(0, key, key, start=0)
    cached_key, cached_value = cache.read(0, end=1)
    assert cached_key.shape == (1, 1, 1, 4)
    assert torch.equal(cached_key, cached_value)


def test_check_cli_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", "--hidden-size", "32", "--n-heads", "4", "--n-kv-heads", "2", "--intermediate-size", "64", "--max-seq-len", "16", "--prompt-len", "4", "--decode-len", "2", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
