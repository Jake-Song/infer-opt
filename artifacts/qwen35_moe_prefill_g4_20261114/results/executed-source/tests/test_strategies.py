import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F

from infer_opt.config import ModelConfig, RunConfig
from infer_opt.runner import check_cached_decode
from infer_opt.strategies import (
    available_attention_strategies,
    causal_mask,
    expand_kv,
    make_attention_strategy,
    register_attention_strategy,
)

CONFIG = ModelConfig(
    vocab_size=64, hidden_size=32, n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=64, max_seq_len=32
)
#: Strategies exercised on CPU. ``decode_nomask`` is excluded because it only
#: serves a single query, and ``sdpa_mem_efficient`` because its backend is
#: CUDA-only -- pinning it on CPU raises "no viable backend". Both are covered
#: by :func:`test_every_strategy_agrees_on_cuda` where the hardware allows it.
PREFILL_STRATEGIES = ("eager_sdpa", "sdpa_math", "gqa_native")
ALL_STRATEGIES = (*PREFILL_STRATEGIES, "sdpa_mem_efficient", "decode_nomask")


def _qkv(*, query_len: int, key_len: int, n_heads: int = 4, n_kv_heads: int = 2, head_dim: int = 8):
    torch.manual_seed(0)
    query = torch.randn(2, n_heads, query_len, head_dim)
    key = torch.randn(2, n_kv_heads, key_len, head_dim)
    value = torch.randn(2, n_kv_heads, key_len, head_dim)
    return query, key, value


def _reference(query, key, value, query_start: int) -> torch.Tensor:
    key, value = expand_kv(query, key, value)
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=causal_mask(query, key, query_start)
    )


@pytest.mark.parametrize("name", PREFILL_STRATEGIES)
def test_every_prefill_strategy_matches_the_reference(name: str) -> None:
    query, key, value = _qkv(query_len=6, key_len=6)
    produced = make_attention_strategy(name)(query, key, value, query_start=0)
    assert torch.allclose(produced, _reference(query, key, value, 0), atol=1e-6)


@pytest.mark.parametrize("name", (*PREFILL_STRATEGIES, "decode_nomask"))
def test_every_strategy_matches_the_reference_while_decoding(name: str) -> None:
    query, key, value = _qkv(query_len=1, key_len=9)
    produced = make_attention_strategy(name)(query, key, value, query_start=8)
    assert torch.allclose(produced, _reference(query, key, value, 8), atol=1e-6)


def test_dropping_the_mask_is_only_valid_for_a_single_query() -> None:
    strategy = make_attention_strategy("decode_nomask")
    query, key, value = _qkv(query_len=4, key_len=4)
    with pytest.raises(ValueError, match="one query token"):
        strategy(query, key, value, query_start=0)


def test_dropping_the_mask_requires_the_query_at_the_end_of_the_cache() -> None:
    strategy = make_attention_strategy("decode_nomask")
    query, key, value = _qkv(query_len=1, key_len=9)
    with pytest.raises(ValueError, match="end of the cache"):
        strategy(query, key, value, query_start=3)


def test_expanding_kv_is_a_no_op_without_grouping() -> None:
    query, key, value = _qkv(query_len=2, key_len=2, n_heads=4, n_kv_heads=4)
    assert expand_kv(query, key, value) == (key, value)


def test_expanding_kv_repeats_each_head_for_its_group() -> None:
    query, key, value = _qkv(query_len=2, key_len=2)
    wide_key, _ = expand_kv(query, key, value)
    assert wide_key.shape[1] == query.shape[1]
    assert torch.equal(wide_key[:, 0], wide_key[:, 1])


def test_the_causal_mask_moves_with_the_query_position() -> None:
    query, key, _ = _qkv(query_len=1, key_len=5)
    assert causal_mask(query, key, query_start=4).all()
    assert causal_mask(query, key, query_start=2).sum() == 3


@pytest.mark.parametrize("name", PREFILL_STRATEGIES)
def test_cached_decode_matches_full_forward_for_every_strategy(name: str) -> None:
    result = check_cached_decode(
        CONFIG, RunConfig(batch_size=2, prompt_len=5, decode_len=3, device="cpu", strategy=name)
    )
    assert result["passed"], result


def test_registering_a_duplicate_strategy_is_refused() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_attention_strategy(make_attention_strategy("eager_sdpa").__class__)


def test_unknown_strategies_name_the_available_ones() -> None:
    with pytest.raises(ValueError, match="unknown attention strategy"):
        make_attention_strategy("does_not_exist")
    assert "eager_sdpa" in available_attention_strategies()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_every_strategy_agrees_on_cuda(name: str) -> None:
    """Covers the two strategies CPU cannot run, at the shapes decode really uses."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    query = torch.randn(2, 8, 1, 64, device=device, dtype=torch.float16)
    key = torch.randn(2, 2, 33, 64, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    produced = make_attention_strategy(name)(query, key, value, query_start=32)
    assert torch.allclose(produced, _reference(query, key, value, 32), atol=2e-3)
