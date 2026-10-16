import torch

from infer_opt.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def tiny_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        rope_theta=10000.0, partial_rotary_factor=0.25, mrope_section=[1, 1, 0],
    )


def make_model() -> Qwen3_5ForCausalLM:
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(tiny_config()).double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.randn_like(p) * 0.5)
            elif "norm" in name or name.endswith("dt_bias"):
                p.copy_(torch.randn_like(p) * 0.1)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
    return model


def test_token_by_token_matches_full_forward():
    model = make_model()
    ids = torch.randint(0, 97, (2, 150))  # >2 delta-rule chunks of 64
    with torch.no_grad():
        full = model(ids)
        cache = model.new_cache()
        steps = [model(ids[:, :100], cache)] + [model(ids[:, i : i + 1], cache) for i in range(100, 150)]
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)  # delta rule runs in fp32 internally
    assert cache.seen_tokens == 150


def test_chunked_prefill_matches_full_forward():
    model = make_model()
    ids = torch.randint(0, 97, (1, 140))
    with torch.no_grad():
        full = model(ids)
        cache = model.new_cache()
        parts = [model(ids[:, a:b], cache) for a, b in [(0, 37), (37, 101), (101, 140)]]
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-4)  # delta rule runs in fp32 internally


def test_short_prompt_smaller_than_conv_kernel():
    model = make_model()
    ids = torch.randint(0, 97, (1, 2))
    with torch.no_grad():
        full = model(torch.cat([ids, ids], dim=1))
        cache = model.new_cache()
        out = torch.cat([model(ids, cache), model(ids[:, :1], cache), model(ids[:, 1:], cache)], dim=1)
    assert torch.allclose(full, out, atol=1e-4)  # delta rule runs in fp32 internally


def test_cached_greedy_matches_uncached():
    model = make_model()
    ids = torch.randint(0, 97, (1, 9))
    assert torch.equal(model.generate_greedy(ids, 12, use_cache=True), model.generate_greedy(ids, 12, use_cache=False))


def test_recurrent_state_is_constant_size():
    model = make_model()
    cache = model.new_cache()
    with torch.no_grad():
        model(torch.randint(0, 97, (1, 10)), cache)
        before = [t.numel() for t in cache.recurrent_states if t is not None]
        model(torch.randint(0, 97, (1, 1)), cache)
    assert before == [t.numel() for t in cache.recurrent_states if t is not None]
    assert len(before) == 3
