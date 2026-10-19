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


def test_static_cache_matches_dynamic_cache():
    model = make_model()
    ids = torch.randint(0, 97, (2, 70))
    with torch.no_grad():
        dyn, sta = model.new_cache(), model.new_cache(batch_size=2, max_seq_len=80)
        out_d = [model(ids[:, :60], dyn)] + [model(ids[:, i : i + 1], dyn) for i in range(60, 70)]
        out_s = [model(ids[:, :60], sta)] + [model(ids[:, i : i + 1], sta) for i in range(60, 70)]
    assert torch.equal(torch.cat(out_d, 1), torch.cat(out_s, 1))
    assert sta.keys[3].shape[2] == 80 and sta.seen_tokens == 70


def test_static_cache_buffers_are_updated_in_place():
    model = make_model()
    cache = model.new_cache(max_seq_len=16)
    addresses = [t.data_ptr() for t in cache.keys + cache.conv_states + cache.recurrent_states if t is not None]
    with torch.no_grad():
        model(torch.randint(0, 97, (1, 5)), cache)
        model(torch.randint(0, 97, (1, 1)), cache)
    assert addresses == [t.data_ptr() for t in cache.keys + cache.conv_states + cache.recurrent_states if t is not None]


def test_static_cache_reset_and_overflow():
    model = make_model()
    ids = torch.randint(0, 97, (1, 12))
    cache = model.new_cache(max_seq_len=12)
    with torch.no_grad():
        first = model(ids, cache)
        cache.reset()
        second = model(ids, cache)  # a dirty cache would change this
        assert torch.equal(first, second)
        try:
            model(ids[:, :1], cache)
        except ValueError:
            pass
        else:
            raise AssertionError("expected overflow error")


def test_static_greedy_matches_dynamic_greedy():
    model = make_model()
    ids = torch.randint(0, 97, (1, 9))
    assert torch.equal(model.generate_greedy(ids, 12), model.generate_greedy(ids, 12, static_cache=True))


def test_fixed_shape_decode_matches_sliced_decode():
    model = make_model()
    ids = torch.randint(0, 97, (2, 70))
    with torch.no_grad():
        ref_cache = model.new_cache(batch_size=2, max_seq_len=90)
        fix_cache = model.new_cache(batch_size=2, max_seq_len=90)
        model(ids[:, :40], ref_cache)
        model(ids[:, :40], fix_cache)
        assert fix_cache.position.item() == 40
        for i in range(40, 70):
            want = model(ids[:, i : i + 1], ref_cache)
            got = model.decode_step(ids[:, i : i + 1], fix_cache)
            assert torch.allclose(want, got, atol=1e-4), i
    assert fix_cache.position.item() == fix_cache.seen_tokens == 70


def test_fixed_shape_decode_after_one_token_prompt_and_reset():
    model = make_model()
    cache = model.new_cache(max_seq_len=12)
    ids = torch.randint(0, 97, (1, 12))
    with torch.no_grad():
        first = [model(ids[:, :1], cache)] + [model.decode_step(ids[:, i : i + 1], cache) for i in range(1, 12)]
        cache.reset()
        assert cache.position.item() == 0
        second = [model(ids[:, :1], cache)] + [model.decode_step(ids[:, i : i + 1], cache) for i in range(1, 12)]
    assert torch.equal(torch.cat(first, 1), torch.cat(second, 1))
    full = model(ids)
    assert torch.allclose(full, torch.cat(first, 1), atol=1e-4)


def test_decode_step_requires_static_cache_and_checks_capacity():
    model = make_model()
    ids = torch.randint(0, 97, (1, 4))
    with torch.no_grad():
        dynamic = model.new_cache()
        model(ids, dynamic)
        for bad in (lambda: model.decode_step(ids[:, :1], dynamic),):
            try:
                bad()
            except ValueError:
                pass
            else:
                raise AssertionError("expected ValueError for a dynamic cache")
        cache = model.new_cache(max_seq_len=5)
        model(ids, cache)
        model.decode_step(ids[:, :1], cache)
        try:
            model.decode_step(ids[:, :1], cache)
        except ValueError:
            pass
        else:
            raise AssertionError("expected a capacity error")


def test_generate_with_fixed_shape_decoder_matches_default_generate():
    model = make_model()
    ids = torch.randint(0, 97, (1, 9))
    assert torch.equal(model.generate_greedy(ids, 14), model.generate_greedy(ids, 14, decode_mode="eager"))


def test_layer_summary_describes_layout():
    assert tiny_config().layer_summary() == {
        "num_layers": 4, "linear_attention": 3, "full_attention": 1, "pattern": "LLLF",
    }
