import pytest
import torch
from torch import nn

from infer_opt.quant import Int8Linear, quantize_int8, quantize_linears
from tests.test_qwen3_5 import make_model


def test_quantize_int8_error_is_at_most_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(40, 64) * torch.rand(40, 1) * 3  # rows with very different ranges
    q, scale = quantize_int8(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127
    assert torch.all(q.abs().amax(dim=1) == 127)  # every row uses the full range
    err = (q.float() * scale.float().unsqueeze(1) - w).abs()
    assert torch.all(err <= scale.float().unsqueeze(1) / 2 + 1e-6)


def test_quantize_int8_chunking_matches_unchunked(monkeypatch):
    torch.manual_seed(0)
    w = torch.randn(50, 32)
    expected = quantize_int8(w)
    monkeypatch.setattr("infer_opt.quant._ROW_CHUNK", 7)
    got = quantize_int8(w)
    assert torch.equal(expected[0], got[0]) and torch.equal(expected[1], got[1])


def test_zero_row_does_not_produce_nan():
    w = torch.randn(4, 16)
    w[2] = 0
    q, scale = quantize_int8(w)
    assert torch.all(q[2] == 0) and torch.isfinite(scale).all()
    assert torch.isfinite(Int8Linear(q, scale)(torch.randn(1, 16))).all()


def test_int8_linear_matches_dequantized_matmul_and_bias():
    torch.manual_seed(0)
    linear = nn.Linear(64, 48, bias=True)
    q = Int8Linear.from_linear(linear)
    x = torch.randn(3, 5, 64)
    expected = torch.nn.functional.linear(x, q.dequantized_weight(), linear.bias)
    assert torch.allclose(q(x), expected, atol=1e-5)
    assert torch.allclose(q(x), linear(x), atol=0.05)  # close to the original, not equal


def test_quantize_linears_selects_modules_and_counts_bytes():
    model = make_model().float()
    before = sum(p.numel() * p.element_size() for p in model.parameters())
    report = quantize_linears(model, min_out_features=8)
    assert "lm_head" in report.skipped and isinstance(model.lm_head, nn.Linear)
    assert isinstance(model.model.language_model.layers[0].mlp.gate_proj, Int8Linear)
    assert isinstance(model.model.language_model.layers[3].self_attn.q_proj, Int8Linear)
    assert all("in_proj_b" in n or "in_proj_a" in n or "lm_head" in n for n in report.skipped)  # out_features 4 < 8
    assert report.bytes_after < report.bytes_before / 3  # fp32 -> int8 (+ scales)
    after = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers() if b.dtype == torch.int8
    )
    assert after < before


def test_quantized_model_stays_close_and_cache_keeps_float_dtype():
    model = make_model()
    ids = torch.randint(0, 97, (1, 40))
    with torch.no_grad():
        reference = model(ids)
        quantize_linears(model, include_lm_head=True, min_out_features=8)
        out = model(ids)
        cache = model.new_cache(1, 64)
    assert cache.keys[3].dtype == torch.float64  # not int8, although lm_head is now an Int8Linear
    rel = (out - reference).norm() / reference.norm()
    assert rel < 0.05
    assert (out.argmax(-1) == reference.argmax(-1)).float().mean() > 0.8


def test_token_by_token_still_matches_full_forward_after_quantization():
    model = make_model()
    quantize_linears(model, min_out_features=8)
    ids = torch.randint(0, 97, (1, 70))
    with torch.no_grad():
        full = model(ids)
        cache = model.new_cache()
        steps = [model(ids[:, :40], cache)] + [model(ids[:, i : i + 1], cache) for i in range(40, 70)]
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)


def test_rejects_non_matrix_weight():
    with pytest.raises(ValueError):
        quantize_int8(torch.randn(8))
