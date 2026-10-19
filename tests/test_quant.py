import pytest
import torch
from torch import nn

from infer_opt.quant import Int8Linear, int8_linear, quantize_int8, quantize_linears
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


def test_group_wise_scales_cover_each_group_and_beat_per_channel():
    torch.manual_seed(0)
    w = torch.randn(24, 256)
    w[3, 5] = 40.0  # one outlier: per channel it coarsens the whole row, per group only its 64 columns
    q, scale = quantize_int8(w, group_size=64)
    assert scale.shape == (24, 4) and q.dtype == torch.int8
    groups = q.view(24, 4, 64)
    assert torch.all(groups.abs().amax(-1) == 127)  # every group uses the full range
    deq = Int8Linear(q, scale).dequantized_weight()
    err = (deq - w).view(24, 4, 64).abs()
    assert torch.all(err <= scale.unsqueeze(-1) / 2 + 1e-6)
    per_channel = Int8Linear(*quantize_int8(w)).dequantized_weight()
    assert (deq - w)[3, 64:].norm() < (per_channel - w)[3, 64:].norm() / 4  # the groups without the outlier
    assert (deq - w).norm() < (per_channel - w).norm()


def test_group_wise_chunking_matches_unchunked(monkeypatch):
    torch.manual_seed(0)
    w = torch.randn(50, 32)
    expected = quantize_int8(w, group_size=8)
    monkeypatch.setattr("infer_opt.quant._ROW_CHUNK", 7)
    got = quantize_int8(w, group_size=8)
    assert torch.equal(expected[0], got[0]) and torch.equal(expected[1], got[1])


def test_group_wise_linear_matches_dequantized_matmul():
    torch.manual_seed(0)
    linear = nn.Linear(64, 48, bias=True)
    q = Int8Linear.from_linear(linear, group_size=16)
    assert q.group_size == 16 and "group_size=16" in repr(q)
    x = torch.randn(3, 5, 64)
    expected = torch.nn.functional.linear(x, q.dequantized_weight(), linear.bias)
    assert torch.allclose(q(x), expected, atol=1e-5)
    assert torch.allclose(q(x), linear(x), atol=0.05)


def test_group_size_must_divide_in_features():
    with pytest.raises(ValueError):
        quantize_int8(torch.randn(8, 48), group_size=32)
    model = nn.Sequential(nn.Linear(48, 300), nn.Linear(300, 256))
    report = quantize_linears(model, group_size=32)
    assert isinstance(model[0], nn.Linear) and "not divisible" in report.skipped["0"]
    assert isinstance(model[1], nn.Linear) and "not divisible" in report.skipped["1"]  # 300 % 32 != 0
    report = quantize_linears(model, group_size=16)
    assert report.replaced == ["0"] and isinstance(model[0], Int8Linear)  # 300 % 16 != 0 -> second kept


def test_group_wise_model_is_closer_than_per_channel():
    ids = torch.randint(0, 97, (1, 40))
    errors = {}
    for group_size in (None, 16):
        model = make_model()
        with torch.no_grad():
            reference = model(ids)
            quantize_linears(model, include_lm_head=True, min_out_features=8, group_size=group_size)
            errors[group_size] = ((model(ids) - reference).norm() / reference.norm()).item()
    assert errors[16] < errors[None] < 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton GEMV needs CUDA")
@pytest.mark.parametrize("out_f,in_f,group_size", [(96, 256, 64), (300, 384, 128), (4096, 1024, 128)])
def test_cuda_group_gemv_matches_dequantized_matmul(out_f, in_f, group_size):
    torch.manual_seed(0)
    dtype = torch.float16  # bf16 Triton kernels need sm80; the local GPU may be older
    linear = nn.Linear(in_f, out_f, bias=True, device="cuda", dtype=dtype)
    q = Int8Linear.from_linear(linear, group_size=group_size)
    fn = torch.compile(int8_linear, fullgraph=True)
    for shape in ((1, 1, in_f), (in_f,)):
        x = torch.randn(shape, device="cuda", dtype=dtype)
        expected = torch.nn.functional.linear(x.float(), q.dequantized_weight().float(), linear.bias.float())
        for got in (q(x), fn(x, q.weight, q.scale, q.bias)):
            assert got.shape == expected.shape and got.dtype == dtype
            assert torch.allclose(got.float(), expected, atol=2e-2, rtol=2e-3)

