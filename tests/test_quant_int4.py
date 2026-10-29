import pytest
import torch
from torch import nn

from infer_opt.quant import (Int4Linear, Int8Linear, dequantize_int4, int4_linear, pack_int4, quantize_int4,
                             quantize_linears, save_quantized, unpack_int4)
from tests.test_quant import write_hf_config
from tests.test_qwen3_5 import make_model


def test_pack_pairs_column_i_with_column_i_plus_16():
    q = torch.randint(0, 16, (3, 64), dtype=torch.uint8)
    packed = pack_int4(q)
    assert packed.shape == (3, 32) and packed.dtype == torch.uint8
    assert torch.equal(packed[:, 5], q[:, 5] | (q[:, 21] << 4))
    assert torch.equal(packed[:, 16 + 2], q[:, 32 + 2] | (q[:, 48 + 2] << 4))
    assert torch.equal(unpack_int4(packed), q)
    with pytest.raises(ValueError):
        pack_int4(torch.zeros(2, 48, dtype=torch.uint8))


def test_quantize_int4_error_is_at_most_half_a_step_and_uses_the_range():
    torch.manual_seed(0)
    w = torch.randn(40, 256) * torch.rand(40, 1) * 3 + 0.5  # rows with different ranges, not centred on zero
    packed, scale, offset = quantize_int4(w, group_size=64)
    assert packed.shape == (40, 128) and scale.shape == offset.shape == (40, 4)
    q = unpack_int4(packed).view(40, 4, 64)
    assert torch.all(q.amin(-1) == 0) and torch.all(q.amax(-1) == 15)  # min and max land on the end codes
    err = (dequantize_int4(packed, scale, offset, torch.float32) - w).view(40, 4, 64).abs()
    assert torch.all(err <= scale.unsqueeze(-1) / 2 + 1e-5)


def test_clip_search_never_increases_a_groups_error():
    torch.manual_seed(0)
    w = torch.randn(64, 512)
    w[torch.rand_like(w) < 0.01] *= 8  # outliers, which clipping trades against a finer step
    errors = {}
    for clip in (False, True):
        packed, scale, offset = quantize_int4(w, 128, clip_search=clip)
        errors[clip] = ((dequantize_int4(packed, scale, offset, torch.float32) - w) ** 2).view(64, 4, 128).sum(-1)
    assert torch.all(errors[True] <= errors[False] + 1e-6)
    assert errors[True].sum() < 0.95 * errors[False].sum()


def test_quantize_int4_chunking_matches_unchunked(monkeypatch):
    torch.manual_seed(0)
    w = torch.randn(50, 64)
    expected = quantize_int4(w, 32, clip_search=True)
    monkeypatch.setattr("infer_opt.quant._INT4_CHUNK_ELEMENTS", 7 * 64)
    got = quantize_int4(w, 32, clip_search=True)
    assert all(torch.equal(a, b) for a, b in zip(expected, got))


def test_constant_group_does_not_produce_nan():
    w = torch.randn(4, 64)
    w[2] = 0
    w[1, :32] = 0.25
    packed, scale, offset = quantize_int4(w, 32)
    deq = dequantize_int4(packed, scale, offset, torch.float32)
    assert torch.isfinite(deq).all() and torch.equal(deq[2], w[2]) and torch.equal(deq[1, :32], w[1, :32])


def test_group_size_must_be_a_multiple_of_32_dividing_in_features():
    with pytest.raises(ValueError):
        quantize_int4(torch.randn(8, 96), group_size=48)
    with pytest.raises(ValueError):
        quantize_int4(torch.randn(8, 96), group_size=64)
    with pytest.raises(ValueError):
        quantize_linears(nn.Sequential(nn.Linear(64, 300)), bits=4)  # no group size


def test_int4_linear_matches_dequantized_matmul_and_bias():
    torch.manual_seed(0)
    linear = nn.Linear(128, 48, bias=True)
    q = Int4Linear.from_linear(linear, group_size=32)
    assert q.in_features == 128 and q.group_size == 32 and "int4" in repr(q)
    x = torch.randn(3, 5, 128)
    expected = torch.nn.functional.linear(x, q.dequantized_weight(), linear.bias)
    assert torch.allclose(q(x), expected, atol=1e-5)
    assert torch.allclose(q(x), linear(x), atol=0.15)  # close to the original, not equal


def test_int4_model_stays_close_and_fuses_to_the_same_logits():
    ids = torch.randint(0, 97, (1, 50))
    errors = {}
    for bits in (8, 4):
        model = make_model()
        with torch.no_grad():
            reference = model(ids)
            report = quantize_linears(model, min_out_features=8, group_size=32, bits=bits)
            errors[bits] = ((model(ids) - reference).norm() / reference.norm()).item()
    assert "Int4Linear" in report.summary() and report.bytes_after < report.bytes_before / 6  # fp64 -> 4 bit (+ scales)
    assert isinstance(model.model.language_model.layers[0].mlp.down_proj, Int4Linear)
    assert errors[8] < errors[4] < 0.5  # 16 levels per group on a tiny random model
    with torch.no_grad():
        want = model(ids)
        assert model.fuse_projections() == 8
        fused = model.model.language_model.layers[0].mlp.gate_up_proj
        assert isinstance(fused, Int4Linear) and fused.group_size == 32
        assert torch.allclose(model(ids), want, atol=1e-10)


def test_fusion_skips_mixed_int4_and_int8_parts():
    model = make_model()
    quantize_linears(model, min_out_features=8, group_size=32, bits=4, exclude=("model.language_model.layers.0.mlp.up",))
    quantize_linears(model, min_out_features=8, group_size=32, exclude=("model.language_model.layers.1",))
    assert isinstance(model.model.language_model.layers[0].mlp.up_proj, Int8Linear)
    assert model.model.language_model.layers[0].mlp.fuse_projections() is False


def test_save_and_load_int4_round_trip(tmp_path):
    from infer_opt.qwen3_5 import Qwen3_5ForCausalLM

    model = make_model().float()
    quantize_linears(model, min_out_features=8, group_size=32, bits=4, clip_search=True)
    files = save_quantized(model, tmp_path, shard_bytes=20_000)
    assert len(files) > 1 and all(f.name.startswith("model-int4-") for f in files)
    write_hf_config(tmp_path, model.config)
    loaded = Qwen3_5ForCausalLM.from_quantized_dir(tmp_path)
    gate = loaded.model.language_model.layers[0].mlp.gate_proj
    assert isinstance(gate, Int4Linear) and gate.group_size == 32 and gate.weight.dtype == torch.uint8
    assert loaded.lm_head.weight is loaded.model.language_model.embed_tokens.weight
    ids = torch.randint(0, 97, (1, 30))
    with torch.no_grad():
        assert torch.equal(loaded(ids), model(ids))
    for (name, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert a.dtype == b.dtype, name
    loaded = Qwen3_5ForCausalLM.from_quantized_dir(tmp_path, dtype=torch.float64)
    gate = loaded.model.language_model.layers[0].mlp.gate_proj
    assert gate.weight.dtype == torch.uint8 and gate.offset.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton GEMV needs CUDA")
@pytest.mark.parametrize("out_f,in_f,group_size", [(96, 256, 64), (300, 384, 128), (4096, 1024, 128),
                                                  (640, 17408, 128),  # few rows, long K: the 27B down_proj
                                                  (200, 5120, 128), (64, 96, 32)])  # BLOCK_K 1024 / 32
def test_cuda_int4_gemv_matches_dequantized_matmul(out_f, in_f, group_size):
    torch.manual_seed(0)
    dtype = torch.float16  # bf16 Triton kernels need sm80; the local GPU may be older
    linear = nn.Linear(in_f, out_f, bias=True, device="cuda", dtype=dtype)
    q = Int4Linear.from_linear(linear, group_size=group_size)
    torch._dynamo.reset()
    fn = torch.compile(int4_linear, fullgraph=True)
    for shape in ((1, 1, in_f), (in_f,)):
        x = torch.randn(shape, device="cuda", dtype=dtype)
        expected = torch.nn.functional.linear(x.float(), q.dequantized_weight().float(), linear.bias.float())
        for got in (q(x), fn(x, q.weight, q.scale, q.offset, q.bias)):
            assert got.shape == expected.shape and got.dtype == dtype
            assert torch.allclose(got.float(), expected, atol=2e-2, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton GEMV needs CUDA")
@pytest.mark.parametrize("config", [(1, 1024, 1), (4, 512, 2), (8, 256, 4), (1, 64, 1)])
def test_cuda_int4_gemv_launch_configs_agree(config):
    from infer_opt import quant_kernels

    torch.manual_seed(0)
    q = Int4Linear.from_linear(nn.Linear(2048, 301, bias=False, device="cuda", dtype=torch.float16), group_size=128)
    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    expected = torch.nn.functional.linear(x.float(), q.dequantized_weight().float())
    quant_kernels.INT4_LAUNCH_CONFIGS[(301, 2048, 128)] = config
    try:
        got = quant_kernels.w4a16_group_gemv(x, q.weight, q.scale, q.offset)
    finally:
        quant_kernels.INT4_LAUNCH_CONFIGS.clear()
    assert torch.allclose(got.float(), expected, atol=2e-2, rtol=2e-3)
