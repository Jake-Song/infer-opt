import pytest
import torch

from infer_opt.gptq import gptq_quantize_int4, gptq_quantize_qwen3_5
from infer_opt.quant import Int4Linear, dequantize_int4, quantize_int4, quantize_linears
from tests.test_qwen3_5 import make_model


def correlated_inputs(n_tokens: int, in_f: int) -> torch.Tensor:
    """``[tokens, in]`` with correlated channels and a few large ones, like LLM activations."""
    torch.manual_seed(1)
    mix = torch.randn(in_f, in_f) / in_f**0.5 + torch.eye(in_f)
    x = torch.randn(n_tokens, in_f) @ mix
    x[:, :4] *= 10
    return x


def test_diagonal_hessian_gives_min_max_rtn():
    torch.manual_seed(0)
    w = torch.randn(48, 256)
    h = torch.diag(torch.rand(256) + 0.5)
    got = gptq_quantize_int4(w, h, group_size=64)
    want = quantize_int4(w, 64)
    assert all(torch.equal(a, b) for a, b in zip(got, want))


@pytest.mark.parametrize("block_size", [128, 32])  # one block per group, and several blocks per group
def test_gptq_output_error_is_below_rtn(block_size):
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    x = correlated_inputs(2048, 256)
    packed, scale, offset = gptq_quantize_int4(w, x.T @ x, group_size=128, block_size=block_size)
    gptq = dequantize_int4(packed, scale, offset, torch.float32)
    rtn = dequantize_int4(*quantize_int4(w, 128), torch.float32)
    err_gptq = ((x @ (gptq - w).T) ** 2).sum()
    err_rtn = ((x @ (rtn - w).T) ** 2).sum()
    assert err_gptq < 0.7 * err_rtn


def test_dead_input_columns_are_zero():
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    x = torch.randn(100, 64)
    x[:, 3] = 0
    packed, scale, offset = gptq_quantize_int4(w, x.T @ x, group_size=32)
    deq = dequantize_int4(packed, scale, offset, torch.float32)
    assert torch.isfinite(deq).all() and deq[:, 3].abs().max() <= scale[:, 0].abs().max() / 2 + 1e-6


def test_gptq_model_is_closer_than_rtn_on_held_out_tokens():
    torch.manual_seed(0)
    calibration = [torch.randint(0, 97, (1, 48)) for _ in range(8)]
    held_out = torch.randint(0, 97, (1, 64))
    errors = {}
    for method in ("rtn", "gptq"):
        model = make_model()
        with torch.no_grad():
            reference = model(held_out)
            if method == "gptq":
                report = gptq_quantize_qwen3_5(model, calibration, group_size=32, min_out_features=8, log=lambda _: None)
            else:
                quantize_linears(model, min_out_features=8, group_size=32, bits=4)
            errors[method] = ((model(held_out) - reference).norm() / reference.norm()).item()
    assert len(report.replaced) == 4 * 3 + 3 * 3 + 1 * 4  # MLPs, GatedDeltaNet qkv/z/out, attention q/k/v/o
    assert "lm_head" in report.skipped and any("in_proj_a" in n for n in report.skipped)
    assert isinstance(model.model.language_model.layers[3].self_attn.o_proj, Int4Linear)
    assert errors["gptq"] < errors["rtn"]
