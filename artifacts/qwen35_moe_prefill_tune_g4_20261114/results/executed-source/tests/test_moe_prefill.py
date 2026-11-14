"""Independent dequantized expert reference and adversarial route distributions."""
import pytest
import torch
from torch.nn import functional as F


def reference(x, ids, weights, gu, gs, down, ds):
    from infer_opt.quant import dequantize_int8
    result = torch.zeros_like(x)
    for expert in range(gu.shape[0]):
        tokens, slots = torch.where(ids == expert)
        if not tokens.numel():
            continue
        gate, up = F.linear(x[tokens], dequantize_int8(gu[expert], gs[expert], x.dtype)).chunk(2, -1)
        y = F.linear(F.silu(gate) * up, dequantize_int8(down[expert], ds[expert], x.dtype))
        result.index_add_(0, tokens, (y * weights[tokens, slots, None]).to(x.dtype))
    return result


def bank(tokens, dtype, e=13, h=128, m=64, group=32, top=3):
    gu = torch.randint(-127, 128, (e, 2*m, h), device='cuda', dtype=torch.int8)
    down = torch.randint(-127, 128, (e, h, m), device='cuda', dtype=torch.int8)
    gs = (torch.rand(e, 2*m, h//group, device='cuda') * .0003 + .0001).to(dtype)
    ds = (torch.rand(e, h, m//group, device='cuda') * .0003 + .0001).to(dtype)
    x = torch.randn(tokens, h, device='cuda', dtype=dtype)
    weights = torch.softmax(torch.randn(tokens, top, device='cuda'), -1).to(dtype)
    ids = torch.rand(tokens, e, device='cuda').topk(top, -1).indices
    return x, ids, weights, gu, gs, down, ds


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('tokens', [6, 17, 65])
@pytest.mark.parametrize('routing', ['shared', 'spread', 'skewed'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_grouped_prefill(tokens, routing, dtype):
    """Adoption criterion: no farther from the FP64 value than the adopted eager loop."""
    from infer_opt.moe import Int8Experts
    from infer_opt.moe_prefill import float64_gate, float64_int8_experts, grouped_int8_experts
    torch.manual_seed(93)
    x, ids, weights, gu, gs, down, ds = bank(tokens, dtype)
    if routing == 'shared':
        ids[:] = torch.tensor([12, 0, 7], device='cuda')
    elif routing == 'skewed':
        ids[:tokens//2] = torch.tensor([12, 0, 7], device='cuda')
    module = Int8Experts(gu, gs, down, ds, 32)
    def check():
        truth = float64_int8_experts(x, ids, weights, gu, gs, down, ds)
        existing = module.forward_reference(x, ids, weights)
        for block_m in (16, 32):
            got = grouped_int8_experts(x, ids, weights, gu, gs, down, ds, block_m=block_m)
            # A few hundred elements: single rounding flips move the RMS by more than the 1%
            # allowance used at real shapes (scripts/bench_moe_prefill.py), so allow 10% here.
            gate = float64_gate(got, existing, truth, slack=.1)
            assert gate['passes'], gate
            # A routing or tiling bug is a gross error, far above BF16/FP16 rounding.
            assert gate['candidate']['rms_relative'] < .02, gate
    check()
    # Changing routes exercises empty experts and the device tile schedule again.
    ids.copy_((ids + 5) % 13)
    check()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_float64_reference_and_gate():
    from infer_opt.moe_prefill import float64_gate, float64_int8_experts, grouped_int8_experts
    torch.manual_seed(5)
    x, ids, weights, gu, gs, down, ds = bank(33, torch.bfloat16)
    stages = {}
    truth = float64_int8_experts(x, ids, weights, gu, gs, down, ds, stages=stages)
    # Independent FP32 evaluation with exact q * scale weights agrees to FP32 accuracy.
    want = torch.zeros_like(truth, dtype=torch.float32)
    for expert in range(gu.shape[0]):
        tokens, slots = torch.where(ids == expert)
        w1 = gu[expert].float() * gs[expert].float().repeat_interleave(32, -1)
        w2 = down[expert].float() * ds[expert].float().repeat_interleave(32, -1)
        gate, up = F.linear(x[tokens].float(), w1).chunk(2, -1)
        want.index_add_(0, tokens, F.linear(F.silu(gate) * up, w2) * weights[tokens, slots, None].float())
    torch.testing.assert_close(truth.float(), want, rtol=1e-4, atol=1e-6)
    assert stages['routed'].shape == (33 * 3, 128) and stages['gate_up'].dtype == torch.float64
    got_stages = {}
    got = grouped_int8_experts(x, ids, weights, gu, gs, down, ds, stages=got_stages)
    # Scales act on FP32 partial sums: each projection is the rounded exact-weight value.
    delta = got_stages['gate_up'].double() - stages['gate_up']
    spacing = torch.exp2(torch.floor(torch.log2(stages['gate_up'].abs())) - 7)
    assert ((delta.abs() <= .5 * spacing).double().mean() > .99)
    assert not float64_gate(got + .01 * got.abs().max(), got, truth)['passes']
    with pytest.raises(ValueError, match='block_k'):
        grouped_int8_experts(x, ids, weights, gu, gs, down, ds, block_k=64)


def test_prefill_default_and_cpu_fallback():
    from infer_opt.moe import Int8Experts
    from infer_opt.quant import quantize_int8
    torch.manual_seed(17)
    def bank(e, n, k):
        pairs = [quantize_int8(torch.randn(n, k) * .02, 16) for _ in range(e)]
        return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])
    gu, gs = bank(4, 64, 64)
    down, ds = bank(4, 64, 32)
    module = Int8Experts(gu, gs, down, ds, 16)
    assert module.prefill_backend == 'reference'
    x = torch.randn(7, 64)
    ids = torch.arange(14).reshape(7, 2) % 4
    weights = torch.full((7, 2), .5)
    want = reference(x, ids, weights, gu, gs, down, ds)
    module.prefill_backend = 'grouped'
    torch.testing.assert_close(module(x, ids, weights), want, rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('tokens', [6, 17, 65])
def test_sorted_routes_preserve_expert_arithmetic(device, tokens):
    from infer_opt.moe import Int8Experts
    torch.manual_seed(901)
    e, h, mid, top, group = 13, 128, 64, 3, 32
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    gu = torch.randint(-127, 128, (e, 2*mid, h), device=device, dtype=torch.int8)
    down = torch.randint(-127, 128, (e, h, mid), device=device, dtype=torch.int8)
    gs = (torch.rand(e, 2*mid, h//group, device=device) * .0003 + .0001).to(dtype)
    ds = (torch.rand(e, h, mid//group, device=device) * .0003 + .0001).to(dtype)
    module = Int8Experts(gu, gs, down, ds, group)
    x = torch.randn(tokens, h, device=device, dtype=dtype)
    weights = torch.softmax(torch.randn(tokens, top, device=device), -1).to(dtype)
    for routing in ('spread', 'shared', 'skewed'):
        ids = torch.rand(tokens, e, device=device).topk(top, -1).indices
        if routing == 'shared':
            ids[:] = torch.tensor([12, 0, 7], device=device)
        if routing == 'skewed':
            ids[:tokens*3//4] = torch.tensor([12, 0, 7], device=device)
        expected = module.forward_reference(x, ids, weights)
        module.prefill_backend = 'sorted'
        torch.testing.assert_close(module(x, ids, weights), expected, atol=0, rtol=0)


def test_route_statistics():
    from infer_opt.prefill_profile import route_statistics
    stats = route_statistics([0, 1, 8, 9, 40, 0, 70])
    assert stats['rows'] == 128 and stats['used_experts'] == 5 and stats['experts_with_1_to_8_rows'] == 2
    assert (stats['rows_min'], stats['rows_median'], stats['rows_max']) == (1, 9.0, 70)
    assert stats['experts_by_rows'] == {'1-8': 2, '9-16': 1, '17-32': 0, '33-64': 1, '65-128': 1, '129+': 0}
    assert stats['tiles']['16'] == {'tiles': 1 + 1 + 1 + 3 + 5, 'fill': 128 / (11 * 16)}
    assert route_statistics([0, 0]) == {'experts': 2, 'rows': 0, 'used_experts': 0}


def test_prefill_instrumentation():
    from infer_opt.prefill_profile import RouteRecorder, section_times
    from test_moe import model as tiny_model
    m = tiny_model()
    lm = m.model.language_model
    ids = torch.arange(11).reshape(1, 11) % m.config.vocab_size
    def prefill():
        return m.lm_head(lm(ids)[:, -1:]).argmax(-1)
    want = prefill()
    with RouteRecorder(lm) as recorder:
        assert torch.equal(prefill(), want)
    assert len(recorder.counts) == m.config.num_hidden_layers
    assert all(sum(c) == 11 * m.config.num_experts_per_tok for c in recorder.counts.values())
    summary = recorder.summary()
    assert summary['used_experts_max'] <= m.config.num_experts and len(summary['layers']) == len(recorder.counts)
    result = section_times(m, prefill, repeats=1)  # one pass: the sections add up exactly
    times, host = result['total_ms'], result['host_ms']
    parts = [v for k, v in times.items() if k not in ('text_model', 'wall', 'lm_head')]
    assert all(0 <= host[k] <= times[k] for k in host) and 'routed_experts' in host
    assert all(v >= 0 for k, v in times.items() if k != 'glue')
    assert abs(sum(parts) - times['text_model']) < 1e-6 * max(1, times['text_model'])
    assert times['routed_experts'] > 0 and times['wall'] >= times['text_model'] + times['lm_head']
    kinds = set(m.config.layer_types)
    for name in ('gated_delta_conv', 'gated_delta_rule', 'gated_delta_norm', 'gated_delta_other'):
        assert (times[name] > 0) == ('linear_attention' in kinds), name
    assert (times['full_attention_core'] > 0) == ('full_attention' in kinds)
    import infer_opt.qwen3_5 as q  # the timing wrappers are gone
    assert q.F.conv1d is torch.nn.functional.conv1d and q._load_fla.__name__ == '_load_fla'
    assert q.Qwen3_5Cache.set_state.__name__ == 'set_state'
    assert not lm._forward_hooks and not lm.layers[0].mlp.experts._forward_pre_hooks
    assert torch.equal(prefill(), want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_prefill_backends_in_model():
    """The three backends through a whole quantized model: sorted is exact, grouped differs by rounding."""
    from infer_opt.moe import Int8Experts, quantize_moe_experts
    from infer_opt.prefill_profile import RouteRecorder, section_times
    from test_moe import model as tiny_model
    m = tiny_model()
    quantize_moe_experts(m, 16)
    m = m.to('cuda', torch.bfloat16)
    lm = m.model.language_model
    banks = [b for b in lm.modules() if isinstance(b, Int8Experts)]
    ids = (torch.arange(23, device='cuda').reshape(1, 23) * 7) % m.config.vocab_size
    logits = {}
    for backend in ('reference', 'sorted', 'grouped'):
        for b in banks:
            b.prefill_backend = backend
        logits[backend] = m.lm_head(lm(ids)[:, -1:]).float()
    assert torch.equal(logits['sorted'], logits['reference'])
    want, got = logits['reference'].log_softmax(-1), logits['grouped'].log_softmax(-1)
    assert (want.exp() * (want - got)).sum().item() < 1e-3
    with RouteRecorder(lm) as recorder:
        times = section_times(m, lambda: m.lm_head(lm(ids)[:, -1:]), repeats=1)['total_ms']
    assert len(recorder.counts) == len(banks) and times['routed_experts'] > 0
    # Two tile classes evaluate each expert once: same arithmetic as one class.
    for b in banks:
        b.grouped_options = {'large': (2, 32, 64, 16, 4)}
    assert torch.equal(m.lm_head(lm(ids)[:, -1:]).float(), logits['grouped'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('rows', [0, 3, 20, 1000])
def test_grouped_tile_classes(rows):
    """Experts above the row threshold use the second tile; the result does not depend on the split."""
    from infer_opt.moe_prefill import float64_gate, float64_int8_experts, grouped_int8_experts
    torch.manual_seed(11)
    x, ids, weights, gu, gs, down, ds = bank(65, torch.bfloat16)
    ids[:40] = torch.tensor([12, 0, 7], device='cuda')  # three experts with 40+ rows, the rest with a few
    one = grouped_int8_experts(x, ids, weights, gu, gs, down, ds)
    two = grouped_int8_experts(x, ids, weights, gu, gs, down, ds, large=(rows, 32, 32, 16, 8))
    truth = float64_int8_experts(x, ids, weights, gu, gs, down, ds)
    assert float64_gate(two, one, truth)['passes']
    torch.testing.assert_close(two, one, rtol=.03, atol=2e-3)  # gross errors only; rounding may differ
    with pytest.raises(ValueError, match='large'):
        grouped_int8_experts(x, ids, weights, gu, gs, down, ds, large=(4, 32, 32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('tokens', [1, 17, 70])
def test_w8a16_gemm(tokens):
    from infer_opt.moe_prefill import w8a16_gemm
    from infer_opt.quant import dequantize_int8
    torch.manual_seed(3)
    weight = torch.randint(-127, 128, (96, 128), device='cuda', dtype=torch.int8)
    scale = (torch.rand(96, 4, device='cuda') * .0003 + .0001).bfloat16()
    x = torch.randn(tokens, 128, device='cuda', dtype=torch.bfloat16)
    truth = x.double() @ (weight.double() * scale.double().repeat_interleave(32, -1)).T
    error = lambda got: (got.double() - truth).square().mean().sqrt() / truth.square().mean().sqrt()
    got = w8a16_gemm(x, weight, scale, block_m=16, block_n=32)
    assert got.shape == (tokens, 96) and error(got) < .004
    # no BF16 weight rounding: at least as close to FP64 as dequantize + matmul (10% for tiny shapes)
    assert error(got) <= 1.1 * error(F.linear(x, dequantize_int8(weight, scale, x.dtype)))
