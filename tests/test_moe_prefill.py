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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('tokens', [6, 17, 65])
@pytest.mark.parametrize('routing', ['shared', 'spread', 'skewed'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_grouped_prefill(tokens, routing, dtype):
    from infer_opt.moe_prefill import grouped_int8_experts
    torch.manual_seed(93)
    e, h, m, group, top = 13, 128, 64, 32, 3
    gu = torch.randint(-127, 128, (e, 2*m, h), device='cuda', dtype=torch.int8)
    down = torch.randint(-127, 128, (e, h, m), device='cuda', dtype=torch.int8)
    gs = (torch.rand(e, 2*m, h//group, device='cuda') * .0003 + .0001).to(dtype)
    ds = (torch.rand(e, h, m//group, device='cuda') * .0003 + .0001).to(dtype)
    x = torch.randn(tokens, h, device='cuda', dtype=dtype)
    weights = torch.softmax(torch.randn(tokens, top, device='cuda'), -1).to(dtype)
    ids = torch.rand(tokens, e, device='cuda').topk(top, -1).indices
    if routing == 'shared':
        ids[:] = torch.tensor([12, 0, 7], device='cuda')
    elif routing == 'skewed':
        ids[:tokens//2] = torch.tensor([12, 0, 7], device='cuda')
    run = lambda: grouped_int8_experts(x, ids, weights, gu, gs, down, ds)
    want = reference(x, ids, weights, gu, gs, down, ds)
    torch.testing.assert_close(run(), want, atol=2e-4, rtol=.03)
    # Changing routes exercises empty experts and the device tile schedule again.
    ids.copy_((ids + 5) % e)
    torch.testing.assert_close(run(), reference(x, ids, weights, gu, gs, down, ds), atol=2e-4, rtol=.03)


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
