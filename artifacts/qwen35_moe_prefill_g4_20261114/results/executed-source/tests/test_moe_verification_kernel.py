"""Verification GEMV changes must preserve the original reduction and graph inputs."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('tokens', [2, 3, 4, 5])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('warps', [1, 2])
@pytest.mark.parametrize('h,mid,top', [(512, 256, 3), (2048, 512, 8)])
def test_verification_projection_exact_and_dynamic(tokens, dtype, warps, h, mid, top, monkeypatch):
    from infer_opt import moe_kernels as kernels
    torch.manual_seed(191)
    e, group = 32, 128
    gu = torch.randint(-127, 128, (e, 2*mid, h), dtype=torch.int8, device='cuda')
    down = torch.randint(-127, 128, (e, h, mid), dtype=torch.int8, device='cuda')
    gs = (torch.rand(e, 2*mid, h//group, device='cuda') * .0003 + .0001).to(dtype)
    ds = (torch.rand(e, h, mid//group, device='cuda') * .0003 + .0001).to(dtype)
    x = torch.randn(tokens, h, device='cuda', dtype=dtype)
    weights = torch.softmax(torch.randn(tokens, top, device='cuda'), -1).to(dtype)
    ids = torch.arange(tokens*top, device='cuda').reshape(tokens, top) % (2*top)
    run = lambda: kernels.selected_int8_experts(x, ids, weights, gu, gs, down, ds)
    registry = kernels.PACKED_PAIRS_CONFIGS
    config = (256, warps)
    monkeypatch.setitem(registry, tokens, None)
    expected = run().clone()
    monkeypatch.setitem(registry, tokens, config)
    torch.testing.assert_close(run(), expected, atol=0, rtol=0)
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): captured = run()
    ids.copy_((torch.arange(tokens*top, device='cuda').reshape(tokens, top) + 15) % e)
    monkeypatch.setitem(registry, tokens, None)
    expected = run().clone()
    monkeypatch.setitem(registry, tokens, config)
    graph.replay(); torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)
