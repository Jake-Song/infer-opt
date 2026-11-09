"""Shared-route projection arithmetic, padding, and dynamic graph inputs."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('tokens,top', [(2, 2), (3, 3), (4, 2), (5, 3)])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_shared_routes_match_original_with_dynamic_ids(tokens, top, dtype, monkeypatch):
    from infer_opt import moe_kernels as kernels
    torch.manual_seed(91)
    e, h, mid, group = 16, 64, 32, 16
    gu = torch.randint(-127, 128, (e, 2*mid, h), dtype=torch.int8, device='cuda')
    down = torch.randint(-127, 128, (e, h, mid), dtype=torch.int8, device='cuda')
    gs = (torch.rand(e, 2*mid, h//group, device='cuda') * .0003 + .0001).to(dtype)
    ds = (torch.rand(e, h, mid//group, device='cuda') * .0003 + .0001).to(dtype)
    x = torch.randn(tokens, h, device='cuda', dtype=dtype)
    weights = torch.softmax(torch.randn(tokens, top, device='cuda'), -1).to(dtype)
    ids = torch.arange(tokens*top, device='cuda').reshape(tokens, top) % (2*top)
    run = lambda: kernels.selected_int8_experts(x, ids, weights, gu, gs, down, ds)
    monkeypatch.delitem(kernels.SHARED_ROUTE_CONFIGS, tokens, raising=False)
    expected = run().clone()
    monkeypatch.setitem(kernels.SHARED_ROUTE_CONFIGS, tokens, (256, 2, 4))
    torch.testing.assert_close(run(), expected, atol=0, rtol=0)
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): captured = run()
    # Switch from overlapping routes to disjoint routes in the same graph.
    ids.copy_(torch.arange(tokens*top, device='cuda').reshape(tokens, top))
    monkeypatch.delitem(kernels.SHARED_ROUTE_CONFIGS, tokens)
    expected = run().clone()
    monkeypatch.setitem(kernels.SHARED_ROUTE_CONFIGS, tokens, (256, 2, 4))
    graph.replay(); torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)
