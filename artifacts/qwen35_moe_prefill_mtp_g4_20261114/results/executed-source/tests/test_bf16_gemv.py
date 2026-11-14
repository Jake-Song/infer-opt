import pytest
import torch


def test_bf16_override_preserves_weights_and_cpu_fallback():
    pytest.importorskip('triton')
    from infer_opt.bf16_gemv import install_bf16_gemv,BF16GemvLinear
    m=torch.nn.Sequential(torch.nn.Linear(32,16,bias=True)).bfloat16().eval()
    x=torch.randn(2,32).bfloat16();want=m(x);weight=m[0].weight
    assert install_bf16_gemv(m,{(16,32):(1,32,2)})==['0']
    assert isinstance(m[0],BF16GemvLinear) and m[0].weight is weight and not m[0].training
    torch.testing.assert_close(m(x),want,atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('n',[1,32,256,4096])
def test_bf16_gemv_graph_and_compile(n):
    from infer_opt.bf16_gemv import bf16_gemv
    torch.manual_seed(3)
    x=torch.randn(1,1,2048,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(n,2048,device='cuda',dtype=torch.bfloat16)*.02
    def run(): return bf16_gemv(x,w,2,512,4)
    want=torch.nn.functional.linear(x,w)
    got=run()
    torch.testing.assert_close(got,want,atol=.02,rtol=.02)
    for _ in range(3):run()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=run()
    graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(captured,got,atol=0,rtol=0)
    compiled=torch.compile(run,fullgraph=True)
    torch.testing.assert_close(compiled(),got,atol=0,rtol=0)
    with pytest.raises(ValueError,match='one activation row'):
        bf16_gemv(x.expand(2,1,2048).contiguous(),w,2,512,4)

    with pytest.raises(ValueError,match='one activation row'):
        bf16_gemv(x.reshape(2,1024),w,2,512,4)
