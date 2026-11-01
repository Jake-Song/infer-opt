"""Validate real Qwen3.5-35B expert dimensions without downloading weights."""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer_opt.moe_kernels import selected_int8_experts
from infer_opt.quant import int8_linear

@torch.inference_mode()
def main():
    torch.manual_seed(9)
    e,h,m,g,k=256,2048,512,128,8
    gu=torch.randint(-127,128,(e,2*m,h),device='cuda',dtype=torch.int8)
    down=torch.randint(-127,128,(e,h,m),device='cuda',dtype=torch.int8)
    gs=torch.full((e,2*m,h//g),.0001,device='cuda',dtype=torch.bfloat16)
    ds=torch.full((e,h,m//g),.0001,device='cuda',dtype=torch.bfloat16)
    rows=[]
    for t in (1,2,3,4,5):
        for overlap in (True,False):
            ids=torch.arange(t*k,device='cuda').reshape(t,k)
            if overlap: ids=ids%k
            x=torch.randn(t,h,device='cuda',dtype=torch.bfloat16)
            weights=torch.softmax(torch.randn(t,k,device='cuda'),-1).bfloat16()
            def run(): return selected_int8_experts(x,ids,weights,gu,gs,down,ds)
            want=torch.zeros_like(x)
            for token in range(t):
                for slot in range(k):
                    expert=int(ids[token,slot])
                    gate,up=int8_linear(x[token:token+1],gu[expert],gs[expert]).chunk(2,-1)
                    y=int8_linear(torch.nn.functional.silu(gate)*up,down[expert],ds[expert])
                    want[token]+= (y[0]*weights[token,slot]).bfloat16()
            got=run()
            torch.testing.assert_close(got,want,atol=2e-4,rtol=.03)
            stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): run()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): captured=run()
            graph.replay(); torch.cuda.synchronize()
            torch.testing.assert_close(captured,got,atol=0,rtol=0)
            compiled=torch.compile(run,fullgraph=True)
            torch.testing.assert_close(compiled(),got,atol=0,rtol=0)
            ids.copy_((ids + k) % e)
            changed=run()
            graph.replay(); torch.cuda.synchronize()
            torch.testing.assert_close(captured,changed,atol=0,rtol=0)
            torch.testing.assert_close(compiled(),changed,atol=0,rtol=0)
            rows.append({'dynamic_routes_equal':True,'tokens':t,'overlap':overlap,'max_abs_error':float((got-want).abs().max()),'graph_equal':True,'compiled_equal':True})
            print(json.dumps(rows[-1]),flush=True)
    result={'gpu':torch.cuda.get_device_name(),'memory_gib':torch.cuda.get_device_properties(0).total_memory/2**30,'torch':str(torch.__version__),'rows':rows}
    Path('moe-kernel-validation.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__': main()
