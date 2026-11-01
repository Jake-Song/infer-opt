"""Cold-cache tuning of selected-expert kernels; not an end-to-end benchmark."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from infer_opt import moe_kernels as kernels

@torch.inference_mode()
def main():
    torch.manual_seed(9)
    e,h,m,g,top=256,2048,512,128,8
    gu=torch.randint(-127,128,(e,2*m,h),device='cuda',dtype=torch.int8)
    down=torch.randint(-127,128,(e,h,m),device='cuda',dtype=torch.int8)
    gs=torch.full((e,2*m,h//g),.0001,device='cuda',dtype=torch.bfloat16)
    ds=torch.full((e,h,m//g),.0001,device='cuda',dtype=torch.bfloat16)
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.float32)
    rows=[];best={}
    for t in (1,2,3,4,5):
        x=torch.randn(t,h,device='cuda',dtype=torch.bfloat16)
        ids=torch.arange(t*top,device='cuda').reshape(t,top)
        w=torch.softmax(torch.randn(t,top,device='cuda'),-1).bfloat16()
        def run(): return kernels.selected_int8_experts(x,ids,w,gu,gs,down,ds)
        kernels.LAUNCH_CONFIGS[t]=(256,4)
        want=run()
        candidates=[]
        for block in (128,256,512,1024):
            for warps in (4,8):
                kernels.LAUNCH_CONFIGS[t]=(block,warps)
                got=run();torch.testing.assert_close(got,want,rtol=.03,atol=2e-4)
                for _ in range(3): run()
                torch.cuda.synchronize()
                samples=[]
                for _ in range(20):
                    flush.fill_(0)
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();run();b.record();b.synchronize();samples.append(a.elapsed_time(b))
                median=sorted(samples)[len(samples)//2]
                row={'tokens':t,'block_k':block,'warps':warps,'median_ms':median,'samples_ms':samples}
                rows.append(row);candidates.append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
        winner=min(candidates,key=lambda r:r['median_ms'])
        best[t]=[winner['block_k'],winner['warps']]
    Path('moe-expert-tuning.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'best':best,'rows':rows,'cache_flush_bytes':flush.numel()*4},indent=2))
if __name__=='__main__':main()
