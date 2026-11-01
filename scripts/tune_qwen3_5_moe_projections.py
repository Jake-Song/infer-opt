"""Tune non-expert INT8 projections on real weights with a cold-cache timing boundary."""
import argparse,itertools,json,statistics,sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM
from infer_opt.quant import Int8Linear
from infer_opt import quant_kernels

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--out',required=True)
    p.add_argument('--repeats',type=int,default=10);a=p.parse_args()
    model=Qwen3_5ForCausalLM.from_quantized_dir(a.weights,device='cuda',kernels='fla-delta')
    model.fuse_projections()
    groups=defaultdict(list)
    for module in model.model.modules():
        if isinstance(module,Int8Linear):
            groups[(module.out_features,module.in_features,module.group_size)].append(module)
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.float32)
    result={'gpu':torch.cuda.get_device_name(),'cache_flush_bytes':flush.numel()*4,'shapes':{},'best':[]}
    torch.manual_seed(9)
    for key,modules in groups.items():
        n,k,g=key;x=torch.randn(1,1,k,device='cuda',dtype=torch.bfloat16)
        default=quant_kernels.launch_config(n,k,g)
        ref=modules[0](x).float()
        rows=[];errors=[]
        for cfg in dict.fromkeys([default,*itertools.product((1,2,4,8,16),(128,256,512,1024),(2,4,8))]):
            bn,bk,warps=cfg
            if k%bk or bn*bk<256*warps: continue
            quant_kernels.LAUNCH_CONFIGS[key]=cfg
            try:
                got=modules[0](x).float()
                error=float((got-ref).norm()/ref.norm().clamp_min(1e-12))
                if error>.01: raise ValueError(f'relative L2 {error} exceeds .01')
                for module in modules: module(x)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for module in modules: module(x)
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                samples=[]
                for _ in range(a.repeats):
                    flush.fill_(0);start.record();graph.replay();end.record();end.synchronize()
                    samples.append(start.elapsed_time(end)/len(modules))
                rows.append({'config':list(cfg),'median_ms':statistics.median(samples),'samples_ms':samples,'relative_l2':error})
                del graph
            except Exception as exc:
                errors.append({'config':list(cfg),'error':str(exc)[:200]})
        quant_kernels.LAUNCH_CONFIGS.pop(key,None)
        best=min(rows,key=lambda r:r['median_ms'])
        baseline=next(r for r in rows if r['config']==list(default))
        result['shapes'][str(key)]={'count':len(modules),'default':baseline,'best':best,'candidates':rows,'errors':errors}
        result['best'].append([*key,*best['config']])
        print(json.dumps({'shape':key,'count':len(modules),'default_ms':baseline['median_ms'],'best_ms':best['median_ms'],'best':best['config']}),flush=True)
        Path(a.out).write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
