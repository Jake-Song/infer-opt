"""Cold-cache BF16 GEMV comparison on retained router, gate, and lm_head weights."""
import argparse,itertools,json,statistics,sys
from pathlib import Path
from collections import defaultdict
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM
from infer_opt.bf16_gemv import bf16_gemv

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    model=Qwen3_5ForCausalLM.from_quantized_dir(a.weights,device='cuda')
    model.fuse_projections();groups=defaultdict(list)
    for name,module in model.named_modules():
        if type(module) is torch.nn.Linear and not name.startswith('mtp.'):
            groups[(module.out_features,module.in_features)].append(module)
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.float32)
    result={'gpu':torch.cuda.get_device_name(),'shapes':{},'best':[],'cache_flush_bytes':flush.numel()*4}
    torch.manual_seed(19)
    for (n,k),modules in groups.items():
        x=torch.randn(1,1,k,device='cuda',dtype=torch.bfloat16)
        ref=modules[0](x).float();rows=[]
        for config in [None,*itertools.product((1,2,4,8),(128,256,512,1024),(2,4,8))]:
            if config and (config[1]>k or config[0]*config[1]<32*config[2]): continue
            def run(module):
                return module(x) if config is None else bf16_gemv(x,module.weight,*config)
            try:
                got=run(modules[0]).float()
                error=float((got-ref).norm()/ref.norm().clamp_min(1e-12))
                if error>.01: continue
                for module in modules:run(module)
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for module in modules:run(module)
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                samples=[]
                for _ in range(10):
                    flush.fill_(0);start.record();graph.replay();end.record();end.synchronize()
                    samples.append(start.elapsed_time(end)/len(modules))
                rows.append({'config':config,'median_ms':statistics.median(samples),'samples_ms':samples,'relative_l2':error})
                del graph
            except Exception as exc:
                print('candidate failed',n,k,config,type(exc).__name__,flush=True)
        baseline=rows[0];best=min(rows,key=lambda r:r['median_ms'])
        result['shapes'][str((n,k))]={'count':len(modules),'default':baseline,'best':best,'candidates':rows}
        if best['config'] is not None:result['best'].append([n,k,*best['config']])
        print(json.dumps({'shape':[n,k],'count':len(modules),'default_ms':baseline['median_ms'],'best_ms':best['median_ms'],'best':best['config']}),flush=True)
        Path(a.out).write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
