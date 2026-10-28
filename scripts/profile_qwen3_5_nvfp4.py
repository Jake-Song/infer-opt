"""Profile packed target linears in model order; separate from end-to-end timing."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from infer_opt.nvfp4 import Nvfp4Linear,require_sm120
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    def timeout(signum,frame): raise TimeoutError('linears profiler exceeded 180 seconds')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(180)
    require_sm120()
    model=Qwen3_5ForCausalLM.from_quantized_dir(a.checkpoint,device='cuda')
    linears=[m for m in model.modules() if isinstance(m,Nvfp4Linear)]
    results={'scope':'256 target linears only, model order, synthetic BF16 activations; excludes norms, recurrent state, attention and lm_head',
             'gpu':torch.cuda.get_device_name(),'rows':[]}
    for rows in [1,4]:
        inputs=[torch.randn(rows,m.in_features,device='cuda',dtype=torch.bfloat16) for m in linears]
        for m,x in zip(linears,inputs): m(x)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for m,x in zip(linears,inputs): m(x)
        for _ in range(3): graph.replay()
        torch.cuda.synchronize()
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20): graph.replay()
        end.record()
        end.synchronize()
        milliseconds=start.elapsed_time(end)/20
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            graph.replay()
            torch.cuda.synchronize()
        kernels=defaultdict(float)
        for event in prof.events():
            if event.device_type==torch.autograd.DeviceType.CUDA:
                kernels[event.name]+=event.device_time_total/1000
        results['rows'].append({'activation_rows':rows,'linear_count':len(linears),'aggregate_cuda_ms':milliseconds,
                                'kernel_ms':dict(sorted(kernels.items(),key=lambda item:-item[1]))})
    a.out.write_text(json.dumps(results,indent=2))
    signal.alarm(0)


if __name__=='__main__': main()
