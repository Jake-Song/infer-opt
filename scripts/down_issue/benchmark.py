"""Model-free down projection timing and deterministic profiling entry point."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/infer-opt-models/down-issue/triton')
import torch
import triton

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('down_issue_kernels', ROOT/'infer_opt/moe_kernels.py')
kernels = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kernels
spec.loader.exec_module(kernels)


def routes(tokens, routing):
    rows = []
    for token in range(tokens):
        row = []
        for slot0 in range(8):
            slot = (slot0+token) % 8
            expert = slot if routing == 'shared' or routing == 'partial' and slot < 4 else 4*token+slot
            if routing == 'disjoint':
                expert = 8*token+slot
            row.append(expert)
        assert len(set(row)) == 8
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int32, device='cuda')


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--warm-profile', action='store_true')
    p.add_argument('--tokens', type=int, default=4)
    p.add_argument('--routing', default='partial')
    p.add_argument('--variant', default='packed1', choices=['scalar1','scalar4','packed1','packed4'])
    p.add_argument('--cache', default='hot', choices=['hot','cold'])
    p.add_argument('--batches', type=int, default=3)
    p.add_argument('--repeats', type=int, default=100)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('Expected A100')
    torch.manual_seed(901)
    q = torch.randint(-127,128,(256,2048,512),device='cuda',dtype=torch.int8)
    s = (torch.rand((256,2048,4),device='cuda')*.0002+.0001).bfloat16()
    xs = {t:torch.randn((t*8,512),device='cuda',dtype=torch.bfloat16) for t in [1,2,4]}
    flush = torch.empty(128*2**20,device='cuda',dtype=torch.uint8)
    cases = {}
    variants = ['scalar1','scalar4','packed1','packed4']
    def prepare(t,routing,variant):
        ids = routes(t,routing)
        x = xs[t]
        y = torch.empty((t*8,2048),device='cuda',dtype=torch.bfloat16)
        def run():
            if variant.startswith('packed'):
                return kernels._project_packed[(t*8,2048)](x,q,s,ids,y,2048,512,8,128,False,num_warps=int(variant[-1]),enable_fp_fusion=False)
            return kernels._project[(t*8,2048)](x,q,s,ids,y,2048,512,8,128,False,256,num_warps=int(variant[-1]),enable_fp_fusion=False)
        compiled = run()
        # Always compare against the original four-warp arithmetic.
        reference = torch.empty_like(y)
        kernels._project[(t*8,2048)](x,q,s,ids,reference,2048,512,8,128,False,256,num_warps=4,enable_fp_fusion=False)
        torch.cuda.synchronize()
        exact = torch.equal(y,reference)
        delta = (y.float()-reference.float()).abs()
        # Independent reference for all eight selected experts of token zero.
        rows = []
        for slot, expert in enumerate(ids[0].tolist()):
            w = q[expert].float()*s[expert].float().repeat_interleave(128,dim=1)
            rows.append((w*x[slot].float()).sum(dim=1).bfloat16())
        numerical = torch.stack(rows)
        independent_delta = (y[:8].float()-numerical.float()).abs()
        validation = {'reference_exact':exact,'max_abs_diff':delta.max().item(),
                      'independent_max_abs_diff':independent_delta.max().item(),
                      'finite':bool(torch.isfinite(y).all()),'ids':ids.cpu().tolist(),
                      'output_sha256':hashlib.sha256(y.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}
        if not validation['finite']:
            raise RuntimeError(f'Invalid variant: {t}/{routing}/{variant}: {validation}')
        torch.testing.assert_close(y.float(),reference.float(),atol=0.002,rtol=0.02)
        # Independent check tolerates BF16 output rounding and reduction-order differences.
        torch.testing.assert_close(y[:8].float(),numerical.float(),atol=0.002,rtol=0.02)
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(10): run()
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph): run()
        artifact = out/f't{t}-{routing}-{variant}'
        for asm_name in ['ptx','sass']:
            try:
                code=compiled.asm[asm_name]
                if isinstance(code,str): artifact.with_suffix('.'+asm_name).write_text(code)
            except Exception as error:
                artifact.with_suffix('.'+asm_name+'.error').write_text(repr(error))
        try:
            metadata=compiled.metadata._asdict()
        except Exception:
            metadata=str(compiled.metadata)
        validation['compiled_metadata']=metadata
        return run, graph, y, y.clone(), validation

    if args.profile:
        run,graph,y,reference,validation=prepare(args.tokens,args.routing,args.variant)
        if args.warm_profile:
            for _ in range(1000): graph.replay()
            torch.cuda.synchronize()
        for _ in range(10): run()
        if args.cache=='cold': flush.zero_()
        torch.cuda._sleep(300000)
        torch.cuda.synchronize()
        validation['pre_profile_telemetry']=subprocess.run(['nvidia-smi','--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
        # cudaProfilerStart gates NCU so compilation and validation kernels are excluded.
        torch.cuda.cudart().cudaProfilerStart()
        run()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        (out/'profile-validation.json').write_text(json.dumps(validation,indent=2,default=str))
        print('PROFILE_VALIDATION '+json.dumps(validation,default=str),flush=True)
        return

    for t in [1,2,4]:
        for routing in (['partial'] if t != 4 else ['shared','partial','disjoint']):
            for variant in variants:
                cases[(t,routing,variant)]=prepare(t,routing,variant)
    metadata={'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'triton':triton.__version__,
              'shape':{'experts':256,'hidden':2048,'intermediate':512,'top_k':8,'group':128},
              'seed':901,'flush_bytes':flush.numel(),'prelude_cycles':300000,'batches':args.batches,
              'repeats':args.repeats,'kernel_sha256':hashlib.sha256((ROOT/'infer_opt/moe_kernels.py').read_bytes()).hexdigest(),
              'clock_locked':False,'rows':[]}
    rng=random.Random(20261111)
    order=[]
    for batch in range(args.batches):
        jobs=[(key,cache) for key in cases for cache in ['hot','cold']]
        rng.shuffle(jobs)
        for key,cache in jobs:
            run,graph,y,reference,validation=cases[key]
            # Re-prime this condition immediately before hot samples.
            for _ in range(10): graph.replay()
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            samples=[]
            for _ in range(args.repeats):
                if cache=='cold': flush.zero_()
                torch.cuda._sleep(300000)
                a.record(); graph.replay(); b.record(); b.synchronize()
                samples.append(a.elapsed_time(b)*1000)
            assert torch.equal(y,reference)
            clock=subprocess.run(['nvidia-smi','--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
            row={'batch':batch,'tokens':key[0],'routing':key[1],'variant':key[2],'cache':cache,
                 'median_us':statistics.median(samples),'samples_us':samples,'telemetry':clock,'validation':validation}
            metadata['rows'].append(row)
            (out/'timings.json').write_text(json.dumps(metadata,indent=2,default=str))
            print(json.dumps({k:v for k,v in row.items() if k not in ['samples_us','validation']}),flush=True)
    (out/'complete.json').write_text(json.dumps({'complete':True,'conditions':len(cases)*2,'batches':args.batches,'samples':len(metadata['rows'])*args.repeats}))


if __name__ == '__main__':
    main()
