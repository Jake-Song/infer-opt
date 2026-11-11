"""Matched gate/up, down and routed expert block diagnosis without models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

os.environ.setdefault('TRITON_CACHE_DIR','/tmp/infer-opt-models/expert-issue/triton')
import torch
import triton

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from down_issue.benchmark import kernels,routes


def telemetry():
    p=subprocess.run(['nvidia-smi','--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader'],capture_output=True,text=True)
    return p.stdout.strip()


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--kind',choices=['gate_up','down','block'],default='gate_up')
    parser.add_argument('--tokens',type=int,default=4)
    parser.add_argument('--routing',choices=['shared','partial','disjoint'],default='partial')
    parser.add_argument('--variant',choices=['scalar1','scalar4','packed1','packed4'],default='packed1')
    parser.add_argument('--cache',choices=['hot','cold'],default='hot')
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    assert 'A100' in torch.cuda.get_device_name()
    torch.manual_seed(901)
    # Keep standalone down fixtures identical to the preceding down-only run.
    dq=torch.randint(-127,128,(256,2048,512),device='cuda',dtype=torch.int8)
    ds=(torch.rand((256,2048,4),device='cuda')*.0002+.0001).bfloat16()
    dxs={t:torch.randn((t*8,512),device='cuda',dtype=torch.bfloat16) for t in [1,2,4]}
    gq=torch.randint(-127,128,(256,1024,2048),device='cuda',dtype=torch.int8)
    gs=(torch.rand((256,1024,16),device='cuda')*.0002+.0001).bfloat16()
    xs={t:torch.randn((t,2048),device='cuda',dtype=torch.bfloat16) for t in [1,2,4]}
    weights={t:torch.tensor([[(i+1)/36 for i in range(8)]]*t,device='cuda',dtype=torch.bfloat16) for t in [1,2,4]}
    flush=torch.empty(128*2**20,device='cuda',dtype=torch.uint8)

    def prepare(t,routing,variant,kind):
        ids=routes(t,routing)
        x=xs[t]
        intermediate=torch.empty((t*8,512),device='cuda',dtype=torch.bfloat16)
        down_out=torch.empty((t*8,2048),device='cuda',dtype=torch.bfloat16)
        block_out=torch.empty((t,2048),device='cuda',dtype=torch.bfloat16)
        handles={}
        def project(activation,q,s,y,gate,selected):
            n,k,width=(1024,2048,512) if gate else (2048,512,2048)
            if selected.startswith('packed'):
                handle=kernels._project_packed[(t*8,width)](activation,q,s,ids,y,n,k,8,128,gate,num_warps=int(selected[-1]),enable_fp_fusion=False)
            else:
                handle=kernels._project[(t*8,width)](activation,q,s,ids,y,n,k,8,128,gate,256,num_warps=int(selected[-1]),enable_fp_fusion=False)
            handles['gate_up' if gate else 'down']=handle
        def run(selected=variant):
            if kind in ['gate_up','block']:project(x,gq,gs,intermediate,True,selected)
            if kind in ['down','block']:project(intermediate if kind=='block' else dxs[t],dq,ds,down_out,False,selected)
            if kind=='block':
                handles['combine']=kernels._combine[(t,8)](down_out,weights[t],block_out,2048,8,256,enable_fp_fusion=False)
            return intermediate if kind=='gate_up' else down_out if kind=='down' else block_out
        got=run().clone()
        reference=run('scalar4').clone()
        torch.cuda.synchronize()
        delta=(got.float()-reference.float()).abs()
        validation={'reference_exact':torch.equal(got,reference),'max_abs_diff':delta.max().item(),
                    'different_elements':int((delta!=0).sum().item()),'finite':bool(torch.isfinite(got).all()),
                    'ids':ids.cpu().tolist(),'output_shape':list(got.shape),
                    'output_sha256':hashlib.sha256(got.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}
        torch.testing.assert_close(got.float(),reference.float(),atol=0.002,rtol=0.02)
        assert validation['finite']

        # Independent FP32 reductions, with explicit BF16 rounding at the same boundaries.
        def independent_gate(expert):
            w=gq[expert].float()*gs[expert].float().repeat_interleave(128,dim=1)
            v=(w*x[0].float()).sum(dim=1).bfloat16().float()
            return ((v[:512]*torch.sigmoid(v[:512])).bfloat16().float()*v[512:]).bfloat16()
        numerical=[]
        for slot,expert in enumerate(ids[0].tolist()):
            if kind=='gate_up':
                value=independent_gate(expert)
            else:
                activation=independent_gate(expert) if kind=='block' else dxs[t][slot]
                w=dq[expert].float()*ds[expert].float().repeat_interleave(128,dim=1)
                value=(w*activation.float()).sum(dim=1).bfloat16()
            numerical.append(value)
        numerical=torch.stack(numerical)
        if kind=='block':
            accum=torch.zeros(2048,device='cuda',dtype=torch.bfloat16)
            for slot in range(8):
                weighted=(numerical[slot].float()*weights[t][0,slot].float()).bfloat16()
                accum=(accum.float()+weighted.float()).bfloat16()
            numerical=accum.unsqueeze(0)
            checked=got[:1]
        else:checked=got[:8]
        validation['independent_max_abs_diff']=(checked.float()-numerical.float()).abs().max().item()
        torch.testing.assert_close(checked.float(),numerical.float(),atol=0.002,rtol=0.02)
        if kind=='block':
            # Check that the explicit three-kernel block matches the production dispatcher.
            kernels.LAUNCH_CONFIGS.clear();kernels.PACKED_PAIRS_CONFIGS.clear();kernels.USE_PACKED_VERIFY=False
            if variant.startswith('packed'):kernels.PACKED_PAIRS_CONFIGS[t]=(256,int(variant[-1]))
            else:kernels.LAUNCH_CONFIGS[t]=(256,int(variant[-1]))
            production=kernels.selected_int8_experts(x,ids,weights[t],gq,gs,dq,ds)
            validation['production_dispatch_exact']=torch.equal(got,production)
            assert validation['production_dispatch_exact']
        result=run()
        want=result.clone()
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(10):run()
        torch.cuda.current_stream().wait_stream(side)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run()
        for name,handle in handles.items():
            label=out/f't{t}-{routing}-{kind}-{variant}-{name}'
            try:
                validation.setdefault('compiled',{})[name]={'metadata':handle.metadata._asdict(),'registers':handle.n_regs,'spills':handle.n_spills}
            except Exception:validation.setdefault('compiled',{})[name]={'metadata':str(handle.metadata)}
            for asm in ['ptx','sass']:
                try:
                    code=handle.asm[asm]
                    if isinstance(code,str):label.with_suffix('.'+asm).write_text(code)
                except Exception as error:label.with_suffix('.'+asm+'.error').write_text(str(error))
        return run,graph,result,want,validation

    if args.profile:
        run,graph,result,want,validation=prepare(args.tokens,args.routing,args.variant,args.kind)
        for _ in range(100):graph.replay()
        if args.cache=='cold':flush.zero_()
        torch.cuda._sleep(300000)
        torch.cuda.synchronize()
        assert int(torch.cuda.cudart().cudaProfilerStart())==0
        run();torch.cuda.synchronize()
        assert int(torch.cuda.cudart().cudaProfilerStop())==0
        assert torch.equal(result,want)
        validation['post_profile_telemetry']=telemetry()
        (out/'validation.json').write_text(json.dumps(validation,indent=2,default=str))
        print('PROFILE_VALIDATION '+json.dumps(validation,default=str),flush=True)
        return

    cases={}
    for t in [1,2,4]:
        for variant in ['scalar1','scalar4','packed1','packed4']:
            for kind in ['gate_up','down','block']:
                cases[(t,'partial',variant,kind)]=prepare(t,'partial',variant,kind)
    for routing in ['shared','disjoint']:
        for kind in ['gate_up','down','block']:
            cases[(4,routing,'packed1',kind)]=prepare(4,routing,'packed1',kind)
    preflight=json.loads(Path('/tmp/expert-issue-a100-20261111/preflight.json').read_text())
    data={'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'triton':triton.__version__,
          'seed':901,'clock_lock':preflight['clock_lock'],'shape':{'experts':256,'hidden':2048,'intermediate':512,'top_k':8,'group':128},
          'source_sha256':hashlib.sha256((ROOT/'infer_opt/moe_kernels.py').read_bytes()).hexdigest(),
          'flush_bytes':flush.numel(),'prelude_cycles':300000,'rows':[]}
    rng=random.Random(20261112)
    for batch in range(3):
        jobs=[(key,cache) for key in cases for cache in ['hot','cold']];rng.shuffle(jobs)
        for key,cache in jobs:
            run,graph,result,want,validation=cases[key]
            for _ in range(100):graph.replay()
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            samples=[]
            for _ in range(100):
                if cache=='cold':flush.zero_()
                torch.cuda._sleep(300000)
                a.record();graph.replay();b.record();b.synchronize()
                samples.append(a.elapsed_time(b)*1000)
            assert torch.equal(result,want)
            row={'batch':batch,'tokens':key[0],'routing':key[1],'variant':key[2],'kind':key[3],
                 'cache':cache,'median_us':statistics.median(samples),'samples_us':samples,'validation':validation,'telemetry':telemetry()}
            data['rows'].append(row)
            (out/'timings.json').write_text(json.dumps(data,indent=2,default=str))
            print(json.dumps({k:v for k,v in row.items() if k not in ['samples_us','validation']}),flush=True)
    (out/'complete.json').write_text(json.dumps({'complete':True,'conditions':len(cases)*2,'samples':len(data['rows'])*100}))


if __name__=='__main__':main()
