"""Reload an exported INT8+MTP checkpoint, check parity, and profile decode."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM,FixedShapeDecoder
from infer_opt.speculative import MTPSpeculativeDecoder
from infer_opt.moe import Int8Experts
from bench_qwen3_5_spec_decode import baseline

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    model=Qwen3_5ForCausalLM.from_quantized_dir(a.weights,device='cuda',kernels='fla-delta')
    assert model.mtp is not None
    assert sum(isinstance(m,Int8Experts) for m in model.modules())==41
    model.fuse_projections()
    tok=AutoTokenizer.from_pretrained(a.weights)
    ids=tok('Explain why the sky is blue.',return_tensors='pt').input_ids.cuda()
    decoder=FixedShapeDecoder(model,model.new_cache(1,256),'cudagraph')
    want,ms=baseline(model,decoder,ids,64)
    rows=[]
    for verification in ('sequential','consistent'):
        for k in (1,2):
            spec=MTPSpeculativeDecoder(model,256,k=k,mode='cudagraph',verification=verification)
            got=spec.generate(ids,64)[0,ids.shape[1]:]
            equal=torch.equal(got,want)
            rows.append({'verification':verification,'k':k,'tokens_equal':equal})
            if not equal: raise AssertionError(rows[-1])
            del spec
    token=want[-1:].view(1,1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(16):
            token=decoder.step(token)[:,-1].argmax(-1,keepdim=True)
        torch.cuda.synchronize()
    kernels=defaultdict(lambda:{'count':0,'device_us':0.0})
    for event in prof.events():
        if event.device_type==torch.autograd.DeviceType.CUDA:
            kernels[event.name]['count']+=1
            kernels[event.name]['device_us']+=event.device_time_total
    result={'gpu':torch.cuda.get_device_name(),'weights':a.weights,'expert_modules':41,'mtp':True,
            'allocated_gib':torch.cuda.memory_allocated()/2**30,'baseline_ms_per_token':ms,
            'parity':rows,'text_sample':tok.decode(want.tolist()),
            'profile_steps':16,'kernels':dict(sorted(kernels.items(),key=lambda kv:-kv[1]['device_us']))}
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='kernels'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
