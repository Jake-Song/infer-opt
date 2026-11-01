"""Bounded full-model G4 validation worker; writes logs and progress per stage."""
import json, subprocess, sys, time, traceback
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'moe-full-validation';out.mkdir(exist_ok=True)
started=time.monotonic()
def run(args,name,timeout=1800):
    (out/'progress.json').write_text(json.dumps({'stage':name,'elapsed_s':time.monotonic()-started}))
    with (out/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable]+args,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
try:
    import torch
    assert torch.cuda.get_device_properties(0).total_memory > 90*2**30
    run(['-m','pip','install','transformers','huggingface_hub','flash-linear-attention'],'dependencies',600)
    from huggingface_hub import HfApi,snapshot_download
    repo='Qwen/Qwen3.5-35B-A3B'
    revision=HfApi().model_info(repo).sha
    (out/'environment.json').write_text(json.dumps({'model':repo,'revision':revision,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__)}))
    (out/'progress.json').write_text(json.dumps({'stage':'download','revision':revision}))
    weights=snapshot_download(repo,revision=revision,allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','LICENSE'])
    run(['scripts/bench_qwen3_5_spec_decode.py','--weights',weights,'--repo',repo,'--configs','int8+mtp',
         '--ks','1,2,3,4','--mode','cudagraph','--new-tokens','64','--prompts','short,korean,math',
         '--out',str(out/'spec-smoke.json')],'spec-smoke',1800)
    (out/'progress.json').write_text(json.dumps({'stage':'smoke-complete','elapsed_s':time.monotonic()-started}))
except Exception as exc:
    (out/'error.txt').write_text(traceback.format_exc())
    (out/'progress.json').write_text(json.dumps({'stage':'failed','error':str(exc)}))
    raise
