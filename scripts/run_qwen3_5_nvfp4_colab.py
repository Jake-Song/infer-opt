#!/usr/bin/env python3
"""Bounded G4 experiment. Recover partial artifacts and stop only this run's session."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def snapshot_code(root, archive):
    """Collect partial evidence while the worker runs in its own process."""
    return ("from pathlib import Path\nimport zipfile\n"
            + f"root=Path({str(root)!r})\narchive=Path({str(archive)!r})\n"
            + "temporary=archive.with_suffix('.tmp')\n"
            + "with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:\n"
            + "    for f in root.rglob('*'):\n"
            + "        if f.is_file() and f.suffix in ('.json','.log','.txt'):\n"
            + "            z.write(f,str(f.relative_to(root)))\n"
            + "temporary.replace(archive)\n")


def extract_results(archive, output):
    import zipfile
    with zipfile.ZipFile(archive) as z:
        if any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()):
            raise ValueError('unsafe ZIP')
        z.extractall(output)


def worker_code(run_id, seconds, checkpoint_repo=None, mlp_only=False):
    if mlp_only:
        return ("import os,sys,runpy\nos.chdir('/content/infer-opt-nvfp4')\n"
                + f"sys.argv=['run_qwen3_5_nvfp4_mlp_worker.py','--root','/content/{run_id}',"
                + f"'--budget-seconds',{str(seconds)!r}]\n"
                + "runpy.run_path('scripts/run_qwen3_5_nvfp4_mlp_worker.py',run_name='__main__')\n")
    if checkpoint_repo:
        return ("import os,sys,runpy\nos.chdir('/content/infer-opt-nvfp4')\n"
                +f"sys.argv=['run_qwen3_5_nvfp4_worker.py','--root','/content/{run_id}',"
                +f"'--budget-seconds',{str(seconds)!r},'--checkpoint-repo',{checkpoint_repo!r}]\n"
                +"runpy.run_path('scripts/run_qwen3_5_nvfp4_worker.py',run_name='__main__')\n")
    return f"""
import importlib.metadata as metadata
import json, os, signal, subprocess, sys, time, traceback, zipfile
from pathlib import Path
import torch
root=Path('/content/{run_id}')
root.mkdir(exist_ok=True)
os.chdir('/content/infer-opt-nvfp4')
started=time.monotonic()
deadline=started+{seconds}
def budget_expired(signum, frame):
    raise TimeoutError('G4 worker budget exhausted')
signal.signal(signal.SIGALRM,budget_expired)
signal.setitimer(signal.ITIMER_REAL,{seconds})
status={{'status':'running','stage':'preflight'}}
def save():
    (root/'progress.json').write_text(json.dumps(status,indent=2))
def run(args,label):
    status['stage']=label
    save()
    with (root/(label+'.log')).open('w') as log:
        subprocess.run([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,check=True,
                       timeout=max(1,deadline-time.monotonic()))
try:
    assert 'RTX PRO 6000' in torch.cuda.get_device_name() and torch.cuda.get_device_capability()==(12,0)
    run(['-m','pip','install','-r','scripts/requirements_nvfp4_g4.txt'],'install')
    (root/'environment.json').write_text(json.dumps({{'gpu':torch.cuda.get_device_name(),
       'capability':torch.cuda.get_device_capability(),'memory_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
       'torch':str(torch.__version__),'cuda':torch.version.cuda,'packages':{{d.metadata['Name']:d.version for d in metadata.distributions()}}}},indent=2))
    run(['scripts/bench_qwen3_5_nvfp4.py','--smoke','--out',str(root/'smoke')],'smoke')
    from huggingface_hub import HfApi, snapshot_download
    rev=HfApi().model_info('Qwen/Qwen3.5-27B').sha
    status['model_revision']=rev
    status['stage']='download'
    save()
    weights=snapshot_download('Qwen/Qwen3.5-27B',revision=rev,allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','LICENSE'])
    checkpoint=Path('/content/{run_id}-checkpoint')
    run(['scripts/bench_qwen3_5_nvfp4.py','--precision','nvfp4','--weights',weights,
         '--export',str(checkpoint),'--export-only','--out',str(root/'export')],'quantize_export')
    publish_config=Path('/content/{run_id}-publish.json')
    if publish_config.exists():
        status['stage']='publish'
        save()
        from huggingface_hub import HfApi
        config=json.loads(publish_config.read_text())
        api=HfApi(token=config['token'])
        repo=config['repo']
        (checkpoint/'README.md').write_text('''---
license: apache-2.0
base_model: Qwen/Qwen3.5-27B
base_model_relation: quantized
tags:
- nvfp4
- qwen3.5
- experimental
---
# Qwen3.5-27B native NVFP4 + INT8 MTP

Experimental custom infer-opt checkpoint. This is **not a Transformers/vLLM checkpoint**.
Target large Linear weights and activations use native NVFP4 W4A4, block size 16,
with ModelOpt calibration. MTP uses group-128 INT8. Embeddings, lm_head and small
delta-rule gates stay BF16. No vision tower is included.

Validated on RTX PRO 6000 Blackwell Server Edition SM120, torch 2.11 CUDA 13,
FlashInfer 0.6.6 CUTLASS. Prefill GEMMs are tiled at 128 rows to avoid a large-M
CUDA Graph discrepancy. Saved/reloaded final-position logits are identical on a
short smoke prompt; see validation.json. The full accuracy gate and performance
evaluation are pending. No quality or speed improvement is claimed.

```python
from huggingface_hub import snapshot_download
from pathlib import Path
import sys
path = Path(snapshot_download("REPO_ID"))
sys.path.insert(0, str(path / "code"))
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM
model = Qwen3_5ForCausalLM.from_quantized_dir(path, device="cuda", kernels="fla-delta")
```

Install flashinfer-python==0.6.6, flash-linear-attention, transformers>=5,
safetensors and a compatible CUDA PyTorch. CUDA SM120 is required for this
native execution path. Calibration details and source hashes accompany the weights.
'''.replace('REPO_ID',repo))
        (checkpoint/'source_revision.json').write_text(json.dumps({{'model':'Qwen/Qwen3.5-27B','revision':rev}}))
        api.create_repo(repo,repo_type='model',exist_ok=True)
        commit=api.upload_folder(repo_id=repo,folder_path=str(checkpoint),commit_message='Upload experimental native NVFP4 27B with INT8 MTP')
        status['published_repo']=repo
        status['published_commit']=commit.oid
        save()
        publish_config.unlink()
    for precision in ['nvfp4','int8','bf16']:
        run(['scripts/bench_qwen3_5_nvfp4.py','--precision',precision,'--weights',weights,
             '--out',str(root/precision)]+(['--nvfp4-from',str(checkpoint)] if precision=='nvfp4' else [])
             +(['--ks',''] if precision=='bf16' else []),precision)
    run(['scripts/validate_qwen3_5_decode_modes.py','--weights',weights,'--repo','Qwen/Qwen3.5-27B',
         '--configs','fla-delta:compile-reduce-overhead:bf16,fla-delta:compile-reduce-overhead:int8,fla-delta:compile-reduce-overhead:nvfp4',
         '--group-size','128','--fuse','--autotune','--nvfp4-from',str(checkpoint),
         '--out',str(root/'accuracy.json')],'accuracy')
    status['status']='finished'
except Exception as exc:
    status['status']='failed'
    status['error']=type(exc).__name__+': '+str(exc)
    (root/'error.txt').write_text(traceback.format_exc())
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
    status['elapsed_seconds']=time.monotonic()-started
    save()
    with zipfile.ZipFile('/content/{run_id}.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in root.rglob('*'):
            if f.is_file(): z.write(f,str(f.relative_to(root)))
    (root/'done.json').write_text(json.dumps(status))
"""


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--session',help='Explicitly transfer ownership of this G4 session; it will be stopped')
    p.add_argument('--budget-minutes',type=float,default=120)
    p.add_argument('--publish-repo',help='Publish the validated native checkpoint to this authorized HF model repo')
    p.add_argument('--checkpoint-repo',help='Evaluate this existing native Hub checkpoint instead of quantizing again')
    p.add_argument('--mlp-only',action='store_true',help='Recalibrate MLP-only NVFP4 and compare accuracy to BF16 and full NVFP4')
    a=p.parse_args()
    if a.mlp_only and (a.checkpoint_repo or a.publish_repo):
        p.error('--mlp-only is a local accuracy experiment; do not combine with checkpoint/publication options')
    if not 15<=a.budget_minutes<=120: p.error('budget must be 15..120 minutes')
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    run_id='nvfp4-'+uuid.uuid4().hex[:10]
    session=a.session or run_id
    state={'session':session,'budget_minutes':a.budget_minutes,'status':'starting'}
    import os
    (a.output/'supervisor.json').write_text(json.dumps({'pid':os.getpid(),'session':session,
        'run_id':run_id,'started_unix':time.time(),'mlp_only':a.mlp_only},indent=2))
    def cmd(*args,check=True,timeout=120):
        r=subprocess.run(['colab',*args],capture_output=True,text=True,timeout=timeout)
        with (a.output/'colab.log').open('a') as f: f.write(r.stdout+r.stderr)
        if check and r.returncode: raise RuntimeError('colab '+args[0]+' failed: '+r.stdout[-1000:]+r.stderr[-1000:])
        return r
    owned=False
    try:
        listing=cmd('sessions').stdout
        if a.session:
            if f'[{session}]' not in listing: raise RuntimeError('requested session is not active')
            owned=True
        else:
            if 'No active sessions found on server.' not in listing: raise RuntimeError('active sessions exist; select --session')
            owned=True
            cmd('new','-s',session,'--gpu','G4',timeout=300)
        bundle=a.output/'source.tar.gz'
        manifest={}
        with tarfile.open(bundle,'w:gz') as tar:
            for folder in ['infer_opt','scripts','tests']:
                for f in sorted((ROOT/folder).rglob('*')):
                    if f.is_file() and f.suffix in ('.py','.txt'):
                        rel=str(f.relative_to(ROOT))
                        tar.add(f,arcname=rel)
                        manifest[rel]=hashlib.sha256(f.read_bytes()).hexdigest()
        (a.output/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
        cmd('upload','-s',session,str(bundle),'content/'+run_id+'.tar.gz')
        if a.publish_repo:
            from huggingface_hub import get_token
            import tempfile, os
            token=get_token()
            if not token: raise RuntimeError('Hugging Face login required for publication')
            with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as secret:
                secret.write(json.dumps({'repo':a.publish_repo,'token':token}))
                secret_path=secret.name
            try:
                cmd('upload','-s',session,secret_path,f'content/{run_id}-publish.json')
            finally:
                os.unlink(secret_path)
        bootstrap=a.output/'bootstrap.py'
        remaining=int(a.budget_minutes*60-(time.monotonic()-started)-600)
        worker=worker_code(run_id,remaining,a.checkpoint_repo,mlp_only=a.mlp_only)
        bootstrap.write_text('import tarfile, pathlib, subprocess, sys\n'
            +"dest=pathlib.Path('/content/infer-opt-nvfp4'); dest.mkdir(exist_ok=True)\n"
            +f"with tarfile.open('/content/{run_id}.tar.gz') as t: t.extractall(dest,filter='data')\n"
            +f"p=pathlib.Path('/content/{run_id}-worker.py'); p.write_text({worker!r})\n"
            +f"log=open('/content/{run_id}-supervisor.log','w')\n"
            +"subprocess.Popen([sys.executable,str(p)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n")
        cmd('exec','-s',session,'-f',str(bootstrap),'--timeout','60')
        snapshot=a.output/'snapshot.py'
        snapshot.write_text(snapshot_code('/content/'+run_id,'/content/'+run_id+'-partial.zip'))
        last_snapshot=0
        last_progress=None
        while time.monotonic()-started<a.budget_minutes*60-300:
            r=cmd('download','-s',session,f'content/{run_id}/done.json',str(a.output/'done.json'),check=False)
            if r.returncode==0: break
            if "Session '" in r.stdout+r.stderr and 'not found' in r.stdout+r.stderr:
                raise RuntimeError('Colab session disappeared; latest partial snapshot is preserved locally')
            if time.monotonic()-last_snapshot>=60:
                cmd('exec','-s',session,'-f',str(snapshot),'--timeout','30')
                cmd('download','-s',session,f'content/{run_id}-partial.zip',str(a.output/'partial.zip'))
                extract_results(a.output/'partial.zip',a.output)
                last_snapshot=time.monotonic()
            if (a.output/'progress.json').exists():
                progress=(a.output/'progress.json').read_text()
                if progress!=last_progress:
                    print(progress,flush=True)
                    last_progress=progress
            time.sleep(20)
        cmd('download','-s',session,f'content/{run_id}.zip',str(a.output/'results.zip'))
        extract_results(a.output/'results.zip',a.output)
        state.update(json.loads((a.output/'done.json').read_text()))
    except KeyboardInterrupt:
        state.update(status='interrupted',error='Local supervisor interrupted; see partial evidence')
        raise
    except Exception as exc:
        state.update(status='failed',error=str(exc))
        raise
    finally:
        if owned:
            state['stop_returncode']=cmd('stop','-s',session,check=False).returncode
        state['elapsed_seconds']=time.monotonic()-started
        (a.output/'lifecycle.json').write_text(json.dumps(state,indent=2))
        if a.mlp_only:
            from summarize_qwen3_5_nvfp4_mlp import summarize
        else:
            from summarize_qwen3_5_nvfp4 import summarize
        summarize(a.output)
    return 0 if state['status']=='finished' else 1


if __name__=='__main__':
    raise SystemExit(main())
