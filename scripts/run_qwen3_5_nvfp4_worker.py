"""Evaluate an already-published native checkpoint; preserve partial results on failure."""
import argparse
import importlib.metadata as metadata
import json
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint-repo',required=True)
    p.add_argument('--budget-seconds',type=int,required=True)
    a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    deadline=started+a.budget_seconds
    status={'status':'running','stage':'install','checkpoint_repo':a.checkpoint_repo}
    def save(): (a.root/'progress.json').write_text(json.dumps(status,indent=2))
    def expired(signum,frame): raise TimeoutError('G4 evaluation budget exhausted')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,a.budget_seconds)
    def run(args,label):
        status['stage']=label
        save()
        with (a.root/(label+'.log')).open('w') as log:
            subprocess.run([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=max(1,deadline-time.monotonic()))
    try:
        run(['-m','pip','install','-r','scripts/requirements_nvfp4_g4.txt'],'install')
        import torch
        from huggingface_hub import HfApi,snapshot_download
        assert 'RTX PRO 6000' in torch.cuda.get_device_name() and torch.cuda.get_device_capability()==(12,0)
        (a.root/'environment.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(),
            'capability':torch.cuda.get_device_capability(),
            'memory_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
            'torch':str(torch.__version__),'cuda':torch.version.cuda,
            'packages':{d.metadata['Name']:d.version for d in metadata.distributions()}},indent=2))
        status['stage']='download_native'
        save()
        api=HfApi()
        rev=api.model_info(a.checkpoint_repo).sha
        checkpoint=snapshot_download(a.checkpoint_repo,revision=rev)
        status['checkpoint_revision']=rev
        provenance=json.loads((Path(checkpoint)/'source_revision.json').read_text())
        run(['scripts/profile_qwen3_5_nvfp4.py','--checkpoint',checkpoint,
             '--out',str(a.root/'linear_profile.json')],'linear_profile')
        status['stage']='download_int8'
        save()
        int8_repo='Jakemu/Qwen3.5-27B-int8-g128-mtp'
        int8_rev=api.model_info(int8_repo).sha
        int8=snapshot_download(int8_repo,revision=int8_rev)
        status['int8_revision']=int8_rev
        status['stage']='download_reference'
        save()
        weights=snapshot_download(provenance['model'],revision=provenance['revision'],
                                  allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','LICENSE'])
        status['model_revision']=provenance['revision']
        # Evaluate quality before spending the remaining budget on repeated timings.
        # A failed quality gate is a completed experiment, not a subprocess failure.
        run(['scripts/validate_qwen3_5_decode_modes.py','--weights',weights,'--repo',provenance['model'],
             '--configs','fla-delta:compile-reduce-overhead:bf16,fla-delta:compile-reduce-overhead:int8,fla-delta:compile-reduce-overhead:nvfp4',
             '--group-size','128','--fuse','--autotune','--nvfp4-from',checkpoint,'--int8-from',int8,
             '--out',str(a.root/'accuracy.json')],'accuracy')
        for precision,source in [('bf16',weights),('int8',int8),('nvfp4',checkpoint)]:
            run(['scripts/bench_qwen3_5_nvfp4.py','--precision',precision,'--weights',source,
                 '--out',str(a.root/precision)]
                + (['--ks',''] if precision=='bf16' else ['--nvfp4-from',source]),precision)
        status['status']='finished'
    except Exception as exc:
        status.update(status='failed',error=type(exc).__name__+': '+str(exc))
        (a.root/'error.txt').write_text(traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        status['elapsed_seconds']=time.monotonic()-started
        save()
        with zipfile.ZipFile(str(a.root)+'.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for file in a.root.rglob('*'):
                if file.is_file(): archive.write(file,str(file.relative_to(a.root)))
        (a.root/'done.json').write_text(json.dumps(status))


if __name__=='__main__': main()
