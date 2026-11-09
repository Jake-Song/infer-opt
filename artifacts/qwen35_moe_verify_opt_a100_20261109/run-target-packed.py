import hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
root=Path('/content/q35moe-verify-opt-20261109');out=root/'results';source=root/'infer-opt'
cache=Path('/tmp/infer-opt-models/q35moe-verify-opt-20261109')
def save(n,v):
 p=out/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
try:
 save('target-packed-controller-progress.json',{'stage':'await_checkpoint_and_micro'})
 deadline=time.time()+1800
 while not (out/'checkpoint.json').exists() or not (out/'launch-tuning.exit.json').exists():
  if (out/'download-error.json').exists():raise RuntimeError((out/'download-error.json').read_text())
  if time.time()>deadline:raise RuntimeError('checkpoint wait deadline exceeded')
  time.sleep(2)
 if json.loads((out/'launch-tuning.exit.json').read_text())['exit_code']!=0:raise RuntimeError('launch tuning failed')
 os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),HF_HOME=str(cache/'hf-home'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',TORCH_LOGS='perf_hints')
 for src,name in [(source/'infer_opt/moe_kernels.py','target-packed-executed-moe-kernels.py'),(source/'scripts/bench_moe_target_verify.py','target-packed-executed-benchmark.py')]:shutil.copy2(src,out/name)
 save('target-packed-harness.json',{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['target-packed-executed-moe-kernels.py','target-packed-executed-benchmark.py']})
 weights=json.loads((out/'checkpoint.json').read_text())['weights']
 cmd=[sys.executable,str(source/'scripts/bench_moe_target_verify.py'),'--weights',weights,'--inputs',str(out/'target-inputs.json'),'--out',str(out/'target-packed'),'--kernel','packed','--config','256,1:256,2','--repeats','30','--capacity','256']
 save('target-packed.command.json',cmd);save('target-packed-controller-progress.json',{'stage':'running'})
 with (out/'target-packed.stdout').open('w') as so,(out/'target-packed.stderr').open('w') as se:p=subprocess.run(cmd,stdout=so,stderr=se,timeout=2400)
 save('target-packed.exit.json',{'exit_code':p.returncode});save('target-packed-controller-progress.json',{'stage':'complete' if p.returncode==0 else 'failed'})
except BaseException:save('target-packed.exit.json',{'exit_code':1,'traceback':traceback.format_exc()});raise
