import hashlib,json,os,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
root=Path('/content/q35moe-round-profile-20261109');out=root/'results';cache=Path('/tmp/infer-opt-models/q35moe-round-profile-20261109');source=root/'infer-opt/scripts/profile_qwen3_5_moe_round.py'
def save(name,value):
 p=out/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def snapshot():
 path=Path('/content/q35moe-round-profile-20261109-results.tar.gz');tmp=path.with_suffix('.tmp')
 with tarfile.open(tmp,'w:gz') as a:
  for p in sorted(out.rglob('*')):
   if p.is_file() and p.suffix!='.tmp':a.add(p,arcname=str(p.relative_to(out)))
 tmp.replace(path)
try:
 save('calibrated-progress.json',{'stage':'waiting_for_initial_profiles'})
 deadline=time.time()+3600
 while not (out/'exit.json').exists():
  if time.time()>deadline:raise RuntimeError('initial profiling deadline exceeded')
  time.sleep(2)
 initial=json.loads((out/'exit.json').read_text())
 if initial['exit_code']!=0:raise RuntimeError('initial profiling failed')
 expected=json.loads((out/'diagnostic-harness.json').read_text())['sha256']
 actual=hashlib.sha256(source.read_bytes()).hexdigest()
 if actual!=expected:raise RuntimeError('calibrated source hash mismatch')
 save('calibrated-harness.json',{'sha256':actual,'executed_source':'calibrated-profile-source.py','initial_diagnostic_executed_source':'profile_qwen3_5_moe_round.py','note':'Initial diagnostic process had loaded v1 before v2 upload; this fresh process executes v2 and adds calibrated manual full-round control.'})
 shutil.copy2(source,out/'calibrated-profile-source.py')
 save('calibrated-progress.json',{'stage':'running'})
 env=os.environ.copy();env.update(HF_HOME=str(cache/'hf-home'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',TORCH_LOGS='perf_hints')
 weights=json.loads((out/'production/manifest.json').read_text())['args']['weights']
 command=[sys.executable,str(source),'--stage','diagnostic','--weights',weights,'--out',str(out/'calibrated'),'--ks','1,3','--prompts','math,korean','--cache-capacity','2048','--event-repeats','20','--micro-repeats','10']
 save('calibrated.command.json',command)
 with (out/'calibrated.stdout').open('w') as stdout,(out/'calibrated.stderr').open('w') as stderr:
  result=subprocess.run(command,stdout=stdout,stderr=stderr,env=env,timeout=3600)
 if result.returncode:raise RuntimeError('calibrated profiling failed: '+(out/'calibrated.stderr').read_text()[-4000:])
 save('calibrated-exit.json',{'exit_code':0});save('calibrated-progress.json',{'stage':'complete'})
except BaseException:
 save('calibrated-exit.json',{'exit_code':1,'traceback':traceback.format_exc()});traceback.print_exc()
finally:snapshot()
