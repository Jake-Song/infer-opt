import hashlib,json,os,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
root=Path('/content/q35moe-verify-opt-20261109');out=root/'results';src=root/'infer-opt';cache=Path('/tmp/infer-opt-models/q35moe-verify-opt-20261109')
def save(n,v):
 p=out/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def run(name,cmd):
 save('final-progress.json',{'stage':name});save(name+'.command.json',cmd)
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:p=subprocess.run(cmd,stdout=so,stderr=se,timeout=1800)
 save(name+'.exit.json',{'exit_code':p.returncode})
 if p.returncode:raise RuntimeError(name+' failed: '+(out/(name+'.stderr')).read_text()[-4000:])
try:
 save('final-progress.json',{'stage':'waiting_for_target_packed'})
 deadline=time.time()+1800
 while not (out/'target-packed.exit.json').exists():
  if time.time()>deadline:raise RuntimeError('wait timeout')
  time.sleep(2)
 if json.loads((out/'target-packed.exit.json').read_text())['exit_code']!=0:raise RuntimeError('target packed failed')
 os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),HF_HOME=str(cache/'hf-home'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',TORCH_LOGS='perf_hints')
 files=[('moe-kernels-final.py','infer_opt/moe_kernels.py'),('bench-moe-final.py','scripts/bench_moe_verification_kernel.py'),('bench-target-final.py','scripts/bench_moe_target_verify.py'),('test-kernel-final.py','tests/test_moe_verification_kernel.py')]
 manifests=[]
 for name,dst in files:
  shutil.copy2(root/name,src/dst);shutil.copy2(root/name,out/name)
  manifests.append({'file':dst,'source_file':name,'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()})
 save('final-harness.json',manifests)
 run('final-tests',[sys.executable,'-m','pytest',str(src/'tests/test_moe_verification_kernel.py'),str(src/'tests/test_moe.py'),'-q','-k','verification_projection or selected_int8_kernel'])
 run('final-micro',[sys.executable,str(src/'scripts/bench_moe_verification_kernel.py'),'--out',str(out/'final-micro'),'--kernel','packed','--repeats','100'])
 weights=json.loads((out/'checkpoint.json').read_text())['weights']
 run('final-target',[sys.executable,str(src/'scripts/bench_moe_target_verify.py'),'--weights',weights,'--inputs',str(out/'target-inputs.json'),'--out',str(out/'final-target'),'--kernel','auto','--config','256,1','--repeats','100','--capacity','256'])
 save('final-exit.json',{'exit_code':0});save('final-progress.json',{'stage':'complete'})
except BaseException:save('final-exit.json',{'exit_code':1,'traceback':traceback.format_exc()});raise
finally:
 p=Path('/content/q35moe-verify-opt-20261109-all-results.tar.gz');tmp=p.with_suffix('.tmp')
 with tarfile.open(tmp,'w:gz') as a:
  for f in sorted(out.rglob('*')):
   if f.is_file() and f.suffix!='.tmp':a.add(f,arcname=str(f.relative_to(out)))
 tmp.replace(p)
