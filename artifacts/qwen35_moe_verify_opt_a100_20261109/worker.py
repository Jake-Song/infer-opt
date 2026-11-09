import json,os,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
BUNDLE=Path(__file__).resolve().parent
ROOT=Path('/content/q35moe-verify-opt-20261109')
OUT=ROOT/'results'
CACHE=Path('/tmp/infer-opt-models/q35moe-verify-opt-20261109')
START=time.time()
def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def snapshot():
 p=Path('/content/q35moe-verify-opt-20261109-results.tar.gz');t=p.with_suffix('.tmp')
 with tarfile.open(t,'w:gz') as a:
  for f in OUT.rglob('*'):
   if f.is_file() and f.suffix!='.tmp':a.add(f,arcname=str(f.relative_to(OUT)))
 t.replace(p)
def stage(name):save('progress.json',{'stage':name,'elapsed_s':time.time()-START});print(name,flush=True);snapshot()
def run(cmd,name,timeout=1800):
 save(name+'.command.json',cmd)
 with (OUT/(name+'.stdout')).open('w') as so,(OUT/(name+'.stderr')).open('w') as se:p=subprocess.run(cmd,stdout=so,stderr=se,timeout=timeout)
 save(name+'.exit.json',{'exit_code':p.returncode});snapshot()
 if p.returncode:raise RuntimeError(name+' failed: '+(OUT/(name+'.stderr')).read_text()[-4000:])
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 src=ROOT/'infer-opt';src.mkdir()
 with tarfile.open(BUNDLE/'source-v1.tar.gz') as a:a.extractall(src,filter='data')
 shutil.copy2(BUNDLE/'source-v1-manifest.json',OUT/'source-v1-manifest.json')
 shutil.copy2(src/'infer_opt/moe_kernels.py',OUT/'moe-kernels-v1.py')
 os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(CACHE/'inductor'),TRITON_CACHE_DIR=str(CACHE/'triton'),HF_HOME=str(CACHE/'hf-home'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
 import torch
 gpu=torch.cuda.get_device_properties(0)
 if 'A100' not in gpu.name:raise RuntimeError('Expected A100, got '+gpu.name)
 save('initial-runtime.json',{'gpu':gpu.name,'vram_gib':gpu.total_memory/2**30,'torch':str(torch.__version__)})
 stage('dependencies')
 run([sys.executable,'-m','pip','uninstall','-y','torchvision','torchaudio'],'remove-unused-av')
 run([sys.executable,'-m','pip','install','torch==2.11.0','triton==3.6.0','--index-url','https://download.pytorch.org/whl/cu130'],'install-torch')
 run([sys.executable,'-m','pip','install','pytest','transformers==5.22.0','accelerate==1.15.0','safetensors==0.8.0','huggingface_hub==1.33.0','flash-linear-attention==0.6.0','fla-core==0.6.0'],'install-tools')
 run([sys.executable,'-c','import torch,triton,json; print(json.dumps({"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),"triton":triton.__version__}))'],'environment')
 stage('expert_micro_tuning')
 run([sys.executable,str(src/'scripts/bench_moe_shared_routes.py'),'--out',str(OUT/'micro'),'--repeats','20'],'micro',timeout=1800)
 stage('kernel_tests')
 run([sys.executable,'-m','pytest',str(src/'tests/test_moe_shared_routes.py'),'-q'],'tests')
 stage('complete')
if __name__=='__main__':
 try:main();save('exit.json',{'exit_code':0})
 except BaseException:save('exit.json',{'exit_code':1,'traceback':traceback.format_exc()});traceback.print_exc()
 finally:snapshot()
