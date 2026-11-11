"""Measure native contiguous weights and inspect captured CUDA graph kernels."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import worker

src = worker.ROOT/'llama.cpp'
worker.stage('native-layout-build')
for relative in ['tests/test-backend-ops.cpp','tests/CMakeLists.txt','ggml/src/ggml-cuda/ggml-cuda.cu']:
    original = subprocess.check_output(['git','-C',str(src),'show','HEAD:'+relative])
    (src/relative).write_bytes(original)
worker.run([sys.executable,str(worker.BUNDLE/'scripts/llamacpp_moe/patch_source.py'),str(src)],'native-patch')
worker.run(['git','-C',str(src),'diff'],'native-source-patch')
worker.run(['cmake','--build',str(src/'build'),'-j','12','--target','test-backend-ops'],'native-build')
bench = str(src/'build/bin/test-backend-ops')
env = dict(os.environ,INFER_MOE_CASES='1',INFER_MOE_NATIVE_LAYOUT='1')
worker.stage('native-layout-correctness')
worker.run([bench,'test','-b','CUDA0'],'native-correctness',env=env,timeout=1200)
worker.stage('native-layout-perf')
dot = worker.OUT/'native-graphs'
dot.mkdir(exist_ok=True)
worker.run([bench],'native-isolated-graph-perf',env=dict(env,INFER_MOE_MEASURE='1',INFER_MOE_DOT_DIR=str(dot)),timeout=1200)
worker.stage('view-layout-graph-control')
dot = worker.OUT/'view-graphs'
dot.mkdir(exist_ok=True)
control = dict(os.environ,INFER_MOE_CASES='1',INFER_MOE_MEASURE='1',INFER_MOE_DOT_DIR=str(dot),
               INFER_MOE_SELECT='tokens=4,kind=gate_up,routing=partial',INFER_MOE_REPEATS='2')
worker.run([bench],'view-graph-control',env=control)
worker.save('native-source-manifest.json',{str(p.relative_to(worker.BUNDLE)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in worker.BUNDLE.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
worker.run(['nvidia-smi','-q'],'gpu-final')
worker.stage('native-followup-complete')
worker.save('native-exit.json',{'returncode':0})
worker.snapshot()
