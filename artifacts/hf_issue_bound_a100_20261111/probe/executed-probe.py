"""Small HF Jobs A100 performance-counter permission probe; no models."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path('/tmp/hf-issue-check')
ROOT.mkdir(parents=True,exist_ok=True)
os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH']


def run(command,name):
    p=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
    (ROOT/(name+'.stdout')).write_text(p.stdout)
    (ROOT/(name+'.stderr')).write_text(p.stderr)
    (ROOT/(name+'.json')).write_text(json.dumps({'command':command,'returncode':p.returncode},indent=2))
    return p


def export():
    memory=io.BytesIO()
    with tarfile.open(fileobj=memory,mode='w:gz') as tar:
        for p in sorted(ROOT.iterdir()):
            if p.is_file() and p.name!='probe':tar.add(p,arcname=p.name)
    data=memory.getvalue();encoded=base64.b64encode(data).decode()
    chunks=[encoded[i:i+12000] for i in range(0,len(encoded),12000)]
    print('HF_ARTIFACT_BEGIN '+json.dumps({'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'chunks':len(chunks)}),flush=True)
    for i,chunk in enumerate(chunks):print('HF_ARTIFACT_CHUNK '+json.dumps({'index':i,'data':chunk}),flush=True)
    print('HF_ARTIFACT_END',flush=True)


try:
    print('STAGE permission diagnostics',flush=True)
    diag={'uid':os.geteuid(),'nvcc':shutil.which('nvcc'),'ncu':shutil.which('ncu')}
    for name in ['CapEff','CapPrm','CapBnd']:
        line=next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith(name+':'))
        value=int(line.split()[1],16)
        diag[name]={'hex':line.split()[1],'SYS_ADMIN':bool(value&(1<<21)),'PERFMON':bool(value&(1<<38))}
    params=Path('/proc/driver/nvidia/params')
    diag['driver_profiling_params']=[x for x in params.read_text().splitlines() if 'Profil' in x] if params.exists() else None
    (ROOT/'permissions.json').write_text(json.dumps(diag,indent=2))
    print('DIAGNOSTICS '+json.dumps(diag),flush=True)
    run(['nvidia-smi','-q'],'nvidia-smi')
    run(['nvcc','--version'],'nvcc')
    run(['ncu','--version'],'ncu-version')
    source=ROOT/'probe.cu'
    source.write_text(r'''
#include <cuda_runtime.h>
#include <cstdio>
__global__ void counter_probe(float * a,const float * b,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {float v=a[i];for(int k=0;k<64;++k)v=fmaf(v,1.00001f,b[i]);a[i]=v;}
}
int main(){
    float *a,*b;int n=1<<20;
    if(cudaMalloc(&a,n*4)!=cudaSuccess||cudaMalloc(&b,n*4)!=cudaSuccess)return 2;
    cudaMemset(a,0,n*4);cudaMemset(b,0,n*4);
    for(int i=0;i<5;++i)counter_probe<<<(n+255)/256,256>>>(a,b,n);
    auto s=cudaDeviceSynchronize();printf("probe_cuda_status=%d\n",int(s));
    cudaFree(a);cudaFree(b);return s==cudaSuccess?0:3;
}
''')
    compiled=run(['nvcc','-O3','-lineinfo','-arch=sm_80',str(source),'-o',str(ROOT/'probe')],'compile')
    if compiled.returncode:raise RuntimeError(compiled.stderr)
    print('STAGE counter collection',flush=True)
    prof=run(['ncu','--set','basic','--launch-skip','3','--launch-count','1','--csv',str(ROOT/'probe')],'ncu-probe')
    denied='ERR_NVGPUCTRPERM' in prof.stdout+prof.stderr
    result={'returncode':prof.returncode,'counter_permission_denied':denied,
            'counter_collection_succeeded':prof.returncode==0 and 'Metric Name' in prof.stdout}
    (ROOT/'result.json').write_text(json.dumps(result,indent=2))
    print('RESULT '+json.dumps(result),flush=True)
except BaseException as error:
    (ROOT/'error.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error)},indent=2))
    print('ERROR '+str(error),flush=True)
finally:
    export()
