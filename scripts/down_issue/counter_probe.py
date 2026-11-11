"""Compile and profile one model-free CUDA kernel before larger experiments."""
import json
from pathlib import Path
import subprocess

root = Path('/tmp/down-issue-a100-20261111')
root.mkdir(exist_ok=True)
source = root/'probe.cu'
source.write_text(r'''
#include <cuda_runtime.h>
#include <cstdio>
__global__ void counter_probe(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = a[i]; for (int k=0; k<64; ++k) v=fmaf(v,1.00001f,b[i]); a[i]=v; }
}
int main() {
    float *a, *b; int n=1<<20;
    if(cudaMalloc(&a,n*4)!=cudaSuccess || cudaMalloc(&b,n*4)!=cudaSuccess) return 2;
    cudaMemset(a,0,n*4); cudaMemset(b,0,n*4);
    for(int i=0;i<5;++i) counter_probe<<<(n+255)/256,256>>>(a,b,n);
    auto s=cudaDeviceSynchronize(); printf("cuda_status=%d\n",int(s));
    cudaFree(a); cudaFree(b); return int(s);
}
''')
def run(name, command):
    p = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (root/(name+'.stdout')).write_text(p.stdout)
    (root/(name+'.stderr')).write_text(p.stderr)
    (root/(name+'.command.json')).write_text(json.dumps({'command':command,'returncode':p.returncode},indent=2))
    return p
run('ncu-version',['ncu','--version'])
p=run('compile',['nvcc','-O3','-lineinfo','-arch=sm_80',str(source),'-o',str(root/'probe')])
if p.returncode:
    raise RuntimeError(p.stderr)
bare=run('cuda-probe',[str(root/'probe')])
p=run('ncu-probe',['ncu','--set','basic','--launch-skip','3','--launch-count','1','--csv',str(root/'probe')])
result={'cuda_returncode':bare.returncode,'profiler_returncode':p.returncode,
        'permission_denied':'ERR_NVGPUCTRPERM' in p.stdout+p.stderr,
        'counters_collected':p.returncode==0 and 'Metric Name' in p.stdout}
(root/'probe-result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
print(p.stdout[-6000:]+p.stderr[-2000:],flush=True)
