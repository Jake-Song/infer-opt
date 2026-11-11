
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
