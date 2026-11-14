"""Optional BF16 GEMV runtime override; weights and checkpoint names stay unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl
from torch.library import custom_op,wrap_triton

@triton.jit
def _kernel(X,W,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BN+tl.arange(0,BN)
    cols=tl.arange(0,BK)
    acc=tl.zeros((BN,BK),tl.float32)
    for start in range(tl.cdiv(K,BK)):
        col=start*BK+cols
        x=tl.load(X+col,col<K,0).to(tl.float32)
        w=tl.load(W+rows[:,None].to(tl.int64)*K+col[None,:],(rows[:,None]<N)&(col[None,:]<K),0).to(tl.float32)
        acc+=w*x[None,:]
    tl.store(Y+rows,tl.sum(acc,1),rows<N)

@custom_op('infer_opt::bf16_gemv',mutates_args=())
def bf16_gemv(x:torch.Tensor,weight:torch.Tensor,bn:int,bk:int,warps:int)->torch.Tensor:
    if weight.ndim != 2 or x.ndim < 1 or x.shape[-1] != weight.shape[1] or x.numel() != weight.shape[1]:
        raise ValueError("bf16_gemv requires one activation row and a 2-D weight")
    if not x.is_cuda or x.device != weight.device or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("bf16_gemv requires BF16 tensors on the same CUDA device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("bf16_gemv requires contiguous inputs")
    if bn <= 0 or bn & (bn - 1) or bk <= 0 or bk & (bk - 1) or warps not in (1, 2, 4, 8, 16):
        raise ValueError("invalid GEMV launch configuration")
    n,k=weight.shape
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    wrap_triton(_kernel)[(triton.cdiv(n,bn),)](x,weight,y,n,k,bn,bk,num_warps=warps,enable_fp_fusion=False)
    return y

@bf16_gemv.register_fake
def _fake(x,weight,bn,bk,warps):
    return x.new_empty((*x.shape[:-1],weight.shape[0]))

class BF16GemvLinear(nn.Linear):
    launch=(1,512,4)
    def forward(self,x):
        if (x.is_cuda and self.weight.is_cuda and x.dtype==torch.bfloat16
                and self.weight.dtype==torch.bfloat16 and self.weight.is_contiguous()
                and x.ndim >= 1 and x.shape[-1]==self.in_features and x.numel()==self.in_features):
            output=bf16_gemv(x.contiguous(),self.weight,*self.launch)
            return output if self.bias is None else output+self.bias
        return F.linear(x,self.weight,self.bias)


def install_bf16_gemv(model,configs):
    """Install only shapes supplied by an explicit tuning table; keep BF16 storage."""
    names=[n for n,m in model.named_modules() if type(m) is nn.Linear and (m.out_features,m.in_features) in configs]
    for name in names:
        old=model.get_submodule(name)
        if old.weight.dtype!=torch.bfloat16: continue
        with torch.device('meta'):
            new=BF16GemvLinear(old.in_features,old.out_features,old.bias is not None)
        new.weight,new.bias=old.weight,old.bias
        new.train(old.training)
        new.launch=tuple(configs[(old.out_features,old.in_features)])
        parent,_,child=name.rpartition('.')
        setattr(model.get_submodule(parent),child,new)
    return names
