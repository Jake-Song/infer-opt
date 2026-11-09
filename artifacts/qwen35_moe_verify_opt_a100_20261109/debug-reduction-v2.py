import torch,triton,json
import triton.language as tl
from pathlib import Path
@triton.jit
def reduce(X,Y,KIND:tl.constexpr):
 r=tl.program_id(0); c=tl.arange(0,256);v=tl.load(X+r*256+c)
 if KIND==0:z=tl.sum(v,0)
 elif KIND==1:z=tl.sum(tl.sum(tl.sum(tl.reshape(v,(2,4,32)),0),1),0)
 elif KIND==2:z=tl.sum(tl.sum(tl.sum(tl.reshape(v,(4,32,2)),2),1),0)
 elif KIND==4:
  z4=tl.sum(tl.sum(tl.reshape(v,(4,32,2)),2),1);i=tl.arange(0,4)
  z=tl.sum(tl.where((i%2)==0,z4,0),0)+tl.sum(tl.where((i%2)==1,z4,0),0)
 elif KIND==3:
  for shift in tl.static_range(8):
   if shift==0:s=128
   elif shift==1:s=16
   elif shift==2:s=8
   elif shift==3:s=4
   elif shift==4:s=2
   elif shift==5:s=1
   elif shift==6:s=64
   else:s=32
   v=v+tl.gather(v,c^s,0)
  z=tl.sum(tl.where(c==0,v,0),0)
 tl.store(Y+r,z)
torch.manual_seed(9);x=torch.randn(4096,256,device="cuda");base=torch.empty(4096,device="cuda");reduce[(4096,)](x,base,0,num_warps=4,enable_fp_fusion=False)
rows=[]
for kind in [4]:
 for w in [1,2,4]:
  y=torch.empty_like(base);reduce[(4096,)](x,y,kind,num_warps=w,enable_fp_fusion=False);rows.append({"kind":kind,"warps":w,"exact":bool(torch.equal(base,y)),"different":(base!=y).sum().item(),"max_abs_diff":(base-y).abs().max().item()})
print(json.dumps(rows),flush=True)
Path("/content/q35moe-verify-opt-20261109/results/debug-reduction-v2.json").write_text(json.dumps(rows,indent=2))
