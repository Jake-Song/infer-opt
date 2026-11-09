"""CPU interpreter arithmetic check; does not establish CUDA equality or speed."""
import os
os.environ['TRITON_INTERPRET'] = '1'
import json
from pathlib import Path
import sys
import torch
import triton
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from infer_opt.moe_kernels import _project, _project_shared_routes, _combine

root = Path(__file__).resolve().parent
rows = []
torch.manual_seed(117)
e, h, mid, group, top = 16, 16, 8, 4, 3
for dtype in [torch.bfloat16, torch.float16]:
    gu = torch.randint(-127, 128, (e, 2*mid, h), dtype=torch.int8)
    down = torch.randint(-127, 128, (e, h, mid), dtype=torch.int8)
    gs = (torch.rand(e, 2*mid, h//group)*.0003+.0001).to(dtype)
    ds = (torch.rand(e, h, mid//group)*.0003+.0001).to(dtype)
    for t in [1, 2, 3, 4, 5]:
        x = torch.randn(t, h, dtype=dtype)
        w = torch.randn(t, top).softmax(-1).to(dtype)
        for routing in ['shared', 'partial', 'disjoint']:
            ids = torch.arange(t*top).reshape(t, top)
            if routing == 'shared': ids = (ids + torch.arange(t)[:, None]) % top
            if routing == 'partial': ids[:, 0] = 15
            a = torch.empty(t*top, mid, dtype=dtype)
            b = torch.empty_like(a)
            ya = torch.empty(t*top, h, dtype=dtype)
            yb = torch.empty_like(ya)
            oa, ob = torch.empty_like(x), torch.empty_like(x)
            _project[(t*top, mid)](x,gu,gs,ids,a,2*mid,h,top,group,True,h)
            _project[(t*top, h)](a,down,ds,ids,ya,h,mid,top,group,False,mid)
            _combine[(t,1)](ya,w,oa,h,top,16)
            bt,br,bs = triton.next_power_of_2(t),triton.next_power_of_2(t*top),4
            _project_shared_routes[(t*top,mid//2)](x,gu,gs,ids,b,2*mid,h,top,t,group,True,h,2,bt,br,bs)
            _project_shared_routes[(t*top,h//2)](b,down,ds,ids,yb,h,mid,top,t,group,False,mid,2,bt,br,bs)
            _combine[(t,1)](yb,w,ob,h,top,16)
            assert torch.equal(a,b), ('gate',dtype,t,routing)
            assert torch.equal(ya,yb), ('down',dtype,t,routing)
            assert torch.equal(oa,ob), ('combine',dtype,t,routing)
            rows.append({'dtype':str(dtype),'tokens':t,'top':top,'routing':routing,'gate_exact':True,'down_exact':True,'combined_exact':True})
result={'triton':triton.__version__,'scope':'CPU interpreter only; not actual GPU execution or performance.','rows':rows}
(root/'interpreter-validation.json').write_text(json.dumps(result,indent=2))
print('CPU interpreter:',len(rows),'cases exact')
