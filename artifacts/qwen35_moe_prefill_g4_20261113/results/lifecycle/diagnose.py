from pathlib import Path
import sys,json,torch
sys.path[:0]=['/tmp/moe-prefill-g4-20261113/source','/tmp/moe-prefill-g4-20261113/source/tests']
from test_moe_prefill import reference
from infer_opt.moe_prefill import grouped_int8_experts
from infer_opt.moe import Int8Experts
from scripts.bench_moe_prefill import differences
with torch.inference_mode():
 torch.manual_seed(901)
 e,h,m,g,top=256,2048,512,128,8
 gu=torch.randint(-127,128,(e,2*m,h),device='cuda',dtype=torch.int8)
 down=torch.randint(-127,128,(e,h,m),device='cuda',dtype=torch.int8)
 gs=(torch.rand(e,2*m,h//g,device='cuda')*.0003+.0001).bfloat16()
 ds=(torch.rand(e,h,m//g,device='cuda')*.0003+.0001).bfloat16()
 mod=Int8Experts(gu,gs,down,ds,g)
 rows=[]
 for t in [128,512,2048]:
  x=torch.randn(t,h,device='cuda',dtype=torch.bfloat16)
  ids=torch.rand(t,e,device='cuda').topk(top,-1).indices
  w=torch.softmax(torch.randn(t,top,device='cuda'),-1).bfloat16()
  pure=reference(x,ids,w,gu,gs,down,ds)
  old=mod.forward_reference(x,ids,w)
  candidate=grouped_int8_experts(x,ids,w,gu,gs,down,ds)
  counts=torch.bincount(ids.flatten(),minlength=e)
  row={'tokens':t,'experts_with_1_to_8_rows':int(((counts>0)&(counts<=8)).sum()),'grouped_vs_dequantized':differences(candidate,pure),'legacy_vs_dequantized':differences(old,pure)}
  rows.append(row)
  print(json.dumps(row),flush=True)
 Path('/tmp/moe-prefill-g4-20261113/results/dequant-diagnosis.json').write_text(json.dumps(rows,indent=2))
