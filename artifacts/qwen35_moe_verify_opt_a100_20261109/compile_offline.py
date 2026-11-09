"""Offline SM80 compilation with explicit target; no GPU is queried or launched."""
import os
os.environ['TRITON_CACHE_DIR']='/tmp/infer-opt-models/q35moe-verify-opt-20261109/offline-triton'
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from infer_opt.moe_kernels import _project_shared_routes
root=Path(__file__).resolve().parent
out=root/'offline-sm80';out.mkdir(exist_ok=True)
rows=[]
for t in [1,2,3,4,5]:
 for gate in [True,False]:
  for bn in [1,2,4]:
   n,k=(1024,2048) if gate else (2048,512)
   c={'N':n,'K':k,'TOP':8,'TOKENS':t,'GROUP':128,'GATE':gate,'BK':256,'BN':bn,
      'BT':triton.next_power_of_2(t),'BR':triton.next_power_of_2(t*8),'BS':8}
   src=ASTSource(_project_shared_routes,{'X':'*bf16','Q':'*i8','S':'*bf16','IDS':'*i64','Y':'*bf16'},c)
   compiled=triton.compile(src,target=GPUTarget('cuda',80,32),options={'num_warps':4,'enable_fp_fusion':False})
   name=f't{t}-gate{int(gate)}-bn{bn}.ptx'
   data=compiled.asm['ptx'].encode();(out/name).write_bytes(data)
   rows.append({'tokens':t,'gate':gate,'bn':bn,'shared_bytes':compiled.metadata.shared,
                'ptx':name,'ptx_sha256':hashlib.sha256(data).hexdigest(),'compiled':True})
result={'triton':triton.__version__,'target':'sm80','executed_on_gpu':False,'rows':rows}
(root/'offline-compilation.json').write_text(json.dumps(result,indent=2))
print('Offline',triton.__version__,'SM80 compiled',len(rows),'variants')
