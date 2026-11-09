import os,json,subprocess,sys,hashlib,shutil,tarfile
from pathlib import Path
r=Path("/content/q35moe-verify-opt-20261109");o=r/"results";s=r/"infer-opt";c=Path("/tmp/infer-opt-models/q35moe-verify-opt-20261109")
os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(c/"inductor"),TRITON_CACHE_DIR=str(c/"triton"),HF_HOME=str(c/"hf-home"),HF_HUB_DISABLE_IMPLICIT_TOKEN="1",OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
k=s/"infer_opt/moe_kernels.py";shutil.copy2(k,o/"moe-kernels-v3.py")
(o/"grouped-source.json").write_text(json.dumps({"sha256":hashlib.sha256(k.read_bytes()).hexdigest()}))
cmd=[sys.executable,str(s/"scripts/bench_moe_shared_routes.py"),"--kernel","grouped","--out",str(o/"grouped"),"--repeats","30"]
(o/"grouped.command.json").write_text(json.dumps(cmd))
with (o/"grouped.stdout").open("w") as so,(o/"grouped.stderr").open("w") as se:p=subprocess.run(cmd,stdout=so,stderr=se,timeout=1800)
(o/"grouped.exit.json").write_text(json.dumps({"exit_code":p.returncode}))
