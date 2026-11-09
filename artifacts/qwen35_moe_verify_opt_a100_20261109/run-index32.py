import os,json,subprocess,sys,hashlib,shutil,tarfile
from pathlib import Path
r=Path("/content/q35moe-verify-opt-20261109");o=r/"results";s=r/"infer-opt";c=Path("/tmp/infer-opt-models/q35moe-verify-opt-20261109")
os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(c/"inductor"),TRITON_CACHE_DIR=str(c/"triton"),HF_HOME=str(c/"hf-home"),HF_HUB_DISABLE_IMPLICIT_TOKEN="1",OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
import time
while not (o/"target.exit.json").exists():time.sleep(2)
k=r/"moe-kernels-index32.py";shutil.copy2(k,o/"moe-kernels-v5.py")
(o/"index32-source.json").write_text(json.dumps({"sha256":hashlib.sha256(k.read_bytes()).hexdigest()}))
cmd=[sys.executable,str(r/"bench-index32.py"),"--kernel","index32","--out",str(o/"index32"),"--repeats","50"]
(o/"index32.command.json").write_text(json.dumps(cmd))
with (o/"index32.stdout").open("w") as so,(o/"index32.stderr").open("w") as se:p=subprocess.run(cmd,stdout=so,stderr=se,timeout=1800)
(o/"index32.exit.json").write_text(json.dumps({"exit_code":p.returncode}))
