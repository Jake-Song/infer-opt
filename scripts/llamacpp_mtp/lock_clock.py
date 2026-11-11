import json, subprocess
from pathlib import Path
r=subprocess.run(["nvidia-smi","-lgc","1095,1095"],capture_output=True,text=True)
d={"requested_sm_mhz":1095,"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
Path("/tmp/llama-mtp-a100-20261111/results/clock-lock.json").write_text(json.dumps(d,indent=2))
print(json.dumps(d))
assert r.returncode==0,d
