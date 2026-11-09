from pathlib import Path
import json,time
r=Path("/content/q35moe-verify-opt-20261109/results")
for n in ["download-progress.json","download-error.json","target-packed-controller-progress.json","target-packed/progress.json","target-packed.exit.json","launch-tuning.exit.json"]:
 p=r/n
 if p.exists():print(n,p.read_text())
p=r/"target-packed/summary.json"
if p.exists():
 d=json.loads(p.read_text());print("rows",len(d["rows"]));
 for x in d["rows"]:print({k:v for k,v in x.items() if k!="samples_ms"})
p=r/"target-packed.stderr"
if p.exists():print("stderr",p.read_text()[-1400:])
p=Path("/content/q35moe-verify-opt-20261109/download.log")
if p.exists():print("download",p.read_text()[-500:])

import subprocess
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu,power.draw,clocks.sm","--format=csv"],capture_output=True,text=True).stdout)
ps=subprocess.run(["ps","-eo","pid,etime,pcpu,stat,comm"],capture_output=True,text=True).stdout
print("\n".join(x for x in ps.splitlines() if "python" in x))
