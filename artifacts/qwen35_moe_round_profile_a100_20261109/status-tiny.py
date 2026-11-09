from pathlib import Path
import json
r=Path("/content/q35moe-round-profile-20261109/results")
for n in ["calibrated-progress.json","calibrated-exit.json","calibrated/progress.json"]:
 p=r/n
 if p.exists(): print(n,p.read_text())
p=r/"calibrated/summary.json"
if p.exists():
 for x in json.loads(p.read_text())["rows"]:
  print({k:x[k] for k in ["k","prompt","frozen_production_host_ms","full_manual_graph_gpu_ms","full_manual_graph_host_ms","full_manual_graph_outputs_vs_production","outputs_vs_production","segmented_over_full_manual_gpu"]},x["segmented_phase_medians_ms"])

import subprocess,time
print("utc",time.strftime("%H:%M:%S",time.gmtime()))
print(subprocess.run(["ps","-p","11947","-o","pid,etime,pcpu,pmem,stat"],capture_output=True,text=True).stdout)
p=r/"calibrated.stderr"
if p.exists(): print(p.read_text()[-500:])
