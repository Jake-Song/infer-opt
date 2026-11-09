from pathlib import Path
import json,time
r=Path("/content/q35moe-verify-opt-20261109/results")
for n in ["initial-runtime.json","progress.json","exit.json","environment.stdout","micro/complete.json","tests.stdout"]:
 p=r/n
 if p.exists(): print(n,p.read_text()[-1500:])
p=r/"micro/tuning.json"
if p.exists():
 d=json.loads(p.read_text()); print("tuning_rows",len(d["rows"])); print("last",{k:v for k,v in d["rows"][-1].items() if k!="samples_ms"})
for n in ["install-torch.stderr","install-tools.stderr","micro.stderr","tests.stderr"]:
 p=r/n
 if p.exists():print(n,p.read_text()[-800:])

for n in ["install-torch.stdout","install-tools.stdout"]:
 p=r/n
 if p.exists():print(n,p.read_text()[-500:])

for n in ["tiled.exit.json","tiled/complete.json"]:
 p=r/n
 if p.exists():print(n,p.read_text())
p=r/"tiled/tuning.json"
if p.exists():
 d=json.loads(p.read_text());print("tiled rows",len(d["rows"]));print({k:v for k,v in d["rows"][-1].items() if k!="samples_ms"})
p=r/"tiled.stderr"
if p.exists():print(p.read_text()[-1200:])
