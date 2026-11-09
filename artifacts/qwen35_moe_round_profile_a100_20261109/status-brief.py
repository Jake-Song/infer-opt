from pathlib import Path
import json
r=Path("/content/q35moe-round-profile-20261109/results")
for n in ["progress.json","exit.json","diagnostic/progress.json","calibrated-progress.json","calibrated-exit.json","calibrated/progress.json"]:
 p=r/n
 if p.exists(): print(n,p.read_text())
for n in ["diagnostic/summary.json","calibrated/summary.json"]:
 p=r/n
 if p.exists():
  for row in json.loads(p.read_text())["rows"]: print(n,{k:v for k,v in row.items() if "samples" not in k})
for n in ["diagnostic.stderr","calibrated.stderr"]:
 p=r/n
 if p.exists(): print(n,p.read_text()[-700:])
