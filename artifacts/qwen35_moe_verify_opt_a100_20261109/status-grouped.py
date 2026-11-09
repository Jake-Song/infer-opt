import json
from pathlib import Path
r=Path("/content/q35moe-verify-opt-20261109/results")
for n in ["grouped.exit.json","grouped/complete.json","grouped/validation.json"]:
 p=r/n
 if p.exists():print(n,p.read_text()[-1400:])
p=r/"grouped/tuning.json"
if p.exists():
 d=json.loads(p.read_text());print("rows",len(d["rows"]));
 for t in [2,4]:
  for route in ["shared","partial","disjoint"]:
   print(t,route,[(x["config"],round(x.get("speedup",1),3),x.get("max_abs_diff")) for x in d["rows"] if x["tokens"]==t and x["routing"]==route])
p=r/"grouped.stderr"
if p.exists():print(p.read_text()[-1600:])
