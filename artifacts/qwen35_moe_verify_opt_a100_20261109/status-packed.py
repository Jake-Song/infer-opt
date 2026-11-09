from pathlib import Path
import json
r=Path("/content/q35moe-verify-opt-20261109/results")
for n in ["packed.exit.json","packed/validation.json"]:
 p=r/n
 if p.exists():print(n,p.read_text()[-1200:])
p=r/"packed/tuning.json"
if p.exists():
 rows=json.loads(p.read_text())["rows"];print("rows",len(rows))
 for t in [2,4]:
  for route in ["shared","partial","disjoint"]:
   print(t,route,[(x["config"],round(x.get("speedup",1),3),x.get("exact")) for x in rows if x["tokens"]==t and x["routing"]==route])
p=r/"packed.stderr"
if p.exists():print(p.read_text()[-1800:])
