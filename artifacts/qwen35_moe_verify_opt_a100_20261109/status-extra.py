from pathlib import Path
import json
r=Path("/content/q35moe-verify-opt-20261109/results")
for s in ["index32","ordered","pairs"]:
 for f in [s+".exit.json",s+"/validation.json"]:
  p=r/f
  if p.exists():print(f,p.read_text()[-1000:])
 p=r/s/"tuning.json"
 if p.exists():
  rows=json.loads(p.read_text())["rows"];print(s,"rows",len(rows))
  for t in [2,4]:
   for route in ["shared","partial","disjoint"]:
    print(t,route,[(x["config"],round(x.get("speedup",1),3),x.get("exact")) for x in rows if x["tokens"]==t and x["routing"]==route])
 p=r/(s+".stderr")
 if p.exists():print(p.read_text()[-1000:])
