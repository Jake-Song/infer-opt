from pathlib import Path
import tarfile
r=Path("/content/q35moe-verify-opt-20261109/results")
p=Path("/content/q35moe-verify-opt-20261109-all-results.tar.gz");t=p.with_suffix(".tmp")
with tarfile.open(t,"w:gz") as a:
 for f in sorted(r.rglob("*")):
  if f.is_file() and f.suffix!=".tmp":a.add(f,arcname=str(f.relative_to(r)))
t.replace(p)
print({"archive":str(p),"bytes":p.stat().st_size})
