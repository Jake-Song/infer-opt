import os,signal,time,json,tarfile
from pathlib import Path
r=Path("/content/q35moe-verify-opt-20261109/results")
pid=11304
result={"reason":"User requested wrap-up here; leave proper packed/persistent grouped GEMM for the next experiment.","owned_controller_pid":pid}
if not (r/"final-exit.json").exists():
 p=Path(f"/proc/{pid}/cmdline")
 if p.exists() and b"run-final.py" in p.read_bytes():
  os.killpg(pid,signal.SIGTERM);time.sleep(2);result["final_controller_stopped"]=True
else:result["final_exit"]=json.loads((r/"final-exit.json").read_text())
(r/"wrap-status.json").write_text(json.dumps(result,indent=2))
p=Path("/content/q35moe-verify-opt-20261109-all-results.tar.gz");t=p.with_suffix(".tmp")
with tarfile.open(t,"w:gz") as a:
 for f in sorted(r.rglob("*")):
  if f.is_file() and f.suffix!=".tmp":a.add(f,arcname=str(f.relative_to(r)))
t.replace(p)
print(result,{"archive_bytes":p.stat().st_size})
