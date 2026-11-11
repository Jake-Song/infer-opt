import json,os,subprocess
from pathlib import Path
r=Path("/tmp/llama-mtp-a100-20261111")
o=r/"results"
for name in ["profile-progress.json","profile-exit.json"]:
 p=o/name
 print(name,p.read_text() if p.exists() else "not present")
for name in ["profile-worker.stdout","profile-worker.stderr"]:
 p=r/name
 print(name,p.read_text()[-4000:] if p.exists() else "not present")
print("recent files",[(p.name,p.stat().st_size) for p in sorted(o.glob("trace-*.exit.json"),key=lambda p:p.stat().st_mtime)[-8:]])
print(subprocess.run(["ps","-eo","pid,ppid,comm,etime"],capture_output=True,text=True).stdout[-6000:])
