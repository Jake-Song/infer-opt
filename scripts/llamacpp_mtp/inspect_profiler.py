import subprocess
from pathlib import Path
root=Path("/tmp/llama-mtp-a100-20261111/results")
for command, tag in [("launch","session"),("launch","trace"),("start","trace"),("start","session"),("shutdown","session")]:
 args=["nsys",command,"--help="+tag]
 r=subprocess.run(args,capture_output=True,text=True,timeout=20)
 (root/("nsys-"+command+"-"+tag+".txt")).write_text(r.stdout+r.stderr)
 print(args,r.returncode,r.stdout[:10000],r.stderr[:500])
