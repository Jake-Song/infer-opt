"""Extract the uploaded source bundle and start a persistent Colab worker."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import shutil

root=Path('/tmp/down-issue-source')
root.mkdir(exist_ok=True)
with tarfile.open('/tmp/down-issue-source.tar.gz') as archive:
    archive.extractall(root,filter='data')
out=Path('/tmp/down-issue-a100-20261111')
out.mkdir(exist_ok=True)
if (out/'worker-error.json').exists():
    attempt=out/'attempt-1'
    attempt.mkdir(exist_ok=True)
    for name in ['worker-error.json','worker.stdout','worker.stderr','timing.stdout','timing.stderr','timing.command.json','timing','progress.json','launch.json']:
        path=out/name
        if path.exists(): shutil.move(str(path),str(attempt/name))
with (out/'worker.stdout').open('w') as stdout,(out/'worker.stderr').open('w') as stderr:
    proc=subprocess.Popen([sys.executable,str(root/'scripts/down_issue/worker.py')],stdout=stdout,stderr=stderr,start_new_session=True)
(out/'launch.json').write_text(json.dumps({'pid':proc.pid,'python':sys.executable}))
print(json.dumps({'pid':proc.pid,'python':sys.executable}),flush=True)
