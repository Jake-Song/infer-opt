"""Unpack an uploaded experiment and start its dedicated Colab worker."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile

source=Path('/tmp/expert-issue-source');source.mkdir(exist_ok=True)
with tarfile.open('/tmp/expert-issue-source.tar.gz') as tar:tar.extractall(source,filter='data')
out=Path('/tmp/expert-issue-a100-20261111');out.mkdir(exist_ok=True)
assert not (out/'launch.json').exists(),'Do not start a duplicate worker'
with (out/'worker.stdout').open('w') as stdout,(out/'worker.stderr').open('w') as stderr:
    p=subprocess.Popen([sys.executable,str(source/'scripts/expert_issue/worker.py')],stdout=stdout,stderr=stderr,start_new_session=True)
(out/'launch.json').write_text(json.dumps({'pid':p.pid,'python':sys.executable}))
print(json.dumps({'pid':p.pid}),flush=True)
