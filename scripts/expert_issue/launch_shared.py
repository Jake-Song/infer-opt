"""Start a waiting follow-up process; it uses the GPU only after the main sweep."""
import json
from pathlib import Path
import subprocess
import sys

out=Path('/tmp/expert-issue-a100-20261111')
assert not (out/'shared-launch.json').exists(),'Duplicate follow-up'
with (out/'shared-worker.stdout').open('w') as stdout,(out/'shared-worker.stderr').open('w') as stderr:
    p=subprocess.Popen([sys.executable,'/tmp/expert-issue-shared-followup.py'],stdout=stdout,stderr=stderr,start_new_session=True)
(out/'shared-launch.json').write_text(json.dumps({'pid':p.pid}))
print(json.dumps({'pid':p.pid,'waits_for_main_sweep':True}),flush=True)
