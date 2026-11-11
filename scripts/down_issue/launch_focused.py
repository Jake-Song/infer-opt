"""After broad hot/cold triage, stop its remaining sweep and start focused probes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

out=Path('/tmp/down-issue-a100-20261111')
profiles=json.loads((out/'profiles.json').read_text())
assert len(profiles)>=2 and all(p['returncode']==0 for p in profiles[:2]),'Broad hot/cold triage not finished'
pid=json.loads((out/'launch.json').read_text())['pid']
cmdline=Path(f'/proc/{pid}/cmdline').read_text()
assert 'down_issue/worker.py' in cmdline,cmdline
interruption={'pid':pid,'completed_profiles':len(profiles),'progress':json.loads((out/'progress.json').read_text()),
              'reason':'Broad hot/cold triage complete. Collect a focused metric set for the remaining comparisons; add explicit longer GPU warmup.'}
os.killpg(pid,signal.SIGTERM)
time.sleep(2)
(out/'triage-interruption.json').write_text(json.dumps(interruption,indent=2))
root=Path('/tmp/down-issue-source')
with tarfile.open('/tmp/down-issue-source-v3.tar.gz') as archive:archive.extractall(root,filter='data')
with (out/'focused-worker.stdout').open('w') as stdout,(out/'focused-worker.stderr').open('w') as stderr:
    proc=subprocess.Popen([sys.executable,str(root/'scripts/down_issue/focused_worker.py')],stdout=stdout,stderr=stderr,start_new_session=True)
(out/'focused-launch.json').write_text(json.dumps({'pid':proc.pid,'python':sys.executable}))
print(json.dumps({'pid':proc.pid,'triage_completed_profiles':len(profiles)}),flush=True)
