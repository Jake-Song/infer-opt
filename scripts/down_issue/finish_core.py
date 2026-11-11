"""Finish after the six primary counter comparisons answer the T=4 question."""
import json
import os
from pathlib import Path
import signal
import time

root=Path('/tmp/down-issue-a100-20261111')
profiles=json.loads((root/'focused-profiles.json').read_text())
expected={f'focused-t4-partial-{variant}-{cache}' for variant,cache in [
    ('packed1','hot'),('packed1','cold'),('scalar4','hot'),('scalar4','cold'),('scalar1','hot'),('packed4','hot')]}
done={p['name'] for p in profiles if p['returncode']==0}
if not expected.issubset(done):
    print(json.dumps({'ready':False,'completed':len(done),'required':len(expected)}),flush=True)
else:
    pid=json.loads((root/'focused-launch.json').read_text())['pid']
    cmdline=Path(f'/proc/{pid}/cmdline').read_text()
    assert 'down_issue/focused_worker.py' in cmdline,cmdline
    progress=json.loads((root/'progress.json').read_text())
    os.killpg(pid,signal.SIGTERM)
    time.sleep(2)
    result={'complete':True,'completed_profiles':len(done),'required_profiles':sorted(expected),
            'interrupted_optional_stage':progress,'pid':pid,
            'reason':'Primary T=4 partial-route counter comparisons are complete. Existing unprofiled route/T=1/2 controls were already measured; additional counter sweeps are unnecessary for the issue-saturation decision.'}
    (root/'core-complete.json').write_text(json.dumps(result,indent=2))
    (root/'progress.json').write_text(json.dumps({'stage':'complete-core','time':time.time()}))
    print(json.dumps(result),flush=True)
