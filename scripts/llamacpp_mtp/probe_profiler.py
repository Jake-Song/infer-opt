"""Check interactive CUDA trace collection before the model trace phase."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path('/tmp/llama-mtp-a100-20261111')
out = root / 'results'
probe = root / 'probe'
probe.mkdir(exist_ok=True)
session = 'llama-mtp-probe'
script = probe / 'gpu.py'
script.write_text('''from pathlib import Path
import time
import torch
root=Path("/tmp/llama-mtp-a100-20261111/probe")
x=torch.ones(1024,device="cuda")
torch.cuda.synchronize()
(root/"ready").touch()
while not (root/"start").exists(): time.sleep(0.05)
for _ in range(20): x.add_(1)
torch.cuda.synchronize()
(root/"done").touch()
time.sleep(120)
''')
commands = []
def run(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=120)
    commands.append({'argv': args, 'returncode': result.returncode,
                     'stdout': result.stdout, 'stderr': result.stderr})
    assert result.returncode == 0, commands[-1]
try:
    run(['nsys', 'launch', '--session-new=' + session, '--trace=cuda',
         '--cuda-graph-trace=node', sys.executable, str(script)])
    for _ in range(600):
        if (probe / 'ready').exists():
            break
        time.sleep(0.05)
    assert (probe / 'ready').exists()
    run(['nsys', 'start', '--session=' + session, '--sample=none', '--cpuctxsw=none',
         '--force-overwrite=true', '--export=sqlite', '--output=' + str(probe / 'report')])
    (probe / 'start').touch()
    for _ in range(600):
        if (probe / 'done').exists():
            break
        time.sleep(0.05)
    assert (probe / 'done').exists()
    run(['nsys', 'stop', '--session=' + session])
    run(['nsys', 'stats', '--report=cuda_gpu_kern_sum', '--format=csv', str(probe / 'report.nsys-rep')])
    assert 'elementwise' in commands[-1]['stdout'], commands[-1]
    result = {'passed': True, 'commands': commands}
except BaseException as error:
    result = {'passed': False, 'error': str(error), 'commands': commands}
finally:
    cleanup = subprocess.run(['nsys', 'shutdown', '--session=' + session, '--kill=sigterm'],
                             capture_output=True, text=True, timeout=120)
    result['cleanup'] = {'returncode': cleanup.returncode, 'stdout': cleanup.stdout, 'stderr': cleanup.stderr}
    (out / 'nsys-probe.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
