"""Inspect a Colab A100 before collecting down-projection counters."""
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess

result = {'uid': os.geteuid(), 'tools': {k: shutil.which(k) for k in ['nvidia-smi', 'nvcc', 'ncu']}}
result['ncu_candidates'] = glob.glob('/usr/local/**/ncu', recursive=True) + glob.glob('/opt/**/ncu', recursive=True)
for source, key in [('/proc/driver/nvidia/params', 'driver_params'), ('/proc/self/status', 'capabilities')]:
    try:
        lines = Path(source).read_text().splitlines()
        result[key] = [line for line in lines if ('Profil' in line if key == 'driver_params' else line.startswith(('CapEff:', 'CapPrm:', 'CapBnd:')))]
    except OSError as error:
        result[key] = str(error)
for name, command in [('gpu', ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']), ('nvcc', ['nvcc', '--version'])]:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
        result[name] = {'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    except OSError as error:
        result[name] = str(error)
try:
    import torch
    import triton
    result['python_gpu'] = {'torch': str(torch.__version__), 'triton': triton.__version__, 'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability()}
except Exception as error:
    result['python_gpu'] = repr(error)
root = Path('/tmp/expert-issue-a100-20261111')
root.mkdir(exist_ok=True)
lock = subprocess.run(['nvidia-smi', '--lock-gpu-clocks=1095,1095'], capture_output=True, text=True, timeout=30)
result['clock_lock'] = {'command': ['nvidia-smi', '--lock-gpu-clocks=1095,1095'], 'returncode':lock.returncode, 'stdout':lock.stdout, 'stderr':lock.stderr, 'requested_mhz':1095}
print('CLOCK_LOCK '+json.dumps(result['clock_lock']),flush=True)
(root/'preflight.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
