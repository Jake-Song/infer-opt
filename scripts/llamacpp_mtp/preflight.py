"""Inspect the dedicated runtime before downloading Qwen weights."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

out = Path('/tmp/llama-mtp-a100-20261111')
out.mkdir(exist_ok=True)
p = torch.cuda.get_device_properties(0)
info = {'gpu': p.name, 'vram_gib': p.total_memory / 2**30,
        'python': sys.version, 'torch': str(torch.__version__),
        'disk_free_gib': shutil.disk_usage('/tmp').free / 2**30,
        'tools': {x: shutil.which(x) for x in ['nvcc', 'cmake', 'nsys', 'ncu']}}
(out / 'preflight.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
assert 'A100' in p.name and p.total_memory >= 75 * 2**30, info
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,clocks.sm,power.limit',
                      '--format=csv,noheader'], capture_output=True, text=True).stdout)
