"""Verify the requested G4 device before downloading the checkpoint."""
import json
from pathlib import Path
import shutil
import sys
import torch

root = Path('/tmp/moe-compiled-g4-20261112')
root.mkdir(exist_ok=True)
gpu = torch.cuda.get_device_properties(0)
info = {'gpu': gpu.name, 'vram_gib': gpu.total_memory / 2**30,
        'capability': torch.cuda.get_device_capability(), 'torch': str(torch.__version__),
        'python': sys.version, 'disk_free_gib': shutil.disk_usage('/tmp').free / 2**30}
(root / 'preflight.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
assert 'RTX PRO 6000' in gpu.name and gpu.total_memory > 90 * 2**30, info
assert info['capability'] == (12, 0), info
