"""Inspect the existing G4 worker; never allocates a new runtime."""
import json
from pathlib import Path
import subprocess

root = Path('/tmp/moe-compiled-g4-20261112')
out = root / 'results'
for name in ['progress.json', 'exit.json', 'benchmark/progress.json']:
    p = out / name
    print(name, p.read_text() if p.exists() else 'not present')
for p in [root / 'worker.stderr', out / 'benchmark.stdout', out / 'benchmark.stderr',
          out / 'install-torch.stdout', out / 'install.stderr']:
    if p.exists():
        with p.open('rb') as stream:
            stream.seek(max(0, p.stat().st_size - 3000))
            print(p.name, stream.read().decode(errors='replace'))
for name in ['summary.json', 'cuda-graph-evidence.json']:
    p = out / 'benchmark' / name
    if p.exists():
        data = json.loads(p.read_text())
        print(name, json.dumps(data.get('rows', data), ensure_ascii=False)[-10000:])
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu,clocks.sm',
                      '--format=csv,noheader'], capture_output=True, text=True).stdout)
