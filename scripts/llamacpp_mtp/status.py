"""Show progress and write a recoverable snapshot on the dedicated runtime."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

root = Path('/tmp/llama-mtp-a100-20261111')
out = root / 'results'
for name in ['progress.json', 'exit.json', 'profile-progress.json', 'profile-exit.json']:
    p = out / name
    if p.exists():
        print(name, p.read_text()[-15000:])
for name in ['worker.stdout', 'worker.stderr', 'profile-worker.stdout', 'profile-worker.stderr']:
    p = root / name
    if p.exists():
        print(name, p.read_text()[-1200:])
for name in ['build.stdout', 'build.stderr']:
    p = out / name
    if p.exists():
        print(name, p.read_text()[-900:])
cache=Path('/tmp/infer-opt-models/llama-mtp-20261111')
for p in cache.rglob('*'):
    if p.is_file() and p.stat().st_size > 2**30:
        print('model_file', p.name, 'bytes', p.stat().st_size)
print(subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,clocks.sm',
                      '--format=csv,noheader'], capture_output=True, text=True).stdout)
tmp = root / 'snapshot.tar.gz.tmp'
with tarfile.open(tmp, 'w:gz') as tar:
    for p in sorted(out.rglob('*')):
        if p.is_file() and not p.name.endswith('.tmp'):
            data = p.read_bytes()
            import io
            info = tarfile.TarInfo(str(p.relative_to(out)))
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    for name in ['worker.stdout', 'worker.stderr', 'preflight.json', 'launch.json']:
        p = root / name
        if p.exists():
            tar.add(p, arcname=name)
tmp.replace(root / 'snapshot.tar.gz')
p = root / 'snapshot.tar.gz'
print(json.dumps({'archive': str(p), 'bytes': p.stat().st_size,
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}))
