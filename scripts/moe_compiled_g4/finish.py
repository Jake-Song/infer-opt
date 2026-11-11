"""Finalize weights-free evidence after the dedicated worker exits."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

root = Path('/tmp/moe-compiled-g4-20261112')
out = root / 'results'
assert (out / 'exit.json').exists(), 'Worker still running'
for name in ['worker.stdout', 'worker.stderr', 'preflight.json', 'launch.json']:
    p = root / name
    if p.exists():
        (out / name).write_bytes(p.read_bytes())
files = {}
for p in sorted(out.rglob('*')):
    if p.is_file() and not p.name.endswith('.tmp') and p.name != 'files-manifest.json':
        files[str(p.relative_to(out))] = {'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
(out / 'files-manifest.json').write_text(json.dumps(files, indent=2))
archive = root / 'final-results.tar.gz'
with tarfile.open(archive.with_suffix('.tmp'), 'w:gz') as tar:
    for p in sorted(out.rglob('*')):
        if p.is_file() and not p.name.endswith('.tmp'):
            data = p.read_bytes()
            entry = tarfile.TarInfo(str(p.relative_to(out)))
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
archive.with_suffix('.tmp').replace(archive)
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': len(files)}, indent=2))
