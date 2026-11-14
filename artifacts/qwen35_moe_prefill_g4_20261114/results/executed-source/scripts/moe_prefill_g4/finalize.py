"""Package completed results and actual executed sources, including raw traces."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

root = max(Path('/tmp').glob('moe-prefill-g4-*/launch.json'), key=lambda p: p.stat().st_mtime).parent
out = root/'results'
assert json.loads((out/'exit.json').read_text())['exit_code'] == 0
assert json.loads((out/'model/complete.json').read_text())['complete']
assert json.loads((out/'experts/complete.json').read_text())['complete']
assert json.loads((out/'numerics/numerics.json').read_text())['rows']
for path in (root/'source').rglob('*'):
    if path.is_file() and '__pycache__' not in path.parts:
        dest = out/'executed-source'/path.relative_to(root/'source')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
for path in root.glob('launch*.json'):
    dest = out/'lifecycle'/path.name
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(path, dest)
for name in ['worker.stdout', 'worker.stderr', 'diagnose.py']:
    path = root/name
    if path.exists():
        shutil.copy2(path, out/'lifecycle'/path.name)
files = {}
for path in sorted(out.rglob('*')):
    if path.is_file() and path.name != 'archive-manifest.json':
        files[str(path.relative_to(out))] = {'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
(out/'archive-manifest.json').write_text(json.dumps({'files': files}, indent=2))
archive = root/'recovered-results.tar.gz'
with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
    for path in sorted(out.rglob('*')):
        if path.is_file():
            tar.add(path, arcname=str(path.relative_to(out)))
print(json.dumps({'run': root.name, 'gates': json.loads((out/'gates.json').read_text()),
    'archive': str(archive), 'bytes': archive.stat().st_size,
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': len(files)}))
