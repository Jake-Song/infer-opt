"""Reset the dedicated GPU clock and finalize a weights-free result archive."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

root = Path('/tmp/llama-mtp-a100-20261111')
out = root / 'results'
assert (out / 'exit.json').exists(), 'Main worker has not finished'
assert (out / 'profile-exit.json').exists(), 'Profile worker has not finished'
sessions = subprocess.run(['nsys', 'sessions', 'list'], capture_output=True, text=True)
(out / 'nsys-final-sessions.json').write_text(json.dumps({'returncode': sessions.returncode,
    'stdout': sessions.stdout, 'stderr': sessions.stderr}, indent=2))
reset = subprocess.run(['nvidia-smi', '-rgc'], capture_output=True, text=True)
(out / 'clock-reset.json').write_text(json.dumps({'returncode': reset.returncode,
    'stdout': reset.stdout, 'stderr': reset.stderr}, indent=2))
assert reset.returncode == 0, reset.stderr
for name in ['worker.stdout', 'worker.stderr', 'profile-worker.stdout', 'profile-worker.stderr',
             'preflight.json', 'launch.json', 'profile-launch.json']:
    p = root / name
    if p.exists():
        (out / name).write_bytes(p.read_bytes())
files = {}
for p in sorted(out.rglob('*')):
    if p.is_file() and not p.name.endswith('.tmp') and p.name != 'files-manifest.json':
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 2**20), b''):
                digest.update(chunk)
        files[str(p.relative_to(out))] = {'bytes': p.stat().st_size, 'sha256': digest.hexdigest()}
(out / 'files-manifest.json').write_text(json.dumps(files, indent=2))
archive = root / 'final-results.tar.gz'
with tarfile.open(archive.with_suffix('.tmp'), 'w:gz') as tar:
    for p in sorted(out.rglob('*')):
        if p.is_file() and not p.name.endswith('.tmp'):
            tar.add(p, arcname=str(p.relative_to(out)))
archive.with_suffix('.tmp').replace(archive)
digest = hashlib.sha256()
with archive.open('rb') as stream:
    for chunk in iter(lambda: stream.read(4 * 2**20), b''):
        digest.update(chunk)
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                  'sha256': digest.hexdigest(), 'clock_reset': reset.returncode,
                  'files': len(files)}, indent=2))
