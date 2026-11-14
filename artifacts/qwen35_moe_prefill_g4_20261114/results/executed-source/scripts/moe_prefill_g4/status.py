"""Read this run's existing state; never allocate or launch another worker."""
import json
from pathlib import Path

# The most recently launched run on this runtime.
ROOT = max(Path('/tmp').glob('moe-prefill-g4-*/launch.json'), key=lambda p: p.stat().st_mtime).parent
OUT = ROOT/'results'
print('run', ROOT)
for name in ['hardware.json', 'progress.json', 'gates.json', 'exit.json']:
    p = OUT/name
    print(name, p.read_text() if p.exists() else 'pending')
for name in ['tests.stdout', 'tests.stderr', 'numerics.stdout', 'numerics.stderr', 'experts.stderr', 'model.stderr']:
    p = OUT/name
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0, p.stat().st_size - 2000))
            print(name, f.read().decode(errors='replace'))
for name in ['experts/experts.json', 'model/model.json']:
    p = OUT/name
    if p.exists():
        data = json.loads(p.read_text())
        print(name, 'rows', len(data['rows']))
        for row in data['rows'][-4:]:
            print({k: v for k, v in row.items() if k not in ('reference', 'sorted', 'grouped')})
            for key in ('reference', 'sorted', 'grouped'):
                if key in row:
                    print(key, {k: v for k, v in row[key].items() if k not in ('samples', 'generated_ids')})
