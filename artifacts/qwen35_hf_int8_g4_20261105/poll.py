import json
import time
from pathlib import Path
root = Path('/content/q35moe-hf-int8')
print('UTC', time.strftime('%H:%M:%S', time.gmtime()))
for name in ['environment.json', 'progress.json', 'load.json', 'exit.json']:
    path = root / name
    if path.exists():
        print(name, path.read_text())
path = root / 'summary.json'
if path.exists():
    rows = json.loads(path.read_text())['rows']
    print('rows', [{k:v for k,v in row.items() if k != 'samples_ms'} for row in rows])
for name in ['worker.log']:
    path = root / name
    if path.exists():
        print(name, path.read_text()[-900:])
cache = Path('/tmp/infer-opt-models/huggingface/hub/models--Jakemu--Qwen3.5-35B-A3B-int8-g128-mtp/blobs')
if cache.exists():
    files = list(cache.iterdir())
    size = 0
    for path in files:
        try:
            size += path.stat().st_size
        except FileNotFoundError:
            pass
    print('model cache files', len(files), 'GiB', size / 2**30)
