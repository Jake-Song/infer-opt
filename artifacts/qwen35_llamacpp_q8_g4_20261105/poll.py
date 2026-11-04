import json
import time
from pathlib import Path
root = Path('/content/q35moe-llama-q8')
print('UTC', time.strftime('%H:%M:%S', time.gmtime()))
for name in ['environment.json', 'progress.json', 'exit.json']:
    path = root / name
    if path.exists():
        print(name, path.read_text())
path = root / 'summary.json'
if path.exists():
    print('completed rows:', json.dumps(json.loads(path.read_text())['rows']))
progress = root / 'progress.json'
if progress.exists():
    name = json.loads(progress.read_text())['stage']
    log = root / (name + '.stderr')
    if log.exists():
        print('current stderr tail:', log.read_text()[-1600:])
path = root / 'worker.log'
if path.exists():
    print('worker.log tail:', path.read_text()[-1800:])
path = root / 'build.stdout'
if path.exists():
    print('build tail:', path.read_text()[-700:])
path = root / 'download.log'
if path.exists():
    print('download tail:', path.read_text()[-500:])
cache = Path('/root/.cache/huggingface/hub')
for path in cache.glob('models--ggml-org--Qwen3.5-35B-A3B-GGUF/blobs/*'):
    print('cached file', path.name[:12], round(path.stat().st_size / 2**30, 3), 'GiB')
