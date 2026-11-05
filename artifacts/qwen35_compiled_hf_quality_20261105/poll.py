from pathlib import Path
import time
root=Path('/content/q35moe-compiled-hf-quality/results')
for name in ('environment.json','progress.json','exit.json','aggregate.json','load.json','setup.json'):
    path=root/name
    if path.exists(): print(name,path.read_text())
for name in ('hf-reference.stdout','quality.stdout','quality.stderr'):
    path=root/name
    if path.exists(): print(name,path.read_text()[-1700:])
log=Path('/content/q35moe-compiled-hf-quality-worker.log')
if log.exists(): print(log.read_text()[-1600:])
p=root/'progress.json'
if p.exists(): print('seconds_since_stage_update',round(time.time()-p.stat().st_mtime,1))
cache=Path('/tmp/infer-opt-models/q35moe-compiled-hf-quality')
blobs=[p for p in cache.rglob('*') if p.is_file() and not p.is_symlink() and 'blobs' in p.parts]
print('download_blob_logical_gib',round(sum(p.stat().st_size for p in blobs)/2**30,3))
