from pathlib import Path
root=Path('/content/q35moe-llama-hf-quality/results')
for name in ('environment.json','progress.json','exit.json','aggregate.json'):
    path=root/name
    if path.exists(): print(name,path.read_text())
for name in ('build.stdout','hf-reference.stdout','comparison.stdout'):
    path=root/name
    if path.exists(): print(name,path.read_text()[-800:])
log=Path('/content/q35moe-llama-hf-quality-worker.log')
if log.exists(): print(log.read_text()[-2200:])
import time
p=root/'progress.json'
if p.exists(): print('seconds_since_stage_update',round(time.time()-p.stat().st_mtime,1))
cache=Path('/tmp/infer-opt-models/q35moe-llama-hf-quality')
blobs=[p for p in cache.rglob('*') if p.is_file() and not p.is_symlink() and ('blobs' in p.parts)]
print('cache_blob_files',len(blobs),'logical_gib',round(sum(p.stat().st_size for p in blobs)/2**30,3))
for p in sorted(blobs,key=lambda p:p.stat().st_size,reverse=True)[:3]: print(p.name,round(p.stat().st_size/2**30,3))
