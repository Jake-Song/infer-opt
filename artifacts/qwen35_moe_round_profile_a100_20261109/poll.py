from pathlib import Path
import json,time
root=Path('/content/q35moe-round-profile-20261109/results')
for name in ['environment.json','progress.json','exit.json','production/progress.json','diagnostic/progress.json','calibrated-progress.json','calibrated-exit.json','calibrated/progress.json']:
 p=root/name
 if p.exists():print(name,p.read_text())
for name in ['production/summary.json','diagnostic/summary.json','calibrated/summary.json']:
 p=root/name
 if p.exists():
  d=json.loads(p.read_text())
  for r in d['rows']:
   if 'profile' in r:
    prof=r['profile'];print('production',r['method'],r['prompt'],'ms/token',r['median_ms_token'],'gpu/call',prof['gpu_kernel_sum_ms_per_call'],'category',prof['gpu_category_ms_per_call'],'events',prof['gpu_events'],'graphs',prof['cuda_graph_launches'])
   else:print('diagnostic',{k:v for k,v in r.items() if 'samples' not in k})
for name in ['production.stderr','diagnostic.stderr','calibrated.stderr']:
 p=root/name
 if p.exists():print(name,p.read_text()[-1200:])
cache=Path('/tmp/infer-opt-models/q35moe-round-profile-20261109')
blobs=[p for p in cache.rglob('*') if p.is_file() and not p.is_symlink() and 'blobs' in p.parts]
print('model_cache_gib',round(sum(p.stat().st_size for p in blobs)/2**30,3))
