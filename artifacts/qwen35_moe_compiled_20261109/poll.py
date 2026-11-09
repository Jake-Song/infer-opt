from pathlib import Path
import json,time
root=Path('/content/q35moe-compiled-20261109/results')
for name in ['environment.json','progress.json','exit.json','quality/aggregate.json','benchmark/progress.json','benchmark/cuda-graph-evidence.json']:
 p=root/name
 if p.exists():print(name,p.read_text())
p=root/'benchmark/summary.json'
if p.exists():
 d=json.loads(p.read_text())
 print('benchmark_rows',[{k:r.get(k) for k in ['method','prompt','median_ms','divergent_runs','valid_speedup','first_divergence']} for r in d['rows']])
progress=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else {}
logs=['benchmark.stdout','benchmark.stderr'] if progress.get('stage')=='compiled_parallel_mtp_k123_benchmark' else ['hf-reference.stdout','hf-reference.stderr','quality.stdout','quality.stderr']
if progress.get('stage')=='dependencies':logs=['install-torch.stdout','install-torch.stderr','install.stderr']
for name in logs:
 p=root/name
 if p.exists():print(name,p.read_text()[-1200:])
cache=Path('/tmp/infer-opt-models/q35moe-compiled-20261109')
blobs=[p for p in cache.rglob('*') if p.is_file() and not p.is_symlink() and 'blobs' in p.parts]
print('model_cache_gib',round(sum(p.stat().st_size for p in blobs)/2**30,3))
