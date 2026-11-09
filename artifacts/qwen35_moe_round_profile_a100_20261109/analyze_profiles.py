"""Reaggregate raw traces with explicit routing classification and coverage checks."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / 'scripts/profile_qwen3_5_moe_round.py'
spec = importlib.util.spec_from_file_location('round_profile_analysis', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
production = ROOT / 'results/production'
summary = json.loads((production / 'summary.json').read_text())
rows = []
for row in summary['rows']:
    method, prompt = row['method'], row['prompt']
    with gzip.open(production / f'{method}-{prompt}-trace.json.gz', 'rt') as stream:
        trace = json.load(stream)
    calls = row['profile']['calls']
    profile = module.summarize_trace(trace, calls)
    gpu = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    projects = sum(e['name'] == '_project' for e in gpu)
    combines = sum(e['name'] == '_combine' for e in gpu)
    k = 0 if method == 'plain' else int(method[3:])
    expected_blocks = 40 + k
    assert projects == 2 * expected_blocks * calls, (method, prompt, projects)
    assert combines == expected_blocks * calls, (method, prompt, combines)
    assert profile['cuda_graph_launches'] >= calls
    profile['expert_blocks_per_call'] = expected_blocks
    profile['expert_kernel_coverage_verified'] = True
    profile['gpu_category_percent'] = {key: value / profile['gpu_kernel_sum_ms_per_call'] * 100
                                      for key, value in profile['gpu_category_ms_per_call'].items()}
    row = {**row, 'profile_reclassified': profile}
    if k:
        frames = [e for e in trace['traceEvents'] if e.get('ph') == 'X' and
                  e.get('cat') == 'user_annotation' and re.fullmatch(method + r'_iteration_\d+', e.get('name', ''))]
        activities = [e for e in trace['traceEvents'] if e.get('ph') == 'X' and
                      e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
        envelopes = []
        for frame in frames:
            contained = [e for e in activities if frame['ts'] <= e['ts'] and
                         e['ts'] + e['dur'] <= frame['ts'] + frame['dur'] + 1.]
            first = min(e['ts'] for e in contained)
            last = max(e['ts'] + e['dur'] for e in contained)
            busy = module.union_us([(e['ts'], e['ts'] + e['dur']) for e in contained])
            envelopes.append({'cpu_annotation_wall_ms': frame['dur'] / 1000,
                'before_first_gpu_ms': (first - frame['ts']) / 1000,
                'gpu_busy_union_ms': busy / 1000,
                'gaps_between_gpu_work_ms': (last - first - busy) / 1000,
                'after_last_gpu_ms': (frame['ts'] + frame['dur'] - last) / 1000,
                'gpu_events': len(contained)})
        assert len(envelopes) == calls
        assert sum(e['gpu_events'] for e in envelopes) == len(activities)
        row['profiled_round_envelope'] = {'samples': envelopes,
            'medians_excluding_first': {key: statistics.median(e[key] for e in envelopes[1:])
                                       for key in envelopes[0]},
            'scope': 'CPU user_annotation only; GPU annotation duplicates excluded. Includes profiler perturbation.'}
    host_path = production / f'{method}-{prompt}-host-stream.json'
    if host_path.exists():
        host = json.loads(host_path.read_text())
        row['host_stream_medians'] = {key: statistics.median(s[key] for s in host['samples'])
            for key in ('wall_ms', 'stream_interval_ms', 'cpu_graph_call_ms', 'after_graph_call_wall_ms')}
    rows.append(row)
    print(method, prompt, 'GPU/category', profile['gpu_category_ms_per_call'])
result = {'analysis_source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'classification_note': 'Routing top-k/sort kept separate from cache copies. BF16 matmul includes fused kernels containing mm and pointwise work.',
          'timing_note': 'Kernel durations include profiler perturbation. CPU waits overlap GPU execution and are not additive.',
          'rows': rows}
(ROOT / 'analysis_summary.json').write_text(json.dumps(result, indent=2))
