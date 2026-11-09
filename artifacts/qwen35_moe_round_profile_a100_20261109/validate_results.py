"""Validate saved samples and provenance without loading model weights."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
r = ROOT / 'results'
load = lambda p: json.loads(p.read_text())
checks = {}
assert load(r / 'exit.json')['exit_code'] == 0
assert load(r / 'calibrated-exit.json')['exit_code'] == 0
for manifest, source in [('profile-harness.json', 'profile_qwen3_5_moe_round.py'),
                         ('calibrated-harness.json', 'calibrated-profile-source.py')]:
    assert hashlib.sha256((r / source).read_bytes()).hexdigest() == load(r / manifest)['sha256']
checks['source_hashes_verified'] = True
summary = load(r / 'production/summary.json')['rows']
assert len(summary) == 6
generated = 0
for row in summary:
    path = r / 'production' / f"{row['method']}-{row['prompt']}-timings.json"
    data = load(path)
    assert len(data['samples']) == 5
    for s in data['samples']:
        assert len(s['generated_ids']) == 256
        generated += 256
        assert math.isclose(s['ms_per_token'], s['decode_ms'] / (s['emitted_tokens'] - 1))
        divergence = next((i for i, (a, b) in enumerate(zip(s['generated_ids'], data['baseline_ids'])) if a != b), None)
        assert divergence == s['first_divergence']
        if row['method'] != 'plain':
            assert sum(int(k) * v for k, v in s['accepted_histogram'].items()) == s['emitted_tokens'] - 1
            assert sum(s['accepted_histogram'].values()) == s['rounds']
    assert statistics.median(s['ms_per_token'] for s in data['samples']) == row['median_ms_token']
checks['timed_generations'] = len(summary) * 5
checks['generated_ids_verified'] = generated
checks['trace_files'] = len(list((r / 'production').glob('*-trace.json.gz')))
assert checks['trace_files'] == 6
checks['phase_diagnostics'] = {}
for stage in ['diagnostic', 'calibrated']:
    rows = load(r / stage / 'summary.json')['rows']
    assert len(rows) == 4
    for row in rows:
        assert len(row['segmented_samples_ms']) == 20
        for sample in row['segmented_samples_ms']:
            assert math.isclose(sum(v for k, v in sample.items() if k != 'total_ms'), sample['total_ms'], abs_tol=1e-4)
        if stage == 'calibrated':
            for key in ['production_samples_ms', 'production_host_samples_ms', 'full_manual_gpu_samples_ms', 'full_manual_host_samples_ms']:
                assert len(row[key]) == 20
    micro_samples = 0
    for prompt in ['math', 'korean']:
        rows_micro = load(r / stage / f'experts-{prompt}.json')['rows']
        assert len(rows_micro) == 6
        for x in rows_micro:
            assert len(x['samples_ms']) == 10
            assert len(x['routing_by_layer']) == 40
            micro_samples += 10
    checks['phase_diagnostics'][stage] = {'frozen_points': len(rows), 'phase_replays': len(rows) * 20, 'expert_sweep_replays': micro_samples}
checks['state_parity'] = [{k: row[k] for k in ['k', 'prompt', 'full_manual_graph_outputs_vs_production', 'outputs_vs_production']}
                          for row in load(r / 'calibrated/summary.json')['rows']]
checks['state_parity_note'] = 'State equality is recorded, not assumed. A mismatch does not invalidate the original production traces, but prevents claiming equivalent optimization.'
compiler = load(r / 'compiler-python-manifest.json')
checks['compiler_python_modules'] = len(compiler)
(ROOT / 'validation.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
