"""Audit recovered measurements from raw token IDs, counts, timings and hashes."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def read(name):
    return json.loads((RESULTS / name).read_text())


assert read('exit.json')['exit_code'] == 0
assert read('benchmark.exit.json')['returncode'] == 0
assert read('quality.exit.json')['returncode'] == 0
assert read('benchmark/progress.json')['stage'] == 'complete'
environment = read('environment.json')
assert environment['gpu'] == 'NVIDIA A100-SXM4-80GB'
assert environment['packages']['torch'] == '2.11.0+cu130'
assert environment['packages']['triton'] == '3.6.0'
summary = read('benchmark/summary.json')
rows = summary['rows']
methods = ('plain', 'mtp1', 'mtp2', 'mtp3')
prompts = ('chat_long', 'code_1k', 'prose_1k', 'math', 'korean')
assert {(r['method'], r['prompt']) for r in rows} == {(m, p) for m in methods for p in prompts}
assert len(rows) == 20
assert summary['manifest']['max_autotune'] is True
assert summary['manifest']['coordinate_descent_tuning'] is True
assert summary['manifest']['verification'] == 'parallel'
assert summary['manifest']['compile'] == {'mode': 'reduce-overhead', 'fullgraph': True, 'dynamic': False}
graph = read('benchmark/cuda-graph-evidence.json')
assert all(graph[method]['cuda_graph_launch_observed'] for method in methods)
baselines = {p: read(f'benchmark/plain-{p}.json')['baseline_ids'] for p in prompts}
for row in rows:
    method, prompt = row['method'], row['prompt']
    raw = read(f'benchmark/{method}-{prompt}.json')
    assert raw['baseline_ids'] == baselines[prompt]
    assert len(raw['samples']) == row['repeats'] == 5
    divergences = []
    agreements = []
    for sample in raw['samples']:
        assert len(sample['generated_ids']) == len(raw['baseline_ids']) == 256
        same = [a == b for a, b in zip(sample['generated_ids'], raw['baseline_ids'])]
        first = next((i for i, match in enumerate(same) if not match), None)
        agreement = sum(same) / 256
        assert sample['first_divergence'] == first
        assert sample['token_agreement'] == agreement
        assert math.isclose(sample['ms_per_token'], sample['decode_ms'] / (sample['emitted_tokens'] - 1), abs_tol=1e-12)
        if method != 'plain':
            histogram = sample['accepted_histogram']
            assert sum(histogram.values()) == sample['rounds']
            emitted = sum(int(n) * count for n, count in histogram.items())
            assert emitted + 1 == sample['emitted_tokens']
            assert sample['tokens_per_round'] == emitted / sample['rounds']
        divergences.append(first)
        agreements.append(agreement)
    assert row['first_divergence'] == divergences
    assert row['token_agreement'] == agreements
    assert row['divergent_runs'] == sum(first is not None for first in divergences)
    assert row['median_ms'] == statistics.median(s['ms_per_token'] for s in raw['samples'])
    if method != 'plain':
        baseline = next(r for r in rows if r['method'] == 'plain' and r['prompt'] == prompt)
        assert math.isclose(row['timing_ratio'], baseline['median_ms'] / row['median_ms'], abs_tol=1e-12)
        assert row['valid_speedup'] == (row['timing_ratio'] if row['divergent_runs'] == 0 else None)

quality = read('quality/quality.json')
assert quality['mtp'] is False and quality['max_autotune'] is True
assert read('quality/load.json')['mtp_called'] is False
assert len(quality['rows']) == 18
for prompt in {r['prompt'] for r in quality['rows']}:
    paired = [r for r in quality['rows'] if r['prompt'] == prompt]
    assert len(paired) == 2 and all(r['teacher_forced_tokens'] == 48 for r in paired)
    assert len({r['hf_logits_sha256'] for r in paired}) == 1
reference = read('reference.json')
inputs = read('benchmark/inputs.json')
for case in reference['cases']:
    if case['name'] in prompts:
        assert case['prompt_ids'] == inputs[case['name']]['input_ids']
source_manifest = read('source-manifest.json')
assert source_manifest == json.loads((ROOT / 'source-manifest.json').read_text())
with tarfile.open(ROOT / 'source.tar.gz') as archive:
    for name, expected in source_manifest['files'].items():
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
assert (ROOT / 'worker.py').read_bytes() == (RESULTS / 'worker.py').read_bytes()

aggregates = {}
for method in methods:
    subset = [r for r in rows if r['method'] == method]
    aggregates[method] = {'min_prompt_median_ms': min(r['median_ms'] for r in subset),
        'max_prompt_median_ms': max(r['median_ms'] for r in subset),
        'matching_runs': sum(r['repeats'] - r['divergent_runs'] for r in subset),
        'total_runs': sum(r['repeats'] for r in subset),
        'cuda_graph_launch_observed': graph[method]['cuda_graph_launch_observed']}
record = {'complete': True, 'benchmark_rows': 20, 'samples_verified': 100,
          'generated_token_ids_verified': 25600, 'paired_hf_reference_hashes_verified': True,
          'source_hashes_verified': True, 'quality': read('quality/aggregate.json'),
          'benchmark': aggregates, 'peak_benchmark_allocated_gib': max(r['peak_allocated_gib'] for r in rows)}
(ROOT / 'comparison_summary.json').write_text(json.dumps(record, indent=2))
(ROOT / 'verification.json').write_text(json.dumps({k: v for k, v in record.items()
    if k not in ('quality', 'benchmark')}, indent=2))
print(json.dumps(record, indent=2))
