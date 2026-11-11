"""Independently verify the recovered G4 compiled parallel experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def analyze(root):
    def read(name):
        return json.loads((root / name).read_text())
    assert read('exit.json')['exit_code'] == 0
    assert read('benchmark.exit.json')['returncode'] == 0
    assert read('benchmark/progress.json')['stage'] == 'complete'
    env = read('environment.json')
    assert 'RTX PRO 6000' in env['gpu']['gpu']
    assert env['gpu']['vram_gib'] > 90
    assert env['packages']['torch'] == '2.11.0+cu130'
    assert env['packages']['triton'] == '3.6.0'
    assert env['packages']['fla-core'] == '0.6.0'
    assert read('contract.json')['verification'] == 'parallel'
    assert read('contract.json')['warp_override'] is False
    source = read('source-manifest.json')
    for name, expected in read('worker-manifest.json').items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    checkpoint = read('checkpoint-hashes.json')
    assert sum(x['file'].endswith('.safetensors') for x in checkpoint) == 22
    assert all(x['hub_lfs_verified'] for x in checkpoint if x['file'].endswith('.safetensors'))
    methods = ['plain', 'mtp1', 'mtp2', 'mtp3']
    prompts = ['chat_long', 'code_1k', 'prose_1k', 'math', 'korean']
    expected_inputs = read('expected-inputs.json')
    inputs = read('benchmark/inputs.json')
    for prompt in prompts:
        assert inputs[prompt]['input_ids'] == expected_inputs[prompt]['input_ids']
    summary = read('benchmark/summary.json')
    assert summary['manifest']['verification'] == 'parallel'
    assert summary['manifest']['max_autotune'] is True
    assert summary['manifest']['coordinate_descent_tuning'] is True
    assert summary['manifest']['compile'] == {'mode': 'reduce-overhead', 'fullgraph': True, 'dynamic': False}
    assert summary['manifest']['cache_capacity'] == 2048
    rows = summary['rows']
    assert len(rows) == 20
    assert {(r['method'], r['prompt']) for r in rows} == {(m, p) for m in methods for p in prompts}
    baselines = {p: read(f'benchmark/plain-{p}.json')['baseline_ids'] for p in prompts}
    for row in rows:
        method, prompt = row['method'], row['prompt']
        record = read(f'benchmark/{method}-{prompt}.json')
        samples = record['samples']
        assert len(samples) == 5
        assert record['baseline_ids'] == baselines[prompt]
        differences, agreement = [], []
        for sample in samples:
            ids = sample['generated_ids']
            assert len(ids) == len(baselines[prompt]) == 256
            same = [a == b for a, b in zip(ids, baselines[prompt])]
            first = next((i for i, equal in enumerate(same) if not equal), None)
            assert sample['first_divergence'] == first
            assert sample['token_agreement'] == sum(same) / 256
            assert math.isclose(sample['ms_per_token'], sample['decode_ms'] / (sample['emitted_tokens'] - 1), abs_tol=1e-12)
            if method != 'plain':
                histogram = sample['accepted_histogram']
                assert sum(histogram.values()) == sample['rounds']
                emitted = sum(int(n) * count for n, count in histogram.items())
                assert emitted + 1 == sample['emitted_tokens']
                assert sample['tokens_per_round'] == emitted / sample['rounds']
            differences.append(first)
            agreement.append(sum(same) / 256)
        assert row['first_divergence'] == differences
        assert row['token_agreement'] == agreement
        assert row['divergent_runs'] == sum(x is not None for x in differences)
        assert row['median_ms'] == statistics.median(s['ms_per_token'] for s in samples)
        if method != 'plain':
            base = next(r for r in rows if r['method'] == 'plain' and r['prompt'] == prompt)
            assert math.isclose(row['timing_ratio'], base['median_ms'] / row['median_ms'], abs_tol=1e-12)
            assert row['valid_speedup'] == (row['timing_ratio'] if row['divergent_runs'] == 0 else None)
    evidence = read('benchmark/cuda-graph-evidence.json')
    assert all(evidence[method]['cuda_graph_launch_observed'] for method in methods)
    record = {'passed': True, 'timed_samples': 100, 'generated_ids_checked': 25600,
              'rows': 20, 'parallel_verification': True, 'matching_input_ids': True,
              'worker_hash_checked': True, 'source_files': len(source['files']),
              'cuda_graph_observed_all_modes': True,
              'exact_runs': {method: sum(r['repeats'] - r['divergent_runs'] for r in rows if r['method'] == method)
                             for method in methods}}
    (root / 'validation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    for prompt in prompts:
        group = {r['method']: r for r in rows if r['prompt'] == prompt}
        print(prompt, ' '.join(f"{m}={group[m]['median_ms']:.4f}" for m in methods),
              'parity', [5 - group[m]['divergent_runs'] for m in methods[1:]])
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    analyze(parser.parse_args().results)
