"""Independently validate recovered native MTP requests and print the table."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics


def analyze(root):
    def read(name):
        return json.loads((root / name).read_text())
    assert read('exit.json')['returncode'] == 0
    manifest = read('model-manifest.json')
    assert manifest['sha256'] == manifest['expected_sha256']
    assert manifest['bytes'] == 37801097376
    source = read('harness-manifest.json')
    for name, expected in source.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    assert read('clock-lock.json')['returncode'] == 0
    assert read('environment.json')['warp_override'] is False
    gguf = read('gguf-audit.json')
    assert any('nextn' in name and int(value) > 0 for name, value in gguf['fields'].items())
    inputs = read('inputs.json')
    records = read('timed-requests.json')
    assert len(records) == 40
    groups = collections.defaultdict(list)
    for row in records:
        prompt, k = row['prompt'], row['k']
        assert prompt in inputs and k in [0, 1, 2, 3]
        response, request = row['response'], row['request']
        timing = response['timings']
        assert request['prompt'] == inputs[prompt]['input_ids']
        assert len(response['tokens']) == response['tokens_predicted'] == timing['predicted_n'] == 256
        assert request['cache_prompt'] is False and timing['cache_n'] == 0
        assert request['temperature'] == 0 and request['ignore_eos'] is True
        per_token = timing['predicted_ms'] / 255
        assert math.isclose(row['ms_token'], per_token, rel_tol=1e-10)
        assert math.isclose(timing['predicted_per_second'], 1000 / per_token, rel_tol=1e-8)
        if k:
            assert 0 <= timing['draft_n_accepted'] <= timing['draft_n']
            assert timing['draft_n'] > 0 and row['rounds'] > 0
            assert math.isclose(row['acceptance'], timing['draft_n_accepted'] / timing['draft_n'])
        groups[prompt, k].append(row)
    assert len(groups) == 8
    summary = []
    for prompt in inputs:
        reference = groups[prompt, 0][0]['response']['tokens']
        off_ms = statistics.median(x['ms_token'] for x in groups[prompt, 0])
        for k in [0, 1, 2, 3]:
            rows = groups[prompt, k]
            assert len(rows) == 5
            assert collections.Counter(r['batch'] for r in rows) == {0: 3, 1: 2}
            differences = [next((i for i, (a, b) in enumerate(zip(reference, row['response']['tokens'])) if a != b), None)
                           for row in rows]
            med = statistics.median(x['ms_token'] for x in rows)
            summary.append({'prompt': prompt, 'k': k, 'ms_token': med, 'tokens_s': 1000 / med,
                            'timing_ratio_vs_off': off_ms / med,
                            'exact_runs': sum(x is None for x in differences), 'first_differences': differences,
                            'acceptance': statistics.median(x['acceptance'] for x in rows) if k else None,
                            'samples': 5,
                            'batch_medians_ms_token': {str(b): statistics.median(x['ms_token'] for x in rows if x['batch'] == b)
                                                       for b in [0, 1]}})
    original = {(x['prompt'], x['k']): x for x in read('summary.json')['rows']}
    for row in summary:
        other = original[row['prompt'], row['k']]
        for key in ['ms_token', 'tokens_s', 'timing_ratio_vs_off', 'acceptance', 'exact_runs', 'first_differences']:
            assert row[key] == other[key], (row, key)
    for batch in [0, 1]:
        for k in [0, 1, 2, 3]:
            label = f'batch{batch}-k{k}'
            args = read(label + '-server.command.json')
            assert args[args.index('--spec-type') + 1] == ('draft-mtp' if k else 'none')
            if k:
                assert args[args.index('--spec-draft-n-max') + 1] == str(k)
            assert args[args.index('-ngl') + 1] == '99'
            for prompt in inputs:
                assert read(label + '-' + prompt + '-tokenize.json')['tokens'] == inputs[prompt]['input_ids']
                for repeat in range(3):
                    warmup = read(f'{label}-{prompt}-warmup{repeat}.json')
                    assert len(warmup['response']['tokens']) == 256
            telemetry = (root / (label + '-telemetry.stdout')).read_text()
            assert '1095 MHz' in telemetry, telemetry
    validation = {'passed': True, 'timed_requests': 40, 'generated_tokens': 10240,
                  'warmup_requests': 48, 'conditions': 8, 'rows': summary}
    (root / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    print('| 입력 | 설정 | ms/token | token/s | off 대비 시간비 | 수락률 | 출력 일치 |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for row in summary:
        acc = f"{100 * row['acceptance']:.1f}%" if row['acceptance'] is not None else '-'
        config = 'off' if row['k'] == 0 else f"k={row['k']}"
        print(f"| {row['prompt']} | {config} | {row['ms_token']:.4f} | {row['tokens_s']:.2f} | "
              f"{row['timing_ratio_vs_off']:.3f} | {acc} | {row['exact_runs']}/5 |")
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    analyze(parser.parse_args().results)
