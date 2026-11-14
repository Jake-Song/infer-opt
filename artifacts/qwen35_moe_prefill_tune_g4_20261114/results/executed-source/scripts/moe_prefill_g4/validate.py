"""Independently check recovered hashes, timing aggregation, sorted-path parity and grouped gates.

Covers the 2026-11-13 layout (reference and sorted) and later runs that add the grouped
backend, the FP64 numerics stage and the route/section breakdown.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True)
    p.add_argument('--archive', help='Read retained raw traces from the recovered archive when not extracted')
    args = p.parse_args()
    root = Path(args.results)
    read = lambda name: json.loads((root/name).read_text())
    archive = read('archive-manifest.json')
    packed = tarfile.open(args.archive) if args.archive else None
    for name, metadata in archive['files'].items():
        path = root/name
        if path.exists():
            assert path.stat().st_size == metadata['bytes'], name
            stream = path.open('rb')
        else:
            assert packed is not None, name
            assert packed.getmember(name).size == metadata['bytes'], name
            stream = packed.extractfile(name)
        digest = hashlib.sha256()
        with stream:
            for chunk in iter(lambda: stream.read(8*2**20), b''):
                digest.update(chunk)
        assert digest.hexdigest() == metadata['sha256'], name
    if packed:
        packed.close()
    assert read('exit.json')['exit_code'] == 0
    assert read('experts/complete.json')['complete']
    assert read('model/complete.json')['complete']
    count = 0
    def check_timing(row):
        nonlocal count
        samples = row['samples']
        assert len(samples) == 5
        for kind in ('wall', 'cuda'):
            values = [s[f'{kind}_ms'] for s in samples]
            assert all(math.isfinite(value) and value > 0 for value in values)
            assert statistics.median(values) == row[f'median_{kind}_ms']
        count += len(samples)
    def fp64_gate(gate):
        passes = gate['candidate']['rms_relative'] <= gate['existing']['rms_relative'] * (1 + gate['slack'])
        assert passes == gate['passes']
        return passes
    grouped = (root/'numerics/numerics.json').exists()
    summary = {'archive_files_verified': len(archive['files'])}
    if grouped:
        numerics = read('numerics/numerics.json')
        gates = [fp64_gate(g) for row in numerics['rows'] for g in row['fp64_gate'].values()]
        assert numerics['all_pass'] == all(gates)
        for row in numerics['rows']:  # the kernels the grouped path replaces one for one
            assert row['swiglu']['grouped_vs_torch']['equal_fraction'] == 1
            assert row['combine']['grouped_vs_loop']['equal_fraction'] == 1
        summary['numerics'] = {'cases': len(numerics['rows']), 'fp64_gates': len(gates), 'passed': sum(gates)}
    expert_rows = read('experts/experts.json')['rows']
    sorted_rows = [row for row in expert_rows if row['backend'] == 'sorted']
    grouped_rows = [row for row in expert_rows if row['backend'] == 'grouped']
    assert len(sorted_rows) == 9 and len(sorted_rows) + len(grouped_rows) == len(expert_rows)
    assert bool(grouped_rows) == grouped
    for row in sorted_rows:
        assert row['numerics']['exact']
        assert row['numerics']['max_abs'] == 0
        for backend in ('reference', 'sorted'):
            check_timing(row[backend])
        assert row['timing_ratio'] == row['reference']['median_wall_ms'] / row['sorted']['median_wall_ms']
    timed = 0
    for row in grouped_rows:
        assert ('grouped' in row) == fp64_gate(row['fp64_gate'])  # rejected candidates carry no timing
        if 'grouped' in row:
            check_timing(row['grouped'])
            assert row['timing_ratio'] == row['reference']['median_wall_ms'] / row['grouped']['median_wall_ms']
            timed += 1
    expert_samples = count
    inputs = read('model/inputs.json')
    model_rows = read('model/model.json')['rows']
    assert {(r['prompt'], r['tokens']) for r in model_rows} == set(itertools.product(
        ['math', 'korean'], [128, 512, 2048, 4096]))
    assert len(model_rows) == 8
    backends = ['reference', 'sorted'] + (['grouped'] if 'grouped' in model_rows[0] else [])
    routes = read('model/routes.json') if grouped else {}
    rows = []
    for row in model_rows:
        key = f"{row['prompt']}-{row['tokens']}"
        assert len(inputs[key]) == row['tokens']
        for backend in backends:
            check_timing(row[backend])
            assert len(row[backend]['generated_ids']) == 16
        candidate = row['sorted']
        assert candidate['numerics']['exact'] and candidate['numerics']['max_abs'] == 0
        assert candidate['kl_reference_to_candidate'] == 0
        assert candidate['first_divergence'] is None and candidate['greedy_parity']
        assert candidate['generated_ids'] == row['reference']['generated_ids']
        ratio = row['reference']['median_wall_ms'] / candidate['median_wall_ms']
        assert ratio == candidate['timing_ratio']
        out = {'prompt': row['prompt'], 'tokens': row['tokens'],
            'reference_ms': row['reference']['median_wall_ms'],
            'sorted_ms': candidate['median_wall_ms'], 'speedup': ratio}
        if 'grouped' in backends:
            g = row['grouped']
            same = g['generated_ids'] == row['reference']['generated_ids']
            assert same == g['greedy_parity'] == (g['first_divergence'] is None)
            gate = g['model_gate']
            assert gate['kl'] == g['kl_reference_to_candidate'] and math.isfinite(gate['kl'])
            assert gate['passes'] == (gate['kl'] <= gate['kl_limit'] and same)
            assert g['timing_ratio'] == row['reference']['median_wall_ms'] / g['median_wall_ms']
            out.update(grouped_ms=g['median_wall_ms'], grouped_speedup=g['timing_ratio'],
                       grouped_over_sorted=candidate['median_wall_ms'] / g['median_wall_ms'],
                       grouped_kl=gate['kl'], grouped_gate=gate['passes'])
        if grouped:
            assert len(routes[key]) == 40  # every routed row lands on exactly one expert
            assert all(len(c) == 256 and sum(c) == row['tokens'] * 8 for c in routes[key].values())
            assert [layer['used_experts'] for layer in row['routes']['layers']] == [
                sum(1 for v in c if v) for c in routes[key].values()]
            for backend in backends:
                t = row[backend]['sections_ms']
                parts = sum(v for k, v in t.items() if k not in ('text_model', 'wall', 'lm_head'))
                # medians over passes are taken per section, so they add up only approximately
                assert math.isclose(parts, t['text_model'], rel_tol=.02) and t['wall'] >= t['text_model']
        rows.append(out)
    for folder in ('experts', 'model'):
        for name, digest in read(folder+'/manifest.json')['source_sha256'].items():
            path = root/'executed-source'/name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    summary.update({'expert_cases': len(sorted_rows),
        'expert_timing_samples': expert_samples, 'model_cases': len(rows),
        'model_timing_samples': count-expert_samples, 'continuation_ids_checked': len(rows)*len(backends)*16,
        'all_last_logits_exact': True, 'all_continuations_equal': True, 'model_rows': rows})
    if grouped:
        del summary['all_last_logits_exact']  # exact for sorted only; grouped is gated by KL
        summary.update(sorted_last_logits_exact=True, grouped_expert_configs=len(grouped_rows),
            grouped_expert_configs_timed=timed, route_cases=len(routes),
            grouped_model_gates_passed=sum(r['grouped_gate'] for r in rows) if 'grouped' in backends else None,
            all_continuations_equal=all(r.get('grouped_gate', True) for r in rows))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
