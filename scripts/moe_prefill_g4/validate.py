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


def validate_mtp(root, read, files):
    """The mtp plan: every divergence, count and gate verdict recomputed from the retained token IDs."""
    diverge = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    out = {'archive_files_verified': files}
    for folder in ('mtp', 'mtp-parallel'):
        if not (root/folder/'mtp-parity.json').exists():
            continue
        result = read(folder + '/mtp-parity.json')
        rows, gate = result['rows'], result['gate']
        backends = list(dict.fromkeys(r['backend'] for r in rows))
        first = {r['prompt']: r for r in rows if r['backend'] == backends[0]}
        new_tokens = read(folder + '/manifest.json')['args']['new_tokens']
        inputs = read(folder + '/inputs.json')
        failures, table, ids = 0, [], 0
        for row in rows:
            plain = row['plain']
            assert len(inputs[row['prompt']]) == row['prompt_tokens']
            assert len(plain['generated_ids']) == len(plain['top2_margins']) == new_tokens and plain['repeatable']
            assert plain['prefill_ms'] > 0 and plain['decode_ms_per_token'] > 0
            ids += new_tokens * (1 + len(row['mtp']))
            for entry in row['mtp']:
                assert len(entry['generated_ids']) == new_tokens and entry['repeatable']
                assert entry['first_divergence_from_plain'] == diverge(entry['generated_ids'], plain['generated_ids'])
            if row['backend'] == backends[0]:
                continue
            ref = first[row['prompt']]
            at = diverge(plain['generated_ids'], ref['plain']['generated_ids'])
            assert at == plain['first_divergence_from_' + backends[0]]
            if at is not None:
                margins = plain['margins_at_divergence']
                assert margins[backends[0]] == ref['plain']['top2_margins'][at]
                assert margins[row['backend']] == plain['top2_margins'][at]
                failures += margins[backends[0]] > gate['strong_margin']
            for entry, want in zip(row['mtp'], ref['mtp']):
                assert (entry['verification'], entry['k']) == (want['verification'], want['k'])
                assert entry['first_divergence_from_' + backends[0]] == diverge(entry['generated_ids'],
                                                                              want['generated_ids'])
                failures += (want['first_divergence_from_plain'] is None
                             and entry['first_divergence_from_plain'] is not None)
            table.append({'prompt': row['prompt'], 'prompt_tokens': row['prompt_tokens'], 'backend': row['backend'],
                'plain_first_divergence': at,
                'reference_margin_at_divergence': None if at is None else ref['plain']['top2_margins'][at],
                'plain_prefill_ms': [ref['plain']['prefill_ms'], plain['prefill_ms']],
                'plain_decode_ms_per_token': [ref['plain']['decode_ms_per_token'], plain['decode_ms_per_token']]})
        assert failures == len(gate['failures']) and gate['passes'] == (failures == 0)
        assert read(folder + '/complete.json') == {'complete': True, 'gate': gate}
        summary = {b: {'mtp_runs': sum(len(r['mtp']) for r in rows if r['backend'] == b),
                       'mtp_runs_equal_to_plain': sum(e['first_divergence_from_plain'] is None
                                                      for r in rows if r['backend'] == b for e in r['mtp']),
                       'all_repeatable': True} for b in backends}
        assert summary == result['summary']
        out[folder] = {'summary': summary, 'gate_passes': gate['passes'], 'gate_failures': gate['failures'],
                       'generated_ids_checked': ids, 'rows': table}
    return out


def validate_tune(root, read, files):
    """The tune plan: tile ranking, dense-projection totals, and tuned tiles in the model."""
    tune = read('tune/tile-tuning.json')
    rows, cases = tune['rows'], tune['cases']
    base = next(r for r in rows if r['kind'] == 'default')
    for row in rows:  # the ranking score, recomputed from the per-case sums
        score = math.exp(statistics.fmean(math.log(row['case_ms'][k] / base['case_ms'][k]) for k in cases))
        assert math.isclose(score, row['relative_time_geomean'], rel_tol=1e-12)
    assert rows == sorted(rows, key=lambda r: r['relative_time_geomean'])
    best = next(r for r in rows if r['name'] == tune['best'])
    gates = [g['candidate']['rms_relative'] <= g['existing']['rms_relative'] * (1 + g['slack'])
             for g in best['fp64_gate']]
    assert all(gates) and all(g['passes'] for g in best['fp64_gate'])
    # nothing ranked ahead of the selection was left unchecked or passing
    assert all(not all(g['passes'] for g in r['fp64_gate']) for r in rows[:rows.index(best)])
    assert read('tune/best-config.json') == best['config']
    projections = read('projections/projections.json')
    for tokens, totals in projections['per_prefill_ms'].items():
        group = [r for r in projections['rows'] if r['tokens'] == int(tokens)]
        for key in ('dequantize_only_ms', 'dequant_linear_ms', 'cached_linear_ms'):
            assert math.isclose(totals[key], sum(r[key] * r['layers'] for r in group), rel_tol=1e-12)
        assert all(r['best_fused']['ms'] == min(f['ms'] for f in r['fused']) for r in group)
        assert math.isclose(totals['best_fused_ms'], sum(r['best_fused']['ms'] * r['layers'] for r in group),
                            rel_tol=1e-12)
    assert read('model/complete.json') == {'complete': True, 'grouped_gate_failures': []}
    inputs, routes = read('model/inputs.json'), read('model/routes.json')
    model_rows = read('model/model.json')['rows']
    assert {(r['prompt'], r['tokens']) for r in model_rows} == set(itertools.product(
        ['math', 'korean'], [128, 512, 2048, 4096]))
    backends, samples, out = ('reference', 'grouped', 'grouped-tuned'), 0, []
    for row in model_rows:
        key = f"{row['prompt']}-{row['tokens']}"
        assert len(inputs[key]) == row['tokens'] and row['grouped_tuned_config'] == best['config']
        assert all(len(c) == 256 and sum(c) == row['tokens'] * 8 for c in routes[key].values())
        for backend in backends:
            timing = row[backend]
            assert len(timing['samples']) == 5 and len(timing['generated_ids']) == 16
            for kind in ('wall', 'cuda'):
                values = [s[f'{kind}_ms'] for s in timing['samples']]
                assert all(math.isfinite(v) and v > 0 for v in values)
                assert statistics.median(values) == timing[f'median_{kind}_ms']
            samples += 5
            t, host = timing['sections_ms'], timing['sections_host_ms']
            parts = sum(v for k, v in t.items() if k not in ('text_model', 'wall', 'lm_head'))
            assert math.isclose(parts, t['text_model'], rel_tol=.02) and t['wall'] >= t['text_model']
            assert all(0 <= host[k] <= t[k] * 1.05 + .05 for k in host)  # medians taken separately
        for backend in backends[1:]:
            g = row[backend]
            same = g['generated_ids'] == row['reference']['generated_ids']
            gate = g['model_gate']
            assert same == g['greedy_parity'] == (g['first_divergence'] is None)
            assert gate['kl'] == g['kl_reference_to_candidate'] and math.isfinite(gate['kl'])
            assert gate['passes'] == (gate['kl'] <= gate['kl_limit'] and same)
            assert g['timing_ratio'] == row['reference']['median_wall_ms'] / g['median_wall_ms']
        out.append({'prompt': row['prompt'], 'tokens': row['tokens'],
            'reference_ms': row['reference']['median_wall_ms'], 'grouped_ms': row['grouped']['median_wall_ms'],
            'tuned_ms': row['grouped-tuned']['median_wall_ms'],
            'tuned_over_grouped': row['grouped']['median_wall_ms'] / row['grouped-tuned']['median_wall_ms'],
            'grouped_kl': row['grouped']['kl_reference_to_candidate'],
            'tuned_kl': row['grouped-tuned']['kl_reference_to_candidate'],
            'gates': [row[b]['model_gate']['passes'] for b in backends[1:]]})
    for name, digest in read('model/manifest.json')['source_sha256'].items():
        assert hashlib.sha256((root/'executed-source'/name).read_bytes()).hexdigest() == digest, name
    return {'archive_files_verified': files, 'tile_configurations': len(rows), 'best_tile': best['name'],
            'best_relative_expert_time': best['relative_time_geomean'], 'best_fp64_gates': len(gates),
            'projection_rows': len(projections['rows']), 'model_cases': len(out), 'model_timing_samples': samples,
            'continuation_ids_checked': len(out) * 3 * 16,
            'model_gates_passed': sum(sum(r['gates']) for r in out), 'model_rows': out}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True)
    p.add_argument('--archive', help='Read retained raw traces from the recovered archive when not extracted; '
                   'without it raw traces and the remote pytest cache are skipped and counted, everything '
                   'else is still checked')
    args = p.parse_args()
    root = Path(args.results)
    read = lambda name: json.loads((root/name).read_text())
    archive = read('archive-manifest.json')
    packed = tarfile.open(args.archive) if args.archive else None
    skipped = cache = 0
    for name, metadata in archive['files'].items():
        path = root/name
        if path.exists():
            assert path.stat().st_size == metadata['bytes'], name
            stream = path.open('rb')
        elif packed is None and name.endswith('.trace.json'):
            skipped += 1  # raw traces live only in the archive, which is not kept in the repository
            continue
        elif packed is None and '.pytest_cache/' in name:
            cache += 1  # the remote test run's cache: archived, self-ignored, never committed
            continue
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
    plan = read('plan.json')['plan'] if (root/'plan.json').exists() else 'gate'
    verified = len(archive['files']) - skipped - cache
    unverified = {key: count for key, count in (('raw_traces_not_verified', skipped),
                                                ('pytest_cache_files_not_verified', cache)) if count}
    if plan in ('tune', 'mtp'):
        check = validate_tune if plan == 'tune' else validate_mtp
        print(json.dumps({**check(root, read, verified), **unverified}, indent=2))
        return
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
    summary = {'archive_files_verified': verified, **unverified}
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
