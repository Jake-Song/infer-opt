"""Tune the route-grouped expert kernel's tiles on recorded per-layer expert loads.

Replays the rows-per-expert counts of real prefills (``routes.json`` from
``bench_moe_prefill.py --breakdown``) on one synthetic real-shape expert bank, so
no model is needed. Each configuration's time is summed over a case's layers.
Which token goes to which expert is synthetic; only the load per expert is real.

The selected configuration minimizes the geometric mean of per-case time relative
to the default tile, among configurations that pass the FP64 gate.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer_opt.moe import Int8Experts
from infer_opt.moe_prefill import float64_gate, float64_int8_experts, grouped_int8_experts

# The reference point of every ratio: the tile the kernel shipped with before tuning.
DEFAULT = {'block_m': 16, 'block_n': 64, 'block_k': 64, 'num_warps': 4}


def ids_from_counts(counts, tokens, top):
    """Routes ``[tokens, top]`` with exactly ``counts[e]`` rows on expert ``e`` and unique experts per token."""
    counts = torch.as_tensor(counts, device='cuda')
    assert int(counts.sum()) == tokens * top and int(counts.max()) <= tokens
    experts = counts.argsort(descending=True, stable=True)
    labels = torch.repeat_interleave(experts, counts[experts])
    # A run of one expert is at most `tokens` long, so positions i mod tokens never repeat it.
    return labels.view(top, tokens).T.contiguous()


def name(config):
    text = '{block_m}x{block_n}x{block_k}w{num_warps}'.format(**config)
    if config.get('large'):
        rows, bm, bn, bk, warps = config['large']
        text += f'+rows>{rows}:{bm}x{bn}x{bk}w{warps}'
    return text


def time_ms(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--routes', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cases', help='comma-separated route keys; default all')
    p.add_argument('--layers', type=int, default=0, help='use only the first N layers of each case (0: all)')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--block-m', default='16,32,64')
    p.add_argument('--block-n', default='64,128')
    p.add_argument('--block-k', default='64,128')
    p.add_argument('--num-warps', default='4,8')
    p.add_argument('--split-rows', default='16,32,64')
    p.add_argument('--shape', default='256,2048,512,128,8', help='experts,hidden,intermediate,group,top_k')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    e, h, m, g, top = (int(v) for v in args.shape.split(','))
    values = lambda text: [int(v) for v in text.split(',')]
    routes = json.loads(Path(args.routes).read_text())
    cases = args.cases.split(',') if args.cases else list(routes)
    torch.manual_seed(901)
    gu = torch.randint(-127, 128, (e, 2*m, h), device='cuda', dtype=torch.int8)
    down = torch.randint(-127, 128, (e, h, m), device='cuda', dtype=torch.int8)
    gs = (torch.rand(e, 2*m, h//g, device='cuda') * .0003 + .0001).bfloat16()
    ds = (torch.rand(e, h, m//g, device='cuda') * .0003 + .0001).bfloat16()
    module = Int8Experts(gu, gs, down, ds, g)
    work = {}
    for key in cases:
        tokens = int(key.rsplit('-', 1)[1])
        layers = list(routes[key].values())[:args.layers or None]
        work[key] = (torch.randn(tokens, h, device='cuda', dtype=torch.bfloat16),
                     torch.softmax(torch.randn(tokens, top, device='cuda'), -1).bfloat16(),
                     [ids_from_counts(counts, tokens, top) for counts in layers])

    def run(config):
        kwargs = {**config, 'large': tuple(config['large'])} if config.get('large') else config
        per_case = {}
        for key, (x, weights, layers) in work.items():
            per_case[key] = sum(time_ms(lambda: grouped_int8_experts(x, ids, weights, gu, gs, down, ds, **kwargs),
                                        args.repeats) for ids in layers)
        return per_case

    rows = []
    def record(config, kind):
        row = {'name': name(config), 'kind': kind, 'config': config, 'case_ms': run(config)}
        rows.append(row)
        print(row['name'], {k: round(v, 2) for k, v in row['case_ms'].items()}, flush=True)
        return row

    baseline = record(DEFAULT, 'default')
    def score(row):  # geometric mean of time relative to the default tile
        return math.exp(statistics.fmean(math.log(row['case_ms'][k] / baseline['case_ms'][k]) for k in cases))
    singles = [baseline]
    for bm, bn, bk, warps in itertools.product(values(args.block_m), values(args.block_n),
                                               values(args.block_k), values(args.num_warps)):
        config = {'block_m': bm, 'block_n': bn, 'block_k': bk, 'num_warps': warps}
        if g % bk == 0 and config != DEFAULT:
            singles.append(record(config, 'single'))
    # Two-class candidates: the best narrow tiles for few rows, the best wide tiles for many.
    small_keys = [k for k in cases if int(k.rsplit('-', 1)[1]) <= 512] or cases
    large_keys = [k for k in cases if int(k.rsplit('-', 1)[1]) > 512] or cases
    total = lambda row, keys: sum(row['case_ms'][k] / baseline['case_ms'][k] for k in keys)
    small = sorted((r for r in singles if r['config']['block_m'] == min(values(args.block_m))),
                   key=lambda r: total(r, small_keys))[:2]
    large = sorted((r for r in singles if r['config']['block_m'] > min(values(args.block_m))),
                   key=lambda r: total(r, large_keys))[:2]
    for a, b, rows_threshold in itertools.product(small, large, values(args.split_rows)):
        c = b['config']
        record({**a['config'], 'large': [rows_threshold, c['block_m'], c['block_n'], c['block_k'], c['num_warps']]},
               'split')
    for row in rows:
        row['relative_time_geomean'] = score(row)
    ranked = sorted(rows, key=lambda r: r['relative_time_geomean'])

    # FP64 gate of the ranked candidates on the first and the middle layer of every case.
    def gate(config):
        kwargs = {**config, 'large': tuple(config['large'])} if config.get('large') else config
        results = []
        for key, (x, weights, layers) in work.items():
            for index in sorted({0, len(layers) // 2}):
                ids = layers[index]
                truth = float64_int8_experts(x, ids, weights, gu, gs, down, ds)
                verdict = float64_gate(grouped_int8_experts(x, ids, weights, gu, gs, down, ds, **kwargs),
                                       module.forward_reference(x, ids, weights), truth)
                results.append({'case': key, 'layer_index': index, **verdict})
        return results
    best = None
    for row in ranked:
        row['fp64_gate'] = gate(row['config'])
        if all(r['passes'] for r in row['fp64_gate']):
            best = row
            break
    baseline['fp64_gate'] = baseline.get('fp64_gate') or gate(DEFAULT)
    result = {'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__), 'shape': args.shape,
              'cases': cases, 'layers_per_case': {k: len(v[2]) for k, v in work.items()},
              'repeats': args.repeats, 'objective': 'geometric mean over cases of time relative to the default tile',
              'default': baseline['name'], 'best': best and best['name'], 'rows': ranked}
    (out / 'tile-tuning.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'best-config.json').write_text(json.dumps(best['config'] if best else DEFAULT, indent=2) + '\n')
    print('best', best and best['name'], best and round(best['relative_time_geomean'], 4))
    if best is None:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
