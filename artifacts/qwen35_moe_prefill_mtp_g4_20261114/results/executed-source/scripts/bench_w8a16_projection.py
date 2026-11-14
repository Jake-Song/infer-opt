"""Dense W8A16 prefill projections: dequantize + matmul against the fused group-scale GEMM.

The eager prefill path rebuilds a BF16 weight from INT8 and scales on every call
and then calls ``F.linear``. This measures that, the matmul alone on a cached
BF16 weight (what keeping a BF16 copy would cost in memory instead), and
``infer_opt.moe_prefill.w8a16_gemm``, which reads INT8 directly. Errors are RMS
relative to an FP64 product with weights exactly ``q * scale``.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer_opt.moe_prefill import w8a16_gemm
from infer_opt.quant import dequantize_int8

# Qwen3.5-35B-A3B dense projections that read one input each prefill: (name, out, in, layers)
SHAPES = [('gated_delta_qkvz', 12288, 2048, 30), ('gated_delta_out', 2048, 4096, 30),
          ('attention_qkv', 9216, 2048, 10), ('attention_out', 2048, 4096, 10),
          ('shared_gate_up', 1024, 2048, 40), ('shared_down', 2048, 512, 40)]


def time_ms(fn, repeats):
    fn()
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
    p.add_argument('--out', required=True)
    p.add_argument('--lengths', default='128,512,2048,4096')
    p.add_argument('--shapes', help='comma-separated shape names; default all')
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--group', type=int, default=128)
    p.add_argument('--block-m', default='16,32,64')
    p.add_argument('--block-n', default='64,128')
    p.add_argument('--block-k', default='64,128')
    p.add_argument('--num-warps', default='4,8')
    args = p.parse_args()
    values = lambda text: [int(v) for v in text.split(',')]
    tiles = list(itertools.product(values(args.block_m), values(args.block_n), values(args.block_k),
                                   values(args.num_warps)))
    lengths = values(args.lengths)
    wanted = args.shapes.split(',') if args.shapes else [s[0] for s in SHAPES]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(901)
    result = {'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__), 'group': args.group,
              'repeats': args.repeats, 'rows': []}
    for label, n, k, layers in SHAPES:
        if label not in wanted:
            continue
        weight = torch.randint(-127, 128, (n, k), device='cuda', dtype=torch.int8)
        scale = (torch.rand(n, k // args.group, device='cuda') * .0003 + .0001).bfloat16()
        cached = dequantize_int8(weight, scale, torch.bfloat16)
        exact = weight.double() * scale.double().repeat_interleave(args.group, -1)
        dequant_ms = time_ms(lambda: dequantize_int8(weight, scale, torch.bfloat16), args.repeats)
        for t in lengths:
            x = torch.randn(t, k, device='cuda', dtype=torch.bfloat16)
            truth = x.double() @ exact.T
            norm = truth.square().mean().sqrt()
            error = lambda got: ((got.double() - truth).square().mean().sqrt() / norm).item()
            row = {'shape': label, 'out': n, 'in': k, 'layers': layers, 'tokens': t,
                   'dequantize_only_ms': dequant_ms,
                   'dequant_linear_ms': time_ms(lambda: F.linear(x, dequantize_int8(weight, scale, x.dtype)), args.repeats),
                   'cached_linear_ms': time_ms(lambda: F.linear(x, cached), args.repeats),
                   'dequant_linear_error': error(F.linear(x, dequantize_int8(weight, scale, x.dtype))),
                   'fused': []}
            for bm, bn, bk, warps in tiles:
                if args.group % bk:
                    continue
                run = lambda: w8a16_gemm(x, weight, scale, block_m=bm, block_n=bn, block_k=bk, num_warps=warps)
                row['fused'].append({'tile': [bm, bn, bk, warps], 'ms': time_ms(run, args.repeats),
                                     'error': error(run())})
            best = min(row['fused'], key=lambda f: f['ms'])
            row['best_fused'] = best
            result['rows'].append(row)
            out.write_text(json.dumps(result, indent=2) + '\n')
            print(label, t, 'dequant+linear %.3f cached %.3f fused %.3f %s | error %.5f / %.5f' % (
                row['dequant_linear_ms'], row['cached_linear_ms'], best['ms'], best['tile'],
                row['dequant_linear_error'], best['error']), flush=True)
            del x, truth
        del weight, scale, cached, exact
    # Per-prefill totals over all layers of each shape, per length.
    result['per_prefill_ms'] = {str(t): {key: sum(r[key] * r['layers'] for r in result['rows'] if r['tokens'] == t)
        for key in ('dequantize_only_ms', 'dequant_linear_ms', 'cached_linear_ms')} |
        {'best_fused_ms': sum(r['best_fused']['ms'] * r['layers'] for r in result['rows'] if r['tokens'] == t)}
        for t in lengths}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['per_prefill_ms'], indent=1))


if __name__ == '__main__':
    main()
