"""Stage-wise FP64-referenced numerics of the MoE prefill expert backends.

Every BF16 path rounds somewhere, so paths are ranked by distance to an unrounded
FP64 evaluation of the same INT8 banks (weights exactly ``q * scale``), not by
distance to each other. Each stage is also isolated: all implementations receive
the same BF16 input and are compared with the FP64 value of that stage alone.

Paths: ``production`` is the adopted eager loop (small-row GEMV, otherwise
dequantize + ``F.linear``); ``dequant`` always dequantizes, which rounds
``q * scale`` to BF16; ``grouped`` is the route-grouped Triton candidate, which
applies scales to FP32 partial sums. ``fp64_gate`` applies the adoption criterion
of ``infer_opt.moe_prefill.float64_gate`` to every tile. No timing is recorded.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer_opt.moe import Int8Experts
from infer_opt.moe_prefill import (_combine, _swiglu, float64_gate, float64_int8_experts,
                                   grouped_int8_experts)
from infer_opt.quant import dequantize_int8, int8_linear


def ulp(value):
    """BF16 spacing (8 significant bits) at each FP64 magnitude."""
    return torch.exp2(torch.floor(torch.log2(value.abs().clamp_min(1e-300))) - 7)


def score(got, truth):
    delta = got.double() - truth
    units = delta / ulp(truth)
    return {'rms_relative': (delta.square().mean().sqrt() / truth.square().mean().sqrt()).item(),
            'max_abs': delta.abs().max().item(),
            # A correctly rounded BF16 result is within half a unit of the FP64 value.
            'within_half_ulp': (units.abs() <= .5).double().mean().item(),
            'within_one_ulp': (units.abs() <= 1).double().mean().item(),
            'max_ulp': units.abs().max().item(),
            # Over results within one unit: round-to-nearest gives about 0, truncation about -0.5.
            'mean_ulp_away_from_zero': (units * truth.sign())[units.abs() <= 1].mean().item()}


def pair(a, b):
    delta = a.double() - b.double()
    return {'equal_fraction': (a == b).double().mean().item(), 'max_abs': delta.abs().max().item(),
            'rms_relative': (delta.square().mean().sqrt() / b.double().square().mean().sqrt()).item()}


def expert_rows(ids, experts):
    """(expert, start, count) for every used expert, in expert-sorted row order."""
    counts = torch.bincount(ids.reshape(-1).long(), minlength=experts).tolist()
    start = 0
    for expert, count in enumerate(counts):
        if count:
            yield expert, start, count
        start += count


def loop_linear(x_rows, weight, scale, linear, ids, experts):
    """Per-expert projection of expert-sorted rows with the eager kernels."""
    out = torch.empty((x_rows.shape[0], weight.shape[1]), device=x_rows.device, dtype=x_rows.dtype)
    for expert, start, count in expert_rows(ids, experts):
        out[start:start + count] = linear(x_rows[start:start + count], weight[expert], scale[expert])
    return out


def loop_float64(x_rows, weight, scale, ids, experts, rounded_weights):
    out = torch.empty((x_rows.shape[0], weight.shape[1]), device=x_rows.device, dtype=torch.float64)
    group = weight.shape[-1] // scale.shape[-1]
    for expert, start, count in expert_rows(ids, experts):
        w = (dequantize_int8(weight[expert], scale[expert], x_rows.dtype).double() if rounded_weights
             else weight[expert].double() * scale[expert].double().repeat_interleave(group, -1))
        out[start:start + count] = x_rows[start:start + count].double() @ w.T
    return out


def by_tile(got, truth, ids, experts, block_m):
    """Grouped-kernel error by row-tile index inside an expert, and for partial tiles."""
    counts = torch.bincount(ids.reshape(-1).long(), minlength=experts)
    starts = counts.cumsum(0) - counts
    expert = torch.repeat_interleave(torch.arange(experts, device=got.device), counts)
    position = torch.arange(got.shape[0], device=got.device) - starts[expert]
    tile = position // block_m
    partial = tile == (counts[expert] - 1) // block_m
    partial &= counts[expert] % block_m != 0
    delta = (got.double() - truth).square().sum(-1)
    power = truth.square().sum(-1)
    rel = lambda mask: (delta[mask].sum() / power[mask].sum()).sqrt().item() if mask.any() else None
    out = {'tile_0': rel(tile == 0), 'tile_1': rel(tile == 1), 'tile_2': rel(tile == 2),
           'tile_3': rel(tile == 3), 'tile_4_plus': rel(tile >= 4),
           'partial_tile': rel(partial), 'full_tile': rel(~partial)}
    return {k: v for k, v in out.items() if v is not None}


def dequant_linear(x, weight, scale):
    return F.linear(x, dequantize_int8(weight, scale, x.dtype))


def diagnose(x, ids, weights, gu, gs, down, ds, tiles):
    e, n, h = gu.shape
    mid, top, group = n // 2, ids.shape[1], h // gs.shape[-1]
    dtype = x.dtype
    module = Int8Experts(gu, gs, down, ds, group)
    truth_stages = {}
    truth = float64_int8_experts(x, ids, weights, gu, gs, down, ds, stages=truth_stages)
    order = truth_stages['order']
    counts = torch.bincount(ids.reshape(-1).long(), minlength=e)
    used = counts[counts > 0]
    report = {'routes': {'used_experts': used.numel(), 'rows_min': used.min().item(),
                         'rows_median': used.float().median().item(), 'rows_max': used.max().item(),
                         'experts_with_1_to_8_rows': (used <= 8).sum().item()}}

    # End to end: each backend against the unrounded FP64 value.
    production = module.forward_reference(x, ids, weights)
    dequant = torch.zeros_like(x)
    for expert, start, count in expert_rows(ids, e):
        selected = order[start:start + count]
        tokens, slots = selected // top, selected % top
        gate, up = dequant_linear(x[tokens], gu[expert], gs[expert]).chunk(2, -1)
        y = dequant_linear(F.silu(gate) * up, down[expert], ds[expert])
        dequant.index_add_(0, tokens, (y * weights[tokens, slots, None]).to(dtype))
    end = {'production': score(production, truth), 'dequant': score(dequant, truth)}
    pairs = {'production_vs_dequant': pair(production, dequant)}
    grouped_stages, outputs = {}, {}
    for tile in tiles:
        name = 'grouped_' + 'x'.join(map(str, tile))
        stages = {} if tile == tiles[0] else None
        got = grouped_int8_experts(x, ids, weights, gu, gs, down, ds, block_m=tile[0],
                                   block_n=tile[1], block_k=tile[2], stages=stages)
        end[name] = score(got, truth)
        outputs[name] = got
        pairs[name + '_vs_production'] = pair(got, production)
        pairs[name + '_vs_dequant'] = pair(got, dequant)
        if stages is not None:
            grouped_stages = stages
    report['end_to_end_vs_fp64'] = end
    report['end_to_end_pairs'] = pairs
    report['fp64_gate'] = {name: float64_gate(got, production, truth) for name, got in outputs.items()}

    # Stage 1: gate/up projection of the same BF16 activations.
    x_rows = x[order // top].contiguous()
    exact = truth_stages['gate_up']
    rounded = loop_float64(x_rows, gu, gs, ids, e, rounded_weights=True)
    g1 = grouped_stages['gate_up']
    d1 = loop_linear(x_rows, gu, gs, dequant_linear, ids, e)
    p1 = loop_linear(x_rows, gu, gs, int8_linear, ids, e)
    report['gate_up'] = {
        'vs_fp64_exact_weights': {'grouped': score(g1, exact), 'dequant': score(d1, exact),
                                  'production': score(p1, exact)},
        'vs_fp64_bf16_rounded_weights': {'grouped': score(g1, rounded), 'dequant': score(d1, rounded),
                                         'production': score(p1, rounded)},
        'weight_rounding_alone': score(rounded.to(dtype), exact),
        'grouped_by_tile_vs_exact_weights': by_tile(g1, exact, ids, e, tiles[0][0]),
        'grouped_vs_dequant': pair(g1, d1), 'grouped_vs_production': pair(g1, p1),
        'production_vs_dequant': pair(p1, d1)}

    # Stage 2: SwiGLU of the grouped kernel's own BF16 gate/up output.
    gate, up = g1.chunk(2, -1)
    truth2 = F.silu(gate.double()) * up.double()
    torch2 = F.silu(gate) * up
    triton2 = torch.empty_like(torch2)
    size = triton2.numel()
    _swiglu[((size + 255) // 256,)](g1, triton2, size, mid, 256, enable_fp_fusion=False)
    report['swiglu'] = {'vs_fp64': {'grouped': score(triton2, truth2), 'torch': score(torch2, truth2)},
                        'grouped_vs_torch': pair(triton2, torch2)}

    # Stage 3: down projection of the same BF16 activations.
    a_rows = grouped_stages['activated']
    exact3 = loop_float64(a_rows, down, ds, ids, e, rounded_weights=False)
    rounded3 = loop_float64(a_rows, down, ds, ids, e, rounded_weights=True)
    g3 = grouped_stages['routed'][order]  # expert-sorted layout
    d3 = loop_linear(a_rows, down, ds, dequant_linear, ids, e)
    p3 = loop_linear(a_rows, down, ds, int8_linear, ids, e)
    report['down'] = {
        'vs_fp64_exact_weights': {'grouped': score(g3, exact3), 'dequant': score(d3, exact3),
                                  'production': score(p3, exact3)},
        'vs_fp64_bf16_rounded_weights': {'grouped': score(g3, rounded3), 'dequant': score(d3, rounded3),
                                         'production': score(p3, rounded3)},
        'weight_rounding_alone': score(rounded3.to(dtype), exact3),
        'grouped_by_tile_vs_exact_weights': by_tile(g3, exact3, ids, e, tiles[0][0]),
        'grouped_vs_dequant': pair(g3, d3), 'grouped_vs_production': pair(g3, p3),
        'production_vs_dequant': pair(p3, d3)}

    # Stage 4: weighted accumulation of the same BF16 expert outputs.
    routed = grouped_stages['routed']
    truth4 = (routed.double().view(-1, top, h) * weights.double()[..., None]).sum(1)
    triton4 = torch.empty_like(x)
    _combine[(x.shape[0], (h + 255) // 256)](routed, ids, weights, triton4, h, top,
        1 << (top - 1).bit_length(), 256, enable_fp_fusion=False)
    loop4 = torch.zeros_like(x)
    for expert, start, count in expert_rows(ids, e):
        selected = order[start:start + count]
        tokens, slots = selected // top, selected % top
        loop4.index_add_(0, tokens, (routed[selected] * weights[tokens, slots, None]).to(dtype))
    report['combine'] = {'vs_fp64': {'grouped': score(triton4, truth4), 'loop': score(loop4, truth4)},
                         'grouped_vs_loop': pair(triton4, loop4)}
    return report


def synthetic_bank(args):
    e, h, m, g = args.experts, args.hidden, args.intermediate, args.group
    gu = torch.randint(-127, 128, (e, 2*m, h), device='cuda', dtype=torch.int8)
    down = torch.randint(-127, 128, (e, h, m), device='cuda', dtype=torch.int8)
    gs = (torch.rand(e, 2*m, h//g, device='cuda') * .0003 + .0001).to(args.dtype)
    ds = (torch.rand(e, h, m//g, device='cuda') * .0003 + .0001).to(args.dtype)
    return gu, gs, down, ds


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--lengths', default='128,512,2048')
    p.add_argument('--routings', default='spread,shared,skewed')
    p.add_argument('--tiles', default='16x64x64,32x64x64,32x128x64,16x64x128,64x64x64')
    p.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16'])
    p.add_argument('--experts', type=int, default=256)
    p.add_argument('--hidden', type=int, default=2048)
    p.add_argument('--intermediate', type=int, default=512)
    p.add_argument('--group', type=int, default=128)
    p.add_argument('--top-k', type=int, default=8)
    args = p.parse_args()
    args.dtype = getattr(torch, args.dtype)
    tiles = [tuple(int(v) for v in tile.split('x')) for tile in args.tiles.split(',')]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import triton
    torch.manual_seed(901)
    gu, gs, down, ds = synthetic_bank(args)
    result = {'environment': {'torch': str(torch.__version__), 'triton': triton.__version__,
                              'gpu': torch.cuda.get_device_name(),
                              'capability': torch.cuda.get_device_capability()},
              'shape': {'experts': args.experts, 'hidden': args.hidden, 'intermediate': args.intermediate,
                        'group': args.group, 'top_k': args.top_k, 'dtype': str(args.dtype)},
              'weights': 'synthetic uniform INT8 banks and scales, standard-normal activations',
              'rows': []}
    for t in (int(v) for v in args.lengths.split(',')):
        x = torch.randn(t, args.hidden, device='cuda', dtype=args.dtype)
        weights = torch.softmax(torch.randn(t, args.top_k, device='cuda'), -1).to(args.dtype)
        for routing in args.routings.split(','):
            ids = torch.rand(t, args.experts, device='cuda').topk(args.top_k, -1).indices
            if routing == 'shared':
                ids[:] = torch.arange(args.top_k - 1, -1, -1, device='cuda')
            if routing == 'skewed':
                ids[:t*3//4] = torch.arange(args.top_k - 1, -1, -1, device='cuda')
            row = {'tokens': t, 'routing': routing, **diagnose(x, ids, weights, gu, gs, down, ds, tiles)}
            result['rows'].append(row)
            out.write_text(json.dumps(result, indent=2) + '\n')
            for name, gate in row['fp64_gate'].items():
                print(t, routing, name, 'candidate %.5f existing %.5f passes %s' % (
                    gate['candidate']['rms_relative'], gate['existing']['rms_relative'], gate['passes']), flush=True)
    result['all_pass'] = all(g['passes'] for row in result['rows'] for g in row['fp64_gate'].values())
    out.write_text(json.dumps(result, indent=2) + '\n')
    if not result['all_pass']:
        print('FP64 gate failed; inspect', out, file=sys.stderr)
        raise SystemExit(3)  # completed, candidate rejected (as scripts/bench_moe_prefill.py)


if __name__ == '__main__':
    main()
