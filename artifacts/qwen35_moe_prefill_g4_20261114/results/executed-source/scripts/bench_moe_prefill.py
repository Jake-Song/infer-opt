"""Paired MoE prefill experiments: synthetic experts or full-model warm TTFT.

Model mode times text-model prefill + last-position vocabulary projection + argmax.
Tokenization, cache reset, warmup, compilation and profiling are excluded.

Acceptance: ``sorted`` keeps the expert arithmetic and must be bit-exact. ``grouped``
changes the reduction and rounding points, so it is judged by distance to an FP64
evaluation of the same INT8 banks (not larger than the adopted loop's), and in the
model by last-position logit KL and greedy continuation parity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer_opt.moe import Int8Experts
from infer_opt.prefill_profile import RouteRecorder, section_times


GATE_FAILED = 3  # exit status of a completed run whose grouped candidate missed its gate


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def differences(got, want):
    delta = got.float() - want.float()
    return {'exact': torch.equal(got, want), 'max_abs': delta.abs().max().item(),
            'rms_relative': (delta.square().mean().sqrt() /
                             want.float().square().mean().sqrt().clamp_min(1e-12)).item(),
            'allclose': torch.allclose(got, want, atol=2e-4, rtol=.03)}


def measure(fn, reset, repeats, warmup):
    for _ in range(warmup):
        reset()
        fn()
    torch.cuda.synchronize()
    samples = []
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    b.record()
    b.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        a.record()
        result = fn()
        b.record()
        b.synchronize()
        samples.append({'wall_ms': (time.perf_counter() - start) * 1000,
                        'cuda_ms': a.elapsed_time(b)})
    return {'samples': samples, 'median_wall_ms': statistics.median(s['wall_ms'] for s in samples),
            'median_cuda_ms': statistics.median(s['cuda_ms'] for s in samples),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}, result


def profile(fn, reset, path, model=None):
    reset()
    torch.cuda.synchronize()
    handles, active = [], {}
    if model is not None:
        def before(module, inputs):
            ctx = torch.profiler.record_function('moe::routed_experts')
            active[module] = ctx
            ctx.__enter__()
        def after(module, inputs, output):
            active.pop(module).__exit__(None, None, None)
        for module in model.modules():
            if isinstance(module, Int8Experts):
                handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as prof:
            fn()
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    prof.export_chrome_trace(str(path.with_suffix('.trace.json')))
    events = prof.key_averages()
    save(path.with_suffix('.json'), [{'key': e.key, 'count': e.count,
        'self_cpu_us': e.self_cpu_time_total,
        'self_device_us': e.self_device_time_total,
        'total_cpu_us': e.cpu_time_total, 'total_device_us': e.device_time_total} for e in events])


def synthetic(args, out):
    from infer_opt.moe_prefill import float64_gate, float64_int8_experts, grouped_int8_experts
    e, h, m, g, top = 256, 2048, 512, 128, 8
    gu = torch.randint(-127, 128, (e, 2*m, h), device='cuda', dtype=torch.int8)
    down = torch.randint(-127, 128, (e, h, m), device='cuda', dtype=torch.int8)
    gs = (torch.rand(e, 2*m, h//g, device='cuda') * .0003 + .0001).bfloat16()
    ds = (torch.rand(e, h, m//g, device='cuda') * .0003 + .0001).bfloat16()
    module = Int8Experts(gu, gs, down, ds, g)
    rows = []
    for t in args.lengths:
        x = torch.randn(t, h, device='cuda', dtype=torch.bfloat16)
        weights = torch.softmax(torch.randn(t, top, device='cuda'), -1).bfloat16()
        for routing in ['spread', 'shared', 'skewed']:
            ids = torch.rand(t, e, device='cuda').topk(top, -1).indices
            if routing == 'shared':
                ids[:] = torch.arange(top - 1, -1, -1, device='cuda')
            if routing == 'skewed':
                ids[:t*3//4] = torch.arange(top - 1, -1, -1, device='cuda')
            baseline = lambda: module.forward_reference(x, ids, weights)
            baseline_time, want = measure(baseline, lambda: None, args.repeats, args.warmup)
            if 'sorted' in args.backends:
                candidate = lambda: module.forward_sorted(x, ids, weights)
                error = differences(candidate(), want)
                if not error['exact']:
                    save(out / 'sorted-failure.json', {'tokens': t, 'routing': routing, 'numerics': error})
                    raise RuntimeError('sorted routing changed expert outputs')
                timing, _ = measure(candidate, lambda: None, args.repeats, args.warmup)
                row = {'backend': 'sorted', 'tokens': t, 'routing': routing, 'numerics': error,
                       'reference': baseline_time, 'sorted': timing,
                       'timing_ratio': baseline_time['median_wall_ms'] / timing['median_wall_ms']}
                rows.append(row)
                save(out / 'experts.json', {'rows': rows})
                print(json.dumps(row), flush=True)
                if args.profile and t == args.lengths[0] and routing == 'spread':
                    profile(baseline, lambda: None, out / 'experts-reference')
                    profile(candidate, lambda: None, out / 'experts-sorted')
            if 'grouped' not in args.backends:
                continue
            configs = [(16, 64, 64), (32, 64, 64), (64, 64, 64), (32, 128, 64), (16, 64, 128)]
            truth = float64_int8_experts(x, ids, weights, gu, gs, down, ds)
            for bm, bn, bk in configs:
                candidate = lambda: grouped_int8_experts(x, ids, weights, gu, gs, down, ds,
                                                         block_m=bm, block_n=bn, block_k=bk)
                got = candidate()
                row = {'backend': 'grouped', 'tokens': t, 'routing': routing, 'tile': [bm, bn, bk],
                       'numerics': differences(got, want), 'fp64_gate': float64_gate(got, want, truth),
                       'reference': baseline_time}
                del got
                if row['fp64_gate']['passes']:
                    timing, _ = measure(candidate, lambda: None, args.repeats, args.warmup)
                    row.update(grouped=timing,
                               timing_ratio=baseline_time['median_wall_ms'] / timing['median_wall_ms'])
                rows.append(row)
                save(out / 'experts.json', {'rows': rows})
                print(json.dumps(row), flush=True)
            if args.profile and t == args.lengths[0] and routing == 'spread':
                profile(baseline, lambda: None, out / 'experts-reference')
                profile(candidate, lambda: None, out / 'experts-grouped')
    return [f"{row['tokens']}-{row['routing']}-{'x'.join(map(str, row['tile']))}" for row in rows
            if row['backend'] == 'grouped' and not row['fp64_gate']['passes']]


def full_model(args, out):
    from transformers import AutoTokenizer
    from infer_opt.qwen3_5 import Qwen3_5ForCausalLM
    from validate_qwen3_5_hf import build_prompts, CODE_PROMPT_SOURCE
    tok = AutoTokenizer.from_pretrained(args.weights)
    texts = build_prompts(tok, CODE_PROMPT_SOURCE.read_text())
    model = Qwen3_5ForCausalLM.from_quantized_dir(args.weights, device='cuda', kernels='fla-delta')
    fused = model.fuse_projections()
    modules = [m for m in model.model.language_model.modules() if isinstance(m, Int8Experts)]
    if len(modules) != 40 or model.config.num_experts != 256:
        raise ValueError('expected Qwen3.5-35B-A3B INT8 target')
    cache = model.new_cache(1, max(args.lengths) + args.decode_tokens + 8)
    rows, inputs, routes, failures = [], {}, {}, []
    for name in args.prompts.split(','):
        raw = tok(texts[name], add_special_tokens=False).input_ids
        for t in args.lengths:
            # Exact-size frozen token prefixes; repeated when the fixture is shorter.
            token_ids = (raw * ((t + len(raw) - 1) // len(raw)))[:t]
            ids = torch.tensor([token_ids], device='cuda')
            key = f'{name}-{t}'
            inputs[key] = token_ids
            save(out / 'inputs.json', inputs)
            reference_logits = None
            reference_ids = None
            row = {'prompt': name, 'tokens': t, 'fused_groups': fused}
            for backend in args.backends:
                for module in modules:
                    module.prefill_backend = backend
                def prefill():
                    hidden = model.model.language_model(ids, cache)
                    logits = model.lm_head(hidden[:, -1:])
                    return logits, logits.argmax(-1)
                timing, (logits, token) = measure(prefill, cache.reset, args.repeats, args.warmup)
                timing['prompt_tokens_per_second'] = t * 1000 / timing['median_wall_ms']
                generated = [token]
                top2 = logits.float().flatten().topk(2).values
                margins = [(top2[0] - top2[1]).item()]  # near ties explain a continuation divergence
                for _ in range(args.decode_tokens - 1):
                    step = model.decode_step(token, cache)[:, -1]
                    token = step.argmax(-1, keepdim=True)
                    top2 = step.float().flatten().topk(2).values
                    margins.append((top2[0] - top2[1]).item())
                    generated.append(token)
                timing['top2_margins'] = margins
                generated_ids = torch.cat(generated, dim=1)[0].tolist()
                timing['generated_ids'] = generated_ids
                if backend == 'reference':
                    reference_logits, reference_ids = logits.clone(), generated_ids
                else:
                    timing['numerics'] = differences(logits, reference_logits)
                    want = reference_logits.float().log_softmax(-1)
                    got = logits.float().log_softmax(-1)
                    timing['kl_reference_to_candidate'] = (want.exp() * (want - got)).sum().item()
                    timing['first_divergence'] = next((i for i, (a, b) in
                        enumerate(zip(generated_ids, reference_ids)) if a != b), None)
                row[backend] = timing
                if args.breakdown:
                    timing['sections_ms'] = section_times(model, prefill, cache.reset, args.breakdown)
                    if 'routes' not in row:  # sorted routes equal the reference's; grouped may differ by rounding
                        cache.reset()
                        with RouteRecorder(model.model.language_model) as recorder:
                            prefill()
                        summary = recorder.summary()
                        routes[key] = recorder.counts
                        row['routes'] = {k: v for k, v in summary.items() if k != 'layers'}
                        row['routes']['layers'] = [{k: v for k, v in layer.items() if k != 'experts'}
                                                   for layer in summary['layers'].values()]
                        save(out / 'routes.json', routes)
                if args.profile and t == args.lengths[0]:
                    profile(prefill, cache.reset, out / f'model-{key}-{backend}', model)
            for backend in args.backends[1:]:
                row[backend]['timing_ratio'] = row['reference']['median_wall_ms'] / row[backend]['median_wall_ms']
                row[backend]['greedy_parity'] = row[backend]['first_divergence'] is None
                if backend == 'sorted' and not (row[backend]['numerics']['exact'] and row[backend]['greedy_parity']):
                    save(out / 'sorted-failure.json', row)
                    raise RuntimeError('sorted routing changed model logits or greedy output')
                if backend == 'grouped':
                    kl = row[backend]['kl_reference_to_candidate']
                    row[backend]['model_gate'] = {'kl_limit': args.kl_limit, 'kl': kl,
                        'greedy_parity': row[backend]['greedy_parity'],
                        'passes': kl <= args.kl_limit and row[backend]['greedy_parity']}
                    if not row[backend]['model_gate']['passes']:
                        failures.append(f'{key}: KL {kl:.3g}, first divergence {row[backend]["first_divergence"]}')
            rows.append(row)
            save(out / 'model.json', {'rows': rows})
            print(json.dumps(row), flush=True)
    return failures


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', help='Omit to benchmark synthetic expert banks without model downloads')
    p.add_argument('--out', required=True)
    p.add_argument('--lengths', default='128,512,2048,4096')
    p.add_argument('--prompts', default='math,korean')
    p.add_argument('--backends', default='reference,sorted', help='reference,sorted[,grouped]; grouped is experimental')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--decode-tokens', type=int, default=16)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--breakdown', type=int, default=0, metavar='REPEATS',
                   help='model mode: also record per-layer expert loads and synchronized section times')
    p.add_argument('--kl-limit', type=float, default=2e-3,
                   help='grouped model gate: last-position KL(reference || candidate) ceiling; the default is '
                        "the scale of this checkpoint's recorded INT8-vs-HF-BF16 mean KL")
    args = p.parse_args()
    args.lengths = [int(x) for x in args.lengths.split(',')]
    args.backends = args.backends.split(',')
    if (args.backends[0] != 'reference' or len(args.backends) < 2 or
            len(set(args.backends)) != len(args.backends) or
            any(b not in ('reference', 'sorted', 'grouped') for b in args.backends)):
        p.error('backends must start with reference and include unique sorted/grouped candidates')
    if min(*args.lengths, args.warmup, args.repeats, args.decode_tokens) <= 0:
        p.error('lengths, warmup, repeats and decode-tokens must be positive')
    if min(args.lengths) <= 5:
        p.error('prefill lengths must be at least 6; decode has a separate dispatcher')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(901)
    save(out / 'manifest.json', {'args': vars(args), 'torch': str(torch.__version__),
        'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
        'timing': 'warm synchronized host wall time and CUDA events, reset excluded',
        'scope': 'synthetic experts' if not args.weights else 'batch-one text-model TTFT, MTP off',
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), ROOT/'infer_opt/moe.py', ROOT/'infer_opt/moe_prefill.py',
                         ROOT/'infer_opt/prefill_profile.py',
                         ROOT/'scripts/validate_qwen3_5_hf.py', ROOT/'scripts/code_prompt_source.txt']}})
    failures = full_model(args, out) if args.weights else synthetic(args, out)
    # A failed grouped gate is a recorded result: every case is measured and saved first.
    save(out / 'complete.json', {'complete': True, 'grouped_gate_failures': failures})
    if failures:
        print('grouped gate failed:', '; '.join(failures), file=sys.stderr)
        raise SystemExit(GATE_FAILED)


if __name__ == '__main__':
    main()
