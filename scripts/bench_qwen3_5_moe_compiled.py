"""Paired plain/parallel-MTP decode with compile + CUDA Graphs on one checkpoint.

Retains every generated token and repetition. Timings exclude prefill, warmup,
compilation and profiling; MTP includes final-round overshoot in its denominator,
as in the dense G4 experiment. Greedy parity is distinct from HF task accuracy.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch._inductor.config as inductor_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer_opt.qwen3_5 import FixedShapeDecoder, Qwen3_5ForCausalLM
from infer_opt.speculative import MTPSpeculativeDecoder
from validate_qwen3_5_hf import CODE_PROMPT_SOURCE, build_prompts
from validate_llamacpp_qwen3_5_quality import save


@torch.inference_mode()
def generate(model, decoder, ids, new_tokens, mtp=False):
    if mtp:
        output = decoder.prefill(ids)[0].tolist()
    else:
        decoder.cache.reset()
        hidden = model.model.language_model(ids, decoder.cache)
        token = model.lm_head(hidden[:, -1:]).argmax(-1)
        output_tensors = [token]
    torch.cuda.synchronize()
    started = time.perf_counter()
    if mtp:
        while len(output) < new_tokens:
            output.extend(decoder.round())
    else:
        for _ in range(new_tokens - 1):
            token = decoder.step(token)[:, -1].argmax(-1, keepdim=True)
            output_tensors.append(token)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not mtp:
        output = torch.cat(output_tensors, dim=1)[0].tolist()
    result = {'generated_ids': output[:new_tokens], 'emitted_tokens': len(output),
              'decode_ms': elapsed_ms, 'ms_per_token': elapsed_ms / (len(output) - 1)}
    if mtp:
        result.update(decoder.stats.as_dict())
        result['round_ms'] = elapsed_ms / decoder.stats.rounds
    return result


def graph_evidence(decoder, token=None):
    # Inspect a warmed step outside timed measurements; retain launch counts only.
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            if token is None:
                decoder.round()
            else:
                decoder.step(token)
            torch.cuda.synchronize()
        events = {event.key: event.count for event in prof.key_averages()
                  if 'graph' in event.key.lower() or 'compiled' in event.key.lower()}
        return {'events': events,
                'cuda_graph_launch_observed': any('cudaGraphLaunch' in key for key in events)}
    except Exception as exc:
        return {'profiling_error': repr(exc), 'cuda_graph_launch_observed': None}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--new-tokens', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--cache-capacity', type=int, default=2048)
    parser.add_argument('--ks', default='1,2,3')
    args = parser.parse_args()
    if min(args.new_tokens - 1, args.repeats, args.warmup) < 1:
        parser.error('new-tokens >=2, repeats >=1 and warmup >=1 required')
    ks = [int(k) for k in args.ks.split(',')]
    if any(k < 1 or k > 4 for k in ks):
        parser.error('k must be in 1..4')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    inductor_config.max_autotune = True
    inductor_config.coordinate_descent_tuning = True
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.weights)
    texts = build_prompts(tok, CODE_PROMPT_SOURCE.read_text())
    names = ('chat_long', 'code_1k', 'prose_1k', 'math', 'korean')
    prompts = {name: tok(texts[name], return_tensors='pt').input_ids.cuda() for name in names}
    if max(ids.shape[1] for ids in prompts.values()) + args.new_tokens + 32 > args.cache_capacity:
        parser.error('cache too small for prompts and speculative overshoot')
    save(out / 'inputs.json', {name: {'text': texts[name], 'input_ids': ids[0].tolist()}
                              for name, ids in prompts.items()})
    model = Qwen3_5ForCausalLM.from_quantized_dir(args.weights, device='cuda', kernels='fla-delta')
    fused = model.fuse_projections()
    if model.config.num_experts != 256 or model.mtp is None:
        raise RuntimeError('expected 35B-A3B checkpoint with bundled MTP')
    manifest = {'args': vars(args), 'mode': 'compile-reduce-overhead',
                'compile': {'mode': 'reduce-overhead', 'fullgraph': True, 'dynamic': False},
                'verification': 'parallel', 'allow_inexact': True, 'fused_groups': fused,
                'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
                'cache_capacity': args.cache_capacity, 'max_autotune': True,
                'coordinate_descent_tuning': True,
                'timing': 'synchronized host wall clock; prefill/warmup/compilation excluded; '
                          'final overshoot included in MTP emitted-token denominator',
                'mtp_precision': 'INT8 group-128', 'lm_head': 'BF16',
                'scope': 'decode throughput and same-checkpoint greedy parity; not task accuracy'}
    save(out / 'manifest.json', manifest)
    baselines, rows, evidence = {}, [], {}
    methods = [('plain', None)] + [(f'mtp{k}', k) for k in ks]
    for method, k in methods:
        save(out / 'progress.json', {'method': method, 'stage': 'compile_and_warmup'})
        if k is None:
            decoder = FixedShapeDecoder(model, model.new_cache(1, args.cache_capacity),
                                        'compile-reduce-overhead')
        else:
            # Constructor adds speculative room; make every target cache exactly equal.
            max_len = args.cache_capacity - 4 * (k + 1) - k
            decoder = MTPSpeculativeDecoder(model, max_len, k=k,
                mode='compile-reduce-overhead', verification='parallel', allow_inexact=True)
        first = next(iter(prompts.values()))
        if k is None:
            decoder.prepare()
        started = time.perf_counter()
        for _ in range(args.warmup):
            generate(model, decoder, first, 16, mtp=k is not None)
        setup_s = time.perf_counter() - started
        for name, ids in prompts.items():
            save(out / 'progress.json', {'method': method, 'prompt': name, 'stage': 'measure'})
            for _ in range(args.warmup):
                generate(model, decoder, ids, args.new_tokens, mtp=k is not None)
            samples = []
            for repeat in range(args.repeats):
                sample = generate(model, decoder, ids, args.new_tokens, mtp=k is not None)
                if k is None:
                    baselines.setdefault(name, sample['generated_ids'])
                want = baselines[name]
                same = [a == b for a, b in zip(sample['generated_ids'], want)]
                sample.update(repeat=repeat, token_agreement=sum(same) / len(same),
                              first_divergence=next((i for i, match in enumerate(same) if not match), None))
                samples.append(sample)
                save(out / f'{method}-{name}.json', {'method': method, 'prompt': name,
                     'prompt_tokens': ids.shape[1], 'baseline_ids': want, 'samples': samples})
            median_ms = statistics.median(s['ms_per_token'] for s in samples)
            row = {'method': method, 'prompt': name, 'prompt_tokens': ids.shape[1],
                   'repeats': len(samples), 'median_ms': median_ms, 'tokens_per_second': 1000 / median_ms,
                   'divergent_runs': sum(s['first_divergence'] is not None for s in samples),
                   'first_divergence': [s['first_divergence'] for s in samples],
                   'token_agreement': [s['token_agreement'] for s in samples],
                   'setup_s': setup_s, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
            if k is not None:
                base_ms = next(r['median_ms'] for r in rows if r['method'] == 'plain' and r['prompt'] == name)
                row['timing_ratio'] = base_ms / median_ms
                row['valid_speedup'] = row['timing_ratio'] if row['divergent_runs'] == 0 else None
                row['tokens_per_round'] = statistics.median(s['tokens_per_round'] for s in samples)
            rows.append(row)
            save(out / 'summary.json', {'manifest': manifest, 'rows': rows})
            print(json.dumps(row), flush=True)
        evidence[method] = graph_evidence(decoder, first[:, :1] if k is None else None)
        save(out / 'cuda-graph-evidence.json', evidence)
        del decoder
        gc.collect()
        torch.cuda.empty_cache()
    save(out / 'progress.json', {'stage': 'complete'})


if __name__ == '__main__':
    main()
