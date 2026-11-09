"""Production fullgraph CUDA traces plus separately calibrated phase diagnostics.

The production decoder is unchanged. Segmented compilation changes fusion
boundaries: its times and output differences are explicitly reported separately.
All model weights/caches are supplied by the caller under /tmp/infer-opt-models.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import gzip
from functools import partial
import json
from pathlib import Path
import re
import statistics
import sys
import time

import torch
import torch._inductor.config as ic

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from infer_opt.qwen3_5 import FixedShapeDecoder, Qwen3_5ForCausalLM
from infer_opt.speculative import MTPSpeculativeDecoder, spec_round
from infer_opt.inductor_choices import wide_persistent_reductions
from validate_qwen3_5_hf import build_prompts, CODE_PROMPT_SOURCE
from validate_llamacpp_qwen3_5_quality import save
from bench_qwen3_5_moe_compiled import generate


def category(name):
    if re.search(r'(^|\W)_(project|combine)(\W|$)', name):
        return 'routed_experts'
    if re.search(r'gatherTopK|bitonicSortKVInPlace|sortKeyValue|topk', name, re.IGNORECASE):
        return 'routing_topk_sort'
    for label, pattern in [
        ('int8_multirow', r'_w8a16_group_gemv_rows'),
        ('int8_single', r'_w8a16_group_gemv_kernel'),
        ('delta_rule', r'fused_recurrent|gated_delta|chunk_'),
        ('attention', r'decode_attention'),
        ('bf16_matmul', r'gemm|gemv|sm80_xmma|cutlass|ampere|triton_mm|splitK|mm_'),
        ('copy_gather', r'fused_copy|Memcpy|memcpy|index_select|gather|scatter|Memset'),
        ('reduction', r'triton_red|triton_per|argmax|reduce'),
        ('pointwise', r'triton_poi'),
    ]:
        if re.search(pattern, name):
            return label
    return 'other'


def union_us(intervals):
    total, left, right = 0., None, None
    for begin, end in sorted(intervals):
        if left is None:
            left, right = begin, end
        elif begin <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = begin, end
    return total + (right - left if left is not None else 0.)


def summarize_trace(trace, calls):
    events = trace['traceEvents']
    gpu = [e for e in events if e.get('ph') == 'X' and
           e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    by_name, counts, by_category = defaultdict(float), defaultdict(int), defaultdict(float)
    for event in gpu:
        by_name[event['name']] += event['dur']
        counts[event['name']] += 1
        by_category[category(event['name'])] += event['dur']
    runtime = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'cuda_runtime']
    sync = [e for e in runtime if re.search(r'Synchronize|Memcpy', e['name'])]
    launches = sum('cudaGraphLaunch' in e['name'] for e in runtime)
    return {'calls': calls, 'gpu_events': len(gpu), 'cuda_graph_launches': launches,
        'gpu_kernel_sum_ms_per_call': sum(e['dur'] for e in gpu) / calls / 1000,
        'gpu_busy_union_ms_per_call': union_us([(e['ts'], e['ts'] + e['dur']) for e in gpu]) / calls / 1000,
        'gpu_category_ms_per_call': {k: v / calls / 1000 for k, v in sorted(by_category.items(), key=lambda x: -x[1])},
        'cpu_sync_api_ms_per_call': sum(e['dur'] for e in sync) / calls / 1000,
        'top_kernels': [{'name': name, 'ms_per_call': us / calls / 1000, 'count_per_call': counts[name] / calls}
                        for name, us in sorted(by_name.items(), key=lambda x: -x[1])[:30]],
        'scope': 'CPU sync API time overlaps GPU execution; it must not be added to GPU kernel time.'}


def spec_decoder(model, k, capacity):
    return MTPSpeculativeDecoder(model, capacity - 4 * (k + 1) - k, k=k,
        mode='compile-reduce-overhead', verification='parallel', allow_inexact=True)


def checkpoint(decoder):
    tensors = decoder._state_tensors() + [decoder.tokens]
    return [x.clone() for x in tensors], decoder.cache.seen_tokens, decoder.mtp_cache.seen_tokens


def restore(decoder, saved):
    for dst, src in zip(decoder._state_tensors() + [decoder.tokens], saved[0]):
        dst.copy_(src)
    decoder.cache.seen_tokens, decoder.mtp_cache.seen_tokens = saved[1:]


def compare_state(decoder, saved):
    tensors = decoder._state_tensors() + [decoder.tokens]
    return {'tensors': len(tensors), 'exact_tensors': sum(bool(torch.equal(a, b)) for a, b in zip(tensors, saved[0])),
            'max_abs_diff': max((a.float() - b.float()).abs().max().item() for a, b in zip(tensors, saved[0]))}


def gpu_event(external=False):
    return torch.cuda.Event(enable_timing=True, external=external)


def production(args, model, prompts, out):
    from torch.profiler import profile, ProfilerActivity, record_function
    # Initialize CUPTI before any graph capture; compilation remains outside profiling.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
        torch.cuda.synchronize()
    rows, baselines = [], {}
    for method, k in [('plain', None)] + [(f'mtp{k}', k) for k in args.ks]:
        save(out / 'progress.json', {'method': method, 'stage': 'compile_warmup'})
        decoder = (FixedShapeDecoder(model, model.new_cache(1, args.cache_capacity), 'compile-reduce-overhead')
                   if k is None else spec_decoder(model, k, args.cache_capacity))
        if k is None:
            decoder.prepare()
        for name, ids in prompts.items():
            for _ in range(args.warmup):
                generate(model, decoder, ids, args.new_tokens, mtp=k is not None)
            samples = [generate(model, decoder, ids, args.new_tokens, mtp=k is not None) for _ in range(args.repeats)]
            if k is None:
                baselines[name] = samples[0]['generated_ids']
            for s in samples:
                s['first_divergence'] = next((i for i, (a, b) in enumerate(zip(s['generated_ids'], baselines[name])) if a != b), None)
            save(out / f'{method}-{name}-timings.json', {'samples': samples, 'baseline_ids': baselines[name]})
            row = {'method': method, 'prompt': name, 'median_ms_token': statistics.median(s['ms_per_token'] for s in samples),
                   'divergent_runs': sum(s['first_divergence'] is not None for s in samples)}
            if k is not None:
                row['round_ms'] = statistics.median(s['round_ms'] for s in samples)
                row['tokens_per_round'] = statistics.median(s['tokens_per_round'] for s in samples)
                decoder.prefill(ids)
                for _ in range(8):
                    decoder.round()
                instrumented = []
                original = decoder.fn
                start, end = gpu_event(), gpu_event()
                def measured_fn(*fn_args):
                    start.record()
                    t0 = time.perf_counter()
                    result = original(*fn_args)
                    timings['cpu_call_ms'] = (time.perf_counter() - t0) * 1000
                    end.record()
                    timings['after_call'] = time.perf_counter()
                    return result
                decoder.fn = measured_fn
                for _ in range(args.profile_rounds):
                    timings = {}
                    t0 = time.perf_counter()
                    emitted = decoder.round()
                    finished = time.perf_counter()
                    elapsed = (finished - t0) * 1000
                    instrumented.append({'wall_ms': elapsed, 'stream_interval_ms': start.elapsed_time(end),
                        'cpu_graph_call_ms': timings['cpu_call_ms'],
                        'after_graph_call_wall_ms': (finished - timings['after_call']) * 1000,
                        'emitted': len(emitted)})
                decoder.fn = original
                save(out / f'{method}-{name}-host-stream.json', {'samples': instrumented,
                    'scope': 'Stream interval includes launch gaps. After-call CPU time includes waiting for GPU; intervals overlap.'})
            save(out / 'progress.json', {'method': method, 'prompt': name, 'stage': 'trace'})
            if k is None:
                decoder.cache.reset()
                hidden = model.model.language_model(ids, decoder.cache)
                token = model.lm_head(hidden[:, -1:]).argmax(-1)
                for _ in range(8):
                    token = decoder.step(token)[:, -1].argmax(-1, keepdim=True)
            else:
                decoder.prefill(ids)
                for _ in range(8):
                    decoder.round()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for i in range(args.profile_rounds):
                    with record_function(f'{method}_iteration_{i}'):
                        if k is None:
                            token = decoder.step(token)[:, -1].argmax(-1, keepdim=True)
                        else:
                            decoder.round()
                torch.cuda.synchronize()
            path = out / f'{method}-{name}-trace.json'
            prof.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            row['profile'] = summarize_trace(trace, args.profile_rounds)
            with gzip.open(str(path) + '.gz', 'wt') as stream:
                json.dump(trace, stream)
            path.unlink()
            save(out / f'{method}-{name}-profile.json', row)
            rows.append(row)
            save(out / 'summary.json', {'rows': rows, 'args': vars(args),
                'scope': 'Unmodified production fullgraph. Timing samples exclude profiler; kernel durations come from profiled replay.'})
            print(json.dumps(row), flush=True)
        del decoder
        gc.collect()
        torch.cuda.empty_cache()
    save(out / 'progress.json', {'stage': 'complete'})


class SegmentedRound:
    """CUDA-event phase diagnostic with separate compiled regions and one outer graph."""
    def __init__(self, model, decoder, k):
        self.model, self.decoder, self.k = model, decoder, k
        cache, mtp = decoder.cache, decoder.mtp_cache
        def verify(tokens):
            return model.model.language_model.forward_verify(tokens, cache)
        def vocab(hidden):
            return model.lm_head(hidden).argmax(-1)
        def commit(tokens, target):
            accepted = (tokens[:, 1:] == target[:, :-1]).long().cumprod(-1).sum(-1) + 1
            cache.commit(accepted)
            return accepted
        def mtp_first(target, hidden, accepted):
            values = model.mtp_hidden(target, hidden, mtp, decode=True)
            mtp.position.add_(accepted)
            last = accepted - 1
            return values.index_select(1, last), last
        def mtp_next(token, hidden):
            values = model.mtp_hidden(token, hidden, mtp, decode=True)
            mtp.position.add_(1)
            return values
        def pack(target, last, first, drafts):
            if k > 1:
                mtp.position.sub_(k - 1)
            decoder.tokens.copy_(torch.cat([target.index_select(1, last), first, *drafts], dim=1))
        def compile_part(fn):
            return torch.compile(fn, fullgraph=True, dynamic=False,
                options={'max_autotune': True, 'coordinate_descent_tuning': True, 'triton.cudagraphs': False})
        self.verify, self.vocab, self.commit, self.mtp_first, self.mtp_next, self.pack = [
            compile_part(fn) for fn in (verify, vocab, commit, mtp_first, mtp_next, pack)]
        self.labels = ['verify_body', 'verify_lm_head', 'accept_commit', 'mtp_first', 'draft_lm_head_0']
        for i in range(1, k):
            self.labels += [f'draft_mtp_{i}', f'draft_lm_head_{i}']
        self.labels += ['pack_next_inputs']
        self.events = [gpu_event(external=True) for _ in range(len(self.labels) + 1)]

    def run(self, mark=False):
        index = 0
        def tick():
            nonlocal index
            if mark:
                self.events[index].record()
            index += 1
        tick()
        with wide_persistent_reductions():
            hidden = self.verify(self.decoder.tokens); tick()
            target = self.vocab(hidden); tick()
            accepted = self.commit(self.decoder.tokens, target); tick()
            hidden, last = self.mtp_first(target, hidden, accepted); tick()
            first = self.vocab(hidden); tick()
            drafts, token = [], first
            for _ in range(1, self.k):
                hidden = self.mtp_next(token, hidden); tick()
                token = self.vocab(hidden); tick()
                drafts.append(token)
            self.pack(target, last, first, drafts); tick()
        return target, accepted


def expert_micro(args, model, decoder, saved, out, name):
    """Cold-cache model-order sweep using real eager-replay activations/routes."""
    from infer_opt.moe_kernels import selected_int8_experts
    captured, hooks = {}, []
    for index, layer in enumerate(model.model.language_model.layers):
        def hook(module, inputs, index=index):
            captured[index] = (module, tuple(x.detach().clone() for x in inputs))
        hooks.append(layer.mlp.experts.register_forward_pre_hook(hook))
    restore(decoder, saved)
    try:
        model.model.language_model.forward_verify(decoder.tokens, decoder.cache)
    finally:
        for h in hooks:
            h.remove()
        restore(decoder, saved)
    assert len(captured) == 40
    flush = torch.empty(128 * 2**20, dtype=torch.uint8, device='cuda')
    rows = []
    for tokens in (1, 2, 4):
        for routing in ('actual', 'forced_shared'):
            entries, routing_rows = [], []
            for index, (module, (x, ids, weights)) in sorted(captured.items()):
                x, ids, weights = x[:tokens].contiguous(), ids[:tokens].contiguous(), weights[:tokens].contiguous()
                if routing == 'forced_shared':
                    ids = ids[:1].expand(tokens, -1).contiguous()
                cpu_ids = ids.cpu().tolist()
                unique = len({expert for row in cpu_ids for expert in row})
                routing_rows.append({'layer': index, 'ids': cpu_ids, 'unique_experts': unique,
                                     'routes': ids.numel(), 'reuse_potential': 1 - unique / ids.numel()})
                entries.append((module, x, ids, weights))
            def sweep():
                return [selected_int8_experts(x, ids, w, m.gate_up_proj, m.gate_up_scale,
                        m.down_proj, m.down_scale) for m, x, ids, w in entries]
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    outputs = sweep()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = sweep()
            times = []
            start, end = gpu_event(), gpu_event()
            for _ in range(args.micro_repeats):
                flush.zero_(); torch.cuda.synchronize()
                start.record(); graph.replay(); end.record(); end.synchronize()
                times.append(start.elapsed_time(end))
            row = {'tokens': tokens, 'routing': routing, 'sweep_median_ms': statistics.median(times),
                   'samples_ms': times, 'routing_by_layer': routing_rows}
            rows.append(row)
            save(out / f'experts-{name}.json', {'rows': rows,
                'scope': '40 target expert blocks only; cold L2 via 128MiB flush before timing. Eager diagnostic replay routes, not fullgraph routing instrumentation.'})
            del graph, outputs
    return rows


def diagnostic(args, model, prompts, out):
    rows = []
    for k in args.ks:
        decoder = spec_decoder(model, k, args.cache_capacity)
        segmented = SegmentedRound(model, decoder, k)
        full_core = torch.compile(partial(spec_round, verification='parallel'), fullgraph=True, dynamic=False,
            options={'max_autotune': True, 'coordinate_descent_tuning': True, 'triton.cudagraphs': False})
        for name, ids in prompts.items():
            save(out / 'progress.json', {'k': k, 'prompt': name, 'stage': 'warmup_and_compile'})
            decoder.prefill(ids)
            for _ in range(8):
                decoder.round()
            saved = checkpoint(decoder)
            with wide_persistent_reductions():
                target, accepted = decoder.fn(model, decoder.tokens, decoder.cache, decoder.mtp_cache)
            expected = (target.clone(), accepted.clone(), decoder.tokens.clone())
            expected_state = checkpoint(decoder)
            del target, accepted
            for _ in range(3):
                restore(decoder, saved)
                with wide_persistent_reductions():
                    full_core(model, decoder.tokens, decoder.cache, decoder.mtp_cache)
            restore(decoder, saved); torch.cuda.synchronize()
            full_events = [gpu_event(external=True), gpu_event(external=True)]
            full_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(full_graph):
                full_events[0].record()
                with wide_persistent_reductions():
                    full_outputs = full_core(model, decoder.tokens, decoder.cache, decoder.mtp_cache)
                full_events[1].record()
            full_graph.replay(); torch.cuda.synchronize()
            full_comparison = {'target_ids_equal': bool(torch.equal(full_outputs[0], expected[0])),
                'accepted_equal': bool(torch.equal(full_outputs[1], expected[1])),
                'next_input_ids_equal': bool(torch.equal(decoder.tokens, expected[2])),
                'state': compare_state(decoder, expected_state)}
            for _ in range(3):
                restore(decoder, saved)
                segmented.run()
            restore(decoder, saved)
            target, accepted = segmented.run()
            comparison = {'target_ids_equal': bool(torch.equal(target, expected[0])),
                          'accepted_equal': bool(torch.equal(accepted, expected[1])),
                          'next_input_ids_equal': bool(torch.equal(decoder.tokens, expected[2])),
                          'state': compare_state(decoder, expected_state)}
            del target, accepted
            restore(decoder, saved); torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = segmented.run(mark=True)
            samples, production_ms, production_host, full_gpu, full_host = [], [], [], [], []
            start, end = gpu_event(), gpu_event()
            for _ in range(args.event_repeats):
                restore(decoder, saved); torch.cuda.synchronize()
                started = time.perf_counter()
                start.record()
                with wide_persistent_reductions():
                    ref = decoder.fn(model, decoder.tokens, decoder.cache, decoder.mtp_cache)
                end.record()
                torch.cat([ref[1], ref[0][0]]).tolist()
                production_host.append((time.perf_counter() - started) * 1000)
                production_ms.append(start.elapsed_time(end))
                del ref
                restore(decoder, saved); torch.cuda.synchronize()
                started = time.perf_counter()
                full_graph.replay()
                torch.cat([full_outputs[1], full_outputs[0][0]]).tolist()
                full_host.append((time.perf_counter() - started) * 1000)
                full_gpu.append(full_events[0].elapsed_time(full_events[1]))
                restore(decoder, saved); torch.cuda.synchronize()
                graph.replay(); torch.cuda.synchronize()
                times = {label: segmented.events[i].elapsed_time(segmented.events[i + 1])
                         for i, label in enumerate(segmented.labels)}
                times['total_ms'] = segmented.events[0].elapsed_time(segmented.events[-1])
                samples.append(times)
            medians = {key: statistics.median(s[key] for s in samples) for key in samples[0]}
            row = {'k': k, 'prompt': name, 'frozen_production_stream_ms': statistics.median(production_ms),
                'frozen_production_host_ms': statistics.median(production_host),
                'full_manual_graph_gpu_ms': statistics.median(full_gpu),
                'full_manual_graph_host_ms': statistics.median(full_host),
                'full_manual_graph_outputs_vs_production': full_comparison,
                'segmented_phase_medians_ms': medians, 'segmented_samples_ms': samples,
                'production_samples_ms': production_ms, 'production_host_samples_ms': production_host,
                'full_manual_gpu_samples_ms': full_gpu, 'full_manual_host_samples_ms': full_host,
                'outputs_vs_production': comparison,
                'segmented_over_full_manual_gpu': medians['total_ms'] / statistics.median(full_gpu),
                'scope': 'Separate compiled regions inside manual CUDA Graph with external CUDA-event nodes. Fusion/tuning boundaries differ from production.'}
            rows.append(row)
            save(out / f'phases-k{k}-{name}.json', row)
            save(out / 'summary.json', {'rows': rows, 'args': vars(args)})
            print(json.dumps({key: value for key, value in row.items() if 'samples' not in key}), flush=True)
            del graph, outputs, expected, expected_state, full_graph, full_outputs
            if k == 3:
                save(out / 'progress.json', {'k': k, 'prompt': name, 'stage': 'expert_micro'})
                expert_micro(args, model, decoder, saved, out, name)
            restore(decoder, saved)
            del saved
        del segmented, decoder, full_core
        gc.collect(); torch.cuda.empty_cache()
    save(out / 'progress.json', {'stage': 'complete'})


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--stage', choices=['production', 'diagnostic'], required=True)
    parser.add_argument('--ks', default='1,3')
    parser.add_argument('--prompts', default='math,korean')
    parser.add_argument('--cache-capacity', type=int, default=2048)
    parser.add_argument('--new-tokens', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--profile-rounds', type=int, default=24)
    parser.add_argument('--event-repeats', type=int, default=20)
    parser.add_argument('--micro-repeats', type=int, default=10)
    args = parser.parse_args(); args.ks = [int(k) for k in args.ks.split(',')]
    torch.set_num_threads(4)
    ic.max_autotune = True; ic.coordinate_descent_tuning = True
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.weights)
    texts = build_prompts(tok, CODE_PROMPT_SOURCE.read_text())
    prompts = {name: tok(texts[name], return_tensors='pt').input_ids.cuda() for name in args.prompts.split(',')}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    save(out / 'inputs.json', {n: {'text': texts[n], 'ids': ids[0].tolist()} for n, ids in prompts.items()})
    model = Qwen3_5ForCausalLM.from_quantized_dir(args.weights, device='cuda', kernels='fla-delta')
    fused = model.fuse_projections()
    save(out / 'manifest.json', {'args': vars(args), 'gpu': torch.cuda.get_device_name(),
        'torch': str(torch.__version__), 'fused_groups': fused, 'mtp': 'INT8 group128',
        'lm_head': 'BF16', 'compile': 'reduce-overhead/fullgraph/static/max_autotune/coordinate_descent'})
    (production if args.stage == 'production' else diagnostic)(args, model, prompts, out)


if __name__ == '__main__':
    main()
