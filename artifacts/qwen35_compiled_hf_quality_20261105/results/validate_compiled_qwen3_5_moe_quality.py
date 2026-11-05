"""Paired eager/compiled INT8 comparison against the same HF BF16 logits.

Uses a published checkpoint without requantization. MTP is loaded but not called.
The first position is eager prefill; all remaining positions use the selected
single-token decoder. Runtime model sources should be pinned by the caller.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validate_llamacpp_qwen3_5_quality import distribution_metrics, save


def eager_equivalence(eager, actual):
    eager = torch.as_tensor(eager).float()
    actual = torch.as_tensor(actual).float()
    metrics = distribution_metrics(eager, actual)
    close = torch.isclose(actual, eager, atol=0.02, rtol=0.02).all(dim=-1)
    return {**metrics, 'reference': 'same-checkpoint fixed-shape eager',
            'atol': 0.02, 'rtol': 0.02, 'allclose_per_step': close.tolist(),
            'allclose_positions': close.sum().item(), 'allclose_passed': bool(close.all())}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--cache-capacity', type=int, default=8456)
    args = parser.parse_args()
    from infer_opt.qwen3_5 import Qwen3_5ForCausalLM, FixedShapeDecoder

    torch.set_num_threads(4)
    ref_dir, out, scratch = Path(args.reference_dir), Path(args.out), Path(args.scratch)
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ref_dir / 'reference.json').read_text())
    required = max(c['prefix'] + len(c['generated_ids']) + 8 for c in reference['cases'])
    if args.cache_capacity < required:
        parser.error(f'cache capacity must be at least {required}')
    # The checkpoint tokenizer must map exactly the same HF-defined token strings to IDs.
    from transformers import AutoTokenizer
    original_tok = AutoTokenizer.from_pretrained(args.weights)
    if original_tok.get_vocab() != json.loads((ref_dir / 'tokenizer-vocab.json').read_text()):
        raise RuntimeError('custom checkpoint/HF reference tokenizer mapping differs')
    model = Qwen3_5ForCausalLM.from_quantized_dir(args.weights, device='cuda', kernels='fla-delta')
    if model.config.num_hidden_layers != 40 or model.config.num_experts != 256:
        raise RuntimeError('unexpected model architecture')
    fused = model.fuse_projections()
    cache = model.new_cache(1, args.cache_capacity)
    save(out / 'load.json', {'strict_reload': True, 'fused_groups': fused,
         'mtp_head_present': model.mtp is not None, 'mtp_called': False,
         'activation_dtype': str(model.model.language_model.embed_tokens.weight.dtype),
         'cache_capacity': args.cache_capacity, 'allocated_gib': torch.cuda.memory_allocated() / 2**30})
    save(out / 'reference.json', reference)
    rows, equivalence, setup = [], [], []
    for mode in ('eager', 'compile-reduce-overhead'):
        print('prepare', mode, flush=True)
        cache.reset()
        decoder = FixedShapeDecoder(model, cache, mode)
        dummy = torch.tensor([[reference['cases'][0]['prompt_ids'][0]]], device='cuda', dtype=torch.long)
        started = time.perf_counter()
        decoder.prepare()
        # First-call compilation/capture and warmups mutate state; reset before evaluation.
        for _ in range(8):
            decoder.step(dummy)
        torch.cuda.synchronize()
        setup.append({'mode': mode, 'warmup_steps': 8, 'setup_s': time.perf_counter() - started})
        save(out / 'setup.json', setup)
        for case in reference['cases']:
            cache.reset()
            ids = torch.tensor([case['prompt_ids']], device='cuda', dtype=torch.long)
            hidden = model.model.language_model(ids, cache)
            actual = [model.lm_head(hidden[:, -1:])[:, -1].float().cpu()]
            del hidden
            for token in case['generated_ids'][:-1]:
                step = torch.tensor([[token]], device='cuda', dtype=torch.long)
                actual.append(decoder.step(step)[:, -1].float().cpu())
            values = torch.cat(actual).numpy()
            want = np.load(ref_dir / (case['name'] + '.npy'), allow_pickle=False)
            row = {'mode': mode, 'prompt': case['name'], 'prefix_tokens': case['prefix'],
                   'hf_logits_sha256': hashlib.sha256(want.tobytes()).hexdigest(),
                   'candidate_logits_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
                   **distribution_metrics(want, values)}
            rows.append(row)
            if mode == 'eager':
                np.save(scratch / (case['name'] + '.npy'), values)
            else:
                eager = np.load(scratch / (case['name'] + '.npy'), allow_pickle=False)
                equivalence.append({'prompt': case['name'], **eager_equivalence(eager, values)})
                save(out / 'compiled-vs-eager.json', equivalence)
            save(out / 'quality.json', {'rows': rows, 'reference': reference['reference'],
                 'reference_revision': reference['model_revision'], 'tokens': reference['tokens'],
                 'kernels': 'fla-delta', 'fused_projections': True, 'mtp': False,
                 'cache_capacity': args.cache_capacity, 'strong_mismatch_margin': 0.5,
                 'scope': 'paired teacher-forced numerical distribution comparison; not task accuracy'})
            print(json.dumps({k: v for k, v in row.items() if k != 'per_step'}), flush=True)
            del actual, values, want, ids
        del decoder
        gc.collect()
    aggregate = {}
    for mode in ('eager', 'compile-reduce-overhead'):
        subset = [r for r in rows if r['mode'] == mode]
        n = sum(r['teacher_forced_tokens'] for r in subset)
        mismatches = sum(len(r['mismatch_steps']) for r in subset)
        aggregate[mode] = {'positions': n,
            'mean_kl': sum(r['mean_kl'] * r['teacher_forced_tokens'] for r in subset) / n,
            'mismatches': mismatches, 'top1_agreement': 1 - mismatches / n,
            'strong_mismatches': sum(r['strong_mismatches'] for r in subset),
            'max_abs_logit_error': max(r['max_abs_logit_error'] for r in subset)}
    aggregate['compiled_vs_eager'] = {
        'positions': sum(r['teacher_forced_tokens'] for r in equivalence),
        'allclose_positions': sum(r['allclose_positions'] for r in equivalence),
        'mismatches': sum(len(r['mismatch_steps']) for r in equivalence),
        'max_abs_logit_error': max(r['max_abs_logit_error'] for r in equivalence),
        'atol': 0.02, 'rtol': 0.02,
    }
    save(out / 'aggregate.json', aggregate)


if __name__ == '__main__':
    main()
