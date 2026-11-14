"""MTP on/off greedy parity of an INT8 MoE checkpoint under each prefill backend.

The prefill backend runs twice when MTP is on: in the target prefill and in the
MTP head's own prefill over the prompt (its expert bank sees every prompt row).
Decode and draft verification use at most five rows and never reach it.

For every backend and prompt this generates with MTP off (``FixedShapeDecoder``)
and with MTP on (``MTPSpeculativeDecoder``) for each verification and ``k``, and
compares tokens. Prefill and decode are timed separately and never added up:
``prefill_ms`` is the call that produces the first token (for MTP on it also
runs the MTP prefill and the first drafts), ``decode_ms_per_token`` covers only
the tokens after it.

Gate: a backend must keep every MTP-on/MTP-off match the first backend has, and
where its MTP-off output leaves the first backend's, the first backend's top-2
logit margin at that token must be a near tie (at most ``--strong-margin``).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer_opt.moe import Int8Experts
from infer_opt.qwen3_5 import FixedShapeDecoder
from infer_opt.speculative import MTPSpeculativeDecoder

GATE_FAILED = 3


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def first_divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def margin(logits):
    top = logits.float().flatten().topk(2).values
    return (top[0] - top[1]).item()


def plain(model, decoder, ids, new_tokens, margins=False):
    """MTP off: (tokens, top-2 margins or None, decode ms per token)."""
    decoder.cache.reset()
    hidden = model.model.language_model(ids, decoder.cache)
    logits = model.lm_head(hidden[:, -1:])
    token = logits.argmax(-1)
    if decoder.graph is None:
        decoder.prepare()
    out, gaps = [token], [margin(logits)] if margins else None
    sync()
    start = time.perf_counter()
    for _ in range(new_tokens - 1):
        logits = decoder.step(token)[:, -1]
        token = logits.argmax(-1, keepdim=True)
        out.append(token)
        if margins:
            gaps.append(margin(logits))
    sync()
    elapsed = (time.perf_counter() - start) * 1000
    return torch.cat(out, dim=1)[0].tolist(), gaps, elapsed / (new_tokens - 1)


def speculative(decoder, ids, new_tokens):
    """MTP on: (tokens, decode ms per emitted token, acceptance statistics)."""
    out = decoder.prefill(ids)[0].tolist()
    if decoder.graph is None:
        decoder.prepare()
    sync()
    start = time.perf_counter()
    while len(out) < new_tokens:
        out.extend(decoder.round())
    sync()
    elapsed = (time.perf_counter() - start) * 1000
    return out[:new_tokens], elapsed / (len(out) - 1), decoder.stats.as_dict()


def time_prefill(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


@torch.inference_mode()
def validate(model, prompts, *, backends, ks, verifications, new_tokens, capacity, mode='cudagraph',
             prefill_repeats=3, strong_margin=.5, progress=None):
    banks = [m for m in model.modules() if isinstance(m, Int8Experts)]
    if not banks or model.mtp is None:
        raise ValueError('expected an INT8 MoE model with an MTP head')
    base = FixedShapeDecoder(model, model.new_cache(1, capacity), mode)
    specs = {(v, k): MTPSpeculativeDecoder(model, capacity - 4 * (k + 1) - k, k=k, mode=mode, verification=v)
             for v in verifications for k in ks}
    lm = model.model.language_model
    rows = []
    for backend in backends:
        for bank in banks:  # the target's banks and the MTP head's
            bank.prefill_backend = backend
            bank.grouped_options = {}
        for name, ids in prompts.items():
            if progress:
                progress({'backend': backend, 'prompt': name})
            def target_prefill():
                base.cache.reset()
                return model.lm_head(lm(ids, base.cache)[:, -1:]).argmax(-1)
            tokens, gaps, _ = plain(model, base, ids, new_tokens, margins=True)
            again, _, decode_ms = plain(model, base, ids, new_tokens)
            row = {'backend': backend, 'prompt': name, 'prompt_tokens': ids.shape[1],
                   'plain': {'generated_ids': tokens, 'top2_margins': gaps, 'repeatable': again == tokens,
                             'prefill_ms': time_prefill(target_prefill, prefill_repeats),
                             'decode_ms_per_token': decode_ms}, 'mtp': []}
            for (verification, k), decoder in specs.items():
                got, _, _ = speculative(decoder, ids, new_tokens)
                again, decode_ms, stats = speculative(decoder, ids, new_tokens)
                entry = {'verification': verification, 'k': k, 'generated_ids': got, 'repeatable': again == got,
                         'first_divergence_from_plain': first_divergence(got, tokens),
                         'decode_ms_per_token': decode_ms, 'tokens_per_round': stats['tokens_per_round']}
                if verification == verifications[0]:  # the prefill does not depend on the verification
                    entry['prefill_ms'] = time_prefill(lambda: decoder.prefill(ids), prefill_repeats)
                row['mtp'].append(entry)
            rows.append(row)
    first = {row['prompt']: row for row in rows if row['backend'] == backends[0]}
    failures = []
    for row in rows:
        if row['backend'] == backends[0]:
            continue
        ref = first[row['prompt']]
        at = first_divergence(row['plain']['generated_ids'], ref['plain']['generated_ids'])
        row['plain']['first_divergence_from_' + backends[0]] = at
        if at is not None:
            row['plain']['margins_at_divergence'] = {backends[0]: ref['plain']['top2_margins'][at],
                                                     row['backend']: row['plain']['top2_margins'][at]}
            if ref['plain']['top2_margins'][at] > strong_margin:
                failures.append(f"{row['backend']} {row['prompt']}: MTP-off output leaves {backends[0]} at token "
                                f"{at} where its margin is {ref['plain']['top2_margins'][at]:.3f}")
        for entry, want in zip(row['mtp'], ref['mtp']):
            entry['first_divergence_from_' + backends[0]] = first_divergence(entry['generated_ids'],
                                                                             want['generated_ids'])
            if want['first_divergence_from_plain'] is None and entry['first_divergence_from_plain'] is not None:
                failures.append(f"{row['backend']} {row['prompt']} {entry['verification']} k={entry['k']}: MTP on "
                                f"leaves MTP off at token {entry['first_divergence_from_plain']}")
    summary = {backend: {
        'mtp_runs': sum(len(r['mtp']) for r in rows if r['backend'] == backend),
        'mtp_runs_equal_to_plain': sum(e['first_divergence_from_plain'] is None
                                       for r in rows if r['backend'] == backend for e in r['mtp']),
        'all_repeatable': all(r['plain']['repeatable'] and all(e['repeatable'] for e in r['mtp'])
                              for r in rows if r['backend'] == backend)} for backend in backends}
    return {'rows': rows, 'summary': summary, 'gate': {'strong_margin': strong_margin, 'failures': failures,
                                                        'passes': not failures}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--backends', default='reference,grouped', help='the first is the yardstick')
    p.add_argument('--prompts', default='chat_long,code_1k,prose_1k,math,korean')
    p.add_argument('--ks', default='1,2')
    p.add_argument('--verifications', default='sequential,consistent')
    p.add_argument('--new-tokens', type=int, default=256)
    p.add_argument('--cache-capacity', type=int, default=2048)
    p.add_argument('--prefill-repeats', type=int, default=3)
    p.add_argument('--strong-margin', type=float, default=.5,
                   help="reference top-2 logit margin above which a changed token is not a near tie "
                        "(the repository's strong-mismatch threshold)")
    args = p.parse_args()
    from transformers import AutoTokenizer
    from infer_opt.qwen3_5 import Qwen3_5ForCausalLM
    from validate_qwen3_5_hf import CODE_PROMPT_SOURCE, build_prompts
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save = lambda name, value: (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(args.weights)
    texts = build_prompts(tok, CODE_PROMPT_SOURCE.read_text())
    prompts = {name: tok(texts[name], return_tensors='pt').input_ids.cuda() for name in args.prompts.split(',')}
    if max(ids.shape[1] for ids in prompts.values()) + args.new_tokens + 32 > args.cache_capacity:
        p.error('cache too small for prompts and speculative overshoot')
    save('inputs.json', {name: ids[0].tolist() for name, ids in prompts.items()})
    model = Qwen3_5ForCausalLM.from_quantized_dir(args.weights, device='cuda', kernels='fla-delta')
    fused = model.fuse_projections()
    if model.config.num_experts != 256 or model.mtp is None:
        raise RuntimeError('expected the 35B-A3B checkpoint with its bundled MTP head')
    save('manifest.json', {'args': vars(args), 'fused_groups': fused, 'mode': 'cudagraph',
        'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
        'default_prefill_backend': next(m for m in model.modules() if isinstance(m, Int8Experts)).prefill_backend,
        'expert_banks': sum(isinstance(m, Int8Experts) for m in model.modules()),
        'timing': 'synchronized host wall clock; prefill and decode separate; MTP decode includes final-round '
                  'overshoot in its token count',
        'scope': 'same-checkpoint greedy parity; not HF BF16 equivalence or task accuracy'})
    result = validate(model, prompts, backends=args.backends.split(','), ks=[int(k) for k in args.ks.split(',')],
                      verifications=args.verifications.split(','), new_tokens=args.new_tokens,
                      capacity=args.cache_capacity, prefill_repeats=args.prefill_repeats,
                      strong_margin=args.strong_margin, progress=lambda state: save('progress.json', state))
    save('mtp-parity.json', result)
    save('complete.json', {'complete': True, 'gate': result['gate']})
    print(json.dumps({'summary': result['summary'], 'gate': result['gate']}, indent=1))
    if not result['gate']['passes']:
        raise SystemExit(GATE_FAILED)


if __name__ == '__main__':
    main()
