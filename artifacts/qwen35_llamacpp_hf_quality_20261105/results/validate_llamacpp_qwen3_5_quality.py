"""HF BF16 cached-greedy reference versus llama.cpp raw-token replay.

Numerical distribution preservation, not task accuracy. Stages run in separate
processes so the 35B HF reference and GGUF never occupy the GPU together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def write_string(stream, value):
    data = value.encode('utf-8')
    stream.write(struct.pack('<I', len(data)))
    stream.write(data)


def read_exact(stream, n):
    data = stream.read(n)
    if len(data) != n:
        raise ValueError('truncated llama.cpp output')
    return data


def read_u32(stream):
    return struct.unpack('<I', read_exact(stream, 4))[0]


def read_string(stream):
    n = read_u32(stream)
    if n > 1 << 20:
        raise ValueError('oversized llama.cpp output name')
    return read_exact(stream, n).decode('utf-8')


def write_replay(path, reference, vocabulary):
    """Pass HF token IDs directly; the final predicted token is not fed back."""
    vocab = reference['vocab_size']
    with Path(path).open('wb') as out:
        out.write(b'IOQ8TF01')
        out.write(struct.pack('<III', vocab, len(vocabulary), len(reference['cases'])))
        for token, index in sorted(vocabulary.items(), key=lambda item: item[1]):
            if not 0 <= index < vocab:
                raise ValueError('HF tokenizer ID outside model vocabulary')
            out.write(struct.pack('<I', index))
            write_string(out, token)
        for case in reference['cases']:
            prompt, generated = case['prompt_ids'], case['generated_ids']
            if not prompt or not generated:
                raise ValueError('empty reference prompt or continuation')
            ids = prompt + generated[:-1]
            if any(not 0 <= index < vocab for index in ids):
                raise ValueError('reference token ID outside model vocabulary')
            write_string(out, case['name'])
            out.write(struct.pack('<II', len(prompt), len(generated)))
            out.write(np.asarray(ids, dtype='<i4').tobytes())


def distribution_metrics(want, actual, margin_threshold=0.5):
    """Same KL direction and strong-mismatch rule as the existing MoE check."""
    want, actual = torch.as_tensor(want).float(), torch.as_tensor(actual).float()
    if want.ndim != 2 or want.shape != actual.shape or min(want.shape) < 1 or want.shape[1] < 2:
        raise ValueError('expected matching nonempty [steps, vocabulary] logits')
    if not torch.isfinite(want).all() or not torch.isfinite(actual).all():
        raise ValueError('nonfinite logits')
    lp, lq = want.log_softmax(-1), actual.log_softmax(-1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    top = want.topk(2, dim=-1).values
    margins = top[:, 0] - top[:, 1]
    reference_top, actual_top = want.argmax(-1), actual.argmax(-1)
    mismatch = reference_top != actual_top
    strong = mismatch & (margins > margin_threshold)
    return {
        'teacher_forced_tokens': want.shape[0],
        'mean_kl': kl.mean().item(),
        'top1_agreement': (~mismatch).float().mean().item(),
        'strong_mismatches': strong.sum().item(),
        'mismatch_steps': mismatch.nonzero().flatten().tolist(),
        'max_abs_logit_error': (want - actual).abs().max().item(),
        'per_step': [
            {'step': i, 'phase': 'prefill_last' if i == 0 else 'decode',
             'kl': kl[i].item(), 'reference_top1': reference_top[i].item(),
             'actual_top1': actual_top[i].item(), 'reference_margin': margins[i].item(),
             'strong_mismatch': strong[i].item()}
            for i in range(want.shape[0])
        ],
    }


@torch.inference_mode()
def reference_stage(args):
    from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM
    from validate_qwen3_5_hf import build_prompts, CODE_PROMPT_SOURCE

    out = Path(args.reference_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.weights)
    prompts = build_prompts(tok, CODE_PROMPT_SOURCE.read_text())
    hf, loading = Qwen3_5MoeForCausalLM.from_pretrained(
        args.weights, dtype=torch.bfloat16, device_map='cuda',
        attn_implementation='sdpa', output_loading_info=True,
    )
    if loading.get('missing_keys') or loading.get('mismatched_keys'):
        raise RuntimeError(f'incomplete HF reference: {loading}')
    hf.eval()
    result = {'reference': 'HF BF16 cached greedy', 'model_revision': args.reference_revision,
              'tokens': args.tokens, 'code_prompt_sha256': hashlib.sha256(CODE_PROMPT_SOURCE.read_bytes()).hexdigest(),
              'attention': 'sdpa', 'mtp': False, 'cases': []}
    for name, text in prompts.items():
        ids = tok(text, return_tensors='pt').input_ids.cuda()
        step, cache, logits, generated = ids, None, [], []
        for _ in range(args.tokens):
            prediction = hf(input_ids=step, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = prediction.past_key_values
            last = prediction.logits[:, -1].float()
            if not torch.isfinite(last).all():
                raise RuntimeError('nonfinite HF logits')
            logits.append(last.cpu())
            step = last.argmax(-1, keepdim=True)
            generated.append(step.item())
        values = torch.cat(logits).numpy()
        if 'vocab_size' in result and result['vocab_size'] != values.shape[1]:
            raise RuntimeError('inconsistent reference vocabulary')
        result['vocab_size'] = values.shape[1]
        np.save(out / f'{name}.npy', values)
        result['cases'].append({'name': name, 'text': text, 'prompt_ids': ids[0].tolist(),
                                'generated_ids': generated, 'prefix': ids.shape[1]})
        save(out / 'reference.json', result)
        print(f'HF {name}: prefix={ids.shape[1]}, steps={args.tokens}', flush=True)
        del prediction, cache, logits, ids, step, last
    write_replay(out / 'replay.bin', result, tok.get_vocab())


def compare_stage(args):
    ref_dir, out = Path(args.reference_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ref_dir / 'reference.json').read_text())
    command = [args.llama_bin, args.gguf, str(ref_dir / 'replay.bin'), str(ref_dir / 'actual.bin'),
               str(args.batch), str(args.ubatch), str(args.threads), str(args.gpu_layers)]
    save(out / 'command.json', command)
    with (out / 'llama.stdout').open('w') as stdout, (out / 'llama.stderr').open('w') as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
    # Preserve inputs and the HF continuation independently of temporary full-logit buffers.
    save(out / 'reference.json', reference)
    rows = []
    with (ref_dir / 'actual.bin').open('rb') as actual:
        if read_exact(actual, 8) != b'IOQ8LG01':
            raise ValueError('bad llama.cpp output magic')
        vocab, count = read_u32(actual), read_u32(actual)
        if vocab != reference['vocab_size'] or count != len(reference['cases']):
            raise ValueError('llama.cpp output dimensions differ from HF reference')
        for case in reference['cases']:
            name, steps = read_string(actual), read_u32(actual)
            if name != case['name'] or steps != len(case['generated_ids']):
                raise ValueError('llama.cpp case/order/step mismatch')
            got = np.frombuffer(read_exact(actual, steps * vocab * 4), dtype='<f4').copy().reshape(steps, vocab)
            want = np.load(ref_dir / f'{name}.npy', allow_pickle=False)
            row = {'precision': 'llamacpp_q8_0', 'prompt': name, 'prefix_tokens': case['prefix'],
                   **distribution_metrics(want, got)}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'per_step'}), flush=True)
            save(out / 'quality.json', {'rows': rows, 'reference': reference['reference'],
                 'reference_revision': reference['model_revision'], 'tokens': reference['tokens'],
                 'strong_mismatch_margin': 0.5, 'flash_attention': True,
                 'cache_k': 'f16', 'cache_v': 'f16', 'mtp': False,
                 'scope': 'teacher-forced numerical distribution comparison, not task accuracy'})
        if actual.read(1):
            raise ValueError('trailing llama.cpp output bytes')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['reference', 'compare'])
    p.add_argument('--reference-dir', required=True)
    p.add_argument('--weights')
    p.add_argument('--reference-revision', default='59d61f3ce65a6d9863b86d2e96597125219dc754')
    p.add_argument('--tokens', type=int, default=48)
    p.add_argument('--out')
    p.add_argument('--llama-bin')
    p.add_argument('--gguf')
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--ubatch', type=int, default=512)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--gpu-layers', type=int, default=99)
    args = p.parse_args()
    if args.tokens < 1 or min(args.batch, args.ubatch, args.threads) < 1:
        p.error('tokens/batch/ubatch/threads must be positive')
    if args.stage == 'reference' and not args.weights:
        p.error('reference requires --weights')
    if args.stage == 'compare' and not all((args.out, args.llama_bin, args.gguf)):
        p.error('compare requires --out, --llama-bin and --gguf')
    torch.set_num_threads(args.threads)
    (reference_stage if args.stage == 'reference' else compare_stage)(args)


if __name__ == '__main__':
    main()
