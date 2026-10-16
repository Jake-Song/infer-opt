"""Compare infer_opt.qwen3_5 (copied HF code) against the Hugging Face Qwen3.5 model.

Needs transformers>=5 (qwen3_5 landed in 5.x):
    pip install -U "transformers>=5" safetensors huggingface_hub
Colab A100:
    python scripts/compare_qwen3_5_hf.py --device cuda --dtype bfloat16
CPU sanity check (fp32, differences should be ~1e-5):
    python scripts/compare_qwen3_5_hf.py --device cpu --dtype float32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infer_opt.qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Explain why the sky is blue in one paragraph. " * 12,  # >64 tokens: several delta-rule chunks
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--weights", default=None, help="local checkpoint dir (downloaded if omitted)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"], help="HF attention implementation")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--out", default=None, help="write JSON results here")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    path = args.weights or snapshot_download(args.repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    tok = AutoTokenizer.from_pretrained(path)

    ours = Qwen3_5ForCausalLM.from_pretrained_dir(path, dtype=dtype, device=args.device)
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation=args.attn).to(args.device).eval()
    print(f"hf class: {type(hf).__name__}, torch {torch.__version__}, device {args.device}, dtype {args.dtype}")

    results = []
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            ref_logits = hf(input_ids=ids).logits.float()
            our_logits = ours(ids).float()
        diff = (ref_logits - our_logits).abs()
        ref_top = ref_logits.argmax(-1)
        our_top = our_logits.argmax(-1)

        with torch.no_grad():
            ref_gen = hf.generate(ids, max_new_tokens=args.new_tokens, do_sample=False, use_cache=True)
        our_gen = ours.generate_greedy(ids, args.new_tokens, use_cache=True)
        our_gen_nocache = ours.generate_greedy(ids, args.new_tokens, use_cache=False)
        n = min(ref_gen.shape[1], our_gen.shape[1])
        mismatch = (ref_gen[0, :n] != our_gen[0, :n]).nonzero()
        first_mismatch = int(mismatch[0]) - ids.shape[1] if len(mismatch) else None

        # cache consistency: prefill all but the last 8 tokens, then decode the rest one token at a time
        # (the first 8 positions of the logits come from the cached path, compared with the full forward)
        split = max(ids.shape[1] - 8, 1)
        with torch.no_grad():
            cache = ours.new_cache()
            steps = [ours(ids[:, :split], cache)]
            steps += [ours(ids[:, i : i + 1], cache) for i in range(split, ids.shape[1])]
            cached_logits = torch.cat(steps, dim=1).float()
        cache_diff = (cached_logits - our_logits).abs().max().item()

        ours_cache_mib = cache.nbytes() / 2**20
        row = {
            "prompt_tokens": ids.shape[1],
            "logits_max_abs_diff": diff.max().item(),
            "logits_mean_abs_diff": diff.mean().item(),
            "logits_ref_absmax": ref_logits.abs().max().item(),
            "argmax_agreement": (ref_top == our_top).float().mean().item(),
            "greedy_tokens_equal": first_mismatch is None,
            "cached_vs_nocache_tokens_equal": bool((our_gen == our_gen_nocache).all()),
            "cached_vs_full_logits_max_abs_diff": cache_diff,
            "cache_mib": ours_cache_mib,
            "first_mismatch_new_token": first_mismatch,
        }
        results.append(row)
        print(json.dumps(row))
        if first_mismatch is not None:
            print("  hf  :", repr(tok.decode(ref_gen[0, ids.shape[1]:])))
            print("  ours:", repr(tok.decode(our_gen[0, ids.shape[1]:])))

    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "results": results}, indent=2))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.1f}s")
