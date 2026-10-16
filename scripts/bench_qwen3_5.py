"""Prefill / decode benchmark for infer_opt.qwen3_5: torch vs FLA kernels, dynamic vs static cache.

    pip install -U "transformers>=5" safetensors huggingface_hub flash-linear-attention
    python scripts/bench_qwen3_5.py --prompt-tokens 1024 --new-tokens 128 --out bench.json
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


def timed(fn, repeats: int) -> float:
    fn()  # warmup
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    dtype = getattr(torch, args.dtype)
    path = args.weights or snapshot_download(args.repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    model = Qwen3_5ForCausalLM.from_pretrained_dir(path, dtype=dtype, device="cuda")
    torch.manual_seed(0)
    ids = torch.randint(1000, 100000, (args.batch, args.prompt_tokens), device="cuda")
    max_len = args.prompt_tokens + args.new_tokens
    print(f"{torch.cuda.get_device_name()} torch {torch.__version__} dtype {args.dtype} "
          f"batch {args.batch} prompt {args.prompt_tokens} new {args.new_tokens}")

    rows = []
    reference_tokens = None
    for kernels in ("torch", "fla"):
        model.set_kernels(kernels)
        for static in (False, True):
            def new_cache():
                return model.new_cache(args.batch, max_len if static else None)

            @torch.no_grad()
            def prefill():
                model(ids, new_cache())

            @torch.no_grad()
            def decode():
                cache = new_cache()
                logits = model(ids, cache)
                token = logits[:, -1].argmax(-1, keepdim=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.new_tokens):
                    token = model(token, cache)[:, -1].argmax(-1, keepdim=True)
                torch.cuda.synchronize()
                return (time.perf_counter() - start) / args.new_tokens

            torch.cuda.reset_peak_memory_stats()
            prefill_s = timed(prefill, args.repeats)
            decode()  # warmup
            decode_s = min(decode() for _ in range(args.repeats))
            tokens = model.generate_greedy(ids[:1], 16, static_cache=static)
            reference_tokens = tokens if reference_tokens is None else reference_tokens
            row = {
                "kernels": kernels, "static_cache": static,
                "prefill_ms": prefill_s * 1e3,
                "prefill_tok_per_s": args.batch * args.prompt_tokens / prefill_s,
                "decode_ms_per_token": decode_s * 1e3,
                "decode_tok_per_s": args.batch / decode_s,
                "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                "greedy16_matches_torch_dynamic": bool(torch.equal(tokens, reference_tokens)),
            }
            rows.append(row)
            print(json.dumps(row))

    if not args.no_hf:
        from transformers import AutoModelForCausalLM

        hf = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).cuda().eval()

        @torch.no_grad()
        def hf_prefill():
            hf(input_ids=ids)

        @torch.no_grad()
        def hf_generate(n):
            hf.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)

        prefill_s = timed(hf_prefill, args.repeats)
        t_full = timed(lambda: hf_generate(args.new_tokens), args.repeats)
        t_one = timed(lambda: hf_generate(1), args.repeats)
        decode_s = (t_full - t_one) / (args.new_tokens - 1)
        row = {"kernels": "hf-default", "static_cache": False, "prefill_ms": prefill_s * 1e3,
               "prefill_tok_per_s": args.batch * args.prompt_tokens / prefill_s,
               "decode_ms_per_token": decode_s * 1e3, "decode_tok_per_s": args.batch / decode_s}
        rows.append(row)
        print(json.dumps(row))

    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "device": torch.cuda.get_device_name(),
                                              "torch": torch.__version__, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
