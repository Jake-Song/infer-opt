"""Where the time of one speculative round goes, next to one plain decode step (A100, int8 Qwen3.5-27B + MTP head).

The model is the reference configuration: target and MTP head int8 (group 128), lm_head bf16; ``--mtp-bf16`` keeps
the MTP head in bf16, ``--lm-head-int8`` quantizes lm_head too.

Both run compiled with CUDA graphs (compile-reduce-overhead, max-autotune) on a real prompt; the CUDA kernels of
``--steps`` replays are profiled and grouped into categories by kernel name (GEMV, multi-row GEMV, bf16 matmul,
delta rule, attention, copies, the rest). Prints ms per step / per round for each category, the top kernels of the
round, and writes them to ``--out``.

    python scripts/profile_qwen3_5_spec_round.py --repo Qwen/Qwen3.5-27B --k 2 --rows-configs rows_tune.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer_opt import quant_kernels  # noqa: E402
from infer_opt.quant import quantize_linears  # noqa: E402
from infer_opt.qwen3_5 import FixedShapeDecoder, Qwen3_5ForCausalLM  # noqa: E402
from infer_opt.speculative import MTPSpeculativeDecoder  # noqa: E402
from validate_qwen3_5_hf import CODE_PROMPT_SOURCE, build_prompts  # noqa: E402

CATEGORIES = [
    ("gemv_rows", r"_w8a16_group_gemv_rows"),
    ("gemv", r"_w8a16_group_gemv_kernel"),  # before bf16_matmul: cuBLAS has gemv kernels of its own
    ("delta_rule", r"fused_recurrent|gated_delta|chunk_"),  # FLA's step kernel and _gated_delta_verify_kernel
    ("attention", r"decode_attention"),
    ("bf16_matmul", r"gemm|gemv|sm80_xmma|cutlass|ampere|triton_mm|splitK|mm_"),
    ("copy", r"fused_copy|Memcpy|memcpy|index_select|gather|scatter"),  # not "_to_copy": that is a dtype cast
    ("reduction", r"triton_red|triton_per|argmax|reduce"),
    ("pointwise", r"triton_poi"),
]


def category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if re.search(pattern, name):
            return label
    return "other"


def profile(fn, steps: int) -> dict[str, float]:
    """Total CUDA kernel time per kernel name (us) over ``steps`` calls of ``fn``."""
    from torch.profiler import ProfilerActivity, profile as torch_profile

    torch.cuda.synchronize()
    with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
    totals = defaultdict(float)
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            totals[event.name] += event.device_time
    return totals


def summarize(totals: dict[str, float], per: int) -> dict:
    by_category = defaultdict(float)
    for name, us in totals.items():
        by_category[category(name)] += us
    top = sorted(totals.items(), key=lambda kv: -kv[1])[:25]
    return {"total_ms": sum(totals.values()) / per / 1e3,
            "by_category_ms": {c: v / per / 1e3 for c, v in sorted(by_category.items(), key=lambda kv: -kv[1])},
            "top_kernels_ms": [(name[:120], us / per / 1e3) for name, us in top]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--kernels", default="fla-delta")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--prompt", default="prose_1k")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--rows-configs", default=None)
    parser.add_argument("--mtp-bf16", action="store_true", help="keep the MTP head in bf16")
    parser.add_argument("--lm-head-int8", action="store_true", help="quantize lm_head too")
    parser.add_argument("--out", default="spec_round_profile.json")
    args = parser.parse_args()

    import torch._inductor.config as inductor_config
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    inductor_config.max_autotune = True
    inductor_config.coordinate_descent_tuning = True
    if args.rows_configs:
        for m, n, k, group, config in json.loads(Path(args.rows_configs).read_text())["rows_best"]:
            quant_kernels.ROWS_LAUNCH_CONFIGS[(m, n, k, group)] = tuple(config)
    path = args.weights or snapshot_download(args.repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok(build_prompts(tok, CODE_PROMPT_SOURCE.read_text())[args.prompt], return_tensors="pt").input_ids.cuda()
    model = Qwen3_5ForCausalLM.from_pretrained_dir(path, dtype=torch.bfloat16, device="cuda", kernels=args.kernels,
                                                   mtp=True)
    model.fuse_projections()
    exclude = ("mtp.",) if args.mtp_bf16 else ()
    print(quantize_linears(model, group_size=args.group_size, exclude=exclude, include_lm_head=args.lm_head_int8)
          .summary(), flush=True)
    torch.cuda.empty_cache()
    max_len = ids.shape[1] + 4 * args.steps * (args.k + 1) + 64

    with torch.no_grad():
        cache = model.new_cache(1, max_len)
        base = FixedShapeDecoder(model, cache, "compile-reduce-overhead")
        token = model(ids, cache)[:, -1].argmax(-1, keepdim=True)
        for _ in range(3):  # compile and warm up
            token = base.step(token)[:, -1].argmax(-1, keepdim=True)
        step_totals = profile(lambda: base.step(token), args.steps)

        spec = MTPSpeculativeDecoder(model, max_len, k=args.k, mode="compile-reduce-overhead")
        spec.prefill(ids)
        for _ in range(3):
            spec.round()
        round_totals = profile(spec.round, args.steps)

    result = {"k": args.k, "prompt": args.prompt, "mtp_bf16": args.mtp_bf16, "lm_head_int8": args.lm_head_int8, "step": summarize(step_totals, args.steps),
              "round": summarize(round_totals, args.steps), "stats": spec.stats.as_dict()}
    for label in ("step", "round"):
        print(label, json.dumps(result[label]["by_category_ms"]), f"total {result[label]['total_ms']:.2f} ms", flush=True)
    for name, ms in result["round"]["top_kernels_ms"]:
        print(f"  {ms:7.3f} ms  {name}", flush=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
