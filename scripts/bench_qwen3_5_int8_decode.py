"""Decode ms/token and numerics of the INT8 weight-only model against the bf16 model (A100).

One process, one model load, configurations in this order (the quantization is in place and one-way):

  bf16            the control, measured in the same session
  int8            quantize_linears(model)                    -> every Linear except lm_head and the 32-wide gates
  int8+lm_head    quantize_linears(model, include_lm_head=True)

Every configuration runs the fixed-shape decode through torch.compile with CUDA graphs. ``--autotune`` turns on
the options of torch.compile(mode="max-autotune"): max_autotune and coordinate_descent_tuning (bench_int8_linear.py
measured the fused int8 kernel under that mode; with max_autotune alone the int8 decode was 5x slower than bf16).
Numerics are teacher-forced against the bf16 sliced-eager logits that are recorded before anything is quantized:
max |dlogit|, argmax agreement and the mean KL(bf16 || quantized) over the checked steps.

    python scripts/bench_qwen3_5_int8_decode.py --repo Qwen/Qwen3.5-9B --kernels fla-delta --autotune --out int8_decode.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_qwen3_5_decode import make_stepper, prefilled, reference, sync, time_decode  # noqa: E402
from infer_opt.quant import quantize_linears  # noqa: E402
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402


@torch.no_grad()
def check_numerics(model, ids, max_len, mode, fed, ref_logits):
    cache, _ = prefilled(model, ids, max_len)
    step, prepare = make_stepper(model, cache, mode)
    prepare()
    max_diff, agree, kl = 0.0, 0, 0.0
    for token, want in zip(fed, ref_logits):
        got = step(token).float()
        max_diff = max(max_diff, (got - want).abs().max().item())
        agree += int((got.argmax(-1) == want.argmax(-1)).all())
        kl += F.kl_div(F.log_softmax(got[:, -1], -1), F.log_softmax(want[:, -1], -1), log_target=True,
                       reduction="batchmean").item()
    return {"logits_max_abs_diff": max_diff, "argmax_agreement": agree / len(fed), "mean_kl_vs_bf16": kl / len(fed)}


def measure(model, label, args, ids, max_len, fed, ref_logits, autotune):
    import torch._inductor.config as inductor_config
    from torch._dynamo.utils import counters

    row = {"config": label, "mode": args.mode, "autotune": autotune}
    try:
        inductor_config.max_autotune = autotune
        inductor_config.coordinate_descent_tuning = autotune
        torch._dynamo.reset()
        counters.clear()
        torch.cuda.reset_peak_memory_stats()
        row.update(check_numerics(model, ids, max_len, args.mode, fed, ref_logits))
        torch._dynamo.reset()
        counters.clear()
        row.update(time_decode(model, ids, max_len, args.mode, args.new_tokens, args.warmup))
        row["graph_breaks"] = sum(counters["graph_break"].values())
        row["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    except Exception as exc:  # noqa: BLE001 - record and continue
        row["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        print(traceback.format_exc()[-1500:], flush=True)
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--kernels", default="fla-delta")
    parser.add_argument("--mode", default="compile-reduce-overhead")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--check-steps", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--autotune", action="store_true", help="max-autotune options (max_autotune + coordinate_descent_tuning) for the int8 configurations")
    parser.add_argument("--bf16-autotune", action="store_true", help="also measure bf16 with max_autotune")
    parser.add_argument("--configs", default="bf16,int8,int8+lm_head")
    parser.add_argument("--out", default="int8_decode.json")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    path = args.weights or snapshot_download(args.repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    model = Qwen3_5ForCausalLM.from_pretrained_dir(path, dtype=torch.bfloat16, device="cuda", kernels=args.kernels)
    torch.manual_seed(0)
    ids = torch.randint(1000, 100000, (1, args.prompt_tokens), device="cuda")
    max_len = args.prompt_tokens + args.warmup + args.new_tokens + args.check_steps + 8
    print(f"{torch.cuda.get_device_name()} torch {torch.__version__} kernels {args.kernels} mode {args.mode} "
          f"prompt {args.prompt_tokens} new {args.new_tokens}", flush=True)

    fed, ref_logits = reference(model, ids, max_len, args.check_steps)  # bf16 sliced-eager, before quantizing
    wanted = args.configs.split(",")
    rows, reports = [], {}
    out = Path(args.out)

    def save():
        out.write_text(json.dumps({"args": vars(args), "device": torch.cuda.get_device_name(), "torch": torch.__version__,
                                   "layers": model.config.layer_summary(), "quantization": reports, "rows": rows}, indent=1))

    if "bf16" in wanted:
        rows.append(measure(model, "bf16", args, ids, max_len, fed, ref_logits, autotune=False))
        save()
        if args.bf16_autotune:
            rows.append(measure(model, "bf16", args, ids, max_len, fed, ref_logits, autotune=True))
            save()
    for label, kwargs in (("int8", {}), ("int8+lm_head", {"include_lm_head": True})):
        if label not in wanted:
            continue
        t0 = time.perf_counter()
        report = quantize_linears(model, **kwargs)
        sync()
        reports[label] = {"summary": report.summary(), "seconds": time.perf_counter() - t0,
                          "replaced": len(report.replaced), "skipped": report.skipped}
        print(label, report.summary(), flush=True)
        torch.cuda.empty_cache()
        rows.append(measure(model, label, args, ids, max_len, fed, ref_logits, autotune=args.autotune))
        save()


if __name__ == "__main__":
    main()
