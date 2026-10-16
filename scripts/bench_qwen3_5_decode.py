"""Decode speed and numerics: sliced eager vs fixed-shape eager / torch.compile / CUDA graph (A100).

For each (kernels, mode) it reports
  * steady-state ms/token of free-running greedy decode after a warm-up (compile / capture time reported apart)
  * numerics against the sliced-eager decode: both fed the same tokens (teacher-forced), max |dlogit|,
    and whether the argmax of every step agrees
  * torch._dynamo counters (unique graphs, graph breaks) for the compiled modes
A mode that fails is recorded with its error instead of aborting the run.

    pip install -U "transformers>=5" safetensors huggingface_hub flash-linear-attention
    python scripts/bench_qwen3_5_decode.py --prompt-tokens 1024 --new-tokens 128 --out bench_decode.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infer_opt.qwen3_5 import DECODE_MODES, FixedShapeDecoder, Qwen3_5ForCausalLM  # noqa: E402

MODES = ("sliced-eager",) + DECODE_MODES


def sync() -> None:
    torch.cuda.synchronize()


@torch.no_grad()
def prefilled(model, ids, max_len):
    cache = model.new_cache(ids.shape[0], max_len)
    first = model(ids, cache)[:, -1].argmax(-1, keepdim=True)
    return cache, first


@torch.no_grad()
def make_stepper(model, cache, mode):
    """Returns step(token) -> logits [B, 1, V]; sliced-eager is the pre-existing decode path."""
    if mode == "sliced-eager":
        return lambda token: model(token, cache), lambda: None
    decoder = FixedShapeDecoder(model, cache, mode)
    return decoder.step, decoder.prepare


@torch.no_grad()
def reference(model, ids, max_len, steps):
    """Sliced-eager greedy decode: the tokens fed at every step and the logits they produced."""
    cache, token = prefilled(model, ids, max_len)
    fed, logits = [], []
    for _ in range(steps):
        fed.append(token)
        out = model(token, cache).float()
        logits.append(out)
        token = out[:, -1].argmax(-1, keepdim=True)
    return fed, logits


@torch.no_grad()
def check_numerics(model, ids, max_len, mode, fed, ref_logits):
    cache, _ = prefilled(model, ids, max_len)
    step, prepare = make_stepper(model, cache, mode)
    prepare()
    max_diff, agree = 0.0, 0
    for token, want in zip(fed, ref_logits):
        got = step(token).float()
        max_diff = max(max_diff, (got - want).abs().max().item())
        agree += int((got.argmax(-1) == want.argmax(-1)).all())
    return {"logits_max_abs_diff": max_diff, "argmax_agreement": agree / len(fed)}


@torch.no_grad()
def time_decode(model, ids, max_len, mode, steps, warmup):
    cache, token = prefilled(model, ids, max_len)
    step, prepare = make_stepper(model, cache, mode)
    sync()
    t0 = time.perf_counter()
    prepare()
    sync()
    prepare_s = time.perf_counter() - t0
    first_s = None
    for i in range(warmup):
        sync()
        t = time.perf_counter()
        token = step(token)[:, -1].argmax(-1, keepdim=True)
        sync()
        if i == 0:
            first_s = time.perf_counter() - t  # includes torch.compile for the compiled modes
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        token = step(token)[:, -1].argmax(-1, keepdim=True)
    sync()
    return {"prepare_s": prepare_s, "first_step_s": first_s, "ms_per_token": (time.perf_counter() - t0) / steps * 1e3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--check-steps", type=int, default=48)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--kernels", default="torch", help="comma separated: torch,fla-delta,fla")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", default="bench_decode.json")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    path = args.weights or snapshot_download(args.repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    model = Qwen3_5ForCausalLM.from_pretrained_dir(path, dtype=torch.bfloat16, device="cuda")
    torch.manual_seed(0)
    ids = torch.randint(1000, 100000, (args.batch, args.prompt_tokens), device="cuda")
    max_len = args.prompt_tokens + args.warmup + args.new_tokens + args.check_steps + 8
    print(f"{torch.cuda.get_device_name()} torch {torch.__version__} batch {args.batch} prompt {args.prompt_tokens} "
          f"new {args.new_tokens}", flush=True)

    rows = []
    out = Path(args.out)
    for kernels in args.kernels.split(","):
        model.set_kernels(kernels)
        fed, ref_logits = reference(model, ids, max_len, args.check_steps)
        for mode in args.modes.split(","):
            row = {"kernels": kernels, "mode": mode}
            try:
                from torch._dynamo.utils import counters

                torch._dynamo.reset()
                counters.clear()
                row.update(check_numerics(model, ids, max_len, mode, fed, ref_logits))
                torch._dynamo.reset()
                counters.clear()
                row.update(time_decode(model, ids, max_len, mode, args.new_tokens, args.warmup))
                if mode.startswith("compile"):
                    row["dynamo_stats"] = {k: v for k, v in counters["stats"].items()}
                    row["graph_breaks"] = sum(counters["graph_break"].values())
            except Exception as exc:  # noqa: BLE001 - record and continue
                row["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
                print(traceback.format_exc()[-1500:], flush=True)
            rows.append(row)
            print(json.dumps(row), flush=True)
            out.write_text(json.dumps({"args": vars(args), "device": torch.cuda.get_device_name(),
                                       "torch": torch.__version__, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
