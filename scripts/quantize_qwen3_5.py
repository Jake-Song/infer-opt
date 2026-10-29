"""Quantize a Qwen3.5 checkpoint to weight-only INT8 (or INT4) and save it for Qwen3_5ForCausalLM.from_quantized_dir.

Loads the bf16 Hugging Face checkpoint on the GPU, quantizes every Linear except the small delta-rule gates
(group-wise scales by default; lm_head stays bf16 unless --lm-head), writes safetensors shards and
quantization.json, and copies config.json and the tokenizer files next to them. It then reloads the directory and
checks that the logits of a short prompt are identical to those of the in-memory quantized model. With
--push-to-hub it then writes a model card and uploads the directory to that Hugging Face repo (needs a write token).

    python scripts/quantize_qwen3_5.py --repo Qwen/Qwen3.5-9B --out qwen3_5_9b_int8_g128 --push-to-hub user/name

``--bits 4`` writes group-wise asymmetric 4-bit weights (two per byte, a bf16 scale and offset per group);
``--clip-search`` picks each group's range by the smallest weight error instead of its min and max.
``--gptq`` quantizes to 4-bit with GPTQ (infer_opt.gptq) on ``--calibration-samples`` UltraChat conversations
(the same 512-token samples as the NVFP4 calibration), layer by layer; the checkpoint format is the same.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer_opt.quant import quantize_linears, save_quantized  # noqa: E402
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

COPY = ("LICENSE", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "chat_template.jinja")
PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.jinja", "LICENSE"]

CARD = """---
license: apache-2.0
base_model: {source}
base_model_relation: quantized
tags:
- int{bits}
- weight-only
- qwen3.5
---

# {name}

Weight-only INT{bits} (W{bits}A16) version of [{source}](https://huggingface.co/{source}): every Linear of the text model
except {kept_note} stores int{bits} weights with {scale_note}. Activations, embeddings,
norms and the delta-rule parameters stay in bf16. The vision tower and the multi-token-prediction head of the
original checkpoint are not included.

**This is not a `transformers` checkpoint.** It is loaded by
[infer-opt](https://github.com/Jake-Song/infer-opt), whose Triton kernel runs the one-token group-wise matmul:

```python
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM
from huggingface_hub import snapshot_download

model = Qwen3_5ForCausalLM.from_quantized_dir(snapshot_download("{repo_id}"), device="cuda", kernels="fla-delta")
```

{summary}

{measured}License: Apache-2.0, as the original model (see `LICENSE`).
"""


# model card section per source repo (left out for others): measured with this repository's scripts
MEASURED = {
    "Qwen/Qwen3.5-9B": """## Measured (Qwen3.5-9B, A100-SXM4-40GB, batch 1)

Decode with torch.compile + CUDA graphs (`fla-delta:compile-reduce-overhead`), 1024-token prompt, mean of 128 steps.
KL is against the bf16 model, teacher-forced over 48 steps.

| | ms/token | KL vs bf16 |
|---|---:|---:|
| bf16 | 15.09 | – |
| group-128 int8, lm_head bf16 (this repo) | 10.79 | 0.00051 |
| group-128 int8 incl. lm_head | 10.15 | 0.00057 |
| per-channel int8 incl. lm_head (for comparison) | 8.92 | 0.00092 |

Against Hugging Face `transformers` on 9 prompts, the mean decode KL of this configuration was 1.55-1.89x that of
the bf16 model in the same run, with no confident (margin > 0.5) teacher-forced mismatch. Per-channel INT8 was 2.14x
with one. These are single runs; see the repository's notebook `qwen3_5_9b_benchmark_tables.ipynb` (§5).

""",
    "Qwen/Qwen3.5-27B": """## Measured (Qwen3.5-27B, A100-SXM4-80GB, batch 1)

Decode with torch.compile + CUDA graphs (`fla-delta:compile-reduce-overhead`), 1024-token prompt, mean of 128 steps.
KL is against the bf16 model, teacher-forced over 48 steps.

| | ms/token | peak GiB | KL vs bf16 |
|---|---:|---:|---:|
| bf16 | 37.65 | 50.9 | – |
| group-128 int8, lm_head bf16 (this repo) | 25.93 | 28.6 | 0.00038 |
| group-128 int8 incl. lm_head | 25.22 | 31.7 | 0.00036 |

Against Hugging Face `transformers` on 9 prompts, the mean decode KL of this configuration was 0.53-0.85x that of
the bf16 model in the same run (three runs, one of them on this saved checkpoint), with no confident (margin > 0.5)
teacher-forced mismatch. The mean is dominated by one code prompt whose KL is high for bf16 too. The int8 weights
need about 29 GiB, so it also runs on a 40 GB GPU. See the repository's `worklog.md` (2026-10-20).

""",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--weights", default=None, help="local checkpoint dir instead of downloading --repo")
    parser.add_argument("--group-size", type=int, default=128, help="0: one scale per output channel")
    parser.add_argument("--lm-head", action="store_true", help="quantize lm_head too")
    parser.add_argument("--bits", type=int, default=8, choices=[8, 4])
    parser.add_argument("--clip-search", action="store_true", help="4-bit: per-group range with the smallest error")
    parser.add_argument("--gptq", action="store_true", help="4-bit with GPTQ on calibration data")
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--out", required=True)
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID", help="upload to this (public) model repo")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    src = Path(args.weights or snapshot_download(args.repo, allow_patterns=PATTERNS))
    out = Path(args.out)
    model = Qwen3_5ForCausalLM.from_pretrained_dir(src, dtype=torch.bfloat16, device="cuda")
    out.mkdir(parents=True, exist_ok=True)
    if args.gptq:
        if args.bits != 4 or args.lm_head or args.clip_search or not args.group_size:
            raise SystemExit("--gptq is 4-bit group-wise, without --lm-head or --clip-search")
        from transformers import AutoTokenizer

        from bench_qwen3_5_nvfp4 import calibration_inputs
        from infer_opt.gptq import gptq_quantize_qwen3_5

        calibration = calibration_inputs(AutoTokenizer.from_pretrained(src), out, args.calibration_samples)
        t0 = time.perf_counter()
        report = gptq_quantize_qwen3_5(model.set_kernels("fla-delta"), calibration, group_size=args.group_size,
                                       log=lambda line: print(line, flush=True))
        model.set_kernels("torch")  # as loaded, so the reload check compares like with like
        print(f"gptq {time.perf_counter() - t0:.0f} s", flush=True)
    else:
        report = quantize_linears(model, include_lm_head=args.lm_head, group_size=args.group_size or None,
                                  bits=args.bits, clip_search=args.clip_search)
    print(report.summary(), flush=True)

    t0 = time.perf_counter()
    files = save_quantized(model, out)
    copied = []
    for name in COPY:
        if (src / name).exists():
            shutil.copy(src / name, out / name)
            copied.append(name)
    (out / "quantize_info.json").write_text(json.dumps({
        "source": args.repo if not args.weights else str(src), "group_size": args.group_size or None,
        "bits": args.bits, "clip_search": args.clip_search, "gptq": args.gptq,
        **({"calibration": json.loads((out / "calibration.json").read_text())} if args.gptq else {}),
        "lm_head_quantized": args.lm_head, "summary": report.summary(), "kept": report.skipped,
        "load": "Qwen3_5ForCausalLM.from_quantized_dir(path, device='cuda', kernels='fla-delta')",
    }, indent=1))
    size = sum(f.stat().st_size for f in files)
    print(f"wrote {len(files)} shards, {size / 2**30:.2f} GiB, in {time.perf_counter() - t0:.0f} s; copied {copied}",
          flush=True)

    torch.manual_seed(0)
    ids = torch.randint(1000, 100000, (1, 64), device="cuda")
    with torch.no_grad():
        expected = model(ids)
        del model
        torch.cuda.empty_cache()
        loaded = Qwen3_5ForCausalLM.from_quantized_dir(out, device="cuda")
        same = torch.equal(loaded(ids), expected)
    print(f"reloaded: logits identical = {same}", flush=True)
    if not same:
        raise SystemExit("reloaded model differs from the quantized model")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        group = args.group_size or None
        gates = loaded.config.linear_num_value_heads  # in_proj_a / in_proj_b: too small to quantize
        if args.bits == 4:
            scale_note = (f"one bf16 scale and offset per output channel and {group} input columns (asymmetric, "
                          + (f"GPTQ on {args.calibration_samples} UltraChat conversations of up to 512 tokens, no act-order)"
                                if args.gptq else
                             "per-group range chosen by the smallest weight error)" if args.clip_search else "min-max)"))
        else:
            scale_note = (f"one bf16 scale per output channel and {group} input columns" if group
                          else "one bf16 scale per output channel") + ", symmetric absmax, no clipping"
        (out / "README.md").write_text(CARD.format(
            bits=args.bits, source=args.repo, name=args.push_to_hub.split("/")[-1], repo_id=args.push_to_hub,
            kept_note=f"lm_head and the {gates}-wide delta-rule gate projections" if not args.lm_head
            else f"the {gates}-wide delta-rule gate projections",
            scale_note=scale_note, summary=f"Quantization: {report.summary()}.",
            measured=MEASURED.get(args.repo, "") if args.bits == 8 else ""))
        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(repo_id=args.push_to_hub, folder_path=str(out), repo_type="model",
                                   commit_message=f"Upload {args.repo} weight-only INT{args.bits} (group {group})")
        print(f"uploaded to https://huggingface.co/{args.push_to_hub} ({commit.oid[:12]})", flush=True)


if __name__ == "__main__":
    main()
