"""Accuracy of the fixed-shape decode path (eager / torch.compile / CUDA graph) against Hugging Face.

Same metrics and prompts as validate_qwen3_5_hf.py, but only the decode path is under test:
  * token-by-token logits of the last 32 prompt positions vs HF (KL, top-1, max |dlogit|)
  * 192-token greedy generation: first free-run mismatch with HF, and next-token agreement when fed HF's tokens
    (mismatch margins recorded; margin <= 0.5 counts as a near-tie)
The prompt is always prefilled by the ordinary eager path; the decoder under test only runs the single-token steps.
A config is ``kernels:mode`` or ``kernels:mode:quant`` with quant in bf16 / int8 / int8+lm_head (weight-only INT8, see
infer_opt.quant; ``--group-size 128`` for one scale per 128 input columns). Quantization is in place and one way,
so quant levels must appear in that order. ``--autotune`` runs the quantized configs with the options of
torch.compile(mode="max-autotune") (max_autotune and coordinate descent tuning), without which the int8 -> bf16 conversion is not fused and decode is several times slower than bf16.
``--dtype float16`` runs our model (control and quantized) in fp16 against the bf16 HF reference, as on GPUs without
bf16; ``--int8-from`` takes the int8 configs from a checkpoint written by scripts/quantize_qwen3_5.py (a directory or a
Hub repo) instead of quantizing in place, so the published artifact itself is checked.
At the end the script evaluates the quantization gate against the bf16 config with the same kernels:mode in this run:
mean decode KL <= ``--kl-ratio`` x the bf16 mean, and zero strong teacher-forced mismatches.
``sliced-eager`` is the original decode path and serves as the baseline in the same run.
Each (kernels, mode) gets its own cache and decoder (one compile / capture), loops over the prompts, and reports
torch._dynamo unique graphs so a silent fall-back to eager after too many recompiles would show.

    python scripts/validate_qwen3_5_decode_modes.py --out decode_modes.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from infer_opt.quant import quantize_linears  # noqa: E402
from infer_opt.qwen3_5 import FixedShapeDecoder, Qwen3_5ForCausalLM  # noqa: E402
from validate_qwen3_5_hf import TIE_MARGIN, build_prompts, kl_and_top1  # noqa: E402

DEFAULT_CONFIGS = (
    "torch:sliced-eager,torch:eager,torch:compile,torch:compile-reduce-overhead,torch:cudagraph,"
    "fla-delta:sliced-eager,fla-delta:cudagraph,fla-delta:compile-reduce-overhead,"
    "fla:sliced-eager,fla:cudagraph,fla:compile-reduce-overhead"
)

QUANT_LEVELS = ("bf16", "int8", "int8+lm_head")


def quantization_gate(rows: list[dict], kl_ratio: float) -> list[dict]:
    """Per quantized config: mean decode KL against the bf16 config with the same kernels and mode, strong mismatches."""
    def stats(kernels, mode, quant):
        mine = [r for r in rows if not r.get("summary") and (r["kernels"], r["mode"], r.get("quant", "bf16")) == (kernels, mode, quant)]
        kls = [r["decode_kl_vs_hf"] for r in mine if "decode_kl_vs_hf" in r]
        return {"prompts": len(mine), "mean_kl": sum(kls) / len(kls) if kls else None,
                "strong_mismatches": sum(r["tf_mismatch_strong"] for r in mine),
                "near_tie_mismatches": sum(r["tf_mismatch_near_tie"] for r in mine),
                "free_run_identical": sum(r["gen_first_mismatch"] is None for r in mine)}

    gates = []
    for kernels, mode, quant in dict.fromkeys((r["kernels"], r["mode"], r.get("quant", "bf16")) for r in rows if not r.get("summary")):
        if quant == "bf16":
            continue
        base, got = stats(kernels, mode, "bf16"), stats(kernels, mode, quant)
        if base["mean_kl"] is None or got["mean_kl"] is None:
            gates.append({"config": f"{kernels}:{mode}:{quant}", "error": "no bf16 control with the same kernels:mode"})
            continue
        ratio = got["mean_kl"] / base["mean_kl"]
        gates.append({"config": f"{kernels}:{mode}:{quant}", "bf16": base, "quantized": got, "kl_ratio": ratio,
                      "pass_kl": ratio <= kl_ratio, "pass_strong": got["strong_mismatches"] == 0,
                      "pass": ratio <= kl_ratio and got["strong_mismatches"] == 0})
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--new-tokens", type=int, default=192)
    parser.add_argument("--decode-positions", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS, help="comma separated kernels:mode[:quant]")
    parser.add_argument("--autotune", action="store_true", help="max-autotune inductor options for the quantized configs")
    parser.add_argument("--group-size", type=int, default=0, help="INT8 scale per this many input columns; 0: per channel")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"], help="dtype of our model")
    parser.add_argument("--int8-from", default=None, help="quantized checkpoint dir or Hub repo for the int8 configs")
    parser.add_argument("--kl-ratio", type=float, default=2.0, help="gate: quantized mean KL / bf16 mean KL")
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--out", default="decode_modes.json")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from torch._dynamo.utils import counters
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = args.weights or snapshot_download(args.repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    tok = AutoTokenizer.from_pretrained(path)
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    prompts = build_prompts(tok, (ROOT / "infer_opt" / "qwen3_5.py").read_text())
    if args.prompts:
        prompts = {k: prompts[k] for k in args.prompts.split(",")}

    # HF reference for every prompt, computed once
    refs = {}
    with torch.no_grad():
        for name, text in prompts.items():
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            T = ids.shape[1]
            split = max(T - args.decode_positions, 1)
            hf_gen = hf.generate(ids, max_new_tokens=args.new_tokens, do_sample=False)
            full = hf(input_ids=hf_gen).logits.float()  # one pass over prompt + generation
            top2 = full[0, T - 1 : -1].topk(2, -1).values
            refs[name] = {
                "ids": ids, "T": T, "split": split, "new_ref": hf_gen[:, T:],
                "hf_decode_logits": full[:, split:T] if split < T else None,
                "margins": (top2[:, 0] - top2[:, 1]).cpu(),
            }
            del full
    del hf
    torch.cuda.empty_cache()
    # loaded after the HF reference is freed so two copies of a large model never share the GPU
    dtype = getattr(torch, args.dtype)
    ours = Qwen3_5ForCausalLM.from_pretrained_dir(path, dtype=dtype, device="cuda")
    print(f"{torch.cuda.get_device_name()} torch {torch.__version__} prompts {list(prompts)}", flush=True)

    rows = []
    out = Path(args.out)

    quant_done = "bf16"
    quant_reports = {}

    def save():
        out.write_text(json.dumps({"args": vars(args), "layers": ours.config.layer_summary(), "quantization": quant_reports,
                                   "gate": quantization_gate(rows, args.kl_ratio), "rows": rows}, indent=1))

    import torch._inductor.config as inductor_config

    for config in args.configs.split(","):
        kernels, mode, *rest = config.split(":")
        quant = rest[0] if rest else "bf16"
        if quant not in QUANT_LEVELS:
            raise SystemExit(f"quant must be one of {QUANT_LEVELS}, got {quant!r}")
        if QUANT_LEVELS.index(quant) < QUANT_LEVELS.index(quant_done):
            raise SystemExit(f"{config}: the model is already {quant_done}; list configs in order {QUANT_LEVELS}")
        if quant != quant_done and args.int8_from:
            if quant != "int8":
                raise SystemExit("--int8-from provides the int8 config only")
            kernels_now = ours.config.kernels
            del ours
            torch.cuda.empty_cache()
            src = args.int8_from if Path(args.int8_from).is_dir() else snapshot_download(args.int8_from)
            ours = Qwen3_5ForCausalLM.from_quantized_dir(src, device="cuda", kernels=kernels_now, dtype=dtype)
            quant_reports[quant] = {"from": args.int8_from}
            print(quant, "loaded from", args.int8_from, flush=True)
            quant_done = quant
        elif quant != quant_done:  # one way, in place; lm_head is the only addition from int8 to int8+lm_head
            report = quantize_linears(ours, include_lm_head=quant == "int8+lm_head", group_size=args.group_size or None)
            quant_reports[quant] = {"summary": report.summary(), "replaced": len(report.replaced),
                                    "group_size": args.group_size or None}
            print(quant, report.summary(), flush=True)
            quant_done = quant
            torch.cuda.empty_cache()
        torch._dynamo.reset()
        counters.clear()
        tuned = args.autotune and quant != "bf16"
        inductor_config.max_autotune = tuned
        inductor_config.coordinate_descent_tuning = tuned
        ours.set_kernels(kernels)
        cache = ours.new_cache(1, args.max_len)
        t0 = time.time()
        decoder = None
        try:
            if mode == "sliced-eager":
                step, decoder = (lambda token: ours(token, cache)), None
            else:
                decoder = FixedShapeDecoder(ours, cache, mode)
                step = decoder.step
                decoder.prepare()  # cudagraph: capture once, on the empty cache

            def prefill(ids):
                cache.reset()
                return ours(ids, cache)

            with torch.no_grad():
                for name, ref in refs.items():
                    ids, T, split, new_ref = ref["ids"], ref["T"], ref["split"], ref["new_ref"]
                    n_new = new_ref.shape[1]
                    row = {"kernels": kernels, "mode": mode, "quant": quant, "prompt": name, "tokens": T}

                    if split < T:
                        prefill(ids[:, :split])
                        got = torch.cat([step(ids[:, i : i + 1]).float() for i in range(split, T)], dim=1)
                        kl, top1, maxd = kl_and_top1(ref["hf_decode_logits"], got)
                        row.update(decode_kl_vs_hf=kl, decode_top1_vs_hf=top1, decode_maxdiff_vs_hf=maxd)

                    # teacher-forced on HF's tokens
                    token = prefill(ids)[:, -1].argmax(-1, keepdim=True)
                    preds = [token]
                    for i in range(n_new - 1):
                        preds.append(step(new_ref[:, i : i + 1])[:, -1].argmax(-1, keepdim=True))
                    wrong = (torch.cat(preds, 1)[0] != new_ref[0]).cpu()
                    margins = ref["margins"][wrong].tolist()

                    # free run
                    token = prefill(ids)[:, -1].argmax(-1, keepdim=True)
                    gen = [token]
                    for _ in range(n_new - 1):
                        token = step(token)[:, -1].argmax(-1, keepdim=True)
                        gen.append(token)
                    same = (torch.cat(gen, 1)[0] == new_ref[0]).cpu()
                    first = None if bool(same.all()) else int((~same).nonzero()[0])
                    near_tie = sum(m <= TIE_MARGIN for m in margins)
                    row.update(
                        new_tokens_compared=n_new, gen_first_mismatch=first,
                        tf_mismatch_steps=int(wrong.sum()), tf_mismatch_near_tie=near_tie,
                        tf_mismatch_strong=int(wrong.sum()) - near_tie,
                        tf_mismatch_hf_margins=[round(m, 3) for m in margins],
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)

            summary = {"kernels": kernels, "mode": mode, "quant": quant, "summary": True, "seconds": time.time() - t0,
                       "dynamo_unique_graphs": counters["stats"].get("unique_graphs"),
                       "dynamo_graph_breaks": sum(counters["graph_break"].values())}
        except Exception as exc:  # noqa: BLE001 - record and go on with the next config
            summary = {"kernels": kernels, "mode": mode, "quant": quant, "summary": True, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
            print(traceback.format_exc()[-1200:], flush=True)
        rows.append(summary)
        print(json.dumps(summary), flush=True)
        save()
        cache = decoder = step = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
