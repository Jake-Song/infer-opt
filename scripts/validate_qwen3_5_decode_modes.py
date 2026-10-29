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
quant ``int4`` / ``int4-clip`` reloads the bf16 weights and quantizes them to group-wise 4-bit (``--int4-group-size``;
``-clip`` searches each group's range, see infer_opt.quant.quantize_int4); ``--int4-from`` takes ``int4`` from a
checkpoint instead.
At the end the script evaluates the quantization gate against the bf16 config with the same kernels:mode in this run:
mean decode KL <= ``--kl-ratio`` x the bf16 mean, and zero strong teacher-forced mismatches.
``--fuse`` stacks the projections that share an input (``Qwen3_5ForCausalLM.fuse_projections``) in every config.
``sliced-eager`` is the original decode path and serves as the baseline in the same run.
Each (kernels, mode) gets its own cache and decoder (one compile / capture), loops over the prompts, and reports
torch._dynamo unique graphs so a silent fall-back to eager after too many recompiles would show.

    python scripts/validate_qwen3_5_decode_modes.py --out decode_modes.json
"""

from __future__ import annotations

import argparse
import gc
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
from validate_qwen3_5_hf import CODE_PROMPT_SOURCE, TIE_MARGIN, build_prompts, kl_and_top1  # noqa: E402

DEFAULT_CONFIGS = (
    "torch:sliced-eager,torch:eager,torch:compile,torch:compile-reduce-overhead,torch:cudagraph,"
    "fla-delta:sliced-eager,fla-delta:cudagraph,fla-delta:compile-reduce-overhead,"
    "fla:sliced-eager,fla:cudagraph,fla:compile-reduce-overhead"
)

QUANT_LEVELS = ("bf16", "int8", "int8+lm_head", "int4", "int4-clip", "nvfp4", "nvfp4-mlp")


@torch.no_grad()
def hf_cached_reference(hf, ids, decode_positions, new_tokens):
    """Use cached HF forwards for prompt logits, greedy tokens and their margins.

    Prompt decode and generation each start from the same prefix split as the
    corresponding infer-opt check. No full-sequence logits supply generation margins.
    """
    T = ids.shape[1]
    split = max(T - decode_positions, 1)
    prompt_logits = []
    if split < T:
        output = hf(input_ids=ids[:, :split], use_cache=True)
        past = output.past_key_values
        for i in range(split, T):
            output = hf(input_ids=ids[:, i:i+1], past_key_values=past, use_cache=True)
            past = output.past_key_values
            prompt_logits.append(output.logits[:, -1:].float().cpu())
        del output, past
    output = hf(input_ids=ids, use_cache=True)
    tokens, margins = [], []
    for i in range(new_tokens):
        logits = output.logits[:, -1:].float()
        token = logits.argmax(-1)
        top = logits.topk(2, dim=-1).values
        tokens.append(token)
        margins.append((top[..., 0] - top[..., 1]).reshape(-1))
        if i + 1 < new_tokens:
            output = hf(input_ids=token, past_key_values=output.past_key_values, use_cache=True)
    return {"ids": ids, "T": T, "split": split, "new_ref": torch.cat(tokens, dim=1),
            "hf_decode_logits": torch.cat(prompt_logits, dim=1).to(ids.device) if prompt_logits else None,
            "margins": torch.cat(margins).cpu()}


def quantization_gate(rows: list[dict], kl_ratio: float) -> list[dict]:
    """Per quantized config: mean decode KL against the bf16 config with the same kernels and mode, strong mismatches."""
    def stats(kernels, mode, quant):
        mine = [r for r in rows if not r.get("summary") and (r["kernels"], r["mode"], r.get("quant", "bf16")) == (kernels, mode, quant)]
        kls = [r["decode_kl_vs_hf"] for r in mine if "decode_kl_vs_hf" in r]
        return {"prompts": len(mine), "mean_kl": sum(kls) / len(kls) if kls else None,
                "evaluated_prompts": len(kls), "prompt_names": sorted(r["prompt"] for r in mine),
                # A one-token prompt has no post-prefill prompt position to decode.
                # It still participates in the teacher-forced and free-run checks.
                "decode_eligible_prompts": sum(r.get("tokens", 2) > 1 for r in mine),
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
        complete = (got["prompts"] == base["prompts"] and got["prompts"] > 0
                    and got["evaluated_prompts"] == got["decode_eligible_prompts"]
                    and base["evaluated_prompts"] == base["decode_eligible_prompts"]
                    and got["prompt_names"] == base["prompt_names"]
                    and not any(r.get("error") for r in rows
                                if (r.get("kernels"), r.get("mode")) == (kernels, mode)
                                and r.get("quant", "bf16") in ("bf16", quant)))
        ratio = got["mean_kl"] / base["mean_kl"] if base["mean_kl"] > 0 else float("inf")
        gates.append({"config": f"{kernels}:{mode}:{quant}", "bf16": base, "quantized": got, "kl_ratio": ratio,
                      "pass_kl": ratio <= kl_ratio, "pass_strong": got["strong_mismatches"] == 0,
                      "complete": complete,
                      "pass": complete and ratio <= kl_ratio and got["strong_mismatches"] == 0})
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
    parser.add_argument("--int4-group-size", type=int, default=128)
    parser.add_argument("--int4-configs", default=None,
                        help="tune_qwen3_5_group_gemv.py --bits 4 output: its best table into INT4_LAUNCH_CONFIGS")
    parser.add_argument("--int4-from", default=None, help="quantized checkpoint dir or Hub repo for the int4 config")
    parser.add_argument("--fuse", action="store_true", help="model.fuse_projections() for every config (bf16 too)")
    parser.add_argument("--kl-ratio", type=float, default=2.0, help="gate: quantized mean KL / bf16 mean KL")
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--nvfp4-from", default=None, help="Validate a native packed checkpoint instead of recalibrating")
    parser.add_argument("--nvfp4-mlp-from", default=None, help="MLP-only native checkpoint; config quant is nvfp4-mlp")
    parser.add_argument("--out", default="decode_modes.json")
    args = parser.parse_args()
    if args.int4_configs:
        from infer_opt import quant_kernels

        for n, k, group, *config in json.loads(Path(args.int4_configs).read_text())["best"]:
            quant_kernels.INT4_LAUNCH_CONFIGS[(n, k, group)] = tuple(config)

    from huggingface_hub import snapshot_download
    from torch._dynamo.utils import counters
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = args.weights or snapshot_download(args.repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    tok = AutoTokenizer.from_pretrained(path)
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    prompts = build_prompts(tok, CODE_PROMPT_SOURCE.read_text())
    if args.prompts:
        prompts = {k: prompts[k] for k in args.prompts.split(",")}

    # HF reference for every prompt, computed once
    refs = {}
    with torch.no_grad():
        for name, text in prompts.items():
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            refs[name] = hf_cached_reference(hf, ids, args.decode_positions, args.new_tokens)
            print(json.dumps({"reference_prompt": name, "tokens": ids.shape[1],
                              "generated_tokens": refs[name]["new_ref"].shape[1]}), flush=True)
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
        temporary=out.with_suffix('.tmp')
        temporary.write_text(json.dumps({"args": vars(args), "reference_protocol": "hf-cached-greedy-v2",
                                   "reference_tokens": {n:r["new_ref"][0].tolist() for n,r in refs.items()},
                                   "reference_margins": {n:r["margins"].tolist() for n,r in refs.items()},
                                   "layers": ours.config.layer_summary(), "quantization": quant_reports,
                                   "gate": quantization_gate(rows, args.kl_ratio), "rows": rows}, indent=1))
        temporary.replace(out)

    import torch._inductor.config as inductor_config

    for config in args.configs.split(","):
        kernels, mode, *rest = config.split(":")
        quant = rest[0] if rest else "bf16"
        if quant not in QUANT_LEVELS:
            raise SystemExit(f"quant must be one of {QUANT_LEVELS}, got {quant!r}")
        if QUANT_LEVELS.index(quant) < QUANT_LEVELS.index(quant_done):
            raise SystemExit(f"{config}: the model is already {quant_done}; list configs in order {QUANT_LEVELS}")
        if quant != quant_done:
            # the compiled steps of the previous config reference its weights: drop them first, or the old weights
            # stay alive next to the new ones (27B: 52 GiB bf16 + 24 GiB int8 does not fit in 80 GB)
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
        if quant != quant_done and quant in ("nvfp4", "nvfp4-mlp"):
            from infer_opt.nvfp4 import quantize_nvfp4
            from bench_qwen3_5_nvfp4 import calibration_inputs
            del ours
            gc.collect()
            torch.cuda.empty_cache()
            source = args.nvfp4_from if quant == "nvfp4" else args.nvfp4_mlp_from
            if source:
                ours = Qwen3_5ForCausalLM.from_quantized_dir(source, device="cuda")
                quant_reports[quant] = {"from": source}
            else:
                ours = Qwen3_5ForCausalLM.from_pretrained_dir(path, device="cuda", dtype=dtype)
                ours.fuse_projections()
                quant_reports[quant] = quantize_nvfp4(ours, calibration_inputs(tok, out.parent, args.calibration_samples),
                                                      scope="mlp" if quant == "nvfp4-mlp" else "all")
            quant_done = quant
        elif quant != quant_done and quant.startswith("int4"):
            del ours
            gc.collect()
            torch.cuda.empty_cache()
            if quant == "int4" and args.int4_from:
                src = args.int4_from if Path(args.int4_from).is_dir() else snapshot_download(args.int4_from)
                ours = Qwen3_5ForCausalLM.from_quantized_dir(src, device="cuda", dtype=dtype)
                quant_reports[quant] = {"from": args.int4_from}
            else:
                ours = Qwen3_5ForCausalLM.from_pretrained_dir(path, dtype=dtype, device="cuda")
                report = quantize_linears(ours, bits=4, group_size=args.int4_group_size, clip_search=quant == "int4-clip")
                quant_reports[quant] = {"summary": report.summary(), "replaced": len(report.replaced),
                                        "group_size": args.int4_group_size, "clip_search": quant == "int4-clip"}
                print(quant, report.summary(), flush=True)
            quant_done = quant
            torch.cuda.empty_cache()
        elif quant != quant_done and args.int8_from:
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
        if args.fuse:
            ours.fuse_projections()  # no-op where already fused; a freshly loaded checkpoint gets fused here
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
