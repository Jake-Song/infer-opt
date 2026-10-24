"""MTP speculative decoding of Qwen3.5 against plain fixed-shape decoding (A100): ms/token, acceptance, agreement.

One process, one bf16 load with the MTP head, ``fuse_projections()``, then configurations in this order (the
quantization is in place and one-way):

  bf16             the control (only with --configs containing it)
  int8             quantize_linears(group_size) on the target; MTP head and lm_head stay bf16
  int8+mtp         the MTP head's Linears quantized too
  int8+mtp+lm_head lm_head quantized too (it is shared by the target and the drafts)

For every configuration and prompt: greedy decoding with :class:`FixedShapeDecoder` (the baseline) and with
:class:`MTPSpeculativeDecoder` for every ``k`` in ``--ks``, both in ``--mode``. Recorded per run: decode ms/token
(prefill excluded), rounds and accepted-token histogram, and the first position where the speculative tokens leave
the baseline's (a ``T``-token forward rounds differently from a one-token one, so near-ties can flip). ``--gemv``
also times the int8 GEMVs of one step with 1 and ``k + 1`` activation rows (every Int8Linear in model order).

The prompts are real text from ``validate_qwen3_5_hf.build_prompts`` (random tokens would make acceptance
meaningless), so the tokenizer is needed (transformers).

    python scripts/bench_qwen3_5_spec_decode.py --repo Qwen/Qwen3.5-27B --autotune --gemv --out spec_decode.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_qwen3_5_decode import sync  # noqa: E402
from infer_opt.quant import Int8Linear, quantize_linears  # noqa: E402
from infer_opt.qwen3_5 import FixedShapeDecoder, Qwen3_5ForCausalLM  # noqa: E402
from infer_opt.speculative import MTPSpeculativeDecoder  # noqa: E402
from validate_qwen3_5_hf import CODE_PROMPT_SOURCE, build_prompts  # noqa: E402

PROMPTS = ("chat_long", "code_1k", "prose_1k", "math", "korean")


@torch.no_grad()
def baseline(model, decoder, ids, new_tokens):
    """Greedy tokens and decode ms/token of a fixed-shape decoder (reused across prompts: one cache size)."""
    decoder.cache.reset()
    token = model(ids, decoder.cache)[:, -1].argmax(-1, keepdim=True)
    if decoder.graph is None:
        decoder.prepare()
    out = [token]
    sync()
    t0 = time.perf_counter()
    for _ in range(new_tokens - 1):
        token = decoder.step(token)[:, -1].argmax(-1, keepdim=True)
        out.append(token)
    sync()
    return torch.cat(out, dim=1)[0], (time.perf_counter() - t0) / (new_tokens - 1) * 1e3


@torch.no_grad()
def speculative(decoder, ids, new_tokens):
    """Greedy tokens, decode ms/token (the rounds after the prefill) and ms per round."""
    out = decoder.prefill(ids)[0].tolist()
    if decoder.graph is None:
        decoder.prepare()
    sync()
    t0 = time.perf_counter()
    while len(out) < new_tokens:
        out += decoder.round()
    sync()
    seconds = time.perf_counter() - t0
    return (torch.tensor(out[:new_tokens], device=ids.device), seconds / (len(out) - 1) * 1e3,
            seconds / decoder.stats.rounds * 1e3)


@torch.no_grad()
def gemv_times(model, rows: int, repeats: int = 20) -> float:
    """ms for every Int8Linear of the model once, on ``rows`` activation rows (model order, CUDA-graph replays)."""
    linears = [m for m in model.modules() if isinstance(m, Int8Linear)]
    xs = [torch.randn(1, rows, m.in_features, device="cuda", dtype=m.scale.dtype) for m in linears]
    for m, x in zip(linears, xs):
        m(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for m, x in zip(linears, xs):
            m(x)
    graph.replay()
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        graph.replay()
    sync()
    return (time.perf_counter() - t0) / repeats * 1e3


def measure(model, label, args, prompts):
    """Every prompt through the baseline and every k. One decoder per method for all prompts (same cache size), so
    each compiles once per configuration; the first prompt is run once more first, untimed, to compile."""
    import torch._inductor.config as inductor_config

    inductor_config.max_autotune = args.autotune
    inductor_config.coordinate_descent_tuning = args.autotune
    torch._dynamo.reset()
    max_len = max(ids.shape[1] for ids in prompts.values()) + args.new_tokens
    rows = []
    try:
        base = FixedShapeDecoder(model, model.new_cache(1, max_len + 8), args.mode)
        specs = {k: MTPSpeculativeDecoder(model, max_len, k=k, mode=args.mode) for k in args.ks}
        first = next(iter(prompts.values()))
        t0 = time.perf_counter()
        baseline(model, base, first, 16)
        for decoder in specs.values():
            speculative(decoder, first, 16)
        compile_s = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001 - record and continue
        print(traceback.format_exc()[-1500:], flush=True)
        return [{"config": label, "error": f"{type(exc).__name__}: {str(exc)[:600]}"}]
    for name, ids in prompts.items():
        row = {"config": label, "prompt": name, "prompt_tokens": ids.shape[1], "compile_s": compile_s}
        try:
            want, row["baseline_ms_per_token"] = baseline(model, base, ids, args.new_tokens)
            row["spec"] = {}
            for k, decoder in specs.items():
                got, ms, round_ms = speculative(decoder, ids, args.new_tokens)
                same = (got == want).tolist()
                spec = {"ms_per_token": ms, "round_ms": round_ms, "speedup": row["baseline_ms_per_token"] / ms,
                        **decoder.stats.as_dict(), "first_divergence": same.index(False) if False in same else None,
                        "token_agreement": sum(same) / len(same)}
                row["spec"][k] = spec
                print(label, name, "k", k, json.dumps(spec), flush=True)
            row["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        except Exception as exc:  # noqa: BLE001 - record and continue
            row["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
            print(traceback.format_exc()[-1500:], flush=True)
        print(json.dumps({k: v for k, v in row.items() if k != "spec"}), flush=True)
        rows.append(row)
    if args.gemv:
        rows.append({"config": label, "gemv_ms": {r: gemv_times(model, r) for r in sorted({1, *(k + 1 for k in args.ks)})}})
        print(json.dumps(rows[-1]), flush=True)
    del base, specs
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--kernels", default="fla-delta")
    parser.add_argument("--mode", default="compile-reduce-overhead")
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--ks", default="1,2,3")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--prompts", default=",".join(PROMPTS))
    parser.add_argument("--configs", default="int8,int8+mtp,int8+mtp+lm_head")
    parser.add_argument("--autotune", action="store_true", help="max_autotune + coordinate_descent_tuning")
    parser.add_argument("--gemv", action="store_true", help="also time the int8 GEMVs of a step with 1 and k+1 rows")
    parser.add_argument("--rows-configs", default=None,
                        help="tune_qwen3_5_group_gemv.py --rows output: its rows_best table into ROWS_LAUNCH_CONFIGS")
    parser.add_argument("--out", default="spec_decode.json")
    args = parser.parse_args()
    args.ks = [int(k) for k in args.ks.split(",")]
    if args.rows_configs:
        from infer_opt import quant_kernels

        for m, n, k, group, config in json.loads(Path(args.rows_configs).read_text())["rows_best"]:
            quant_kernels.ROWS_LAUNCH_CONFIGS[(m, n, k, group)] = tuple(config)

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = args.weights or snapshot_download(args.repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    tok = AutoTokenizer.from_pretrained(path)
    texts = build_prompts(tok, CODE_PROMPT_SOURCE.read_text())
    prompts = {name: tok(texts[name], return_tensors="pt").input_ids.cuda() for name in args.prompts.split(",")}
    model = Qwen3_5ForCausalLM.from_pretrained_dir(path, dtype=torch.bfloat16, device="cuda", kernels=args.kernels,
                                                   mtp=True)
    model.fuse_projections()
    print(f"{torch.cuda.get_device_name()} torch {torch.__version__} kernels {args.kernels} mode {args.mode} "
          f"new {args.new_tokens} ks {args.ks}", flush=True)

    wanted = args.configs.split(",")
    rows, reports = [], {}
    out = Path(args.out)

    def save():
        out.write_text(json.dumps({"args": vars(args), "device": torch.cuda.get_device_name(), "torch": torch.__version__,
                                   "layers": model.config.layer_summary(), "quantization": reports, "rows": rows},
                                  indent=1))

    stages = [("bf16", None),
              ("int8", {"exclude": ("mtp.",)}),
              ("int8+mtp", {}),
              ("int8+mtp+lm_head", {"include_lm_head": True})]
    for label, kwargs in stages:
        if kwargs is not None:
            if not any(w.startswith(label) for w in wanted):
                continue
            torch._dynamo.reset()  # drop compiled steps that reference the weights about to be replaced
            gc.collect()
            torch.cuda.empty_cache()
            report = quantize_linears(model, group_size=args.group_size, **kwargs)
            reports[label] = {"summary": report.summary(), "replaced": len(report.replaced)}
            print(label, report.summary(), flush=True)
            torch.cuda.empty_cache()
        if label in wanted:
            rows += measure(model, label, args, prompts)
            save()


if __name__ == "__main__":
    main()
