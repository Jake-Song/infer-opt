"""Accuracy of the fixed-shape decode path (eager / torch.compile / CUDA graph) against Hugging Face.

Same metrics and prompts as validate_qwen3_5_hf.py, but only the decode path is under test:
  * token-by-token logits of the last 32 prompt positions vs HF (KL, top-1, max |dlogit|)
  * 192-token greedy generation: first free-run mismatch with HF, and next-token agreement when fed HF's tokens
    (mismatch margins recorded; margin <= 0.5 counts as a near-tie)
The prompt is always prefilled by the ordinary eager path; the decoder under test only runs the single-token steps.
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

from infer_opt.qwen3_5 import FixedShapeDecoder, Qwen3_5ForCausalLM  # noqa: E402
from validate_qwen3_5_hf import TIE_MARGIN, build_prompts, kl_and_top1  # noqa: E402

DEFAULT_CONFIGS = (
    "torch:sliced-eager,torch:eager,torch:compile,torch:compile-reduce-overhead,torch:cudagraph,"
    "fla-delta:sliced-eager,fla-delta:cudagraph,fla-delta:compile-reduce-overhead,"
    "fla:sliced-eager,fla:cudagraph,fla:compile-reduce-overhead"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--new-tokens", type=int, default=192)
    parser.add_argument("--decode-positions", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS, help="comma separated kernels:mode")
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
    ours = Qwen3_5ForCausalLM.from_pretrained_dir(path, dtype=torch.bfloat16, device="cuda")
    print(f"{torch.cuda.get_device_name()} torch {torch.__version__} prompts {list(prompts)}", flush=True)

    rows = []
    out = Path(args.out)

    def save():
        out.write_text(json.dumps({"args": vars(args), "layers": ours.config.layer_summary(), "rows": rows}, indent=1))

    for config in args.configs.split(","):
        kernels, mode = config.split(":")
        torch._dynamo.reset()
        counters.clear()
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
                    row = {"kernels": kernels, "mode": mode, "prompt": name, "tokens": T}

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

            summary = {"kernels": kernels, "mode": mode, "summary": True, "seconds": time.time() - t0,
                       "dynamo_unique_graphs": counters["stats"].get("unique_graphs"),
                       "dynamo_graph_breaks": sum(counters["graph_break"].values())}
        except Exception as exc:  # noqa: BLE001 - record and go on with the next config
            summary = {"kernels": kernels, "mode": mode, "summary": True, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
            print(traceback.format_exc()[-1200:], flush=True)
        rows.append(summary)
        print(json.dumps(summary), flush=True)
        save()
        cache = decoder = step = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
