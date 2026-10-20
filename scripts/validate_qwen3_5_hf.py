"""Accuracy validation of infer_opt.qwen3_5 against Hugging Face Qwen3.5 (bf16, A100).

For every prompt and every configuration (kernels x cache type) it measures

  * teacher-forced logits vs HF over the whole prompt (prefill path): top-1 agreement, KL(HF || ours), max |dlogit|
  * the same for the last 32 positions computed token by token through the cache (decode path)
  * 192-token greedy generation: first free-run mismatch with HF, and next-token agreement when ours is fed
    HF's own tokens (so one near-tie flip cannot hide or inflate later steps); the HF top1-top2 margin of every
    mismatching step is recorded
  * a noise floor: the same metrics of HF-bf16 and of ours-bf16 against an fp32 run of our model, i.e. how far
    plain bf16 rounding moves the logits
  * batch=2 (equal-length prompts): ours batched vs HF batched, and ours batched vs ours single

    pip install -U "transformers>=5" safetensors huggingface_hub flash-linear-attention
    python scripts/validate_qwen3_5_hf.py --out validation.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer_opt.qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

TIE_MARGIN = 0.5  # HF top1-top2 logit gap at or below which a flipped argmax counts as a near-tie

PROSE = (
    "Section {i}. The harbour town woke slowly: fishermen hauled nets onto the quay, a baker propped open her "
    "shutters, and the ferry to the islands blew its horn twice. Meanwhile, in a laboratory three streets away, "
    "a team of physicists argued about whether the anomaly in the detector was noise or the first sign of a new "
    "particle. Their disagreement was polite but stubborn, and it would take another eleven months of data to "
    "settle. In a nearby classroom, children learned that the speed of light is about three hundred thousand "
    "kilometres per second, and that a year is the distance light travels in 365 days, not a unit of time. "
)


# The code prompts are cut from this frozen copy of infer_opt/qwen3_5.py (its text as of 9e5c3bd; the first 12 000
# characters, which hold both prompts, are unchanged since 32c1509). They used to be cut from the live file, so editing
# the model's docstring changed code_1k and code_1_5k and with them the gate's mean KL.
CODE_PROMPT_SOURCE = Path(__file__).resolve().with_name("code_prompt_source.txt")


def build_prompts(tok, source_text: str) -> dict[str, str]:
    code_ids = tok(source_text, return_tensors="pt").input_ids[0]
    return {
        "short": "The capital of France is",
        "hello": "Hello",
        "korean": "대한민국의 수도는 서울이며, 한강을 중심으로 발전한 이 도시는 오랜 역사와",
        "math": "Question: A train travels 120 km in 1.5 hours, then 80 km in 1 hour. What is its average speed? "
                "Let's think step by step.\nAnswer:",
        "chat": tok.apply_chat_template(
            [{"role": "user", "content": "Write a haiku about GPUs and explain the syllable count."}],
            tokenize=False, add_generation_prompt=True,
        ),
        "chat_long": tok.apply_chat_template(
            [{"role": "user", "content": "Explain in detail how a transformer language model generates text, "
                                         "covering attention, the KV cache, and sampling."}],
            tokenize=False, add_generation_prompt=True,
        ),
        "prose_1k": "".join(PROSE.format(i=i) for i in range(1, 9)),
        "code_1k": tok.decode(code_ids[:1000]),
        "code_1_5k": tok.decode(code_ids[1000:2500]),
    }


def kl_and_top1(p: torch.Tensor, q: torch.Tensor) -> tuple[float, float, float]:
    """p, q: [N, V] logits. Returns (mean KL(p||q), top-1 agreement, max |p - q|)."""
    p, q = p.float().flatten(0, -2), q.float().flatten(0, -2)
    kls = []
    for s in range(0, p.shape[0], 256):
        lp, lq = F.log_softmax(p[s : s + 256], -1), F.log_softmax(q[s : s + 256], -1)
        kls.append((lp.exp() * (lp - lq)).sum(-1))
    return (
        torch.cat(kls).mean().item(),
        (p.argmax(-1) == q.argmax(-1)).float().mean().item(),
        (p - q).abs().max().item(),
    )


@torch.no_grad()
def decode_logits(model, ids: torch.Tensor, split: int, static: bool) -> torch.Tensor:
    """Logits for positions split..T-1, produced one token at a time through the cache."""
    cache = model.new_cache(ids.shape[0], ids.shape[1] if static else None)
    model(ids[:, :split], cache)
    return torch.cat([model(ids[:, i : i + 1], cache) for i in range(split, ids.shape[1])], dim=1).float()


@torch.no_grad()
def teacher_forced_next_tokens(model, prompt: torch.Tensor, forced: torch.Tensor, static: bool) -> torch.Tensor:
    cache = model.new_cache(1, prompt.shape[1] + forced.shape[1] if static else None)
    preds = [model(prompt, cache)[:, -1].argmax(-1)]
    for i in range(forced.shape[1] - 1):
        preds.append(model(forced[:, i : i + 1], cache)[:, -1].argmax(-1))
    return torch.cat(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--new-tokens", type=int, default=192)
    parser.add_argument("--decode-positions", type=int, default=32)
    parser.add_argument("--prompts", default=None, help="comma separated subset of prompt names")
    parser.add_argument("--out", default="validation.json")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = args.weights or snapshot_download(args.repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    tok = AutoTokenizer.from_pretrained(path)
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    ours = Qwen3_5ForCausalLM.from_pretrained_dir(path, dtype=torch.bfloat16, device="cuda")
    ref32 = Qwen3_5ForCausalLM.from_pretrained_dir(path, dtype=torch.float32, device="cuda")  # kernels="torch"
    prompts = build_prompts(tok, CODE_PROMPT_SOURCE.read_text())
    if args.prompts:
        prompts = {k: prompts[k] for k in args.prompts.split(",")}
    configs = list(itertools.product(("torch", "fla-delta", "fla"), (False, True)))
    print(f"{torch.cuda.get_device_name()} torch {torch.__version__} prompts {list(prompts)} new_tokens {args.new_tokens}",
          flush=True)

    rows, batch_rows = [], []
    out = Path(args.out)

    def save():
        out.write_text(json.dumps({"args": vars(args), "layers": ours.config.layer_summary(), "rows": rows, "batch_rows": batch_rows}, indent=1))

    with torch.no_grad():
        for name, text in prompts.items():
            t0 = time.time()
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            T = ids.shape[1]
            split = max(T - args.decode_positions, 1)
            hf_logits = hf(input_ids=ids).logits.float()
            ref_logits = ref32(ids).float()
            noise_kl, noise_top1, _ = kl_and_top1(ref_logits, hf_logits)
            # no min_new_tokens: suppressing EOS would make HF pick tokens that raw argmax never would
            hf_gen = hf.generate(ids, max_new_tokens=args.new_tokens, do_sample=False)
            new_ref = hf_gen[:, T:]
            n_new = new_ref.shape[1]
            full = hf(input_ids=hf_gen).logits.float()[0, T - 1 : -1]  # HF logits at every generation step
            top2 = full.topk(2, -1).values
            margins = (top2[:, 0] - top2[:, 1]).cpu()

            for kernels, static in configs:
                ours.set_kernels(kernels)
                o_logits = ours(ids).float()
                kl, top1, maxd = kl_and_top1(hf_logits, o_logits)
                kl_ref, top1_ref, maxd_ref = kl_and_top1(ref_logits, o_logits)
                if split < T:  # a 1-token prompt has no decode positions inside the prompt
                    dkl, dtop1, dmax = kl_and_top1(hf_logits[:, split:], decode_logits(ours, ids, split, static))
                else:
                    dkl = dtop1 = dmax = None

                gen = ours.generate_greedy(ids, n_new, static_cache=static)[:, T:]
                same = (gen == new_ref)[0]
                first = int((~same).nonzero()[0]) if not bool(same.all()) else None
                preds = teacher_forced_next_tokens(ours, ids, new_ref, static)
                wrong = (preds != new_ref[0]).cpu()
                wrong_margins = margins[wrong].tolist()
                near_tie = sum(m <= TIE_MARGIN for m in wrong_margins)
                rows.append({
                    "prompt": name, "tokens": T, "kernels": kernels, "static_cache": static,
                    "prefill_kl_vs_hf": kl, "prefill_top1_vs_hf": top1, "prefill_maxdiff_vs_hf": maxd,
                    "prefill_kl_vs_fp32": kl_ref, "prefill_top1_vs_fp32": top1_ref,
                    "hf_bf16_kl_vs_fp32": noise_kl, "hf_bf16_top1_vs_fp32": noise_top1,
                    "decode_kl_vs_hf": dkl, "decode_top1_vs_hf": dtop1, "decode_maxdiff_vs_hf": dmax,
                    "gen_identical": first is None, "gen_first_mismatch": first,
                    "tf_next_token_agreement": 1 - wrong.float().mean().item(),
                    "new_tokens_compared": n_new,
                    "tf_mismatch_steps": int(wrong.sum()),
                    "tf_mismatch_near_tie": near_tie,
                    "tf_mismatch_strong": int(wrong.sum()) - near_tie,
                    "tf_mismatch_hf_margins": [round(m, 3) for m in wrong_margins],
                })
                print(f"[{name:9s} T={T:5d}] {kernels:9s} static={static!s:5s} prefill top1 {top1:.4f} KL {kl:.2e} | "
                      f"decode top1 {dtop1 if dtop1 is None else round(dtop1, 4)} | gen identical {first is None} first {first} | "
                      f"tf wrong {int(wrong.sum())}/{n_new} (tie {near_tie}) margins {rows[-1]['tf_mismatch_hf_margins']} "
                      f"| fp32-ref KL ours {kl_ref:.2e} hf {noise_kl:.2e}", flush=True)
            print(f"  ({time.time() - t0:.0f}s)", flush=True)
            save()
            del hf_logits, ref_logits, full

        # batch=2, equal length, no padding
        tokenized = sorted((tok(t, return_tensors="pt", add_special_tokens=False).input_ids[0] for t in prompts.values()),
                           key=len, reverse=True)
        a, b = tokenized[0][:256], tokenized[1][:256]
        n = min(len(a), len(b))
        batch = torch.stack([a[:n], b[:n]]).cuda()
        hf_b = hf(input_ids=batch).logits.float()
        split = n - args.decode_positions
        for kernels, static in configs:
            ours.set_kernels(kernels)
            ob = ours(batch).float()
            single = torch.cat([ours(batch[i : i + 1]) for i in range(2)]).float()
            kl, top1, maxd = kl_and_top1(hf_b, ob)
            dkl, dtop1, dmax = kl_and_top1(hf_b[:, split:], decode_logits(ours, batch, split, static))
            batch_rows.append({
                "kernels": kernels, "static_cache": static, "tokens": n,
                "prefill_top1_vs_hf": top1, "prefill_kl_vs_hf": kl, "decode_top1_vs_hf": dtop1, "decode_kl_vs_hf": dkl,
                "batched_vs_single_maxdiff": (ob - single).abs().max().item(),
            })
            print("[batch=2]", json.dumps(batch_rows[-1]), flush=True)
        save()

    print("\n=== summary (mean over prompts) ===")
    print(f"{'kernels':10s} {'static':6s} {'pre top1':>9s} {'pre KL':>9s} {'dec top1':>9s} {'gen ident':>9s} "
          f"{'tf wrong (strong)':>18s} {'KL vs fp32 (ours | hf)':>24s}")
    for kernels, static in configs:
        sel = [r for r in rows if r["kernels"] == kernels and r["static_cache"] == static]
        mean = lambda k: sum(r[k] for r in sel) / len(sel)  # noqa: E731
        print(f"{kernels:10s} {static!s:6s} {mean('prefill_top1_vs_hf'):9.4f} {mean('prefill_kl_vs_hf'):9.2e} "
              f"{sum(r['decode_top1_vs_hf'] for r in sel if r['decode_top1_vs_hf'] is not None) / sum(r['decode_top1_vs_hf'] is not None for r in sel):9.4f} {sum(r['gen_identical'] for r in sel):>6d}/{len(sel)} "
              f"{sum(r['tf_mismatch_steps'] for r in sel):>10d} ({sum(r['tf_mismatch_strong'] for r in sel)}) "
              f"{mean('prefill_kl_vs_fp32'):>13.2e} | {mean('hf_bf16_kl_vs_fp32'):.2e}")


if __name__ == "__main__":
    main()
