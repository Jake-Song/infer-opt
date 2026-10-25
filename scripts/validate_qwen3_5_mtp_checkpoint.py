"""Validate a pinned bundled MTP release against the original attach-and-quantize workflow on an A100.

Loads one target model at a time. The reference reuses the verified unchanged target shards on disk; only its
metadata and original BF16 MTP tensors come from the source repos. Timings are intentionally not benchmarked.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_qwen3_5_spec_decode import baseline  # noqa: E402
from infer_opt.hub import fetch_hub_tensors  # noqa: E402
from infer_opt.quant import Int8Linear, quantize_linears  # noqa: E402
from infer_opt.qwen3_5 import FixedShapeDecoder, Qwen3_5ForCausalLM  # noqa: E402
from infer_opt.speculative import MTPSpeculativeDecoder  # noqa: E402
from validate_qwen3_5_hf import CODE_PROMPT_SOURCE, build_prompts  # noqa: E402


def fingerprint(state):
    return {n: {"dtype": str(t.dtype), "shape": list(t.shape),
                "sha256": hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            for n, t in state.items()}


@torch.no_grad()
def diagnose_difference(model, ids, plain, generated, rounds, index, max_len):
    """Recompute both paths on the common prefix; record whether this actually reproduces the disagreement."""
    record = next(r for r in rounds if r["start"] <= index < r["start"] + r["accepted"])
    prefix = torch.tensor([generated[:record["start"] - 1]], device=ids.device, dtype=ids.dtype)
    prefix = torch.cat([ids, prefix], dim=1)
    tokens = torch.tensor([record["inputs"]], device=ids.device)
    sequential, verify = model.new_cache(1, max_len), model.new_cache(1, max_len)
    verify.allocate_spec_buffers(tokens.shape[1])
    model(prefix, sequential)
    model(prefix, verify)
    offset = index - record["start"]
    for i in range(offset + 1):
        one = model.decode_step(tokens[:, i:i + 1], sequential)[0, -1].float()
    many = model.lm_head(model.model.language_model.forward_verify(tokens, verify))[0, offset].float()
    a, b = one.topk(2), many.topk(2)
    return {"method": "recomputed common-prefix single-token versus multi-token verification",
            "single_top2_ids": a.indices.tolist(), "verify_top2_ids": b.indices.tolist(),
            "single_top2_logits": a.values.tolist(), "verify_top2_logits": b.values.tolist(),
            "single_margin": (a.values[0] - a.values[1]).item(),
            "verify_margin": (b.values[0] - b.values[1]).item(),
            "max_abs_logit_difference": (one - many).abs().max().item(),
            "reproduced_choices": a.indices[0].item() == plain[index] and b.indices[0].item() == generated[index]}


@torch.no_grad()
def exercise(model, prompts, new_tokens, mode):
    hashes = fingerprint(model.mtp.state_dict())
    head_modules = sum(isinstance(m, Int8Linear) for m in model.mtp.modules())
    if head_modules != 8 or isinstance(model.lm_head, Int8Linear):
        raise RuntimeError("expected eight INT8 MTP Linear modules and a BF16 lm_head")
    model.fuse_projections()
    max_len = max(p.shape[1] for p in prompts.values()) + new_tokens + 32
    rows, logits = {}, {}
    decoder = FixedShapeDecoder(model, model.new_cache(1, max_len), mode)
    for name, ids in prompts.items():
        logits[name] = model(ids)[:, -1].cpu()
        tokens, _ = baseline(model, decoder, ids, new_tokens)
        rows[name] = {"prompt_tokens": ids.shape[1], "plain": tokens.tolist(), "spec": {}}
    del decoder
    gc.collect()
    torch.cuda.empty_cache()
    for k in (1, 3):
        decoder = MTPSpeculativeDecoder(model, max_len - 16, k=k, mode=mode)
        for name, ids in prompts.items():
            generated = decoder.prefill(ids)[0].tolist()
            if decoder.graph is None:
                decoder.prepare()
            rounds = []
            while len(generated) < new_tokens:
                record = {"start": len(generated), "inputs": decoder.tokens[0].tolist()}
                accepted = decoder.round()
                record["accepted"] = len(accepted)
                rounds.append(record)
                generated += accepted
            generated = generated[:new_tokens]
            plain = rows[name]["plain"]
            mismatch = next((i for i, (a, b) in enumerate(zip(plain, generated)) if a != b), None)
            result = {"tokens": generated, "first_plain_divergence": mismatch,
                      "plain_token_agreement": sum(a == b for a, b in zip(plain, generated)) / new_tokens,
                      **decoder.stats.as_dict()}
            if mismatch is not None and mismatch > 0:
                result["diagnostic"] = diagnose_difference(model, ids, plain, generated, rounds, mismatch, max_len)
            rows[name]["spec"][str(k)] = result
            print(json.dumps({"prompt": name, "k": k, "first_plain_divergence": mismatch}), flush=True)
        del decoder
        gc.collect()
        torch.cuda.empty_cache()
    return {"mtp_tensors": hashes, "mtp_int8_modules": head_modules, "prompts": rows}, logits


def main():
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Jakemu/Qwen3.5-27B-int8-g128-mtp")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--prompts", default="prose_1k,math,korean")
    parser.add_argument("--mode", choices=["eager", "cudagraph", "compile-reduce-overhead"], default="cudagraph")
    args = parser.parse_args()
    gpu = torch.cuda.get_device_properties(0)
    if "A100" not in gpu.name or gpu.total_memory / 2**30 < 38:
        raise RuntimeError(f"expected A100 with at least 38 GiB; got {gpu.name}, {gpu.total_memory / 2**30:.1f} GiB")
    torch.manual_seed(0)
    report = {"repo_id": args.repo_id, "revision": args.revision, "gpu": gpu.name,
              "gpu_memory_gib": gpu.total_memory / 2**30, "torch": torch.__version__, "mode": args.mode,
              "new_tokens": args.new_tokens, "status": "running"}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    save()
    try:
        path = Path(snapshot_download(args.repo_id, revision=args.revision))
        manifest = json.loads((path / "publish_manifest.json").read_text())
        remote = HfApi().model_info(args.repo_id, revision=args.revision, files_metadata=True)
        by_name = {f.rfilename: f for f in remote.siblings}
        for f in manifest["copied_files"]:
            actual = by_name[f["path"]]
            if actual.size != f["size"] or (f["sha256"] and (not actual.lfs or actual.lfs.sha256 != f["sha256"])):
                raise RuntimeError(f"copied file differs: {f['path']}")
        report["target_manifest_verified"] = True
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        texts = build_prompts(tokenizer, CODE_PROMPT_SOURCE.read_text())
        prompts = {n: tokenizer(texts[n], return_tensors="pt").input_ids.cuda() for n in args.prompts.split(",")}
        print("Load bundled checkpoint directly, before any original-model download", flush=True)
        model = Qwen3_5ForCausalLM.from_quantized_dir(path, device="cuda", kernels="fla-delta")
        assert model.mtp is not None
        report["bundled"], bundled_logits = exercise(model, prompts, args.new_tokens, args.mode)
        report["bundled_load_without_original"] = True
        save()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("Load target-only shards and attach original BF16 MTP, then quantize it on GPU", flush=True)
        with tempfile.TemporaryDirectory(prefix="mtp-reference-") as directory:
            reference = Path(directory)
            for f in manifest["copied_files"]:
                if f["path"].endswith(".safetensors") or f["path"] == "config.json":
                    (reference / f["path"]).symlink_to((path / f["path"]).resolve())
            base_spec = Path(hf_hub_download(manifest["base_repo"], "quantization.json", revision=manifest["base_revision"]))
            (reference / "quantization.json").write_bytes(base_spec.read_bytes())
            model = Qwen3_5ForCausalLM.from_quantized_dir(reference, device="cuda", kernels="fla-delta")
            assert model.mtp is None
            model.attach_mtp(fetch_hub_tensors(manifest["source_repo"], "mtp.", manifest["source_revision"]))
            quantize_linears(model, group_size=128)
            report["attached_reference"], reference_logits = exercise(model, prompts, args.new_tokens, args.mode)
        report["mtp_tensors_equal"] = report["bundled"]["mtp_tensors"] == report["attached_reference"]["mtp_tensors"]
        report["prefill_logits_equal"] = {n: torch.equal(bundled_logits[n], reference_logits[n]) for n in prompts}
        report["prefill_max_abs_difference"] = {n: (bundled_logits[n].float() - reference_logits[n].float()).abs().max().item()
                                                 for n in prompts}
        equal = {}
        for name in prompts:
            a, b = report["bundled"]["prompts"][name], report["attached_reference"]["prompts"][name]
            equal[name] = {"plain": a["plain"] == b["plain"],
                           **{f"spec_k{k}": a["spec"][str(k)]["tokens"] == b["spec"][str(k)]["tokens"] for k in (1, 3)}}
        report["same_path_tokens_equal"] = equal
        passed = report["mtp_tensors_equal"] and all(report["prefill_logits_equal"].values()) and all(
            all(row.values()) for row in equal.values())
        report["status"] = "passed" if passed else "failed"
        report["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        save()
        print(json.dumps({k: v for k, v in report.items() if k not in ("bundled", "attached_reference")}), flush=True)
        if not passed:
            raise RuntimeError("bundled checkpoint differs from the attached reference; see report")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
