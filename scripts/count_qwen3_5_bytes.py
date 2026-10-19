"""Byte accounting for one Qwen3.5 batch-1 decode step: what must be read from HBM, by category.

Builds the model on the meta device from config.json (no weights are downloaded or allocated), then
splits parameters into the groups a weight-only quantization could touch and the ones it could not,
and adds the cache/state traffic. The result is an upper bound on what quantization can save and a
bandwidth lower bound for ms/token.

    python scripts/count_qwen3_5_bytes.py --repo Qwen/Qwen3.5-9B --context 1024 --out bytes.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

GIB = 2**30
MIB = 2**20

# (category, name pattern); the first match wins. "quantizable" = a 2-D Linear weight read in full every token.
GROUPS = [
    ("embed_tokens", r"embed_tokens\.weight$", False),
    ("lm_head", r"^lm_head\.weight$", True),
    ("full_attn_proj", r"self_attn\.(q|k|v|o)_proj\.weight$", True),
    ("linear_attn_proj", r"linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)\.weight$", True),
    ("mlp", r"mlp\.(gate|up|down)_proj\.weight$", True),
    ("conv1d", r"conv1d\.weight$", False),
]


def categorize(name: str) -> tuple[str, bool]:
    for group, pattern, quantizable in GROUPS:
        if re.search(pattern, name):
            return group, quantizable
    return "norms_and_small", False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--config", default=None, help="local config.json (skips the hub)")
    parser.add_argument("--context", type=int, default=1024, help="tokens in the KV cache when the step runs")
    parser.add_argument("--weight-bytes", type=float, default=2.0, help="bytes per weight today (bf16 = 2)")
    parser.add_argument("--bandwidth-gbps", type=float, nargs="+", default=[1555.0],
                        help="HBM bandwidths for the lower bound; add a measured value next to the spec one")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(args.repo, "config.json")
    config = Qwen3_5TextConfig.from_json(config_path)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)

    groups: dict[str, dict] = {}
    for name, p in model.named_parameters():
        group, quantizable = categorize(name)
        g = groups.setdefault(group, {"params": 0, "quantizable": quantizable})
        g["params"] += p.numel()
    for g in groups.values():
        g["bytes"] = g["params"] * args.weight_bytes
    total_params = sum(g["params"] for g in groups.values())

    # the embedding table is indexed, not streamed: one row per token
    embed_row_bytes = config.hidden_size * args.weight_bytes
    read_weight_bytes = sum(g["bytes"] for k, g in groups.items() if k != "embed_tokens") + embed_row_bytes
    quantizable_bytes = sum(g["bytes"] for g in groups.values() if g["quantizable"])

    layer = config.layer_summary()
    n_full, n_linear = layer["full_attention"], layer["linear_attention"]
    kv_read = n_full * 2 * config.num_key_value_heads * args.context * config.head_dim * 2  # K and V, bf16
    state_elems = config.linear_num_value_heads * config.linear_key_head_dim * config.linear_value_head_dim
    conv_dim = 2 * config.linear_key_head_dim * config.linear_num_key_heads + (
        config.linear_value_head_dim * config.linear_num_value_heads
    )
    recurrent_rw = n_linear * state_elems * 4 * 2  # fp32, read + write
    conv_rw = n_linear * conv_dim * (config.linear_conv_kernel_dim - 1) * 2 * 2  # bf16, read + write
    cache_bytes = kv_read + recurrent_rw + conv_rw
    total_read = read_weight_bytes + cache_bytes

    print(f"{args.repo}: {total_params / 1e9:.3f} B params, layers {layer['pattern']}")
    print(f"{'group':<18}{'params':>14}{'GiB':>9}{'share':>8}  quantizable")
    for k, g in sorted(groups.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"{k:<18}{g['params']:>14,}{g['bytes'] / GIB:>9.3f}{g['bytes'] / total_params / args.weight_bytes:>8.1%}  "
              f"{'yes' if g['quantizable'] else 'no'}")
    print(f"\nread per decode step (context {args.context}):")
    for label, b in (("weights (embedding: 1 row)", read_weight_bytes), ("full-attn K/V read", kv_read),
                     ("recurrent state r+w (fp32)", recurrent_rw), ("conv state r+w", conv_rw), ("total", total_read)):
        print(f"  {label:<28}{b / MIB:>10.1f} MiB  {b / total_read:>6.2%}")
    print(f"\nquantizable weight bytes: {quantizable_bytes / GIB:.3f} GiB = {quantizable_bytes / total_read:.1%} of the step")

    bounds = {}
    for bw in args.bandwidth_gbps:
        bounds[str(bw)] = {"bf16_ms": total_read / (bw * 1e9) * 1e3}
        row = [f"{bw:g} GB/s: bf16 {bounds[str(bw)]['bf16_ms']:.2f} ms"]
        for label, bits in (("int8", 8.0), ("int4+scale", 4.25)):
            # per-group scale overhead folded into the effective bits; the rest of the step is unchanged
            new_quant = quantizable_bytes * bits / (args.weight_bytes * 8)
            t = (total_read - quantizable_bytes + new_quant) / (bw * 1e9) * 1e3
            bounds[str(bw)][f"{label}_ms"] = t
            row.append(f"{label} {t:.2f} ms")
        print("  " + " | ".join(row))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "repo": args.repo, "context": args.context, "layers": layer, "total_params": total_params,
            "groups": groups, "read_bytes": {"weights": read_weight_bytes, "kv": kv_read,
                                             "recurrent_state": recurrent_rw, "conv_state": conv_rw,
                                             "total": total_read},
            "quantizable_bytes": quantizable_bytes, "lower_bound_ms": bounds,
        }, indent=1))


if __name__ == "__main__":
    main()
