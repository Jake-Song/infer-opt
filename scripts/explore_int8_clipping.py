"""How much does a per-row clipping search reduce INT8 weight error on real Qwen3.5 weights?  (CPU, one shard)

For each Linear weight of the chosen layers, compares three quantizers by relative squared error
||dequant(q) - W||^2 / ||W||^2 (the scale is rounded to bf16 first, as quantize_int8 stores it):

  absmax    scale = max|w| / 127 per output row  (what infer_opt.quant does now)
  clip      per row, the best of a grid of ratios r: scale = r * max|w| / 127, values beyond are clamped
  group128  absmax, but one scale per 128 input columns (needs a different kernel; shown as a reference point)

Weight error is only a proxy: the output error also depends on the activations, which this does not use.

    python scripts/explore_int8_clipping.py --shard model.safetensors-00001-of-00004.safetensors --layers 14,15,22,27
"""

from __future__ import annotations

import argparse
import re

import torch

GRID = (1.0, 0.98, 0.96, 0.94, 0.92, 0.9, 0.87, 0.84, 0.8, 0.75, 0.7)
KINDS = r"(mlp\.(gate|up|down)_proj|self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj))\.weight$"


def quantize_error(w: torch.Tensor, amax: torch.Tensor, ratio: float) -> torch.Tensor:
    """Per-row squared error for scale = ratio * amax / 127 (scale rounded through bf16)."""
    s = (amax * ratio / 127).clamp_min(1e-12).to(torch.bfloat16).float().unsqueeze(1)
    q = (w / s).round().clamp(-127, 127)
    return ((q * s - w) ** 2).sum(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--shard", default="model.safetensors-00001-of-00004.safetensors")
    parser.add_argument("--layers", default="14,15,22,27")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(args.repo, args.shard)
    wanted = {int(x) for x in args.layers.split(",")}
    print(f"{'tensor':<48}{'max/rms':>8}{'absmax':>10}{'clip':>10}{'gain':>7}{'r<1 rows':>10}{'mean r':>8}{'group128':>10}")
    totals = {"absmax": 0.0, "clip": 0.0, "group": 0.0, "norm": 0.0}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in sorted(f.keys()):
            m = re.search(r"language_model\.layers\.(\d+)\.", name)
            if not m or int(m.group(1)) not in wanted or not re.search(KINDS, name):
                continue
            w = f.get_tensor(name).float()
            amax = w.abs().amax(1)
            norm = (w ** 2).sum()
            base = quantize_error(w, amax, 1.0)
            best, best_r = base.clone(), torch.ones_like(base)
            for r in GRID[1:]:
                err = quantize_error(w, amax, r)
                better = err < best
                best, best_r = torch.where(better, err, best), torch.where(better, torch.full_like(best_r, r), best_r)
            # group-wise absmax needs per-group maxima, not per-row ones
            n, k = w.shape
            gmax = w.reshape(n, k // 128, 128).abs().amax(-1)
            gscale = (gmax / 127).clamp_min(1e-12).to(torch.bfloat16).float().unsqueeze(-1)
            wg = w.reshape(n, k // 128, 128)
            gerr = (((wg / gscale).round().clamp(-127, 127) * gscale - wg) ** 2).sum()
            rms = (w ** 2).mean().sqrt()
            print(f"{name.replace('model.language_model.', ''):<48}{(w.abs().max() / rms).item():>8.1f}"
                  f"{(base.sum() / norm).item():>10.2e}{(best.sum() / norm).item():>10.2e}"
                  f"{(1 - best.sum() / base.sum()).item():>7.1%}{(best_r < 1).float().mean().item():>10.1%}"
                  f"{best_r.mean().item():>8.3f}{(gerr / norm).item():>10.2e}")
            totals["absmax"] += base.sum().item()
            totals["clip"] += best.sum().item()
            totals["group"] += gerr.item()
            totals["norm"] += norm.item()
    n = totals["norm"]
    print(f"\ntotal relative error: absmax {totals['absmax'] / n:.3e} | clip {totals['clip'] / n:.3e} "
          f"({1 - totals['clip'] / totals['absmax']:.1%} lower) | group128 {totals['group'] / n:.3e} "
          f"({1 - totals['group'] / totals['absmax']:.1%} lower)")


if __name__ == "__main__":
    main()
