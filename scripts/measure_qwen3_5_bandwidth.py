"""Measured HBM bandwidth for batch-1 decode, and the bf16 floor it implies for Qwen3.5.

The spec-sheet bandwidth is never reached. This script measures three things on the GPU it runs on:

1. copy bandwidth (read + write) and read-only bandwidth (a big sum), the usual "peak" numbers;
2. the bandwidth a batch-1 ``F.linear`` actually reaches for every distinct weight shape in the model
   (rotating through enough weight copies that the L2 cache cannot hold them, replayed from a CUDA graph
   so launch gaps are not counted);
3. the sum of those per-shape times over all Linear layers: a floor for the linear part of one decode step
   that uses the real kernels and real shapes, to compare with the byte-count floor.

    python scripts/measure_qwen3_5_bandwidth.py --repo Qwen/Qwen3.5-9B --out bandwidth.json

Needs a CUDA GPU (run it on the A100); weights are random, only shapes matter.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer_opt.profiling import device_summary, measure_peak_bandwidth, timeit  # noqa: E402
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

MIB = 2**20


@torch.inference_mode()
def read_bandwidth(device: torch.device, num_bytes: int = 1 << 30, iters: int = 20) -> float:
    """GB/s of a pure read: sum over a large bf16 tensor (no write traffic to count)."""
    x = torch.empty(num_bytes // 2, device=device, dtype=torch.bfloat16).normal_()
    t = timeit(lambda: x.sum(dtype=torch.float32), device=device, warmup=3, iters=iters)
    del x
    torch.cuda.empty_cache()
    return t.gbps(num_bytes)


@torch.inference_mode()
def linear_time_ms(out_features: int, in_features: int, dtype: torch.dtype, device: torch.device,
                   min_footprint: int = 512 * MIB, iters: int = 30) -> float:
    """Median time of one batch-1 F.linear, with the weight coming from HBM rather than L2."""
    w_bytes = out_features * in_features * torch.empty((), dtype=dtype).element_size()
    copies = max(2, -(-min_footprint // w_bytes))
    weights = [torch.empty(out_features, in_features, device=device, dtype=dtype).normal_(0, 0.02) for _ in range(copies)]
    x = torch.randn(1, 1, in_features, device=device, dtype=dtype)

    for w in weights:  # warm up cuBLAS heuristics and allocator before capture
        F.linear(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for w in weights:
            F.linear(x, w)
    t = timeit(graph.replay, device=device, warmup=3, iters=iters)
    del weights, graph
    torch.cuda.empty_cache()
    return t.ms / copies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--config", default=None, help="local config.json (skips the hub)")
    parser.add_argument("--spec-gbps", type=float, default=1555.0, help="datasheet HBM bandwidth to compare against")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    if args.config:
        config_path = args.config
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(args.repo, "config.json")
    config = Qwen3_5TextConfig.from_json(config_path)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)

    shapes = Counter((m.out_features, m.in_features) for m in model.modules() if isinstance(m, nn.Linear))
    info = device_summary(device)
    print(f"{info['name']}  torch {info['torch']}  cuda {info['cuda_build']}")

    torch.cuda.empty_cache()
    copy_gbps = measure_peak_bandwidth(device, dtype, num_bytes=512 << 20)
    read_gbps = read_bandwidth(device)
    print(f"copy (r+w) {copy_gbps:7.0f} GB/s  = {copy_gbps / args.spec_gbps:.1%} of spec {args.spec_gbps:g}")
    print(f"read only  {read_gbps:7.0f} GB/s  = {read_gbps / args.spec_gbps:.1%} of spec")

    print(f"\n{'out x in':>16}{'count':>7}{'MiB':>9}{'us':>9}{'GB/s':>9}{'% spec':>8}")
    rows, floor_ms, total_bytes = [], 0.0, 0
    for (out_f, in_f), count in sorted(shapes.items(), key=lambda kv: -kv[0][0] * kv[0][1] * kv[1]):
        ms = linear_time_ms(out_f, in_f, dtype, device)
        nbytes = out_f * in_f * 2
        gbps = nbytes / (ms * 1e-3) / 1e9
        floor_ms += ms * count
        total_bytes += nbytes * count
        rows.append({"out": out_f, "in": in_f, "count": count, "mib": nbytes / MIB, "us": ms * 1e3, "gbps": gbps})
        print(f"{out_f:>8} x {in_f:<5}{count:>7}{nbytes / MIB:>9.1f}{ms * 1e3:>9.1f}{gbps:>9.0f}{gbps / args.spec_gbps:>8.1%}")

    eff = total_bytes / (floor_ms * 1e-3) / 1e9
    print(f"\nall Linear layers: {total_bytes / 2**30:.2f} GiB in {floor_ms:.2f} ms  ->  {eff:.0f} GB/s effective "
          f"({eff / args.spec_gbps:.1%} of spec)")
    print("this is the measured bf16 floor for the Linear part of one decode step; the rest "
          "(embedding row, norms, conv, delta rule, attention) comes on top")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "repo": args.repo, "device": info, "spec_gbps": args.spec_gbps,
            "copy_gbps": copy_gbps, "read_gbps": read_gbps,
            "linear_shapes": rows, "linear_total_bytes": total_bytes,
            "linear_floor_ms": floor_ms, "linear_effective_gbps": eff,
        }, indent=1))


if __name__ == "__main__":
    main()
