"""The one-token group-wise INT8 GEMV in CUDA C++ (infer_opt/quant_cuda.py) against the Triton one (quant_kernels.py).

For every Linear shape that ``quantize_linears(include_lm_head=True)`` turns into Int8Linear in a Qwen3.5 model
(``--fuse``: the stacked gate+up, q+k+v and in_proj_qkv+z of ``fuse_projections``), with random int8 weights (the
time does not depend on the values) in enough copies to exceed L2, replayed from one CUDA graph
(``bench_int8_linear.graph_time_ms``), this times

  bf16     cuBLAS F.linear on a bf16 weight of the shape (the reference: 2 bytes per weight)
  triton   quant_kernels.w8a16_group_gemv with its launch rule (``--triton-sweep``: and the best (BLOCK_N, BLOCK_K,
           num_warps) of the grid of tune_qwen3_5_group_gemv.py)
  cuda     quant_cuda.w8a16_group_gemv_cuda with its launch rule and with every compiled (ROWS, THREADS, UNROLL)

and prints microseconds, GB/s on the int8 weight plus scales, and cuda / triton. The ``best`` table of the output
loads into ``quant_cuda.CUDA_LAUNCH_CONFIGS`` (``bench_qwen3_5_int8_decode.py --cuda-configs``).

    python scripts/bench_int8_gemv_cuda.py --repo Qwen/Qwen3.5-27B --fuse --out gemv_cuda.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_int8_linear import graph_time_ms  # noqa: E402
from infer_opt import quant_cuda, quant_kernels  # noqa: E402
from infer_opt.profiling import device_summary  # noqa: E402
from infer_opt.quant import dequantize_int8  # noqa: E402
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

MIB = 2**20


@torch.inference_mode()
def bench_shape(n: int, k: int, group: int, dtype, device, min_footprint: int, iters: int, bf16: bool,
                triton_sweep: bool) -> dict:
    copies = max(2, -(-min_footprint // (n * k)))
    q = [torch.randint(-127, 128, (n, k), device=device, dtype=torch.int8) for _ in range(copies)]
    s = [torch.rand(n, k // group, device=device, dtype=dtype) * 1e-3 + 1e-4 for _ in range(copies)]
    x = torch.randn(1, 1, k, device=device, dtype=dtype)
    nbytes = n * k + n * (k // group) * s[0].element_size()

    ref = F.linear(x.float(), dequantize_int8(q[0], s[0], torch.float32))
    row = {"n": n, "k": k, "group": group, "copies": copies, "MB": nbytes / 1e6}
    tri = quant_kernels.w8a16_group_gemv(x, q[0], s[0]).float()
    row["triton_rel_err"] = ((tri - ref).norm() / ref.norm()).item()
    row["triton_config"] = quant_kernels.launch_config(n, k, group)
    # untimed pass first: on the A100 the first variant timed on fresh weights came out 10-14% slow ((1, 1024, 2): 121.6 us
    # first, 106.6 us later in the sweep on the same 34816 x 5120 copies)
    graph_time_ms(lambda i: quant_kernels.w8a16_group_gemv(x, q[i], s[i]), copies, iters, device)
    row["triton_us"] = graph_time_ms(lambda i: quant_kernels.w8a16_group_gemv(x, q[i], s[i]), copies, iters, device) * 1e3
    if triton_sweep:
        from tune_qwen3_5_group_gemv import GRID

        times = {}
        for block_n, block_k, warps in GRID:
            if k % block_k or block_n * block_k < 256 * warps:  # the tuning script's filter
                continue
            quant_kernels.LAUNCH_CONFIGS[n, k, group] = (block_n, block_k, warps)
            times[block_n, block_k, warps] = graph_time_ms(
                lambda i: quant_kernels.w8a16_group_gemv(x, q[i], s[i]), copies, iters, device) * 1e3
        del quant_kernels.LAUNCH_CONFIGS[n, k, group]
        best = min(times, key=times.get)
        row.update(triton_best=best, triton_best_us=times[best], triton_top5=sorted(times.items(), key=lambda kv: kv[1])[:5],
                   triton_all_us={str(c): t for c, t in times.items()})

    rule = quant_cuda.cuda_launch_config(n, k, group)
    times = {}
    for config in dict.fromkeys([rule, *quant_cuda.CONFIGS]):
        quant_cuda.CUDA_LAUNCH_CONFIGS[n, k, group] = config
        times[config] = graph_time_ms(lambda i: quant_cuda.w8a16_group_gemv_cuda(x, q[i], s[i]), copies, iters, device) * 1e3
    best = min(times, key=times.get)
    quant_cuda.CUDA_LAUNCH_CONFIGS[n, k, group] = best
    got = quant_cuda.w8a16_group_gemv_cuda(x, q[0], s[0]).float()
    del quant_cuda.CUDA_LAUNCH_CONFIGS[n, k, group]
    row["cuda_rel_err"] = ((got - ref).norm() / ref.norm()).item()
    row["cuda_vs_triton_rel"] = ((got - tri).norm() / tri.norm()).item()
    row.update(cuda_rule=rule, cuda_rule_us=times[rule], cuda_best=best, cuda_best_us=times[best],
               cuda_top5=sorted(times.items(), key=lambda kv: kv[1])[:5], cuda_all_us={str(c): t for c, t in times.items()})
    for name in ("triton", "triton_best", "cuda_rule", "cuda_best"):
        if f"{name}_us" in row:
                row[f"{name}_GBps"] = nbytes / row[f"{name}_us"] / 1e3
    del q, s
    if bf16:  # cuBLAS on the bf16 matrix of the same shape, for scale
        w = [torch.empty(n, k, device=device, dtype=dtype).normal_(0, 0.02) for _ in range(max(2, -(-min_footprint // (2 * n * k))))]
        row["bf16_us"] = graph_time_ms(lambda i: F.linear(x, w[i]), len(w), iters, device) * 1e3
        del w
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--config", default=None, help="config.json instead of downloading the repo's")
    parser.add_argument("--fuse", action="store_true", help="the shapes after fuse_projections()")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--min-footprint-mib", type=int, default=512, help="int8 bytes of the weight copies per shape")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-bf16", action="store_true", help="skip the cuBLAS bf16 reference")
    parser.add_argument("--triton-sweep", action="store_true", help="also time the Triton kernel over its tuning grid")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if args.config:
        config_path = args.config
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(args.repo, "config.json")
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig.from_json(config_path))
        if args.fuse:
            model.fuse_projections()
    group = args.group_size
    shapes = Counter((m.out_features, m.in_features) for m in model.modules()
                     if isinstance(m, nn.Linear) and m.out_features >= 256 and m.in_features % group == 0)
    info = device_summary(device)
    print(f"{info['name']}  torch {info['torch']}  {dtype}  group {group}  fused {args.fuse}", flush=True)
    quant_cuda._extension()  # build before timing

    print(f"{'N x K':>16}{'n':>4}{'MB':>8}{'bf16':>8}{'triton':>8}{'tri best':>9}{'cuda':>8}{'best':>8}  {'best cfg':<14}"
          f"{'tri GB/s':>9}{'cuda GB/s':>10}{'cuda/tri':>9}", flush=True)
    rows = []
    totals = Counter()
    for (n, k), count in sorted(shapes.items(), key=lambda kv: -kv[0][0] * kv[0][1] * kv[1]):
        r = bench_shape(n, k, group, dtype, device, args.min_footprint_mib * MIB, args.iters, not args.no_bf16,
                        args.triton_sweep)
        r["count"] = count
        rows.append(r)
        for name in ("triton", "triton_best", "cuda_rule", "cuda_best", "bf16"):
            totals[name] += r.get(f"{name}_us", 0.0) * count
        print(f"{n:>7} x {k:<6}{count:>4}{r['MB']:>8.1f}{r.get('bf16_us', float('nan')):>8.1f}{r['triton_us']:>8.1f}"
              f"{r.get('triton_best_us', float('nan')):>9.1f}{r['cuda_rule_us']:>8.1f}{r['cuda_best_us']:>8.1f}  {str(r['cuda_best']):<14}{r['triton_GBps']:>9.0f}"
              f"{r['cuda_best_GBps']:>10.0f}{r['cuda_best_us'] / r['triton_us']:>9.3f}", flush=True)

    print(f"\nsum over the model's Linears (us per token): " + ", ".join(f"{k} {v:.0f}" for k, v in totals.items()))
    print(f"cuda (rule) / triton {totals['cuda_rule'] / totals['triton']:.3f}, "
          f"cuda (best per shape) / triton {totals['cuda_best'] / totals['triton']:.3f}")
    if args.triton_sweep:
        print(f"cuda (rule) / triton (best per shape) {totals['cuda_rule'] / totals['triton_best']:.3f}, "
              f"cuda (best) / triton (best) {totals['cuda_best'] / totals['triton_best']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "repo": args.repo, "device": info, "dtype": str(dtype), "group_size": group, "fuse": args.fuse,
            "rows": rows, "totals_us": totals, "best": [[r["n"], r["k"], group, *r["cuda_best"]] for r in rows]}, indent=1))


if __name__ == "__main__":
    main()
