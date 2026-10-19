"""Does torch.compile fuse the int8 -> bf16 conversion into the batch-1 matmul?  (go/no-go gate for INT8)

For every distinct Linear shape of a Qwen3.5 model this times, with the weight coming from HBM
(rotating copies, replayed from a CUDA graph, as in measure_qwen3_5_bandwidth.py):

  bf16        F.linear(x, w)
  int8-eager  Int8Linear, no compile  (materializes a bf16 weight: expected to be slower than bf16)
  int8-comp   Int8Linear.forward under torch.compile

and reports microseconds, the bandwidth reached on the bytes that variant must read, and the speedup over
bf16. A fused kernel reads one byte per weight, so its time should approach half the bf16 time on the large
shapes. If int8-comp stays near or above the bf16 time, inductor did not fuse the conversion.
``--group-size 128`` times the group-wise variant, whose dequantization (cast and group-scale multiply) has to be
fused into the matmul's input instead.

    python scripts/bench_int8_linear.py --repo Qwen/Qwen3.5-9B --out int8_linear.json
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
from infer_opt.profiling import device_summary, timeit  # noqa: E402
from infer_opt.quant import Int8Linear, int8_linear, quantize_int8  # noqa: E402
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

MIB = 2**20


def graph_time_ms(call, copies: int, iters: int, device: torch.device) -> float:
    """Median per-call time of ``call(i)`` for i in range(copies), replayed from one CUDA graph."""
    for i in range(copies):  # compile / autotune / cuBLAS heuristics happen here, outside the capture
        call(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(copies):
            call(i)
    t = timeit(graph.replay, device=device, warmup=3, iters=iters)
    del graph
    return t.ms / copies


@torch.inference_mode()
def bench_shape(out_f: int, in_f: int, device, compile_mode: str | None, min_footprint: int, iters: int,
                group_size: int | None = None) -> dict:
    dtype = torch.bfloat16
    bf16_bytes = out_f * in_f * 2
    copies = max(2, -(-min_footprint // bf16_bytes))
    w = [torch.empty(out_f, in_f, device=device, dtype=dtype).normal_(0, 0.02) for _ in range(copies)]
    q = [Int8Linear(*quantize_int8(x, group_size)) for x in w]
    x = torch.randn(1, 1, in_f, device=device, dtype=dtype)

    # numeric check on copy 0: int8 path against the bf16 weight it was made from
    ref = F.linear(x, w[0]).float()
    err = ((q[0](x).float() - ref).norm() / ref.norm()).item()

    row = {"out": out_f, "in": in_f, "rel_err": err}
    row["bf16_us"] = graph_time_ms(lambda i: F.linear(x, w[i]), copies, iters, device) * 1e3
    row["int8_eager_us"] = graph_time_ms(lambda i: q[i](x), copies, iters, device) * 1e3
    if compile_mode != "none":
        # tensors in, tensors out: every weight copy reuses one compiled kernel instead of guarding on a module
        torch._dynamo.reset()
        fn = torch.compile(int8_linear, mode=None if compile_mode == "default" else compile_mode, dynamic=False)
        row["int8_comp_us"] = graph_time_ms(lambda i: fn(x, q[i].weight, q[i].scale), copies, iters, device) * 1e3
    del w, q
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--config", default=None)
    parser.add_argument("--compile-mode", default="default",
                        help="torch.compile mode: default, max-autotune-no-cudagraphs, or none to skip compiling")
    parser.add_argument("--min-footprint-mib", type=int, default=512)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--skip-below", type=int, default=256, help="ignore shapes with fewer output features")
    parser.add_argument("--group-size", type=int, default=0, help="0: one scale per output channel")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    if args.config:
        config_path = args.config
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(args.repo, "config.json")
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig.from_json(config_path))
    shapes = Counter((m.out_features, m.in_features) for m in model.modules() if isinstance(m, nn.Linear))
    shapes = {k: v for k, v in shapes.items() if k[0] >= args.skip_below}
    info = device_summary(device)
    group_size = args.group_size or None
    print(f"{info['name']}  torch {info['torch']}  compile mode: {args.compile_mode}  group size: {group_size}")

    header = f"{'out x in':>16}{'n':>5}{'MiB bf16':>10}{'bf16 us':>9}{'i8 eager':>10}{'i8 comp':>9}{'comp GB/s':>10}{'speedup':>9}{'rel err':>9}"
    print(header)
    rows, totals = [], {"bf16": 0.0, "eager": 0.0, "comp": 0.0}
    for (out_f, in_f), count in sorted(shapes.items(), key=lambda kv: -kv[0][0] * kv[0][1] * kv[1]):
        r = bench_shape(out_f, in_f, device, args.compile_mode, args.min_footprint_mib * MIB, args.iters, group_size)
        r["count"] = count
        comp = r.get("int8_comp_us")
        if comp:
            scale_bytes = 2 * out_f * (1 if group_size is None else in_f // group_size)
            r["int8_comp_gbps"] = (out_f * in_f + scale_bytes) / (comp * 1e-6) / 1e9  # one byte per weight + scales
            r["speedup"] = r["bf16_us"] / comp
        rows.append(r)
        totals["bf16"] += r["bf16_us"] * count
        totals["eager"] += r["int8_eager_us"] * count
        totals["comp"] += (comp or 0) * count
        print(f"{out_f:>8} x {in_f:<5}{count:>5}{out_f * in_f * 2 / MIB:>10.1f}{r['bf16_us']:>9.1f}"
              f"{r['int8_eager_us']:>10.1f}{(comp or float('nan')):>9.1f}{r.get('int8_comp_gbps', float('nan')):>10.0f}"
              f"{r.get('speedup', float('nan')):>8.2f}x{r['rel_err']:>9.4f}")

    print(f"\nsum over layers (shapes >= {args.skip_below} outputs): bf16 {totals['bf16'] / 1e3:.2f} ms | "
          f"int8-eager {totals['eager'] / 1e3:.2f} ms | int8-comp {totals['comp'] / 1e3:.2f} ms")
    if totals["comp"]:
        print(f"-> compiled INT8 is {totals['bf16'] / totals['comp']:.2f}x the bf16 time; fused would be ~1.6-1.9x")

    if args.out:
        Path(args.out).write_text(json.dumps({"repo": args.repo, "device": info, "compile_mode": args.compile_mode, "group_size": group_size,
                                              "rows": rows, "totals_us": totals}, indent=1))


if __name__ == "__main__":
    main()
