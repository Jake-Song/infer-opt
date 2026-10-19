"""Launch configurations of the group-wise INT8 GEMV on the real weights of a Qwen3.5 model, and decode with them.

For every Int8Linear shape (group-wise quantization, lm_head included), one CUDA graph calls the kernel on every
module of that shape in turn, so the weights together exceed L2 as in a decode step, for each candidate
``(BLOCK_N, UNROLL, num_warps, num_stages)``. Prints the best configuration per shape next to the one
``infer_opt.quant_kernels.launch_config`` picks, then decode ms/token (fixed-shape decoder, inductor max-autotune)
with the rule and with the measured best configurations.

    python scripts/tune_qwen3_5_group_gemv.py --repo Qwen/Qwen3.5-9B --out group_gemv_tune.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_int8_linear import graph_time_ms  # noqa: E402
from bench_qwen3_5_decode import time_decode  # noqa: E402
from infer_opt import quant_kernels  # noqa: E402
from infer_opt.quant import Int8Linear, quantize_linears  # noqa: E402
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

GRID = list(itertools.product((2, 4, 8, 16, 32), (1, 2, 4, 8), (4, 8), (2, 3)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--kernels", default="fla-delta")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--out", default="group_gemv_tune.json")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    path = args.weights or snapshot_download(args.repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    model = Qwen3_5ForCausalLM.from_pretrained_dir(path, dtype=torch.bfloat16, device="cuda", kernels=args.kernels)
    print(quantize_linears(model, include_lm_head=True, group_size=args.group_size).summary(), flush=True)
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    by_shape = defaultdict(list)
    for module in model.modules():
        if isinstance(module, Int8Linear) and module.group_size:
            by_shape[(module.out_features, module.in_features, module.group_size)].append(module)

    result = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "shapes": {}}
    best_table = {}
    with torch.no_grad():
        for key, modules in sorted(by_shape.items(), key=lambda kv: -kv[0][0] * kv[0][1] * len(kv[1])):
            n, k, group = key
            x = torch.randn(1, 1, k, device=device, dtype=torch.bfloat16)
            nbytes = n * k * (1 + 2 / group)

            def run(i):
                return quant_kernels.w8a16_group_gemv(x, modules[i].weight, modules[i].scale)

            rule = quant_kernels.launch_config(n, k, group)
            times = {}
            for config in dict.fromkeys([rule, *GRID]):
                block_n, unroll, _, _ = config
                if block_n * unroll * group > 16384:
                    continue
                quant_kernels.LAUNCH_CONFIGS[key] = config
                try:
                    times[config] = graph_time_ms(run, len(modules), args.iters, device) * 1e3
                except Exception:  # noqa: BLE001 - out of resources for large tiles
                    pass
            del quant_kernels.LAUNCH_CONFIGS[key]
            ranked = sorted(times.items(), key=lambda kv: kv[1])
            best, best_us = ranked[0]
            best_table[key] = best
            result["shapes"][str(key)] = {
                "count": len(modules), "rule": rule, "rule_us": times.get(rule), "best": best, "best_us": best_us,
                "best_GBps": nbytes / best_us / 1e3, "top5": ranked[:5],
            }
            print(f"{n:>7} x {k:<6} x{len(modules):<3} best {best} {best_us:7.1f} us ({nbytes / best_us / 1e3:5.0f} GB/s)"
                  f"  rule {rule} {times.get(rule, float('nan')):7.1f} us", flush=True)

        import torch._inductor.config as inductor_config

        inductor_config.max_autotune = True
        inductor_config.coordinate_descent_tuning = True
        torch.manual_seed(0)
        ids = torch.randint(1000, 100000, (1, args.prompt_tokens), device=device)
        max_len = args.prompt_tokens + args.new_tokens + 16
        for label, table in (("rule", {}), ("best", best_table)):
            quant_kernels.LAUNCH_CONFIGS.clear()
            quant_kernels.LAUNCH_CONFIGS.update(table)
            torch._dynamo.reset()
            result[f"decode_{label}"] = time_decode(model, ids, max_len, "compile-reduce-overhead", args.new_tokens, 3)
            print(label, json.dumps(result[f"decode_{label}"]), flush=True)
        quant_kernels.LAUNCH_CONFIGS.clear()
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
