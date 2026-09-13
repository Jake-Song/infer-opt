"""Regenerate browser parity fixtures from the authoritative Python implementation.

Run from the repository root:
    uv run python web/tests/generate_fixtures.py
"""

import json
from pathlib import Path

import torch

from infer_opt.config import ModelConfig
from infer_opt.profiling import (
    kv_cache_bytes,
    parameter_count,
    ridge_point,
    roofline_ceiling,
    transformer_bytes,
    transformer_flops,
)


def main() -> None:
    cases = []
    for hidden, batch, prompt, output, precision in [
        (1024, 1, 1024, 64, "float16"),
        (1024, 32, 1024, 64, "float16"),
        (1024, 1, 32768, 512, "float16"),
        (256, 2, 16, 1, "float32"),
        (512, 8, 256, 16, "bfloat16"),
        (2048, 64, 4096, 128, "float32"),
        (4096, 128, 32768, 512, "float16"),
    ]:
        config = ModelConfig(
            vocab_size=32000, hidden_size=hidden, n_layers=16,
            n_heads=16, n_kv_heads=4, intermediate_size=hidden * 11 // 4,
            max_seq_len=prompt + output,
        )
        dtype = getattr(torch, precision)
        steps = []
        for new_tokens, context_len in [(prompt, prompt), (1, prompt + 1), (1, prompt + output - 1)]:
            args = dict(batch=batch, new_tokens=new_tokens, context_len=context_len)
            flops = transformer_flops(config, **args)
            traffic = transformer_bytes(config, dtype=dtype, **args)
            intensity = flops["total"] / traffic["total"]
            steps.append(dict(
                new_tokens=new_tokens, context_len=context_len, flops=flops, traffic=traffic,
                intensity=intensity, ceiling=roofline_ceiling(intensity, 100, 1000),
            ))
        model = config.to_dict()
        del model["max_seq_len"], model["rms_norm_eps"]
        cases.append(dict(
            config=model, batch=batch, prompt=prompt, output=output, precision=precision,
            parameter_count=parameter_count(config),
            resident_kv=kv_cache_bytes(config, batch=batch, context_len=prompt + output - 1, dtype=dtype),
            ridge=ridge_point(100, 1000), steps=steps,
        ))
    target = Path(__file__).parents[1] / "src" / "fixtures.json"
    target.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Wrote {len(cases)} cases to {target}")


if __name__ == "__main__":
    main()
