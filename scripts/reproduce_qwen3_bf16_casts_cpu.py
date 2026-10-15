#!/usr/bin/env python3
"""CPU mechanism test with actual Qwen3 weights and saved input token IDs.

Downloads only first-layer norm/projection weights and the 30 used embedding
rows using validated HTTP byte ranges. Does not load a full model or use a GPU.
This is explicitly not a reproduction of A100/PyTorch 2.11 full-model results.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import platform
import time
import urllib.request
from pathlib import Path

REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
BASE = f"https://huggingface.co/Qwen/Qwen3-14B/resolve/{REVISION}/"
SHARD = "model-00001-of-00008.safetensors"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/qwen3-bf16-cpu-inductor")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    import torch
    import torch.nn.functional as F
    from torch._inductor.utils import run_and_get_code
    torch.set_num_threads(8)
    helper_path = Path(__file__).with_name("measure_qwen3_flash_compile.py")
    spec = importlib.util.spec_from_file_location("qwen_measurement_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    fixture = json.loads(args.fixture.read_text())[0]
    ids = fixture["prompt_ids"]
    header_info = json.loads((args.output / "shard-header.json").read_text())
    header = header_info["tensors"]
    offset = header_info["header_length"] + 8
    config = json.loads((args.output / "config.json").read_text())
    manifest_path = args.output / "partial_weights.json"

    def download_range(start, end, path):
        expected = end - start + 1
        if not path.exists():
            if not args.download:
                raise RuntimeError(f"Missing {path}; use --download once")
            request = urllib.request.Request(BASE + SHARD, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError("Server did not honor exact HTTP range; refuse full shard download")
                data = response.read(expected + 1)
            if len(data) != expected:
                raise RuntimeError("Invalid partial tensor length")
            path.write_bytes(data)
        data = path.read_bytes()
        if len(data) != expected:
            raise RuntimeError(f"Corrupt cached tensor {path}")
        return {"file": path.name, "range": [start, end], "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}

    tasks, descriptors = [], {}
    for label, name in (("input_norm", "model.layers.0.input_layernorm.weight"),
                        ("key_projection", "model.layers.0.self_attn.k_proj.weight"),
                        ("key_norm", "model.layers.0.self_attn.k_norm.weight")):
        descriptor = header[name]
        if descriptor["dtype"] != "BF16":
            raise ValueError("Expected BF16 checkpoint weights")
        start, end = descriptor["data_offsets"]
        path = args.output / f"{label}.bf16"
        tasks.append((offset + start, offset + end - 1, path))
        descriptors[label] = {"shape": descriptor["shape"], "file": path.name}
    embedding = header["model.embed_tokens.weight"]
    row_bytes = embedding["shape"][1] * 2
    for token in sorted(set(ids)):
        start = offset + embedding["data_offsets"][0] + token * row_bytes
        tasks.append((start, start + row_bytes - 1, args.output / f"embedding-{token}.bf16"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        manifest = list(pool.map(lambda task: download_range(*task), tasks))
    manifest_path.write_text(json.dumps({"model": "Qwen/Qwen3-14B", "revision": REVISION,
                                        "shard": SHARD, "total_bytes": sum(x["bytes"] for x in manifest),
                                        "ranges": manifest}, indent=2) + "\n")
    print(f"Partial weights loaded: {sum(x['bytes'] for x in manifest)} bytes", flush=True)

    def load(name, shape):
        data = bytearray((args.output / name).read_bytes())
        return torch.frombuffer(data, dtype=torch.bfloat16).clone().reshape(shape)

    weights = {key: load(d["file"], d["shape"]) for key, d in descriptors.items()}
    hidden = torch.stack([load(f"embedding-{token}.bf16", (config["hidden_size"],)) for token in ids]).unsqueeze(0)
    eps = config["rms_norm_eps"]
    head_dim, heads = config["head_dim"], config["num_key_value_heads"]

    def rmsnorm(value, weight):
        dtype = value.dtype
        promoted = value.to(torch.float32)
        variance = promoted.pow(2).mean(-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + eps)
        return weight * normalized.to(dtype)

    def input_norm(value):
        return rmsnorm(value, weights["input_norm"])

    def key_norm(value):
        return rmsnorm(value, weights["key_norm"])

    inv_freq = 1.0 / (config["rope_theta"] ** (torch.arange(0, head_dim, 2).float() / head_dim))
    frequency = torch.arange(len(ids)).float()[None, :, None] * inv_freq[None, None, :]
    embedding = torch.cat((frequency, frequency), -1)
    cos = embedding.cos().to(torch.bfloat16).unsqueeze(1)
    sin = embedding.sin().to(torch.bfloat16).unsqueeze(1)

    def rotate_half(value):
        first, last = value.chunk(2, dim=-1)
        return torch.cat((-last, first), dim=-1)

    def rope(value):
        return value * cos + rotate_half(value) * sin

    def key_path(value):
        value = input_norm(value)
        value = F.linear(value, weights["key_projection"]).view(1, len(ids), heads, head_dim)
        value = key_norm(value).transpose(1, 2)
        return rope(value)

    result = {"scope": "CPU mechanism reproduction only; not A100 or full-model validation",
              "torch": str(torch.__version__), "platform": platform.platform(),
              "device": "cpu", "model_revision": REVISION,
              "input_ids": ids, "weight_bytes": sum(x["bytes"] for x in manifest),
              "dtype": "bfloat16", "atol": 0.02, "rtol": 0.02, "operations": {}}
    started = time.perf_counter()
    with torch.inference_mode():
        projected = F.linear(input_norm(hidden), weights["key_projection"]).view(1, len(ids), heads, head_dim)
        normalized_key = key_norm(projected).transpose(1, 2)
        operations = [("input_rmsnorm", input_norm, hidden), ("key_rmsnorm", key_norm, projected),
                      ("rope", rope, normalized_key), ("first_layer_key_path", key_path, hidden)]
        for name, function, value in operations:
            reference = function(value)
            record = {"shape": list(reference.shape), "reference_absmax": reference.abs().max().item()}
            for preserve in (False, True):
                label = "preserve_casts" if preserve else "default"
                compiled = torch.compile(function, fullgraph=True, dynamic=False,
                                         options={"emulate_precision_casts": preserve})
                with torch._inductor.config.patch({"fx_graph_cache": False}):
                    actual, generated = run_and_get_code(compiled, value)
                record[label] = helper.compare(actual, reference)
                (args.output / f"cpu-{name}-{label}.txt").write_text("\n\n".join(generated))
                if name == "rope":
                    # Product rounding is the only deliberate difference in this
                    # eager control; it leaves the same rotation and coefficients.
                    cast_elided = (value.float()*cos.float() + rotate_half(value).float()*sin.float()).to(value.dtype)
                    record[label + "_vs_cast_elided_eager"] = helper.compare(actual, cast_elided)
            result["operations"][name] = record
            helper.write_json(args.output / "cpu_diagnosis.json", result)
            print(json.dumps({"operation": name, **record}), flush=True)
        torch.save({"hidden": hidden, "projected_key": projected, "key_norm": normalized_key,
                    "cos": cos, "sin": sin}, args.output / "cpu-inputs.pt")
    result["elapsed_seconds"] = time.perf_counter() - started
    helper.write_json(args.output / "cpu_diagnosis.json", result)


if __name__ == "__main__":
    main()
