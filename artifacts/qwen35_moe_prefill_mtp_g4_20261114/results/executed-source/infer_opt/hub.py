"""Read a few tensors out of safetensors shards without downloading the shards.

A safetensors file is an 8-byte little-endian header length, a JSON header (``{name: {"dtype", "shape",
"data_offsets": [start, end]}}``, offsets relative to the end of the header) and the raw data. So the tensors of one
module can be read with one ranged read each. On the Hugging Face Hub that is an HTTP range request through
``HfFileSystem``: the MTP head of Qwen3.5-27B (15 tensors, 0.8 GB in bf16) is spread over 4 of its 11 shards (17 GB),
and only its bytes are fetched.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import BinaryIO

import torch

__all__ = ["read_safetensors", "fetch_hub_tensors"]

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I8": torch.int8, "I32": torch.int32,
           "I64": torch.int64, "U8": torch.uint8, "BOOL": torch.bool}


def read_safetensors(f: BinaryIO, prefix: str) -> dict[str, torch.Tensor]:
    """The tensors of an open safetensors file whose name starts with ``prefix`` (CPU tensors, one read each)."""
    f.seek(0)
    (header_len,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(header_len))
    base = 8 + header_len
    header.pop("__metadata__", None)
    wanted = sorted((entry["data_offsets"], name, entry) for name, entry in header.items() if name.startswith(prefix))
    out = {}
    for (start, end), name, entry in wanted:  # in file order, so the reads move forward
        f.seek(base + start)
        data = bytearray(f.read(end - start))
        dtype = _DTYPES[entry["dtype"]]
        out[name] = (torch.frombuffer(data, dtype=dtype).reshape(entry["shape"]) if data
                     else torch.empty(entry["shape"], dtype=dtype))
    return out


def fetch_hub_tensors(repo_id: str, prefix: str, revision: str | None = None) -> dict[str, torch.Tensor]:
    """The tensors named ``prefix*`` of a sharded Hub checkpoint (``model.safetensors.index.json``), read with range
    requests from the shards that hold them."""
    from huggingface_hub import HfFileSystem, hf_hub_download

    index = json.loads(Path(hf_hub_download(repo_id, "model.safetensors.index.json", revision=revision)).read_text())
    shards = sorted({shard for name, shard in index["weight_map"].items() if name.startswith(prefix)})
    fs = HfFileSystem()
    out = {}
    for shard in shards:
        with fs.open(f"{repo_id}/{shard}", "rb", revision=revision) as f:  # seekable; reads ahead one block at most
            out.update(read_safetensors(f, prefix))
    return out
