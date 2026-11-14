from __future__ import annotations

import torch


class KVCache:
    """Preallocated per-layer KV storage with one shared sequence position."""

    def __init__(
        self,
        *,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.position = 0
        shape = (n_layers, batch_size, n_kv_heads, max_seq_len, head_dim)
        self.keys = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty_like(self.keys)

    @property
    def batch_size(self) -> int:
        return self.keys.shape[1]

    def reset(self) -> None:
        self.position = 0

    def validate_append(self, token_count: int) -> None:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.position + token_count > self.max_seq_len:
            raise ValueError(
                f"KV cache capacity exceeded: {self.position + token_count} > {self.max_seq_len}"
            )

    def write(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, start: int) -> None:
        if start != self.position:
            raise ValueError("all layers must write at the current cache position")
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("key and value must be rank-4 tensors with equal shapes")
        if key.shape[0] != self.batch_size:
            raise ValueError("cache batch size does not match key/value batch size")
        end = start + key.shape[-2]
        self.keys[layer_idx, :, :, start:end].copy_(key)
        self.values[layer_idx, :, :, start:end].copy_(value)

    def read(self, layer_idx: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.keys[layer_idx, :, :, :end], self.values[layer_idx, :, :, :end]

    def advance(self, token_count: int) -> None:
        self.validate_append(token_count)
        self.position += token_count
