"""KV cache stored in fixed-size blocks, the way PagedAttention arranges it.

:class:`~infer_opt.cache.KVCache` reserves ``max_seq_len`` positions for every
sequence the moment it is built. A server that must admit a 2048-token request
therefore pays 2048 tokens of memory for a request that turns out to be 300 --
and pays it per sequence, per layer, for the whole life of the batch. The
memory is not lost to a leak; it is lost to a promise nobody collected on.

Blocks replace that promise with an allocation. A sequence holds only the
blocks it has actually filled, drawn from one pool shared by the batch, and the
only waste left is the unused tail of its last block -- bounded by
``block_size`` rather than by ``max_seq_len``. The block table is the price:
one indirection from logical position to physical block.

The class satisfies the same ``write``/``read``/``advance`` contract as
:class:`~infer_opt.cache.KVCache`, so a model runs on either without changing a
line. What it does *not* yet do is let sequences in one batch reach different
lengths mid-flight; :attr:`PagedKVCache.lengths` tracks them individually, but
the model's decode path still advances the batch as a unit. Continuous
batching is what closes that gap.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

__all__ = ["BlockAllocator", "PagedKVCache", "blocks_for", "contiguous_reserved_bytes"]


def blocks_for(tokens: int, block_size: int) -> int:
    """Number of blocks needed to hold ``tokens`` positions."""
    return (tokens + block_size - 1) // block_size


def contiguous_reserved_bytes(
    *, n_layers: int, batch: int, n_kv_heads: int, max_seq_len: int, head_dim: int, element_size: int
) -> int:
    """Bytes :class:`~infer_opt.cache.KVCache` reserves up front, used or not."""
    return 2 * n_layers * batch * n_kv_heads * max_seq_len * head_dim * element_size


class BlockAllocator:
    """Hands out and reclaims fixed-size KV blocks from one shared pool.

    Deliberately a free list rather than a bitmap: the interesting property is
    that a freed block is immediately reusable by *any* sequence, which is what
    lets a server run more concurrent requests than a contiguous cache of the
    same size could hold.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self._free: list[int] = list(reversed(range(num_blocks)))

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    def allocate(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("count cannot be negative")
        if count > len(self._free):
            raise ValueError(
                f"KV block pool exhausted: asked for {count}, {len(self._free)} free"
            )
        return [self._free.pop() for _ in range(count)]

    def free(self, blocks: Iterable[int]) -> None:
        for block in blocks:
            if not 0 <= block < self.num_blocks:
                raise ValueError(f"block {block} is not from this pool")
            self._free.append(block)

    def reset(self) -> None:
        self._free = list(reversed(range(self.num_blocks)))


class PagedKVCache:
    """Paged KV storage with a per-sequence block table.

    Interchangeable with :class:`~infer_opt.cache.KVCache` from the model's
    point of view. ``num_blocks`` defaults to exactly what the batch would need
    at full length, which makes the two caches the same size and the comparison
    about *layout*; pass a smaller pool to model a server that oversubscribes.
    """

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
        block_size: int = 16,
        num_blocks: int | None = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.max_seq_len, self.block_size = max_seq_len, block_size
        self.position = 0
        self.blocks_per_seq = blocks_for(max_seq_len, block_size)
        self.num_blocks = num_blocks if num_blocks is not None else batch_size * self.blocks_per_seq
        self.allocator = BlockAllocator(self.num_blocks)
        shape = (n_layers, self.num_blocks, block_size, n_kv_heads, head_dim)
        self.keys = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty_like(self.keys)
        self.block_table = torch.zeros(
            (batch_size, self.blocks_per_seq), device=device, dtype=torch.long
        )
        self.lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
        self._allocated = 0

    @property
    def batch_size(self) -> int:
        return self.block_table.shape[0]

    @property
    def n_kv_heads(self) -> int:
        return self.keys.shape[3]

    @property
    def head_dim(self) -> int:
        return self.keys.shape[4]

    @property
    def n_layers(self) -> int:
        return self.keys.shape[0]

    def reset(self) -> None:
        self.position = 0
        self._allocated = 0
        self.lengths.zero_()
        self.block_table.zero_()
        self.allocator.reset()

    def validate_append(self, token_count: int) -> None:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.position + token_count > self.max_seq_len:
            raise ValueError(
                f"KV cache capacity exceeded: {self.position + token_count} > {self.max_seq_len}"
            )

    def _ensure_blocks(self, end: int) -> None:
        """Give every sequence enough blocks to reach position ``end``.

        Called from :meth:`write`, so the first layer of a step allocates and
        the remaining layers find the table already covering their positions --
        every layer shares one block table because the layer index is a
        separate dimension of the pool.
        """
        needed = blocks_for(end, self.block_size)
        if needed <= self._allocated:
            return
        for block_idx in range(self._allocated, needed):
            allocated = self.allocator.allocate(self.batch_size)
            self.block_table[:, block_idx] = torch.tensor(
                allocated, device=self.block_table.device, dtype=torch.long
            )
        self._allocated = needed

    def write(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, start: int) -> None:
        if start != self.position:
            raise ValueError("all layers must write at the current cache position")
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("key and value must be rank-4 tensors with equal shapes")
        if key.shape[0] != self.batch_size:
            raise ValueError("cache batch size does not match key/value batch size")
        tokens = key.shape[-2]
        self.validate_append(tokens)
        self._ensure_blocks(start + tokens)
        positions = torch.arange(start, start + tokens, device=self.keys.device)
        block_idx, offsets = positions // self.block_size, positions % self.block_size
        physical = self.block_table[:, block_idx]
        offsets = offsets.expand(self.batch_size, tokens)
        self.keys[layer_idx][physical, offsets] = key.permute(0, 2, 1, 3).to(self.keys.dtype)
        self.values[layer_idx][physical, offsets] = value.permute(0, 2, 1, 3).to(self.values.dtype)

    def read(self, layer_idx: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the first ``end`` positions back into ``[batch, kv_heads, end, head_dim]``.

        A real PagedAttention kernel reads the blocks where they lie and never
        builds this tensor. Doing it in plain PyTorch costs a copy, which is
        why the paged path measures slower here than a contiguous one; the
        memory argument survives that, the speed argument is the kernel's.
        """
        if not 0 < end <= self.max_seq_len:
            raise ValueError(f"cannot read {end} positions from a cache of {self.max_seq_len}")
        needed = blocks_for(end, self.block_size)
        if needed > self._allocated:
            raise ValueError(f"positions up to {end} have not been written yet")
        table = self.block_table[:, :needed]
        batch, span = self.batch_size, needed * self.block_size
        shape = (batch, span, self.n_kv_heads, self.head_dim)
        key = self.keys[layer_idx][table].reshape(shape).permute(0, 2, 1, 3)[:, :, :end]
        value = self.values[layer_idx][table].reshape(shape).permute(0, 2, 1, 3)[:, :, :end]
        return key, value

    def advance(self, token_count: int) -> None:
        self.validate_append(token_count)
        self.position += token_count
        self.lengths += token_count

    @property
    def allocated_blocks(self) -> int:
        """Blocks currently held by the batch, across all sequences."""
        return self._allocated * self.batch_size

    @property
    def bytes_per_token(self) -> int:
        """Bytes one cached position occupies, both tensors and every layer."""
        return (
            2 * self.n_layers * self.n_kv_heads * self.head_dim * self.keys.element_size()
        )

    @property
    def reserved_bytes(self) -> int:
        """Bytes the allocated blocks occupy, filled or not."""
        return self.allocated_blocks * self.block_size * self.bytes_per_token

    @property
    def used_bytes(self) -> int:
        """Bytes holding real cached positions."""
        return int(self.lengths.sum().item()) * self.bytes_per_token

    @property
    def wasted_bytes(self) -> int:
        """Internal fragmentation: the unfilled tail of each sequence's last block.

        Bounded by ``block_size - 1`` positions per sequence however long the
        context grows, which is the whole point of paging.
        """
        return self.reserved_bytes - self.used_bytes
