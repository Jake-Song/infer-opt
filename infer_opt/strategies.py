from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class AttentionStrategy(ABC):
    """Replace this unit to evaluate alternative attention implementations."""

    name: str

    @abstractmethod
    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, query_start: int
    ) -> torch.Tensor: ...


def causal_mask(query: torch.Tensor, key: torch.Tensor, query_start: int) -> torch.Tensor:
    """Which keys each query is allowed to see, as a ``[query_len, key_len]`` bool.

    ``query_start`` is what makes one mask serve both phases: during decode the
    single query sits at the end of the cache, not at row zero.
    """
    query_len, key_len = query.shape[-2], key.shape[-2]
    q_positions = torch.arange(query_start, query_start + query_len, device=query.device)
    k_positions = torch.arange(key_len, device=query.device)
    return k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)


def expand_kv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise one K/V head per query head, the way plain MHA kernels want them.

    Under GQA this writes out ``n_heads / n_kv_heads`` copies of tensors the
    kernel could have re-read in place, which is bandwidth spent to satisfy an
    interface rather than to compute anything.
    """
    groups = query.shape[1] // key.shape[1]
    if groups == 1:
        return key, value
    return key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)


class EagerSDPAStrategy(AttentionStrategy):
    """The baseline: expand K/V, build an explicit mask, let PyTorch choose a kernel."""

    name = "eager_sdpa"

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, query_start: int
    ) -> torch.Tensor:
        key, value = expand_kv(query, key, value)
        mask = causal_mask(query, key, query_start)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)


class _PinnedBackendStrategy(AttentionStrategy):
    """Shared body for the strategies that force one SDPA backend."""

    backend: SDPBackend

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, query_start: int
    ) -> torch.Tensor:
        key, value = expand_kv(query, key, value)
        mask = causal_mask(query, key, query_start)
        with sdpa_kernel(self.backend):
            return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)


class MathSDPAStrategy(_PinnedBackendStrategy):
    """Unfused attention: the whole ``query_len x key_len`` score matrix is built.

    Correct, portable, and quadratic in memory. It is the control the
    FlashAttention family is measured against.
    """

    name = "sdpa_math"
    backend = SDPBackend.MATH


class MemEfficientSDPAStrategy(_PinnedBackendStrategy):
    """Tiled attention that never materialises the score matrix.

    Same family as FlashAttention -- block over keys, keep a running softmax
    maximum and sum, accumulate the output -- so its memory grows with context
    rather than with context squared. PyTorch's own FlashAttention backend
    needs sm80 or newer; this cutlass kernel is what an older card gets.
    """

    name = "sdpa_mem_efficient"
    backend = SDPBackend.EFFICIENT_ATTENTION


class NativeGQAStrategy(AttentionStrategy):
    """Let the kernel handle the head grouping instead of copying K/V out.

    Identical arithmetic to :class:`EagerSDPAStrategy`; the difference is the
    ``repeat_interleave`` that no longer happens.
    """

    name = "gqa_native"

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, query_start: int
    ) -> torch.Tensor:
        mask = causal_mask(query, key, query_start)
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, enable_gqa=query.shape[1] != key.shape[1]
        )


class DecodeNoMaskStrategy(AttentionStrategy):
    """Decode attention without a mask, because decode does not need one.

    A causal mask exists to stop a query from reading tokens that come after
    it. During decode there is exactly one query and it sits at the end of the
    cache, so every cached key is already in its past and nothing is masked
    out. Building the mask, shipping it to the device and testing it per
    element is pure overhead -- and it is overhead that grows with context
    length, in the phase that is already the bottleneck.
    """

    name = "decode_nomask"

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, query_start: int
    ) -> torch.Tensor:
        if query.shape[-2] != 1:
            raise ValueError(
                "decode_nomask handles one query token; prefill still needs a causal mask"
            )
        if query_start + 1 != key.shape[-2]:
            raise ValueError("decode_nomask expects the query to sit at the end of the cache")
        return F.scaled_dot_product_attention(
            query, key, value, enable_gqa=query.shape[1] != key.shape[1]
        )


_REGISTRY: dict[str, type[AttentionStrategy]] = {
    strategy.name: strategy
    for strategy in (
        EagerSDPAStrategy,
        MathSDPAStrategy,
        MemEfficientSDPAStrategy,
        NativeGQAStrategy,
        DecodeNoMaskStrategy,
    )
}


def register_attention_strategy(strategy: type[AttentionStrategy]) -> None:
    if not strategy.name:
        raise ValueError("attention strategy must define a name")
    if strategy.name in _REGISTRY:
        raise ValueError(f"attention strategy already registered: {strategy.name}")
    _REGISTRY[strategy.name] = strategy


def make_attention_strategy(name: str) -> AttentionStrategy:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"unknown attention strategy {name!r}; choices: {', '.join(sorted(_REGISTRY))}") from exc


def available_attention_strategies() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
