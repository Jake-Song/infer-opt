from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class AttentionStrategy(ABC):
    """Replace this unit to evaluate alternative attention implementations."""

    name: str

    @abstractmethod
    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, query_start: int
    ) -> torch.Tensor: ...


class EagerSDPAStrategy(AttentionStrategy):
    name = "eager_sdpa"

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, query_start: int
    ) -> torch.Tensor:
        groups = query.shape[1] // key.shape[1]
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        query_len, key_len = query.shape[-2], key.shape[-2]
        q_positions = torch.arange(query_start, query_start + query_len, device=query.device)
        k_positions = torch.arange(key_len, device=query.device)
        causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=causal_mask)


_REGISTRY: dict[str, type[AttentionStrategy]] = {EagerSDPAStrategy.name: EagerSDPAStrategy}


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
