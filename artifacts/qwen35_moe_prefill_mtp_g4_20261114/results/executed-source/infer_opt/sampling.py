"""Token selection, the last operation of a decode step.

Sampling looks like an afterthought next to sixteen transformer layers, and in
FLOPs it is one. In time it is not: top-p sorts the whole vocabulary, and a
sort over 32000 logits costs about as much as an entire decode step's
attention. That gap is the reason this module exists -- a per-operation
breakdown that stopped at the LM head would miss it.

These are the straightforward implementations, deliberately unfused, so the
measurements the curriculum takes describe the algorithms rather than one
particular kernel's cleverness.
"""

from __future__ import annotations

import torch

__all__ = ["greedy", "sample_top_k", "sample_top_p", "last_token_logits"]


def last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    """The ``[batch, vocab]`` slice a sampler acts on, from ``[batch, tokens, vocab]``."""
    if logits.ndim == 2:
        return logits
    if logits.ndim != 3:
        raise ValueError("logits must be [batch, vocab] or [batch, tokens, vocab]")
    return logits[:, -1]


def _validate(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("sampling expects [batch, vocab]; use last_token_logits first")
    return logits


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Take the highest-scoring token. One reduction, no sort, no randomness."""
    return _validate(logits).argmax(dim=-1, keepdim=True)


def sample_top_k(
    logits: torch.Tensor,
    *,
    k: int,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample from the ``k`` highest-scoring tokens.

    ``topk`` is a partial selection rather than a full sort, so this stays
    cheaper than :func:`sample_top_p` however large the vocabulary grows.
    """
    logits = _validate(logits)
    if k <= 0:
        raise ValueError("k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    k = min(k, logits.shape[-1])
    values, indices = logits.topk(k, dim=-1)
    probabilities = torch.softmax(values.float() / temperature, dim=-1)
    choice = torch.multinomial(probabilities, num_samples=1, generator=generator)
    return indices.gather(-1, choice)


def sample_top_p(
    logits: torch.Tensor,
    *,
    p: float,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample from the smallest set of tokens whose probability mass reaches ``p``.

    The candidate set is data-dependent, so unlike top-k it cannot be found
    without ranking every token: the sort over the full vocabulary is the cost,
    and it does not shrink when the distribution is sharp.
    """
    logits = _validate(logits)
    if not 0 < p <= 1:
        raise ValueError("p must lie in (0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ordered, indices = (logits.float() / temperature).sort(dim=-1, descending=True)
    probabilities = torch.softmax(ordered, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    # Keep every token up to and including the one that crosses p, so the
    # candidate set is never empty even when one token already exceeds it.
    excess = cumulative - probabilities > p
    probabilities = probabilities.masked_fill(excess, 0.0)
    choice = torch.multinomial(probabilities, num_samples=1, generator=generator)
    return indices.gather(-1, choice)
