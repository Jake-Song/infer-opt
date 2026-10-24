"""Inductor heuristics for one-token decode: persistent reductions for wide rows.

Inductor keeps a reduction in registers in one pass ("persistent") only when it is at most 1024 wide; a wider one is
a loop that reads the row twice (sum of squares, then normalize). Decode reduces single rows that are hidden_size
wide (5120 on Qwen3.5-27B), so every RMSNorm becomes a looped kernel run by one program. On the int8 27B (A100) the
128 residual-add + RMSNorm kernels of a step took 0.87 ms looped and 0.40 ms persistent, and decode went from 20.31 to
19.91 ms/token. Allowing it only for single rows made the 48-row in_proj_a/b reductions that inductor fuses with a
norm three times slower, so the rule is by width and row count: lm_head (248 320 rows) stays looped (35 us faster).

:func:`wide_persistent_reductions` applies the rule for the duration of a ``with`` block; compilation happens on the
first call of a ``torch.compile``'d function, so that call has to be inside it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch._inductor.config as inductor_config
from torch._inductor.choices import InductorChoices
from torch._inductor.runtime.hints import ReductionHint
from torch._inductor.virtualized import V

__all__ = ["WidePersistentReductions", "wide_persistent_reductions"]

MAX_WIDTH = 8192
MAX_ROWS = 1024


class WidePersistentReductions(InductorChoices):
    """Inductor's choices, plus persistent inner reductions up to ``MAX_WIDTH`` wide over at most ``MAX_ROWS`` rows."""

    @staticmethod
    def should_use_persistent_reduction(features, cooperative_reduction) -> bool:
        if InductorChoices.should_use_persistent_reduction(features, cooperative_reduction):
            return True
        if cooperative_reduction or not inductor_config.triton.persistent_reductions:
            return False
        if features.get_reduction_hint() != ReductionHint.INNER:
            return False
        sizes = V.graph.sizevars
        # features.numel counts the rows (the non-reduced elements)
        return sizes.statically_known_leq(features.reduction_numel, MAX_WIDTH) and sizes.statically_known_leq(
            features.numel, MAX_ROWS)

    def uuid(self) -> str:  # part of the FX graph cache key on newer torch
        return f"wide-persistent-reductions-{MAX_WIDTH}-{MAX_ROWS}"


@contextlib.contextmanager
def wide_persistent_reductions() -> Iterator[None]:
    """Compile with :class:`WidePersistentReductions`. The handler makes the decisions; the config entry puts the class
    in the FX graph cache key, so a kernel compiled without the rule is not reused (inductor caches its default
    choices per thread on first use, so the config entry alone would not take effect after an earlier compile)."""
    with inductor_config.patch(inductor_choices_class=WidePersistentReductions), \
            V.set_choices_handler(WidePersistentReductions()):
        yield
