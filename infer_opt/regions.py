"""Named spans of the forward pass, for per-operation attribution.

A profiler sees kernels, not intentions: ``q_proj``, ``o_proj`` and the two
MLP branches all arrive as ``aten::mm`` and cannot be told apart afterwards.
Naming the spans while they run is what makes a per-operation breakdown
possible at all.

The annotations are off by default and cost nothing then -- :func:`region`
yields directly rather than entering ``record_function`` -- so the annotated
model runs at full speed everywhere except inside :func:`profiled_regions`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from torch.profiler import record_function

__all__ = ["REGION_NAMES", "region", "regions_enabled", "profiled_regions"]

#: Every span :mod:`infer_opt.model` annotates, in forward-pass order. The
#: analytic counterparts in :mod:`infer_opt.opprofile` use these same names.
REGION_NAMES: tuple[str, ...] = (
    "embedding",
    "norm",
    "qkv_proj",
    "rope",
    "kv_write",
    "kv_read",
    "attention",
    "o_proj",
    "mlp",
    "lm_head",
)

_PREFIX = "infer_opt::"
_ENABLED = False


def regions_enabled() -> bool:
    """Whether :func:`region` is currently emitting profiler annotations."""
    return _ENABLED


@contextmanager
def region(name: str) -> Iterator[None]:
    """Name the enclosed span so a profiler can attribute device time to it."""
    if not _ENABLED:
        yield
        return
    with record_function(_PREFIX + name):
        yield


@contextmanager
def profiled_regions() -> Iterator[None]:
    """Turn :func:`region` into real annotations for the duration of the block."""
    global _ENABLED
    previous, _ENABLED = _ENABLED, True
    try:
        yield
    finally:
        _ENABLED = previous


def annotation_key(name: str) -> str:
    """The profiler key :func:`region` records for ``name``."""
    return _PREFIX + name


def region_of(key: str) -> str | None:
    """Recover the region name from a profiler key, or ``None`` if unrelated."""
    return key[len(_PREFIX) :] if key.startswith(_PREFIX) else None
