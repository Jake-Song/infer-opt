"""Weight-only INT8 linear layers (W8A16): int8 weights with one scale per output channel, bf16 activations.

Batch-1 decode streams every weight from HBM once per token, so halving the weight bytes is the lever.
It only pays if the int8 -> bf16 conversion happens inside the matmul kernel. ``Int8Linear.forward`` is
written as ``F.linear(x, w.to(x.dtype)) * scale`` so that ``torch.compile`` can fuse the conversion; run
eagerly it materializes a bf16 copy of the weight on every call and is slower than bf16.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["Int8Linear", "int8_linear", "quantize_int8", "quantize_linears", "QuantizationReport"]

INT8_MAX = 127
_ROW_CHUNK = 16384  # bounds the fp32 temporary when quantizing a vocab-sized matrix


def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel quantization of a ``[out, in]`` matrix.

    Returns ``(int8 weight, scale)`` with ``scale`` in the weight's dtype, so that ``q * scale ~= weight``.
    The int8 values are computed with the already-rounded scale, so what is stored is consistent with it.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    q = torch.empty(weight.shape, dtype=torch.int8, device=weight.device)
    scale = torch.empty(weight.shape[0], dtype=weight.dtype, device=weight.device)
    for start in range(0, weight.shape[0], _ROW_CHUNK):
        rows = weight[start : start + _ROW_CHUNK].float()
        s = (rows.abs().amax(dim=1) / INT8_MAX).clamp_min(torch.finfo(torch.float32).tiny).to(weight.dtype)
        scale[start : start + _ROW_CHUNK] = s
        q[start : start + _ROW_CHUNK] = (rows / s.float().unsqueeze(1)).round().clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q, scale


def int8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """The W8A16 matmul as a pure function of tensors (what ``Int8Linear.forward`` runs, and what gets compiled).

    The scale multiplies the output (it is per output channel), so the int8 -> x.dtype cast is the only
    elementwise work on the weight and can be absorbed into the matmul kernel.
    """
    y = F.linear(x, weight.to(x.dtype)) * scale.to(x.dtype)
    return y if bias is None else y + bias.to(x.dtype)


class Int8Linear(nn.Module):
    """``y = x @ (q * scale)^T`` with ``q`` stored as int8 and ``scale`` per output channel."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        q, scale = quantize_int8(linear.weight.detach())
        bias = None if linear.bias is None else linear.bias.detach()
        return cls(q, scale, bias)

    def dequantized_weight(self) -> torch.Tensor:
        return self.weight.to(self.scale.dtype) * self.scale.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return int8_linear(x, self.weight, self.scale, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8"


@dataclass
class QuantizationReport:
    replaced: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  # module name -> reason
    bytes_before: int = 0
    bytes_after: int = 0

    def summary(self) -> str:
        return (f"{len(self.replaced)} Linear -> Int8Linear, {len(self.skipped)} kept; "
                f"{self.bytes_before / 2**30:.3f} GiB -> {self.bytes_after / 2**30:.3f} GiB")


@torch.no_grad()
def quantize_linears(
    model: nn.Module,
    *,
    include_lm_head: bool = False,
    min_out_features: int = 256,
) -> QuantizationReport:
    """Replace ``nn.Linear`` modules by :class:`Int8Linear`, one at a time so only one weight is duplicated.

    ``lm_head`` is kept in bf16 unless ``include_lm_head`` is set: it is the most quality-sensitive matrix and
    its logits go straight to sampling. Linears with fewer than ``min_out_features`` outputs (the 32-wide
    delta-rule gate projections) are kept: their weights are tiny, so quantizing them saves nothing.
    """
    report = QuantizationReport()
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    for name, module in targets:
        if name.endswith("lm_head") and not include_lm_head:
            report.skipped[name] = "lm_head kept in bf16"
            continue
        if module.out_features < min_out_features:
            report.skipped[name] = f"out_features {module.out_features} < {min_out_features}"
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        before = module.weight.numel() * module.weight.element_size()
        quantized = Int8Linear.from_linear(module)
        setattr(parent, child_name, quantized)
        report.replaced.append(name)
        report.bytes_before += before
        report.bytes_after += quantized.weight.numel() + quantized.scale.numel() * quantized.scale.element_size()
    return report
