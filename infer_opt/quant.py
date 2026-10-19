"""Weight-only INT8 linear layers (W8A16): int8 weights, bf16 activations, one scale per output channel or per group.

Batch-1 decode streams every weight from HBM once per token, so halving the weight bytes is the lever.
It only pays if the int8 -> bf16 conversion happens inside the matmul kernel. ``Int8Linear.forward`` is
written as ``F.linear(x, w.to(x.dtype)) * scale`` so that ``torch.compile`` can fuse the conversion; run
eagerly it materializes a bf16 copy of the weight on every call and is slower than bf16.

With ``group_size`` the scale is per output channel *and* per ``group_size`` input columns (``scale`` is
``[out, in // group_size]``). On Qwen3.5-9B MLP weights a group of 128 has about half the squared weight error of
one scale per row; one outlier then only coarsens its own 128 columns. The scale can no longer multiply the output:
one token on CUDA runs the Triton GEMV in :mod:`infer_opt.quant_kernels`, anything else dequantizes the weight
before the matmul.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["Int8Linear", "int8_linear", "quantize_int8", "dequantize_int8", "quantize_linears", "QuantizationReport"]

INT8_MAX = 127
_ROW_CHUNK = 16384  # bounds the fp32 temporary when quantizing a vocab-sized matrix


def quantize_int8(weight: torch.Tensor, group_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric quantization of a ``[out, in]`` matrix, per output channel or per ``group_size`` input columns.

    Returns ``(int8 weight, scale)`` with ``scale`` in the weight's dtype, shaped ``[out]`` or
    ``[out, in // group_size]``, so that ``q * scale ~= weight`` (the scale repeated over each group).
    The int8 values are computed with the already-rounded scale, so what is stored is consistent with it.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    out_f, in_f = weight.shape
    if group_size is not None and (group_size <= 0 or in_f % group_size):
        raise ValueError(f"group_size {group_size} does not divide in_features {in_f}")
    groups = 1 if group_size is None else in_f // group_size
    q = torch.empty(weight.shape, dtype=torch.int8, device=weight.device)
    scale = torch.empty(out_f, groups, dtype=weight.dtype, device=weight.device)
    for start in range(0, out_f, _ROW_CHUNK):
        rows = weight[start : start + _ROW_CHUNK].float().reshape(-1, groups, in_f // groups)
        s = (rows.abs().amax(dim=-1) / INT8_MAX).clamp_min(torch.finfo(torch.float32).tiny).to(weight.dtype)
        scale[start : start + _ROW_CHUNK] = s
        rows = (rows / s.float().unsqueeze(-1)).round_().clamp_(-INT8_MAX, INT8_MAX)
        q[start : start + _ROW_CHUNK] = rows.reshape(-1, in_f).to(torch.int8)
    return q, (scale.squeeze(1) if group_size is None else scale)


def dequantize_int8(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``q * scale`` in ``dtype``, for a per-channel ``[out]`` or a group-wise ``[out, groups]`` scale."""
    if scale.dim() == 1:
        return weight.to(dtype) * scale.to(dtype).unsqueeze(1)
    out_f, in_f = weight.shape
    groups = scale.shape[1]
    return (weight.view(out_f, groups, in_f // groups).to(dtype) * scale.to(dtype).unsqueeze(-1)).view(out_f, in_f)


def int8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """The W8A16 matmul as a pure function of tensors (what ``Int8Linear.forward`` runs, and what gets compiled).

    A per-channel scale multiplies the output, so the int8 -> x.dtype cast is the only elementwise work on the
    weight and can be absorbed into the matmul kernel. A group-wise scale varies along the reduction: one token on
    CUDA goes to the Triton GEMV (inductor's own forms of it were 1.4-2x slower than per-channel on an A100), any
    other input dequantizes the weight before the matmul (prefill, where the matmul is compute-bound, and CPU).
    """
    if scale.dim() == 1:
        y = F.linear(x, weight.to(x.dtype)) * scale.to(x.dtype)
    elif x.is_cuda and x.numel() == x.shape[-1]:
        from infer_opt.quant_kernels import w8a16_group_gemv  # imports triton, which a CPU-only install may lack

        y = w8a16_group_gemv(x, weight, scale)
    else:
        y = F.linear(x, dequantize_int8(weight, scale, x.dtype))
    return y if bias is None else y + bias.to(x.dtype)


class Int8Linear(nn.Module):
    """``y = x @ (q * scale)^T`` with ``q`` stored as int8 and ``scale`` per output channel (``[out]``) or group."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.group_size = None if scale.dim() == 1 else self.in_features // scale.shape[1]
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int | None = None) -> "Int8Linear":
        q, scale = quantize_int8(linear.weight.detach(), group_size)
        bias = None if linear.bias is None else linear.bias.detach()
        return cls(q, scale, bias)

    def dequantized_weight(self) -> torch.Tensor:
        return dequantize_int8(self.weight, self.scale, self.scale.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return int8_linear(x, self.weight, self.scale, self.bias)

    def extra_repr(self) -> str:
        group = "" if self.group_size is None else f", group_size={self.group_size}"
        return f"in_features={self.in_features}, out_features={self.out_features}, int8{group}"


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
    group_size: int | None = None,
) -> QuantizationReport:
    """Replace ``nn.Linear`` modules by :class:`Int8Linear`, one at a time so only one weight is duplicated.

    ``lm_head`` is kept in bf16 unless ``include_lm_head`` is set: it is the most quality-sensitive matrix and
    its logits go straight to sampling. Linears with fewer than ``min_out_features`` outputs (the 32-wide
    delta-rule gate projections) are kept: their weights are tiny, so quantizing them saves nothing.
    ``group_size`` selects group-wise scales (see :func:`quantize_int8`); a Linear whose ``in_features`` it does
    not divide is kept in its original dtype.
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
        if group_size is not None and module.in_features % group_size:
            report.skipped[name] = f"in_features {module.in_features} not divisible by group_size {group_size}"
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        before = module.weight.numel() * module.weight.element_size()
        quantized = Int8Linear.from_linear(module, group_size)
        setattr(parent, child_name, quantized)
        report.replaced.append(name)
        report.bytes_before += before
        report.bytes_after += quantized.weight.numel() + quantized.scale.numel() * quantized.scale.element_size()
    return report
