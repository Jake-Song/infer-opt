"""Weight-only INT8 linear layers (W8A16): int8 weights, bf16 activations, one scale per output channel or per group.

Batch-1 decode streams every weight from HBM once per token, so halving the weight bytes is the lever.
It only pays if the int8 -> bf16 conversion happens inside the matmul kernel. ``Int8Linear.forward`` is
written as ``F.linear(x, w.to(x.dtype)) * scale`` so that ``torch.compile`` can fuse the conversion; run
eagerly it materializes a bf16 copy of the weight on every call and is slower than bf16.

With ``group_size`` the scale is per output channel *and* per ``group_size`` input columns (``scale`` is
``[out, in // group_size]``). On Qwen3.5-9B MLP weights a group of 128 has about half the squared weight error of
one scale per row; one outlier then only coarsens its own 128 columns. The scale can no longer multiply the output:
one token on CUDA runs the Triton GEMV in :mod:`infer_opt.quant_kernels` (a few tokens, as when verifying drafts, its
multi-row form), anything else dequantizes the weight before the matmul.

:class:`Int4Linear` (W4A16) halves the weight bytes again: 4-bit codes ``q`` in 0..15, two per byte, with a scale and an
offset per output channel and ``group_size`` input columns (asymmetric, ``w ~= q * scale + offset``). One token on CUDA
runs :func:`infer_opt.quant_kernels.w4a16_group_gemv`; anything else dequantizes the weight before the matmul.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["Int8Linear", "int8_linear", "quantize_int8", "dequantize_int8", "Int4Linear", "int4_linear", "quantize_int4",
           "dequantize_int4", "int4_from_gptq", "pack_int4", "unpack_int4", "quantize_linears", "QuantizationReport",
           "save_quantized", "prepare_quantized", "QUANTIZATION_FILE"]

QUANTIZATION_FILE = "quantization.json"

INT8_MAX = 127
INT4_MAX = 15  # unsigned 4-bit codes 0..15
INT4_CHUNK = 32  # pack_int4 pairs column i with column i + 16 of every 32-column chunk
INT4_CLIP_RATIOS = tuple(1 - 0.02 * i for i in range(16))  # quantize_int4(clip_search=True) tries ranges down to 70%
_INT4_CHUNK_ELEMENTS = 1 << 24  # bounds the fp32 temporaries of quantize_int4 (the clip search keeps a few)
_GEMV_ROWS = 8  # infer_opt.quant_kernels.MAX_ROWS, without importing triton here
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
    CUDA goes to the Triton GEMV (inductor's own forms of it were 1.4-2x slower than per-channel on an A100), 2-8 tokens
    to its multi-row form, any other input dequantizes the weight before the matmul (prefill, where the matmul is
    compute-bound, and CPU).
    """
    if scale.dim() == 1:
        y = F.linear(x, weight.to(x.dtype)) * scale.to(x.dtype)
    elif x.is_cuda and x.numel() == x.shape[-1]:
        from infer_opt.quant_kernels import w8a16_group_gemv  # imports triton, which a CPU-only install may lack

        y = w8a16_group_gemv(x, weight, scale)
    elif x.is_cuda and x.numel() <= _GEMV_ROWS * x.shape[-1] and _rows_supported(weight, scale):
        from infer_opt.quant_kernels import w8a16_group_gemv_rows

        y = w8a16_group_gemv_rows(x, weight, scale)
    else:
        y = F.linear(x, dequantize_int8(weight, scale, x.dtype))
    return y if bias is None else y + bias.to(x.dtype)


def _rows_supported(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    group = weight.shape[1] // scale.shape[1]  # quant_kernels.rows_supported, without importing triton here
    return group >= 16 and group & (group - 1) == 0


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


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """``[rows, k]`` codes in 0..15 (``k`` a multiple of 32) -> ``[rows, k // 2]`` uint8.

    Byte ``16 c + i`` holds column ``32 c + i`` in its low nibble and column ``32 c + 16 + i`` in its high nibble, so a
    thread that loads 16 contiguous bytes multiplies them with two contiguous 16-element runs of the activation."""
    rows, k = q.shape
    if k % INT4_CHUNK:
        raise ValueError(f"in_features {k} is not a multiple of {INT4_CHUNK}")
    q = q.to(torch.uint8).view(rows, k // INT4_CHUNK, 2, INT4_CHUNK // 2)
    return (q[:, :, 0] | (q[:, :, 1] << 4)).view(rows, k // 2)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_int4`: ``[rows, k // 2]`` uint8 -> ``[rows, k]`` uint8 codes."""
    rows, half = packed.shape
    p = packed.view(rows, half // (INT4_CHUNK // 2), 1, INT4_CHUNK // 2)
    return torch.cat([p & 15, p >> 4], dim=2).view(rows, half * 2)


def _int4_range(lo: torch.Tensor, hi: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale and offset (rounded to ``dtype``) that map codes 0..15 onto ``[lo, hi]``."""
    scale = ((hi - lo) / INT4_MAX).clamp_min(torch.finfo(torch.float32).tiny).to(dtype)
    return scale, lo.to(dtype)


def _int4_codes(rows: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """fp32 codes of ``rows [r, groups, group]`` for per-group ``scale``/``offset`` ``[r, groups]``."""
    return ((rows - offset.float().unsqueeze(-1)) / scale.float().unsqueeze(-1)).round_().clamp_(0, INT4_MAX)


def quantize_int4(weight: torch.Tensor, group_size: int = 128, clip_search: bool = False
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric 4-bit quantization of a ``[out, in]`` matrix with one scale and offset per ``group_size`` columns.

    Returns ``(packed uint8 [out, in // 2], scale, offset)``, the last two ``[out, in // group_size]`` in the weight's
    dtype, so that ``q * scale + offset ~= weight`` with ``q`` in 0..15 (see :func:`pack_int4` for the byte layout).
    By default a group's range is its min and max. ``clip_search`` instead tries the range shrunk by each of
    :data:`INT4_CLIP_RATIOS` and keeps, per group, the one with the smallest squared weight error: clipping the
    extremes of a group buys a finer step for all its other weights. The codes are computed with the rounded scale and
    offset, so what is stored is consistent with them.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    out_f, in_f = weight.shape
    if group_size <= 0 or group_size % INT4_CHUNK or in_f % group_size:
        raise ValueError(f"group_size {group_size} must be a multiple of {INT4_CHUNK} dividing in_features {in_f}")
    groups = in_f // group_size
    packed = torch.empty(out_f, in_f // 2, dtype=torch.uint8, device=weight.device)
    scale = torch.empty(out_f, groups, dtype=weight.dtype, device=weight.device)
    offset = torch.empty_like(scale)
    chunk = max(1, _INT4_CHUNK_ELEMENTS // in_f)
    for start in range(0, out_f, chunk):
        rows = weight[start : start + chunk].float().view(-1, groups, group_size)
        lo, hi = rows.amin(dim=-1), rows.amax(dim=-1)
        s, o = _int4_range(lo, hi, weight.dtype)
        if clip_search:
            best = ((_int4_codes(rows, s, o) * s.float().unsqueeze(-1) + o.float().unsqueeze(-1) - rows) ** 2).sum(-1)
            for ratio in INT4_CLIP_RATIOS[1:]:
                s_r, o_r = _int4_range(lo * ratio, hi * ratio, weight.dtype)
                err = ((_int4_codes(rows, s_r, o_r) * s_r.float().unsqueeze(-1) + o_r.float().unsqueeze(-1) - rows)
                       ** 2).sum(-1)
                better = err < best
                best = torch.where(better, err, best)
                s, o = torch.where(better, s_r, s), torch.where(better, o_r, o)
        scale[start : start + chunk], offset[start : start + chunk] = s, o
        packed[start : start + chunk] = pack_int4(_int4_codes(rows, s, o).view(-1, in_f).to(torch.uint8))
    return packed, scale, offset


def dequantize_int4(weight: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``q * scale + offset`` in ``dtype`` for packed ``weight [out, in // 2]`` and ``[out, groups]`` scale/offset."""
    q = unpack_int4(weight)
    out_f, in_f = q.shape
    groups = scale.shape[1]
    w = q.view(out_f, groups, in_f // groups).to(dtype) * scale.to(dtype).unsqueeze(-1) + offset.to(dtype).unsqueeze(-1)
    return w.view(out_f, in_f)


def int4_from_gptq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, g_idx: torch.Tensor | None = None,
                   sym: bool = False, dtype: torch.dtype = torch.bfloat16
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One 4-bit GPTQ Linear (AutoGPTQ / GPTQModel ``gptq`` format) as :func:`quantize_int4` output.

    GPTQ stores ``qweight`` int32 ``[in // 8, out]`` with eight codes per word along the input (code ``j`` in bits
    ``4j..4j+3``), ``qzeros`` int32 ``[groups, out // 8]`` packed the same way along the output and holding the zero
    point minus one, and ``scales`` ``[groups, out]``: ``w = (q - zero) * scale``. A symmetric checkpoint's zero is 8
    whatever ``qzeros`` holds, as vLLM's GPTQ-Marlin reads it (Qwen/Qwen3.5-27B-GPTQ-Int4 stores 8, which the
    minus-one rule would turn into 9). Here ``offset = -zero * scale``, from the scale rounded to ``dtype``; that
    rounding (fp16 -> bf16) is the only change, as in vLLM's bf16 GPTQ path. Act-order checkpoints (``g_idx`` not
    ``i // group_size``) are not supported.
    """
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qweight.device)
    q = ((qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 15).reshape(-1, qweight.shape[1])  # [in, out]
    if sym:
        zero = torch.full_like(scales, 8, dtype=torch.int32)
    else:
        zero = ((qzeros.unsqueeze(2) >> shifts.view(1, 1, 8)) & 15).reshape(qzeros.shape[0], -1) + 1  # [groups, out]
    in_f, groups = q.shape[0], scales.shape[0]
    group_size = in_f // groups
    if g_idx is not None and not torch.equal(g_idx.long().cpu(), torch.arange(in_f) // group_size):
        raise ValueError("act-order GPTQ (permuted g_idx) is not supported")
    scale = scales.t().to(dtype).contiguous()
    offset = (-zero.t().float() * scale.float()).to(dtype).contiguous()
    return pack_int4(q.t().contiguous()), scale, offset


def int4_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
    """The W4A16 matmul as a pure function of tensors (what ``Int4Linear.forward`` runs, and what gets compiled).

    One token on CUDA goes to the Triton GEMV, which reads the packed weight once; any other input (prefill, several
    drafted tokens, CPU) dequantizes the weight before the matmul."""
    if x.is_cuda and x.numel() == x.shape[-1]:
        from infer_opt.quant_kernels import w4a16_group_gemv  # imports triton, which a CPU-only install may lack

        y = w4a16_group_gemv(x, weight, scale, offset)
    else:
        y = F.linear(x, dequantize_int4(weight, scale, offset, x.dtype))
    return y if bias is None else y + bias.to(x.dtype)


class Int4Linear(nn.Module):
    """``y = x @ (q * scale + offset)^T`` with 4-bit ``q`` packed two per byte and ``scale``/``offset`` per group."""

    bits = 4

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.out_features, self.in_features = weight.shape[0], weight.shape[1] * 2
        self.group_size = self.in_features // scale.shape[1]
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.register_buffer("offset", offset)
        self.register_buffer("bias", bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128, clip_search: bool = False) -> "Int4Linear":
        q, scale, offset = quantize_int4(linear.weight.detach(), group_size, clip_search)
        bias = None if linear.bias is None else linear.bias.detach()
        return cls(q, scale, offset, bias)

    def dequantized_weight(self) -> torch.Tensor:
        return dequantize_int4(self.weight, self.scale, self.offset, self.scale.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return int4_linear(x, self.weight, self.scale, self.offset, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int4, group_size={self.group_size}"


@dataclass
class QuantizationReport:
    replaced: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  # module name -> reason
    bytes_before: int = 0
    bytes_after: int = 0
    bits: int = 8

    def summary(self) -> str:
        return (f"{len(self.replaced)} Linear -> Int{self.bits}Linear, {len(self.skipped)} kept; "
                f"{self.bytes_before / 2**30:.3f} GiB -> {self.bytes_after / 2**30:.3f} GiB")


@torch.no_grad()
def quantize_linears(
    model: nn.Module,
    *,
    include_lm_head: bool = False,
    min_out_features: int = 256,
    group_size: int | None = None,
    exclude: tuple[str, ...] = (),
    bits: int = 8,
    clip_search: bool = False,
) -> QuantizationReport:
    """Replace ``nn.Linear`` modules by :class:`Int8Linear` (or :class:`Int4Linear` with ``bits=4``), one at a time so
    only one weight is duplicated.

    ``lm_head`` is kept in bf16 unless ``include_lm_head`` is set: it is the most quality-sensitive matrix and
    its logits go straight to sampling. Linears with fewer than ``min_out_features`` outputs (the 32-wide
    delta-rule gate projections) are kept: their weights are tiny, so quantizing them saves nothing.
    ``group_size`` selects group-wise scales (see :func:`quantize_int8`); a Linear whose ``in_features`` it does
    not divide is kept in its original dtype. Modules whose name starts with one of ``exclude`` are kept too.
    ``bits=4`` needs a ``group_size`` (a multiple of 32); ``clip_search`` is passed to :func:`quantize_int4`.
    """
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    if bits == 4 and group_size is None:
        raise ValueError("4-bit quantization needs a group_size")
    report = QuantizationReport(bits=bits)
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    for name, module in targets:
        if exclude and name.startswith(exclude):
            report.skipped[name] = "excluded"
            continue
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
        if bits == 4:
            quantized = Int4Linear.from_linear(module, group_size, clip_search)
        else:
            quantized = Int8Linear.from_linear(module, group_size)
        setattr(parent, child_name, quantized)
        report.replaced.append(name)
        report.bytes_before += before
        report.bytes_after += sum(t.numel() * t.element_size() for t in quantized.buffers() if t is not quantized.bias)
    return report


@torch.no_grad()
def save_quantized(model: nn.Module, path: str | Path, *, shard_bytes: int = 2 << 30) -> list[Path]:
    """Write ``model``'s state dict as safetensors shards plus :data:`QUANTIZATION_FILE`, which lists every
    :class:`Int8Linear` / :class:`Int4Linear` with its shape, group size and bits so :func:`prepare_quantized` can
    rebuild the module tree. A model with any 4-bit Linear is written as format ``infer_opt.int4``.

    A tensor shared between two names (tied embeddings) is stored once, under its first name. The ``mtp`` flag
    records an attached MTP module, including one whose weights are still floating point. Returns the shard paths.
    """
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    modules = {name: {"out_features": m.out_features, "in_features": m.in_features, "group_size": m.group_size,
                      "bias": m.bias is not None, **({"bits": 4} if isinstance(m, Int4Linear) else {})}
               for name, m in model.named_modules() if isinstance(m, (Int8Linear, Int4Linear))}
    fmt = "int4" if any(m.get("bits") == 4 for m in modules.values()) else "int8"
    seen, tied, shards, current, size = {}, {}, [], {}, 0
    for name, tensor in model.state_dict().items():
        key = (tensor.data_ptr(), tensor.dtype, tuple(tensor.shape))
        if tensor.data_ptr() and key in seen:
            tied[name] = seen[key]
            continue
        seen[key] = name
        nbytes = tensor.numel() * tensor.element_size()
        if current and size + nbytes > shard_bytes:
            shards.append(current)
            current, size = {}, 0
        current[name] = tensor.detach().contiguous().cpu()
        size += nbytes
    if current:
        shards.append(current)
    files = []
    for i, shard in enumerate(shards, 1):
        files.append(path / f"model-{fmt}-{i:05d}-of-{len(shards):05d}.safetensors")
        save_file(shard, str(files[-1]))
    (path / QUANTIZATION_FILE).write_text(json.dumps({
        "format": f"infer_opt.{fmt}", "modules": modules, "tied": tied,
        "mtp": getattr(model, "mtp", None) is not None,
    }, indent=1))
    return files


def prepare_quantized(model: nn.Module, spec: dict) -> None:
    """Replace the ``nn.Linear`` modules named in ``spec["modules"]`` by empty :class:`Int8Linear` (``Int4Linear`` for
    ``bits: 4``) of the same shape (on the Linear's device, typically meta), ready for
    ``load_state_dict(..., assign=True)``."""
    for name, m in spec["modules"].items():
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        device = getattr(parent, child_name).weight.device
        out_f, in_f, group = m["out_features"], m["in_features"], m["group_size"]
        weight = torch.empty(out_f, in_f, dtype=torch.int8, device=device)
        scale_shape = (out_f,) if group is None else (out_f, in_f // group)
        scale = torch.empty(scale_shape, dtype=torch.bfloat16, device=device)
        bias = torch.empty(out_f, dtype=torch.bfloat16, device=device) if m["bias"] else None
        if m.get("bits", 8) == 4:
            weight = torch.empty(out_f, in_f // 2, dtype=torch.uint8, device=device)
            setattr(parent, child_name, Int4Linear(weight, scale, torch.empty_like(scale), bias))
        else:
            setattr(parent, child_name, Int8Linear(weight, scale, bias))
