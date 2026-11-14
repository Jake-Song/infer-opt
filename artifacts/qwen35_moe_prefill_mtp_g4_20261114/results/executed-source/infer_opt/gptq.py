"""GPTQ for the group-wise 4-bit format of :class:`infer_opt.quant.Int4Linear`.

Rounding each weight to its nearest code (RTN, :func:`infer_opt.quant.quantize_int4`) minimizes ``||W - Q||^2``, but
what matters is the output error on real inputs, ``||(W - Q) X||^2``. GPTQ (Frantar et al., 2022) minimizes that with
``H = X X^T`` from calibration inputs: it quantizes one input column at a time and spreads the column's rounding error
over the columns not quantized yet, weighted by the inverse of ``H`` (the optimal update for the remaining weights).
Columns go in their natural order (no act-order), so every group of ``group_size`` columns is contiguous and its scale
and offset, taken from the min and max of the group's already-updated weights when the group starts, are exactly
what the Triton GEMV reads. The output is the same ``(packed, scale, offset)`` as RTN: kernel and checkpoint format
are unchanged.

:func:`gptq_quantize_qwen3_5` runs it layer by layer through a Qwen3.5 model: the calibration hidden states enter a
decoder layer, every Linear of the layer collects its ``H``, the layer is quantized, and the *quantized* layer's
outputs are the next layer's inputs, so later layers compensate for the errors of earlier ones.
"""

from __future__ import annotations

import time

import torch
from torch import nn

from infer_opt.quant import INT4_CHUNK, Int4Linear, QuantizationReport, _int4_codes, _int4_range, pack_int4

__all__ = ["gptq_quantize_int4", "gptq_quantize_qwen3_5"]


@torch.no_grad()
def gptq_quantize_int4(weight: torch.Tensor, hessian: torch.Tensor, group_size: int = 128, damp: float = 0.01,
                       block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ of a ``[out, in]`` weight to asymmetric 4-bit codes with one scale/offset per ``group_size`` columns.

    ``hessian`` is ``[in, in]``, proportional to ``X X^T`` over the calibration tokens (its scale does not matter).
    ``damp`` times the mean of its diagonal is added to the diagonal for a stable inverse. Columns that never saw a
    nonzero input are set to zero. Returns ``(packed uint8 [out, in // 2], scale, offset)`` like
    :func:`infer_opt.quant.quantize_int4`; with a diagonal ``hessian`` the result is the same as its min-max RTN.
    """
    out_f, in_f = weight.shape
    if group_size <= 0 or group_size % INT4_CHUNK or in_f % group_size:
        raise ValueError(f"group_size {group_size} must be a multiple of {INT4_CHUNK} dividing in_features {in_f}")
    if block_size % group_size and group_size % block_size:
        raise ValueError("block_size and group_size must divide one another")
    dtype = weight.dtype
    W = weight.float().clone()
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H.diagonal().add_(damp * torch.mean(torch.diag(H)))
    # upper Cholesky factor of H^-1: row i holds the update of columns i.. for the error of column i
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    del H

    groups = in_f // group_size
    codes = torch.empty(out_f, in_f, dtype=torch.uint8, device=weight.device)
    scale = torch.empty(out_f, groups, dtype=dtype, device=weight.device)
    offset = torch.empty_like(scale)
    s = o = None
    for i1 in range(0, in_f, block_size):
        i2 = min(i1 + block_size, in_f)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.empty_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group_size == 0:  # the group's range from its weights as updated so far
                g = col // group_size
                group = W1[:, i : i + group_size] if block_size >= group_size else W[:, col : col + group_size]
                s, o = _int4_range(group.amin(dim=1), group.amax(dim=1), dtype)
                scale[:, g], offset[:, g] = s, o
            w = W1[:, i]
            q = _int4_codes(w.view(-1, 1, 1), s.view(-1, 1), o.view(-1, 1)).view(-1)
            codes[:, col] = q.to(torch.uint8)
            err = (w - (q * s.float() + o.float())) / Hinv1[i, i]
            W1[:, i:].addr_(err, Hinv1[i, i:], alpha=-1)
            Err1[:, i] = err
        W[:, i2:].addmm_(Err1, Hinv[i1:i2, i2:], alpha=-1)
        if block_size < group_size:
            W[:, i1:i2] = W1  # a later block of the same group reads its range from W
    return pack_int4(codes), scale, offset


def _hessian_hook(store: dict, name: str):
    def hook(module, args, output):
        x = args[0].reshape(-1, args[0].shape[-1]).float()
        h = store.get(name)
        if h is None:
            store[name] = h = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device)
        h.addmm_(x.T, x)
    return hook


@torch.no_grad()
def gptq_quantize_qwen3_5(model: nn.Module, calibration: list[torch.Tensor], *, group_size: int = 128,
                          damp: float = 0.01, min_out_features: int = 256, log=print) -> QuantizationReport:
    """Quantize every decoder-layer Linear of a Qwen3.5 model (bf16, on the GPU) to :class:`Int4Linear` with GPTQ.

    ``calibration`` is a list of ``[1, T]`` token ids. lm_head, the embeddings, the MTP head and Linears with fewer
    than ``min_out_features`` outputs (the delta-rule gates) stay as they are, as in
    :func:`infer_opt.quant.quantize_linears`. Holds one hidden state per sample (``T x hidden`` in the model dtype).
    """
    lm = model.model.language_model
    report = QuantizationReport(bits=4)
    hidden = [lm.embed_tokens(ids) for ids in calibration]
    rope = []
    for ids, h in zip(calibration, hidden):
        position_ids = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, 1, -1)
        rope.append(lm.rotary_emb(h, position_ids))
    for index, layer in enumerate(lm.layers):
        t0 = time.perf_counter()
        targets = {}
        for name, module in layer.named_modules():
            if type(module) is not nn.Linear:
                continue
            full = f"model.language_model.layers.{index}.{name}"
            if module.out_features < min_out_features:
                report.skipped[full] = f"out_features {module.out_features} < {min_out_features}"
            elif module.in_features % group_size:
                report.skipped[full] = f"in_features {module.in_features} not divisible by group_size {group_size}"
            else:
                targets[name] = module
        hessians: dict[str, torch.Tensor] = {}
        hooks = [m.register_forward_hook(_hessian_hook(hessians, n)) for n, m in targets.items()]
        try:
            for h, pe in zip(hidden, rope):
                layer(h, pe)
        finally:
            for hook in hooks:
                hook.remove()
        for name, module in targets.items():
            packed, scale, offset = gptq_quantize_int4(module.weight, hessians.pop(name), group_size, damp)
            parent_name, _, child_name = name.rpartition(".")
            parent = layer.get_submodule(parent_name) if parent_name else layer
            bias = None if module.bias is None else module.bias.detach()
            quantized = Int4Linear(packed, scale, offset, bias)
            setattr(parent, child_name, quantized)
            report.replaced.append(f"model.language_model.layers.{index}.{name}")
            report.bytes_before += module.weight.numel() * module.weight.element_size()
            report.bytes_after += sum(t.numel() * t.element_size() for t in (packed, scale, offset))
        del hessians
        hidden = [layer(h, pe) for h, pe in zip(hidden, rope)]  # through the quantized layer
        log(f"gptq layer {index} ({layer.block_type}): {len(targets)} Linear in {time.perf_counter() - t0:.1f} s")
    for name, module in model.named_modules():
        if type(module) is nn.Linear and name not in report.skipped and not name.startswith("model.language_model.layers"):
            report.skipped[name] = "not a decoder-layer Linear"
    return report
