"""Prefill instrumentation: routed-expert load per layer and synchronized section times.

Both are diagnostic passes, separate from TTFT measurement: the route recorder
reads counts back to the host, and the section timer synchronizes the device at
every section boundary, which removes the overlap of host dispatch and device
work that an unprofiled prefill has.
"""
from __future__ import annotations

from collections import defaultdict
import statistics
import time

import torch
from torch import nn

__all__ = ["RouteRecorder", "route_statistics", "section_times"]

_ROW_BUCKETS = ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, None))


def route_statistics(counts, block_ms=(16, 32, 64)) -> dict:
    """Summarize one layer's assignments per expert (``counts[e]`` rows routed to expert ``e``)."""
    counts = torch.as_tensor(counts, dtype=torch.long)
    used = counts[counts > 0]
    rows = int(counts.sum())
    out = {"experts": counts.numel(), "rows": rows, "used_experts": used.numel()}
    if not used.numel():
        return out
    ordered = used.sort(descending=True).values
    out.update(
        rows_min=int(used.min()), rows_median=float(used.float().median()), rows_mean=rows / used.numel(),
        rows_max=int(used.max()),
        # the eager loop sends these experts to the small-row GEMV instead of dequantize + matmul
        experts_with_1_to_8_rows=int((used <= 8).sum()),
        rows_in_top8_experts=int(ordered[:8].sum()),
        experts_by_rows={f"{lo}-{hi}" if hi else f"{lo}+": int(((used >= lo) & (used <= (hi or rows))).sum())
                         for lo, hi in _ROW_BUCKETS},
        # row tiles a grouped kernel launches, and how full they are
        tiles={str(bm): {"tiles": int(((used + bm - 1) // bm).sum()),
                         "fill": rows / (int(((used + bm - 1) // bm).sum()) * bm)} for bm in block_ms},
    )
    return out


class RouteRecorder:
    """Record every expert bank's per-expert row counts during the enclosed forward passes."""

    def __init__(self, model: nn.Module):
        from .moe import Int8Experts, Qwen3_5Experts

        self.banks = [(name, m) for name, m in model.named_modules() if isinstance(m, (Int8Experts, Qwen3_5Experts))]
        self.counts: dict[str, list[int]] = {}
        self._handles = []

    def __enter__(self) -> "RouteRecorder":
        for name, module in self.banks:
            def hook(module, args, name=name):
                indices = args[1]
                self.counts[name] = torch.bincount(indices.reshape(-1).long(),
                                                   minlength=module.gate_up_proj.shape[0]).tolist()
            self._handles.append(module.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def summary(self) -> dict:
        layers = {name: route_statistics(counts) for name, counts in self.counts.items()}
        used = [s["used_experts"] for s in layers.values()]
        return {"layers": layers, "used_experts_min": min(used), "used_experts_median": statistics.median(used),
                "used_experts_max": max(used), "used_experts_total": sum(used)}


def _sections(model: nn.Module) -> dict[nn.Module, str]:
    from .quant import Int4Linear, Int8Linear

    lm = model.model.language_model
    sections = {lm: "text_model", lm.embed_tokens: "embedding", lm.rotary_emb: "rotary", lm.norm: "final_norm",
                model.lm_head: "lm_head"}
    for layer in lm.layers:
        linear = getattr(layer, "linear_attn", None)
        kind, block = ("gated_delta", linear) if linear is not None else ("full_attention", layer.self_attn)
        sections[block] = kind
        for module in block.modules():
            if isinstance(module, (nn.Linear, Int8Linear, Int4Linear)):
                sections[module] = kind + "_projections"
        sections[layer.input_layernorm] = sections[layer.post_attention_layernorm] = "layer_norms"
        if hasattr(layer.mlp, "experts"):
            sections.update({layer.mlp: "moe_block", layer.mlp.experts: "routed_experts",
                             layer.mlp.shared_expert: "shared_expert", layer.mlp.gate: "router",
                             layer.mlp.shared_expert_gate: "shared_expert_gate"})
        else:
            sections[layer.mlp] = "mlp"
    return sections


def section_times(model: nn.Module, fn, reset=None, repeats: int = 3) -> dict:
    """Median synchronized milliseconds of ``fn`` (one prefill of ``model``) by exclusive section.

    ``*_core`` is an attention block without its projections; ``moe_other`` is routing softmax/top-k, the
    shared-expert gate product and the sum; ``glue`` is residual adds and Python between sections. The sections add
    up to ``text_model`` plus ``lm_head``; ``wall`` also includes whatever ``fn`` does after them.
    """
    sections = _sections(model)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    inclusive: defaultdict[str, float] = defaultdict(float)
    starts: dict[nn.Module, float] = {}

    def before(module, args):
        sync()
        starts[module] = time.perf_counter()

    def after(module, args, output):
        sync()
        inclusive[sections[module]] += (time.perf_counter() - starts.pop(module)) * 1000

    handles = [h for module in sections for h in (module.register_forward_pre_hook(before),
                                                  module.register_forward_hook(after))]
    samples = []
    try:
        for _ in range(repeats):
            if reset is not None:
                reset()
            inclusive.clear()
            sync()
            start = time.perf_counter()
            fn()
            sync()
            wall = (time.perf_counter() - start) * 1000
            t = defaultdict(float, inclusive)
            sample = {name: t[name] for name in ("embedding", "rotary", "final_norm", "layer_norms", "lm_head")}
            for kind in ("gated_delta", "full_attention"):
                sample[kind + "_projections"] = t[kind + "_projections"]
                sample[kind + "_core"] = t[kind] - t[kind + "_projections"]
            if "moe_block" in t:
                parts = ("router", "routed_experts", "shared_expert", "shared_expert_gate")
                sample.update({name: t[name] for name in parts})
                sample["moe_other"] = t["moe_block"] - sum(t[name] for name in parts)
            else:
                sample["mlp"] = t["mlp"]
            sample["glue"] = t["text_model"] - sum(v for name, v in sample.items() if name != "lm_head")
            sample["text_model"], sample["wall"] = t["text_model"], wall
            samples.append(sample)
    finally:
        for handle in handles:
            handle.remove()
    return {name: statistics.median(s[name] for s in samples) for name in samples[0]}
