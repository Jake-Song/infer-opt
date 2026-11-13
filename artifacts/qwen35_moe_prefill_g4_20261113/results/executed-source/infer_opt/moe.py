"""Qwen3.5 routed experts and checkpoint normalization.

The eager path is a correctness reference, not an optimized decode kernel.
"""
from __future__ import annotations

import dataclasses
import torch
from torch import nn
from torch.nn import functional as F


class Qwen3_5Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        e, h, m = config.num_experts, config.hidden_size, config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(e, 2 * m, h))
        self.down_proj = nn.Parameter(torch.empty(e, h, m))

    def forward(self, x, indices, weights):
        output = torch.zeros_like(x)
        for expert in range(self.gate_up_proj.shape[0]):
            tokens, slots = torch.where(indices == expert)
            if tokens.numel() == 0:
                continue
            gate, up = F.linear(x[tokens], self.gate_up_proj[expert]).chunk(2, -1)
            y = F.linear(F.silu(gate) * up, self.down_proj[expert])
            output.index_add_(0, tokens, (y * weights[tokens, slots, None]).to(x.dtype))
        return output


class Qwen3_5SparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        from .qwen3_5 import Qwen3_5MLP
        if not 0 < config.num_experts_per_tok <= config.num_experts:
            raise ValueError('invalid number of selected experts')
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3_5Experts(config)
        shared = dataclasses.replace(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert = Qwen3_5MLP(shared)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, x, consistent_rows: bool = False):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        small_consistent = consistent_rows and x.shape[0] <= 5
        logits = (torch.cat([self.gate(x[i:i + 1]) for i in range(x.shape[0])])
                  if small_consistent else self.gate(x))
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = (weights / weights.sum(-1, keepdim=True)).to(x.dtype)
        routed = self.experts(x, indices, weights)
        if small_consistent:
            shared = torch.cat([self.shared_expert(x[i:i + 1]) * self.shared_expert_gate(x[i:i + 1]).sigmoid()
                                for i in range(x.shape[0])])
        else:
            shared = self.shared_expert(x) * self.shared_expert_gate(x).sigmoid()
        return (routed + shared).reshape(shape)


def normalize_expert_state(state, config):
    """Normalize separate HF MTP expert Linears into the target's packed layout.

    Reject incomplete or mixed layouts instead of silently dropping tensors.
    The input dictionary is consumed to avoid retaining duplicate expert banks.
    """
    suffix = '.experts.0.gate_proj.weight'
    prefixes = [k[:-len(suffix)] + '.experts' for k in state if k.endswith(suffix)]
    for prefix in prefixes:
        if prefix + '.gate_up_proj' in state or prefix + '.down_proj' in state:
            raise ValueError(f'mixed expert layouts: {prefix}')
        names = [f'{prefix}.{i}.{p}_proj.weight' for i in range(config.num_experts)
                 for p in ('gate', 'up', 'down')]
        missing = [n for n in names if n not in state]
        if missing:
            raise ValueError(f'incomplete expert layout: {missing[:3]}')
        h, m = config.hidden_size, config.moe_intermediate_size
        first = state[names[0]]
        gu = first.new_empty(config.num_experts, 2 * m, h)
        down = first.new_empty(config.num_experts, h, m)
        for i in range(config.num_experts):
            for projection, target in [('gate', gu[i, :m]), ('up', gu[i, m:]), ('down', down[i])]:
                tensor = state.pop(f'{prefix}.{i}.{projection}_proj.weight')
                if tensor.shape != target.shape:
                    raise ValueError(f'invalid expert shape: {prefix}.{i}.{projection}')
                target.copy_(tensor)
        state[prefix + '.gate_up_proj'] = gu
        state[prefix + '.down_proj'] = down
    return state


class Int8Experts(nn.Module):
    """Packed W8A16 banks; grouped prefill is an explicit experimental opt-in."""
    def __init__(self, gate_up, gate_up_scale, down, down_scale, group_size):
        super().__init__()
        self.group_size = group_size
        self.prefill_backend = 'reference'
        self.register_buffer('gate_up_proj', gate_up)
        self.register_buffer('gate_up_scale', gate_up_scale)
        self.register_buffer('down_proj', down)
        self.register_buffer('down_scale', down_scale)

    @classmethod
    @torch.no_grad()
    def from_float(cls, experts, group_size=128):
        from .quant import quantize_int8
        def convert(bank):
            q = torch.empty_like(bank, dtype=torch.int8)
            scales = bank.new_empty(*bank.shape[:-1], bank.shape[-1] // group_size)
            for i in range(bank.shape[0]):
                q[i], scales[i] = quantize_int8(bank[i], group_size)
            return q, scales
        gu, gs = convert(experts.gate_up_proj)
        down, ds = convert(experts.down_proj)
        return cls(gu, gs, down, ds, group_size)

    def forward(self, x, indices, weights):
        if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and x.shape[0] <= 5:
            from .moe_kernels import selected_int8_experts
            return selected_int8_experts(x, indices, weights, self.gate_up_proj,
                                        self.gate_up_scale, self.down_proj, self.down_scale)
        if self.prefill_backend == 'grouped' and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
            from .moe_prefill import grouped_int8_experts
            return grouped_int8_experts(x, indices, weights, self.gate_up_proj,
                                       self.gate_up_scale, self.down_proj, self.down_scale)
        if self.prefill_backend == 'sorted':
            return self.forward_sorted(x, indices, weights)
        return self.forward_reference(x, indices, weights)

    def forward_sorted(self, x, indices, weights):
        """Keep expert arithmetic/order; replace per-expert nonzero syncs with one count read.

        Stable sorting preserves torch.where's token/slot order within each expert.
        This eager prefill path intentionally synchronizes once for CPU loop bounds.
        """
        from .quant import int8_linear
        experts = self.gate_up_proj.shape[0]
        top = indices.shape[1]
        flat = indices.reshape(-1).long()
        order = torch.argsort(flat, stable=True)
        tokens = order // top
        slots = order % top
        counts = torch.zeros(experts, device=x.device, dtype=torch.long)
        counts.scatter_add_(0, flat, torch.ones_like(flat))
        output = torch.zeros_like(x)
        start = 0
        for expert, count in enumerate(counts.tolist()):
            if count:
                selected = tokens[start:start + count]
                selected_slots = slots[start:start + count]
                gate, up = int8_linear(x[selected], self.gate_up_proj[expert], self.gate_up_scale[expert]).chunk(2, -1)
                y = int8_linear(F.silu(gate) * up, self.down_proj[expert], self.down_scale[expert])
                output.index_add_(0, selected, (y * weights[selected, selected_slots, None]).to(x.dtype))
                start += count
        return output

    def forward_reference(self, x, indices, weights):
        from .quant import int8_linear
        output = torch.zeros_like(x)
        for expert in range(self.gate_up_proj.shape[0]):
            tokens, slots = torch.where(indices == expert)
            if tokens.numel() == 0:
                continue
            gate, up = int8_linear(x[tokens], self.gate_up_proj[expert], self.gate_up_scale[expert]).chunk(2, -1)
            y = int8_linear(F.silu(gate) * up, self.down_proj[expert], self.down_scale[expert])
            output.index_add_(0, tokens, (y * weights[tokens, slots, None]).to(x.dtype))
        return output


@torch.no_grad()
def quantize_moe_experts(model, group_size=128):
    """Quantize target and MTP banks, releasing each replaced BF16 module."""
    names = [n for n, m in model.named_modules() if isinstance(m, Qwen3_5Experts)]
    for name in names:
        module = model.get_submodule(name)
        if group_size <= 0 or any(w.shape[-1] % group_size for w in (module.gate_up_proj, module.down_proj)):
            raise ValueError('group size must divide both expert input dimensions')
    for name in names:
        module = model.get_submodule(name)
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent), child, Int8Experts.from_float(module, group_size))
        del module
    return names
