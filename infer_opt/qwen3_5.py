"""Standalone Qwen3.5 text model, copied from Hugging Face transformers 5.20.0.

Source: transformers/models/qwen3_5/modeling_qwen3_5.py (Qwen3_5ForCausalLM path).
Only the transformers dependencies were removed; the math is unchanged:
  * kernel-hub dispatch is dropped: the torch fallbacks are the default, and
    ``set_kernels("fla")`` switches GatedDeltaNet to flash-linear-attention Triton kernels
  * the cache is a plain Qwen3_5Cache (KV for full attention, conv + recurrent state for
    GatedDeltaNet), dynamic or preallocated (static); attention-mask utilities (padding), vision tower and MTP head are not included
  * attention uses torch SDPA with is_causal=True (inputs are unpadded); the fixed-shape decode step attends with
    plain fp32 reductions over the grouped query heads instead (no repeat_kv)
  * ``fuse_projections()`` optionally stacks the projections that share an input (gate+up, q+k+v, in_proj_qkv+z)

Module names follow the checkpoint (``model.language_model.*``, tied ``lm_head``),
so ``load_state_dict`` takes the safetensors keys as they are.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


KERNEL_CHOICES = ("torch", "fla", "fla-delta")


def _load_fla():
    try:
        from fla.modules.convolution import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    except ImportError as exc:
        raise ImportError(
            'kernels="fla" needs flash-linear-attention and a CUDA GPU: pip install flash-linear-attention'
        ) from exc
    return causal_conv1d, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


@dataclass
class Qwen3_5TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    layer_types: list[str]
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: list[int]
    tie_word_embeddings: bool = True
    # GatedDeltaNet kernels: "torch" (reference), "fla" (flash-linear-attention delta rule + causal conv),
    # or "fla-delta" (FLA delta rule only, torch conv)
    kernels: str = "torch"

    def layer_summary(self) -> dict:
        """Layer layout for result files: counts plus a one-letter-per-layer pattern (L linear, F full)."""
        code = {"linear_attention": "L", "full_attention": "F"}
        return {
            "num_layers": len(self.layer_types),
            "linear_attention": self.layer_types.count("linear_attention"),
            "full_attention": self.layer_types.count("full_attention"),
            "pattern": "".join(code.get(t, "?") for t in self.layer_types),
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3_5TextConfig":
        raw = json.loads(Path(path).read_text())
        text = raw["text_config"]
        rope = text["rope_parameters"]
        return cls(
            vocab_size=text["vocab_size"],
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            rms_norm_eps=text["rms_norm_eps"],
            layer_types=list(text["layer_types"]),
            linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"],
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_num_value_heads=text["linear_num_value_heads"],
            rope_theta=rope["rope_theta"],
            partial_rotary_factor=rope.get("partial_rotary_factor", 1.0),
            mrope_section=list(rope.get("mrope_section", [11, 11, 10])),
            tie_word_embeddings=text.get("tie_word_embeddings", raw.get("tie_word_embeddings", True)),
        )


class Qwen3_5RMSNorm(nn.Module):
    """Zero-centered RMSNorm: y = norm(x) * (1 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class Qwen3_5TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        dim = int(config.head_dim * config.partial_rotary_factor)
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = config.mrope_section

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (3, bs, positions) -> freqs: (3, bs, positions, dim // 2)
        position_ids = position_ids.expand(3, -1, -1)
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        cos = self.recomposition_frequencies(freqs.cos())
        sin = self.recomposition_frequencies(freqs.sin())
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def recomposition_frequencies(self, freq):
        freqs_thw = freq[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _fuse_into(module: nn.Module, fused_name: str, names: tuple[str, ...]) -> bool:
    """Replace the Linears ``names`` of ``module`` (same input, no bias) by one Linear ``fused_name`` whose output is
    theirs side by side: the weight rows (and int8 scales) stacked. One batch-1 GEMV over the stacked matrix streams
    the same bytes as the separate ones with one launch, ramp and tail instead of several. Returns whether it fused:
    the parts must be all ``nn.Linear`` of one dtype, or all :class:`~infer_opt.quant.Int8Linear` of one group size."""
    from infer_opt.quant import Int8Linear

    if getattr(module, fused_name) is not None:
        return False
    parts = [getattr(module, name) for name in names]
    if any(p.bias is not None for p in parts):
        return False
    if all(isinstance(p, Int8Linear) for p in parts) and len({(p.group_size, p.scale.dtype) for p in parts}) == 1:
        fused = Int8Linear(torch.cat([p.weight for p in parts]), torch.cat([p.scale for p in parts]))
    elif all(type(p) is nn.Linear for p in parts) and len({p.weight.dtype for p in parts}) == 1:
        weight = torch.cat([p.weight.detach() for p in parts])
        fused = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
        fused.weight = nn.Parameter(weight, requires_grad=parts[0].weight.requires_grad)
    else:
        return False
    for name in names:
        delattr(module, name)
    setattr(module, fused_name, fused)
    return True


class Qwen3_5Attention(nn.Module):
    """Full attention with output gate (q_proj emits query and gate)."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.qkv_proj = None  # q_proj, k_proj and v_proj stacked, after fuse_projections()
        self.qkv_sizes = [self.q_proj.out_features, self.k_proj.out_features, self.v_proj.out_features]

    def fuse_projections(self) -> bool:
        return _fuse_into(self, "qkv_proj", ("q_proj", "k_proj", "v_proj"))

    def forward(self, hidden_states, position_embeddings, cache: Qwen3_5Cache | None = None, decode: bool = False):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.qkv_proj is not None:
            q_out, k_out, v_out = self.qkv_proj(hidden_states).split(self.qkv_sizes, dim=-1)
        else:
            q_out, k_out, v_out = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        query_states, gate = torch.chunk(q_out.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(k_out.view(hidden_shape)).transpose(1, 2)
        value_states = v_out.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if decode:
            # fixed shapes: write at the device-side position, attend over the whole buffer, mask the unfilled tail
            position = cache.position
            cache.keys[self.layer_idx].index_copy_(2, position, key_states)
            cache.values[self.layer_idx].index_copy_(2, position, value_states)
            key_states, value_states = cache.keys[self.layer_idx], cache.values[self.layer_idx]
            max_len = key_states.shape[2]
            # One query row per head: group the query heads by their KV head, so the cache is read once per KV head.
            # repeat_kv would copy the cache once per query head, and SDPA's kernels parallelize over query rows (one
            # here): on the 27B (24 query / 4 KV heads, head_dim 256) that path took ~110 us per layer for a 4.5 MB cache.
            # On CUDA a split-K Triton kernel; elsewhere the same math as fp32 reductions.
            batch, heads, _, dim = query_states.shape
            if query_states.is_cuda and dim & (dim - 1) == 0:
                from infer_opt.attention_kernels import decode_attention

                query = query_states.reshape(batch, -1, self.num_key_value_groups, dim)
                attn_output = decode_attention(query, key_states, value_states, position, self.scaling)
            else:
                query = query_states.reshape(batch, -1, self.num_key_value_groups, 1, dim).float()
                scores = (query * key_states.unsqueeze(2).float()).sum(-1) * self.scaling  # [B, KV heads, groups, L]
                unfilled = torch.arange(max_len, device=hidden_states.device) > position
                probs = scores.masked_fill(unfilled, float("-inf")).softmax(-1)
                attn_output = (probs.unsqueeze(-1) * value_states.unsqueeze(2).float()).sum(-2)  # [B, KV heads, groups, D]
                attn_output = attn_output.to(value_states.dtype)
            attn_output = attn_output.view(batch, heads, 1, dim)
        else:
            if cache is not None:
                key_states, value_states = cache.update_kv(self.layer_idx, key_states, value_states)

            q_len, kv_len = query_states.shape[2], key_states.shape[2]
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            if q_len == kv_len:
                attn_mask, is_causal = None, True
            elif q_len == 1:
                attn_mask, is_causal = None, False  # the new token sees the whole cache
            else:  # chunked prefill on top of a non-empty cache
                attn_mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=hidden_states.device)
                attn_mask = attn_mask.tril(diagonal=kv_len - q_len)
                is_causal = False
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attn_mask, is_causal=is_causal, scale=self.scaling
            )
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated delta rule by chunking the sequence (prefill path).

    query/key: [B, T, H, Dk], value: [B, T, H, Dv], g/beta: [B, T, H],
    initial_state: [B, H, Dk, Dv] or None. Returns (output [B, T, H, Dv], final_state).
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
    decay = g

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)

    cum_decay = decay.cumsum(dim=3)

    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    pairwise_decay = pairwise_decay.exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)

    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)

    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same math as the chunk rule, one token at a time (decode path)."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query / (query.shape[-1] ** 0.5)

    if initial_state is None:
        last_recurrent_state = torch.zeros(
            batch_size, num_v_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device
        )
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)

    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        last_recurrent_state = last_recurrent_state * decay[:, :, i].exp()[..., None, None]
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5Cache:
    """Per-layer inference state.

    full_attention  : K/V tensors, [B, kv_heads, S, head_dim]
    linear_attention: conv state [B, conv_dim, kernel - 1] (last pre-conv inputs)
                      and recurrent state [B, v_heads, Dk, Dv] in fp32 (constant size)

    Dynamic (default): K/V grow with ``torch.cat`` and states are rebound every step.
    Static (``Qwen3_5Cache.allocate``): every buffer is allocated once for ``max_seq_len``;
    K/V are written in place at ``seen_tokens`` and states are updated with ``copy_``, so a
    step allocates nothing and the addresses stay fixed (CUDA-graph friendly).
    """

    def __init__(self, num_layers: int):
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers
        self.conv_states: list[torch.Tensor | None] = [None] * num_layers
        self.recurrent_states: list[torch.Tensor | None] = [None] * num_layers
        self.seen_tokens = 0
        self.max_seq_len: int | None = None
        # static caches only: the write index as a device tensor, so a captured graph can advance it
        self.position: torch.Tensor | None = None

    @classmethod
    def allocate(cls, config: Qwen3_5TextConfig, batch_size: int, max_seq_len: int, dtype, device) -> "Qwen3_5Cache":
        cache = cls(config.num_hidden_layers)
        cache.max_seq_len = max_seq_len
        cache.position = torch.zeros(1, dtype=torch.long, device=device)
        conv_dim = 2 * config.linear_key_head_dim * config.linear_num_key_heads + (
            config.linear_value_head_dim * config.linear_num_value_heads
        )
        for i, layer_type in enumerate(config.layer_types):
            if layer_type == "full_attention":
                shape = (batch_size, config.num_key_value_heads, max_seq_len, config.head_dim)
                cache.keys[i] = torch.zeros(shape, dtype=dtype, device=device)
                cache.values[i] = torch.zeros(shape, dtype=dtype, device=device)
            else:
                cache.conv_states[i] = torch.zeros(
                    batch_size, conv_dim, config.linear_conv_kernel_dim - 1, dtype=dtype, device=device
                )
                cache.recurrent_states[i] = torch.zeros(
                    batch_size, config.linear_num_value_heads, config.linear_key_head_dim,
                    config.linear_value_head_dim, dtype=torch.float32, device=device,
                )
        return cache

    @property
    def is_static(self) -> bool:
        return self.max_seq_len is not None

    def reset(self) -> None:
        """Start a new sequence. Static K/V need no clearing (they are only read below ``seen_tokens``)."""
        self.seen_tokens = 0
        if self.is_static:
            self.position.zero_()
            for t in self.conv_states + self.recurrent_states:
                if t is not None:
                    t.zero_()
        else:
            n = len(self.keys)
            self.keys, self.values = [None] * n, [None] * n
            self.conv_states, self.recurrent_states = [None] * n, [None] * n

    def update_kv(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        if self.is_static:
            start, end = self.seen_tokens, self.seen_tokens + key.shape[2]
            if end > self.max_seq_len:
                raise ValueError(f"cache is full: {end} tokens > max_seq_len={self.max_seq_len}")
            self.keys[layer_idx][:, :, start:end] = key
            self.values[layer_idx][:, :, start:end] = value
            return self.keys[layer_idx][:, :, :end], self.values[layer_idx][:, :, :end]
        if self.keys[layer_idx] is None:
            self.keys[layer_idx], self.values[layer_idx] = key, value
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value], dim=2)
        return self.keys[layer_idx], self.values[layer_idx]

    def set_state(self, states: list, layer_idx: int, new: torch.Tensor) -> None:
        if self.is_static:
            states[layer_idx].copy_(new)
        else:
            states[layer_idx] = new.contiguous()

    def nbytes(self) -> int:
        tensors = [t for group in (self.keys, self.values, self.conv_states, self.recurrent_states) for t in group]
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.kernels = config.kernels

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_qkvz = None  # in_proj_qkv and in_proj_z stacked, after fuse_projections()

    def fuse_projections(self) -> bool:
        return _fuse_into(self, "in_proj_qkvz", ("in_proj_qkv", "in_proj_z"))

    def forward(self, hidden_states: torch.Tensor, cache: Qwen3_5Cache | None = None, decode: bool = False):
        batch_size, seq_len, _ = hidden_states.shape
        use_fla_conv = self.kernels == "fla"
        use_fla_delta = self.kernels in ("fla", "fla-delta")
        if use_fla_conv or use_fla_delta:
            fla_conv, fla_chunk, fla_recurrent = _load_fla()

        if self.in_proj_qkvz is not None:
            mixed_qkv, z = self.in_proj_qkvz(hidden_states).split([self.conv_dim, self.value_dim], dim=-1)
        else:
            mixed_qkv, z = self.in_proj_qkv(hidden_states), self.in_proj_z(hidden_states)  # [B, T, conv_dim], [B, T, value_dim]
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # causal depthwise conv + silu; the cache carries the last (kernel - 1) inputs
        weight = self.conv1d.weight.squeeze(1)  # [conv_dim, kernel]
        # an empty cache (seen_tokens == 0) holds all-zero states: pass None instead of reading them
        fresh = cache is None or (not decode and cache.seen_tokens == 0)
        conv_state = None if fresh else cache.conv_states[self.layer_idx]
        if use_fla_conv:
            # FLA keeps a state of `kernel` inputs; ours holds kernel - 1, so pad one zero column
            initial = None if conv_state is None else F.pad(conv_state, (1, 0))
            mixed_qkv, final = fla_conv(
                mixed_qkv, weight, None, initial_state=initial, output_final_state=cache is not None,
                activation="silu",
            )
            if cache is not None:
                cache.set_state(cache.conv_states, self.layer_idx, final[:, :, 1:])
        else:
            x = mixed_qkv.transpose(1, 2)
            if conv_state is None:
                conv_state = x.new_zeros(batch_size, self.conv_dim, self.conv_kernel_size - 1)
            conv_input = torch.cat([conv_state, x], dim=-1)
            if cache is not None:
                cache.set_state(cache.conv_states, self.layer_idx, conv_input[:, :, -(self.conv_kernel_size - 1):])
            if seq_len == 1:  # one output per channel: a dot product over the kernel window, which inductor can fuse
                conv_out = (conv_input.float() * weight.float()).sum(-1, keepdim=True).to(weight.dtype)
            else:
                conv_out = F.conv1d(conv_input.to(weight.dtype), weight=weight.unsqueeze(1), bias=None, groups=self.conv_dim)
            mixed_qkv = F.silu(conv_out).to(hidden_states.dtype).transpose(1, 2)

        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        recurrent_state = None if fresh else cache.recurrent_states[self.layer_idx]
        if use_fla_delta:
            rule = fla_recurrent if seq_len == 1 else fla_chunk
            core_attn_out, last_recurrent_state = rule(
                query.contiguous(), key.contiguous(), value.contiguous(), g=g, beta=beta,
                initial_state=recurrent_state, output_final_state=cache is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            rule = torch_recurrent_gated_delta_rule if seq_len == 1 else torch_chunk_gated_delta_rule
            core_attn_out, last_recurrent_state = rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state, use_qk_l2norm_in_kernel=True
            )
        if cache is not None:
            cache.set_state(cache.recurrent_states, self.layer_idx, last_recurrent_state)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.gate_up_proj = None  # gate_proj and up_proj stacked, after fuse_projections()

    def fuse_projections(self) -> bool:
        return _fuse_into(self, "gate_up_proj", ("gate_proj", "up_proj"))

    def forward(self, x):
        if self.gate_up_proj is not None:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"unknown layer type {self.block_type!r}")
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, cache: Qwen3_5Cache | None = None, decode: bool = False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, cache, decode)
        else:
            hidden_states = self.self_attn(hidden_states, position_embeddings, cache, decode)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)

    def forward(self, input_ids: torch.LongTensor, cache: Qwen3_5Cache | None = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        past = cache.seen_tokens if cache is not None else 0
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device) + past
        position_ids = position_ids.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, cache)
        if cache is not None:
            cache.seen_tokens += input_ids.shape[1]
            if cache.is_static:
                cache.position.fill_(cache.seen_tokens)
        return self.norm(hidden_states)

    def forward_decode(self, input_ids: torch.LongTensor, cache: Qwen3_5Cache) -> torch.Tensor:
        """One token per sequence with fixed shapes. The position lives on the device (``cache.position``) and
        is advanced here, so nothing in this function depends on a Python int that changes between steps."""
        hidden_states = self.embed_tokens(input_ids)
        position_ids = cache.position.view(1, 1, 1).expand(3, input_ids.shape[0], 1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, cache, decode=True)
        cache.position.add_(1)
        return self.norm(hidden_states)


class _Backbone(nn.Module):
    """Holds ``language_model`` so state-dict keys read ``model.language_model.*``."""

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.language_model = Qwen3_5TextModel(config)


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.model = _Backbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def new_cache(self, batch_size: int = 1, max_seq_len: int | None = None) -> Qwen3_5Cache:
        """Dynamic cache by default; pass ``max_seq_len`` for preallocated (static) buffers."""
        if max_seq_len is None:
            return Qwen3_5Cache(self.config.num_hidden_layers)
        weight = self.model.language_model.embed_tokens.weight  # lm_head may be quantized; the embedding never is
        return Qwen3_5Cache.allocate(self.config, batch_size, max_seq_len, weight.dtype, weight.device)

    def set_kernels(self, kernels: str) -> "Qwen3_5ForCausalLM":
        if kernels not in KERNEL_CHOICES:
            raise ValueError(f"kernels must be one of {KERNEL_CHOICES}, got {kernels!r}")
        if kernels != "torch":
            _load_fla()
        self.config.kernels = kernels
        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.kernels = kernels
        return self

    @torch.no_grad()
    def fuse_projections(self) -> int:
        """Stack the projections that read the same input into one Linear each: gate+up (MLP), q+k+v (full attention)
        and in_proj_qkv+in_proj_z (GatedDeltaNet). Works before or after :func:`~infer_opt.quant.quantize_linears`
        (group-wise scales are per row, so they stack too). Decoding the int8 Qwen3.5-27B, gate+up alone saves
        ~26 us of the 140 us its two GEMVs take. The state-dict keys change: save checkpoints before fusing.
        Returns the number of fused groups."""
        return sum(m.fuse_projections() for m in self.modules()
                   if isinstance(m, (Qwen3_5MLP, Qwen3_5Attention, Qwen3_5GatedDeltaNet)))

    def forward(self, input_ids: torch.LongTensor, cache: Qwen3_5Cache | None = None) -> torch.Tensor:
        return self.lm_head(self.model.language_model(input_ids, cache))

    def decode_logits(self, input_ids: torch.LongTensor, cache: Qwen3_5Cache) -> torch.Tensor:
        """Fixed-shape single-token step on a static cache (the function that gets compiled / graph-captured).
        It does not touch ``cache.seen_tokens``; use :meth:`decode_step` or :class:`FixedShapeDecoder`."""
        return self.lm_head(self.model.language_model.forward_decode(input_ids, cache))

    @torch.no_grad()
    def decode_step(self, input_ids: torch.LongTensor, cache: Qwen3_5Cache) -> torch.Tensor:
        if not cache.is_static:
            raise ValueError("decode_step needs a static cache: model.new_cache(batch, max_seq_len)")
        if cache.seen_tokens + 1 > cache.max_seq_len:
            raise ValueError(f"cache is full: max_seq_len={cache.max_seq_len}")
        logits = self.decode_logits(input_ids, cache)
        cache.seen_tokens += 1
        return logits

    @torch.no_grad()
    def generate_greedy(
        self, input_ids: torch.LongTensor, max_new_tokens: int, use_cache: bool = True, static_cache: bool = False,
        decode_mode: str | None = None,
    ) -> torch.LongTensor:
        """Greedy decoding. With a cache: one prefill, then one token per step.
        Without: re-runs the full prefix each step (reference for testing the cache).
        ``static_cache`` preallocates K/V for prompt + new tokens. ``decode_mode`` (see
        :class:`FixedShapeDecoder`) runs the decode steps through the fixed-shape path."""
        if decode_mode is not None:
            cache = self.new_cache(input_ids.shape[0], input_ids.shape[1] + max_new_tokens)
            decoder = FixedShapeDecoder(self, cache, decode_mode)
            token = self(input_ids, cache)[:, -1].argmax(-1, keepdim=True)
            decoder.prepare()
            out = [input_ids, token]
            for _ in range(max_new_tokens - 1):
                token = decoder.step(token)[:, -1].argmax(-1, keepdim=True)
                out.append(token)
            return torch.cat(out, dim=-1)
        if use_cache:
            max_len = input_ids.shape[1] + max_new_tokens if static_cache else None
            cache = self.new_cache(input_ids.shape[0], max_len)
        else:
            cache = None
        step_input = input_ids
        for _ in range(max_new_tokens):
            next_token = self(step_input, cache)[:, -1].argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            step_input = next_token if use_cache else input_ids
        return input_ids

    @classmethod
    def from_pretrained_dir(
        cls, path: str | Path, dtype=torch.bfloat16, device="cpu", kernels: str = "torch"
    ) -> "Qwen3_5ForCausalLM":
        """Load safetensors shards from a local HF checkpoint dir; keys are used as-is.

        Tensors are read straight onto ``device``, so a large checkpoint (52 GiB for Qwen3.5-27B) is not staged in host
        memory first."""
        from safetensors import safe_open

        path = Path(path)
        config = Qwen3_5TextConfig.from_json(path / "config.json")
        with torch.device("meta"):
            model = cls(config)
        state = {}
        for shard in sorted(path.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device=str(device)) as f:
                for key in f.keys():
                    # vision tower and multi-token-prediction head are not part of the text model
                    if key.startswith(("model.visual.", "mtp.")):
                        continue
                    state[key] = f.get_tensor(key)
        model.load_state_dict(state, strict=False, assign=True)
        # the non-persistent rope buffer is not in the checkpoint and was created on meta
        model.model.language_model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        if config.tie_word_embeddings:
            model.lm_head.weight = model.model.language_model.embed_tokens.weight
        missing = [k for k, p in model.named_parameters() if p.is_meta]
        if missing:
            raise RuntimeError(f"parameters not loaded: {missing}")
        model.to(device=device, dtype=dtype)
        return model.set_kernels(kernels).eval()

    @classmethod
    def from_quantized_dir(
        cls, path: str | Path, device="cpu", kernels: str = "torch", dtype: torch.dtype | None = None
    ) -> "Qwen3_5ForCausalLM":
        """Load a checkpoint written by :func:`infer_opt.quant.save_quantized` (``config.json`` next to it).

        Tensors are read straight onto ``device`` with the dtypes they were saved with (int8 weights, bf16 scales and
        the rest), unless ``dtype`` is given: then every floating-point tensor is cast to it and the int8 weights stay
        int8. ``dtype=torch.float16`` is for GPUs without bf16 (Turing: T4, RTX 20xx).
        """
        from safetensors import safe_open

        from infer_opt.quant import QUANTIZATION_FILE, prepare_quantized

        path = Path(path)
        config = Qwen3_5TextConfig.from_json(path / "config.json")
        spec = json.loads((path / QUANTIZATION_FILE).read_text())
        with torch.device("meta"):
            model = cls(config)
        prepare_quantized(model, spec)
        state = {}
        for shard in sorted(path.glob("model-int8-*.safetensors")):
            with safe_open(shard, framework="pt", device=str(device)) as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    state[key] = tensor.to(dtype) if dtype is not None and tensor.is_floating_point() else tensor
        for name, source in spec["tied"].items():
            state[name] = state[source]
        model.load_state_dict(state, strict=True, assign=True)
        # from_pretrained_dir casts the whole model, the rope buffer included; match it so the logits are the same
        rotary = Qwen3_5TextRotaryEmbedding(config)
        rotary.inv_freq = rotary.inv_freq.to(model.model.language_model.embed_tokens.weight.dtype)
        model.model.language_model.rotary_emb = rotary
        if config.tie_word_embeddings and not spec["modules"].get("lm_head"):
            model.lm_head.weight = model.model.language_model.embed_tokens.weight
        return model.to(device).set_kernels(kernels).eval()


DECODE_MODES = ("eager", "compile", "compile-reduce-overhead", "cudagraph")


class FixedShapeDecoder:
    """Runs single-token decode steps on a static cache through the fixed-shape path.

    modes: ``eager`` (no compilation), ``compile`` (torch.compile), ``compile-reduce-overhead``
    (torch.compile with CUDA graphs) and ``cudagraph`` (one hand-captured CUDA graph).
    Call :meth:`prepare` once (for ``cudagraph`` it captures the graph) and :meth:`step` per token.
    ``step`` returns a buffer that the next step overwrites when the mode is ``cudagraph``.
    """

    def __init__(self, model: Qwen3_5ForCausalLM, cache: Qwen3_5Cache, mode: str = "eager", fullgraph: bool = True):
        if mode not in DECODE_MODES:
            raise ValueError(f"mode must be one of {DECODE_MODES}, got {mode!r}")
        if not cache.is_static:
            raise ValueError("FixedShapeDecoder needs a static cache")
        self.model, self.cache, self.mode = model, cache, mode
        self.batch_size = next(t.shape[0] for t in cache.recurrent_states if t is not None)
        self.graph = None
        # every mode feeds the model from this one buffer: callers pass views with ever-changing strides/offsets,
        # which would otherwise trip torch.compile's tensor guards and force a recompile per distinct view
        self.static_token = torch.zeros(self.batch_size, 1, dtype=torch.long, device=cache.position.device)
        self.fn = model.decode_logits
        if mode.startswith("compile"):
            for t in self._state_tensors() + [self.static_token]:
                torch._dynamo.mark_static_address(t)
            options = {"mode": "reduce-overhead"} if mode == "compile-reduce-overhead" else {}
            self.fn = torch.compile(model.decode_logits, fullgraph=fullgraph, dynamic=False, **options)

    def _state_tensors(self) -> list[torch.Tensor]:
        c = self.cache
        return [t for group in (c.keys, c.values, c.conv_states, c.recurrent_states) for t in group
                if t is not None] + [c.position]

    @torch.no_grad()
    def prepare(self) -> None:
        """Capture the CUDA graph (``cudagraph`` mode). Can run at any time: the warm-up steps scribble on the
        recurrent/conv state and position, which are restored afterwards (it needs 4 free cache slots)."""
        if self.mode != "cudagraph":
            return
        cache = self.cache
        if cache.seen_tokens + 4 > cache.max_seq_len:
            raise ValueError("need 4 free cache slots for the capture warm-up")
        saved = [t.clone() for t in cache.conv_states + cache.recurrent_states if t is not None]
        saved_position = cache.position.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.model.decode_logits(self.static_token, cache)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits = self.model.decode_logits(self.static_token, cache)
        live = [t for t in cache.conv_states + cache.recurrent_states if t is not None]
        for dst, src in zip(live, saved):
            dst.copy_(src)
        cache.position.copy_(saved_position)

    @torch.no_grad()
    def step(self, input_ids: torch.LongTensor) -> torch.Tensor:
        cache = self.cache
        if cache.seen_tokens + 1 > cache.max_seq_len:
            raise ValueError(f"cache is full: max_seq_len={cache.max_seq_len}")
        self.static_token.copy_(input_ids)
        if self.mode == "cudagraph":
            if self.graph is None:
                raise RuntimeError("call prepare() after prefill before stepping a cudagraph decoder")
            self.graph.replay()
            logits = self.static_logits
        elif self.mode.startswith("compile"):
            from .inductor_choices import wide_persistent_reductions

            with wide_persistent_reductions():  # compiles on the first call (and on any recompile)
                logits = self.fn(self.static_token, cache)
        else:
            logits = self.fn(self.static_token, cache)
        cache.seen_tokens += 1
        return logits
