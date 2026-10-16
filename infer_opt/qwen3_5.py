"""Standalone Qwen3.5 text model, copied from Hugging Face transformers 5.20.0.

Source: transformers/models/qwen3_5/modeling_qwen3_5.py (Qwen3_5ForCausalLM path).
Only the transformers dependencies were removed; the math is unchanged:
  * kernel-hub / FLA / causal-conv1d dispatch is dropped, the torch fallbacks are used
  * the cache is a plain Qwen3_5Cache (KV for full attention, conv + recurrent state for
    GatedDeltaNet); attention-mask utilities (padding), vision tower and MTP head are not included
  * attention uses torch SDPA with is_causal=True (inputs are unpadded)

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

    def forward(self, hidden_states, position_embeddings, cache: Qwen3_5Cache | None = None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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

    full_attention  : growing K/V tensors, [B, kv_heads, S, head_dim]
    linear_attention: conv state [B, conv_dim, kernel - 1] (last pre-conv inputs)
                      and recurrent state [B, v_heads, Dk, Dv] in fp32 (constant size)
    """

    def __init__(self, num_layers: int):
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers
        self.conv_states: list[torch.Tensor | None] = [None] * num_layers
        self.recurrent_states: list[torch.Tensor | None] = [None] * num_layers
        self.seen_tokens = 0

    def update_kv(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        if self.keys[layer_idx] is None:
            self.keys[layer_idx], self.values[layer_idx] = key, value
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value], dim=2)
        return self.keys[layer_idx], self.values[layer_idx]

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

    def forward(self, hidden_states: torch.Tensor, cache: Qwen3_5Cache | None = None):
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # causal depthwise conv + silu; the cache carries the last (kernel - 1) inputs
        weight = self.conv1d.weight.squeeze(1)
        state_len = self.conv_kernel_size - 1
        conv_state = cache.conv_states[self.layer_idx] if cache is not None else None
        if conv_state is None:
            conv_state = mixed_qkv.new_zeros(batch_size, self.conv_dim, state_len)
        conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)
        if cache is not None:
            cache.conv_states[self.layer_idx] = conv_input[:, :, -state_len:].contiguous()
        mixed_qkv = F.silu(
            F.conv1d(conv_input.to(weight.dtype), weight=weight.unsqueeze(1), bias=None, groups=self.conv_dim)
        ).to(hidden_states.dtype)

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        recurrent_state = cache.recurrent_states[self.layer_idx] if cache is not None else None
        rule = torch_recurrent_gated_delta_rule if seq_len == 1 else torch_chunk_gated_delta_rule
        core_attn_out, last_recurrent_state = rule(
            query, key, value, g=g, beta=beta, initial_state=recurrent_state, use_qk_l2norm_in_kernel=True
        )
        if cache is not None:
            cache.recurrent_states[self.layer_idx] = last_recurrent_state

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

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


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

    def forward(self, hidden_states, position_embeddings, cache: Qwen3_5Cache | None = None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, cache)
        else:
            hidden_states = self.self_attn(hidden_states, position_embeddings, cache)
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

    def new_cache(self) -> Qwen3_5Cache:
        return Qwen3_5Cache(self.config.num_hidden_layers)

    def forward(self, input_ids: torch.LongTensor, cache: Qwen3_5Cache | None = None) -> torch.Tensor:
        return self.lm_head(self.model.language_model(input_ids, cache))

    @torch.no_grad()
    def generate_greedy(
        self, input_ids: torch.LongTensor, max_new_tokens: int, use_cache: bool = True
    ) -> torch.LongTensor:
        """Greedy decoding. With a cache: one prefill, then one token per step.
        Without: re-runs the full prefix each step (reference for testing the cache)."""
        cache = self.new_cache() if use_cache else None
        step_input = input_ids
        for _ in range(max_new_tokens):
            next_token = self(step_input, cache)[:, -1].argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            step_input = next_token if use_cache else input_ids
        return input_ids

    @classmethod
    def from_pretrained_dir(cls, path: str | Path, dtype=torch.bfloat16, device="cpu") -> "Qwen3_5ForCausalLM":
        """Load safetensors shards from a local HF checkpoint dir; keys are used as-is."""
        from safetensors import safe_open

        path = Path(path)
        config = Qwen3_5TextConfig.from_json(path / "config.json")
        with torch.device("meta"):
            model = cls(config)
        state = {}
        for shard in sorted(path.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as f:
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
        return model.eval()
