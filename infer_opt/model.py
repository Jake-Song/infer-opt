from __future__ import annotations

import torch
from torch import nn

from .cache import KVCache
from .config import ModelConfig
from .strategies import AttentionStrategy, make_attention_strategy


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    angles = torch.outer(positions.float(), inv_freq).to(x.dtype)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, strategy: AttentionStrategy) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = config.n_heads, config.n_kv_heads, config.head_dim
        self.strategy = strategy
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, layer_idx: int, cache: KVCache | None = None) -> torch.Tensor:
        batch, tokens, _ = x.shape
        if cache is not None and batch != cache.batch_size:
            raise ValueError("input batch size differs from KV cache batch size")
        start = cache.position if cache is not None else 0
        positions = torch.arange(start, start + tokens, device=x.device)
        q = self.q_proj(x).view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, positions), apply_rope(k, positions)
        if cache is not None:
            cache.write(layer_idx, k, v, start)
            k, v = cache.read(layer_idx, start + tokens)
        output = self.strategy(q, k, v, query_start=start)
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch, tokens, -1))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, strategy: AttentionStrategy) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config, strategy)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, layer_idx: int, cache: KVCache | None = None) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x), layer_idx, cache)
        return x + self.mlp(self.mlp_norm(x))


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig, strategy_name: str = "eager_sdpa") -> None:
        super().__init__()
        self.config = config
        strategy = make_attention_strategy(strategy_name)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config, strategy) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        x = self.embedding(token_ids)
        for idx, layer in enumerate(self.layers):
            x = layer(x, idx, cache)
        return self.lm_head(self.norm(x))

    def forward_prefill(self, token_ids: torch.Tensor, cache: KVCache) -> torch.Tensor:
        if cache.position != 0:
            raise ValueError("prefill requires an empty KV cache; call cache.reset() first")
        cache.validate_append(token_ids.shape[1])
        logits = self(token_ids, cache)
        cache.advance(token_ids.shape[1])
        return logits

    def forward_decode(self, token_ids: torch.Tensor, cache: KVCache) -> torch.Tensor:
        if token_ids.shape[1] != 1:
            raise ValueError("forward_decode accepts exactly one token; use forward_prefill for prompts")
        cache.validate_append(1)
        logits = self(token_ids, cache)
        cache.advance(1)
        return logits

    def new_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> KVCache:
        return KVCache(
            n_layers=self.config.n_layers,
            batch_size=batch_size,
            n_kv_heads=self.config.n_kv_heads,
            max_seq_len=self.config.max_seq_len,
            head_dim=self.config.head_dim,
            device=device,
            dtype=dtype,
        )


def make_random_model(config: ModelConfig, strategy_name: str, seed: int) -> DecoderOnlyTransformer:
    torch.manual_seed(seed)
    return DecoderOnlyTransformer(config, strategy_name)
