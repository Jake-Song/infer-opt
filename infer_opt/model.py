from __future__ import annotations

import torch
from torch import nn

from .cache import KVCache
from .paged import PagedKVCache
from .config import ModelConfig
from .regions import region
from .strategies import AttentionStrategy, make_attention_strategy


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


def apply_rope_naive(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` by rebuilding the angle table on every call.

    Kept as the chapter-2 exhibit, not as the model's path. It is wrong twice
    over: it recomputes a table that never changes, and it rounds the *angle*
    to ``x.dtype`` before taking cos/sin. In float16 the spacing at 1024 is a
    full 1.0, so a position of 2047 lands up to half a radian off and its
    cosine is wrong by 0.16 -- an error that grows with position, exactly
    where a long context lives. :class:`RotaryEmbedding` fixes both.
    """
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    angles = torch.outer(positions.float(), inv_freq).to(x.dtype)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    return _rotate(x, cos, sin)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` by angle tables already sliced to its positions."""
    return _rotate(x, cos.to(x.dtype)[None, None], sin.to(x.dtype)[None, None])


class RotaryEmbedding(nn.Module):
    """Position angle tables for RoPE, built once per device and kept in float32.

    The tables are plain attributes rather than registered buffers on purpose:
    ``model.to(torch.float16)`` converts buffers, and halving the angles is the
    very bug this class exists to avoid. Only the cosine and sine are cast, at
    the point of use, where their values are bounded by one.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim, self.max_seq_len, self.base = head_dim, max_seq_len, base
        self._tables: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def _build(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        steps = torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (steps / self.head_dim))
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        return angles.cos(), angles.sin()

    def slice(self, start: int, tokens: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Cosine and sine for ``tokens`` positions beginning at ``start``."""
        if start + tokens > self.max_seq_len:
            raise ValueError(
                f"rope table covers {self.max_seq_len} positions; asked for {start + tokens}"
            )
        if device not in self._tables:
            self._tables[device] = self._build(device)
        cos, sin = self._tables[device]
        return cos[start : start + tokens], sin[start : start + tokens]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, strategy: AttentionStrategy, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = config.n_heads, config.n_kv_heads, config.head_dim
        self.strategy = strategy
        self.rope = rope
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, layer_idx: int, cache: KVCache | None = None) -> torch.Tensor:
        batch, tokens, _ = x.shape
        if cache is not None and batch != cache.batch_size:
            raise ValueError("input batch size differs from KV cache batch size")
        start = cache.position if cache is not None else 0
        with region("qkv_proj"):
            q = self.q_proj(x).view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch, tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch, tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)
        with region("rope"):
            cos, sin = self.rope.slice(start, tokens, x.device)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            with region("kv_write"):
                cache.write(layer_idx, k, v, start)
            with region("kv_read"):
                k, v = cache.read(layer_idx, start + tokens)
        with region("attention"):
            output = self.strategy(q, k, v, query_start=start)
        with region("o_proj"):
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
    def __init__(self, config: ModelConfig, strategy: AttentionStrategy, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config, strategy, rope)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, layer_idx: int, cache: KVCache | None = None) -> torch.Tensor:
        with region("norm"):
            normed = self.attn_norm(x)
        x = x + self.attention(normed, layer_idx, cache)
        with region("norm"):
            normed = self.mlp_norm(x)
        with region("mlp"):
            return x + self.mlp(normed)


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig, strategy_name: str = "eager_sdpa") -> None:
        super().__init__()
        self.config = config
        strategy = make_attention_strategy(strategy_name)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
        self.layers = nn.ModuleList(
            DecoderLayer(config, strategy, self.rope) for _ in range(config.n_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        with region("embedding"):
            x = self.embedding(token_ids)
        for idx, layer in enumerate(self.layers):
            x = layer(x, idx, cache)
        with region("norm"):
            x = self.norm(x)
        with region("lm_head"):
            return self.lm_head(x)

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

    def _cache_arguments(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict:
        return {
            "n_layers": self.config.n_layers,
            "batch_size": batch_size,
            "n_kv_heads": self.config.n_kv_heads,
            "max_seq_len": self.config.max_seq_len,
            "head_dim": self.config.head_dim,
            "device": device,
            "dtype": dtype,
        }

    def new_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> KVCache:
        return KVCache(**self._cache_arguments(batch_size, device, dtype))

    def new_paged_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        block_size: int = 16,
        num_blocks: int | None = None,
    ) -> PagedKVCache:
        """A block-structured cache the forward pass accepts in place of :class:`KVCache`."""
        return PagedKVCache(
            **self._cache_arguments(batch_size, device, dtype),
            block_size=block_size,
            num_blocks=num_blocks,
        )


def make_random_model(config: ModelConfig, strategy_name: str, seed: int) -> DecoderOnlyTransformer:
    torch.manual_seed(seed)
    return DecoderOnlyTransformer(config, strategy_name)
