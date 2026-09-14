from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    intermediate_size: int = 352
    max_seq_len: int = 256
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_size % self.n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if min(self.vocab_size, self.n_layers, self.intermediate_size, self.max_seq_len) <= 0:
            raise ValueError("model dimensions must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    batch_size: int = 1
    prompt_len: int = 32
    decode_len: int = 16
    iterations: int = 10
    warmup: int = 2
    seed: int = 0
    device: str = "auto"
    dtype: str = "float32"
    strategy: str = "eager_sdpa"
    cache: str = "contiguous"
    block_size: int = 16

    def __post_init__(self) -> None:
        if min(self.batch_size, self.prompt_len, self.decode_len, self.iterations) <= 0:
            raise ValueError("batch_size, prompt_len, decode_len, and iterations must be positive")
        if self.warmup < 0:
            raise ValueError("warmup cannot be negative")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if self.cache not in {"contiguous", "paged"}:
            raise ValueError("cache must be contiguous or paged")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)
