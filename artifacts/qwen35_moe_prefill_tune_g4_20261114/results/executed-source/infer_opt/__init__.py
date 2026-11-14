"""A small, correctness-first decoder-only inference research scaffold."""

from .config import ModelConfig, RunConfig
from .model import DecoderOnlyTransformer

__all__ = ["DecoderOnlyTransformer", "ModelConfig", "RunConfig"]
