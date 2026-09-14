import pytest

torch = pytest.importorskip("torch")

from infer_opt.config import ModelConfig
from infer_opt.model import RotaryEmbedding, apply_rope, apply_rope_naive, make_random_model

HEAD_DIM, MAX_SEQ = 64, 2048


def _reference(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """RoPE computed entirely in float64: the answer both paths approximate."""
    head_dim = x.shape[-1]
    steps = torch.arange(0, head_dim, 2, dtype=torch.float64)
    inv_freq = 1.0 / (10000 ** (steps / head_dim))
    angles = torch.outer(positions.to(torch.float64), inv_freq)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    wide = x.to(torch.float64)
    even, odd = wide[..., 0::2], wide[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def _errors(position: int) -> tuple[float, float]:
    torch.manual_seed(0)
    x = torch.randn(1, 4, 1, HEAD_DIM, dtype=torch.float16)
    positions = torch.tensor([position])
    rope = RotaryEmbedding(HEAD_DIM, MAX_SEQ)
    cos, sin = rope.slice(position, 1, torch.device("cpu"))
    exact = _reference(x, positions)
    naive = (apply_rope_naive(x, positions).to(torch.float64) - exact).abs().max().item()
    table = (apply_rope(x, cos, sin).to(torch.float64) - exact).abs().max().item()
    return naive, table


@pytest.mark.parametrize("position", [512, 1024, 2047])
def test_table_rope_beats_naive_rope_at_distant_positions(position: int) -> None:
    """The naive path rounds the angle to float16 before taking its cosine.

    Half precision steps by 1.0 near 1024, so the further out the position the
    worse the rotation. The float32 table does not have that failure mode.
    """
    naive, table = _errors(position)
    assert table < naive / 10


def test_the_two_rope_paths_agree_at_small_positions() -> None:
    naive, table = _errors(4)
    assert naive < 1e-2 and table < 1e-2


def test_rope_error_of_the_table_path_does_not_grow_with_position() -> None:
    near, far = _errors(16)[1], _errors(2047)[1]
    assert far < 10 * max(near, 1e-6)


def test_naive_rope_error_grows_with_position() -> None:
    assert _errors(2047)[0] > 10 * _errors(16)[0]


def test_rope_table_refuses_positions_it_does_not_cover() -> None:
    rope = RotaryEmbedding(HEAD_DIM, 32)
    with pytest.raises(ValueError, match="rope table covers"):
        rope.slice(30, 4, torch.device("cpu"))


def test_one_rope_table_is_shared_by_every_layer() -> None:
    config = ModelConfig(hidden_size=32, n_heads=4, n_kv_heads=2, intermediate_size=64, n_layers=3, max_seq_len=16)
    model = make_random_model(config, "eager_sdpa", seed=0)
    assert all(layer.attention.rope is model.rope for layer in model.layers)


def test_rope_tables_survive_a_half_precision_cast() -> None:
    """``model.to(float16)`` converts buffers; the angle tables must not be."""
    config = ModelConfig(hidden_size=32, n_heads=4, n_kv_heads=2, intermediate_size=64, n_layers=2, max_seq_len=16)
    model = make_random_model(config, "eager_sdpa", seed=0).to(dtype=torch.float16)
    cos, sin = model.rope.slice(0, 8, torch.device("cpu"))
    assert cos.dtype is torch.float32 and sin.dtype is torch.float32
