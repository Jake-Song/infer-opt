import pytest

torch = pytest.importorskip("torch")

from infer_opt.sampling import greedy, last_token_logits, sample_top_k, sample_top_p


def _logits() -> torch.Tensor:
    return torch.tensor([[0.0, 5.0, 1.0, -2.0, 3.0], [4.0, 0.5, 0.0, 2.0, -1.0]])


def test_greedy_picks_the_highest_logit() -> None:
    chosen = greedy(_logits())
    assert chosen.shape == (2, 1)
    assert chosen.flatten().tolist() == [1, 0]


def test_top_k_never_chooses_outside_the_top_k() -> None:
    logits = _logits()
    allowed = [set(row.topk(2).indices.tolist()) for row in logits]
    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        chosen = sample_top_k(logits, k=2, generator=generator).flatten().tolist()
        assert all(token in allowed[row] for row, token in enumerate(chosen))


def test_top_k_of_one_is_greedy() -> None:
    logits = _logits()
    assert torch.equal(sample_top_k(logits, k=1), greedy(logits))


def test_top_k_larger_than_the_vocabulary_is_clamped() -> None:
    logits = _logits()
    generator = torch.Generator().manual_seed(0)
    chosen = sample_top_k(logits, k=999, generator=generator)
    assert chosen.shape == (2, 1)
    assert int(chosen.max()) < logits.shape[-1]


def test_top_p_keeps_the_token_that_crosses_the_threshold() -> None:
    """A tiny p must still leave one candidate, never an empty set."""
    logits = _logits()
    generator = torch.Generator().manual_seed(0)
    chosen = sample_top_p(logits, p=1e-6, generator=generator)
    assert torch.equal(chosen, greedy(logits))


def test_top_p_of_one_can_reach_every_token() -> None:
    logits = torch.zeros(1, 6)
    generator = torch.Generator().manual_seed(0)
    seen = {int(sample_top_p(logits, p=1.0, generator=generator)) for _ in range(200)}
    assert seen == set(range(6))


def test_sampling_is_reproducible_from_a_seeded_generator() -> None:
    logits = _logits()
    first = sample_top_p(logits, p=0.9, generator=torch.Generator().manual_seed(7))
    second = sample_top_p(logits, p=0.9, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


def test_last_token_logits_selects_the_final_position() -> None:
    logits = torch.arange(24.0).reshape(2, 3, 4)
    assert torch.equal(last_token_logits(logits), logits[:, -1])
    assert torch.equal(last_token_logits(logits[:, -1]), logits[:, -1])


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: sample_top_k(_logits(), k=0), "k must be positive"),
        (lambda: sample_top_p(_logits(), p=0.0), r"p must lie in \(0, 1\]"),
        (lambda: sample_top_p(_logits(), p=0.5, temperature=0), "temperature must be positive"),
        (lambda: greedy(torch.zeros(2, 3, 4)), "use last_token_logits first"),
    ],
)
def test_sampling_rejects_invalid_arguments(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
