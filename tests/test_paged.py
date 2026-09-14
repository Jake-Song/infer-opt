import pytest

torch = pytest.importorskip("torch")

from infer_opt.cache import KVCache
from infer_opt.config import ModelConfig, RunConfig
from infer_opt.paged import BlockAllocator, PagedKVCache, blocks_for, contiguous_reserved_bytes
from infer_opt.runner import check_cached_decode

CONFIG = ModelConfig(
    vocab_size=64, hidden_size=32, n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=64, max_seq_len=32
)
CPU = torch.device("cpu")
SHAPE = {"n_layers": 2, "batch_size": 3, "n_kv_heads": 2, "max_seq_len": 32, "head_dim": 8}


def _pair(block_size: int = 4) -> tuple[KVCache, PagedKVCache]:
    contiguous = KVCache(**SHAPE, device=CPU, dtype=torch.float32)
    paged = PagedKVCache(**SHAPE, device=CPU, dtype=torch.float32, block_size=block_size)
    return contiguous, paged


def test_paged_reads_reproduce_the_contiguous_cache_exactly() -> None:
    """Written through the same calls, the two caches must return the same tensors.

    The chunk sizes deliberately straddle block boundaries, and the block table
    is not the identity, so a layout mistake cannot pass by accident.
    """
    torch.manual_seed(0)
    contiguous, paged = _pair(block_size=4)
    position = 0
    for chunk in (7, 1, 1, 3, 5):
        key = torch.randn(3, 2, chunk, 8)
        value = torch.randn(3, 2, chunk, 8)
        for layer in range(2):
            contiguous.write(layer, key + layer, value + layer, position)
            paged.write(layer, key + layer, value + layer, position)
        contiguous.advance(chunk)
        paged.advance(chunk)
        position += chunk
        for layer in range(2):
            want_key, want_value = contiguous.read(layer, position)
            got_key, got_value = paged.read(layer, position)
            assert torch.equal(got_key, want_key)
            assert torch.equal(got_value, want_value)


def test_the_block_table_is_not_the_identity() -> None:
    """Otherwise the parity test above would prove nothing about indirection."""
    _, paged = _pair(block_size=4)
    paged.write(0, torch.randn(3, 2, 9, 8), torch.randn(3, 2, 9, 8), 0)
    table = paged.block_table[:, : blocks_for(9, 4)]
    assert table.unique().numel() == table.numel()
    assert not torch.equal(table, torch.arange(table.numel()).reshape(table.shape))


def test_both_caches_expose_the_same_contract() -> None:
    contiguous, paged = _pair()
    for name in ("batch_size", "max_seq_len", "position", "reset", "validate_append", "write", "read", "advance"):
        assert hasattr(contiguous, name) and hasattr(paged, name)
    assert paged.batch_size == contiguous.batch_size == 3


def test_paged_cache_drives_a_full_cached_decode() -> None:
    result = check_cached_decode(
        CONFIG,
        RunConfig(batch_size=2, prompt_len=5, decode_len=3, device="cpu", cache="paged", block_size=4),
    )
    assert result["passed"], result


def test_advancing_tracks_each_sequence_length() -> None:
    _, paged = _pair()
    paged.advance(5)
    paged.advance(2)
    assert paged.position == 7
    assert paged.lengths.tolist() == [7, 7, 7]


def test_resetting_returns_every_block_to_the_pool() -> None:
    _, paged = _pair(block_size=4)
    paged.write(0, torch.randn(3, 2, 9, 8), torch.randn(3, 2, 9, 8), 0)
    paged.advance(9)
    assert paged.allocator.used_blocks > 0
    paged.reset()
    assert paged.allocator.used_blocks == 0
    assert paged.position == 0 and paged.lengths.sum() == 0


def test_waste_is_bounded_by_the_block_size_not_the_context_limit() -> None:
    """The whole memory argument for paging, as an assertion."""
    _, paged = _pair(block_size=4)
    paged.write(0, torch.randn(3, 2, 9, 8), torch.randn(3, 2, 9, 8), 0)
    paged.advance(9)
    # 9 tokens fill two blocks and three slots of a third: 3 slots wasted each.
    assert paged.wasted_bytes == 3 * 3 * paged.bytes_per_token
    assert paged.used_bytes == 3 * 9 * paged.bytes_per_token
    assert paged.reserved_bytes == paged.used_bytes + paged.wasted_bytes

    # Contiguous reserves all 32 positions per sequence; paged reserves the
    # three blocks (12 positions) the 9 tokens actually reached.
    reserved = contiguous_reserved_bytes(
        n_layers=2, batch=3, n_kv_heads=2, max_seq_len=32, head_dim=8, element_size=4
    )
    assert paged.reserved_bytes == reserved * 12 // 32
    assert paged.reserved_bytes < reserved / 2


def test_a_full_length_sequence_wastes_nothing() -> None:
    _, paged = _pair(block_size=4)
    paged.write(0, torch.randn(3, 2, 8, 8), torch.randn(3, 2, 8, 8), 0)
    paged.advance(8)
    assert paged.wasted_bytes == 0


def test_reading_positions_that_were_never_written_is_refused() -> None:
    _, paged = _pair()
    with pytest.raises(ValueError, match="have not been written yet"):
        paged.read(0, 4)


def test_writing_past_the_context_limit_is_refused() -> None:
    _, paged = _pair()
    paged.advance(30)
    with pytest.raises(ValueError, match="capacity exceeded"):
        paged.write(0, torch.randn(3, 2, 4, 8), torch.randn(3, 2, 4, 8), 30)


def test_layers_must_write_at_the_same_position() -> None:
    _, paged = _pair()
    with pytest.raises(ValueError, match="current cache position"):
        paged.write(0, torch.randn(3, 2, 2, 8), torch.randn(3, 2, 2, 8), 5)


def test_an_exhausted_pool_reports_what_it_had() -> None:
    paged = PagedKVCache(**SHAPE, device=CPU, dtype=torch.float32, block_size=4, num_blocks=3)
    with pytest.raises(ValueError, match="block pool exhausted"):
        paged.write(0, torch.randn(3, 2, 8, 8), torch.randn(3, 2, 8, 8), 0)


def test_the_allocator_recycles_freed_blocks() -> None:
    allocator = BlockAllocator(4)
    first = allocator.allocate(4)
    assert allocator.free_blocks == 0
    allocator.free(first[:2])
    assert allocator.free_blocks == 2
    assert len(allocator.allocate(2)) == 2
    with pytest.raises(ValueError, match="block pool exhausted"):
        allocator.allocate(1)


def test_the_allocator_refuses_blocks_from_elsewhere() -> None:
    with pytest.raises(ValueError, match="not from this pool"):
        BlockAllocator(2).free([7])


@pytest.mark.parametrize(("tokens", "size", "expected"), [(0, 4, 0), (1, 4, 1), (4, 4, 1), (5, 4, 2), (16, 16, 1)])
def test_block_count_rounds_up(tokens: int, size: int, expected: int) -> None:
    assert blocks_for(tokens, size) == expected
