"""Small HF-model checks; no checkpoint download or 27B allocation required."""

import ast
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_hf_notebook_is_standalone_and_valid_python():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(ROOT / "notebooks/qwen3_5_27b_hf_colab.ipynb", as_version=4)
    nbformat.validate(nb)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    for source in code:
        ast.parse(source)
    assert (ROOT / "scripts/qwen3_5_hf_graph.py").read_text() in code
    assert "infer_opt" not in "\n".join(code)
    assert all(not c.outputs and c.execution_count is None for c in nb.cells if c.cell_type == "code")


@pytest.fixture
def hf_decoder():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.20.2":
        pytest.skip("The standalone notebook pins transformers==5.20.2")
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from scripts.qwen3_5_hf_graph import HFGraphDecoder

    torch.manual_seed(42)
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 1]},
    )
    config._attn_implementation = "sdpa"
    model = Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
    return HFGraphDecoder(model, max_cache_len=32)


def test_static_decode_matches_hf_dynamic_after_reset(hf_decoder):
    from scripts.qwen3_5_hf_graph import validate_decoder

    for length in (9, 17, 3):
        ids = torch.randint(1, 128, (1, length))
        rows = validate_decoder(hf_decoder, ids, steps=5, modes=("eager",), max_mean_kl=1e-6)
        assert all(row["passed"] for row in rows), rows
        if hf_decoder.addresses is None:
            hf_decoder.addresses = hf_decoder.check_addresses()
        hf_decoder.check_addresses()


def test_fullgraph_trace_preserves_mutable_cache(hf_decoder):
    from scripts.qwen3_5_hf_graph import validate_decoder

    ids = torch.randint(1, 128, (1, 9))
    hf_decoder.prefill(ids)
    hf_decoder.addresses = hf_decoder.check_addresses()
    # Exercise Dynamo's fullgraph constraints on CPU, without claiming CUDA capture coverage.
    hf_decoder.compiled = torch.compile(hf_decoder._decode, backend="eager", fullgraph=True, dynamic=False)
    rows = validate_decoder(hf_decoder, ids, steps=5, modes=("compiled",), max_mean_kl=1e-6)
    assert all(row["passed"] for row in rows), rows
    torch._dynamo.reset()


def test_cache_capacity_and_prepare_guards(hf_decoder):
    ids = torch.ones((1, 31), dtype=torch.long)
    hf_decoder.prefill(ids)
    hf_decoder.step(mode="eager")
    with pytest.raises(ValueError, match="capacity"):
        hf_decoder.step(mode="eager")
    with pytest.raises(RuntimeError, match="CUDA"):
        hf_decoder.prepare(ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph needs an actual GPU")
def test_cuda_graph_replay_and_reset(hf_decoder):
    from scripts.qwen3_5_hf_graph import HFGraphDecoder, validate_decoder

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    decoder = HFGraphDecoder(hf_decoder.model.to(device="cuda", dtype=dtype), max_cache_len=32)
    ids = torch.randint(1, 128, (1, 9), device="cuda")
    setup = decoder.prepare(ids)
    assert setup["explicit_cuda_graph"]
    for length in (9, 13):
        ids = torch.randint(1, 128, (1, length), device="cuda")
        rows = validate_decoder(decoder, ids, steps=5)
        assert all(row["passed"] for row in rows), rows
