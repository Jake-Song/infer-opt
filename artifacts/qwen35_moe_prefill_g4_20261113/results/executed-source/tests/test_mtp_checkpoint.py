import dataclasses
import json
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

from infer_opt.quant import Int8Linear, quantize_linears, save_quantized
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM
from infer_opt.speculative import MTPSpeculativeDecoder
from scripts.publish_qwen3_5_mtp import HEAD_FILE, merge_specs, quantize_head, sha256
from test_quant import write_hf_config
from test_speculative import make_model


@pytest.mark.parametrize("quantized_head", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_mtp_checkpoint_round_trip(tmp_path, quantized_head, tied):
    source = make_model()
    model = Qwen3_5ForCausalLM(dataclasses.replace(source.config, tie_word_embeddings=tied), mtp=True).double().eval()
    model.load_state_dict(source.state_dict())
    quantize_linears(model, group_size=16, min_out_features=8, exclude=() if quantized_head else ("mtp.",))
    save_quantized(model, tmp_path, shard_bytes=20_000)
    write_hf_config(tmp_path, model.config)
    loaded = Qwen3_5ForCausalLM.from_quantized_dir(tmp_path)
    assert loaded.mtp is not None
    assert isinstance(loaded.mtp.fc, Int8Linear) == quantized_head
    assert (loaded.lm_head.weight is loaded.model.language_model.embed_tokens.weight) == tied
    for name, tensor in model.state_dict().items():
        got = loaded.state_dict()[name]
        assert tensor.dtype == got.dtype and torch.equal(tensor, got), name
    ids = torch.randint(0, model.config.vocab_size, (1, 9))
    with torch.no_grad():
        assert torch.equal(model(ids), loaded(ids))
        for k in (1, 3):
            want = MTPSpeculativeDecoder(model, 25, k=k).generate(ids, 12)
            got = MTPSpeculativeDecoder(loaded, 25, k=k).generate(ids, 12)
            assert torch.equal(got, want)
    cast = Qwen3_5ForCausalLM.from_quantized_dir(tmp_path, dtype=torch.float32)
    assert cast.mtp.norm.weight.dtype == torch.float32
    if quantized_head:
        assert cast.mtp.fc.weight.dtype == torch.int8 and cast.mtp.fc.scale.dtype == torch.float32


def test_old_checkpoint_with_mtp_config_keeps_target_only(tmp_path):
    model = make_model().float()
    model.mtp = None
    quantize_linears(model, group_size=16, min_out_features=8)
    save_quantized(model, tmp_path)
    write_hf_config(tmp_path, model.config)
    spec = json.loads((tmp_path / "quantization.json").read_text())
    del spec["mtp"]
    (tmp_path / "quantization.json").write_text(json.dumps(spec))
    loaded = Qwen3_5ForCausalLM.from_quantized_dir(tmp_path)
    assert loaded.config.mtp_num_hidden_layers == 1 and loaded.mtp is None


def test_append_head_preserves_target_shards_and_matches_attached_model(tmp_path):
    source = make_model().float()
    state = {f"mtp.{k}": v for k, v in source.mtp.state_dict().items()}
    source.mtp = None
    quantize_linears(source, group_size=16, min_out_features=8)
    target_dir, head_dir = tmp_path / "target", tmp_path / "head"
    files = save_quantized(source, target_dir, shard_bytes=20_000)
    write_hf_config(target_dir, source.config)
    before = {f.name: sha256(f) for f in files}
    head_spec, checked = quantize_head(source.config, state, head_dir, group_size=16, min_out_features=8)
    assert checked["round_trip_equal"]
    base = json.loads((target_dir / "quantization.json").read_text())
    merged = merge_specs(base, head_spec, group_size=16)
    assert base["mtp"] is False  # merging must not mutate the original metadata
    (target_dir / "quantization.json").write_text(json.dumps(merged))
    shutil.copyfile(head_dir / HEAD_FILE, target_dir / HEAD_FILE)
    loaded = Qwen3_5ForCausalLM.from_quantized_dir(target_dir)
    source.attach_mtp(state)
    quantize_linears(source, group_size=16, min_out_features=8)
    assert {f.name: sha256(f) for f in files} == before
    assert loaded.state_dict().keys() == source.state_dict().keys()
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name]), name
    with pytest.raises(ValueError, match="already contains MTP"):
        merge_specs(merged, head_spec, group_size=16)
    missing = load_file(str(target_dir / HEAD_FILE))
    missing.pop("mtp.norm.weight")
    save_file(missing, str(target_dir / HEAD_FILE))
    with pytest.raises(RuntimeError, match="Missing key"):
        Qwen3_5ForCausalLM.from_quantized_dir(target_dir)


def test_loader_rejects_duplicate_tensor_shards(tmp_path):
    model = make_model().float()
    quantize_linears(model, group_size=16, min_out_features=8)
    files = save_quantized(model, tmp_path)
    write_hf_config(tmp_path, model.config)
    shutil.copyfile(files[0], tmp_path / "model-int8-duplicate.safetensors")
    with pytest.raises(ValueError, match="duplicate checkpoint tensor"):
        Qwen3_5ForCausalLM.from_quantized_dir(tmp_path)
