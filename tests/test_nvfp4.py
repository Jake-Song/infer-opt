import importlib.util
from pathlib import Path

import pytest
import torch

from infer_opt.nvfp4 import Nvfp4Linear, require_sm120


def test_nvfp4_fake_preserves_leading_dimensions():
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        layer=Nvfp4Linear(torch.empty(32,16,dtype=torch.uint8),torch.empty(128,dtype=torch.uint8),
                          torch.ones(1),torch.ones(1),32,32)
        x=torch.empty(2,3,32,dtype=torch.bfloat16)
        assert layer(x).shape==(2,3,32)
        assert layer(x).dtype==torch.bfloat16


def test_nvfp4_unsupported_hardware_fails_explicitly(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='SM120'):
        require_sm120()


def test_gate_rejects_incomplete_precision():
    path=Path(__file__).resolve().parents[1]/'scripts/validate_qwen3_5_decode_modes.py'
    import sys
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('nvfp4_gate_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def row(quant,prompt):
        return dict(kernels='fla-delta',mode='compile',quant=quant,prompt=prompt,
                    decode_kl_vs_hf=.001,tf_mismatch_strong=0,tf_mismatch_near_tie=0,gen_first_mismatch=None)
    rows=[row('bf16','a'),row('bf16','b'),row('nvfp4','a')]
    assert not module.quantization_gate(rows,2)[0]['pass']
    rows.append(row('nvfp4','b'))
    assert module.quantization_gate(rows,2)[0]['pass']


def test_native_checkpoint_restores_packed_buffers_without_gpu(tmp_path):
    from tests.test_qwen3_5 import make_model
    from tests.test_quant import write_hf_config
    from infer_opt.nvfp4 import save_nvfp4
    from infer_opt.qwen3_5 import Qwen3_5ForCausalLM
    model=make_model().float()
    model.fuse_projections()
    mlp=model.model.language_model.layers[0].mlp
    old=mlp.down_proj
    mlp.down_proj=Nvfp4Linear(torch.zeros(old.out_features,old.in_features//2,dtype=torch.uint8),
                              torch.zeros(128,dtype=torch.uint8),torch.ones(1),torch.ones(1),
                              old.in_features,old.out_features)
    dest=tmp_path/'checkpoint'
    save_nvfp4(model,dest)
    write_hf_config(dest,model.config)
    loaded=Qwen3_5ForCausalLM.from_quantized_dir(dest)
    want=model.state_dict()
    got=loaded.state_dict()
    assert got.keys()==want.keys()
    for name in want:
        assert torch.equal(want[name],got[name]),name


def test_supervisor_is_valid_python():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_qwen3_5_nvfp4_colab import worker_code
    compile(worker_code('test',60),'<worker>','exec')
