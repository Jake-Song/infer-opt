"""Guard the scope ablation and the reference-logit provenance."""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from infer_opt.nvfp4 import Nvfp4Linear, nvfp4_targets, quantize_nvfp4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from validate_qwen3_5_decode_modes import hf_cached_reference


class ScopeModel(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Module()
        layer.mlp = nn.ModuleDict({'gate_up_proj': nn.Linear(256, 512, bias=False),
                                  'down_proj': nn.Linear(256, 256, bias=False)})
        layer.linear_attn = nn.ModuleDict({'in_proj_qkvz': nn.Linear(256, 512, bias=False),
                                           'out_proj': nn.Linear(256, 256, bias=False)})
        self.layers = nn.ModuleList([layer])
        self.mtp = nn.ModuleDict({'mlp': nn.ModuleDict({'down_proj': nn.Linear(256, 256)})})
        self.lm_head = nn.Linear(256, 512, bias=False)

    def forward(self, ids):
        return ids


@pytest.mark.parametrize('list_config', [False, True])
def test_mlp_scope_matches_calibration_and_packing(monkeypatch, list_config):
    nv = importlib.import_module('infer_opt.nvfp4')
    monkeypatch.setattr(nv, 'require_sm120', lambda: None)
    names = ['modelopt', 'modelopt.torch', 'modelopt.torch.quantization',
             'modelopt.torch.export', 'modelopt.torch.export.quant_utils', 'flashinfer']
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(modules[parent], child, module)
    mtq = modules['modelopt.torch.quantization']
    mtq.NVFP4_DEFAULT_CFG = {'quant_cfg': [] if list_config else {}}
    observed = {}

    def calibrate(model, config, loop):
        observed.update(config)
        loop(model)

    mtq.quantize = calibrate
    utils = modules['modelopt.torch.export.quant_utils']
    utils.get_activation_scaling_factor = lambda model: torch.tensor(.001)
    utils.get_weight_scaling_factor_2 = lambda model: torch.tensor(.002)
    fi = modules['flashinfer']
    fi.SfLayout = SimpleNamespace(layout_128x4=0)
    fi.nvfp4_quantize = lambda w, *a, **kw: (torch.zeros(w.shape[0], w.shape[1]//2, dtype=torch.uint8),
                                             torch.zeros(w.numel()//16, dtype=torch.uint8))
    model = ScopeModel()
    expected = {'layers.0.mlp.gate_up_proj', 'layers.0.mlp.down_proj'}
    assert set(nvfp4_targets(model, 'mlp')) == expected
    assert len(nvfp4_targets(model, 'all')) == 4
    report = quantize_nvfp4(model, [torch.ones(1, 3)], scope='mlp')
    assert set(report['modules']) == expected
    assert all(isinstance(model.get_submodule(n), Nvfp4Linear) for n in expected)
    assert isinstance(model.layers[0].linear_attn.out_proj, nn.Linear)
    entries = observed['quant_cfg']
    if list_config:
        disabled = {e['quantizer_name'] for e in entries if e.get('enable') is False}
    else:
        disabled = {n for n, c in entries.items() if c.get('enable') is False}
    for prefix in ['layers.0.linear_attn.in_proj_qkvz', 'layers.0.linear_attn.out_proj', 'lm_head',
                   'mtp.mlp.down_proj']:
        assert prefix + '.weight_quantizer' in disabled
        assert prefix + '.input_quantizer' in disabled
    assert not any(n + '.input_quantizer' in disabled for n in expected)
    with pytest.raises(ValueError, match='scope'):
        nvfp4_targets(model, 'invalid')


class FakeHF:
    """Full forwards have a different margin, making margin provenance observable."""
    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        assert use_cache
        prefix = input_ids if past_key_values is None else torch.cat([past_key_values, input_ids], 1)
        winner = prefix.sum(1) % 7
        logits = torch.zeros(1, input_ids.shape[1], 7)
        logits[:, -1, winner.item()] = 2 + input_ids.shape[1]
        return SimpleNamespace(logits=logits, past_key_values=prefix)


def test_reference_tokens_and_margins_use_the_same_cached_forwards():
    result = hf_cached_reference(FakeHF(), torch.tensor([[1, 2, 3]]), 2, 3)
    assert result['new_ref'].tolist() == [[6, 5, 3]]
    assert result['margins'].tolist() == [5, 3, 3]
    assert result['hf_decode_logits'].argmax(-1).tolist() == [[3, 6]]
    assert result['split'] == 1


def test_single_token_reference_still_evaluates_generation():
    result = hf_cached_reference(FakeHF(), torch.tensor([[2]]), 32, 3)
    assert result['hf_decode_logits'] is None
    assert result['new_ref'].tolist() == [[2, 4, 1]]
    assert result['margins'].tolist() == [3, 3, 3]


def test_ablation_pass_requires_protocol_scope_and_complete_rows(tmp_path):
    import json
    from summarize_qwen3_5_nvfp4_mlp import summarize
    prompts = ['short', 'hello', 'korean', 'math', 'chat', 'chat_long', 'prose_1k', 'code_1k', 'code_1_5k']
    stats = {'prompt_names': prompts}
    gates = [{'config': 'fla-delta:compile-reduce-overhead:' + q, 'complete': True,
              'pass': q == 'nvfp4-mlp', 'bf16': stats, 'quantized': stats, 'kl_ratio': 1.}
             for q in ['nvfp4', 'nvfp4-mlp']]
    accuracy = {'reference_protocol': 'hf-cached-greedy-v2', 'gate': gates, 'rows': [
        {'quant': q, 'prompt': p, 'decode_kl_vs_hf': .001, 'tf_mismatch_strong': 0,
         'tf_mismatch_near_tie': 0, 'gen_first_mismatch': None}
        for q in ['bf16', 'nvfp4', 'nvfp4-mlp'] for p in prompts]}

    def write(name, value):
        file = tmp_path / name
        file.parent.mkdir(exist_ok=True, parents=True)
        file.write_text(json.dumps(value))

    write('accuracy.json', accuracy)
    write('lifecycle.json', {'status': 'finished'})
    write('export/export_validation.json', {'reload_logits_identical': True})
    scope = {'native_mlp_linears': 128, 'native_attention_linears': 0,
             'bf16_attention_linears': 128, 'calibration_token_hashes_identical': True}
    write('scope_validation.json', scope)
    assert summarize(tmp_path)['mlp_accuracy_passed']
    accuracy['reference_protocol'] = 'old-full-sequence-reference'
    write('accuracy.json', accuracy)
    assert not summarize(tmp_path)['mlp_accuracy_passed']
    accuracy['reference_protocol'] = 'hf-cached-greedy-v2'
    accuracy['rows'].pop()
    write('accuracy.json', accuracy)
    assert not summarize(tmp_path)['experiment_complete']
