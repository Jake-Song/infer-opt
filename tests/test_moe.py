import dataclasses
import pytest
import torch
from infer_opt.moe import Qwen3_5SparseMoeBlock, normalize_expert_state, quantize_moe_experts, Int8Experts
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM
from infer_opt.quant import quantize_linears, save_quantized, prepare_quantized
from test_qwen3_5 import tiny_config


def config():
    return dataclasses.replace(tiny_config(), num_experts=4, num_experts_per_tok=2,
                               moe_intermediate_size=32, shared_expert_intermediate_size=32,
                               mtp_num_hidden_layers=1)


def model():
    torch.manual_seed(17)
    m = Qwen3_5ForCausalLM(config(), mtp=True).float().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, .03)
    return m


def test_routing_matches_explicit_sum():
    m = model().model.language_model.layers[0].mlp
    x = torch.randn(1, 5, 64)
    flat = x[0]
    p = m.gate(flat).softmax(-1)
    weights, ids = p.topk(2, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    want = m.shared_expert(flat) * m.shared_expert_gate(flat).sigmoid()
    for t in range(5):
        for j in range(2):
            e = ids[t, j]
            gate, up = torch.nn.functional.linear(flat[t], m.experts.gate_up_proj[e]).chunk(2)
            want[t] += weights[t, j] * torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, m.experts.down_proj[e])
    torch.testing.assert_close(m(x)[0], want)


def test_expert_layout():
    cfg = config()
    state = {}
    for i in range(4):
        for p, shape in [('gate', (32,64)), ('up',(32,64)), ('down',(64,32))]:
            state[f'mtp.layers.0.mlp.experts.{i}.{p}_proj.weight'] = torch.randn(*shape)
    gate = state['mtp.layers.0.mlp.experts.0.gate_proj.weight'].clone()
    normalize_expert_state(state, cfg)
    assert len(state) == 2
    torch.testing.assert_close(state['mtp.layers.0.mlp.experts.gate_up_proj'][0,:32], gate)
    with pytest.raises(ValueError, match='incomplete'):
        normalize_expert_state({'x.experts.0.gate_proj.weight': gate}, cfg)


def test_int8_roundtrip(tmp_path):
    import json
    m = model()
    quantize_linears(m, group_size=16, min_out_features=4)
    names = quantize_moe_experts(m, 16)
    assert len(names) == 5
    assert not m.mtp.layers[0].mlp.experts.gate_up_scale.requires_grad
    assert not m.mtp.layers[0].mlp.experts.down_scale.requires_grad
    assert isinstance(m.mtp.layers[0].mlp.gate, torch.nn.Linear)
    save_quantized(m, tmp_path)
    clone = model()
    prepare_quantized(clone, json.loads((tmp_path/'quantization.json').read_text()))
    clone.load_state_dict(m.state_dict(), assign=True)
    x = torch.randn(1, 5, 64)
    torch.testing.assert_close(m.mtp.layers[0].mlp(x), clone.mtp.layers[0].mlp(x), rtol=0, atol=0)
    assert isinstance(clone.mtp.layers[0].mlp.experts, Int8Experts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('tokens', [1, 2, 3, 4, 5])
def test_selected_int8_kernel(tokens):
    from infer_opt.moe_kernels import selected_int8_experts
    m = model().bfloat16()
    quantize_moe_experts(m, 16)
    e = m.mtp.layers[0].mlp.experts
    x = torch.randn(tokens, 64, dtype=torch.bfloat16)
    ids = torch.arange(tokens * 2).reshape(tokens, 2) % 4
    weights = torch.full((tokens, 2), .5, dtype=torch.bfloat16)
    want = e(x, ids, weights)
    e.cuda()
    got = selected_int8_experts(x.cuda(), ids.cuda(), weights.cuda(), e.gate_up_proj,
                               e.gate_up_scale, e.down_proj, e.down_scale)
    torch.testing.assert_close(got.cpu(), want, atol=2e-4, rtol=.03)


def test_streaming_conversion_with_separate_mtp_experts(tmp_path):
    import json
    import importlib.util
    from pathlib import Path
    from safetensors.torch import save_file
    from test_quant import write_hf_config
    spec = importlib.util.spec_from_file_location('streaming', Path(__file__).resolve().parents[1] / 'scripts/quantize_qwen3_5_moe_streaming.py')
    streaming = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(streaming)
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    original = model().bfloat16()
    original.config = dataclasses.replace(original.config, tie_word_embeddings=False)
    original.lm_head.weight = torch.nn.Parameter(original.lm_head.weight.detach().clone())
    write_hf_config(source, original.config)
    raw = json.loads((source / "config.json").read_text())
    del raw["text_config"]["intermediate_size"]  # real MoE configs omit the dense dimension
    (source / "config.json").write_text(json.dumps(raw))
    state = {n: t.clone() for n, t in original.state_dict().items()}
    prefix = 'mtp.layers.0.mlp.experts'
    gu, down = state.pop(prefix+'.gate_up_proj'), state.pop(prefix+'.down_proj')
    for i in range(4):
        state[f'{prefix}.{i}.gate_proj.weight'] = gu[i,:32].clone()
        state[f'{prefix}.{i}.up_proj.weight'] = gu[i,32:].clone()
        state[f'{prefix}.{i}.down_proj.weight'] = down[i].clone()
    save_file(state, str(source/'model.safetensors'))
    (source/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{n:'model.safetensors' for n in state}}))
    files=streaming.convert(source, output, group=16, shard_bytes=20000)
    assert len(files)>1
    loaded=Qwen3_5ForCausalLM.from_quantized_dir(output)
    quantize_linears(original,group_size=16)
    quantize_moe_experts(original,16)
    for n,t in original.state_dict().items():
        assert torch.equal(t,loaded.state_dict()[n]), n
    with pytest.raises(ValueError,match='empty'):
        streaming.convert(source,output,16)


@pytest.mark.parametrize('verification', ['sequential', 'consistent'])
@pytest.mark.parametrize('k',[1,2,3,4])
def test_moe_speculative_parity(k, verification):
    from infer_opt.speculative import MTPSpeculativeDecoder
    m=model().double()
    quantize_moe_experts(m,16)
    ids=torch.tensor([[1,3,5,7]])
    with torch.no_grad():
        want=m.generate_greedy(ids,12)
        decoder=MTPSpeculativeDecoder(m,32,k=k,verification=verification)
        assert torch.equal(decoder.generate(ids,12),want)
        assert torch.equal(decoder.generate(ids,12),want)


@pytest.mark.parametrize('accepted',[1,2,5])
@pytest.mark.parametrize('verification',['sequential','consistent'])
def test_moe_verify_commit_matches_decode(accepted, verification):
    m=model().double()
    quantize_moe_experts(m,16)
    ids=torch.tensor([[1,3,5,7,9,11,13,15,17,19]])
    with torch.no_grad():
        baseline=m.new_cache(1,20); candidate=m.new_cache(1,20)
        candidate.allocate_spec_buffers(5)
        m(ids[:,:4],baseline);m(ids[:,:4],candidate)
        want=torch.cat([m.decode_step(ids[:,4+i:5+i],baseline) for i in range(accepted)],1)
        verify=getattr(m.model.language_model,'forward_verify_'+verification)
        hidden=verify(ids[:,4:9],candidate)
        got=torch.cat([m.lm_head(hidden[:,i:i+1]) for i in range(accepted)],1)
        torch.testing.assert_close(got,want,atol=1e-10,rtol=1e-10)
        candidate.commit(torch.tensor([accepted]))
        torch.testing.assert_close(candidate.position,baseline.position)
        for a,b in zip(candidate.conv_states+candidate.recurrent_states,baseline.conv_states+baseline.recurrent_states):
            if a is not None: torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(m.decode_step(ids[:,9:],candidate),m.decode_step(ids[:,9:],baseline),atol=1e-10,rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('mode',['cudagraph','compile'])
@pytest.mark.parametrize('verification',['sequential','consistent'])
def test_gpu_moe_speculative_parity(mode,verification):
    from infer_opt.qwen3_5 import FixedShapeDecoder
    from infer_opt.speculative import MTPSpeculativeDecoder
    m=model().bfloat16().cuda()
    quantize_moe_experts(m,16)
    ids=torch.tensor([[1,3,5,7]],device='cuda')
    with torch.no_grad():
        base=FixedShapeDecoder(m,m.new_cache(1,64),'eager')
        token=m(ids,base.cache)[:,-1].argmax(-1,keepdim=True)
        want=[token]
        for _ in range(15):
            token=base.step(token)[:,-1].argmax(-1,keepdim=True);want.append(token)
        want=torch.cat(want,1)
        spec=MTPSpeculativeDecoder(m,64,k=2,mode=mode,verification=verification,allow_inexact=True)
        got=spec.generate(ids,16)[:,ids.shape[1]:]
        assert torch.equal(got,want)


def test_moe_rejects_unsupported_verify_width():
    from infer_opt.speculative import MTPSpeculativeDecoder
    with pytest.raises(ValueError,match='k=1 through k=4'):
        MTPSpeculativeDecoder(model(),64,k=5)


def test_compiled_moe_requires_diagnostic_opt_in():
    from infer_opt.speculative import MTPSpeculativeDecoder
    with pytest.raises(ValueError, match='failed full-model greedy parity'):
        MTPSpeculativeDecoder(model(),64,mode='compile')
