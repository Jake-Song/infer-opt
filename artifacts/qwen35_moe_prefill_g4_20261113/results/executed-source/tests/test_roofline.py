from types import SimpleNamespace

import pytest
import torch
from torch.autograd import DeviceType

from infer_opt.config import ModelConfig, RunConfig
from infer_opt.model import make_random_model
from infer_opt.profiling import transformer_flops
from infer_opt.roofline import (LINEARS, assess, attribute_events, event_owner, operation_annotations,
                                operation_costs, plot_results, profile_step, run_roofline, write_report)

CONFIG = ModelConfig(vocab_size=64, hidden_size=32, n_layers=2, n_heads=4,
                     n_kv_heads=2, intermediate_size=64, max_seq_len=32)


@pytest.mark.parametrize("tokens,context", [(8, 8), (1, 17), (4, 12)])
def test_matmul_costs_match_model_and_count_both_gqa_products(tokens, context):
    costs = operation_costs(CONFIG, tokens=tokens, context=context)
    coarse = transformer_flops(CONFIG, batch=1, new_tokens=tokens, context_len=context)
    assert sum(costs[name]['flops'] for name in LINEARS) == coarse['linear'] + coarse['lm_head']
    assert costs['attention']['flops'] == coarse['attention']
    assert costs['q_proj']['logical_bytes'] == 2 * (tokens*32 + 32*32 + tokens*32)*2
    assert costs['kv_read']['logical_bytes'] == 0
    assert costs['kv_write']['logical_bytes'] == 4*2*tokens*16*2


def test_embedding_reads_selected_rows_not_entire_table():
    costs = operation_costs(CONFIG, tokens=1, context=17)
    assert costs['embedding']['logical_bytes'] == 2*32*2 + 8
    assert costs['embedding']['logical_bytes'] < CONFIG.vocab_size*CONFIG.hidden_size*2


def test_attention_charges_expansion_and_context_growth():
    short = operation_costs(CONFIG, tokens=1, context=8)
    long = operation_costs(CONFIG, tokens=1, context=16)
    assert long['attention']['logical_bytes'] > short['attention']['logical_bytes']
    assert long['k_proj']['logical_bytes'] == short['k_proj']['logical_bytes']
    assert long['kv_write']['logical_bytes'] == short['kv_write']['logical_bytes']


def test_prediction_does_not_claim_measured_saturation_or_apply_tensor_roof_to_norm():
    costs = operation_costs(CONFIG, tokens=1, context=8)
    row = assess(costs['q_proj'], 10, 200, 1000)
    assert row['predicted_bound'] == 'memory-bound (모형)'
    assert 'roof에서 멂' in row['evidence']
    assert assess(costs['norm'], 1, 200, 1000)['roof_fraction'] is None
    assert assess(costs['kv_read'], 0, 200, 1000)['effective_gbps'] is None
    assert 'GPU 시간 없음' in assess(costs['q_proj'], 0, 200, 1000)['evidence']
    assert 'roof 초과' in assess(costs['q_proj'], 1e-9, 200, 1000)['evidence']


def cpu(name, parent=None, kernels=()):
    return SimpleNamespace(name=name, cpu_parent=parent, device_type=DeviceType.CPU,
                           self_cpu_time_total=1.0, kernels=list(kernels))


def gpu(us):
    return SimpleNamespace(device_type=DeviceType.CUDA, is_user_annotation=False,
                           time_range=SimpleNamespace(elapsed_us=lambda: us))


def test_exclusive_kernel_attribution_does_not_double_count_parent_spans():
    parent = cpu('infer_opt::qkv_proj')
    child = cpu('roofline::q_proj', parent)
    operator = cpu('aten::mm', child, [SimpleNamespace(name='gemm', duration=6.0)])
    events = [parent, child, operator, gpu(6.0)]
    result = attribute_events(events, steps=2)
    assert result['gpu_ms']['q_proj'] == pytest.approx(0.003)
    assert result['gpu_sum_ms'] == pytest.approx(0.003)
    assert result['accounting_passed']
    assert result['kernel_counts']['q_proj'] == 1
    assert event_owner(cpu('aten::contiguous', cpu('infer_opt::o_proj'))) == 'layout'
    # A missing correlation must fail the GPU accounting gate.
    assert not attribute_events([*events, gpu(4.0)], steps=2)['accounting_passed']
    assert not attribute_events([parent, child], steps=1)['accounting_passed']


def test_temporary_annotations_preserve_model_outputs_and_hooks():
    model = make_random_model(CONFIG, 'eager_sdpa', 0).eval()
    tokens = torch.tensor([[1,2,3,4]])
    with torch.inference_mode():
        expected = model(tokens)
        with operation_annotations(model):
            actual = model(tokens)
        assert torch.equal(expected, actual)
        with pytest.raises(RuntimeError, match='intentional'):
            with operation_annotations(model):
                raise RuntimeError('intentional')
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_cpu_profile_has_named_operations_but_no_gpu_performance_claim(tmp_path):
    model = make_random_model(CONFIG, 'eager_sdpa', 0).eval()
    device = torch.device('cpu')
    tokens = torch.tensor([[1,2,3,4]])
    cache = model.new_cache(1, device, torch.float32)
    with torch.inference_mode():
        result = profile_step(model, cache.reset, lambda: model.forward_prefill(tokens, cache),
                              device=device, iters=2, trace_path=tmp_path/'trace.json.gz')
    assert result['gpu_sum_ms'] == 0
    assert not result['accounting_passed']
    assert all(result['cpu_self_ms'][name] > 0 for name in LINEARS)
    assert (tmp_path/'trace.json.gz').stat().st_size > 0


def test_cuda_absence_saves_diagnostic_and_blocks_roofline(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    result = run_roofline(tmp_path, prompt_lengths=(4,), validation_decode_len=2, config=CONFIG)
    assert result['status'] == 'error'
    assert result['conditions'] == []
    assert 'CUDA GPU required' in result['error']
    assert (tmp_path/'roofline_results.json').exists()
    assert (tmp_path/'operations.csv').exists()


def test_compute_prediction_and_report_units(tmp_path):
    cost = dict(flops=2*4096**3, logical_bytes=3*4096**2*2, kind='gemm', shape='4096 cubed')
    row = assess(cost, 1.0, 200, 1000)
    assert row['predicted_bound'] == 'compute-bound (모형)'
    assert row['achieved_tflops'] == pytest.approx(137.438953472)
    assert row['effective_gbps'] == pytest.approx(100.663296)
    result = dict(status='passed', conditions=[dict(prompt_len=4096, phase='prefill',status='passed',
        operations=[dict(prompt_len=4096,phase='prefill',operation='q_proj',gpu_ms=1,gpu_share_pct=100,
                         **cost, **row)])])
    write_report(tmp_path, result)
    assert '137.438953472' in (tmp_path/'operations.csv').read_text()
    assert 'compute-bound (모형)' in (tmp_path/'report.md').read_text()


def test_plots_export_both_phases_and_zero_flop_operations(tmp_path):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    # Synthetic values for rendering tests only; never a GPU benchmark result.
    result = dict(status='passed', model=CONFIG.to_dict(),
                  calibration=dict(compute_tflops=200, copy_gbps=1000), conditions=[])
    for phase, tokens, context in [('prefill', 8, 8), ('decode', 1, 9)]:
        costs = operation_costs(CONFIG, tokens=tokens, context=context)
        rows = [dict(operation=name, gpu_ms=0.01 if cost['kind']!='view' else 0,
                     gpu_share_pct=6.25, **cost,
                     **assess(cost, 0.01 if cost['kind']!='view' else 0, 200, 1000))
                for name,cost in costs.items()]
        result['conditions'].append(dict(status='passed', prompt_len=8, phase=phase, tokens=tokens,
            context=context, operations=rows, timing={'wall_ms':{'median':1}}, profile={'gpu_sum_ms':0.16}))
    paths = plot_results(result, tmp_path)
    assert {p.name for p in paths} == {'roofline-8.png','operation-times-8.png','wall-vs-kernels.png'}
    assert all(p.read_bytes().startswith(b'\x89PNG') for p in paths)
    assert plot_results(dict(conditions=[]), tmp_path) == []
