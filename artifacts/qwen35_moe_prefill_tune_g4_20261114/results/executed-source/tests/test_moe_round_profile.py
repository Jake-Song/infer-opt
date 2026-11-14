"""Profiler accounting must distinguish expert work and overlapping CPU waits."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/profile_qwen3_5_moe_round.py'
spec = importlib.util.spec_from_file_location('moe_round_profile', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_expert_kernels_are_not_lumped_into_other():
    assert module.category('_project') == 'routed_experts'
    assert module.category('_combine') == 'routed_experts'
    assert module.category('_w8a16_group_gemv_rows_kernel') == 'int8_multirow'
    assert module.category('triton_mm_12') == 'bf16_matmul'
    assert module.category('void at::native::sbtopk::gatherTopK<float>') == 'routing_topk_sort'
    assert module.category('void at::native::bitonicSortKVInPlace<float>') == 'routing_topk_sort'


def test_trace_accounting_keeps_overlapping_wait_separate():
    trace = {'traceEvents': [
        {'ph': 'X', 'cat': 'kernel', 'name': '_project', 'ts': 0., 'dur': 100.},
        {'ph': 'X', 'cat': 'kernel', 'name': '_combine', 'ts': 80., 'dur': 40.},
        {'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaStreamSynchronize', 'ts': 5., 'dur': 115.},
        {'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaGraphLaunch', 'ts': 0., 'dur': 2.},
        {'ph': 'X', 'cat': 'cpu_op', 'name': 'large unrelated CPU op', 'ts': 0., 'dur': 9000.},
    ]}
    result = module.summarize_trace(trace, 2)
    assert result['gpu_kernel_sum_ms_per_call'] == pytest.approx(.070)
    assert result['gpu_busy_union_ms_per_call'] == pytest.approx(.060)
    assert result['cpu_sync_api_ms_per_call'] == pytest.approx(.0575)
    assert result['gpu_category_ms_per_call']['routed_experts'] == pytest.approx(.070)
    assert result['cuda_graph_launches'] == 1


def test_union_handles_empty_and_nested_intervals():
    assert module.union_us([]) == 0
    assert module.union_us([(20, 30), (1, 10), (2, 4), (9, 21)]) == 29
