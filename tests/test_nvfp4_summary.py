"""Evidence completeness must not be confused with a passing accuracy result."""
import importlib.util
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'scripts/summarize_qwen3_5_nvfp4.py'
spec = importlib.util.spec_from_file_location('nvfp4_summary', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def gate(passed=False):
    return {'config': 'fla-delta:compile-reduce-overhead:nvfp4', 'complete': True,
            'pass': passed, 'bf16': {'prompt_names': list(module.ACCURACY_PROMPTS)},
            'quantized': {'prompt_names': list(module.ACCURACY_PROMPTS)}}


def test_missing_results_are_not_zero_error_passes(tmp_path):
    status = module.summarize(tmp_path)
    assert status['accuracy_status'] == 'not_evaluated'
    assert not status['experiment_complete']
    assert not status['mtp_greedy_equivalent']
    assert '이 실행에 증거 없음' in (tmp_path / 'report.md').read_text()


def test_flattened_release_evidence_is_found(tmp_path):
    write(tmp_path, 'native_smoke.json', [{'m': 1}])
    write(tmp_path, 'export_validation.json', {'reload_logits_identical': True})
    write(tmp_path, 'evaluation_lifecycle.json', {'status': 'stopped_by_user'})
    write(tmp_path, 'nvfp4_measurements.json', {'rows': [
        {'prompt': 'chat_long', 'method': 'mtp3', 'median_ms': 10., 'min_ms': 9.,
         'max_ms': 11., 'tokens_per_second': 100., 'token_agreement': .5}]})
    status = module.summarize(tmp_path)
    assert status['smoke_shapes'] == 1
    assert status['export']['reload_logits_identical']
    assert status['mtp_divergent_cases'] == 1
    assert not status['mtp_greedy_equivalent']


def test_incomplete_control_cannot_pass_even_when_gate_says_pass(tmp_path):
    g = gate(True)
    g['bf16']['prompt_names'].pop()
    write(tmp_path, 'accuracy.json', {'gate': [g]})
    status = module.summarize(tmp_path)
    assert status['accuracy_status'] == 'incomplete'
    assert not status['accuracy_passed']


def test_completed_negative_experiment_is_complete(tmp_path):
    write(tmp_path, 'lifecycle.json', {'status': 'finished'})
    int8_gate = gate(True)
    int8_gate['config'] = 'fla-delta:compile-reduce-overhead:int8'
    write(tmp_path, 'accuracy.json', {'gate': [gate(False), int8_gate]})
    write(tmp_path, 'linear_profile.json', {'rows': [
        {'activation_rows': n, 'linear_count': 256, 'aggregate_cuda_ms': 1.} for n in [1, 4]]})
    for precision in ['bf16', 'int8', 'nvfp4']:
        methods = ['plain'] + ([] if precision == 'bf16' else ['mtp1', 'mtp2', 'mtp3'])
        rows = [{'prompt': p, 'method': m, 'samples_ms': [10.] * 5, 'median_ms': 10.,
                 'min_ms': 10., 'max_ms': 10., 'tokens_per_second': 100., 'token_agreement': .5}
                for p in module.PROMPTS for m in methods]
        write(tmp_path, f'{precision}/measurements.json', {'rows': rows, 'prefill': [
            {'tokens': n, 'samples_ms': [10.] * 5, 'median_ms': 10.} for n in [512, 1024, 4096]]})
    status = module.summarize(tmp_path)
    assert status['experiment_complete']
    assert status['accuracy_status'] == 'failed'
    assert not status['accuracy_passed']
    assert not status['mtp_greedy_equivalent']
    # A duplicate cannot substitute for a missing prompt/method combination.
    path = tmp_path / 'nvfp4/measurements.json'
    result = json.loads(path.read_text())
    result['rows'][-1] = result['rows'][0]
    write(tmp_path, 'nvfp4/measurements.json', result)
    assert not module.summarize(tmp_path)['experiment_complete']


def test_partial_snapshot_recovers_completed_measurements(tmp_path):
    import sys
    sys.path.insert(0, str(PATH.parent))
    from run_qwen3_5_nvfp4_colab import snapshot_code, extract_results
    remote = tmp_path / 'remote'
    write(remote, 'nvfp4/measurements.json', {'rows': [{'method': 'plain'}]})
    (remote / 'nvfp4/measurements.tmp').write_text('{')
    archive = tmp_path / 'partial.zip'
    exec(compile(snapshot_code(remote, archive), '<snapshot>', 'exec'), {})
    local = tmp_path / 'local'
    extract_results(archive, local)
    assert json.loads((local / 'nvfp4/measurements.json').read_text())['rows']
    assert not (local / 'nvfp4/measurements.tmp').exists()


def test_partial_snapshot_rejects_path_traversal(tmp_path):
    import sys
    import zipfile
    import pytest
    sys.path.insert(0, str(PATH.parent))
    from run_qwen3_5_nvfp4_colab import extract_results
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../outside.json', '{}')
    with pytest.raises(ValueError, match='unsafe ZIP'):
        extract_results(archive, tmp_path / 'local')
