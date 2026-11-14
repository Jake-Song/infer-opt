"""Numerical and replay alignment contracts, without model downloads or CUDA."""
import importlib.util
import io
from pathlib import Path
import struct

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/validate_llamacpp_qwen3_5_quality.py'
spec = importlib.util.spec_from_file_location('llamacpp_quality', SOURCE)
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def test_distribution_metrics_separates_near_ties_from_strong_mismatches():
    want = np.array([[3., 0., -1.], [0.1, 0., -1.], [0., 4., -1.]], dtype=np.float32)
    got = np.array([[3., 0., -1.], [0., 0.1, -1.], [4., 0., -1.]], dtype=np.float32)
    result = quality.distribution_metrics(want, got)
    assert result['mismatch_steps'] == [1, 2]
    assert result['strong_mismatches'] == 1
    assert result['top1_agreement'] == pytest.approx(1 / 3)
    p = np.exp(want.astype(np.float64) - want.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    q = np.exp(got.astype(np.float64) - got.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    assert result['mean_kl'] == pytest.approx((p * np.log(p / q)).sum(1).mean(), abs=1e-6)
    assert result['per_step'][0]['phase'] == 'prefill_last'
    assert result['per_step'][1]['phase'] == 'decode'


def test_shifted_logits_preserve_distribution():
    want = np.array([[1., 2., 3.], [-2., 4., 0.]], dtype=np.float32)
    result = quality.distribution_metrics(want, want + 10)
    assert result['mean_kl'] == pytest.approx(0, abs=1e-7)
    assert result['top1_agreement'] == 1
    assert result['max_abs_logit_error'] == 10


def test_replay_uses_prompt_then_all_but_last_hf_prediction(tmp_path):
    reference = {'vocab_size': 8, 'cases': [
        {'name': 'example', 'prompt_ids': [1, 2, 3], 'generated_ids': [4, 5, 6]},
    ]}
    path = tmp_path / 'replay.bin'
    quality.write_replay(path, reference, {'token-a': 1, 'token-b': 2})
    data = io.BytesIO(path.read_bytes())
    assert data.read(8) == b'IOQ8TF01'
    assert struct.unpack('<III', data.read(12)) == (8, 2, 1)
    for index, token in [(1, 'token-a'), (2, 'token-b')]:
        assert quality.read_u32(data) == index
        assert quality.read_string(data) == token
    assert quality.read_string(data) == 'example'
    assert (quality.read_u32(data), quality.read_u32(data)) == (3, 3)
    assert struct.unpack('<5i', data.read()) == (1, 2, 3, 4, 5)


@pytest.mark.parametrize('want,got', [
    ([[1., 2.]], [[float('nan'), 0.]]),
    ([[1., 2.]], [[1., 2.], [3., 4.]]),
    ([], []),
])
def test_invalid_logits_fail_instead_of_reporting_quality(want, got):
    with pytest.raises(ValueError):
        quality.distribution_metrics(want, got)


def test_truncated_native_output_is_rejected():
    with pytest.raises(ValueError, match='truncated'):
        quality.read_u32(io.BytesIO(b'\0\0'))
