"""Distribution preservation and elementwise tolerance are distinct contracts."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('compiled_quality', SCRIPTS / 'validate_compiled_qwen3_5_moe_quality.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def test_equal_distribution_can_fail_elementwise_logit_tolerance():
    eager = np.array([[0., 1., 2.], [2., -1., 0.]], dtype=np.float32)
    result = quality.eager_equivalence(eager, eager + 1.)
    assert result['mean_kl'] == pytest.approx(0, abs=1e-7)
    assert result['top1_agreement'] == 1
    assert result['allclose_per_step'] == [False, False]
    assert result['allclose_passed'] is False
    assert result['max_abs_logit_error'] == 1


def test_allclose_is_checked_per_position():
    eager = np.array([[1., 2., 3.], [1., 2., 3.]], dtype=np.float32)
    actual = eager.copy()
    actual[1, 0] += 0.5
    result = quality.eager_equivalence(eager, actual)
    assert result['allclose_per_step'] == [True, False]
    assert result['allclose_positions'] == 1
