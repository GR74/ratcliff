"""Budget/provenance safeguards for the new reference experiment."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module(name):
    path = Path(__file__).resolve().parents[2] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_reference_recovery_includes_sv():
    runner = module('run_fft_reference_recovery')
    truth = np.array(runner.TRUTH, float)
    fitted = truth.copy(); fitted[6] *= 2
    result = runner.recovery(fitted, truth)
    assert result['worst_relative_error'] == 1
    assert result['mean_relative_error'] == pytest.approx(1/13)


def test_late_evaluation_cannot_improve_earlier_checkpoint(tmp_path):
    runner = module('run_fft_reference_recovery')
    now = [0.]
    calls = [0]
    def loss(p):
        calls[0] += 1
        now[0] += 11
        return 10. if calls[0] == 1 else 1.
    def minimize(fn, x0, **kwargs):
        fn(x0)
        fn(x0)  # cap must be enforced before this objective is called
        raise AssertionError('cap not enforced')
    protocol = dict(x0=runner.TRUTH, truth=runner.TRUTH, checkpoint_seconds=[10],
                    fit_cap_seconds=10, maxiter=1000)
    result = runner.fit_one('test', protocol, tmp_path, loss, np.asarray,
                           lambda *a, **kw: None, clock=lambda: now[0], minimize=minimize)
    assert result['termination']['capped']
    assert calls[0] == 2  # warmup plus one completed evaluation
    assert result['checkpoints'][0]['loss'] == 10
    assert result['best']['loss'] == 1


def test_preparation_rejects_ambiguous_anchors():
    prepare = module('prepare_fft_reference')
    with pytest.raises(ValueError):
        prepare.replace_once('same same', 'same', 'new')
