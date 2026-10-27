"""Small deterministic checks of fit timing, recovery and preservation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_exploratory_fft_fits import fit_one, recovery


def write(path, value):
    path.write_text(json.dumps(value))


def test_inert_sv_reported_but_not_used_for_recovery():
    truth = np.arange(1., 14.)
    p = truth.copy()
    p[6] += 100
    result = recovery(p, truth)
    assert result['recovery_pass']
    assert result['signed_errors'][6] == 100
    p[7] *= 1.2
    assert not recovery(p, truth)['recovery_pass']


def test_time_checkpoint_does_not_use_future_result_and_cap_preserves_best(tmp_path):
    truth = np.arange(1., 14.)
    protocol = dict(truth=truth.tolist(), x0=(truth*1.2).tolist(), checkpoint_seconds=300,
                    fit_cap_seconds=600, maxiter=1000)
    now = [0.]
    calls = []
    def objective(p):
        calls.append(np.array(p))
        now[0] += 200
        return float(np.sum((p-truth)**2))
    def minimize(loss, x0, **kwargs):
        loss(np.array(x0))  # available at training t=200
        loss(truth)  # at 400, must not count at t=300
        loss(truth*1.1)  # at 600
        loss(truth)  # rejected by cap before evaluation
        pytest.fail('Cap should interrupt optimizer')
    result = fit_one('fft200', protocol, tmp_path, objective, np.asarray, minimize,
                     write, lambda *a, **k: None, lambda: now[0])
    assert result['warmup_seconds'] == 200
    assert result['evaluations'] == 3
    assert result['termination']['capped']
    assert result['checkpoints'][0]['seconds'] == 300
    assert result['checkpoints'][0]['params'] == protocol['x0']
    assert result['best']['params'] == truth.tolist()
    assert len(calls) == 4  # one warmup, three objective evaluations
    trace = [json.loads(line) for line in (tmp_path/'fft200-trace.jsonl').read_text().splitlines()]
    assert len(trace) == 3
    assert trace[0]['raw_params'] == protocol['x0']
    assert fit_one('fft200', protocol, tmp_path, None, None, None, write, None) == result


def test_interrupted_trace_is_not_overwritten(tmp_path):
    path = tmp_path/'fft199-trace.jsonl'
    path.write_text('retained evidence')
    with pytest.raises(RuntimeError, match='Interrupted fit retained'):
        fit_one('fft199', {}, tmp_path, None, None, None, write, None)
    assert path.read_text() == 'retained evidence'
