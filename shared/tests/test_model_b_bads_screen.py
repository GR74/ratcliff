"""Tests for resource limits and honest comparisons in the BADS screen."""
import json

import numpy as np
import pytest

from scripts.model_b_bads_support import (
    FitDeadline, RecordedObjective, classify, paired_analysis, select_candidate,
)


def test_checkpoint_excludes_late_results_and_breaks_ties_by_completion():
    rows = [dict(eval=1, elapsed=2., loss=4., params=[1]),
            dict(eval=2, elapsed=3., loss=4., params=[2]),
            dict(eval=3, elapsed=4.01, loss=1., params=[3])]
    assert select_candidate(rows, 4)['eval'] == 1
    assert select_candidate(rows, 4.01)['eval'] == 3
    assert select_candidate(rows, 1) is None


def test_deadline_preserves_overrun_but_does_not_include_it_at_cutoff(tmp_path):
    clock = [0.]
    def evaluate(p):
        clock[0] += 6
        return p[0]
    runner = RecordedObjective(evaluate, lambda p: p, tmp_path/'trace.jsonl',
                               budget=10, clock=lambda: clock[0])
    assert runner([2]) == 2
    with pytest.raises(FitDeadline):
        runner([1])
    assert len(runner.rows) == 2 and runner.rows[-1]['elapsed'] == 12
    assert select_candidate(runner.rows, 10)['loss'] == 2
    with pytest.raises(FitDeadline):
        runner([0])
    assert runner.requests == 2


def test_invalid_evaluation_is_archived_and_not_selected(tmp_path):
    runner = RecordedObjective(lambda p: float('nan'), lambda p: np.maximum(p, 1),
                               tmp_path/'trace.jsonl')
    with pytest.raises(ValueError, match='Nonfinite'):
        runner([0])
    row = json.loads((tmp_path/'trace.jsonl').read_text())
    assert row['effective_params'] == [1] and not row['valid']
    assert select_candidate(runner.rows, 1e9) is None


def test_interrupted_fit_cannot_be_silently_restarted(tmp_path):
    path = tmp_path/'trace.jsonl'
    path.write_text('')
    with pytest.raises(FileExistsError):
        RecordedObjective(lambda p: 0, lambda p: p, path)


def score_rows(value=2., missing=0):
    return [dict(seed=seed, g2=value, sw1_pixels_by_condition=[value, value],
        metrics=dict(probability_max_absolute_error_pp=value,
            quantile_max_absolute_error_ms=value, missing_quantiles=missing))
        for seed in range(6001, 6021)]


def test_paired_analysis_uses_matching_seeds_and_detects_improvement():
    a, b = score_rows(1), list(reversed(score_rows(2)))
    result = paired_analysis(a, b)
    assert result['metrics']['g2']['interval_99'] == [-1, -1]
    assert classify({'recovery_screen_pass': True}, result)['classification'] == 'promising_for_replication'


def test_missing_quantiles_cannot_pass_the_screen():
    result = paired_analysis(score_rows(1, missing=1), score_rows(2))
    assert classify({'recovery_screen_pass': True}, result)['classification'] == 'inconclusive'


def test_predictive_regression_and_recovery_failure_are_both_retained():
    result = paired_analysis(score_rows(3), score_rows(2))
    decision = classify({'recovery_screen_pass': False}, result)
    assert decision['flags'] == ['recovery_target_missed', 'predictive_regression']


def test_unmatched_or_duplicate_seeds_are_rejected():
    rows = score_rows()
    rows[0]['seed'] = 6002
    with pytest.raises(ValueError, match='unique'):
        paired_analysis(rows, score_rows())


def test_early_completion_retains_candidate_without_restarting():
    rows = [dict(eval=1, elapsed=1, loss=2, params=[1]),
            dict(eval=2, elapsed=2, loss=1, params=[2])]
    assert select_candidate(rows, 8100) == rows[1]


def test_real_bads_unbounded_fixed_seed_and_trace(tmp_path):
    pybads = pytest.importorskip('pybads')
    x0 = np.array([2., 2.])
    def run(name):
        recorder = RecordedObjective(lambda p: float(np.sum((p-1)**2)), lambda p: p,
                                     tmp_path/name, budget=60)
        optimizer = pybads.BADS(recorder, x0=x0, lower_bounds=None, upper_bounds=None,
            plausible_lower_bounds=.5*x0, plausible_upper_bounds=1.5*x0,
            options=dict(random_seed=1701, uncertainty_handling=False,
                specify_target_noise=False, nonlinear_scaling=False, max_fun_evals=25,
                max_iter=10, display='off', show_tips=False))
        result = optimizer.optimize()
        return recorder.rows, result
    a, _ = run('a.jsonl')
    b, _ = run('b.jsonl')
    assert [r['params'] for r in a] == [r['params'] for r in b]
    assert len(a) > 1
    assert select_candidate(a, 60)['loss'] < a[0]['loss']
