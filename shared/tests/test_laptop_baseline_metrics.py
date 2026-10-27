"""Guard against misleading recovery and sparse-response baseline summaries."""
import numpy as np

from scripts.baseline_model_b_laptop import median_interval, observable_metrics, recovery
from scripts.model_b_baseline_observer import sliced_wasserstein


def test_single_bad_parameter_fails_despite_small_mean_error():
    truth = np.ones(13)
    estimate = truth.copy()
    estimate[3] = 1.2
    result = recovery(estimate, truth)
    assert result['mean_absolute_relative_error'] < .05
    assert result['worst_absolute_relative_error'] > .10
    assert not result['recovery_screen_pass']


def test_inactive_parameter_does_not_determine_historical_recovery():
    truth = np.ones(13)
    estimate = truth.copy()
    estimate[6] = 10
    assert recovery(estimate, truth)['recovery_screen_pass']


def test_empty_categories_have_missing_quantiles_not_zero_errors():
    rt = np.full((2, 20), 210.)
    categories = np.ones((2, 20), dtype=int)
    data = {'prop': np.full((2, 5), .2), 'quant': np.full((2, 5, 5), 210.)}
    result = observable_metrics(rt, categories, data)
    assert result['missing_quantiles'] == 40
    assert result['probability_max_absolute_error_pp'] == 80
    assert result['conditional_quantiles_ms'][0][0][1] is None


def test_sliced_distance_excludes_timeout_placeholders():
    directions = np.eye(2)
    a = [[0., 0.], [1., 1.], [np.nan, np.nan]]
    b = [[0., 0.], [1., 1.]]
    assert sliced_wasserstein(a, b, directions) == 0
    assert sliced_wasserstein([[np.nan, np.nan]], b, directions) is None


def test_timing_summary_retains_raw_values_and_constant_interval():
    result = median_interval([2., 2., 2.])
    assert result['raw_seconds'] == [2., 2., 2.]
    assert result['median_95_percent_bootstrap_ci_seconds'] == [2., 2.]
