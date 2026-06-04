import numpy as np
import pytest

from shared import validation


def test_proportions_match_when_identical():
    a = np.array([0.3, 0.5, 0.2])
    ok, report = validation.proportions_match(a, a, abs_tol=0.005)
    assert ok
    assert report["max_abs_diff"] == 0.0


def test_proportions_match_within_tolerance():
    a = np.array([0.3, 0.5, 0.2])
    b = np.array([0.302, 0.499, 0.199])
    ok, _ = validation.proportions_match(a, b, abs_tol=0.005)
    assert ok


def test_proportions_match_fails_outside_tolerance():
    a = np.array([0.3, 0.5, 0.2])
    b = np.array([0.31, 0.49, 0.20])
    ok, report = validation.proportions_match(a, b, abs_tol=0.005)
    assert not ok
    assert report["max_abs_diff"] > 0.005


def test_quantiles_match_within_relative_tolerance():
    a = np.array([300.0, 400.0, 500.0])
    b = np.array([301.0, 401.0, 501.0])
    ok, _ = validation.quantiles_match(a, b, rel_tol=0.01)
    assert ok


def test_quantiles_match_fails_outside_relative_tolerance():
    a = np.array([300.0, 400.0, 500.0])
    b = np.array([330.0, 400.0, 500.0])  # 10% off on first
    ok, _ = validation.quantiles_match(a, b, rel_tol=0.01)
    assert not ok


def test_aggregate_match_combines_both():
    obs_prop = np.array([0.3, 0.5, 0.2])
    obs_quant = np.array([[300.0, 400.0], [310.0, 410.0]])  # (n_quantiles=2, n_cat=2)... ignored shape, just illustrative
    sim_prop = np.array([0.302, 0.499, 0.199])
    sim_quant = obs_quant.copy()
    result = validation.aggregate_match(
        obs_prop, sim_prop, obs_quant, sim_quant, prop_abs_tol=0.005, quant_rel_tol=0.01
    )
    assert result["passed"]
    assert result["prop_passed"]
    assert result["quant_passed"]
