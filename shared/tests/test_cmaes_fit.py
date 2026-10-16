"""CMA-ES driver: minimises, respects bounds, is seeded, copes with plateaus."""
import numpy as np
import pytest

from shared.cmaes_fit import minimize_cmaes

TARGET = np.array([3.0, -2.0, 0.5, 10.0])


def sphere(x, nsim):
    return float(np.sum((x - TARGET) ** 2))


def staircase(x, nsim):
    # Piecewise-constant like a hard-count G2 under common random numbers.
    return float(np.sum(np.floor(4 * np.abs(x - TARGET)) ** 2))


def test_minimises_sphere():
    res = minimize_cmaes(sphere, np.zeros(4), np.ones(4) * 3, nsim=1,
                         max_evals=3000, seed=0, sigma0=1.0)
    assert res.loss < 1e-4
    np.testing.assert_allclose(res.params, TARGET, atol=1e-2)


def test_respects_box():
    upper = np.array([1.0, 5.0, 5.0, 20.0])
    res = minimize_cmaes(sphere, np.zeros(4), np.ones(4) * 3, nsim=1,
                         max_evals=3000, seed=1, sigma0=1.0,
                         lower=-np.full(4, 5.0), upper=upper)
    assert np.all(res.params <= upper + 1e-9)
    assert res.params[0] == pytest.approx(1.0, abs=1e-2)


def test_seeded_runs_are_identical():
    a = minimize_cmaes(sphere, np.zeros(4), np.ones(4), nsim=1, max_evals=400, seed=7)
    b = minimize_cmaes(sphere, np.zeros(4), np.ones(4), nsim=1, max_evals=400, seed=7)
    np.testing.assert_array_equal(a.params, b.params)
    assert a.loss == b.loss


def test_reaches_bottom_step_of_staircase():
    res = minimize_cmaes(staircase, np.zeros(4), np.ones(4) * 3, nsim=1,
                         max_evals=4000, seed=0, sigma0=1.0, restarts=2)
    assert res.loss == 0.0
