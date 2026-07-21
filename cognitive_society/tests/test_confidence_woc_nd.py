"""Tests for Brick 3 — n-dimensional beliefs."""
import numpy as np

from cognitive_society.confidence_woc_nd import run, crowd_read, _specialist_kappas


def test_confidence_weighting_generalizes_to_nd():
    kap = np.random.default_rng(0).uniform(0.5, 3.0, (12, 8))
    r = run(kap, 1200, seed=0)
    assert r["confidence-weighted"] < r["simple mean"]
    assert r["confidence-weighted"] < r["median"]
    assert r["confidence-weighted"] <= r["oracle"] * 1.15


def test_division_of_labor_beats_best_single_specialist():
    """The n-D payoff: per-dimension confidence routes each dim to its expert, so a crowd of
    narrow specialists collectively crushes the best individual."""
    kap = _specialist_kappas(8, 8, np.random.default_rng(100), expert_per_agent=2)
    r = run(kap, 1200, seed=0)
    assert r["confidence-weighted"] < 0.5 * r["best single agent"]


def test_circumscribed_delusion_isolated_by_outcome_trust():
    kap = np.random.default_rng(0).uniform(0.5, 3.0, (12, 8))
    cg = np.ones((12, 8)); bs = np.zeros((12, 8))
    cg[:4, 0] = 10.0; bs[:4, 0] = 0.8            # 4 agents deluded ONLY on dim 0
    r = run(kap, 1200, conf_gains=cg, biases=bs, seed=0)
    assert r["outcome-trust"] < r["confidence-weighted"]


def test_crowd_read_shapes():
    kap = np.random.default_rng(0).uniform(0.5, 3.0, (5, 4))
    est, conf, prec = crowd_read(np.zeros(4), kap, 20, np.random.default_rng(0))
    assert est.shape == (5, 4) and conf.shape == (5, 4) and prec.shape == (5, 4)
    assert np.all(conf > 0) and np.all(prec > 0)
