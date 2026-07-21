"""Tests for Brick 2 — the iterated belief<->evidence<->confidence loop."""
import numpy as np

from cognitive_society import topology as topo
from cognitive_society.belief_loop import belief_loop


def _setup(K=24, deg=4):
    kappas = np.random.default_rng(0).uniform(0.5, 3.0, K)
    adj = topo.watts_strogatz(K, deg, 0.3, np.random.default_rng(1))
    return kappas, adj


def test_healthy_loop_is_finite_and_accurate():
    kappas, adj = _setup(12)
    r = belief_loop(kappas, adj, n_problems=500, gate="confidence", seed=0)
    assert np.isfinite(r["collective_rmse"]) and r["collective_rmse"] < 0.2


def test_delusion_propagates_and_outcome_trust_defends():
    """The loop lets a deluded cluster PROPAGATE into a confident-but-wrong consensus (echo
    chamber) under confidence-gating; outcome-trust keeps honest agents accurate."""
    kappas, adj = _setup(24)
    K = len(kappas)
    cg, bs = np.ones(K), np.zeros(K)
    cg[:8], bs[:8] = 10.0, 0.8                      # 8/24 deluded cluster
    conf = belief_loop(kappas, adj, 800, gate="confidence", conf_gains=cg, biases=bs, seed=0)
    ot = belief_loop(kappas, adj, 800, gate="outcome-trust", conf_gains=cg, biases=bs, seed=0)
    assert conf["honest_rmse"] > 0.2                # delusion drags honest agents off-truth
    assert ot["honest_rmse"] < conf["honest_rmse"] / 3   # outcome-trust defends
