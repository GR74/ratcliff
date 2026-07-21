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


def test_precision_bound_fixes_double_counting_overconfidence():
    """Bounding precision at the total independent evidence removes the naive loop's double-
    counting overconfidence WITHOUT costing accuracy (Golub-Jackson fix)."""
    kappas, adj = _setup(24)
    unb = belief_loop(kappas, adj, 800, gate="confidence", bound_precision=False, seed=0)
    bnd = belief_loop(kappas, adj, 800, gate="confidence", bound_precision=True, seed=0)
    assert bnd["mean_precision"] < unb["mean_precision"] / 3      # overconfidence tamed
    assert bnd["collective_rmse"] < unb["collective_rmse"] * 1.2  # accuracy preserved


def test_homeostasis_maintains_diversity_and_blunts_delusion():
    """The RSL set-point principle ported to the belief loop: homeostatic coupling keeps a
    diversity 'gap' (doubters) and partially resists the delusion WITHOUT any outcome signal."""
    kappas, adj = _setup(24)
    K = len(kappas)
    cg, bs = np.ones(K), np.zeros(K)
    cg[:8], bs[:8] = 10.0, 0.8
    plain = belief_loop(kappas, adj, 800, gate="confidence", conf_gains=cg, biases=bs, seed=0)
    homeo = belief_loop(kappas, adj, 800, gate="confidence", conf_gains=cg, biases=bs,
                        homeostatic=True, seed=0)
    assert homeo["belief_spread"] > plain["belief_spread"]   # doubters persist (diversity kept)
    assert homeo["honest_rmse"] < plain["honest_rmse"]       # delusion blunted, outcome-free


def test_robust_precision_plus_homeostasis_stacks():
    """The two outcome-free levers stack: robust precision cap (kill aberrant confidence) +
    homeostatic diversity cut the delusion's grip far more than either alone — no outcomes."""
    kappas, adj = _setup(24)
    K = len(kappas)
    cg, bs = np.ones(K), np.zeros(K)
    cg[:8], bs[:8] = 10.0, 0.8
    none = belief_loop(kappas, adj, 800, gate="confidence", conf_gains=cg, biases=bs, seed=0)
    both = belief_loop(kappas, adj, 800, gate="confidence", conf_gains=cg, biases=bs,
                       homeostatic=True, target_spread=1.0, robust_cap=1.5, seed=0)
    assert both["honest_rmse"] < 0.6 * none["honest_rmse"]     # substantial, outcome-free


def test_outcome_free_defense_does_not_hurt_healthy_crowd():
    """The robust+homeostatic defense must be ~free on a healthy crowd (no over-skepticism cost)."""
    kappas, adj = _setup(24)
    base = belief_loop(kappas, adj, 800, gate="confidence", seed=0)["collective_rmse"]
    defended = belief_loop(kappas, adj, 800, gate="confidence", homeostatic=True,
                           target_spread=1.0, robust_cap=1.5, seed=0)["collective_rmse"]
    assert defended < 1.3 * base


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
