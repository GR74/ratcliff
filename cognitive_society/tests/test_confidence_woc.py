"""Tests for Brick 1 — confidence-weighted wisdom of crowds. Pure NumPy, fast."""
import numpy as np

from cognitive_society.confidence_woc import run, agent_read


def test_confidence_is_calibrated_and_beats_naive_aggregators():
    """The load-bearing claim: DDM-family confidence tracks accuracy, so confidence-weighting
    beats simple mean AND median and lands near the oracle inverse-variance estimator."""
    kappas = np.random.default_rng(0).uniform(0.5, 3.0, 12)
    out = run(kappas, n_problems=2000, seed=0)
    # confidence is negatively correlated with error (higher confidence -> lower error)
    assert out["calibration_spearman"] < -0.1
    r = out["rmse"]
    assert r["confidence-weighted"] < r["simple mean"]
    assert r["confidence-weighted"] < r["median"]
    # within ~15% of the oracle that KNOWS the true variances
    assert r["confidence-weighted"] <= r["oracle (true precision)"] * 1.15


def test_delusion_inverts_confidence_weighting():
    """A spreading DELUSION (shared false belief + aberrant precision) captures the confidence-
    weighted crowd — it becomes WORSE than a naive average. The computational-psychiatry hook."""
    K = 15
    kappas = np.random.default_rng(1).uniform(0.5, 3.0, K)
    idx = np.argsort(kappas)[:8]        # 8/15 deluded
    cg, bs = np.ones(K), np.zeros(K)
    cg[idx], bs[idx] = 10.0, 0.8
    out = run(kappas, n_problems=2000, conf_gains=cg, biases=bs, seed=0)
    assert out["rmse"]["confidence-weighted"] > out["rmse"]["simple mean"]


def test_outcome_trust_defends_against_delusion():
    """Outcome-grounded meta-trust (learned from revealed outcomes) stays robust where self-
    reported confidence-weighting is CAPTURED by a spreading delusion — the defense, and the
    same outcome-vs-self-report distinction as the swarm work."""
    K = 15
    kappas = np.random.default_rng(1).uniform(0.5, 3.0, K)
    idx = np.argsort(kappas)[:8]        # 8/15 deluded
    cg, bs = np.ones(K), np.zeros(K)
    cg[idx], bs[idx] = 10.0, 0.8
    r = run(kappas, n_problems=2000, conf_gains=cg, biases=bs, seed=0)["rmse"]
    assert r["outcome-trust (learned)"] < r["confidence-weighted"] / 3   # dramatically better
    assert r["outcome-trust (learned)"] < r["simple mean"]               # stays useful


def test_outcome_trust_matches_confidence_when_calibrated():
    """When everyone is calibrated, outcome-trust ~= confidence-weighting: you don't NEED
    outcomes. (Only miscalibration makes the outcome grounding pay off.)"""
    kappas = np.random.default_rng(0).uniform(0.5, 3.0, 12)
    r = run(kappas, n_problems=2000, seed=0)["rmse"]
    ot, cw = r["outcome-trust (learned)"], r["confidence-weighted"]
    assert abs(ot - cw) < 0.4 * cw     # within ~40% of each other; both near oracle


def test_agent_read_shapes_and_bias():
    est, conf, prec = agent_read(0.5, kappa=2.0, n_samples=20,
                                 rng=np.random.default_rng(0), conf_gain=1.0, bias=0.0)
    assert conf > 0 and prec > 0
    # a strong systematic bias pulls the estimate off the truth
    est_biased, _, _ = agent_read(0.5, kappa=50.0, n_samples=200,
                                  rng=np.random.default_rng(0), bias=0.8)
    assert abs(est_biased - (0.5 + 0.8)) < 0.05
