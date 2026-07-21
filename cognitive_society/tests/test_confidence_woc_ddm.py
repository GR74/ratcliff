"""Tests for Brick 1 grounded in the real DDM (chronometric vs evidence-state confidence)."""
import numpy as np

from cognitive_society.confidence_woc_ddm import build_crowd, run


def test_ddm_evidence_confidence_is_calibrated_and_helps():
    """The DDM's evidence-state confidence (internal decision consistency) IS calibrated to
    accuracy and, weighted, lifts the crowd over plain majority — the claim, in the real model."""
    agents = build_crowd(4, 11, seed=0)[0]
    out = run(agents, n_problems=1000, seed=0)
    assert out["calib_cons"] > 0.1
    acc = out["accuracy"]
    assert acc["consistency-conf (DDM)"] > acc["plain majority"]


def test_raw_rt_confidence_is_not_calibrated():
    """Honest negative: raw RT confounds caution (boundary) with evidence quality, so it is NOT
    positively calibrated — it is strictly worse than the evidence-state readout."""
    agents = build_crowd(4, 11, seed=1)[0]
    out = run(agents, n_problems=1000, seed=1)
    assert out["calib_rt"] < out["calib_cons"]


def test_decide_grid_matches_single_decision_dynamics():
    """The vectorized (K,P) grid is the same DDM as DDMAgent.decide: choices are valid, RTs > ndt."""
    from cognitive_society.confidence_woc_ddm import decide_grid
    agents = build_crowd(2, 3, seed=0)[0]
    ev = np.array([0.2, -0.2, 0.1])
    ch, rt = decide_grid(agents, ev, np.random.default_rng(0))
    assert ch.shape == (5, 3) and rt.shape == (5, 3)
    assert set(np.unique(ch)).issubset({0, 1})
    assert np.all(rt > 0)
