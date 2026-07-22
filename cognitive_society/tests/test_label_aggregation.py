"""Tests for the label-aggregation (RLHF/crowdsourcing) companion result."""
import numpy as np

from cognitive_society.label_aggregation import run


def test_confidence_weighting_cleans_labels_when_calibrated():
    r = run(np.random.default_rng(0).uniform(0.3, 2.0, 10), 3000, seed=0)
    assert r["confidence"] > r["majority"]          # cleaner labels than plain majority
    assert r["confidence"] >= 0.95 * r["oracle"]    # tracks the oracle without gold


def test_careless_confident_annotators_poison_confidence_not_track_record():
    rng = np.random.default_rng(1)
    skill = np.concatenate([rng.uniform(0.3, 2.0, 10), np.zeros(5)])   # +5 careless (chance)
    cg = np.concatenate([np.zeros(10), np.full(5, 4.0)])              # ...loudly confident
    r = run(skill, 3000, conf_gains=cg, seed=1)
    assert r["confidence"] < r["majority"]          # confidence-weighting is poisoned below majority
    assert r["track-record"] > r["confidence"]      # gold-grounded track-record stays robust
