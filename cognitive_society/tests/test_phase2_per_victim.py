"""Smoke tests for the per-victim isolation experiment (Phase 2). Pure NumPy, fast."""
import numpy as np

from cognitive_society.baselines import AgreementTrustModel
from cognitive_society.phase2_per_victim import build_and_run, victim_aucs


def test_build_and_run_returns_valid_state():
    soc, gt, victims = build_and_run(16, 4, 5, seed=0, agreement=False,
                                     prior=False, n_problems=6)
    assert soc.K == 16
    assert set(np.unique(gt)).issubset({0, 1})
    for _, auc in victim_aucs(soc, gt, victims):
        assert 0.0 <= auc <= 1.0  # AUC is a probability


def test_agreement_arm_swaps_trust_models():
    soc, _, _ = build_and_run(16, 4, 5, seed=1, agreement=True, prior=False, n_problems=6)
    assert all(isinstance(t, AgreementTrustModel) for t in soc.trust)


def test_outcome_arm_keeps_default_trust():
    soc, _, _ = build_and_run(16, 4, 5, seed=2, agreement=False, prior=False, n_problems=6)
    assert not any(isinstance(t, AgreementTrustModel) for t in soc.trust)
