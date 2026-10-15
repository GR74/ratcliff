"""Smoke + invariant tests for Phase 4 iterative consensus. Pure NumPy, fast."""
import numpy as np

from cognitive_society.phase4_consensus import (
    clustered_graph, honest_graph, run, METHODS,
)


def test_run_returns_probability():
    adj, gt = clustered_graph(24, 4, 6, seed=0)
    for m in METHODS:
        acc = run(adj, gt, m, n_problems=10, seed=0)
        assert 0.0 <= acc <= 1.0


def test_all_honest_sanity_every_method_converges():
    """The load-bearing bug-check: with NO adversaries the swarm must converge to
    truth regardless of trust rule. If any method is low here, the dynamics are broken
    and the cascade gap would be an artifact."""
    adj, gt = honest_graph(40, 4, seed=0)
    for m in METHODS:
        acc = run(adj, gt, m, n_problems=20, seed=0)
        assert acc > 0.85, f"{m} only {acc:.2f} on all-honest graph"


def test_outcome_trust_beats_agreement_in_cascade():
    """Core claim: a contiguous confidently-wrong cluster cascades; outcome-grounded
    trust resists it, agreement-based trust does not. Averaged over a few seeds so the
    assertion is not seed-fragile."""
    N, deg, n_adv, seeds = 60, 4, 18, range(4)   # 30% cluster
    out = np.mean([run(*clustered_graph(N, deg, n_adv, s), "outcome", seed=s) for s in seeds])
    agr = np.mean([run(*clustered_graph(N, deg, n_adv, s), "agreement", seed=s) for s in seeds])
    assert out > agr + 0.30, f"outcome {out:.2f} not >> agreement {agr:.2f}"
    assert out > 0.80, f"outcome should hold the swarm, got {out:.2f}"
