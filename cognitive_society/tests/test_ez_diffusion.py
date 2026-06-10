"""Cognitive-mapping tests — EZ-diffusion recovers an agent's parameters from
its observed behavior. Pure NumPy, fast. This is the 'infer a peer's style'
engine for the society (1D closed form; the 2D model needs SBI instead)."""
import numpy as np
import pytest

from cognitive_society.agent import DDMAgent, DDMParams
from cognitive_society.ez_diffusion import ez_recover, recover_from_agent_observations


def _rng(seed=0):
    return np.random.default_rng(seed)


def _observe(agent, evidence, n, seed):
    choices, rts = agent.decide_batch(evidence, n, _rng(seed))
    return choices, rts


def test_recovers_boundary_ordering():
    """Cautious agent should be recovered with a higher boundary than decisive."""
    decisive = DDMAgent(DDMParams(boundary=0.6, drift_scale=1.0, ndt=0.2, sigma=1.0))
    cautious = DDMAgent(DDMParams(boundary=1.4, drift_scale=1.0, ndt=0.2, sigma=1.0))

    rec_d = ez_recover(*_observe(decisive, 0.6, 4000, 1), evidence=0.6, sigma=1.0)
    rec_c = ez_recover(*_observe(cautious, 0.6, 4000, 2), evidence=0.6, sigma=1.0)

    assert rec_c["boundary"] > rec_d["boundary"], (
        f"cautious boundary {rec_c['boundary']:.2f} should exceed "
        f"decisive {rec_d['boundary']:.2f}"
    )


def test_recovers_boundary_magnitude_roughly():
    """Recovered boundary should be in the right ballpark of the true value."""
    agent = DDMAgent(DDMParams(boundary=1.0, drift_scale=1.0, ndt=0.2, sigma=1.0))
    rec = ez_recover(*_observe(agent, 0.5, 6000, 3), evidence=0.5, sigma=1.0)
    # EZ is approximate; allow a generous band but it must be the right order.
    assert 0.5 < rec["boundary"] < 1.8, f"recovered boundary {rec['boundary']:.2f}"


def test_recovers_positive_drift_for_positive_evidence():
    agent = DDMAgent(DDMParams(boundary=1.0, drift_scale=1.0, ndt=0.2))
    rec = ez_recover(*_observe(agent, 0.6, 4000, 4), evidence=0.6, sigma=1.0)
    assert rec["drift"] > 0


def test_drift_increases_with_evidence():
    agent = DDMAgent(DDMParams(boundary=1.0, drift_scale=1.0, ndt=0.2))
    rec_low = ez_recover(*_observe(agent, 0.3, 5000, 5), evidence=0.3, sigma=1.0)
    rec_high = ez_recover(*_observe(agent, 0.9, 5000, 6), evidence=0.9, sigma=1.0)
    assert rec_high["drift"] > rec_low["drift"], (
        f"drift should rise with evidence: {rec_low['drift']:.2f} -> {rec_high['drift']:.2f}"
    )


def test_recovers_ndt_roughly():
    agent = DDMAgent(DDMParams(boundary=1.0, drift_scale=1.0, ndt=0.3, sigma=1.0))
    rec = ez_recover(*_observe(agent, 0.6, 5000, 7), evidence=0.6, sigma=1.0)
    # ndt is the easiest to recover; should be within ~0.15s of truth.
    assert abs(rec["ndt"] - 0.3) < 0.15, f"recovered ndt {rec['ndt']:.3f}"


def test_pooled_recovery_across_evidence_levels():
    """Pooling several evidence levels gives a robust style estimate."""
    agent = DDMAgent(DDMParams(boundary=1.2, drift_scale=1.0, ndt=0.2, sigma=1.0))
    obs = [
        (*_observe(agent, 0.4, 3000, 10), 0.4),
        (*_observe(agent, 0.6, 3000, 11), 0.6),
        (*_observe(agent, 0.8, 3000, 12), 0.8),
    ]
    rec = recover_from_agent_observations(obs, sigma=1.0)
    assert rec["n_batches"] == 3
    assert 0.7 < rec["boundary"] < 1.9, f"pooled boundary {rec['boundary']:.2f}"


def test_two_distinct_agents_get_distinct_maps():
    """The core 'cognitive map' property: different agents -> distinguishable
    recovered profiles."""
    a = DDMAgent(DDMParams(boundary=0.6, drift_scale=1.4, ndt=0.15))
    b = DDMAgent(DDMParams(boundary=1.5, drift_scale=0.8, ndt=0.30))
    rec_a = ez_recover(*_observe(a, 0.6, 5000, 20), evidence=0.6, sigma=1.0)
    rec_b = ez_recover(*_observe(b, 0.6, 5000, 21), evidence=0.6, sigma=1.0)
    # a is more decisive (lower boundary) and faster (lower ndt) than b
    assert rec_a["boundary"] < rec_b["boundary"]
    assert rec_a["ndt"] < rec_b["ndt"]
