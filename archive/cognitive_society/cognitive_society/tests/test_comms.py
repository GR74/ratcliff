"""Checkpoint 2 tests — DDM-coupled communication + trust. Pure NumPy, fast."""
import numpy as np

from cognitive_society.comms import TrustModel, leaning, social_drift


def test_social_drift_bounded_by_gain():
    trust = np.ones(4)
    comp = np.ones(4)
    leanings = np.array([1, 1, -1, 1])
    d = social_drift(trust, leanings, comp, social_gain=0.5)
    assert abs(d) <= 0.5 + 1e-9


def test_social_drift_zero_when_no_trust():
    d = social_drift(np.zeros(3), np.array([1, -1, 1]), np.ones(3), social_gain=1.0)
    assert d == 0.0


def test_social_drift_follows_trusted_peers():
    # peer 0 trusted + competent and leans +1; others untrusted -> drift positive
    trust = np.array([1.0, 0.0, 0.0])
    comp = np.array([1.0, 1.0, 1.0])
    leanings = np.array([1, -1, -1])
    assert social_drift(trust, leanings, comp, 1.0) > 0


def test_trust_rises_for_agreeing_peer_falls_for_disagreeing():
    tm = TrustModel(n_peers=2)
    # outcome +1; peer 0 always agrees (+1), peer 1 always disagrees (-1)
    for _ in range(10):
        tm.update(np.array([1, -1]), outcome=1)
    t = tm.trust()
    assert t[0] > 0.8, "consistently-agreeing peer should become trusted"
    assert t[1] < 0.2, "consistently-disagreeing peer should lose trust"


def test_trust_is_bounded():
    tm = TrustModel(n_peers=1, bound=3.0)
    for _ in range(100):
        tm.update(np.array([1]), outcome=1)
    assert tm.e[0] <= 3.0 + 1e-9  # accumulator stays bounded -> no blow-up


def test_prior_from_competence_orders_trust():
    tm = TrustModel(n_peers=3)
    tm.set_prior_from_competence([0.9, 0.5, 0.1])
    t = tm.trust()
    assert t[0] > t[1] > t[2], "higher inferred competence -> higher prior trust"


def test_adversary_identified_even_when_confident():
    """A confidently-wrong peer (always leans against the truth) is driven to low
    trust — the robustness the society needs."""
    tm = TrustModel(n_peers=2)
    truth = 1  # outcome +1 every round
    for _ in range(15):
        # peer 0 honest (leans +1), peer 1 adversary (confidently leans -1)
        tm.update(np.array([leaning(1), leaning(0)]), outcome=truth)
    t = tm.trust()
    assert t[0] > 0.85 and t[1] < 0.15
