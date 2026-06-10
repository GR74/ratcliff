"""Checkpoint 2 — DDM-coupled communication + trust.

Communication is not a separate message bus with a bolt-on trust scalar; it is
modelled as DDM interactions. A peer's decision (its signed leaning, weighted by
how competent the agent believes that peer is) enters the agent's evidence
accumulation as a trust-weighted SOCIAL DRIFT. Trust itself is a slow
meta-accumulator: it rises when a peer's leaning matched the outcome, falls when
it didn't.

Stability (coupled accumulators can diverge / echo-chamber): the social drift is
trust-normalized and gain-capped, and trust evidence is bounded. The cap is the
`social_gain` passed in per round; the society raises it with the agent's own
uncertainty (up to ~1.5x the base gain), so under high uncertainty social
evidence *can* intentionally outweigh weak private evidence — bounded deference,
not a runaway loop. The non-pathology guarantees are: bounded {-1,+1} leanings,
a finite per-round (not within-trial) gain cap, and OUTCOME-driven (not
agreement-driven) trust. See docs/notes/2026-06-05-ddm-coupled-comms-design-notes.md.
"""
import numpy as np


def social_drift(trust_row, leanings, competences, social_gain: float) -> float:
    """Trust-weighted social drift entering one agent's DDM.

    trust_row   : (K,) trust this agent places in each peer, in [0, 1].
    leanings    : (K,) each peer's signed decision direction, in {-1, +1}.
    competences : (K,) each peer's believed reliability, in [0, 1] (from the
                  cognitive map; uniform before the map is built).
    social_gain : cap on this round's social influence. The society may pass an
                  uncertainty-boosted gain (up to ~1.5x the configured base) so
                  deference rises when the agent is unsure — see Society.round.

    Returns a scalar drift term with |value| <= social_gain (the value passed in).
    """
    trust_row = np.asarray(trust_row, dtype=float)
    leanings = np.asarray(leanings, dtype=float)
    competences = np.asarray(competences, dtype=float)
    w = trust_row * competences
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s  # normalize -> sum(w*leaning) in [-1, 1] -> social drift bounded
    return float(social_gain * np.sum(w * leanings))


class TrustModel:
    """Per-agent trust over peers as a slow, bounded meta-accumulator.

    Trust evidence `e[j]` rises when peer j's leaning agreed with the outcome and
    falls otherwise; `trust()` maps it through a logistic to (0, 1). The slow
    timescale (one update per decision round, not per ms) separates it from the
    fast decision DDM.
    """

    def __init__(self, n_peers: int, lr: float = 0.4, bound: float = 5.0):
        self.e = np.zeros(n_peers, dtype=float)
        self.lr = lr
        self.bound = bound

    def trust(self) -> np.ndarray:
        # Clip the logit before the logistic so trust() is overflow-safe for any
        # e, independent of the [-bound, bound] clamp applied in update()/prior.
        return 1.0 / (1.0 + np.exp(-np.clip(self.e, -30.0, 30.0)))

    def set_prior_from_competence(self, competences, gain: float = 3.0):
        """Seed trust evidence from cognitive-map competence (in [0,1]).

        Grounding trust in inferred competence — not just outcome history — is
        the part that lets the society flag confidently-wrong / adversarial
        agents quickly: a peer whose *inferred* competence is low starts with
        low trust even before its bad advice has cost anything.
        """
        c = np.clip(np.asarray(competences, dtype=float), 1e-3, 1 - 1e-3)
        self.e = np.clip(gain * (2.0 * c - 1.0), -self.bound, self.bound)

    def update(self, peer_leanings, outcome: int):
        """Update trust from one round. outcome in {-1, +1} (the true/consensus
        direction); peer_leanings in {-1, +1}. outcome == 0 (no signed outcome)
        is treated as 'no information' and leaves trust unchanged."""
        leanings = np.sign(np.asarray(peer_leanings, dtype=float))
        agree = leanings * np.sign(outcome)  # +1 agreed, -1 disagreed, 0 no-op
        self.e = np.clip(self.e + self.lr * agree, -self.bound, self.bound)


def leaning(choice: int) -> int:
    """Map a 2-choice decision {0,1} to a signed leaning {-1,+1}."""
    return 1 if choice == 1 else -1
