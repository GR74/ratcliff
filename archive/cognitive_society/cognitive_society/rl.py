"""Checkpoint 4 (RL) — a learned adaptive deference policy.

The hand-tuned society gates deference by a fixed rule: defer + grow caution in
proportion to the agent's own uncertainty (``sg = social_gain * (0.5 + unc)``).
That rule is good but STATIC — it can't notice that in *this* regime the peers
became unreliable (so it should defer less) or unusually trustworthy (defer more).
Its social-gain floor means it keeps deferring even to a fully compromised group.

Here each decision instead consults a small REINFORCE policy that maps an agent's
observable context -> how much to defer, and learns online from whether deferring
led to a correct decision. Pure NumPy. The point is ADAPTATION: when the
environment shifts (peers turn adversarial, problems get harder) the learned
policy retunes its deference where the fixed rule cannot.

State features phi(s) — all observable at decision time, in ~[0, 1]:
    unc          1 - own confidence (how split the agent's private vote was)
    mean_trust   average trust this agent currently places in its peers
    consensus    |mean(peer leanings)| — how strongly the peers agree with each other
    bias         constant 1.0
Action: a deference multiplier ``m`` in (0, m_max); the round's social gain is
    ``sg = social_gain * m``   (m replaces the fixed (0.5 + unc) factor).
Reward: +1 if the agent's own final decision matched the truth, else 0.

This is a deliberately small, transparent RL: a linear policy + Gaussian
exploration + a running-baseline REINFORCE update. It is enough to *learn the
deference shape from reward* and to *adapt it online* — the claim being made —
without a neural net or any GPU.
"""
import numpy as np


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class DeferencePolicy:
    """Linear-Gaussian REINFORCE policy over a scalar deference action.

    One shared policy pools experience across all agents/rounds (fast to learn
    with a handful of agents). Per agent-decision call :meth:`act` to get a
    deference multiplier (it stashes the state/action); after the round, call
    :meth:`update` with the per-decision rewards in the same order.

    Action transform: ``m = m_max * sigmoid(z)`` where ``z ~ N(mu, sigma)`` and
    ``mu = theta . phi``. At ``theta = 0`` the mean action is ``m_max/2`` (=1.0 by
    default), i.e. neutral — ``sg = social_gain`` — so an untrained policy starts
    in the middle of the hand-tuned rule's range and learns to move from there.
    """

    N_FEATURES = 4  # unc, mean_trust, consensus, bias

    def __init__(self, m_max: float = 2.0, sigma: float = 0.5, lr: float = 0.05,
                 baseline_decay: float = 0.9, seed: int = 0):
        self.m_max = m_max
        self.sigma = sigma
        self.lr = lr
        self.baseline_decay = baseline_decay
        self.theta = np.zeros(self.N_FEATURES)
        self.baseline = 0.0
        self.rng = np.random.default_rng(seed)
        self._buffer = []  # (phi, z, mu) awaiting their reward

    @staticmethod
    def features(unc: float, mean_trust: float, consensus: float) -> np.ndarray:
        """Assemble the state feature vector (with the constant bias term)."""
        return np.array([unc, mean_trust, consensus, 1.0], dtype=float)

    def act(self, phi: np.ndarray) -> float:
        """Sample a deference multiplier ``m`` in (0, m_max) for context ``phi``
        and cache (phi, z, mu) so the next :meth:`update` can credit it."""
        mu = float(self.theta @ phi)
        z = mu + self.sigma * float(self.rng.standard_normal())
        self._buffer.append((phi, z, mu))
        return self.m_max * _sigmoid(z)

    def act_greedy(self, phi: np.ndarray) -> float:
        """Deterministic action (no exploration, no caching) — for evaluation."""
        return self.m_max * _sigmoid(float(self.theta @ phi))

    def update(self, rewards) -> None:
        """REINFORCE update over the buffered (state, action) pairs.

        ``rewards`` is aligned with the :meth:`act` calls since the last update.
        Gradient of the Gaussian log-policy wrt theta is ``(z - mu)/sigma^2 * phi``;
        we step theta by ``lr * (R - baseline) * grad``, then move the running
        baseline toward this batch's mean reward (variance reduction).
        """
        if not self._buffer:
            return
        rewards = np.asarray(rewards, dtype=float)
        if len(rewards) != len(self._buffer):
            raise ValueError(
                f"rewards ({len(rewards)}) must align with buffered actions "
                f"({len(self._buffer)})"
            )
        b = self.baseline
        inv_var = 1.0 / (self.sigma ** 2)
        for (phi, z, mu), R in zip(self._buffer, rewards):
            grad_logp = (z - mu) * inv_var * phi
            self.theta += self.lr * (R - b) * grad_logp
        self.baseline = (self.baseline_decay * b
                         + (1.0 - self.baseline_decay) * float(rewards.mean()))
        self._buffer.clear()

    def clear(self) -> None:
        """Drop any buffered actions without learning from them (e.g. when a
        round is run in evaluation mode)."""
        self._buffer.clear()
