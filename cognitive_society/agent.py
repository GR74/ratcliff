"""Lightweight 1D diffusion-decision agents for the Cognitive Society.

Each agent decides a 2-alternative choice by accumulating noisy evidence to a
boundary. Parameters define its "personality":
    boundary    a  — caution (high = waits for more evidence; slower, more accurate)
    drift_scale    — how strongly the true signal pulls the accumulator
    ndt        t0  — non-decision time (processing/motor lag, seconds)
    sigma          — within-trial noise standard deviation

This is the cheap per-agent primitive for a society of many agents. The 2D
spatial simulator (model_b) is the academic engine; here we want fast, simple,
many-agent decisions with NO compile/GPU dependency — pure NumPy.

Checkpoint 1 of docs/plans/2026-06-05-OVERALL-PLAN-cognitive-society.md.
"""
from dataclasses import dataclass

import numpy as np

# Simulation granularity. DT small enough for stable accumulation; MAX_STEPS is a
# hard cap so every trial terminates (a non-crossing trial commits to its sign).
DT = 0.01
MAX_STEPS = 2000


@dataclass
class DDMParams:
    """Decision parameters = an agent's cognitive style."""
    boundary: float = 1.0       # distance from start (0) to each bound (+a / -a)
    drift_scale: float = 1.0    # multiplies evidence to get the drift rate
    ndt: float = 0.2            # non-decision time, seconds
    sigma: float = 1.0          # within-trial noise sd

    def style(self) -> str:
        """A human label for the personality, from the boundary (caution)."""
        if self.boundary >= 1.4:
            return "cautious"
        if self.boundary <= 0.6:
            return "decisive"
        return "balanced"


class DDMAgent:
    """A single diffusion-decision agent.

    Decision: accumulate `x` from 0; positive evidence drifts toward +boundary
    (choice 1), negative toward -boundary (choice 0). Crossing a bound commits
    the choice and yields a reaction time.
    """

    def __init__(self, params: DDMParams, agent_id: int = 0):
        self.params = params
        self.agent_id = agent_id

    def decide(self, evidence: float, rng: np.random.Generator):
        """One decision at a given evidence level.

        evidence > 0 favors choice 1, < 0 favors choice 0, 0 is ambiguous.
        Returns (choice in {0, 1}, rt in seconds).
        """
        choices, rts = self.decide_batch(evidence, 1, rng)
        return int(choices[0]), float(rts[0])

    def decide_batch(self, evidence: float, n: int, rng: np.random.Generator):
        """Vectorized: n independent decisions at one evidence level.

        Returns (choices: int array (n,), rts: float array (n,)). Fast — all
        trials step in lockstep until each crosses a bound.
        """
        p = self.params
        drift = p.drift_scale * evidence
        noise_sd = p.sigma * np.sqrt(DT)

        x = np.zeros(n)
        rt = np.full(n, np.nan)
        choice = np.full(n, -1, dtype=np.int64)
        active = np.ones(n, dtype=bool)

        for step in range(1, MAX_STEPS + 1):
            k = int(active.sum())
            if k == 0:
                break
            x[active] += drift * DT + noise_sd * rng.standard_normal(k)

            hi = active & (x >= p.boundary)
            lo = active & (x <= -p.boundary)
            t = p.ndt + step * DT
            choice[hi] = 1
            rt[hi] = t
            choice[lo] = 0
            rt[lo] = t
            active &= ~(hi | lo)

        # Any trial that never crossed commits to the sign of its accumulator.
        if active.any():
            choice[active] = (x[active] >= 0).astype(np.int64)
            rt[active] = p.ndt + MAX_STEPS * DT

        return choice, rt


def make_population(styles, rng_seed: int = 0):
    """Build a list of agents from (style_name, DDMParams) pairs or DDMParams.

    Convenience for assembling a heterogeneous society.
    """
    agents = []
    for i, item in enumerate(styles):
        params = item[1] if isinstance(item, (tuple, list)) else item
        agents.append(DDMAgent(params, agent_id=i))
    return agents
