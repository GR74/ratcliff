"""Engine-agnostic decision interface for society agents.

The whole point: the society's logic (communication, trust, cognitive mapping,
adaptation) depends ONLY on this `DecisionEngine` contract — never on a specific
model. So the two engines are fully interchangeable:

  - `DDMAgent` (agent.py)        — fast 1D NumPy DDM. No compile, no GPU. The
                                    default for building + iterating society logic.
  - `SpatialDDMEngine` (here)    — the validated 2D spatial diffusion model
                                    (model_b, JAX, GPU-preferred). Heavyweight;
                                    5 spatial categories. The "real cognitive
                                    model" option for final results.

Prototype the society on 1D (seconds per run); swap in the 2D engine for
scale/validation (the GPU version stays fully available). Cognitive mapping uses
EZ-diffusion (closed form) for the 1D engine and amortized SBI (Track B) for the
2D engine.
"""
from typing import Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class DecisionEngine(Protocol):
    """The contract every agent decision engine satisfies.

    n_choices    : number of response alternatives (2 for 1D DDM, 5 for spatial).
    decide       : one decision -> (choice_index, rt_seconds).
    decide_batch : n independent decisions -> (choices array, rts array).
    """

    n_choices: int

    def decide(self, stimulus, rng) -> Tuple[int, float]:
        ...

    def decide_batch(self, stimulus, n: int, rng) -> Tuple[np.ndarray, np.ndarray]:
        ...


class SpatialDDMEngine:
    """Adapter: the 2D spatial diffusion model (model_b) as a society engine.

    Each spatial agent carries a fixed parameter set (its personality). A
    decision runs the validated 2D simulator and maps its (rt_ms, category 1..5)
    output to the society contract (choice 0..4, rt seconds).

    Heavyweight: JAX, GPU-preferred (set use_kl=True on GPU for the K-L fast
    path). Runs are seconds-to-minutes, not microseconds — use this for final
    results, not for iterating society logic. Cognitive mapping for this engine
    uses amortized SBI (Track B), since the 2D model has no EZ-style closed form.

    params: dict with keys ter, st, cr, crsd, av1, av2, av3, sis, sig, si
            (the model_b.simulate_b signature, minus key/nsim/chunk_size).
    """

    n_choices = 5

    def __init__(self, params: dict, chunk_size: int = 16, use_kl: bool = False):
        required = {"ter", "st", "cr", "crsd", "av1", "av2", "av3", "sis", "sig", "si"}
        missing = required - set(params)
        if missing:
            raise ValueError(f"SpatialDDMEngine params missing: {sorted(missing)}")
        self.params = dict(params)
        self.chunk_size = chunk_size
        self.use_kl = use_kl

    def decide(self, stimulus, rng) -> Tuple[int, float]:
        c, r = self.decide_batch(stimulus, 1, rng)
        return int(c[0]), float(r[0])

    def decide_batch(self, stimulus, n: int, rng) -> Tuple[np.ndarray, np.ndarray]:
        # Imported lazily so importing this module doesn't require JAX.
        import jax

        from model_b import simulate as sim_b

        seed = int(rng.integers(0, 2**31 - 1))
        key = jax.random.key(seed)
        rt, cat = sim_b.simulate_b(
            key, nsim=int(n), chunk_size=self.chunk_size, use_kl=self.use_kl,
            **self.params,
        )
        choices = np.asarray(cat).astype(np.int64) - 1   # categories 1..5 -> 0..4
        rts = np.asarray(rt, dtype=float) / 1000.0        # ms -> seconds
        return choices, rts


def is_decision_engine(obj) -> bool:
    """True if obj satisfies the DecisionEngine contract (has n_choices +
    decide + decide_batch)."""
    return (
        hasattr(obj, "n_choices")
        and callable(getattr(obj, "decide", None))
        and callable(getattr(obj, "decide_batch", None))
    )
