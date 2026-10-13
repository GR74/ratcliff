"""Explicit coarse-to-fine Nelder-Mead budgets, ending at full simulation size."""
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class TrialStage:
    nsim: int
    maxiter: int


@dataclass
class SimplexResult:
    params: np.ndarray
    loss: float
    n_iters: int
    n_evals: int
    converged: bool
    stages: list[dict]


def validate_schedule(nsim, maxiter, schedule):
    if nsim < 1 or maxiter < 1:
        raise ValueError("nsim and maxiter must be positive")
    stages = (TrialStage(nsim, maxiter),) if schedule is None else tuple(schedule)
    if not stages or any(s.nsim < 1 or s.maxiter < 1 for s in stages):
        raise ValueError("each trial stage must have positive nsim and maxiter")
    if any(a.nsim >= b.nsim for a, b in zip(stages, stages[1:])):
        raise ValueError("stage trial counts must increase strictly")
    if stages[-1].nsim != nsim:
        raise ValueError("final stage must use the full requested nsim")
    if sum(s.maxiter for s in stages) != maxiter:
        raise ValueError("stage iteration budgets must sum to maxiter")
    return stages


def minimize_staged(loss: Callable, x0, *, nsim, maxiter, tol,
                    schedule: Sequence[TrialStage] | None = None,
                    on_update=None, on_stage=None):
    """loss(params, stage_nsim) must use the same fixed data and root seed.

    Carry only parameter guesses forward. Rebuild the simplex and recompute
    its losses at every new trial count; never mix losses from two budgets.
    The final stage performs optimization at full nsim, not just a final check.
    """
    from scipy.optimize import minimize

    stages = validate_schedule(nsim, maxiter, schedule)
    params = np.asarray(x0, dtype=float)
    evaluations = 0
    iterations = 0
    history = []
    for index, stage in enumerate(stages):
        initial_evals = evaluations

        def objective(p):
            nonlocal evaluations
            evaluations += 1
            value = float(loss(p, stage.nsim))
            if on_update is not None:
                on_update(evaluations, value, p.copy())
            return value

        result = minimize(
            objective, params, method="Nelder-Mead",
            options={"maxiter": stage.maxiter, "xatol": tol,
                     "fatol": tol, "disp": False},
        )
        params = result.x.copy()
        iterations += int(result.nit)
        details = {
            "stage": index, "nsim": stage.nsim,
            "maxiter": stage.maxiter, "n_iters": int(result.nit),
            "n_evals": evaluations - initial_evals, "loss": float(result.fun),
            "converged": bool(result.success),
        }
        history.append(details)
        if on_stage is not None:
            on_stage(dict(details))
    return SimplexResult(params, float(result.fun), iterations, evaluations,
                         bool(result.success), history)
