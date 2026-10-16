"""CMA-ES alternative to minimize_staged (pass 3 optimizer bake-off).

Same loss contract as shared.staged_fit: loss(params, nsim) with fixed data
and a fixed simulator root seed (common random numbers). CMA-ES ranks a whole
population per generation, so it tolerates the rough, piecewise-constant G2
surface that stalls Nelder-Mead's simplex.

Search runs in scaled coordinates z = (x - x0) / scale, so one step size suits
parameters measured in ms and in drift units alike. Restarts (IPOP) double the
population from the best point found so far, each with its own seeded key.
Population members are evaluated one by one: each has its own sig, so they
cannot share one noise basis in a single vmap.
"""
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class CMAESResult:
    params: np.ndarray
    loss: float
    n_evals: int
    n_generations: int
    restarts: list[dict] = field(default_factory=list)
    converged: bool = False


def minimize_cmaes(loss: Callable, x0, scale, *, nsim, max_evals, seed=0,
                   popsize=None, sigma0=0.3, restarts=0, sigma_tol=1e-3,
                   lower=None, upper=None, penalty=1e3, on_update=None,
                   on_generation=None):
    """Minimise loss(params, nsim) with (IPOP-)CMA-ES.

    scale     : per-parameter unit for the search (e.g. a plausible error size).
    sigma0    : initial step size in units of scale.
    popsize   : default 4 + floor(3 ln d) (Hansen's default).
    restarts  : extra IPOP runs, each with twice the previous population.
    sigma_tol : stop a run when its step size falls below this (scaled units).
    lower/upper : optional box. Out-of-box points are scored at the clipped
        point plus penalty * squared scaled distance, which keeps the search
        distribution inside the box without changing the loss there.
    """
    from evosax.algorithms import CMA_ES

    x0 = np.asarray(x0, dtype=float)
    scale = np.asarray(scale, dtype=float)
    d = x0.size
    lower = np.full(d, -np.inf) if lower is None else np.asarray(lower, float)
    upper = np.full(d, np.inf) if upper is None else np.asarray(upper, float)
    popsize = popsize or 4 + int(3 * np.log(d))

    evaluations = 0
    best_x, best_f = x0.copy(), np.inf
    generations = 0
    history = []

    def score(x):
        nonlocal evaluations, best_x, best_f
        clipped = np.clip(x, lower, upper)
        excess = np.sum(((x - clipped) / scale) ** 2)
        value = float(loss(clipped, nsim)) + penalty * excess
        evaluations += 1
        if value < best_f:
            best_x, best_f = clipped.copy(), value
        if on_update is not None:
            on_update(evaluations, value, clipped.copy())
        return value

    key = jax.random.key(seed)
    start = x0.copy()
    for run in range(restarts + 1):
        if evaluations >= max_evals:
            break
        lam = popsize * 2 ** run
        es = CMA_ES(population_size=lam, solution=jnp.zeros(d))
        params = es.default_params.replace(std_init=sigma0)
        key, k_init = jax.random.split(key)
        state = es.init(k_init, jnp.asarray((start - x0) / scale, jnp.float32), params)
        run_evals0, stop = evaluations, "budget"
        while evaluations + lam <= max_evals:
            key, k_ask, k_tell = jax.random.split(key, 3)
            z, state = es.ask(k_ask, state, params)
            xs = x0 + np.asarray(z, float) * scale
            fitness = jnp.asarray([score(x) for x in xs], jnp.float32)
            state, _ = es.tell(k_tell, z, fitness, state, params)
            generations += 1
            std = float(state.std)
            if on_generation is not None:
                on_generation(dict(run=run, generation=generations, std=std,
                                   best_loss=best_f, gen_min=float(fitness.min()),
                                   n_evals=evaluations))
            if std < sigma_tol:
                stop = "sigma_tol"
                break
        history.append(dict(run=run, popsize=lam, n_evals=evaluations - run_evals0,
                            best_loss=best_f, stop=stop))
        start = best_x.copy()
    converged = bool(history) and history[-1]["stop"] == "sigma_tol"
    return CMAESResult(best_x, best_f, evaluations, generations, history, converged)
