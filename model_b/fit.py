"""
Fit drivers for Model B.

Simplex-only (Nelder-Mead via scipy). L-BFGS is deferred until Stage 3.5
resolves the gradient-zero issue (jax.grad(fofs_b_new) is structurally zero
due to discrete categorization and indicator-CDF in the objective).
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from model_b import objective as obj_b
from shared.staged_fit import minimize_staged


@dataclass
class FitResult:
    params: jnp.ndarray
    loss: float
    n_iters: int
    converged: bool
    n_evals: int = 0
    stages: list[dict] | None = None


def fit_simplex_b(data, key, x0, nsim: int = 256, maxiter: int = 2000,
                  tol: float = 1e-7, chunk_size: int = 4, use_kl: bool = False,
                  on_update=None, trial_schedule=None, time_block_size=0,
                  stop_early=True, compact=True, on_stage=None,
                  optimize_active_only=False, kl_bucketed=False, cumulative_kl=False,
                  stable_kl_noise=False, separable_kl=False):
    """
    Scipy Nelder-Mead simplex fit for Model B.

    data       : dict with "prop" (2, 5), "count" (2, 5), "quant" (2, 5, 5).
    key        : JAX PRNG key for the Monte Carlo simulator.
    x0         : (13,) initial parameter vector.
    nsim       : trials per condition.
    maxiter    : scipy NM maxiter.
    tol        : scipy NM x/f tolerance.
    chunk_size : trial chunk for simulate_b (small for laptop, larger for H100).
    use_kl     : if True, use Stage 6 K-L low-rank GRF inside fofs_b_new.
    on_update  : optional callable(eval_n, loss, x_array) invoked after every
                 function evaluation. Enables live progress reporting from the
                 UI without modifying scipy.

    Returns FitResult.
    optimize_active_only removes the inert sv coordinate from the search while
    preserving the public 13-parameter result/callbacks and its initial sv value.
    kl_bucketed/cumulative_kl are optional blocked K-L simulation extensions.
    separable_kl evaluates the same K-L field in factored row x column form.
    """
    template = np.asarray(x0, dtype=float).copy()
    if template.shape != (13,):
        raise ValueError("Model B x0 must have 13 parameters")
    active = np.asarray([i for i in range(13) if i != 6])

    def expand(p_np):
        if not optimize_active_only:
            return np.asarray(p_np)
        full = template.copy()
        full[active] = p_np
        return full

    def loss_numpy(p_np, stage_nsim):
        p = jnp.asarray(expand(p_np))
        return obj_b.fofs_b_new(
            p, data, key, nsim=stage_nsim, chunk_size=chunk_size, use_kl=use_kl,
            time_block_size=time_block_size, stop_early=stop_early, compact=compact,
            kl_bucketed=kl_bucketed, cumulative_kl=cumulative_kl,
            stable_kl_noise=stable_kl_noise, separable_kl=separable_kl,
        )

    def progress(number, loss, params):
        if on_update is not None:
            on_update(number, loss, expand(params).copy())

    res = minimize_staged(
        loss_numpy, template[active] if optimize_active_only else template,
        nsim=nsim, maxiter=maxiter, tol=tol,
        schedule=trial_schedule, on_update=progress, on_stage=on_stage,
    )
    return FitResult(
        params=jnp.asarray(expand(res.params)), loss=res.loss, n_iters=res.n_iters,
        converged=res.converged, n_evals=res.n_evals, stages=res.stages,
    )
