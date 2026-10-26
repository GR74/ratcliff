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
                  stable_kl_noise=False, separable_kl=False, soft_tau_ms=0.0,
                  maxfev=None, fortran_variant=None):
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
    sv is inert only in the original port: under fortran_variant ("bench" or
    "fit", see objective.fofs_b_new) it is active and always kept.
    kl_bucketed/cumulative_kl are optional blocked K-L simulation extensions.
    separable_kl evaluates the same K-L field in factored row x column form.
    soft_tau_ms > 0 fits the soft-count G2 (see objective.condition_g2_b).
    """
    template = np.asarray(x0, dtype=float).copy()
    if template.shape != (13,):
        raise ValueError("Model B x0 must have 13 parameters")
    optimize_active_only = optimize_active_only and fortran_variant is None
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
            soft_tau_ms=soft_tau_ms, fortran_variant=fortran_variant,
        )

    def progress(number, loss, params):
        if on_update is not None:
            on_update(number, loss, expand(params).copy())

    res = minimize_staged(
        loss_numpy, template[active] if optimize_active_only else template,
        nsim=nsim, maxiter=maxiter, tol=tol,
        schedule=trial_schedule, on_update=progress, on_stage=on_stage,
        maxfev=maxfev,
    )
    return FitResult(
        params=jnp.asarray(expand(res.params)), loss=res.loss, n_iters=res.n_iters,
        converged=res.converged, n_evals=res.n_evals, stages=res.stages,
    )


# Search units for CMA-ES, one per parameter (layout as in objective.clamp_b):
# roughly the size of a meaningful change, so one step size fits them all.
CMAES_SCALE_B = np.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 0.1,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# clamp_b's fixed bounds (st <= 1.5 ter and crsd <= cr/2 stay with clamp_b).
LOWER_B = np.array([175.0, 10.0, 1.0, 0.01, -np.inf, 0.2, 0.2] + [0.01] * 6)
UPPER_B = np.array([np.inf] * 5 + [17.0] + [np.inf] * 7)


def fit_cmaes_b(data, key, x0, nsim: int = 2048, max_evals: int = 2000,
                seed: int = 0, popsize=None, sigma0: float = 0.5, restarts: int = 0,
                chunk_size: int = 64, time_block_size: int = 32,
                stable_kl_noise: bool = True, separable_kl: bool = True,
                soft_tau_ms: float = 0.0, optimize_active_only: bool = True,
                on_update=None, on_generation=None, use_kl: bool = True,
                fortran_variant=None):
    """CMA-ES fit for Model B with the same objective as fit_simplex_b.

    key is the simulator root key (fixed across evaluations); seed drives the
    CMA-ES sampling only. Returns FitResult; n_iters counts generations and
    stages holds one summary per (IPOP) run. use_kl=False uses the exact
    circulant (FFT) field. Under fortran_variant sv is active and always kept
    in the search, and "fit" lowers the ter bound to 150 as that program does.
    """
    from shared.cmaes_fit import minimize_cmaes

    template = np.asarray(x0, dtype=float).copy()
    if template.shape != (13,):
        raise ValueError("Model B x0 must have 13 parameters")
    optimize_active_only = optimize_active_only and fortran_variant is None
    active = (np.asarray([i for i in range(13) if i != 6]) if optimize_active_only
              else np.arange(13))
    lower = LOWER_B.copy()
    lower[0] = obj_b.TER_MIN_MS.get(fortran_variant, lower[0])

    def expand(p_np):
        full = template.copy()
        full[active] = p_np
        return full

    def loss_numpy(p_np, stage_nsim):
        return float(obj_b.fofs_b_new(
            jnp.asarray(expand(p_np)), data, key, nsim=stage_nsim,
            chunk_size=chunk_size, use_kl=use_kl, time_block_size=time_block_size,
            stable_kl_noise=stable_kl_noise and use_kl,
            separable_kl=separable_kl and use_kl,
            soft_tau_ms=soft_tau_ms, fortran_variant=fortran_variant))

    def progress(number, loss, params):
        if on_update is not None:
            on_update(number, loss, expand(params).copy())

    res = minimize_cmaes(
        loss_numpy, template[active], CMAES_SCALE_B[active], nsim=nsim,
        max_evals=max_evals, seed=seed, popsize=popsize, sigma0=sigma0,
        restarts=restarts, lower=lower[active], upper=UPPER_B[active],
        on_update=progress, on_generation=on_generation)
    return FitResult(params=jnp.asarray(expand(res.params)), loss=res.loss,
                     n_iters=res.n_generations, converged=res.converged,
                     n_evals=res.n_evals, stages=res.restarts)
