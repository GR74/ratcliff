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


@dataclass
class FitResult:
    params: jnp.ndarray
    loss: float
    n_iters: int
    converged: bool


def fit_simplex_b(data, key, x0, nsim: int = 256, maxiter: int = 2000,
                  tol: float = 1e-7, chunk_size: int = 4, use_kl: bool = False,
                  on_update=None):
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
    """
    from scipy.optimize import minimize

    n_evals = [0]

    def loss_numpy(p_np):
        n_evals[0] += 1
        p = jnp.asarray(p_np)
        val = obj_b.fofs_b_new(p, data, key, nsim=nsim,
                                chunk_size=chunk_size, use_kl=use_kl)
        val_f = float(val)
        if on_update is not None:
            on_update(n_evals[0], val_f, p_np.copy())
        return val_f

    res = minimize(
        loss_numpy, np.asarray(x0),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": tol, "fatol": tol, "disp": False},
    )
    return FitResult(
        params=jnp.asarray(res.x),
        loss=float(res.fun),
        n_iters=int(res.nit),
        converged=bool(res.success),
    )
