"""
Fit drivers for Model A.

Three optimizers:
- `fit_simplex`        : scipy Nelder-Mead on the discrete `fofs_new` (gradient-free).
- `fit_lbfgs_smooth`   : jaxopt L-BFGS on the smooth surrogate `fofs_smooth`.
- `fit_hybrid`         : L-BFGS-smooth coarse pass → simplex polish on discrete fofs.

The hybrid is the recommended default: gets into the basin via cheap gradient
descent on the smooth surrogate, then refines on the true discrete objective.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from model_a import objective, objective_smooth


@dataclass
class FitResult:
    params: jnp.ndarray
    loss: float
    n_iters: int
    converged: bool
    optimizer: str


def fit_simplex(data, key, x0, nsim: int = 256, maxiter: int = 2000,
                tol: float = 1e-7, chunk_size: int = 256):
    """Scipy Nelder-Mead simplex on the discrete `fofs_new`."""
    from scipy.optimize import minimize

    def loss_np(p_np):
        p = jnp.asarray(p_np)
        return float(objective.fofs_new(p, data, key, nsim=nsim, chunk_size=chunk_size))

    res = minimize(
        loss_np, np.asarray(x0),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": tol, "fatol": tol, "disp": False},
    )
    return FitResult(
        params=jnp.asarray(res.x),
        loss=float(res.fun),
        n_iters=int(res.nit),
        converged=bool(res.success),
        optimizer="simplex",
    )


def fit_lbfgs_smooth(data, key, x0, nsim: int = 512, maxiter: int = 100,
                     tol: float = 1e-5, chunk_size: int = 256,
                     tau_step: float = 2.0, tau_pos: float = 20.0,
                     sigma_cdf: float = 50.0):
    """jaxopt L-BFGS on the smooth surrogate `fofs_smooth`."""
    import jaxopt

    def loss_fn(p):
        return objective_smooth.fofs_smooth(
            p, data, key, nsim=nsim, chunk_size=chunk_size,
            tau_step=tau_step, tau_pos=tau_pos, sigma_cdf=sigma_cdf,
        )

    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=maxiter, tol=tol)
    res = solver.run(x0)
    return FitResult(
        params=res.params,
        loss=float(res.state.value),
        n_iters=int(res.state.iter_num),
        converged=bool(res.state.error < tol),
        optimizer="lbfgs_smooth",
    )


def fit_hybrid(data, key, x0, nsim: int = 512,
               lbfgs_maxiter: int = 50, polish_maxiter: int = 200,
               polish_tol: float = 1e-5, chunk_size: int = 256,
               **smooth_kwargs):
    """
    Hybrid fit: smooth L-BFGS coarse pass → discrete simplex polish.

    The smooth-L-BFGS finds a basin cheaply via gradient descent on the
    biased smooth objective; the simplex polish then refines on the true
    discrete `fofs_new`. Best wall-clock of the three drivers.
    """
    coarse = fit_lbfgs_smooth(
        data, key, x0,
        nsim=nsim, maxiter=lbfgs_maxiter, chunk_size=chunk_size,
        **smooth_kwargs,
    )
    polish = fit_simplex(
        data, key, coarse.params,
        nsim=nsim, maxiter=polish_maxiter, tol=polish_tol, chunk_size=chunk_size,
    )
    return FitResult(
        params=polish.params,
        loss=polish.loss,
        n_iters=coarse.n_iters + polish.n_iters,
        converged=polish.converged,
        optimizer="hybrid",
    )
