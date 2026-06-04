"""
Fast Monte Carlo simulator for the Ratcliff 1D spatially-extended diffusion.

Rewrite of `model_a/jax_port.simulate`: pre-generates the full noise block per
chunk, applies the Cholesky factor as one big GEMM, demeans per step, cumsums
along time, and finds the first crossing via argmax. No per-step lax.scan
state; XLA fuses the whole chunk into a single program.

Memory: at default chunk_size=256, peak working set is ~240 MB for N=72,
NSTEP=400, fp64. Tunable via the static `chunk_size` argument.
"""
from functools import partial

import jax
import jax.numpy as jnp

from shared import prng

# ----------------------------------------------------------------------
# Fixed model structure (mirrors jax_port.py exactly)
# ----------------------------------------------------------------------
N = 72
NSTEP = 400
E = 10.0
U = 180.0 / 5.0
MC = 3
IPA, IPB, IPC, IPD = 150 / 5, 210 / 5, 100 / 5, 260 / 5  # 30, 42, 20, 52
IDX = jnp.arange(1, N + 1, dtype=jnp.float64)


def chol_factor(sig):
    """Cholesky factor L of the GP kernel K; noise = L @ z ~ N(0, K)."""
    d = IDX[:, None] - IDX[None, :]
    K = jnp.exp(-0.5 * d * d / (sig * sig)) + 1e-12 * jnp.eye(N)
    return jnp.linalg.cholesky(K)


def drift_profile(av, si):
    """Spatial drift bump v(i) = av * Normal(i; U, si)."""
    return av * jnp.exp(-(IDX - U) ** 2 / (2.0 * si * si)) / (si * jnp.sqrt(2.0 * jnp.pi))
