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


def _simulate_chunk(key, ter, st, cr, crsd, L, v, chunk_size):
    """
    Simulate `chunk_size` trials in one fused XLA program.

    Returns (rt, cat) each of shape (chunk_size,).
    """
    ku, kz = jax.random.split(key)

    # Per-trial uniforms (gu1 in the Fortran)
    u = jax.random.uniform(ku, (chunk_size, 10))
    crr = cr + crsd * (u[:, 4] - 0.5)                  # (chunk_size,) per-trial threshold
    ndt = (ter + st * (0.5 - u[:, 9])) / E             # (chunk_size,) per-trial nondecision in steps

    # One big GEMM: all noise for chunk × NSTEP × N
    z = jax.random.normal(kz, (chunk_size, NSTEP, N))  # iid normals
    noise = z @ L.T                                     # (chunk_size, NSTEP, N) correlated normals

    # Build demeaned per-step increments
    incr = v[None, None, :] + 5.0 * noise               # broadcast drift bump + scaled noise
    incr = incr - incr.mean(axis=-1, keepdims=True)     # demean per (trial, step)

    # Accumulator paths
    a = jnp.cumsum(incr, axis=1)                        # (chunk_size, NSTEP, N)

    # First crossing
    max_per_step = a.max(axis=-1)                       # (chunk_size, NSTEP)
    crossed = max_per_step > crr[:, None]               # (chunk_size, NSTEP) bool
    any_crossed = crossed.any(axis=1)                   # (chunk_size,)
    # argmax of bool returns the first True; if none, returns 0, hence the where
    jstop = jnp.where(any_crossed, jnp.argmax(crossed, axis=1) + 1, NSTEP)

    # Position at crossing (or at NSTEP if never crossed)
    pos = jnp.argmax(a[jnp.arange(chunk_size), jstop - 1, :], axis=-1) + 1

    # RT in ms; categorize by position band (mirrors jax_port.one_trial)
    rt = (jstop + ndt) * E
    cat = jnp.where((pos > IPA) & (pos < IPB), 1,
          jnp.where((pos <= IPC) | (pos >= IPD), 3, 2))
    return rt, cat
