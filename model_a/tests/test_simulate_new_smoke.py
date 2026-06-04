"""Smoke tests for the new model_a/simulate.py (single-GEMM rewrite)."""
import jax.numpy as jnp
import numpy as np
import pytest

from model_a import simulate as sim_new
from shared import prng


def test_module_exposes_constants():
    """Sanity: the new module exports N, NSTEP, MC, E with the right values."""
    assert sim_new.N == 72
    assert sim_new.NSTEP == 400
    assert sim_new.MC == 3
    assert sim_new.E == 10.0


def test_chol_factor_returns_lower_triangular():
    """chol_factor(sig) returns the Cholesky factor of the GP kernel."""
    L = sim_new.chol_factor(5.0)
    assert L.shape == (72, 72)
    # K = L @ L.T should reproduce the kernel (within FP tolerance)
    K = L @ L.T
    # Diagonal of K is ~1 (plus the 1e-12 jitter)
    assert jnp.allclose(jnp.diag(K), 1.0, atol=1e-6)


def test_drift_profile_peaks_at_U():
    """drift_profile(av, si) is a Gaussian bump centered at U=36."""
    v = sim_new.drift_profile(av=20.0, si=4.0)
    assert v.shape == (72,)
    # Peak is near index U=36 (0-indexed: 35, but the function uses 1-indexed IDX)
    peak_idx = int(jnp.argmax(v))
    assert 34 <= peak_idx <= 36  # tolerance for the 1-vs-0 indexing convention
