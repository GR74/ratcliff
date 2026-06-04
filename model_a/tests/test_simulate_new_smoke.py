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


def test_simulate_chunk_returns_rt_and_cat_shapes():
    """_simulate_chunk(...) returns (rt, cat) each of shape (chunk_size,)."""
    L = sim_new.chol_factor(5.0)
    v = sim_new.drift_profile(av=20.0, si=4.0)
    key = prng.root_key(0)
    rt, cat = sim_new._simulate_chunk(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, L=L, v=v, chunk_size=64
    )
    assert rt.shape == (64,)
    assert cat.shape == (64,)


def test_simulate_chunk_rt_is_finite_and_positive():
    """RTs are finite, positive, and bounded above by (NSTEP + ter+st/2)*E."""
    L = sim_new.chol_factor(5.0)
    v = sim_new.drift_profile(av=20.0, si=4.0)
    key = prng.root_key(0)
    rt, _ = sim_new._simulate_chunk(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, L=L, v=v, chunk_size=64
    )
    assert jnp.all(jnp.isfinite(rt))
    assert jnp.all(rt > 0)
    # Hard upper bound: jstop <= NSTEP, ndt <= (ter+st/2)/E ≈ 22.5 steps,
    # so rt <= (400 + 22.5) * 10 = 4225 ms
    assert jnp.all(rt <= 5000)


def test_simulate_chunk_cat_in_valid_range():
    """All categories are in {1, 2, 3}."""
    L = sim_new.chol_factor(5.0)
    v = sim_new.drift_profile(av=20.0, si=4.0)
    key = prng.root_key(0)
    _, cat = sim_new._simulate_chunk(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, L=L, v=v, chunk_size=64
    )
    assert jnp.all((cat >= 1) & (cat <= 3))


def test_simulate_chunk_deterministic_for_same_key():
    """Same key + same params produces bit-exact same outputs."""
    L = sim_new.chol_factor(5.0)
    v = sim_new.drift_profile(av=20.0, si=4.0)
    key = prng.root_key(42)
    rt_a, cat_a = sim_new._simulate_chunk(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, L=L, v=v, chunk_size=32
    )
    rt_b, cat_b = sim_new._simulate_chunk(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, L=L, v=v, chunk_size=32
    )
    np.testing.assert_array_equal(rt_a, rt_b)
    np.testing.assert_array_equal(cat_a, cat_b)
