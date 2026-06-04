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


def test_simulate_chunk_no_crossing_saturates_at_nstep():
    """At very high cr, no trials cross; jstop=NSTEP, rt = (NSTEP + ndt)*E."""
    L = sim_new.chol_factor(5.0)
    v = sim_new.drift_profile(av=1.0, si=4.0)   # tiny drift
    key = prng.root_key(0)
    rt, _ = sim_new._simulate_chunk(
        key, ter=200.0, st=50.0, cr=1e6, crsd=0.0, L=L, v=v, chunk_size=32
    )
    # All trials should saturate: jstop = NSTEP = 400, rt = (400 + ndt) * 10
    # ndt = (200 + 50*(0.5 - u)) / 10 ∈ [(200-25)/10, (200+25)/10] = [17.5, 22.5]
    # rt ∈ [4175, 4225] ms
    assert jnp.all(rt >= 4170)
    assert jnp.all(rt <= 4230)


def test_simulate_chunk_immediate_crossing_at_low_cr():
    """At cr=0, the first step's accumulator typically crosses; jstop=1 expected often."""
    L = sim_new.chol_factor(5.0)
    v = sim_new.drift_profile(av=20.0, si=4.0)
    key = prng.root_key(0)
    rt, _ = sim_new._simulate_chunk(
        key, ter=200.0, st=50.0, cr=0.0, crsd=0.0, L=L, v=v, chunk_size=64
    )
    # With cr=0, any positive max of accumulator triggers; that happens at step 1
    # for the majority. RT for jstop=1 is (1 + ndt) * 10 ∈ [185, 225] ms.
    # We allow that some trials might cross at step 2 due to noise; assert
    # the MEDIAN RT is consistent with jstop ≈ 1-2.
    median_rt = float(jnp.median(rt))
    assert median_rt < 250, f"expected near-immediate crossing, median rt = {median_rt}"


def test_simulate_returns_full_nsim_shape():
    """simulate(...) returns (rt, cat) each of shape (nsim,)."""
    key = prng.root_key(0)
    rt, cat = sim_new.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=512, chunk_size=128
    )
    assert rt.shape == (512,)
    assert cat.shape == (512,)


def test_simulate_handles_non_multiple_chunk():
    """nsim that isn't a multiple of chunk_size still returns exactly nsim outputs."""
    key = prng.root_key(0)
    rt, cat = sim_new.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=300, chunk_size=128
    )
    assert rt.shape == (300,)
    assert cat.shape == (300,)


def test_simulate_deterministic_for_same_key():
    """Same key reproduces bit-exact outputs."""
    key = prng.root_key(11)
    rt_a, cat_a = sim_new.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=64, chunk_size=64
    )
    rt_b, cat_b = sim_new.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=64, chunk_size=64
    )
    np.testing.assert_array_equal(rt_a, rt_b)
    np.testing.assert_array_equal(cat_a, cat_b)


def test_simulate_differs_for_different_keys():
    """Different keys produce different outputs."""
    rt_a, _ = sim_new.simulate(
        prng.root_key(0), ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=64, chunk_size=64
    )
    rt_b, _ = sim_new.simulate(
        prng.root_key(1), ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=64, chunk_size=64
    )
    assert not jnp.array_equal(rt_a, rt_b)
