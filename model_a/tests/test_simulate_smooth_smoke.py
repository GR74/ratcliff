"""Smoke tests for model_a/simulate_smooth.py."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_a import simulate_smooth as sim_sm
from shared import prng


def test_simulate_smooth_returns_shapes():
    """simulate_smooth(...) returns (rt, cat_probs) with right shapes."""
    key = prng.root_key(0)
    rt, cat_probs = sim_sm.simulate_smooth(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0,
        nsim=32, chunk_size=32,
        tau_step=0.5, tau_pos=2.0,
    )
    assert rt.shape == (32,)
    assert cat_probs.shape == (32, 3)
    # cat_probs should sum to ~1 across categories (with floor tolerance)
    sums = cat_probs.sum(axis=1)
    np.testing.assert_allclose(np.asarray(sums), np.ones(32), atol=0.05)


def test_simulate_smooth_rt_is_finite_and_positive():
    key = prng.root_key(0)
    rt, _ = sim_sm.simulate_smooth(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=32, chunk_size=32,
        tau_step=0.5, tau_pos=2.0,
    )
    assert jnp.all(jnp.isfinite(rt))
    assert jnp.all(rt > 0)


def test_simulate_smooth_cat_probs_are_valid():
    """cat_probs are nonneg and finite."""
    key = prng.root_key(0)
    _, cat_probs = sim_sm.simulate_smooth(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=32, chunk_size=32,
        tau_step=0.5, tau_pos=2.0,
    )
    assert jnp.all(jnp.isfinite(cat_probs))
    assert jnp.all(cat_probs >= 0)
    assert jnp.all(cat_probs <= 1.0)


def test_simulate_smooth_deterministic():
    key = prng.root_key(42)
    rt_a, _ = sim_sm.simulate_smooth(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=16, chunk_size=16,
        tau_step=0.5, tau_pos=2.0,
    )
    rt_b, _ = sim_sm.simulate_smooth(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, nsim=16, chunk_size=16,
        tau_step=0.5, tau_pos=2.0,
    )
    np.testing.assert_array_equal(rt_a, rt_b)
