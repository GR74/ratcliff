"""Tests for simulate_b's use_kl flag (Stage 6 K-L path)."""
import jax
import jax.numpy as jnp
import numpy as np

from model_b import simulate as sim_b


def test_simulate_b_accepts_use_kl_flag():
    """simulate_b should accept use_kl=False and behave identically to current."""
    key = jax.random.key(0)
    rt_a, cat_a = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=16, chunk_size=4, use_kl=False,
    )
    rt_b, cat_b = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=16, chunk_size=4,  # use_kl defaults to False
    )
    np.testing.assert_array_equal(rt_a, rt_b)
    np.testing.assert_array_equal(cat_a, cat_b)


def test_simulate_b_kl_path_runs_and_returns_finite():
    """use_kl=True should run end-to-end without errors."""
    key = jax.random.key(0)
    rt, cat = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=16, chunk_size=4, use_kl=True,
    )
    assert rt.shape == (16,)
    assert cat.shape == (16,)
    assert jnp.all(jnp.isfinite(rt))
    assert jnp.all((cat >= 1) & (cat <= 5))


def test_simulate_b_kl_deterministic_for_same_key():
    """K-L path with same key should produce same output."""
    key = jax.random.key(7)
    rt_a, cat_a = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=8, chunk_size=4, use_kl=True,
    )
    rt_b, cat_b = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=8, chunk_size=4, use_kl=True,
    )
    np.testing.assert_array_equal(rt_a, rt_b)
    np.testing.assert_array_equal(cat_a, cat_b)
