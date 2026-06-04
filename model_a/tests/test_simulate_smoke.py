import jax
import jax.numpy as jnp
import pytest

from model_a import jax_port
from shared import prng


def test_simulate_returns_finite_rt_and_categories():
    key = prng.root_key(0)
    rt, cat = jax_port.simulate(
        key,
        ter=200.0,
        st=50.0,
        cr=50.0,
        crsd=10.0,
        si=4.0,
        sig=5.0,
        av=20.0,
        sv=0.7,
        nsim=64,
    )
    assert rt.shape == (64,)
    assert cat.shape == (64,)
    assert jnp.all(jnp.isfinite(rt))
    # Categories are {1, 2, 3} per the Fortran convention
    assert jnp.all((cat >= 1) & (cat <= 3))


def test_simulate_is_deterministic_for_same_key():
    key = prng.root_key(42)
    a_rt, a_cat = jax_port.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, si=4.0, sig=5.0, av=20.0, sv=0.7, nsim=32
    )
    b_rt, b_cat = jax_port.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0, si=4.0, sig=5.0, av=20.0, sv=0.7, nsim=32
    )
    assert jnp.array_equal(a_rt, b_rt)
    assert jnp.array_equal(a_cat, b_cat)


def test_simulate_differs_for_different_keys():
    rt_a, _ = jax_port.simulate(
        prng.root_key(0),
        ter=200.0, st=50.0, cr=50.0, crsd=10.0, si=4.0, sig=5.0, av=20.0, sv=0.7, nsim=32,
    )
    rt_b, _ = jax_port.simulate(
        prng.root_key(1),
        ter=200.0, st=50.0, cr=50.0, crsd=10.0, si=4.0, sig=5.0, av=20.0, sv=0.7, nsim=32,
    )
    assert not jnp.array_equal(rt_a, rt_b)
