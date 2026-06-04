"""Smoke tests for model_b/simulate.py."""
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import simulate as sim_b


def test_simulate_b_exposes_constants():
    assert sim_b.N == 100
    assert sim_b.M == 160
    assert sim_b.NSTEP == 400
    assert sim_b.E == 10.0


def test_drift_bumps_shape_and_peak():
    """Three drift Gaussians at (uj1=80, uj2=30, uj3=130), all with ui=50."""
    v1, v2, v3 = sim_b.drift_bumps(sis=12.0)
    assert v1.shape == (100, 160)
    assert v2.shape == (100, 160)
    assert v3.shape == (100, 160)
    # Peaks at the documented positions (row 50, col 80/30/130)
    p1 = int(jnp.argmax(v1))
    p2 = int(jnp.argmax(v2))
    p3 = int(jnp.argmax(v3))
    assert p1 // 160 == 50 and p1 % 160 == 80, f"v1 peak at flat {p1}"
    assert p2 // 160 == 50 and p2 % 160 == 30, f"v2 peak at flat {p2}"
    assert p3 // 160 == 50 and p3 % 160 == 130, f"v3 peak at flat {p3}"


def test_zone_array_has_5_categories():
    """k(i,j) classifies positions into {1, 2, 3, 4, 5}."""
    k = sim_b.zone_array(si=6.0)
    assert k.shape == (100, 160)
    unique = set(int(x) for x in jnp.unique(k))
    assert unique == {1, 2, 3, 4, 5}


def test_zone_array_cat1_inside_cat2_inside_5():
    """Category 1 (innermost ring) is at the very center; cat 2 surrounds it."""
    k = sim_b.zone_array(si=6.0)
    # Cell at (50, 80) should be in cat 1 (innermost ring around UJ1)
    assert int(k[50, 80]) == 1
    # Cell far from any bump (e.g., (0, 0)) should be cat 5
    assert int(k[0, 0]) == 5


def test_simulate_chunk_b_returns_shapes():
    from model_b import grf
    import jax
    LAM = grf.calc_LAM(s1=10.0, s2=10.0)
    v1, v2, v3 = sim_b.drift_bumps(sis=12.0)
    k_zone = sim_b.zone_array(si=6.0)
    key = jax.random.key(0)
    rt, cat = sim_b._simulate_chunk_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        LAM=LAM, v1=v1, v2=v2, v3=v3, k_zone=k_zone,
        chunk_size=2,
    )
    assert rt.shape == (2,)
    assert cat.shape == (2,)
    assert jnp.all(jnp.isfinite(rt))


def test_simulate_chunk_b_cat_in_valid_range():
    from model_b import grf
    import jax
    LAM = grf.calc_LAM(s1=10.0, s2=10.0)
    v1, v2, v3 = sim_b.drift_bumps(sis=12.0)
    k_zone = sim_b.zone_array(si=6.0)
    key = jax.random.key(0)
    _, cat = sim_b._simulate_chunk_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        LAM=LAM, v1=v1, v2=v2, v3=v3, k_zone=k_zone,
        chunk_size=4,
    )
    # cat in {1..5}
    assert jnp.all((cat >= 1) & (cat <= 5))


def test_simulate_chunk_b_deterministic():
    from model_b import grf
    import jax
    LAM = grf.calc_LAM(s1=10.0, s2=10.0)
    v1, v2, v3 = sim_b.drift_bumps(sis=12.0)
    k_zone = sim_b.zone_array(si=6.0)
    key = jax.random.key(42)
    rt_a, _ = sim_b._simulate_chunk_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        LAM=LAM, v1=v1, v2=v2, v3=v3, k_zone=k_zone,
        chunk_size=2,
    )
    rt_b, _ = sim_b._simulate_chunk_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        LAM=LAM, v1=v1, v2=v2, v3=v3, k_zone=k_zone,
        chunk_size=2,
    )
    np.testing.assert_array_equal(rt_a, rt_b)
