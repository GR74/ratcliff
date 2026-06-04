"""Smoke tests for model_b/objective.py."""
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import objective as obj_b


def test_clamp_b_floors():
    """Bounds match Fortran fofs in benchtwod3mpi: ter>=175, st in [10, 1.5*ter], cr>=1, ..."""
    p = obj_b.clamp_b(jnp.array([100., 5., 0.5, 0.0, 1.0, 0.1, 0.1, 0., 0., 0., 0., 0., 0.]))
    assert float(p[0]) >= 175.0  # ter floor
    assert float(p[1]) >= 10.0   # st floor
    assert float(p[2]) >= 1.0    # cr floor (a1)
    assert float(p[3]) >= 0.01   # crsd (sa) floor
    assert float(p[5]) >= 0.2 and float(p[5]) <= 17.0  # sig clipped to [0.2, 17.0]
    assert float(p[6]) >= 0.2    # sv floor
    # Drift params (indices 7-12)
    for i in range(7, 13):
        assert float(p[i]) >= 0.01


def test_clamp_b_passes_valid_params_through():
    """Already-valid params should be unchanged."""
    x = jnp.array([200., 50., 10., 2., 12., 10., 0.5, 15., 10., 8., 14., 11., 9.])
    p = obj_b.clamp_b(x)
    np.testing.assert_array_almost_equal(np.asarray(p), np.asarray(x))


def test_condition_g2_b_returns_finite_scalar():
    """condition_g2_b returns a finite scalar G² value."""
    import jax
    from model_b import simulate as sim_b
    key = jax.random.key(0)
    rt, cat = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=64, chunk_size=8,
    )
    # Synthetic observed-data for ONE condition (5 categories)
    obs_prop = jnp.array([0.3, 0.2, 0.15, 0.2, 0.15])
    obs_count = jnp.array([30, 20, 15, 20, 15], dtype=jnp.int64)
    obs_quant = jnp.array([
        [300., 310., 320., 330., 340.],
        [340., 350., 360., 370., 380.],
        [380., 390., 400., 410., 420.],
        [420., 430., 440., 450., 460.],
        [460., 470., 480., 490., 500.],
    ])  # shape (5 quantiles, 5 categories)
    val = float(obj_b.condition_g2_b(rt, cat, obs_prop, obs_count, obs_quant))
    assert np.isfinite(val)
    assert val > 0


def test_condition_g2_b_handles_5_categories():
    """The result should sum 5 per-category contributions."""
    import jax
    from model_b import simulate as sim_b
    key = jax.random.key(1)
    rt, cat = sim_b.simulate_b(
        key, ter=200.0, st=50.0, cr=10.0, crsd=2.0,
        av1=15.0, av2=10.0, av3=8.0,
        sis=12.0, sig=10.0, si=6.0,
        nsim=32, chunk_size=4,
    )
    obs_prop = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
    obs_count = jnp.array([20, 20, 20, 20, 20], dtype=jnp.int64)
    obs_quant = jnp.ones((5, 5)) * 400.0  # placeholder quantiles
    val = float(obj_b.condition_g2_b(rt, cat, obs_prop, obs_count, obs_quant))
    assert np.isfinite(val)


def test_fofs_b_new_returns_finite_scalar():
    """fofs_b_new on real twod3datanew data (subject 0) returns finite positive scalar."""
    import jax
    from pathlib import Path
    from shared import data_io
    from model_b import objective as obj_b

    path = Path(__file__).resolve().parents[2] / "data" / "twod3datanew"
    raw = data_io.load_twod3datanew(path)
    data = {
        "prop": jnp.asarray(raw["prop"][0]),       # (2, 5)
        "count": jnp.asarray(raw["count"][0]),     # (2, 5)
        "quant": jnp.asarray(raw["quant"][0]),     # (2, 5, 5)
    }
    params = jnp.array([200., 50., 10., 2., 12., 10., 0.5,
                        15., 10., 8., 14., 11., 9.])
    key = jax.random.key(0)
    val = float(obj_b.fofs_b_new(params, data, key, nsim=16, chunk_size=4))
    assert np.isfinite(val)
    assert val > 0, f"G2 should be positive, got {val}"


def test_fofs_b_new_deterministic():
    """Same key + same params -> same fofs value."""
    import jax
    from pathlib import Path
    from shared import data_io
    from model_b import objective as obj_b

    path = Path(__file__).resolve().parents[2] / "data" / "twod3datanew"
    raw = data_io.load_twod3datanew(path)
    data = {
        "prop": jnp.asarray(raw["prop"][0]),
        "count": jnp.asarray(raw["count"][0]),
        "quant": jnp.asarray(raw["quant"][0]),
    }
    params = jnp.array([200., 50., 10., 2., 12., 10., 0.5,
                        15., 10., 8., 14., 11., 9.])
    key = jax.random.key(7)
    a = float(obj_b.fofs_b_new(params, data, key, nsim=16, chunk_size=4))
    b = float(obj_b.fofs_b_new(params, data, key, nsim=16, chunk_size=4))
    assert a == b, f"fofs_b_new non-deterministic: a={a!r}, b={b!r}"


def test_fofs_b_new_responds_to_param_change():
    """Changing ter should change fofs value (not zero gradient case at least)."""
    import jax
    from pathlib import Path
    from shared import data_io
    from model_b import objective as obj_b

    path = Path(__file__).resolve().parents[2] / "data" / "twod3datanew"
    raw = data_io.load_twod3datanew(path)
    data = {
        "prop": jnp.asarray(raw["prop"][0]),
        "count": jnp.asarray(raw["count"][0]),
        "quant": jnp.asarray(raw["quant"][0]),
    }
    params_a = jnp.array([200., 50., 10., 2., 12., 10., 0.5,
                          15., 10., 8., 14., 11., 9.])
    params_b = params_a.at[0].set(250.0)  # ter changed
    key = jax.random.key(0)
    val_a = float(obj_b.fofs_b_new(params_a, data, key, nsim=16, chunk_size=4))
    val_b = float(obj_b.fofs_b_new(params_b, data, key, nsim=16, chunk_size=4))
    assert val_a != val_b, f"fofs insensitive to ter change: both = {val_a}"
