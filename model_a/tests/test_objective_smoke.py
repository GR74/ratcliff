"""Smoke tests for model_a/objective.py."""
import jax.numpy as jnp
import pytest


def test_objective_module_imports():
    from model_a import objective
    # Confirm public API
    assert hasattr(objective, "condition_g2_vectorized")
    assert hasattr(objective, "fofs_new")
    assert hasattr(objective, "COND_MAP")


def test_cond_map_has_4_conditions():
    from model_a import objective
    assert len(objective.COND_MAP) == 4
    for (di, bi) in objective.COND_MAP:
        assert isinstance(di, int)
        assert isinstance(bi, int)


def test_condition_g2_matches_jax_port_at_realistic():
    """The vectorized condition_g2 must match jax_port's scalar output."""
    import numpy as np
    from model_a import jax_port, objective
    from shared import prng

    # Generate one condition's worth of trials
    key = prng.root_key(42)
    rt, cat = jax_port.simulate(
        key, ter=200.0, st=50.0, cr=50.0, crsd=10.0,
        si=4.0, sig=5.0, av=20.0, sv=0.7, nsim=512
    )

    # Synthetic observed-data triplet for one condition (matches fofs's inner call)
    obs_prop = jnp.array([0.5, 0.3, 0.2])
    obs_count = jnp.array([50, 30, 20], dtype=jnp.int64)
    obs_quant = jnp.array([
        [300.0, 320.0, 360.0],
        [340.0, 360.0, 400.0],
        [380.0, 400.0, 440.0],
        [420.0, 440.0, 480.0],
        [460.0, 480.0, 520.0],
    ])  # shape (NQ=5, MC=3)

    g2_old = float(jax_port.condition_g2(rt, cat, obs_prop, obs_count, obs_quant))
    g2_new = float(objective.condition_g2_vectorized(rt, cat, obs_prop, obs_count, obs_quant))
    rel_diff = abs(g2_new - g2_old) / abs(g2_old) if abs(g2_old) > 1e-9 else abs(g2_new - g2_old)
    assert rel_diff < 1e-4, f"condition_g2 mismatch: old={g2_old}, new={g2_new}, rel_diff={rel_diff}"
