"""
Smoke + gradient tests for model_a/objective_smooth.py.

The KEY test is `test_fofs_smooth_gradient_is_finite_and_nonzero` — if this passes,
the L-BFGS path is unlocked. Stage 3 found jax.grad(fofs_new) is structurally zero.
This test confirms the smooth surrogate restores useful gradients.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_a import objective_smooth as obj_sm
from shared import data_io, prng


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "twod24data"


def _first_subject_data():
    d = data_io.load_twod24data(DATA_PATH)
    g = data_io.group_by_subject(d, conditions_per_subject=4)
    return {
        "prop": jnp.asarray(g["prop"][0]),
        "count": jnp.asarray(g["count"][0]),
        "quant": jnp.asarray(g["quant"][0]),
    }


def test_fofs_smooth_returns_finite_scalar():
    """fofs_smooth on real data returns a finite positive scalar."""
    data = _first_subject_data()
    params = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])
    key = prng.root_key(0)
    val = float(obj_sm.fofs_smooth(params, data, key, nsim=64, chunk_size=64))
    assert np.isfinite(val), f"non-finite: {val}"
    assert val > 0, f"G² should be positive, got {val}"


def test_fofs_smooth_deterministic():
    """Same key + params → same scalar."""
    data = _first_subject_data()
    params = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])
    key = prng.root_key(7)
    a = float(obj_sm.fofs_smooth(params, data, key, nsim=32, chunk_size=32))
    b = float(obj_sm.fofs_smooth(params, data, key, nsim=32, chunk_size=32))
    assert a == b


def test_fofs_smooth_gradient_is_finite_and_nonzero():
    """
    THE Stage 3.5 test: jax.grad(fofs_smooth) must produce a finite, non-zero
    gradient vector on at least 8 of 10 params. (sv may be inert; clamp at floor
    components can be zero.)
    """
    data = _first_subject_data()
    params = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])
    key = prng.root_key(0)

    grad_fn = jax.grad(lambda p: obj_sm.fofs_smooth(p, data, key, nsim=64, chunk_size=64))
    g = grad_fn(params)

    assert g.shape == (10,)
    assert jnp.all(jnp.isfinite(g)), f"gradient has non-finite: {g}"
    n_active = int(jnp.sum(jnp.abs(g) > 1e-4))
    assert n_active >= 8, (
        f"only {n_active}/10 params have meaningful gradient: {g}. "
        f"Stage 3.5's headline win requires 8+ active gradients."
    )


def test_fofs_smooth_changes_with_params():
    """Perturbing ter changes the smooth fofs (sanity for the gradient test)."""
    data = _first_subject_data()
    key = prng.root_key(0)
    params_a = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])
    params_b = params_a.at[0].set(250.0)
    a = float(obj_sm.fofs_smooth(params_a, data, key, nsim=64, chunk_size=64))
    b = float(obj_sm.fofs_smooth(params_b, data, key, nsim=64, chunk_size=64))
    assert a != b, f"fofs_smooth insensitive to ter change: both {a}"
