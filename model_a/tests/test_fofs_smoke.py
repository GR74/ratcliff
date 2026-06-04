from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from model_a import jax_port
from shared import data_io, prng

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "twod24data"


def _first_subject_4_conditions():
    """fofs in jax_port expects {"prop", "count", "quant"} for 4 conditions."""
    d = data_io.load_twod24data(DATA_PATH)
    g = data_io.group_by_subject(d, conditions_per_subject=4)
    return {
        "prop": jnp.asarray(g["prop"][0]),    # (4, 3)
        "count": jnp.asarray(g["count"][0]),  # (4, 3)
        "quant": jnp.asarray(g["quant"][0]),  # (4, 5, 3)
    }


def test_fofs_returns_finite_scalar():
    data = _first_subject_4_conditions()
    params = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])
    key = prng.root_key(0)
    val = jax_port.fofs(params, data, key, nsim=512)
    val = float(val)
    assert np.isfinite(val), f"fofs returned non-finite: {val}"
    assert val > 0, f"fofs is a G^2 statistic and should be positive: {val}"


def test_fofs_is_deterministic_for_same_key():
    data = _first_subject_4_conditions()
    params = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])
    key = prng.root_key(7)
    a = float(jax_port.fofs(params, data, key, nsim=256))
    b = float(jax_port.fofs(params, data, key, nsim=256))
    assert a == b, f"fofs non-deterministic: a={a!r}, b={b!r}"


def test_fofs_changes_when_params_change():
    data = _first_subject_4_conditions()
    key = prng.root_key(0)
    params_a = jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])
    params_b = jnp.array([250., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])  # ter different
    # Note: params[6] (sv) is currently inert because jax_port.SV_ACTIVE = False
    # (mirrors a Fortran dummy-arg mismatch). Do not change the perturbation
    # index to 6; use ter (0) or any other active parameter.
    a = float(jax_port.fofs(params_a, data, key, nsim=512))
    b = float(jax_port.fofs(params_b, data, key, nsim=512))
    assert a != b, f"fofs insensitive to ter change: both returned {a!r}"
