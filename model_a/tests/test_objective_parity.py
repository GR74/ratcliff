"""
Parity tests: fofs_new must match jax_port.fofs to MC tolerance at the same params.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from model_a import jax_port, objective
from shared import data_io, prng
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "twod24data"


def _first_subject_data():
    """Same helper as test_fofs_smoke; returns 4-condition data dict for subject 0."""
    d = data_io.load_twod24data(DATA_PATH)
    g = data_io.group_by_subject(d, conditions_per_subject=4)
    return {
        "prop": jnp.asarray(g["prop"][0]),
        "count": jnp.asarray(g["count"][0]),
        "quant": jnp.asarray(g["quant"][0]),
    }


PARAM_SETS = [
    ("realistic", jnp.array([200., 50., 50., 10., 4., 5., 0.7, 20., 10., 60.])),
    ("high_drift", jnp.array([200., 50., 50., 10., 4., 5., 0.7, 60., 50., 60.])),
    ("low_drift", jnp.array([200., 50., 50., 10., 4., 5., 0.7, 5., 3., 60.])),
]


@pytest.mark.parametrize("name,params", PARAM_SETS, ids=[p[0] for p in PARAM_SETS])
def test_fofs_new_matches_jax_port_fofs(name, params):
    """fofs_new and jax_port.fofs return scalar G2 values within 5% relative."""
    data = _first_subject_data()
    key = prng.root_key(1337)

    val_old = float(jax_port.fofs(params, data, key, nsim=2048))
    val_new = float(objective.fofs_new(params, data, key, nsim=2048))

    rel_diff = abs(val_new - val_old) / abs(val_old)
    assert rel_diff < 0.05, (
        f"[{name}] fofs mismatch: old={val_old:.3f}, new={val_new:.3f}, "
        f"rel_diff={rel_diff:.4f} (tol 0.05)"
    )
