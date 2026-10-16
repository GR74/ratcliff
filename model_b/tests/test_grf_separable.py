"""Separable K-L projection must reproduce the dense grf_kl basis field."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import grf_kl, grf_separable


@pytest.mark.parametrize("sig", [5.0, 10.0, 10.000001, 15.0])
def test_separable_matches_dense_basis(sig):
    V, K, var, _ = grf_kl.calc_kl_basis(sig=sig, pad_to_k_max=True, return_mode_ids=True)
    sb = grf_separable.separable_basis(sig)
    assert sb.K == K
    assert sb.capacity * 2 == V.shape[1]
    assert sb.variance_captured == pytest.approx(var)
    z = jax.random.normal(jax.random.key(7), (8, V.shape[1]), dtype=jnp.float32)
    # Full fp32 matmuls on both sides: GPU default float32 matmul is TF32
    # (~1e-3 relative), which would dominate this comparison.
    with jax.default_matmul_precision("highest"):
        dense = np.asarray(z @ V.T)
        sep = np.asarray(grf_separable.project(z, sb.P, sb.Q, sb.flat_index, sb.scale))
    rel_max_err = np.abs(sep - dense).max() / dense.std()
    assert rel_max_err < 1e-4, rel_max_err


def test_padded_slots_contribute_nothing():
    sb = grf_separable.separable_basis(10.0)
    z = jnp.zeros((1, 2 * sb.capacity), dtype=jnp.float32)
    z = z.at[0, 2 * sb.K:].set(1.0)                 # only padded slots nonzero
    field = grf_separable.project_jit(z, sb.P, sb.Q, sb.flat_index, sb.scale)
    assert float(jnp.abs(field).max()) == 0.0
