"""Smoke tests for model_b/grf.py."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_b import grf


def test_calc_LAM_returns_correct_shape():
    LAM = grf.calc_LAM(n=100, m=160, s1=15.0, s2=15.0)
    assert LAM.shape == (199, 319)  # (2n-1, 2m-1)


def test_calc_LAM_is_nonneg():
    """LAM should be non-negative everywhere (it's the sqrt of a spectral density)."""
    LAM = grf.calc_LAM(n=100, m=160, s1=15.0, s2=15.0)
    assert jnp.all(LAM >= 0)


def test_calc_LAM_fails_on_large_s():
    """Per Russ's notes, s > 17.95 breaks the positive-definite embedding."""
    with pytest.raises(ValueError, match="positive definite"):
        grf.calc_LAM(n=100, m=160, s1=20.0, s2=20.0)


def test_calc_LAM_smaller_field_works_at_smaller_s():
    """Sanity: at smaller field, also smaller s; should still produce a valid LAM."""
    LAM = grf.calc_LAM(n=40, m=64, s1=5.0, s2=5.0)
    assert LAM.shape == (79, 127)
    assert jnp.all(LAM >= 0)


def test_circulant_grf_returns_two_grfs():
    LAM = grf.calc_LAM(n=100, m=160, s1=10.0, s2=10.0)
    key = jax.random.key(0)
    g_pair = jax.random.normal(key, (2, 199, 319))
    F1, F2 = grf.circulant_grf(LAM, g_pair[0], g_pair[1])
    assert F1.shape == (100, 160)
    assert F2.shape == (100, 160)


def test_circulant_grf_outputs_are_real_finite():
    LAM = grf.calc_LAM(n=100, m=160, s1=10.0, s2=10.0)
    g_pair = jax.random.normal(jax.random.key(0), (2, 199, 319))
    F1, F2 = grf.circulant_grf(LAM, g_pair[0], g_pair[1])
    assert jnp.all(jnp.isfinite(F1))
    assert jnp.all(jnp.isfinite(F2))


def test_circulant_grf_aggregate_variance_near_one():
    """At zero displacement, kernel rho(0,0)=1, so empirical variance approx 1.0."""
    LAM = grf.calc_LAM(n=100, m=160, s1=10.0, s2=10.0)
    key = jax.random.key(42)
    n_samples = 200
    variances = []
    for s in range(n_samples):
        k = jax.random.fold_in(key, s)
        g = jax.random.normal(k, (2, 199, 319))
        F1, F2 = grf.circulant_grf(LAM, g[0], g[1])
        variances.append(float(F1[50, 80] ** 2))
        variances.append(float(F2[50, 80] ** 2))
    emp_var = np.mean(variances)
    # Tolerance ~10% — MC noise at 400 samples is sqrt(2/400) ~ 7%, so 10% is comfortable.
    assert 0.85 < emp_var < 1.15, f"empirical variance {emp_var:.3f}, expected ~1.0"
