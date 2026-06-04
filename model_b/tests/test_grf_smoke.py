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
