"""Smoke tests for model_b/grf_kl.py — K-L basis builder + sampler."""
import jax.numpy as jnp

from model_b import grf_kl


def test_calc_kl_basis_returns_correct_shapes():
    """K-L basis V should be (NM, 2K) fp32 real."""
    V, K, var_captured = grf_kl.calc_kl_basis(sig=10.0, n=100, m=160,
                                              k_max=200, variance_threshold=0.999)
    assert V.shape[0] == 100 * 160, f"V row count should be N*M=16000, got {V.shape[0]}"
    assert V.shape[1] == 2 * K, f"V col count should be 2*K={2*K}, got {V.shape[1]}"
    assert V.dtype == jnp.float32, f"V should be fp32, got {V.dtype}"
    assert 1 <= K <= 200, f"K should be in [1, 200], got {K}"
    assert 0.0 <= var_captured <= 1.0, f"variance_captured should be in [0, 1], got {var_captured}"
