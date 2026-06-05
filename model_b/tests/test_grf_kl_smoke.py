"""Smoke tests for model_b/grf_kl.py — K-L basis builder + sampler."""
import jax.numpy as jnp

from model_b import grf_kl


def test_calc_kl_basis_returns_correct_shapes():
    """K-L basis V should be (NM, 2K) fp32 real."""
    V, K, var_captured = grf_kl.calc_kl_basis(sig=10.0, n=100, m=160,
                                              k_max=2000, variance_threshold=0.99)
    assert V.shape[0] == 100 * 160, f"V row count should be N*M=16000, got {V.shape[0]}"
    assert V.shape[1] == 2 * K, f"V col count should be 2*K={2*K}, got {V.shape[1]}"
    assert V.dtype == jnp.float32, f"V should be fp32, got {V.dtype}"
    assert 1 <= K <= 2000, f"K should be in [1, 2000], got {K}"
    assert 0.0 <= var_captured <= 1.0, f"variance_captured should be in [0, 1], got {var_captured}"


def test_calc_kl_basis_variance_threshold():
    """Variance captured should hit the requested threshold (or hit k_max cap)."""
    V, K, vc = grf_kl.calc_kl_basis(sig=10.0, variance_threshold=0.99, k_max=2000)
    # Either threshold met, OR we hit the cap (in which case test just records that)
    assert vc >= 0.99 or K == 2000, (
        f"variance_captured {vc:.4f} below threshold 0.99 and K={K} did not hit k_max=2000"
    )
    # Empirically at sig=10: 99% needs ~1325 modes
    assert K < 2000, f"At sig=10 with k_max=2000, K={K} should not be capped"


def test_calc_kl_basis_k_grows_with_higher_threshold():
    """Higher threshold should need more modes."""
    _, K_95, _ = grf_kl.calc_kl_basis(sig=10.0, variance_threshold=0.95, k_max=2000)
    _, K_99, _ = grf_kl.calc_kl_basis(sig=10.0, variance_threshold=0.99, k_max=2000)
    assert K_95 <= K_99, f"K should grow with threshold (95%={K_95}, 99%={K_99})"
