"""Parity tests for K-L GRF vs circulant GRF (model_b/grf.py).

Both generators should produce GRFs with the same marginal variance and
autocorrelation structure, up to Monte Carlo sampling noise and the K-L
truncation bias (variance_threshold=0.99 means ~1% variance underestimate).
"""
import jax
import jax.numpy as jnp
import numpy as np

from model_b import grf as grf_circ
from model_b import grf_kl


def _sample_circulant_grfs(key, n_samples, sig=10.0, n=100, m=160):
    """Generate n_samples GRFs from the circulant generator using F1/F2 trick."""
    LAM = grf_circ.calc_LAM(n=n, m=m, s1=sig, s2=sig)
    n_pad, m_pad = LAM.shape
    grfs = []
    n_fft_calls = (n_samples + 1) // 2
    for i in range(n_fft_calls):
        k = jax.random.fold_in(key, i)
        g1 = jax.random.normal(jax.random.fold_in(k, 0), (n_pad, m_pad))
        g2 = jax.random.normal(jax.random.fold_in(k, 1), (n_pad, m_pad))
        F1, F2 = grf_circ.circulant_grf(LAM, g1, g2)
        grfs.append(np.asarray(F1))
        if len(grfs) < n_samples:
            grfs.append(np.asarray(F2))
    return np.stack(grfs[:n_samples])  # (n_samples, n, m)


def _sample_kl_grfs(key, n_samples, sig=10.0, n=100, m=160):
    """Generate n_samples GRFs from the K-L generator."""
    V, K, _ = grf_kl.calc_kl_basis(sig=sig, n=n, m=m)
    z = jax.random.normal(key, (n_samples, 2 * K), dtype=jnp.float32)
    return np.asarray(grf_kl.sample_kl_grf(V, z, n=n, m=m))


def test_marginal_variance_parity():
    """Per-cell variance should match between circulant and K-L generators.

    The K-L generator has variance ~1% lower than circulant due to the
    variance_threshold=0.99 truncation. Tolerance allows 5% to also absorb
    Monte Carlo noise from 2000 samples.
    """
    n_samples = 2000
    key_c = jax.random.key(0)
    key_k = jax.random.key(1)
    grfs_c = _sample_circulant_grfs(key_c, n_samples)
    grfs_k = _sample_kl_grfs(key_k, n_samples)

    var_c = grfs_c.var(axis=0)  # (n, m)
    var_k = grfs_k.var(axis=0)

    mean_var_c = var_c.mean()
    mean_var_k = var_k.mean()
    rel_err = abs(mean_var_k - mean_var_c) / mean_var_c
    assert rel_err < 0.05, (
        f"Mean variance mismatch: circulant={mean_var_c:.6f}, "
        f"K-L={mean_var_k:.6f}, rel_err={rel_err:.4f}"
    )
