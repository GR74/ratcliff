"""
Karhunen-Loève low-rank GRF generator for Model B.

For our block-circulant covariance, the K-L eigendecomposition equals the FFT
of the kernel. Top-K modes by spectral magnitude capture ~99.9% of variance at
sig=10 with K ≈ 100. Runtime sampling is one batched GEMM instead of a 2D FFT.

See docs/plans/2026-06-05-model-b-stage-6-design.md for math + design rationale.
"""
import jax
import jax.numpy as jnp
import numpy as np

from model_b import grf as grf_circulant


def calc_kl_basis(sig: float,
                  n: int = 100,
                  m: int = 160,
                  k_max: int = 200,
                  variance_threshold: float = 0.999):
    """
    Build the truncated K-L basis for the circulant-embedded covariance.

    Returns:
        V : (n*m, 2*K) fp32 real basis with √λ folded in
        K : int, actual number of complex modes retained
        variance_captured : float, fraction of total variance retained
    """
    LAM = grf_circulant.calc_LAM(n=n, m=m, s1=sig, s2=sig)
    n_pad, m_pad = LAM.shape

    eigvals = (LAM ** 2).flatten()
    eigvals_np = np.asarray(eigvals)

    sort_idx = np.argsort(-eigvals_np)
    sorted_eigvals = eigvals_np[sort_idx]
    cumvar = np.cumsum(sorted_eigvals) / sorted_eigvals.sum()
    K_by_thresh = int(np.searchsorted(cumvar, variance_threshold) + 1)
    K = min(K_by_thresh, k_max)
    variance_captured = float(cumvar[K - 1])

    top_idx = sort_idx[:K]
    i_star = top_idx // m_pad
    j_star = top_idx % m_pad
    lam_k = np.sqrt(sorted_eigvals[:K]).astype(np.float32)

    norm = 1.0 / np.sqrt(n_pad * m_pad)
    n_grid = np.arange(n)[:, None]
    m_grid = np.arange(m)[None, :]

    V = np.empty((n * m, 2 * K), dtype=np.float32)
    for k in range(K):
        phase = 2.0 * np.pi * (
            i_star[k] * n_grid / n_pad + j_star[k] * m_grid / m_pad
        )
        re = (np.cos(phase) * norm).astype(np.float32).flatten()
        im = (np.sin(phase) * norm).astype(np.float32).flatten()
        V[:, 2 * k] = lam_k[k] * re
        V[:, 2 * k + 1] = lam_k[k] * im

    return jnp.asarray(V), K, variance_captured
