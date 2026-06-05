"""
Karhunen-Loève low-rank GRF generator for Model B.

For our block-circulant covariance, the K-L eigendecomposition equals the FFT
of the kernel. Top-K modes by spectral magnitude capture ~99.9% of variance at
sig=10 with K ≈ 100. Runtime sampling is one batched GEMM instead of a 2D FFT.

See docs/plans/2026-06-05-model-b-stage-6-design.md for math + design rationale.
"""
import jax.numpy as jnp
import numpy as np

from model_b import grf as grf_circulant


def calc_kl_basis(sig: float,
                  n: int = 100,
                  m: int = 160,
                  k_max: int = 2000,
                  variance_threshold: float = 0.99):
    """
    Build the truncated K-L basis for the circulant-embedded covariance.

    Returns:
        V : (n*m, 2*K) fp32 real basis with √λ folded in
        K : int, actual number of complex modes retained
        variance_captured : float, fraction of total variance retained

    Default K is governed by variance_threshold=0.99 at k_max=2000. Empirically
    on the Kroese §2.2 kernel at sig=10 on a 100×160 grid, ~1325 modes hit 99%
    of total variance and ~1850 hit 99.9%. The K-L speedup comes mostly from
    memory bandwidth (smaller noise tensors) rather than compute (GEMM cost
    grows linearly with K). 99% captures enough variance to keep simulator
    statistics within ~1% of the exact circulant generator while still cutting
    memory traffic ~5-8× vs the batched FFT path.
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

    # NOTE on normalization: the circulant_grf path computes
    #   F = fft2(LAM * z)  (unnormalized forward FFT)
    # whose variance per cell is sum_k LAM[k]^2. To match that, the K-L
    # basis must use the UNNORMALIZED Fourier eigenvectors (no 1/sqrt(N)
    # factor) — those are the actual eigenvectors of the circulant matrix.
    # They're orthogonal but each has norm sqrt(N_pad*M_pad), not unit-norm.
    n_grid = np.arange(n)[:, None]
    m_grid = np.arange(m)[None, :]

    V = np.empty((n * m, 2 * K), dtype=np.float32)
    for k in range(K):
        phase = 2.0 * np.pi * (
            i_star[k] * n_grid / n_pad + j_star[k] * m_grid / m_pad
        )
        re = np.cos(phase).astype(np.float32).flatten()
        im = np.sin(phase).astype(np.float32).flatten()
        V[:, 2 * k] = lam_k[k] * re
        V[:, 2 * k + 1] = lam_k[k] * im

    return jnp.asarray(V), K, variance_captured


def sample_kl_grf(V, z, n: int = 100, m: int = 160):
    """
    Generate GRF samples from the truncated K-L basis.

    V : (NM, 2K) fp32 basis from calc_kl_basis (with √λ already folded in)
    z : (batch, 2K) iid N(0,1) random samples
    n, m : output grid dimensions

    Returns: (batch, n, m) fp32 GRF realizations.
    """
    grf_flat = z @ V.T   # (batch, NM)
    return grf_flat.reshape(-1, n, m)
