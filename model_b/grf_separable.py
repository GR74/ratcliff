"""
Separable (row x column) evaluation of Model B's truncated K-L random field.

The K-L basis in grf_kl is a set of 2D Fourier modes. Mode k contributes
    s_k * (a_k * cos(phi_k) + b_k * sin(phi_k)),
    phi_k(r, c) = 2*pi*(i_k * r / n_pad + j_k * c / m_pad),
which equals Re[ s_k (a_k - i b_k) e^{i 2 pi i_k r / n_pad} e^{i 2 pi j_k c / m_pad} ].
The exponential factors into a row term times a column term, so the whole field is
    field = Re( P @ W @ Q^T ),
with P (n x r1) and Q (m x r2) holding only the row/column frequencies the
retained modes use, and W (r1 x r2) the scattered complex coefficients.

Cost per sample: ~4 * (n*r1*r2 + n*r2*m) real MACs instead of n*m*2*capacity for
the dense basis (about 16x fewer at sig=10). Mode selection, ordering and the
random coefficient layout are taken from grf_kl unchanged, so for the same z the
field matches the dense basis up to float32 rounding.

P and Q do not depend on sig beyond which frequencies are retained; s_k carries
the sig dependence. Shapes are padded to buckets so a fit does not recompile on
every new sig value.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from model_b import grf as grf_circulant
from model_b import grf_kl

ROW_BUCKET = 16
COL_BUCKET = 16


@dataclass(frozen=True)
class SeparableBasis:
    P: jnp.ndarray          # (n, r1_pad) complex64 row exponentials
    Q: jnp.ndarray          # (m, r2_pad) complex64 column exponentials
    flat_index: jnp.ndarray  # (capacity,) int32 position of each mode in W (r1_pad*r2_pad)
    scale: jnp.ndarray      # (capacity,) float32 sqrt-eigenvalue; 0 for padded slots
    K: int                  # retained complex modes
    capacity: int           # padded mode slots (matches grf_kl capacity)
    variance_captured: float
    mode_ids: jnp.ndarray   # (capacity,) uint32, identical to grf_kl return_mode_ids

    @property
    def arrays(self):
        """Device arrays the jitted simulator consumes (a pytree)."""
        return (self.P, self.Q, self.flat_index, self.scale)


def _round_up(x, bucket):
    return int(-(-x // bucket) * bucket)


def selected_modes(sig, n=100, m=160, k_max=2000, variance_threshold=0.99):
    """Same retained-mode selection and order as grf_kl._calc_kl_basis_impl."""
    LAM = grf_circulant.calc_LAM(n=n, m=m, s1=sig, s2=sig)
    n_pad, m_pad = LAM.shape
    eigvals = np.asarray(LAM ** 2).flatten()
    total = float(eigvals.sum())
    if total <= 0.0:
        raise ValueError(f"non-positive total variance at sig={sig}")
    sort_idx = np.argsort(-eigvals)
    sorted_eigvals = eigvals[sort_idx]
    cumvar = np.cumsum(sorted_eigvals) / total
    K = min(int(np.searchsorted(cumvar, variance_threshold) + 1), k_max)
    top_idx = sort_idx[:K]
    return (top_idx, np.sqrt(sorted_eigvals[:K]), n_pad, m_pad, K,
            float(cumvar[K - 1]))


# Capacity for separable_kl given as a variance fraction (e.g. 0.999): 99.9%
# needs 2284 modes at sig 9, 2891 at 8, 3775 at 7, so 4096 covers sig >= ~6.9;
# below that the cap binds and variance_captured < the requested fraction.
K_MAX_HIGH_VARIANCE = 4096

_cache = {}
_CACHE_MAXSIZE = 256   # entries are ~0.2 MB, unlike the 256 MB dense basis


def separable_basis(sig, n=100, m=160, k_max=2000, variance_threshold=0.99,
                    capacity=None):
    """Factored basis, cached by sig rounded to 6 decimals like grf_kl."""
    key = (round(float(sig), 6), n, m, k_max, variance_threshold, capacity)
    if key not in _cache:
        if len(_cache) >= _CACHE_MAXSIZE:
            _cache.pop(next(iter(_cache)))
        _cache[key] = _build(*key)
    return _cache[key]


def _build(sig, n, m, k_max, variance_threshold, capacity):
    """Build the factored basis. capacity defaults to k_max (grf_kl pad_to_k_max)."""
    top_idx, lam_k, n_pad, m_pad, K, var = selected_modes(
        sig, n, m, k_max, variance_threshold)
    i_star, j_star = top_idx // m_pad, top_idx % m_pad
    capacity = k_max if capacity is None else capacity
    if capacity < K:
        raise ValueError(f"capacity {capacity} < retained modes {K}")
    rows, row_of = np.unique(i_star, return_inverse=True)
    cols, col_of = np.unique(j_star, return_inverse=True)
    r1, r2 = _round_up(len(rows), ROW_BUCKET), _round_up(len(cols), COL_BUCKET)
    P = np.zeros((n, r1), dtype=np.complex128)
    Q = np.zeros((m, r2), dtype=np.complex128)
    P[:, :len(rows)] = np.exp(2j * np.pi * np.outer(np.arange(n), rows) / n_pad)
    Q[:, :len(cols)] = np.exp(2j * np.pi * np.outer(np.arange(m), cols) / m_pad)
    flat_index = np.zeros(capacity, dtype=np.int32)   # padded slots add 0 at cell 0
    flat_index[:K] = row_of * r2 + col_of
    scale = np.zeros(capacity, dtype=np.float32)
    scale[:K] = lam_k
    return SeparableBasis(jnp.asarray(P, jnp.complex64), jnp.asarray(Q, jnp.complex64),
                          jnp.asarray(flat_index), jnp.asarray(scale), K, capacity, var,
                          jnp.asarray(grf_kl.mode_ids_for(top_idx, n_pad, m_pad, capacity)))


def project(z, P, Q, flat_index, scale):
    """Field values (..., n*m) for coefficients z (..., 2*capacity), interleaved re/im.

    Equivalent to z @ V.T for the dense grf_kl basis V with the same capacity.
    """
    coef = scale * (z[..., 0::2] - 1j * z[..., 1::2])          # (..., capacity)
    r1, r2 = P.shape[1], Q.shape[1]
    W = jnp.zeros(z.shape[:-1] + (r1 * r2,), dtype=jnp.complex64)
    W = W.at[..., flat_index].add(coef.astype(jnp.complex64))
    W = W.reshape(z.shape[:-1] + (r1, r2))
    rows = jnp.einsum("nu,...uv->...nv", P, W)                  # (..., n, r2)
    field = jnp.einsum("...nv,mv->...nm", rows, Q).real          # (..., n, m)
    return field.reshape(z.shape[:-1] + (P.shape[0] * Q.shape[0],)).astype(jnp.float32)


project_jit = jax.jit(project)
