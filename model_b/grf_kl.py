"""
Karhunen-Loève low-rank GRF generator for Model B.

For our block-circulant covariance, the K-L eigenvalues come from the FFT of
the kernel. About 1325 modes capture 99% of variance at sig=10. Runtime
sampling is one batched GEMM instead of a 2D FFT.

See docs/plans/2026-06-05-model-b-stage-6-design.md for math + design rationale.
"""
import jax
import jax.numpy as jnp
import numpy as np

from model_b import grf as grf_circulant


# Module-level LRU cache for calc_kl_basis. Bounded so it doesn't grow without
# limit if a fit visits many distinct sig values. Each entry holds a
# (n*m, 2*k_max) fp32 array — 256 MB at NM=16000, k_max=2000.
_BASIS_CACHE_MAXSIZE = 64
DEFAULT_K_MAX = 2000
DEFAULT_MODE_BUCKETS = (512, 1024, 1536, 2000)
MODE_ORIENTATION_BIT = np.uint32(0x80000000)
MODE_KEY_MASK = np.uint32(0x7fffffff)
_basis_cache = {}
_basis_cache_order = []


def _cached_calc_kl_basis(sig_key, n, m, k_max, variance_threshold, pad_to_k_max,
                          mode_buckets=None, return_mode_ids=False):
    cache_key = (sig_key, n, m, k_max, variance_threshold, pad_to_k_max, mode_buckets,
                 return_mode_ids)
    cached = _basis_cache.get(cache_key)
    if cached is not None:
        return cached
    result = _calc_kl_basis_impl(sig_key, n, m, k_max, variance_threshold,
                               pad_to_k_max, mode_buckets, return_mode_ids)
    _basis_cache[cache_key] = result
    _basis_cache_order.append(cache_key)
    if len(_basis_cache_order) > _BASIS_CACHE_MAXSIZE:
        evicted = _basis_cache_order.pop(0)
        _basis_cache.pop(evicted, None)
    return result


def calc_kl_basis(sig: float,
                  n: int = 100,
                  m: int = 160,
                  k_max: int = 2000,
                  variance_threshold: float = 0.99,
                  pad_to_k_max: bool = False, mode_buckets=None,
                  return_mode_ids: bool = False):
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

    pad_to_k_max : if True, zero-pad V to shape (n*m, 2*k_max). This is
        critical for callers inside JIT — keeps the basis tensor shape
        constant across distinct sig values, preventing repeated tracing
        and XLA recompilation. Default False preserves the original tight
        shape for tests + standalone use.

    Cached by (rounded sig, n, m, k_max, variance_threshold, pad_to_k_max)
    so a fit visiting the same sig twice doesn't rebuild.

    mode_buckets optionally pads to the smallest listed complex-mode capacity
    that covers K. It changes padding only, never the retained modes or variance
    threshold. Bucket configuration is included in the cache key.

    return_mode_ids additionally returns a uint32 code for each complex mode
    column pair. Its low bits identify a canonical conjugate family and an
    independent component within that family. The high bit reverses the sine
    coefficient for a conjugate orientation. Thus a singleton's representative
    may flip without changing its random field. If both members are retained,
    they receive independent components. Padding is 0xffffffff, with zero columns.
    The historical three-value return and basis values remain unchanged.
    """
    if mode_buckets is not None:
        mode_buckets = tuple(mode_buckets)
        if (not mode_buckets or any(not isinstance(k, int) or k < 1 for k in mode_buckets)
                or any(a >= b for a, b in zip(mode_buckets, mode_buckets[1:]))
                or mode_buckets[-1] < k_max):
            raise ValueError("mode buckets must increase strictly and cover k_max")
    # Round sig key to 6 decimals so floating-point jitter still hits cache.
    sig_key = round(float(sig), 6)
    return _cached_calc_kl_basis(sig_key, n, m, k_max,
                                  variance_threshold, pad_to_k_max, mode_buckets,
                                  return_mode_ids)


def _calc_kl_basis_impl(sig: float, n: int, m: int, k_max: int,
                         variance_threshold: float, pad_to_k_max: bool,
                         mode_buckets=None, return_mode_ids=False):
    """The actual K-L basis construction. Wrapped by calc_kl_basis with caching."""
    LAM = grf_circulant.calc_LAM(n=n, m=m, s1=sig, s2=sig)
    n_pad, m_pad = LAM.shape

    eigvals_np = np.asarray(LAM ** 2).flatten()
    total = float(eigvals_np.sum())
    if total <= 0.0:
        raise ValueError(
            f"calc_kl_basis: covariance has non-positive total variance "
            f"(sum={total:.3e}) at sig={sig}. The embedding may be degenerate; "
            f"check that sig is in the valid range (see grf.assert_pd_embedding)."
        )

    sort_idx = np.argsort(-eigvals_np)
    sorted_eigvals = eigvals_np[sort_idx]
    cumvar = np.cumsum(sorted_eigvals) / total
    K_by_thresh = int(np.searchsorted(cumvar, variance_threshold) + 1)
    K = min(K_by_thresh, k_max)
    variance_captured = float(cumvar[K - 1])

    top_idx = sort_idx[:K]
    i_star = top_idx // m_pad           # (K,)
    j_star = top_idx % m_pad            # (K,)
    lam_k = np.sqrt(sorted_eigvals[:K]).astype(np.float32)  # √λ_k, (K,)

    # NOTE on normalization: the circulant_grf path computes
    #   F = fft2(LAM * z)  (unnormalized forward FFT)
    # whose variance per cell is sum_k LAM[k]^2. To match that, the K-L
    # basis must use the UNNORMALIZED Fourier eigenvectors (no 1/sqrt(N)
    # factor) — those are the actual eigenvectors of the circulant matrix.
    # They're orthogonal but each has norm sqrt(N_pad*M_pad), not unit-norm.
    #
    # Vectorized over all K modes at once instead of a Python loop: build the
    # (K, n, m) phase tensor by broadcasting, then cos/sin, fold in √λ, and
    # interleave real/imag into the (n*m, 2K) basis. Materially faster than the
    # per-mode loop at K≈1325-2000.
    n_grid = np.arange(n)[None, :, None]          # (1, n, 1)
    m_grid = np.arange(m)[None, None, :]          # (1, 1, m)
    ii = i_star[:, None, None]                    # (K, 1, 1)
    jj = j_star[:, None, None]                    # (K, 1, 1)
    phase = 2.0 * np.pi * (ii * n_grid / n_pad + jj * m_grid / m_pad)  # (K, n, m)
    scale = lam_k[:, None, None]
    re = (np.cos(phase) * scale).reshape(K, n * m).astype(np.float32)
    im = (np.sin(phase) * scale).reshape(K, n * m).astype(np.float32)

    # Pad output to (n*m, 2*k_max) when requested; the extra columns are zeros
    # so any random noise multiplied against them contributes nothing.
    capacity = (next(k for k in mode_buckets if k >= K) if mode_buckets is not None
                else k_max if pad_to_k_max else K)
    cols = 2 * capacity
    V = np.zeros((n * m, cols), dtype=np.float32)
    V[:, 0 : 2 * K : 2] = re.T        # even columns = real parts
    V[:, 1 : 2 * K : 2] = im.T        # odd columns  = imag parts

    result = (jnp.asarray(V), K, variance_captured)
    if return_mode_ids:
        mode_ids = np.full(capacity, np.uint32(0xffffffff), dtype=np.uint32)
        conjugates = ((-i_star) % n_pad) * m_pad + ((-j_star) % m_pad)
        canonical = np.minimum(top_idx, conjugates)
        selected = set(top_idx.tolist())
        paired = np.asarray([c != i and c in selected for i, c in zip(top_idx, conjugates)])
        reverse = top_idx != canonical
        components = (paired & reverse).astype(np.uint32)
        codes = 2 * canonical.astype(np.uint32) + components
        codes |= np.where(reverse, MODE_ORIENTATION_BIT, np.uint32(0))
        mode_ids[:K] = codes
        return (*result, jnp.asarray(mode_ids))
    return result


def mode_normal_blocks(keys, mode_ids, block_size, dtype=jnp.float32):
    """Independent normal coefficient pairs indexed by physical Fourier mode.

    keys are per-trial (and, for blocked simulation, per-block) keys. Each
    physical mode component receives its own folded key before generating
    time x real/imaginary draws. Conjugate orientation flips only the sine
    coefficient, compensating the basis's sine sign. Changing rank, retained
    set or capacity does not reassign shared components. Different block lengths
    remain different random layouts; use the same length in paired controls.
    """
    codes = jnp.asarray(mode_ids, dtype=jnp.uint32)
    identities = codes & MODE_KEY_MASK
    orientation = jnp.where((codes & MODE_ORIENTATION_BIT) != 0, -1., 1.).astype(dtype)

    def sample_trial(key):
        mode_keys = jax.vmap(lambda mode: jax.random.fold_in(key, mode))(identities)
        samples = jax.vmap(lambda k: jax.random.normal(
            k, (block_size, 2), dtype=dtype))(mode_keys)
        samples = samples.at[:, :, 1].multiply(orientation[:, None])
        return samples.transpose(1, 0, 2).reshape(block_size, 2 * len(mode_ids))

    return jax.vmap(sample_trial)(keys)


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
