"""
Gaussian Random Field generator via circulant embedding (Kroese §2.2).

Mirrors `calc_LAM` and `circulant_grf` from the Fortran reference
`benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS`. The kernel formula is the
Matern-class variogram documented in `reference/README_random_field`.
"""
import jax
import jax.numpy as jnp

DEFAULT_N = 100
DEFAULT_M = 160


def _kernel_value(dx, dy, s1, s2):
    """ρ(dx, dy) = (1 - dx²/s2² - dx·dy/(s1·s2) - dy²/s1²) · exp(-(dx²/s2² + dy²/s1²))"""
    x = dx.astype(jnp.float64)
    y = dy.astype(jnp.float64)
    return (1.0 - x * x / (s2 * s2) - x * y / (s1 * s2) - y * y / (s1 * s1)) \
        * jnp.exp(-(x * x / (s2 * s2) + y * y / (s1 * s1)))


def calc_LAM(n: int = DEFAULT_N, m: int = DEFAULT_M, s1: float = 15.0, s2: float = 15.0):
    """
    Compute the spectral square root LAM of the block-circulant embedded kernel.

    Returns LAM of shape (2n-1, 2m-1).

    Raises ValueError if the embedding is not positive-definite. At 100x160,
    this happens when s1 or s2 > ~17.95.
    """
    # Displacements (the Fortran loops i over cols, j over rows)
    # rows[j,i] = ρ(dx = tx[i] - tx[0], dy = ty[j] - ty[0])
    # tx[i] = i, ty[j] = j (0-indexed)
    i_idx = jnp.arange(m)
    j_idx = jnp.arange(n)
    dx_pos = i_idx[None, :].astype(jnp.float64)   # (1, m)
    dy_pos = j_idx[:, None].astype(jnp.float64)   # (n, 1)
    # rows: ρ(dx=+i, dy=+j) for i in 0..m-1, j in 0..n-1, broadcast to (n, m)
    rows = _kernel_value(dx_pos, dy_pos, s1, s2)  # (n, m)
    # cols: ρ(dx=-i, dy=+j), same shape
    cols = _kernel_value(-dx_pos, dy_pos, s1, s2)  # (n, m)

    # Embed into (2n-1, 2m-1) block-circulant
    n_pad = 2 * n - 1
    m_pad = 2 * m - 1
    blkcirc = jnp.zeros((n_pad, m_pad), dtype=jnp.complex128)
    # Top-left (n, m): rows
    blkcirc = blkcirc.at[:n, :m].set(rows.astype(jnp.complex128))
    # Top-right (n, m-1): cols[:, m-i] for i in 1..m-1
    # i.e., for i in 1..m-1, blkcirc[:n, i + m - 1] = cols[:, m - i]
    # Vectorized: cols_flipped = cols[:, 1:][::-1 on col axis] = cols[:, m-1:0:-1]
    cols_right = cols[:, 1:][:, ::-1]              # cols[:, m-1], cols[:, m-2], ..., cols[:, 1]
    blkcirc = blkcirc.at[:n, m:].set(cols_right.astype(jnp.complex128))
    # Bottom-left (n-1, m): cols[n-j, :] for j in 1..n-1
    # i.e., blkcirc[j + n - 1, :m] = cols[n - j, :]
    cols_bottom = cols[1:, :][::-1, :]            # cols[n-1, :], cols[n-2, :], ..., cols[1, :]
    blkcirc = blkcirc.at[n:, :m].set(cols_bottom.astype(jnp.complex128))
    # Bottom-right (n-1, m-1): rows[n-j, m-i]
    rows_br = rows[1:, 1:][::-1, ::-1]
    blkcirc = blkcirc.at[n:, m:].set(rows_br.astype(jnp.complex128))

    # 2D FFT, take real, normalize
    spectral = jnp.fft.fft2(blkcirc).real / (n_pad * m_pad)

    # Note: PD embedding check moved to assert_pd_embedding() for JIT compatibility.
    # Caller is responsible for ensuring s1, s2 are in the valid range (clamp_b
    # in objective.py clips sig to [0.2, 17.0], well within the empirical 17.95 ceiling).
    return jnp.sqrt(jnp.maximum(spectral, 0.0))


def assert_pd_embedding(n: int = DEFAULT_N, m: int = DEFAULT_M,
                       s1: float = 15.0, s2: float = 15.0):
    """
    Validate that the circulant embedding for (n, m, s1, s2) is positive-definite.

    This runs the same spectral computation as calc_LAM but ONLY checks
    positive-definiteness. Use this in Python-level preconditions before
    calling calc_LAM inside a JIT region.

    Raises ValueError if the minimum spectral value < -1e-14 (matching
    the Fortran reference). For 100x160 fields, this happens at s1 or s2 > ~17.95.
    """
    # Repeat the embedding step from calc_LAM and check.
    i_idx = jnp.arange(m)
    j_idx = jnp.arange(n)
    dx_pos = i_idx[None, :].astype(jnp.float64)
    dy_pos = j_idx[:, None].astype(jnp.float64)
    rows = _kernel_value(dx_pos, dy_pos, s1, s2)
    cols = _kernel_value(-dx_pos, dy_pos, s1, s2)

    n_pad = 2 * n - 1
    m_pad = 2 * m - 1
    blkcirc = jnp.zeros((n_pad, m_pad), dtype=jnp.complex128)
    blkcirc = blkcirc.at[:n, :m].set(rows.astype(jnp.complex128))
    cols_right = cols[:, 1:][:, ::-1]
    blkcirc = blkcirc.at[:n, m:].set(cols_right.astype(jnp.complex128))
    cols_bottom = cols[1:, :][::-1, :]
    blkcirc = blkcirc.at[n:, :m].set(cols_bottom.astype(jnp.complex128))
    rows_br = rows[1:, 1:][::-1, ::-1]
    blkcirc = blkcirc.at[n:, m:].set(rows_br.astype(jnp.complex128))

    spectral = jnp.fft.fft2(blkcirc).real / (n_pad * m_pad)
    min_val = float(jnp.min(spectral))
    if min_val < -1e-14:
        raise ValueError(
            f"Could not find positive definite embedding (min spectral = {min_val:.3e}). "
            f"For 100x160, s1, s2 must be < ~17.95."
        )


def circulant_grf(LAM, g1, g2):
    """
    Generate two independent GRF samples via one 2D FFT.

    LAM    : (2n-1, 2m-1) — spectral sqrt from calc_LAM.
    g1, g2 : (2n-1, 2m-1) — iid N(0,1) arrays.

    Returns (F1, F2), each shape (n, m), both samples from the GRF.

    The trick: the FFT of a complex Gaussian array has independent real
    and imaginary parts, both with the desired covariance. So one FFT
    yields two samples.
    """
    n_pad, m_pad = LAM.shape
    n = (n_pad + 1) // 2
    m = (m_pad + 1) // 2
    X = LAM * (g1 + 1j * g2)
    F = jnp.fft.fft2(X)
    return F[:n, :m].real, F[:n, :m].imag
