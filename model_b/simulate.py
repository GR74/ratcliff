"""
2D Gaussian-random-field diffusion simulator (Model B).

Mirrors `accum` from benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS. The simulator:
1. Builds 3 drift Gaussian bumps at fixed positions (uj1=80, uj2=30, uj3=130).
2. Builds 5-category zone array k(i,j) from bump positions.
3. Per timestep: generate one GRF via circulant_grf, accumulate, demean,
   check for crossing.

Note: the Fortran uses an F1/F2 caching trick to halve FFT count. We skip
this trick for simpler code (Stage 5 GPU benchmark will revisit if needed).
"""
from functools import partial

import jax
import jax.numpy as jnp

from model_b import grf
from shared import prng

# Model B field dimensions
N = 100
M = 160
NSTEP = 400
E = 10.0

# Drift bump positions (mirrors Fortran accum lines 432-450)
UI = 50.0       # row center, common to all 3 bumps
UJ1 = 80.0      # cat 1 bump column
UJ2 = 30.0      # cat 3 bump column
UJ3 = 130.0     # cat 4 bump column

# Cached coordinate grids
_I = jnp.arange(N, dtype=jnp.float64)
_J = jnp.arange(M, dtype=jnp.float64)
_I_GRID, _J_GRID = jnp.meshgrid(_I, _J, indexing="ij")  # both (N, M)


def drift_bumps(sis: float):
    """
    Three drift Gaussian bumps centered at (UI=50, UJ=80/30/130) with stddev `sis`.

    Returns (v1, v2, v3), each shape (N, M).
    """
    s3 = 2.0 * sis * sis
    s4 = sis * jnp.sqrt(2.0 * jnp.pi)
    def bump(uj):
        return jnp.exp(-((_J_GRID - uj) ** 2 + (_I_GRID - UI) ** 2) / s3) / s4
    return bump(UJ1), bump(UJ2), bump(UJ3)


def zone_array(si: float = 6.0):
    """
    5-category zone array k(i,j) from the drift bump positions.

    Mirrors Fortran accum lines 432-450:
      - Initialize k = 5 (background).
      - Around UJ1: b > .0003 -> 2 (broader ring), b > .03 -> 1 (innermost).
      - Around UJ2: b > .0003 -> 3.
      - Around UJ3: b > .0003 -> 4.
    """
    s1 = 2.0 * si * si
    s2 = si * jnp.sqrt(2.0 * jnp.pi)
    def b_field(uj):
        return jnp.exp(-((_J_GRID - uj) ** 2 + (_I_GRID - UI) ** 2) / s1) / s2

    k = jnp.full((N, M), 5, dtype=jnp.int32)
    # Around UJ1 (cat 1 innermost, cat 2 outer ring)
    b1 = b_field(UJ1)
    k = jnp.where(b1 > 0.0003, jnp.int32(2), k)
    k = jnp.where(b1 > 0.03,   jnp.int32(1), k)
    # Around UJ2 -> cat 3
    b2 = b_field(UJ2)
    k = jnp.where(b2 > 0.0003, jnp.int32(3), k)
    # Around UJ3 -> cat 4
    b3 = b_field(UJ3)
    k = jnp.where(b3 > 0.0003, jnp.int32(4), k)
    return k


def _simulate_chunk_b(key, ter, st, cr, crsd, av1, av2, av3,
                     LAM, v1, v2, v3, k_zone, chunk_size):
    """
    Simulate `chunk_size` trials of the 2D GRF accumulator model — FAST PATH.

    Three optimizations stacked over the naive per-step scan:
      1. F1/F2 caching: one FFT yields two independent GRF samples (real + imag).
         Halves the FFT count from NSTEP=400 to ceil(NSTEP/2)=200 per trial.
      2. Pre-generate all GRFs as ONE batched FFT call, then cumsum over time
         instead of lax.scan. Eliminates the serial-over-time dependency that
         the scan version had.
      3. Mixed precision: noise generation + FFT + accumulator in fp32 (H100
         fp32 throughput ~= 2x fp64). Threshold comparison stays in fp32 too.

    Returns (rt, cat) each of shape (chunk_size,). cat in {1..5}. RT in ms.

    Memory at chunk=64, fp32: ~25 GB peak working set (z, F, grf_path, incr, a).
    Memory at chunk=16: ~7 GB. Use chunk_size to dial memory vs throughput.

    NOT decorated with @jit — the outer simulate_b wraps and jits.
    """
    ku, kg = jax.random.split(key)

    # Per-trial uniforms (mirrors Fortran gu1)
    u = jax.random.uniform(ku, (chunk_size, 10))
    crr = (cr + crsd * (u[:, 4] - 0.5)).astype(jnp.float32)
    ndt = (ter + st * (0.5 - u[:, 9])) / E              # keep ndt fp64 for RT precision

    n_pad, m_pad = LAM.shape

    # Half the FFTs by using both F1 (real) and F2 (imag) parts.
    # NSTEP=400 -> n_fft=200. If NSTEP odd, the last F2 is unused.
    n_fft = (NSTEP + 1) // 2

    # Step 1: generate ALL noise upfront in fp32, shape (chunk, n_fft, 2, n_pad, m_pad).
    # The "2" is g1, g2 — the real and imaginary parts of the complex FFT input.
    z = jax.random.normal(kg, (chunk_size, n_fft, 2, n_pad, m_pad),
                          dtype=jnp.float32)

    # Step 2: ONE big batched 2D FFT.
    # Build complex input X = LAM * (g1 + 1j*g2), then fft2 over (-2, -1).
    LAM_f32 = LAM.astype(jnp.float32)
    X = LAM_f32 * (z[:, :, 0, :, :] + 1j * z[:, :, 1, :, :])   # (chunk, n_fft, n_pad, m_pad) complex64
    F = jnp.fft.fft2(X)                                         # cuFFT batched 2D FFT

    # Step 3: extract F1 (real) and F2 (imag), both valid GRF samples in (N, M).
    F1 = F[:, :, :N, :M].real                                   # (chunk, n_fft, N, M)
    F2 = F[:, :, :N, :M].imag

    # Interleave to get NSTEP timesteps of GRF: F1 at odd, F2 at even.
    # stack -> (chunk, n_fft, 2, N, M), reshape -> (chunk, n_fft*2, N, M), slice to NSTEP.
    grf_path = jnp.stack([F1, F2], axis=2).reshape(
        chunk_size, n_fft * 2, N, M
    )[:, :NSTEP, :, :]                                          # (chunk, NSTEP, N, M) fp32

    # Step 4: build demeaned per-step increment.
    # drift_const = av1*v1 + av2*v2 + av3*v3  (fp32 throughout)
    v1_f32 = v1.astype(jnp.float32)
    v2_f32 = v2.astype(jnp.float32)
    v3_f32 = v3.astype(jnp.float32)
    av1_f32 = jnp.float32(av1)
    av2_f32 = jnp.float32(av2)
    av3_f32 = jnp.float32(av3)
    drift_const = av1_f32 * v1_f32 + av2_f32 * v2_f32 + av3_f32 * v3_f32
    incr = drift_const[None, None, :, :] + grf_path             # (chunk, NSTEP, N, M)

    # Demean per (trial, step) over spatial axes — same as Fortran's recentering.
    incr = incr - incr.mean(axis=(-2, -1), keepdims=True)

    # Step 5: accumulator path = cumsum over time. No scan.
    a = jnp.cumsum(incr, axis=1)                                # (chunk, NSTEP, N, M)

    # Step 6: first crossing detection.
    # max over (N, M) -> (chunk, NSTEP); compare to threshold; argmax of bool gives
    # first True; fall back to NSTEP if never crossed.
    max_per_step = a.reshape(chunk_size, NSTEP, -1).max(axis=-1)
    crossed = max_per_step > crr[:, None]
    any_crossed = crossed.any(axis=1)
    jstop = jnp.where(any_crossed, jnp.argmax(crossed, axis=1) + 1, NSTEP)

    # Step 7: position at crossing time.
    # Fancy index a[trial, jstop-1, :, :] for each trial, argmax over spatial.
    a_at_crossing = a[jnp.arange(chunk_size), jstop - 1, :, :]   # (chunk, N, M)
    pos_flat = jnp.argmax(a_at_crossing.reshape(chunk_size, -1), axis=-1)

    row = pos_flat // M
    col = pos_flat % M
    cat = jnp.where(any_crossed, k_zone[row, col], jnp.int32(5))
    rt = (jstop.astype(jnp.float64) + ndt) * E                  # back to fp64 for output

    return rt, cat


@partial(jax.jit, static_argnums=(11, 12))
def simulate_b(key, ter, st, cr, crsd, av1, av2, av3,
               sis, sig, si, nsim, chunk_size=4):
    """
    Run `nsim` Model B trials with the given parameters.

    Parameters:
        sis : drift bump width
        sig : GRF correlation length (s1=s2=sig)
        si  : zone-array width
    Returns (rt, cat) each shape (nsim,). cat in {1..5}. RT in ms.

    Memory note: chunk_size=4 default for laptop CPU. H100 can use much larger.
    """
    LAM = grf.calc_LAM(s1=sig, s2=sig)
    v1, v2, v3 = drift_bumps(sis=sis)
    k_zone = zone_array(si=si)

    n_chunks = (nsim + chunk_size - 1) // chunk_size
    keys = prng.trial_keys(key, n_chunks)

    def run_chunk(k):
        return _simulate_chunk_b(
            k, ter, st, cr, crsd, av1, av2, av3,
            LAM, v1, v2, v3, k_zone, chunk_size,
        )

    rts, cats = jax.lax.map(run_chunk, keys)
    return rts.reshape(-1)[:nsim], cats.reshape(-1)[:nsim]
