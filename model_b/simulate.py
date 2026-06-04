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
    Simulate `chunk_size` trials of the 2D GRF accumulator model.

    Returns (rt, cat) each of shape (chunk_size,). cat in {1..5}. RT in ms.

    NOT decorated with @jit — the outer simulate_b wraps and jits.
    """
    ku, kt = jax.random.split(key)

    # Per-trial uniforms (mirrors Fortran gu1)
    u = jax.random.uniform(ku, (chunk_size, 10))
    crr = cr + crsd * (u[:, 4] - 0.5)       # (chunk_size,) per-trial threshold
    ndt = (ter + st * (0.5 - u[:, 9])) / E   # (chunk_size,) nondecision in steps

    n_pad, m_pad = LAM.shape

    # Per-trial key derivation: one key per trial, then per-step split inside scan
    trial_keys = jax.random.split(kt, chunk_size)   # (chunk_size,) typed keys

    def per_trial(trial_key, trial_crr, trial_ndt):
        # Per-step keys for THIS trial
        step_keys = jax.random.split(trial_key, NSTEP)  # (NSTEP,) typed keys

        def step(carry, scan_inputs):
            a, jstop, pos_flat, done = carry
            sk, t = scan_inputs
            # Generate fresh GRF for this step
            g_pair = jax.random.normal(sk, (2, n_pad, m_pad))
            F1, _ = grf.circulant_grf(LAM, g_pair[0], g_pair[1])
            # Accumulate drift bumps + GRF noise
            a = a + av1 * v1 + av2 * v2 + av3 * v3 + F1
            # Demean (match Fortran's per-step recentering)
            a = a - a.mean()
            # Check crossing
            am = a.max()
            crossed = am > trial_crr
            newly = crossed & (~done)
            jstop = jnp.where(newly, t + 1, jstop)
            pos_flat = jnp.where(newly, jnp.argmax(a), pos_flat)
            return (a, jstop, pos_flat, done | crossed), None

        init = (jnp.zeros((N, M)), NSTEP, 0, False)
        ts = jnp.arange(NSTEP)
        (a_final, jstop, pos_flat, done), _ = jax.lax.scan(
            step, init, (step_keys, ts),
        )
        # Compute (row, col) from flat index
        row = pos_flat // M
        col = pos_flat % M
        # For non-crossing trials, k_zone at pos_flat may be anything; default cat=5
        cat = jnp.where(done, k_zone[row, col], jnp.int32(5))
        rt = (jstop + trial_ndt) * E
        return rt, cat

    rt_chunk, cat_chunk = jax.vmap(per_trial)(trial_keys, crr, ndt)
    return rt_chunk, cat_chunk


@partial(jax.jit, static_argnums=(8, 9, 10, 11, 12))
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
    # calc_LAM has a Python-side `if min_val < -1e-14: raise` that requires a
    # concrete (non-traced) value. Since sis/sig/si are static args, force JAX
    # to evaluate these at trace time so calc_LAM sees concrete arrays.
    with jax.ensure_compile_time_eval():
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
