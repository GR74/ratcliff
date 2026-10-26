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
from model_b import grf_kl
from model_b import grf_separable
from shared import blocked, prng

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


@jax.jit
def drift_bumps(sis: float):
    """
    Three drift Gaussian bumps centered at (UI=50, UJ=80/30/130) with stddev `sis`.

    Returns (v1, v2, v3), each shape (N, M).

    Jitted (pass 3): run eagerly, each call dispatched ~20 tiny grid ops and
    cost ~0.4 s per fit evaluation on CPU, dwarfing small-nsim simulation.
    """
    s3 = 2.0 * sis * sis
    s4 = sis * jnp.sqrt(2.0 * jnp.pi)
    def bump(uj):
        return jnp.exp(-((_J_GRID - uj) ** 2 + (_I_GRID - UI) ** 2) / s3) / s4
    return bump(UJ1), bump(UJ2), bump(UJ3)


def zone_array(si: float = 6.0, inner=0.03, outer=0.0003):
    """
    5-category zone array k(i,j) from the drift bump positions.

    inner/outer are the b thresholds for zone 1 and zones 2-4. The defaults
    are the active lines in benchtwod3mpi.f; it also carries commented-out
    alternatives .05/.016 (ZONES_COMMENTED_OUT).

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
    k = jnp.where(b1 > outer, jnp.int32(2), k)
    k = jnp.where(b1 > inner, jnp.int32(1), k)
    # Around UJ2 -> cat 3
    b2 = b_field(UJ2)
    k = jnp.where(b2 > outer, jnp.int32(3), k)
    # Around UJ3 -> cat 4
    b3 = b_field(UJ3)
    k = jnp.where(b3 > outer, jnp.int32(4), k)
    return k


# benchtwod3mpi.f's commented-out zone thresholds (C  if(b(i,j).gt..05)k(i,j)=1,
# C  if(b(i,j).gt..016)k(i,j)=2..4): radius ~4.5 for zone 1, ~10.1 for 2-4.
ZONES_COMMENTED_OUT = dict(inner=0.05, outer=0.016)


def zone_array_fit(si: float = 6.0):
    """Zones as in the fitting program twod3fitmpi.f (all versions), not bench.

    Thresholds come from the zone Gaussian itself: ar = mean b at distance 4
    and 5 from the centre (b(50,76), b(50,75)), ab = mean b at 14 and 15
    (b(50,66), b(50,65)). For si = 6 that is radius ~4.5 for zone 1 and ~14.5
    for zones 2-4, versus ~7.6 and ~19.7 for benchtwod3mpi.f's fixed .03/.0003.
    """
    s1 = 2.0 * si * si
    s2 = si * jnp.sqrt(2.0 * jnp.pi)

    def at_distance(d):
        return jnp.exp(-(d * d) / s1) / s2

    ar = (at_distance(4.0) + at_distance(5.0)) / 2.0
    ab = (at_distance(14.0) + at_distance(15.0)) / 2.0

    def b_field(uj):
        return jnp.exp(-((_J_GRID - uj) ** 2 + (_I_GRID - UI) ** 2) / s1) / s2

    k = jnp.full((N, M), 5, dtype=jnp.int32)
    b1 = b_field(UJ1)
    k = jnp.where(b1 > ab, jnp.int32(2), k)
    k = jnp.where(b1 > ar, jnp.int32(1), k)
    k = jnp.where(b_field(UJ2) > ab, jnp.int32(3), k)
    k = jnp.where(b_field(UJ3) > ab, jnp.int32(4), k)
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


def _simulate_chunk_b_kl(key, ter, st, cr, crsd, av1, av2, av3,
                          V_kl, v1, v2, v3, k_zone, chunk_size, mode_ids=None):
    """K-L low-rank variant of _simulate_chunk_b.

    Replaces the (chunk, n_fft, 2, n_pad, m_pad) noise + batched 2D FFT with
    (chunk, NSTEP, 2K) noise + a single batched GEMM via V_kl. Memory
    footprint drops ~16x at K=1325 vs the FFT path. Downstream logic
    (drift + cumsum + first-crossing + categorization) is identical.

    K (number of complex modes retained) is recovered from V_kl.shape[1] // 2,
    which is static at trace time since V_kl has a fixed shape.
    """
    ku, kg = jax.random.split(key)
    two_K = V_kl.shape[1]   # static: V_kl shape is fixed at trace time

    # Per-trial uniforms (mirrors Fortran gu1)
    u = jax.random.uniform(ku, (chunk_size, 10))
    crr = (cr + crsd * (u[:, 4] - 0.5)).astype(jnp.float32)
    ndt = (ter + st * (0.5 - u[:, 9])) / E

    # Sample (chunk, NSTEP, 2K) iid normals
    if mode_ids is None:
        z = jax.random.normal(kg, (chunk_size, NSTEP, two_K), dtype=jnp.float32)
    else:
        noise_keys = blocked.indexed_keys(kg, jnp.arange(chunk_size))
        z = grf_kl.mode_normal_blocks(noise_keys, mode_ids, NSTEP, jnp.float32)

    # One batched GEMM: (chunk, NSTEP, 2K) @ (2K, NM) -> (chunk, NSTEP, NM)
    grf_flat = jnp.einsum("csk,Nk->csN", z, V_kl)
    grf_path = grf_flat.reshape(chunk_size, NSTEP, N, M)

    # Drift + accumulator (identical to _simulate_chunk_b)
    v1_f32 = v1.astype(jnp.float32)
    v2_f32 = v2.astype(jnp.float32)
    v3_f32 = v3.astype(jnp.float32)
    av1_f32 = jnp.float32(av1)
    av2_f32 = jnp.float32(av2)
    av3_f32 = jnp.float32(av3)
    drift_const = av1_f32 * v1_f32 + av2_f32 * v2_f32 + av3_f32 * v3_f32
    incr = drift_const[None, None, :, :] + grf_path

    incr = incr - incr.mean(axis=(-2, -1), keepdims=True)
    a = jnp.cumsum(incr, axis=1)

    max_per_step = a.reshape(chunk_size, NSTEP, -1).max(axis=-1)
    crossed = max_per_step > crr[:, None]
    any_crossed = crossed.any(axis=1)
    jstop = jnp.where(any_crossed, jnp.argmax(crossed, axis=1) + 1, NSTEP)

    a_at_crossing = a[jnp.arange(chunk_size), jstop - 1, :, :]
    pos_flat = jnp.argmax(a_at_crossing.reshape(chunk_size, -1), axis=-1)
    row = pos_flat // M
    col = pos_flat % M
    cat = jnp.where(any_crossed, k_zone[row, col], jnp.int32(5))
    rt = (jstop.astype(jnp.float64) + ndt) * E

    return rt, cat


@partial(jax.jit, static_argnums=(11, 12))
def _simulate_b_fft(key, ter, st, cr, crsd, av1, av2, av3,
                    sis, sig, si, nsim, chunk_size):
    """FFT-path inner core for simulate_b. Bit-for-bit identical to the
    pre-Stage-6 simulate_b body."""
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


@partial(jax.jit, static_argnums=(11, 12))
def _simulate_b_kl_inner(key, ter, st, cr, crsd, av1, av2, av3,
                          V_kl, sis, si, nsim, chunk_size, mode_ids=None):
    """K-L-path inner core for simulate_b. V_kl is passed as a traced array
    (built outside jit by grf_kl.calc_kl_basis, which uses numpy)."""
    v1, v2, v3 = drift_bumps(sis=sis)
    k_zone = zone_array(si=si)

    n_chunks = (nsim + chunk_size - 1) // chunk_size
    keys = prng.trial_keys(key, n_chunks)

    def run_chunk(k):
        return _simulate_chunk_b_kl(
            k, ter, st, cr, crsd, av1, av2, av3,
            V_kl, v1, v2, v3, k_zone, chunk_size, mode_ids,
        )

    rts, cats = jax.lax.map(run_chunk, keys)
    return rts.reshape(-1)[:nsim], cats.reshape(-1)[:nsim]


def simulate_b(key, ter, st, cr, crsd, av1, av2, av3,
               sis, sig, si, nsim, chunk_size=4, use_kl=False,
               stable_kl_noise=False):
    """
    Run `nsim` Model B trials with the given parameters.

    Parameters:
        sis : drift bump width
        sig : GRF correlation length (s1=s2=sig)
        si  : zone-array width
        use_kl : if True, use K-L low-rank GRF path (model_b/grf_kl.py) for
                 reduced memory footprint at large chunk sizes. Default False
                 preserves the original FFT path bit-for-bit.
    Returns (rt, cat) each shape (nsim,). cat in {1..5}. RT in ms.

    Memory note: chunk_size=4 default for laptop CPU. H100 can use much larger.

    Note: this outer wrapper is intentionally NOT jit-compiled. When use_kl=True
    the K-L basis is constructed via numpy in grf_kl.calc_kl_basis, which cannot
    run inside jax tracing. The compute-heavy inner cores are jitted.
    """
    if stable_kl_noise and not use_kl:
        raise ValueError("stable K-L noise requires use_kl=True")
    if use_kl:
        # pad_to_k_max=True keeps V_kl shape constant across sig values, so
        # _simulate_b_kl_inner JIT compiles ONCE and stays cached across the
        # fit instead of recompiling per new sig vertex scipy visits.
        # Cost: ~50% larger GEMM (K=k_max=2000 vs ~1325 at sig=10); benefit:
        # no JIT recompile per eval (~2-4 sec saved per new sig).
        result = grf_kl.calc_kl_basis(
            sig=float(sig), n=N, m=M, pad_to_k_max=True,
            return_mode_ids=stable_kl_noise,
        )
        V_kl = result[0]
        mode_ids = result[3] if stable_kl_noise else None
        return _simulate_b_kl_inner(
            key, ter, st, cr, crsd, av1, av2, av3,
            V_kl, sis, si, nsim, chunk_size, mode_ids,
        )
    return _simulate_b_fft(
        key, ter, st, cr, crsd, av1, av2, av3,
        sis, sig, si, nsim, chunk_size,
    )


@partial(jax.jit, static_argnums=(8, 9, 10, 11, 12, 13, 14, 17))
def _simulate_b_blocked_core(key, ter, st, cr, crsd, drift, k_zone, basis,
                             nsim, chunk_size, time_block_size,
                             stop_early, compact, use_kl, cumulative_kl=False,
                             mode_ids=None, sv=0.0, fortran_variant=None):
    n_chunks = (nsim + chunk_size - 1) // chunk_size

    def sample_noise(keys):
        if use_kl:
            # Keep normal draws fixed across basis capacities; padding removal
            # must not silently replace the retained modes' random weights.
            if mode_ids is None:
                width = (2 * basis[3].shape[0] if isinstance(basis, tuple)
                         else basis.shape[1])
                z = blocked.normal_blocks(keys, time_block_size, 2 * grf_kl.DEFAULT_K_MAX,
                                           jnp.float32)[..., :width]
            else:
                z = grf_kl.mode_normal_blocks(keys, mode_ids, time_block_size, jnp.float32)
            if cumulative_kl:
                z = jnp.cumsum(z, axis=1)
            if isinstance(basis, tuple):   # separable K-L (grf_separable)
                return grf_separable.project(z, *basis)
            return jnp.einsum("bsk,Nk->bsN", z, basis)
        n_pad, m_pad = basis.shape
        n_fft = (time_block_size + 1) // 2
        z = jax.vmap(lambda k: jax.random.normal(
            k, (n_fft, 2, n_pad, m_pad), dtype=jnp.float32,
        ))(keys)
        fields = jnp.fft.fft2(basis * (z[:, :, 0] + 1j * z[:, :, 1]))
        paired = jnp.stack([fields[:, :, :N, :M].real,
                            fields[:, :, :N, :M].imag], axis=2)
        return paired.reshape(len(keys), n_fft * 2, N * M)[:, :time_block_size]

    def run_chunk(chunk_index):
        indices = chunk_index * chunk_size + jnp.arange(chunk_size)
        u, noise_keys = blocked.trial_inputs(key, indices, jnp.float64)
        thresholds = (cr + crsd * (u[:, 4] - 0.5)).astype(jnp.float32)
        ndt = (ter + st * (0.5 - u[:, 9])) / E
        fortran = fortran_variant is not None
        # Fortran gu1(8): per-trial drift scale vs (benchtwod3mpi.f and twod3fitmpi.f).
        drift_scale = (1.0 + sv * (u[:, 7] - 0.5)) if fortran else None
        result = blocked.first_passage(
            noise_keys, thresholds, drift, indices < nsim, sample_noise,
            n_steps=NSTEP, block_size=time_block_size,
            stop_early=stop_early, compact=compact,
            noise_is_path=cumulative_kl,
            drift_scale=drift_scale, center_noise=fortran_variant != "fit",
        )
        if fortran:
            # Fortran: a trial that never crosses ends its loop at j = nstep+1
            # and is categorised by where its maximum sits at that point.
            cat = jnp.where(result.crossed, k_zone[result.positions],
                            k_zone[result.final_positions])
            steps = jnp.where(result.crossed, result.steps, NSTEP + 1)
        else:
            cat = jnp.where(result.crossed, k_zone[result.positions], 5)
            steps = result.steps
        rt = (steps.astype(jnp.float64) + ndt) * E
        return rt, cat, result.computed_trial_steps

    rts, cats, work = jax.lax.map(run_chunk, jnp.arange(n_chunks))
    return rts.reshape(-1)[:nsim], cats.reshape(-1)[:nsim], work.sum()


@jax.jit
def _drift_field(sis, av1, av2, av3):
    """Weighted sum of the three drift bumps, compiled once (see drift_bumps)."""
    v1, v2, v3 = drift_bumps(sis=sis)
    return (jnp.asarray(av1, dtype=jnp.float32) * v1.astype(jnp.float32)
            + jnp.asarray(av2, dtype=jnp.float32) * v2.astype(jnp.float32)
            + jnp.asarray(av3, dtype=jnp.float32) * v3.astype(jnp.float32))


def simulate_b_blocked_with_work(key, ter, st, cr, crsd, av1, av2, av3,
                                 sis, sig, si, nsim, chunk_size=64,
                                 use_kl=True, time_block_size=32,
                                 stop_early=True, compact=True,
                                 kl_bucketed=False, cumulative_kl=False,
                                 stable_kl_noise=False, separable_kl=False,
                                 sv=0.0, fortran_variant=None):
    """Optional blocked Model B with unchanged FFT/K-L covariance/categories.

    stop_early=False is the paired full-horizon control. Keep time_block_size
    fixed in paired comparisons: another block size changes the noise layout.

    fortran_variant (default None = the original port, unchanged):
      "bench": benchtwod3mpi.f, the program behind Ratcliff's TACC subject-1
               fit: adds the per-trial drift scale sv and the Fortran
               non-crosser rule (zones and joint centring already match).
      "fit":   twod3fitmpi.f (Aug 2017): as "bench", plus its smaller zones
               (zone_array_fit) and drift-only centring.
      "bench_alt_zones": as "bench" with its commented-out zone thresholds.
      "fit1":  twod3fitmpi1.f (Jan 19 2017, two days after the TACC log, same
               6-restart schedule): as "bench" with the smaller zones only,
               joint centring kept.
    """
    if nsim < 1 or chunk_size < 1:
        raise ValueError("nsim and chunk_size must be positive")
    if (kl_bucketed or cumulative_kl or stable_kl_noise or separable_kl) and not use_kl:
        raise ValueError("K-L extensions require use_kl=True")
    if separable_kl and kl_bucketed:
        raise ValueError("separable_kl always uses k_max capacity")
    drift = _drift_field(sis, av1, av2, av3)
    if fortran_variant not in (None, "bench", "bench_alt_zones", "fit", "fit1"):
        raise ValueError(f"unknown fortran_variant {fortran_variant!r}")
    if fortran_variant == "fit":
        drift = drift - drift.mean()   # twod3fitmpi: a += drift - mean(drift) + noise
    if fortran_variant in ("fit", "fit1"):
        zones = zone_array_fit(si)
    elif fortran_variant == "bench_alt_zones":
        zones = zone_array(si, **ZONES_COMMENTED_OUT)
    else:
        zones = zone_array(si)
    mode_ids = None
    if separable_kl:
        # Same modes, scaling and random coefficients as the dense K-L basis,
        # evaluated as Re(P W Q^T) (see grf_separable). Builds in milliseconds.
        if separable_kl is True:   # pass-3 default: 99%, k_max 2000
            sb = grf_separable.separable_basis(float(sig), n=N, m=M)
        else:                       # a float: that retained-variance fraction
            sb = grf_separable.separable_basis(
                float(sig), n=N, m=M, k_max=grf_separable.K_MAX_HIGH_VARIANCE,
                variance_threshold=float(separable_kl))
        basis = sb.arrays
        mode_ids = sb.mode_ids if stable_kl_noise else None
    elif use_kl:
        result = grf_kl.calc_kl_basis(
            sig=float(sig), n=N, m=M, pad_to_k_max=True,
            mode_buckets=grf_kl.DEFAULT_MODE_BUCKETS if kl_bucketed else None,
            return_mode_ids=stable_kl_noise,
        )
        basis = result[0]
        mode_ids = result[3] if stable_kl_noise else None
    else:
        basis = grf.calc_LAM(s1=sig, s2=sig).astype(jnp.float32)
    return _simulate_b_blocked_core(
        key, ter, st, cr, crsd, drift.reshape(-1), zones.reshape(-1),
        basis, nsim, chunk_size, time_block_size, stop_early, compact, use_kl,
        cumulative_kl, mode_ids, jnp.asarray(sv, jnp.float64), fortran_variant,
    )


def simulate_b_blocked(key, ter, st, cr, crsd, av1, av2, av3,
                       sis, sig, si, nsim, chunk_size=64, use_kl=True,
                       time_block_size=32, stop_early=True, compact=True,
                       kl_bucketed=False, cumulative_kl=False, stable_kl_noise=False,
                       separable_kl=False, sv=0.0, fortran_variant=None):
    """RT/category-only interface for optional time blocking."""
    rt, cat, _work = simulate_b_blocked_with_work(
        key, ter, st, cr, crsd, av1, av2, av3, sis, sig, si, nsim,
        chunk_size, use_kl, time_block_size, stop_early, compact,
        kl_bucketed, cumulative_kl, stable_kl_noise, separable_kl,
        sv, fortran_variant,
    )
    return rt, cat
