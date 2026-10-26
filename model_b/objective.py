"""
G² objective for Model B (2D GRF model).

5-category version, summed across 2 conditions. Each condition uses all
three drift bumps (av1, av2, av3); conditions differ only by parameter values.

Param layout (13 elements):
  [ter, st, cr, crsd, sis, sig, sv, av1_c1, av2_c1, av3_c1, av1_c2, av2_c2, av3_c2]

Note: `si` (zone width) is fixed at 6.0, not a fit parameter.
"""
from functools import partial

import jax
import jax.numpy as jnp

# Two conditions, each using 3 drifts. Param indices (di1, di2, di3) per cond.
COND_MAP_B = [
    (7, 8, 9),     # cond 1: av1=p[7], av2=p[8], av3=p[9]
    (10, 11, 12),  # cond 2: av1=p[10], av2=p[11], av3=p[12]
]

MC = 5
NQ = 5
NCUT = 10  # Fortran benchtwod3mpi uses ncut=10 (vs ncut=8 for Model A)
PQQ = jnp.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def clamp_b(params, ter_min=175.0):
    """
    Bounds for Model B's 13-parameter vector.

    Mirrors benchtwod3mpi.f::fofs (lines 158-184):
      ter >= 175
      st  in [10, 1.5*ter]
      cr (a1) >= 1
      crsd (sa) in [0.01, cr/2]
      sig in [0.2, 17.0]   (the 17.0 cap is conservative for 100x160 PD)
      sv  >= 0.2
      av1..av6 (indices 7-12) >= 0.01
    """
    ter = jnp.maximum(params[0], ter_min)   # twod3fitmpi.f uses 150
    st = jnp.clip(params[1], 10.0, ter * 1.5)
    cr = jnp.maximum(params[2], 1.0)
    crsd = jnp.clip(params[3], 0.01, cr / 2.0)
    sis = params[4]
    sig = jnp.clip(params[5], 0.2, 17.0)
    sv = jnp.maximum(params[6], 0.2)
    drifts = jnp.maximum(params[7:13], 0.01)
    return jnp.concatenate([
        jnp.array([ter, st, cr, crsd, sis, sig, sv]),
        drifts,
    ])


E_MS = 10.0      # ms per simulator step (simulate.E)
# ter floor per Fortran program: twod3fitmpi.f 150, twod3fitmpi1.f 160, else 175.
TER_MIN_MS = {"fit": 150.0, "fit1": 160.0}
NSTEP_BINS = 400  # Fortran nstep: RT histogram bins used by qant1


def _cdf_fortran_binned(rt, in_cat, q_ms):
    """CDF at observed quantile q_ms as twod3fitmpi.f/accum + qant1 compute it.

    RTs (in steps, ndt included) are truncated to integer bins 1..400; bins
    beyond 400 are ignored. rr(1) = 0 and rr(i) = rr(i-1) + r(i), normalised
    by rr(400); the CDF at art = q/E is linearly interpolated between
    rr(int(art)) and rr(int(art)+1), floored at 0.01.
    """
    bins = jnp.floor(rt / E_MS).astype(jnp.int32)
    counts = jnp.zeros(NSTEP_BINS + 2, rt.dtype).at[
        jnp.clip(bins, 0, NSTEP_BINS + 1)].add(in_cat.astype(rt.dtype))
    r = counts[1:NSTEP_BINS + 1]                      # r(1..400)
    rr = jnp.concatenate([jnp.zeros(1, rt.dtype), jnp.cumsum(r[1:])])
    rr = rr / jnp.maximum(rr[-1], 1e-12)
    art = q_ms / E_MS
    irt = jnp.clip(jnp.floor(art).astype(jnp.int32), 1, NSTEP_BINS - 1)
    q = rr[irt - 1] + (art - irt) * (rr[irt] - rr[irt - 1])
    return jnp.maximum(q, 0.01)


def condition_g2_b(rt, cat, obs_prop, obs_count, obs_quant, soft_tau_ms=0.0,
                   fortran_binned=False):
    """
    G² contribution from one Model B condition (5 categories).

    Same structure as model_a.objective.condition_g2_vectorized but MC=5.

    rt        : (nsim,) RTs from simulate_b.
    cat       : (nsim,) categories in {1,2,3,4,5}.
    obs_prop  : (5,) observed proportions per category.
    obs_count : (5,) observed counts per category.
    obs_quant : (5, 5) observed RT quantiles per category. (NQ, MC)
    soft_tau_ms : 0 keeps the hard count of rt <= quantile (Fortran-identical).
        > 0 replaces each indicator by sigmoid((quantile - rt) / tau), so the
        loss changes smoothly as simulated RTs move past an observed quantile
        (pass 3; tau -> 0 recovers the hard count). Category assignment stays hard.
    fortran_binned : use the Fortran binned + interpolated CDF instead
        (see _cdf_fortran_binned); overrides soft_tau_ms.
    """
    mmn = obs_count.sum()

    def per_cat(i):
        in_cat = (cat == (i + 1))
        pxy = jnp.mean(in_cat)
        denom = jnp.maximum(in_cat.sum(), 1)
        if fortran_binned:
            qc = jnp.array([_cdf_fortran_binned(rt, in_cat, obs_quant[j, i])
                            for j in range(NQ)])
        elif soft_tau_ms > 0:
            w = in_cat.astype(rt.dtype)
            qc = jnp.array([(w * jax.nn.sigmoid((obs_quant[j, i] - rt) / soft_tau_ms)
                             ).sum() / denom for j in range(NQ)])
        else:
            rt_i = jnp.where(in_cat, rt, jnp.inf)
            qc = jnp.array([(rt_i <= obs_quant[j, i]).sum() / denom for j in range(NQ)])

        c_full = mmn * obs_prop[i] * PQQ[0] * jnp.log(
            obs_prop[i] * PQQ[0] / (pxy * qc[0] + 1e-5))
        for j in range(1, NQ):
            yy = jnp.maximum(qc[j] - qc[j - 1], 1e-3)
            c_full = c_full + mmn * obs_prop[i] * PQQ[j] * jnp.log(
                obs_prop[i] * PQQ[j] / (pxy * yy + 1e-5))
        c_full = c_full + mmn * obs_prop[i] * PQQ[NQ] * jnp.log(
            obs_prop[i] * PQQ[NQ] / (pxy * (1.0 - qc[NQ - 1]) + 1e-5))

        c_lumped = mmn * (obs_prop[i] + 0.002) * jnp.log(
            (obs_prop[i] + 0.002) / (pxy + 1e-12))
        return jnp.where(obs_count[i] >= NCUT, c_full, c_lumped)

    return jnp.array([per_cat(i) for i in range(MC)]).sum()


def simulate_conditions_b(params, key, nsim=512, chunk_size=4, use_kl=False,
                time_block_size=0, stop_early=True, compact=True,
                kl_bucketed=False, cumulative_kl=False, stable_kl_noise=False,
                separable_kl=False, fortran_variant=None):
    """Simulated (rts, cats), each (2, nsim), for both conditions of Model B.

    The simulation half of fofs_b_new; see it for the arguments.
    fortran_variant "bench" or "fit" follows benchtwod3mpi.f or twod3fitmpi.f
    (see simulate_b_blocked_with_work) and makes sv (params[6]) active; "fit"
    also lowers the ter bound to 150, as that program does.
    """
    # Local imports to avoid hard top-level dependency cycle risk
    from model_b import simulate as sim_b
    from shared import prng

    if (kl_bucketed or cumulative_kl) and (not time_block_size or not use_kl):
        raise ValueError("K-L extensions require blocked simulation and use_kl=True")
    if (stable_kl_noise or separable_kl) and not use_kl:
        raise ValueError("stable/separable K-L noise requires use_kl=True")
    if separable_kl and not time_block_size:
        raise ValueError("separable K-L requires blocked simulation")
    if fortran_variant is not None and not time_block_size:
        raise ValueError("fortran_variant requires blocked simulation")

    p = clamp_b(params, ter_min=TER_MIN_MS.get(fortran_variant, 175.0))
    ter, st, cr, crsd, sis, sig, sv = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    si = 6.0  # zone width, fixed

    # Per-condition drift vectors (2 conditions x 3 drifts each)
    avs = jnp.stack([
        jnp.array([p[d1], p[d2], p[d3]]) for (d1, d2, d3) in COND_MAP_B
    ])  # shape (2, 3)

    # One subkey per condition
    cond_keys = jnp.stack([prng.split_for_condition(key, ci) for ci in range(2)])

    # sis and si flow through as JAX tensors so the JIT inner cores trace
    # once and reuse the compiled graph across distinct parameter values
    # (drift_bumps and zone_array are pure jnp ops). sig stays a Python
    # float because the K-L path's calc_kl_basis uses numpy and must see
    # a concrete value; the FFT path's calc_LAM still accepts it fine.
    sis_arr = jnp.asarray(sis, dtype=jnp.float64)
    si_arr = jnp.asarray(si, dtype=jnp.float64)
    sig_py = float(sig)

    # vmap simulate_b over (key, av1, av2, av3) — others are condition-invariant.
    # simulate_b signature: (key, ter, st, cr, crsd, av1, av2, av3, sis, sig, si, nsim, chunk_size, use_kl)
    if time_block_size:
        # Preserve runtime branch selection; vmap can evaluate all branches.
        outputs = [sim_b.simulate_b_blocked(
            cond_keys[ci], ter, st, cr, crsd,
            avs[ci, 0], avs[ci, 1], avs[ci, 2],
            sis_arr, sig_py, si_arr, nsim, chunk_size, use_kl,
            time_block_size, stop_early, compact,
            kl_bucketed, cumulative_kl, stable_kl_noise, separable_kl,
            sv, fortran_variant,
        ) for ci in range(2)]
        rts = jnp.stack([out[0] for out in outputs])
        cats = jnp.stack([out[1] for out in outputs])
    else:
        sim_vmap = jax.vmap(
            sim_b.simulate_b,
            in_axes=(0, None, None, None, None, 0, 0, 0, None, None, None, None, None, None, None),
        )
        rts, cats = sim_vmap(
            cond_keys, ter, st, cr, crsd,
            avs[:, 0], avs[:, 1], avs[:, 2],
            sis_arr, sig_py, si_arr, nsim, chunk_size, use_kl, stable_kl_noise,
        )
    # rts, cats: (2, nsim)

    return rts, cats


def fofs_b_new(params, data, key, nsim=512, chunk_size=4, use_kl=False,
                time_block_size=0, stop_early=True, compact=True,
                kl_bucketed=False, cumulative_kl=False, stable_kl_noise=False,
                separable_kl=False, soft_tau_ms=0.0, fortran_variant=None):
    """
    Vectorized G2 objective for Model B, summed across 2 conditions.

    params : (13,) parameter vector. See clamp_b docstring for layout.
    data   : dict with "prop" (2, 5), "count" (2, 5), "quant" (2, 5, 5).
    key    : JAX typed key.
    use_kl : if True, use Stage 6 K-L low-rank GRF inside simulate_b.
    soft_tau_ms : soft-count width for condition_g2_b (0 = hard count).
    fortran_variant : None (original port), "bench" (benchtwod3mpi.f, the
        program behind Ratcliff's TACC fit) or "fit" (twod3fitmpi*.f); both
        Fortran variants also use the Fortran binned CDF for G2.
    Returns scalar G2.
    """
    rts, cats = simulate_conditions_b(
        params, key, nsim, chunk_size, use_kl, time_block_size, stop_early,
        compact, kl_bucketed, cumulative_kl, stable_kl_noise, separable_kl,
        fortran_variant)
    return _g2_total(rts, cats, data["prop"], data["count"], data["quant"],
                     float(soft_tau_ms), fortran_variant is not None)


@partial(jax.jit, static_argnums=(5, 6))
def _g2_total(rts, cats, prop, count, quant, soft_tau_ms=0.0, fortran_binned=False):
    """Sum of condition_g2_b over conditions, compiled once per nsim.

    Pass 3: the eager vmap dispatched ~20,000 tiny ops per evaluation and took
    ~60% of a small-nsim fit evaluation on CPU. Same math, now one XLA program.
    """
    return jax.vmap(partial(condition_g2_b, soft_tau_ms=soft_tau_ms,
                            fortran_binned=fortran_binned),
                    in_axes=(0, 0, 0, 0, 0))(rts, cats, prop, count, quant).sum()
