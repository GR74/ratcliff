"""
G² objective for Model B (2D GRF model).

5-category version, summed across 2 conditions. Each condition uses all
three drift bumps (av1, av2, av3); conditions differ only by parameter values.

Param layout (13 elements):
  [ter, st, cr, crsd, sis, sig, sv, av1_c1, av2_c1, av3_c1, av1_c2, av2_c2, av3_c2]

Note: `si` (zone width) is fixed at 6.0, not a fit parameter.
"""
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


def clamp_b(params):
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
    ter = jnp.maximum(params[0], 175.0)
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


def condition_g2_b(rt, cat, obs_prop, obs_count, obs_quant):
    """
    G² contribution from one Model B condition (5 categories).

    Same structure as model_a.objective.condition_g2_vectorized but MC=5.

    rt        : (nsim,) RTs from simulate_b.
    cat       : (nsim,) categories in {1,2,3,4,5}.
    obs_prop  : (5,) observed proportions per category.
    obs_count : (5,) observed counts per category.
    obs_quant : (5, 5) observed RT quantiles per category. (NQ, MC)
    """
    mmn = obs_count.sum()

    def per_cat(i):
        in_cat = (cat == (i + 1))
        pxy = jnp.mean(in_cat)
        denom = jnp.maximum(in_cat.sum(), 1)
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


def fofs_b_new(params, data, key, nsim=512, chunk_size=4, use_kl=False):
    """
    Vectorized G2 objective for Model B, summed across 2 conditions.

    params : (13,) parameter vector. See clamp_b docstring for layout.
    data   : dict with "prop" (2, 5), "count" (2, 5), "quant" (2, 5, 5).
    key    : JAX typed key.
    use_kl : if True, use Stage 6 K-L low-rank GRF inside simulate_b.
    Returns scalar G2.
    """
    # Local imports to avoid hard top-level dependency cycle risk
    from model_b import simulate as sim_b
    from shared import prng

    p = clamp_b(params)
    ter, st, cr, crsd, sis, sig = p[0], p[1], p[2], p[3], p[4], p[5]
    si = 6.0  # zone width, fixed

    # Per-condition drift vectors (2 conditions x 3 drifts each)
    avs = jnp.stack([
        jnp.array([p[d1], p[d2], p[d3]]) for (d1, d2, d3) in COND_MAP_B
    ])  # shape (2, 3)

    # One subkey per condition
    cond_keys = jnp.stack([prng.split_for_condition(key, ci) for ci in range(2)])

    # Convert sis/sig/si to Python floats so they remain static for simulate_b's JIT
    # (simulate_b has static_argnums on positions 8, 9, 10 = sis, sig, si).
    # In a JIT'd outer caller, this conversion must be done at trace time, not runtime.
    # For now (not JIT'd), use Python floats directly.
    sis_py = float(sis)
    sig_py = float(sig)
    si_py = float(si)

    # vmap simulate_b over (key, av1, av2, av3) — others are condition-invariant.
    # simulate_b signature: (key, ter, st, cr, crsd, av1, av2, av3, sis, sig, si, nsim, chunk_size, use_kl)
    sim_vmap = jax.vmap(
        sim_b.simulate_b,
        in_axes=(0, None, None, None, None, 0, 0, 0, None, None, None, None, None, None),
    )
    rts, cats = sim_vmap(
        cond_keys, ter, st, cr, crsd,
        avs[:, 0], avs[:, 1], avs[:, 2],
        sis_py, sig_py, si_py, nsim, chunk_size, use_kl,
    )
    # rts, cats: (2, nsim)

    # vmap condition_g2_b over (rts[ci], cats[ci], data[ci, :])
    g2_vmap = jax.vmap(condition_g2_b, in_axes=(0, 0, 0, 0, 0))
    g2_per_cond = g2_vmap(rts, cats, data["prop"], data["count"], data["quant"])
    return g2_per_cond.sum()
