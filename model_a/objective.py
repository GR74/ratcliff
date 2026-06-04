"""
Vectorized G² objective for the Ratcliff 1D model.

Replaces the Python loops in `jax_port.fofs` and `jax_port.condition_g2` with
`jax.vmap` over conditions and `jnp.where`-based category aggregation. The
output `fofs_new(params, data, key, nsim)` matches `jax_port.fofs`'s contract
and is differentiable via `jax.grad`.
"""
from functools import partial

import jax
import jax.numpy as jnp

from model_a import simulate as sim_new
from model_a.jax_port import clamp  # reuse the bounds function
from shared import prng

# 4 fitting conditions: (drift_param_idx, boundary_param_idx) per condition.
# Matches jax_port.fofs:
#   cond 1: drift=params[7], boundary=params[2]  (drift1, a1)
#   cond 2: drift=params[8], boundary=params[2]  (drift2, a1)
#   cond 3: drift=params[7], boundary=params[9]  (drift1, a2)
#   cond 4: drift=params[8], boundary=params[9]  (drift2, a2)
COND_MAP = [(7, 2), (8, 2), (7, 9), (8, 9)]

MC = 3
NQ = 5
NCUT = 8
PQQ = jnp.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def condition_g2_vectorized(rt, cat, obs_prop, obs_count, obs_quant):
    """
    Compute G² contribution from one condition.

    rt        : (nsim,) RTs from simulate.
    cat       : (nsim,) categories in {1,2,3}.
    obs_prop  : (3,) observed proportions per category.
    obs_count : (3,) observed counts per category.
    obs_quant : (5, 3) observed RT quantiles per category.

    Returns scalar G² (sum over the 3 categories).
    """
    mmn = obs_count.sum()

    def per_cat(i):
        in_cat = (cat == (i + 1))
        pxy = jnp.mean(in_cat)
        denom = jnp.maximum(in_cat.sum(), 1)
        rt_i = jnp.where(in_cat, rt, jnp.inf)
        # qc[j] = empirical CDF of in-cat RTs at obs_quant[j, i], length NQ=5
        qc = jnp.array([(rt_i <= obs_quant[j, i]).sum() / denom for j in range(NQ)])

        # full quantile-by-quantile G² term
        c_full = mmn * obs_prop[i] * PQQ[0] * jnp.log(
            obs_prop[i] * PQQ[0] / (pxy * qc[0] + 1e-5))
        for j in range(1, NQ):
            yy = jnp.maximum(qc[j] - qc[j - 1], 1e-3)
            c_full = c_full + mmn * obs_prop[i] * PQQ[j] * jnp.log(
                obs_prop[i] * PQQ[j] / (pxy * yy + 1e-5))
        c_full = c_full + mmn * obs_prop[i] * PQQ[NQ] * jnp.log(
            obs_prop[i] * PQQ[NQ] / (pxy * (1.0 - qc[NQ - 1]) + 1e-5))

        # lumped term for small-count categories
        c_lumped = mmn * (obs_prop[i] + 0.002) * jnp.log(
            (obs_prop[i] + 0.002) / (pxy + 1e-12))

        return jnp.where(obs_count[i] >= NCUT, c_full, c_lumped)

    contribs = jnp.array([per_cat(i) for i in range(MC)])
    return contribs.sum()


def fofs_new(params, data, key, nsim=4000, chunk_size=256):
    """
    Vectorized G² objective summed across 4 conditions.

    params : (10,) parameter vector — see clamp() docs.
    data   : dict with "prop" (4,3), "count" (4,3), "quant" (4,5,3).
    key    : JAX typed key.
    nsim   : trials per condition.
    chunk_size : trial chunk for the Stage 2 simulator.

    Returns scalar G² (sum over conditions).
    """
    p = clamp(params)
    ter, st, sa, si, sig = p[0], p[1], p[3], p[4], p[5]
    # Per-condition (drift, boundary) from COND_MAP
    drifts = jnp.stack([p[di] for (di, _) in COND_MAP])     # (4,)
    boundaries = jnp.stack([p[bi] for (_, bi) in COND_MAP])  # (4,)

    # One subkey per condition, deterministic from `key`
    cond_keys = jnp.stack([prng.split_for_condition(key, ci) for ci in range(4)])

    # vmap simulate over (key, cr, av); other params are condition-invariant.
    # simulate signature: (key, ter, st, cr, crsd, si, sig, av, nsim, chunk_size)
    sim_vmap = jax.vmap(
        sim_new.simulate,
        in_axes=(0, None, None, 0, None, None, None, 0, None, None),
    )
    rts, cats = sim_vmap(
        cond_keys, ter, st, boundaries, sa, si, sig, drifts, nsim, chunk_size
    )
    # rts, cats: (4, nsim)

    # vmap condition_g2 over the 4 (rts[ci], cats[ci], data[ci, :])
    g2_vmap = jax.vmap(
        condition_g2_vectorized,
        in_axes=(0, 0, 0, 0, 0),
    )
    g2_per_cond = g2_vmap(
        rts, cats, data["prop"], data["count"], data["quant"]
    )
    return g2_per_cond.sum()
