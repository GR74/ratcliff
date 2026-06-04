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


def fofs_new(params, data, key, nsim=4000):
    """Placeholder — implemented in Task 3.C.1."""
    raise NotImplementedError
