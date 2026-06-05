"""
Smooth-surrogate G² objective for differentiable L-BFGS fitting.

Mirrors `model_a/objective.py::fofs_new` but:
- Uses `simulate_smooth` (returns rt + cat_probs, fully differentiable).
- Replaces (rt <= q).sum() indicator with sigmoid((q - rt) / sigma).sum() — smooth CDF.
- Aggregates per-category quantities by weighting with cat_probs instead of (cat == c) mask.

The smooth objective's optimum is biased relative to the discrete one. Use it
to get into the right basin (cheap L-BFGS), then polish with the discrete
simplex on `fofs_new` (`fit_hybrid` in model_a/fit.py).
"""
from functools import partial

import jax
import jax.numpy as jnp

from model_a import simulate_smooth as sim_sm
from model_a.jax_port import clamp
from shared import prng

COND_MAP = [(7, 2), (8, 2), (7, 9), (8, 9)]
MC = 3
NQ = 5
NCUT = 8
PQQ = jnp.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def condition_g2_smooth(rt, cat_probs, obs_prop, obs_count, obs_quant, sigma_cdf):
    """
    Smooth surrogate loss for one condition. Approximates G² with a
    differentiable, NaN-safe formulation:

      - Proportion-match term: weighted squared deviation between predicted
        soft proportions and observed proportions.
      - Median RT-match term: squared deviation between observed median
        (3rd quantile, index 2) and predicted median (via smooth weighted
        rank), weighted by observed count.

    This is biased relative to the full quantile-G² discrete objective, but
    gradient-stable and sufficient to land L-BFGS in the right basin for
    a subsequent discrete-simplex polish pass.

    rt        : (nsim,) RTs (smooth in params via soft_jstop).
    cat_probs : (nsim, 3) soft category memberships (sums ~1).
    obs_prop  : (3,) observed proportions per category.
    obs_count : (3,) observed counts per category.
    obs_quant : (5, 3) observed RT quantiles per category (index 2 = median).
    sigma_cdf : sigmoid bandwidth for soft median.
    """
    mmn = jnp.maximum(obs_count.sum(), 1.0)

    def per_cat(i):
        weight = cat_probs[:, i]                       # (nsim,)
        pred_prop = jnp.mean(weight)
        denom = jnp.maximum(weight.sum(), 1e-6)

        # Proportion-match contribution (weighted by total trials)
        prop_term = mmn * (pred_prop - obs_prop[i]) ** 2

        # Median-RT contribution: only if observed count >= NCUT
        # Smoothly weighted (no hard conditional).
        cat_weight_for_rt = jax.nn.sigmoid((obs_count[i] - NCUT) / 2.0)

        # Predicted "CDF at observed median" — should be ~0.5 if medians match.
        # Use a soft empirical CDF: average of sigmoid((q_median - rt) / sigma)
        # weighted by cat membership.
        q_median = obs_quant[2, i]  # the central quantile is the median
        pred_cdf_at_median = (
            jax.nn.sigmoid((q_median - rt) / sigma_cdf) * weight
        ).sum() / denom
        # If medians match: pred_cdf_at_median ≈ 0.5
        rt_term = mmn * obs_prop[i] * (pred_cdf_at_median - 0.5) ** 2

        return prop_term + cat_weight_for_rt * rt_term

    return jnp.sum(jnp.array([per_cat(i) for i in range(MC)]))


def fofs_smooth(params, data, key, nsim=512, chunk_size=256,
                tau_step=2.0, tau_pos=20.0, sigma_cdf=50.0):
    """
    Smooth-surrogate G² summed across 4 conditions.

    params : (10,) parameter vector — see clamp() docs.
    data   : dict with "prop" (4, 3), "count" (4, 3), "quant" (4, 5, 3).
    key    : JAX typed key.
    nsim   : trials per condition.
    Returns scalar G² (differentiable w.r.t. params).
    """
    p = clamp(params)
    ter, st, sa, si, sig = p[0], p[1], p[3], p[4], p[5]
    drifts = jnp.stack([p[di] for (di, _) in COND_MAP])
    boundaries = jnp.stack([p[bi] for (_, bi) in COND_MAP])

    cond_keys = jnp.stack([prng.split_for_condition(key, ci) for ci in range(4)])

    # simulate_smooth signature:
    #   (key, ter, st, cr, crsd, si, sig, av, nsim, chunk_size, tau_step, tau_pos)
    sim_vmap = jax.vmap(
        sim_sm.simulate_smooth,
        in_axes=(0, None, None, 0, None, None, None, 0, None, None, None, None),
    )
    rts, cat_probs = sim_vmap(
        cond_keys, ter, st, boundaries, sa, si, sig, drifts,
        nsim, chunk_size, tau_step, tau_pos,
    )
    # rts: (4, nsim);  cat_probs: (4, nsim, 3)

    g2_vmap = jax.vmap(condition_g2_smooth, in_axes=(0, 0, 0, 0, 0, None))
    g2_per_cond = g2_vmap(
        rts, cat_probs, data["prop"], data["count"], data["quant"], sigma_cdf,
    )
    return g2_per_cond.sum()
