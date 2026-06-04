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
    """Placeholder — implemented in Task 3.B.1."""
    raise NotImplementedError


def fofs_new(params, data, key, nsim=4000):
    """Placeholder — implemented in Task 3.C.1."""
    raise NotImplementedError
