"""
Smooth-surrogate simulator for differentiable L-BFGS fitting.

Returns soft jstop (continuous expected first-crossing time), soft cat
probabilities (continuous category memberships from softmaxed positions),
and smooth rt. All three are differentiable functions of all params,
unlike the discrete model_a.simulate (whose jax.grad is structurally zero).

Two temperatures:
  tau_step : softness of the first-crossing time. Smaller -> sharper.
  tau_pos  : softness of the position-to-category mapping. Smaller -> sharper.

The mathematical model is identical to model_a/simulate.py; only the
discrete reductions (argmax, where) are replaced with continuous analogs.
"""
from functools import partial

import jax
import jax.numpy as jnp

from shared import prng

# Mirror constants from model_a.simulate
N = 72
NSTEP = 400
E = 10.0
U = 180.0 / 5.0
MC = 3
IPA, IPB, IPC, IPD = 150 / 5, 210 / 5, 100 / 5, 260 / 5
IDX = jnp.arange(1, N + 1, dtype=jnp.float64)


def chol_factor(sig):
    d = IDX[:, None] - IDX[None, :]
    K = jnp.exp(-0.5 * d * d / (sig * sig)) + 1e-12 * jnp.eye(N)
    return jnp.linalg.cholesky(K)


def drift_profile(av, si):
    return av * jnp.exp(-(IDX - U) ** 2 / (2.0 * si * si)) / (si * jnp.sqrt(2.0 * jnp.pi))


def _simulate_chunk_smooth(key, ter, st, cr, crsd, L, v, chunk_size,
                            tau_step, tau_pos):
    ku, kz = jax.random.split(key)
    u = jax.random.uniform(ku, (chunk_size, 10))
    crr = cr + crsd * (u[:, 4] - 0.5)
    ndt = (ter + st * (0.5 - u[:, 9])) / E

    # Pre-generate noise, same as discrete simulator
    z = jax.random.normal(kz, (chunk_size, NSTEP, N))
    noise = z @ L.T
    incr = v[None, None, :] + 5.0 * noise
    incr = incr - incr.mean(axis=-1, keepdims=True)
    a = jnp.cumsum(incr, axis=1)                          # (chunk, NSTEP, N)

    # Soft first-crossing weights — fully log-space for numerical stability.
    # log P(first crossing at t) = log P(cross at t) + sum_{s<t} log P(not cross at s)
    crossed_score = a.max(axis=-1) - crr[:, None]         # (chunk, NSTEP)
    log_post_cross = jax.nn.log_sigmoid(crossed_score / tau_step)   # (chunk, NSTEP)
    log_not_cross = jax.nn.log_sigmoid(-crossed_score / tau_step)   # (chunk, NSTEP)
    log_not_yet = jnp.cumsum(log_not_cross, axis=1)
    log_not_yet_prev = jnp.concatenate([
        jnp.zeros((chunk_size, 1)), log_not_yet[:, :-1]
    ], axis=1)
    log_w = log_post_cross + log_not_yet_prev              # (chunk, NSTEP)
    # Normalize via softmax (numerically stable)
    weights = jax.nn.softmax(log_w, axis=1)

    # Soft jstop = E[first crossing time]
    timesteps = jnp.arange(1, NSTEP + 1, dtype=jnp.float64)
    soft_jstop = (weights * timesteps).sum(axis=1)         # (chunk,)
    rt = (soft_jstop + ndt) * E

    # Soft category probs.
    # tau_pos is in absolute accumulator-value units. With typical
    # a values in [-cr, cr] = [-50, +50] and tau_pos=20, softmax is in the
    # smooth regime everywhere.
    pos_probs = jax.nn.softmax(a / tau_pos, axis=-1)              # (chunk, NSTEP, N)
    # Band masks (N,)
    positions = jnp.arange(1, N + 1, dtype=jnp.float64)
    mask_1 = (positions > IPA) & (positions < IPB)
    mask_3 = (positions <= IPC) | (positions >= IPD)
    mask_2 = ~(mask_1 | mask_3)
    cat_at_step = jnp.stack([
        (pos_probs * mask_1).sum(axis=-1),
        (pos_probs * mask_2).sum(axis=-1),
        (pos_probs * mask_3).sum(axis=-1),
    ], axis=-1)                                            # (chunk, NSTEP, 3)
    # Weighted average over time (using same first-crossing weights)
    cat_probs = (cat_at_step * weights[:, :, None]).sum(axis=1)  # (chunk, 3)
    return rt, cat_probs


@partial(jax.jit, static_argnums=(8, 9))
def simulate_smooth(key, ter, st, cr, crsd, si, sig, av, nsim,
                    chunk_size=256, tau_step=2.0, tau_pos=20.0):
    """
    Differentiable smooth-surrogate simulator. Returns (rt, cat_probs).
    rt        : (nsim,) smooth expected RT in ms.
    cat_probs : (nsim, 3) soft category memberships, ~sum to 1.
    """
    L = chol_factor(sig)
    v = drift_profile(av, si)
    n_chunks = (nsim + chunk_size - 1) // chunk_size
    keys = prng.trial_keys(key, n_chunks)

    def run_chunk(k):
        return _simulate_chunk_smooth(k, ter, st, cr, crsd, L, v, chunk_size,
                                       tau_step, tau_pos)

    rts, cat_probs = jax.lax.map(run_chunk, keys)
    return rts.reshape(-1)[:nsim], cat_probs.reshape(-1, 3)[:nsim]
