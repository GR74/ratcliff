# Imported verbatim from reference/twod24_jax.py. Reference copy is the immutable original.
"""
twod24_jax.py
=============
GPU/vectorized port of Roger Ratcliff's spatially-extended diffusion /
competing-accumulator model. Original: gpgsq5deg3twod24.f (Fortran + MKL).

KEY IDEA
--------
Because the Fortran re-centers the accumulator to mean-zero every timestep,
the accumulator path is exactly the cumulative sum of *demeaned per-step
increments*. So the whole simulator collapses to three GPU-friendly ops:

    (1) one correlated-noise matmul   iid normals -> N(0,K) noise
    (2) a cumulative sum over time     the accumulator path
    (3) argmax + threshold reduction   first-passage time & choice position

`one_trial` is written for a single trial; `jax.vmap` stacks
trials x conditions x subjects (x optimizer population) as batch dims, and
`jax.jit` fuses the whole thing into one XLA program. Runs on CPU for
validation; the *same code* saturates an H100.

STATUS: first faithful port. MUST be validated against the patched Fortran
oracle on real `twod24data` before any fit is trusted. See NOTES at bottom
for the two things that need confirmation from Roger -- especially the
`sv`/`as` dummy-argument question, which likely makes across-trial drift
variability inert in his code. To *match his numbers* we replicate his
behavior (SV_ACTIVE = False) until he says otherwise.
"""

import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)  # FP64 for the objective; sim can drop to f32

# ----------------------------------------------------------------------
# Fixed model structure (lifted verbatim from the Fortran)
# ----------------------------------------------------------------------
N      = 72            # spatial positions
NSTEP  = 400           # max timesteps
E      = 10.0          # ms per step
U      = 180.0 / 5.0   # drift bump center  (= 36)
NQ     = 5             # number of RT quantiles
MC     = 3             # response categories
NCUT   = 8             # min count to use full quantile G2 (else lumped term)

# response-category position bands (ipa..ipd in the Fortran, /5)
IPA, IPB, IPC, IPD = 150/5, 210/5, 100/5, 260/5   # 30, 42, 20, 52
# defective bin masses for quantiles {.1,.3,.5,.7,.9}: [.1,.2,.2,.2,.2,.1]
PQQ = jnp.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])

SV_ACTIVE = False   # see NOTES -- his code's `sv` appears inert (name mismatch)

IDX = jnp.arange(1, N + 1, dtype=jnp.float64)


def chol_factor(sig):
    """Cholesky factor L of the GP kernel K; noise = L @ z ~ N(0, K)."""
    d = IDX[:, None] - IDX[None, :]
    K = jnp.exp(-0.5 * d * d / (sig * sig)) + 1e-12 * jnp.eye(N)
    return jnp.linalg.cholesky(K)                      # K = L L^T


def drift_profile(av, si):
    """Spatial drift bump v(i) = av * Normal(i; U, si)."""
    return av * jnp.exp(-(IDX - U) ** 2 / (2.0 * si * si)) / (si * jnp.sqrt(2.0 * jnp.pi))


# ----------------------------------------------------------------------
# One Monte Carlo trial  ->  (RT in ms, response category in {1,2,3})
# ----------------------------------------------------------------------
def one_trial(key, ter, st, cr, crsd, si, sig, sv, L, v):
    ku, kz = jax.random.split(key)
    u = jax.random.uniform(ku, (10,))                  # per-trial uniforms (gu1)
    crr = cr + crsd * (u[4] - 0.5)                     # trial boundary  (gu1(5))
    ssv = (sv * (u[7] - 0.5)) if SV_ACTIVE else 0.0    # drift variability (gu1(8))
    ndt = (ter + st * (0.5 - u[9])) / E                # nondecision time, in steps (gu1(10))
    base = (1.0 + ssv) * v                             # constant part of increment (N,)

    # scan over time: carry only the accumulator state (N,), not the whole path.
    # memory is O(trials * N); the per-step L @ z batches into one GEMM per step.
    def step(carry, sk):
        a, done, jstop, pos, t = carry
        z = jax.random.normal(sk, (N,))
        incr = base + 5.0 * (L @ z)                    # correlated noise, N(0,K)
        incr = incr - incr.mean()                      # demean (his recentering)
        a = a + incr                                   # == cumsum of demeaned increments
        crossed = a.max() > crr
        newly = crossed & (~done)
        jstop = jnp.where(newly, t + 1, jstop)         # 1-based first-passage step
        pos = jnp.where(newly, jnp.argmax(a) + 1, pos)
        return (a, done | crossed, jstop, pos, t + 1), None

    init = (jnp.zeros(N), False, NSTEP, 1, 0)
    (a, _, jstop, pos, _), _ = jax.lax.scan(step, init, jax.random.split(kz, NSTEP))
    rt = (jstop + ndt) * E                             # RT in ms
    cat = jnp.where((pos > IPA) & (pos < IPB), 1,
          jnp.where((pos <= IPC) | (pos >= IPD), 3, 2))
    return rt, cat


# vectorize over trials (shared params/noise-factor), then jit
_trials = jax.vmap(one_trial, in_axes=(0, None, None, None, None, None, None, None, None, None))


@partial(jax.jit, static_argnums=(9,))
def simulate(key, ter, st, cr, crsd, si, sig, av, sv, nsim):
    """Run `nsim` trials for ONE condition. Returns (rt[nsim], cat[nsim])."""
    L = chol_factor(sig)
    v = drift_profile(av, si)
    keys = jax.random.split(key, nsim)
    return _trials(keys, ter, st, cr, crsd, si, sig, sv, L, v)


# ----------------------------------------------------------------------
# Predicted moments + G^2 for one condition (matches FOFS' chi term)
# ----------------------------------------------------------------------
def condition_g2(rt, cat, obs_prop, obs_count, obs_quant):
    """
    rt, cat        : simulated (nsim,) RTs (ms) and categories {1,2,3}
    obs_prop[i]    : observed proportion of category i           (acc)
    obs_count[i]   : observed count of category i                (mn)
    obs_quant[j,i] : observed j-th RT quantile for category i    (rry)
    Returns scalar G^2 contribution (his `chi`, summed over the 3 categories).
    """
    mmn = obs_count.sum()
    chi = 0.0
    for i in range(MC):
        in_cat = (cat == (i + 1))
        pxy = jnp.mean(in_cat)                               # predicted proportion
        # predicted conditional CDF at each observed quantile
        rt_i = jnp.where(in_cat, rt, jnp.inf)
        denom = jnp.maximum(in_cat.sum(), 1)
        qc = jnp.array([(rt_i <= obs_quant[j, i]).sum() / denom for j in range(NQ)])

        def full_term():
            c = mmn * obs_prop[i] * PQQ[0] * jnp.log(
                obs_prop[i] * PQQ[0] / (pxy * qc[0] + 1e-5))
            for j in range(1, NQ):
                yy = jnp.maximum(qc[j] - qc[j - 1], 1e-3)
                c += mmn * obs_prop[i] * PQQ[j] * jnp.log(
                    obs_prop[i] * PQQ[j] / (pxy * yy + 1e-5))
            c += mmn * obs_prop[i] * PQQ[NQ] * jnp.log(
                obs_prop[i] * PQQ[NQ] / (pxy * (1.0 - qc[NQ - 1]) + 1e-5))
            return c

        def lumped_term():
            return mmn * (obs_prop[i] + 0.002) * jnp.log((obs_prop[i] + 0.002) / (pxy + 1e-12))

        chi += jnp.where(obs_count[i] >= NCUT, full_term(), lumped_term())
    return chi


# ----------------------------------------------------------------------
# Full objective: 10 params, 4 conditions (2 drifts x 2 boundaries), summed G^2
#   params: [ter, st, a1, sa, si, sig, sv, drift1, drift2, a2]
#   cond 1: drift1, boundary a1 | cond 2: drift2, a1
#   cond 3: drift1, a2          | cond 4: drift2, a2
# ----------------------------------------------------------------------
COND_MAP = [(7, 2), (8, 2), (7, 9), (8, 9)]   # (drift_idx, boundary_idx) per condition


def fofs(params, data, key, nsim=4000):
    p = clamp(params)
    ter, st, sa, si, sig, sv = p[0], p[1], p[3], p[4], p[5], p[6]
    chi = 0.0
    for ci, (di, bi) in enumerate(COND_MAP):
        rt, cat = simulate(jax.random.fold_in(key, ci),
                           ter, st, p[bi], sa, si, sig, p[di], sv, nsim)
        chi += condition_g2(rt, cat,
                            data["prop"][ci], data["count"][ci], data["quant"][ci])
    return chi


def clamp(p):
    """Parameter floors/caps copied from the top of FOFS."""
    ter = jnp.maximum(p[0], 175.0)
    st  = jnp.clip(p[1], 20.0, ter * 1.5)
    a1  = jnp.maximum(p[2], 1.0)
    sa  = jnp.clip(p[3], 0.01, a1 / 2.0)
    si  = p[4]
    sig = jnp.maximum(p[5], 1.0)
    sv  = jnp.maximum(p[6], 0.3)
    d1, d2 = jnp.maximum(p[7], 0.01), jnp.maximum(p[8], 0.01)
    a2  = jnp.maximum(p[9], 0.01)
    return jnp.array([ter, st, a1, sa, si, sig, sv, d1, d2, a2])


# ----------------------------------------------------------------------
# NOTES (need Roger's confirmation)
# ----------------------------------------------------------------------
# 1. `sv`/`as` dummy-arg mismatch: subroutine accum(...,av,as,...) but the body
#    uses `sv` (an undeclared implicit local), so across-trial drift variability
#    (x(7)) is very likely inert in his code. We replicate that (SV_ACTIVE=False)
#    to MATCH his numbers; flip to True if he confirms it should be active.
# 2. Data layout for twod24data assumed: per subject, 4 condition-lines, each
#    = 3 categories x (prop, count, q1..q5, junk, junk). Parser keys off this.
