"""
Stage 5 Section 4 ONLY — verbose fit_simplex_b on H100.

Runs the headline fit with per-evaluation progress printing so we don't have
to fly blind. Drops maxiter from 200 to 100 to cut wall-clock in half while
still doing a real parameter recovery.

Usage on H100 (assumes setup from h100_setup_and_run.sh already done):
    python scripts/h100_section4_verbose.py 2>&1 | tee /workspace/h100_section4.txt
"""
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

print("=" * 72)
print("Stage 5 Section 4 ONLY — verbose fit_simplex_b")
print("=" * 72)
print(f"JAX version : {jax.__version__}")
print(f"JAX devices : {jax.devices()}")
print(f"x64 enabled : {jax.config.read('jax_enable_x64')}")

from model_b import simulate as sim_b, objective as obj_b
from model_b.objective import COND_MAP_B, clamp_b
from shared import prng

NSIM_PROD = 9000
CHUNK = 16     # safe for vmap-of-2 memory
MAXITER = 100  # half of the headline 200, still a real fit

TRUE_PARAMS_B = jnp.array([
    200.0, 50.0, 10.0, 2.0,
    12.0, 10.0, 0.5,
    15.0, 10.0, 8.0,
    14.0, 11.0, 9.0,
])


def _generate_synthetic_data_b(true_params, key, nsim=1024, chunk_size=16):
    p = clamp_b(true_params)
    ter, st, cr, crsd, sis, sig = p[0], p[1], p[2], p[3], p[4], p[5]
    si = 6.0
    props_l, counts_l, quants_l = [], [], []
    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    for ci, (d1, d2, d3) in enumerate(COND_MAP_B):
        ck = prng.split_for_condition(key, ci)
        rt, cat = sim_b.simulate_b(
            ck, ter, st, cr, crsd, p[d1], p[d2], p[d3],
            sis, sig, si, nsim=nsim, chunk_size=chunk_size,
        )
        cat_np = np.asarray(cat); rt_np = np.asarray(rt)
        props = np.array([(cat_np == c).mean() for c in (1, 2, 3, 4, 5)])
        counts = np.array([(cat_np == c).sum() for c in (1, 2, 3, 4, 5)], dtype=np.int64)
        quants = np.zeros((5, 5))
        for ki, c in enumerate((1, 2, 3, 4, 5)):
            mask = cat_np == c
            if mask.sum() >= 5:
                quants[:, ki] = np.quantile(rt_np[mask], qs)
        props_l.append(jnp.asarray(props))
        counts_l.append(jnp.asarray(counts))
        quants_l.append(jnp.asarray(quants))
    return {
        "prop": jnp.stack(props_l),
        "count": jnp.stack(counts_l),
        "quant": jnp.stack(quants_l),
    }


print(f"\nGenerating synthetic data (nsim=1024, chunk={CHUNK}) ...")
key_data = jax.random.key(0)
syn_data = _generate_synthetic_data_b(TRUE_PARAMS_B, key_data,
                                       nsim=1024, chunk_size=CHUNK)
print(f"Data shapes: prop={syn_data['prop'].shape}, count={syn_data['count'].shape}")

np.random.seed(0)
x0 = TRUE_PARAMS_B * jnp.asarray(np.random.uniform(0.9, 1.1, size=13))
print(f"Perturbed start: +/-10% from truth")

# JIT warmup with one call so we don't count compile in the eval loop.
print(f"\nJIT-compiling fofs_b_new at nsim={NSIM_PROD}, chunk={CHUNK} ...")
key_fit = jax.random.key(1)
t0 = time.perf_counter()
_ = float(obj_b.fofs_b_new(jnp.asarray(x0), syn_data, key_fit,
                            nsim=NSIM_PROD, chunk_size=CHUNK))
print(f"Compile + first eval: {time.perf_counter()-t0:.1f}s")

# Per-evaluation counter
n_evals = [0]
t_fit_start = time.perf_counter()
last_print = [t_fit_start]


def loss_with_progress(p_np):
    n_evals[0] += 1
    p = jnp.asarray(p_np)
    val = float(obj_b.fofs_b_new(p, syn_data, key_fit,
                                  nsim=NSIM_PROD, chunk_size=CHUNK))
    now = time.perf_counter()
    # Print every eval (10.7s per eval, low overhead, max visibility)
    elapsed_s = now - t_fit_start
    rate = elapsed_s / n_evals[0]
    print(f"  eval {n_evals[0]:3d}  loss = {val:10.2f}  "
          f"avg = {rate:.1f}s/eval  total = {elapsed_s/60:.1f} min",
          flush=True)
    last_print[0] = now
    return val


print(f"\nRunning Nelder-Mead, maxiter={MAXITER} (will be 200-400 function evals)...")
print(f"Each eval ~10.7s, total expected ~35-70 min")
print("-" * 72)
t0 = time.perf_counter()
res = minimize(
    loss_with_progress, np.asarray(x0),
    method="Nelder-Mead",
    options={"maxiter": MAXITER, "xatol": 1e-7, "fatol": 1e-7, "disp": True},
)
wall = time.perf_counter() - t0
print("-" * 72)

print(f"\nfit wall-clock      : {wall:.1f}s ({wall/60:.1f} min)")
print(f"total function evals : {n_evals[0]}")
print(f"avg s/eval           : {wall/n_evals[0]:.1f}")
print(f"n_iters              : {res.nit}")
print(f"final loss           : {res.fun:.2f}")
print(f"converged            : {res.success}")
print(f"message              : {res.message}")

print(f"\nPer-parameter recovery:")
active_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
for i in active_indices:
    rel_err = abs(float(res.x[i]) - float(TRUE_PARAMS_B[i])) / float(TRUE_PARAMS_B[i])
    label = ["ter", "st", "cr", "crsd", "sis", "sig", "sv",
             "av1c1", "av2c1", "av3c1", "av1c2", "av2c2", "av3c2"][i]
    print(f"  {i:2d} {label:7s} true={float(TRUE_PARAMS_B[i]):7.3f}  "
          f"got={float(res.x[i]):7.3f}  err={rel_err*100:5.1f}%")

print("\n" + "=" * 72)
print("DONE. Paste this output to Claude for write-up.")
print("=" * 72)
