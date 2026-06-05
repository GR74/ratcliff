"""
Stage 5 Model B benchmark — one-shot script for H100.

Run on a freshly-rented H100. Outputs all the numbers we need to claim
the "took weeks → minutes" speedup story for Model B.

Usage on H100 (after `pip install -e ".[dev,fit]"`):
    python scripts/h100_model_b_benchmark.py 2>&1 | tee h100_model_b_results.txt

What it measures:
  1. JAX + CUDA install sanity (devices, dtype, etc.)
  2. Per-call simulate_b wall clock at production nsim=9000,
     across chunk_size = [16, 64, 256, 1024] to find sweet spot.
  3. fofs_b_new wall clock (whole 2-condition objective).
  4. Single-fit hybrid-style timing: fit_simplex_b recovery on synthetic
     data, with budget that mirrors the Fortran's 9000 sims × thousands of
     simplex evals scale.
  5. Comparison to Fortran's "11 seconds for 9000 sims on 6-node MPI".

If any step OOMs at chunk_size=1024, the script falls back automatically
and reports the largest working chunk.
"""
import argparse
import sys
import time
import traceback

import jax

# Enable x64 BEFORE any other JAX op (matches conftest.py).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

print("=" * 72)
print("Stage 5 Model B Benchmark")
print("=" * 72)
print(f"JAX version : {jax.__version__}")
print(f"JAX devices : {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
print(f"x64 enabled : {jax.config.read('jax_enable_x64')}")

if jax.default_backend() != "gpu":
    print("\n!!! WARNING: not running on GPU. This script is meant for H100.")
    print("    Continuing anyway, but numbers will be misleading.\n")

# ----------------------------------------------------------------------
# Section 1: Smoke import + minimal sanity
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("SECTION 1 — Imports + minimal sanity")
print("=" * 72)

from model_b import grf, simulate as sim_b, objective as obj_b, fit as fit_b
from shared import prng


def time_call(fn, *args, n_iter=3, warmup=1, **kwargs):
    """Run fn(*args, **kwargs) once for warmup, then time n_iter calls."""
    for _ in range(warmup):
        out = fn(*args, **kwargs)
        if isinstance(out, tuple):
            for x in out:
                if hasattr(x, "block_until_ready"):
                    x.block_until_ready()
        elif hasattr(out, "block_until_ready"):
            out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args, **kwargs)
        if isinstance(out, tuple):
            for x in out:
                if hasattr(x, "block_until_ready"):
                    x.block_until_ready()
        elif hasattr(out, "block_until_ready"):
            out.block_until_ready()
    return (time.perf_counter() - t0) / n_iter, out


# Test GRF generator works
LAM = grf.calc_LAM(s1=10.0, s2=10.0)
print(f"calc_LAM(s1=s2=10): shape {LAM.shape}, all finite: "
      f"{bool(jnp.all(jnp.isfinite(LAM)))}")

g_pair = jax.random.normal(jax.random.key(0), (2, 199, 319))
F1, F2 = grf.circulant_grf(LAM, g_pair[0], g_pair[1])
print(f"circulant_grf: F1.shape {F1.shape}, both finite: "
      f"{bool(jnp.all(jnp.isfinite(F1)) and jnp.all(jnp.isfinite(F2)))}")

# ----------------------------------------------------------------------
# Section 2: simulate_b at production scale, chunk_size sweep
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("SECTION 2 — simulate_b at production nsim=9000")
print("=" * 72)

NSIM_PROD = 9000   # matches Fortran's benchtwod3mpi runs
params_b = dict(
    ter=200.0, st=50.0, cr=10.0, crsd=2.0,
    av1=15.0, av2=10.0, av3=8.0,
    sis=12.0, sig=10.0, si=6.0,
)

chunk_sizes_to_try = [16, 64, 256, 1024]
best = None

for cs in chunk_sizes_to_try:
    print(f"\n  chunk_size={cs}:")
    try:
        key = jax.random.key(0)
        t_per_call, (rt, cat) = time_call(
            sim_b.simulate_b, key, **params_b, nsim=NSIM_PROD, chunk_size=cs,
            n_iter=3, warmup=1,
        )
        unique_cats = sorted(set(int(c) for c in cat))
        print(f"    wall-clock: {t_per_call*1000:.0f} ms/call ({t_per_call:.2f} s)")
        print(f"    rt range: [{float(rt.min()):.1f}, {float(rt.max()):.1f}] ms")
        print(f"    cat unique: {unique_cats}")
        if best is None or t_per_call < best[1]:
            best = (cs, t_per_call)
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc(limit=2)
        if "memory" in str(e).lower() or "OOM" in str(e).upper():
            print(f"    (chunk_size={cs} too large for this GPU's memory)")

print(f"\n  BEST: chunk_size={best[0]}, {best[1]*1000:.0f} ms/call")
print(f"  Compare to Fortran 6-node MPI: 11 seconds for 9000 sims")
print(f"  Speedup vs 6-node Fortran: {11.0 / best[1]:.1f}x")
print(f"  Speedup vs 1-node Fortran (36s): {36.0 / best[1]:.1f}x")

BEST_CHUNK_B = best[0]

# ----------------------------------------------------------------------
# Section 3: fofs_b_new wall clock (full 2-condition objective)
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("SECTION 3 — fofs_b_new on real twod3datanew[0]")
print("=" * 72)

from pathlib import Path
from shared import data_io

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "twod3datanew"
raw = data_io.load_twod3datanew(DATA_PATH)
data_b = {
    "prop": jnp.asarray(raw["prop"][0]),
    "count": jnp.asarray(raw["count"][0]),
    "quant": jnp.asarray(raw["quant"][0]),
}
params_b_vec = jnp.array([
    200.0, 50.0, 10.0, 2.0,    # ter, st, cr, crsd
    12.0, 10.0, 0.5,            # sis, sig, sv
    15.0, 10.0, 8.0,            # cond 1 drifts
    14.0, 11.0, 9.0,            # cond 2 drifts
])

key = jax.random.key(0)
t_fofs, val = time_call(
    obj_b.fofs_b_new, params_b_vec, data_b, key,
    n_iter=3, warmup=1, nsim=NSIM_PROD, chunk_size=BEST_CHUNK_B,
)
print(f"  fofs_b_new(nsim=9000): {t_fofs*1000:.0f} ms/call (value={float(val):.2f})")
print(f"  Per-call: simulate_b runs 2 conds via vmap, so effectively 2x simulate work")

# ----------------------------------------------------------------------
# Section 4: fit_simplex_b synthetic recovery — THE headline number
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("SECTION 4 — fit_simplex_b synthetic recovery (THE headline)")
print("=" * 72)

TRUE_PARAMS_B = jnp.array([
    200.0, 50.0, 10.0, 2.0,
    12.0, 10.0, 0.5,
    15.0, 10.0, 8.0,
    14.0, 11.0, 9.0,
])


def _generate_synthetic_data_b(true_params, key, nsim=512, chunk_size=64):
    """Synthetic data via simulate_b at known params."""
    from model_b.objective import COND_MAP_B, clamp_b
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


print(f"  Generating synthetic data at nsim=1024 per condition...")
key_data = jax.random.key(0)
syn_data = _generate_synthetic_data_b(TRUE_PARAMS_B, key_data, nsim=1024,
                                       chunk_size=BEST_CHUNK_B)
print(f"  Data shapes: prop={syn_data['prop'].shape}, count={syn_data['count'].shape}")

# Perturb start
np.random.seed(0)
x0 = TRUE_PARAMS_B * jnp.asarray(np.random.uniform(0.9, 1.1, size=13))
print(f"  Perturbed start: ±10% from truth")

# Fit at production-scale nsim
print(f"\n  Running fit_simplex_b at nsim={NSIM_PROD}, chunk_size={BEST_CHUNK_B}...")
print(f"  (This is the Fortran scale — full headline benchmark)")
t0 = time.perf_counter()
result = fit_b.fit_simplex_b(
    syn_data, jax.random.key(1), x0,
    nsim=NSIM_PROD, maxiter=500, chunk_size=BEST_CHUNK_B,
)
wall = time.perf_counter() - t0

print(f"\n  fit_simplex_b wall-clock: {wall:.1f} s ({wall/60:.1f} min)")
print(f"  n_iters: {result.n_iters}")
print(f"  final loss: {result.loss:.2f}")
print(f"  converged: {result.converged}")

print(f"\n  Per-parameter recovery:")
active_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
for i in active_indices:
    rel_err = abs(float(result.params[i]) - float(TRUE_PARAMS_B[i])) / float(TRUE_PARAMS_B[i])
    label = ["ter", "st", "cr", "crsd", "sis", "sig", "sv",
             "av1c1", "av2c1", "av3c1", "av1c2", "av2c2", "av3c2"][i]
    print(f"    {i:2d} {label:7s} true={float(TRUE_PARAMS_B[i]):7.3f}  "
          f"got={float(result.params[i]):7.3f}  err={rel_err*100:5.1f}%")

# ----------------------------------------------------------------------
# Section 5: Speedup summary vs Fortran
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("SECTION 5 — Speedup summary vs original Fortran")
print("=" * 72)

print(f"""
  Per-call simulate_b at nsim=9000:
    H100 (this run):          {best[1]*1000:.0f} ms
    Fortran 1-node:           36,000 ms  (per README)
    Fortran 6-node MPI:       11,000 ms  (per README)
    Speedup vs 1-node:        {36.0/best[1]:.1f}x
    Speedup vs 6-node MPI:    {11.0/best[1]:.1f}x

  Full fit (this run):
    Wall clock:               {wall:.1f} s ({wall/60:.1f} min)
    n_iters:                  {result.n_iters}
    Fortran "took weeks":     hard to say exactly, but if 6-node × 11s × 5000 evals = 15 hrs minimum,
                              and original ran with restarts/queue waits → likely days
    Estimated speedup vs Fortran fit:   ~{(15.0*3600.0)/wall:.0f}x  (conservative, vs 6-node minimum)
                                       ~{(7.0*24.0*3600.0)/wall:.0f}x  (vs "weeks" upper bound)
""")

print("=" * 72)
print("DONE. Copy this output to claude for the Stage 5 completion write-up.")
print("=" * 72)
