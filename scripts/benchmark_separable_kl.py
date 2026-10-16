"""Pass 3: dense vs separable K-L noise, end-to-end objective evaluations.

Each evaluation uses a NEW sig (as a Nelder-Mead fit does), so the dense path
pays its basis rebuild and the separable path pays its own (much smaller) one.
Engines alternate eval by eval so drift in machine load hits both equally.

    python scripts/benchmark_separable_kl.py --nsim 2048 --evals 12
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jax

jax.config.update("jax_enable_x64", True)   # as in backend/main.py (production)
import jax.numpy as jnp
import numpy as np

from model_b import objective as obj_b
from shared import data_io

# Ratcliff's TACC fit of subject 1 (twod3fitmpi.tacc.out, final simplex vertex):
# the benchmark is evaluated in the region a real fit spends its time.
TRUE_PARAMS = np.array([188.064, 47.726, 17.492, 6.894, 10.864, 9.192, 0.521,
                        16.213, 11.340, 7.386, 14.138, 11.320, 7.814])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=2048)
    ap.add_argument("--evals", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subject", type=int, default=0, help="0-based twod3datanew row")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    real = data_io.load_twod3datanew(os.path.join(ROOT, "data", "twod3datanew"))
    data = {k: jnp.asarray(v[args.subject]) for k, v in real.items()}
    key = jax.random.key(args.seed)

    def evaluate(params, separable):
        t0 = time.perf_counter()
        g2 = obj_b.fofs_b_new(jnp.asarray(params), data, key, nsim=args.nsim,
                              chunk_size=args.chunk, use_kl=True, time_block_size=32,
                              stable_kl_noise=True, separable_kl=separable)
        g2 = float(jax.block_until_ready(g2))
        return time.perf_counter() - t0, g2

    # Warm-up compiles both engines (separable recompiles per row/col bucket).
    for sep in (False, True):
        evaluate(TRUE_PARAMS, sep)
    rows = []
    for i in range(args.evals):
        params = TRUE_PARAMS * (1 + 0.03 * rng.standard_normal(13))
        params[5] = TRUE_PARAMS[5] + rng.uniform(-1, 1)   # new sig every eval
        order = (False, True) if i % 2 == 0 else (True, False)
        out = {sep: evaluate(params, sep) for sep in order}
        rows.append(dict(eval=i, sig=float(params[5]),
                         dense_s=out[False][0], separable_s=out[True][0],
                         dense_g2=out[False][1], separable_g2=out[True][1]))
        print(f"eval {i:2d} sig={params[5]:6.3f}  dense {out[False][0]:6.3f} s "
              f"sep {out[True][0]:6.3f} s  G2 {out[False][1]:9.3f} vs {out[True][1]:9.3f}",
              flush=True)
    d = np.array([r["dense_s"] for r in rows]); s = np.array([r["separable_s"] for r in rows])
    ratio = d / s
    summary = dict(device=str(jax.devices()[0]), nsim=args.nsim, evals=args.evals,
                   dense_median_s=float(np.median(d)), separable_median_s=float(np.median(s)),
                   speedup_median=float(np.median(ratio)),
                   speedup_min=float(ratio.min()), speedup_max=float(ratio.max()),
                   max_abs_g2_diff=float(max(abs(r["dense_g2"] - r["separable_g2"]) for r in rows)))
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(summary=summary, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
