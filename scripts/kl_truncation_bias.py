"""How much of the K-L G2 bias goes away with more retained variance, and at what cost.

At Ratcliff's subject-1 checkpoints, the separable K-L engine (99% of field
variance) scores G2 5-8 above the exact circulant FFT field
(replicate_ratcliff_subject1.py fidelity). This sweeps variance_threshold and
reports G2 (mean +- s.e.m. over keys, same keys as the fidelity run) and
seconds per evaluation, against the exact FFT engine.

    python scripts/kl_truncation_bias.py --thresholds 0.99 0.995 0.999 0.9999
"""
import argparse
import functools
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

from model_b import grf_separable
from model_b import objective as obj_b
from scripts.replicate_ratcliff_subject1 import (CHECKPOINTS, ENGINES, RATCLIFF_NSIM,
                                                 SCORE_SEED0, subject_data)

_ORIGINAL_BASIS = grf_separable.separable_basis


def modes_needed(sig, threshold):
    return grf_separable.selected_modes(sig, k_max=10 ** 9,
                                        variance_threshold=threshold)[4]


def evaluate(params, data, engine, nsim, keys):
    g2, secs = [], []
    for k in keys:
        t0 = time.perf_counter()
        v = obj_b.fofs_b_new(jnp.asarray(params, jnp.float64), data, jax.random.key(k),
                             nsim=nsim, **ENGINES[engine])
        g2.append(float(jax.block_until_ready(v)))
        secs.append(time.perf_counter() - t0)
    # First key includes compilation; time the rest.
    return g2, float(np.median(secs[1:])) if len(secs) > 1 else secs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.99, 0.995, 0.999, 0.9999])
    ap.add_argument("--points", type=int, nargs="+", default=[1, 6],
                    help="checkpoint numbers (1-6)")
    ap.add_argument("--keys", type=int, default=5)
    ap.add_argument("--nsim", type=int, default=RATCLIFF_NSIM)
    ap.add_argument("--variant", default="bench", help="fortran_variant suffix")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pass3",
                                                   "kl_truncation_bias.json"))
    args = ap.parse_args()

    data = subject_data(0)
    keys = [SCORE_SEED0 + i for i in range(args.keys)]
    exact, approx = f"fft_{args.variant}", f"separable_kl_{args.variant}"
    rows = []
    for p in args.points:
        params = CHECKPOINTS[p - 1][0]
        g2, sec = evaluate(params, data, exact, args.nsim, keys)
        rows.append(dict(point=p, engine=exact, threshold=None, g2=g2, s_per_eval=sec))
        print(f"cp{p} exact FFT          G2 {np.mean(g2):7.2f} +- "
              f"{np.std(g2, ddof=1) / np.sqrt(len(g2)):.2f}  {sec:.2f} s/eval", flush=True)
        for t in args.thresholds:
            k = modes_needed(params[5], t)
            grf_separable.separable_basis = functools.partial(
                _ORIGINAL_BASIS, k_max=max(k, 2000), variance_threshold=t)
            try:
                g2_t, sec_t = evaluate(params, data, approx, args.nsim, keys)
            finally:
                grf_separable.separable_basis = _ORIGINAL_BASIS
            # Same key list, but the two engines draw different noise from a
            # key, so this is an unpaired difference reported per key.
            diff = np.asarray(g2_t) - np.asarray(g2)
            rows.append(dict(point=p, engine=approx, threshold=t, modes=k, g2=g2_t,
                             s_per_eval=sec_t, diff_vs_exact=diff.tolist()))
            print(f"cp{p} K-L {t:<7g} K={k:5d} G2 {np.mean(g2_t):7.2f}  "
                  f"minus exact {diff.mean():+6.2f} +- "
                  f"{diff.std(ddof=1) / np.sqrt(len(diff)):.2f}  {sec_t:.2f} s/eval",
                  flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(rows=rows, settings=vars(args) | dict(
            device=str(jax.devices()[0]))), f, indent=1)


if __name__ == "__main__":
    main()
