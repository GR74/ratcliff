"""Pass 3: does the soft-count G2 smooth the real Model B loss surface?

One-parameter slices through Ratcliff's subject-1 fit, with the simulator key
fixed (common random numbers, exactly as inside a Nelder-Mead fit). Each grid
point is simulated once; the same trials are scored with the hard count and
with soft counts of several widths, so differences are due to the loss alone.

    python scripts/soft_g2_loss_slices.py --nsim 2048 --points 61
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jax

jax.config.update("jax_enable_x64", True)   # as in backend/main.py (production)
import jax.numpy as jnp
import numpy as np

from model_b import objective as obj_b
from shared import data_io

CENTER = np.array([188.064, 47.726, 17.492, 6.894, 10.864, 9.192, 0.521,
                   16.213, 11.340, 7.386, 14.138, 11.320, 7.814])
NAMES = ["ter_ms", "st_ms", "cr", "crsd", "sis", "sig", "sv",
         "av1_c1", "av2_c1", "av3_c1", "av1_c2", "av2_c2", "av3_c2"]
# Slice half-widths, chosen to stay inside clamp_b's bounds around CENTER.
SLICES = {"ter_ms": 10.0, "st_ms": 10.0, "cr": 1.5, "crsd": 2.0, "sig": 1.0,
          "av1_c1": 1.5}
TAUS_MS = [0.0, 2.0, 5.0, 10.0]


def roughness(y):
    """Summary of how optimizer-hostile a 1D slice is."""
    d = np.diff(y)
    d2 = np.diff(y, 2)
    interior_min = int(np.sum((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])))
    return dict(flat_step_frac=float(np.mean(d == 0)),
                local_minima=interior_min,
                # second-difference energy relative to the slice's range
                jaggedness=float(np.sum(np.abs(d2)) / max(np.ptp(y), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=2048)
    ap.add_argument("--points", type=int, default=61)
    ap.add_argument("--subject", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pass3",
                                                  "soft_g2_slices.json"))
    args = ap.parse_args()

    real = data_io.load_twod3datanew(os.path.join(ROOT, "data", "twod3datanew"))
    data = {k: jnp.asarray(v[args.subject]) for k, v in real.items()}
    key = jax.random.key(args.seed)

    out = dict(settings=dict(nsim=args.nsim, points=args.points, subject=args.subject,
                             seed=args.seed, taus_ms=TAUS_MS, center=CENTER.tolist(),
                             engine="separable_kl, stable_kl_noise, block 32",
                             device=str(jax.devices()[0])),
               slices={})
    for name, half in SLICES.items():
        idx = NAMES.index(name)
        grid = CENTER[idx] + np.linspace(-half, half, args.points)
        g2 = {tau: [] for tau in TAUS_MS}
        for value in grid:
            params = CENTER.copy(); params[idx] = value
            rts, cats = obj_b.simulate_conditions_b(
                jnp.asarray(params), key, nsim=args.nsim, chunk_size=64, use_kl=True,
                time_block_size=32, stable_kl_noise=True, separable_kl=True)
            for tau in TAUS_MS:
                g2[tau].append(float(obj_b._g2_total(
                    rts, cats, data["prop"], data["count"], data["quant"], tau)))
        out["slices"][name] = dict(
            grid=grid.tolist(),
            g2={str(t): v for t, v in g2.items()},
            roughness={str(t): roughness(np.array(v)) for t, v in g2.items()},
            argmin={str(t): float(grid[int(np.argmin(v))]) for t, v in g2.items()})
        r = out["slices"][name]["roughness"]
        print(f"{name:7s} " + "  ".join(
            f"tau={t:>4}: flat {r[str(t)]['flat_step_frac']:.2f} "
            f"minima {r[str(t)]['local_minima']:2d} jag {r[str(t)]['jaggedness']:.1f}"
            for t in TAUS_MS), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
