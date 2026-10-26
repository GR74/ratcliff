"""Case study: replicate Ratcliff's TACC fit of subject 1 (twod3fitmpi).

Ratcliff's run (twod3fitmpi.tacc.out; Stampede, 78 nodes x 64 cores, 24 min)
used nsim = 78*64*2 = 9984 trials per condition and the exact circulant FFT
random field. Despite the log's name, its data file (twod3datanew), start
point and restart schedule are those of benchtwod3mpi.f, not twod3fitmpi.f.
Its log gives the start point and six checkpoints (best vertex
after each simplex restart) with the G2 the Fortran computed for them.

  fidelity : score every checkpoint with this JAX port, several fresh keys,
             at nsim 9984. If the port is faithful its G2 tracks the Fortran's.
             Engines: fft (exact field, as in Fortran) and separable K-L (99%).
  fit      : fit subject 1 from Ratcliff's start, then rescore the result and
             his final point with the same scorer (same keys, hard count).

    python scripts/replicate_ratcliff_subject1.py fidelity --keys 5
    python scripts/replicate_ratcliff_subject1.py fit --optimizer cmaes --max-evals 1500 \
        --fit-engine separable_kl_bench --engines fft_bench
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

from model_b import fit as fit_b
from model_b import objective as obj_b
from shared import data_io

RATCLIFF_NSIM = 78 * 64 * 2          # trials per condition in the TACC run
RATCLIFF_WALL_MIN = 24.05            # "Run time 00:24:03"
RATCLIFF_CORES = 78 * 64
START = [200.0, 43.0, 22.0, 7.0, 11.0, 8.6, 0.5, 15.1, 10.1, 8.0, 14.0, 11.0, 9.5]
# (params, Fortran G2) for the best vertex after each restart, in log order.
CHECKPOINTS = [
    ([200.000, 43.000, 17.044, 7.504, 10.908, 8.593, 0.503, 15.356, 10.201, 7.909, 14.196, 11.166, 9.641], 174.659),
    ([196.578, 43.339, 17.047, 7.265, 10.994, 8.532, 0.505, 15.477, 10.270, 7.936, 14.308, 11.235, 9.717], 167.547),
    ([195.722, 45.412, 17.033, 7.242, 10.950, 8.494, 0.504, 15.438, 10.247, 7.913, 14.252, 11.210, 9.681], 168.894),
    ([195.149, 45.279, 16.983, 7.221, 10.918, 8.910, 0.503, 15.393, 10.217, 7.890, 14.210, 11.177, 9.653], 165.935),
    ([191.424, 45.213, 17.296, 7.227, 10.879, 8.941, 0.503, 15.370, 10.245, 7.924, 14.213, 11.072, 9.597], 152.323),
    ([188.064, 47.726, 17.492, 6.894, 10.864, 9.192, 0.521, 16.213, 11.340, 7.386, 14.138, 11.320, 7.814], 139.097),
]
ENGINES = {
    "fft": dict(chunk_size=16, use_kl=False, time_block_size=32),
    "separable_kl": dict(chunk_size=64, use_kl=True, time_block_size=32,
                         stable_kl_noise=True, separable_kl=True),
    # Simulating and scoring as Ratcliff's Fortran. "bench" = benchtwod3mpi.f,
    # the program whose start point, data file and schedule match the TACC log.
    "fft_bench": dict(chunk_size=16, use_kl=False, time_block_size=32,
                      fortran_variant="bench"),
    "separable_kl_bench": dict(chunk_size=64, use_kl=True, time_block_size=32,
                               stable_kl_noise=True, separable_kl=True,
                               fortran_variant="bench"),
    "fft_fitprog": dict(chunk_size=16, use_kl=False, time_block_size=32,
                        fortran_variant="fit"),
    # twod3fitmpi1.f: same 6-restart schedule as the log (restart 1 freezes
    # ter and st, as the log shows); smaller zones, joint centring.
    "fft_bench_alt_zones": dict(chunk_size=16, use_kl=False, time_block_size=32,
                                fortran_variant="bench_alt_zones"),
    "fft_fit1": dict(chunk_size=16, use_kl=False, time_block_size=32,
                     fortran_variant="fit1"),
}
SCORE_SEED0 = 7000


def subject_data(subject):
    real = data_io.load_twod3datanew(os.path.join(ROOT, "data", "twod3datanew"))
    return {k: jnp.asarray(v[subject]) for k, v in real.items()}


def score(params, data, engine, nsim, keys):
    out = []
    for k in keys:
        g2 = obj_b.fofs_b_new(jnp.asarray(params, dtype=jnp.float64), data,
                              jax.random.key(k), nsim=nsim, **ENGINES[engine])
        out.append(float(jax.block_until_ready(g2)))
    return out


def fidelity(args):
    data = subject_data(args.subject)
    keys = [SCORE_SEED0 + i for i in range(args.keys)]
    rows = []
    for engine in args.engines:
        for i, (params, fortran_g2) in enumerate([(START, None)] + CHECKPOINTS):
            t0 = time.perf_counter()
            g2 = score(params, data, engine, args.nsim, keys)
            rows.append(dict(engine=engine, point="start" if i == 0 else f"checkpoint{i}",
                             params=params, fortran_g2=fortran_g2, jax_g2=g2,
                             jax_mean=float(np.mean(g2)),
                             jax_sem=float(np.std(g2, ddof=1) / np.sqrt(len(g2))),
                             seconds=time.perf_counter() - t0))
            r = rows[-1]
            fortran = float("nan") if fortran_g2 is None else fortran_g2
            print(f"{engine:13s} {r['point']:12s} Fortran {fortran:7.1f}  "
                  f"JAX {r['jax_mean']:7.1f} +- {r['jax_sem']:.1f} (s.e.m., {len(keys)} keys)",
                  flush=True)
    return dict(mode="fidelity", rows=rows)


def fit(args):
    data = subject_data(args.subject)
    trace = []
    t0 = time.perf_counter()

    def on_update(n, loss, params):
        trace.append((n, time.perf_counter() - t0, float(loss)))

    common = dict(nsim=args.fit_nsim, soft_tau_ms=args.soft_tau, on_update=on_update)
    eng = ENGINES[args.fit_engine]
    if args.optimizer == "nm":
        res = fit_b.fit_simplex_b(data, jax.random.key(args.fit_key), START,
                                  maxiter=10 * args.max_evals, maxfev=args.max_evals,
                                  tol=1e-7, optimize_active_only=True,
                                  **eng, **common)
    else:
        res = fit_b.fit_cmaes_b(data, jax.random.key(args.fit_key), START,
                                max_evals=args.max_evals, seed=args.fit_key,
                                restarts=args.restarts, **eng, **common)
    wall_s = time.perf_counter() - t0
    keys = [SCORE_SEED0 + i for i in range(args.keys)]
    fitted = np.asarray(res.params).tolist()
    scores = {name: {e: score(p, data, e, args.nsim, keys) for e in args.engines}
              for name, p in (("ours", fitted), ("ratcliff_final", CHECKPOINTS[-1][0]))}
    for name, by_engine in scores.items():
        for e, g2 in by_engine.items():
            print(f"{name:15s} {e:13s} hard G2 {np.mean(g2):7.1f} +- "
                  f"{np.std(g2, ddof=1) / np.sqrt(len(g2)):.1f}", flush=True)
    print(f"fit wall {wall_s / 60:.1f} min on {jax.devices()[0]} vs Ratcliff "
          f"{RATCLIFF_WALL_MIN} min on {RATCLIFF_CORES} cores", flush=True)
    return dict(mode="fit", params=fitted, fit_loss=res.loss, n_evals=res.n_evals,
                converged=res.converged, wall_s=wall_s, scores=scores, trace=trace,
                stages=res.stages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fidelity", "fit"])
    ap.add_argument("--subject", type=int, default=0)
    ap.add_argument("--nsim", type=int, default=RATCLIFF_NSIM, help="scoring nsim")
    ap.add_argument("--keys", type=int, default=5)
    ap.add_argument("--engines", nargs="+", default=list(ENGINES))
    ap.add_argument("--optimizer", choices=["nm", "cmaes"], default="cmaes")
    ap.add_argument("--fit-nsim", type=int, default=2048)
    ap.add_argument("--fit-engine", choices=list(ENGINES), default="separable_kl_bench")
    ap.add_argument("--fit-key", type=int, default=321)
    ap.add_argument("--max-evals", type=int, default=1500)
    ap.add_argument("--restarts", type=int, default=0)
    ap.add_argument("--soft-tau", type=float, default=0.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    result = fidelity(args) if args.mode == "fidelity" else fit(args)
    result["settings"] = vars(args) | dict(device=str(jax.devices()[0]),
                                           ratcliff_nsim=RATCLIFF_NSIM)
    out = args.out or os.path.join(ROOT, "results", "pass3", "replication",
                                   f"subject{args.subject + 1}_{args.mode}"
                                   + (f"_{args.optimizer}_{args.fit_engine}"
                                      if args.mode == "fit" else "")
                                   + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
