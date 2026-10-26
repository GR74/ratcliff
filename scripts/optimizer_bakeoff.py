"""Pass 3 optimizer bake-off: Nelder-Mead vs CMA-ES on Model B recovery.

One run = one optimizer from one perturbed start, on synthetic data simulated
at TRUTH with its own key (data key != fit key, so the fit cannot reuse the
data's noise). Equal evaluation budgets. Writes one JSON per run; aggregate
across start seeds (the start, not the eval, is the unit of replication).

Data and the final hard-G2 rescoring use the REFERENCE engine: the exact FFT
field simulated as benchtwod3mpi.f (fortran_variant "bench", sv active), the
model the port was shown to reproduce (replicate_ratcliff_subject1.py). The
fit uses --engine; a K-L fit engine therefore carries its truncation bias
into recovery, which is the point of comparing them.

    python scripts/optimizer_bakeoff.py --optimizer cmaes --start-seed 0
    python scripts/optimizer_bakeoff.py --optimizer nm    --start-seed 0
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

# Truth = Ratcliff's subject-1 TACC fit, so recovery is tested where real
# subjects live rather than at a convenient synthetic point.
TRUTH = np.array([188.064, 47.726, 17.492, 6.894, 10.864, 9.192, 0.521,
                  16.213, 11.340, 7.386, 14.138, 11.320, 7.814])
NAMES = ["ter_ms", "st_ms", "cr", "crsd", "sis", "sig", "sv",
         "av1_c1", "av2_c1", "av3_c1", "av1_c2", "av2_c2", "av3_c2"]
DATA_KEY, FIT_KEY, SCORE_KEYS = 999, 321, (5001, 5002, 5003)
REFERENCE = "fft_bench"
ENGINES = {
    "fft_bench": dict(chunk_size=16, use_kl=False, time_block_size=32,
                      fortran_variant="bench"),
    "separable_kl_bench": dict(chunk_size=64, use_kl=True, time_block_size=32,
                               stable_kl_noise=True, separable_kl=True,
                               fortran_variant="bench"),
    # 99.9% of field variance: about half the truncation bias, 1.7-2.1x the
    # 99% engine's time, 2.2-2.7x faster than fft_bench (4090, nsim 2048).
    "separable_kl999_bench": dict(chunk_size=64, use_kl=True, time_block_size=32,
                                  stable_kl_noise=True, separable_kl=0.999,
                                  fortran_variant="bench"),
    # The original port (no sv, unbinned G2, non-crossers -> category 5): the
    # model of the pass-1/2 fits, kept only to reproduce them.
    "separable_kl_legacy": dict(chunk_size=64, use_kl=True, time_block_size=32,
                                stable_kl_noise=True, separable_kl=True),
}


def summary_from_samples(rts, cats):
    prop, count, quant = [], [], []
    for rt, cat in zip(rts, cats):
        count.append([(cat == c).sum() for c in range(1, 6)])
        prop.append([(cat == c).mean() for c in range(1, 6)])
        q = np.zeros((5, 5))
        for c in range(1, 6):
            if (cat == c).sum() >= 5:
                q[:, c - 1] = np.quantile(rt[cat == c], [.1, .3, .5, .7, .9])
        quant.append(q)
    return {"prop": jnp.asarray(np.array(prop)), "count": jnp.asarray(np.array(count)),
            "quant": jnp.asarray(np.array(quant))}


def hard_g2(params, data, nsim, seed, engine):
    return float(obj_b.fofs_b_new(jnp.asarray(params), data, jax.random.key(seed),
                                  nsim=nsim, **ENGINES[engine]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["nm", "cmaes"], required=True)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--data-nsim", type=int, default=400,
                    help="observed trials per condition (real subjects: ~310-440)")
    ap.add_argument("--fit-nsim", type=int, default=2048)
    ap.add_argument("--score-nsim", type=int, default=9000)
    ap.add_argument("--max-evals", type=int, default=1500)
    ap.add_argument("--soft-tau", type=float, default=0.0)
    ap.add_argument("--popsize", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=0)
    ap.add_argument("--engine", choices=list(ENGINES), default="separable_kl999_bench",
                    help="fit engine; data and rescoring use " + REFERENCE)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "pass3", "bakeoff"))
    args = ap.parse_args()

    # Legacy runs keep the legacy model throughout, so they reproduce pass 1/2.
    ref = "separable_kl_legacy" if args.engine == "separable_kl_legacy" else REFERENCE
    engine = ENGINES[args.engine]
    rts, cats = obj_b.simulate_conditions_b(jnp.asarray(TRUTH), jax.random.key(DATA_KEY),
                                            nsim=args.data_nsim, **ENGINES[ref])
    data = summary_from_samples(np.asarray(rts), np.asarray(cats))
    x0 = TRUTH * np.random.default_rng(args.start_seed).uniform(.9, 1.1, 13)

    trace = []
    def on_update(n, loss, params):
        trace.append((n, time.perf_counter() - t0, float(loss)))

    t0 = time.perf_counter()
    if args.optimizer == "nm":
        res = fit_b.fit_simplex_b(
            data, jax.random.key(FIT_KEY), x0, nsim=args.fit_nsim, maxiter=10 * args.max_evals,
            maxfev=args.max_evals, tol=1e-7, optimize_active_only=True,
            soft_tau_ms=args.soft_tau, on_update=on_update, **engine)
    else:
        res = fit_b.fit_cmaes_b(
            data, jax.random.key(FIT_KEY), x0, nsim=args.fit_nsim, max_evals=args.max_evals,
            seed=args.start_seed, popsize=args.popsize or None, restarts=args.restarts,
            soft_tau_ms=args.soft_tau, on_update=on_update,
            **engine)
    wall_s = time.perf_counter() - t0

    fitted = np.asarray(res.params)
    err_pct = 100 * np.abs(fitted - TRUTH) / np.abs(TRUTH)
    # sv is inert, so not recoverable, only in the legacy model.
    active = [i for i in range(13) if i != 6 or "fortran_variant" in engine]
    # Rescore with the HARD count on fresh keys: comparable across tau and optimizers.
    score_fit = [hard_g2(fitted, data, args.score_nsim, s, ref) for s in SCORE_KEYS]
    score_truth = [hard_g2(TRUTH, data, args.score_nsim, s, ref) for s in SCORE_KEYS]
    out = dict(
        settings=vars(args) | dict(truth=TRUTH.tolist(), x0=x0.tolist(), engine=engine,
                                   reference_engine=ENGINES[ref],
                                   device=str(jax.devices()[0]),
                                   keys=dict(data=DATA_KEY, fit=FIT_KEY, score=SCORE_KEYS)),
        params=dict(zip(NAMES, fitted.tolist())),
        err_pct=dict(zip(NAMES, err_pct.tolist())),
        mean_err_pct_active=float(err_pct[active].mean()),
        fit_loss=res.loss, n_evals=res.n_evals, converged=res.converged, wall_s=wall_s,
        hard_g2_fit=score_fit, hard_g2_truth=score_truth,
        trace=trace, stages=res.stages)
    os.makedirs(args.out_dir, exist_ok=True)
    name = (f"{args.optimizer}_{args.engine}_tau{args.soft_tau:g}"
            f"_start{args.start_seed}.json")
    with open(os.path.join(args.out_dir, name), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"{args.optimizer} start {args.start_seed}: mean err {out['mean_err_pct_active']:.2f}% "
          f"evals {res.n_evals} wall {wall_s:.0f}s  hard G2 fit {np.median(score_fit):.1f} "
          f"vs truth {np.median(score_truth):.1f}", flush=True)


if __name__ == "__main__":
    main()
