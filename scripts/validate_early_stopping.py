"""Memory-bounded GPU validation of early stopping and observed time to quality.

Keep all other speedup flags disabled. Training checkpoints are chosen using
training loss; independent scoring never feeds back into the optimization.
Recorded checkpoint times exclude later diagnostic scoring and are upper
bounds on the first time a target could have been reached between checkpoints.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_speedups as bench

SCENARIOS = {
    "baseline": bench.TRUE["b"],
    "lower_boundary": [200., 50., 7., 1.5, 12., 8., .5, 15., 10., 8., 14., 11., 9.],
    "higher_boundary": [200., 50., 14., 2., 12., 12., .5, 12., 8., 6., 11., 9., 7.],
}


def meets_target(error, loss, truth_loss, error_limit=.05):
    import math
    return (all(math.isfinite(v) for v in (error, loss, truth_loss))
            and error <= error_limit
            and loss <= truth_loss + .1 * max(abs(truth_loss), 1.) + 1.)


def worker(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from model_b import grf_kl, fit, objective

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "default")
    if not args.allow_cpu and (jax.default_backend() != "gpu"
                              or args.gpu_kind.lower() not in jax.devices()[0].device_kind.lower()):
        raise RuntimeError(f"GPU containing {args.gpu_kind!r} required; got {jax.devices()}")
    # The production default can cache 16 GB of bases. Bound this experiment's
    # cache equally for both variants; do not change the library defaults.
    grf_kl._BASIS_CACHE_MAXSIZE = args.basis_cache_size
    grf_kl._basis_cache.clear()
    grf_kl._basis_cache_order.clear()
    out = Path(args.output)
    params = np.asarray(SCENARIOS[args.scenario])
    cfg = {} if args.variant == "legacy" else dict(time_block_size=32, stop_early=True, compact=True)
    result = dict(action=args.action, scenario=args.scenario, variant=args.variant,
                  start_seed=args.start_seed, status="completed", settings=vars(args))
    settings = SimpleNamespace(chunk_a=256, chunk_b=args.chunk, block_size=32,
                               kl_bucketed=False, cumulative_kl=False)
    data_path = out / f"data-{args.scenario}.npz"
    if args.action == "data":
        rt, cat, _ = bench.simulate_conditions(jax, "b", params, args.data_nsim,
                                              123, "legacy", settings)
        np.savez_compressed(data_path, **bench.data_from_samples(rt, cat, 5))
        result.update(data_nsim=args.data_nsim, data_seed=123)
    elif args.action == "micro":
        started = time.perf_counter()
        bench.simulate_conditions(jax, "b", params, args.nsim, 0, args.variant, settings)
        result["compile_and_first_call_seconds"] = time.perf_counter() - started
        bench.simulate_conditions(jax, "b", params, args.nsim, 0, args.variant, settings)
        times = []
        for seed in range(args.sim_seeds):
            for _ in range(args.repeats if seed == 0 else 1):
                started = time.perf_counter()
                rt, cat, work = bench.simulate_conditions(jax, "b", params, args.nsim,
                                                        seed, args.variant, settings)
                if seed == 0:
                    times.append(time.perf_counter() - started)
            np.savez_compressed(out / f"samples-{args.scenario}-{args.variant}-{seed}.npz", rt=rt, cat=cat)
        result.update(timings_seconds=times, computed_trial_steps=work)
    else:
        with np.load(data_path) as archive:
            data = {k: jnp.asarray(v) for k, v in archive.items()}
        options = dict(nsim=args.nsim, chunk_size=args.chunk, use_kl=True)
        active = np.asarray([i for i in range(13) if i != 6])

        def recovery_error(vector):
            effective = np.asarray(objective.clamp_b(jnp.asarray(vector)))
            return float(np.mean(np.abs(effective[active] - params[active]) / params[active]))

        if args.action == "fit":
            x0 = np.asarray(objective.clamp_b(jnp.asarray(params * np.random.default_rng(
                args.start_seed).uniform(1 - args.start_spread, 1 + args.start_spread, 13))))
            warm = time.perf_counter()
            jax.block_until_ready(objective.fofs_b_new(jnp.asarray(x0), data,
                                  jax.random.key(321), **options, **cfg))
            result["compile_and_warmup_seconds"] = time.perf_counter() - warm
            checkpoints, best = [], dict(loss=float("inf"), params=x0.copy(), eval=0)
            trace_path = out / f"trace-{args.scenario}-{args.start_seed}-{args.variant}.jsonl"
            started = time.perf_counter()

            class FitDeadline(Exception):
                pass

            def checkpoint(number, elapsed):
                record = dict(eval=number, fit_elapsed_seconds=elapsed,
                              training_loss=best["loss"], params=best["params"].tolist(),
                              mean_active_effective_parameter_error=recovery_error(best["params"]))
                if not checkpoints or record["params"] != checkpoints[-1]["params"]:
                    checkpoints.append(record)

            def update(number, loss, current):
                elapsed = time.perf_counter() - started
                if loss < best["loss"]:
                    best.update(loss=loss, params=current.copy())
                best["eval"] = number
                record = dict(eval=number, loss=loss, params=current.tolist(), elapsed=elapsed)
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                if number == 1 or number % args.checkpoint_every == 0:
                    checkpoint(number, elapsed)
                    bench.write_json(Path(args.result_file).with_suffix(".progress.json"),
                                     dict(checkpoints=checkpoints, latest_eval=number,
                                          elapsed_seconds=elapsed))
                    print("FIT", args.scenario, args.start_seed, args.variant,
                          "eval", number, "seconds", round(elapsed, 1), flush=True)
                if elapsed >= args.fit_seconds_limit:
                    raise FitDeadline()

            try:
                fitted = fit.fit_simplex_b(data, jax.random.key(321), x0,
                    maxiter=args.maxiter, tol=1e-5, on_update=update, **options, **cfg)
                result.update(converged=fitted.converged, n_iters=fitted.n_iters,
                              n_evals=fitted.n_evals)
            except FitDeadline:
                result.update(status="fit_time_limit", converged=False,
                              n_iters=None, n_evals=best["eval"])
            elapsed = time.perf_counter() - started
            checkpoint(best["eval"], elapsed)
            result.update(fit_seconds=elapsed, params=best["params"].tolist(),
                          x0=x0.tolist(), true_params=params.tolist(),
                          checkpoints=checkpoints, loss=best["loss"],
                          mean_active_effective_parameter_error=recovery_error(best["params"]))
        else:
            fitted = json.loads(Path(args.fit_file).read_text(encoding="utf-8"))
            seeds = [999, 5001, 5002]
            scores = {}

            def score(vector):
                identity = tuple(vector)
                if identity not in scores:
                    scores[identity] = [float(objective.fofs_b_new(
                        jnp.asarray(vector), data, jax.random.key(seed),
                        nsim=args.score_nsim, chunk_size=args.chunk, use_kl=True)) for seed in seeds]
                return scores[identity]

            truth_scores = score(params)
            eligible = [c for c in fitted["checkpoints"]
                        if c["mean_active_effective_parameter_error"] <= args.error_limit]
            if len(eligible) > args.max_score_checkpoints:
                indices = np.linspace(0, len(eligible)-1, args.max_score_checkpoints, dtype=int)
                eligible = [eligible[i] for i in indices]
            final = fitted["checkpoints"][-1]
            if not eligible or eligible[-1]["params"] != final["params"]:
                eligible.append(final)
            checked = []
            for c in eligible:
                values = score(c["params"])
                r = dict(c, independent_g2_by_seed=values, independent_g2_median=float(np.median(values)))
                r["target_pass"] = meets_target(r["mean_active_effective_parameter_error"],
                    r["independent_g2_median"], float(np.median(truth_scores)), args.error_limit)
                checked.append(r)
                print("SCORE", args.scenario, args.start_seed, args.variant,
                      "eval", c["eval"], "pass", r["target_pass"], flush=True)
            hits = [c for c in checked if c["target_pass"]]
            result.update(truth_g2_by_seed=truth_scores, score_seeds=seeds,
                          scored_checkpoints=checked,
                          observed_time_to_target_seconds=min(c["fit_elapsed_seconds"] for c in hits) if hits else None,
                          target_reached=bool(hits), final_params=fitted["params"],
                          score_nsim=args.score_nsim,
                          target_note="Earliest scored checkpoint; not an exact crossing time or global accuracy guarantee.")
    result["metadata"] = bench.metadata(jax)
    result["metadata"]["x64"] = bool(jax.config.jax_enable_x64)
    result["metadata"]["validation_driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["metadata"]["basis_cache_size"] = args.basis_cache_size
    result["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest() if data_path.exists() else None
    return result


def analyse(out, rows, args):
    import numpy as np
    from shared.benchmark_stats import compare_distributions
    summary = dict(scope="RTX 3060 synthetic Model B K-L validation; one laptop, limited cases",
                   micro={}, fits=[r for r in rows if r["action"] == "fit"],
                   scores=[r for r in rows if r["action"] == "score"],
                   counts={status: sum(r["status"] == status for r in rows)
                           for status in sorted({r["status"] for r in rows})})
    for scenario in args.scenarios.split(","):
        micro = {r["variant"]: r for r in rows if r["action"] == "micro"
                 and r["scenario"] == scenario and r["status"] == "completed"}
        if not micro:
            continue
        info = {v: float(np.median(r["timings_seconds"])) for v, r in micro.items()}
        if "legacy" in info and "early_stop" in info:
            info["legacy_to_early_speedup"] = info["legacy"] / info["early_stop"]
        if "blocked_full" in micro and "early_stop" in micro:
            samples = []
            for variant in ("blocked_full", "early_stop"):
                data = [np.load(out / f"samples-{scenario}-{variant}-{s}.npz") for s in range(args.sim_seeds)]
                samples.append(tuple(np.concatenate([d[k] for d in data], axis=1) for k in ("rt", "cat")))
            info["paired_distributions"] = compare_distributions(*samples[0], *samples[1], 5, paired=True)
            info["exact_rt_fraction"] = float(np.mean(samples[0][0] == samples[1][0]))
            info["exact_choice_fraction"] = float(np.mean(samples[0][1] == samples[1][1]))
        summary["micro"][scenario] = info
    return summary


def main(args):
    out = Path(args.output).resolve()
    if args.worker:
        bench.write_json(args.result_file, worker(args))
        return
    if (out / "run-config.json").exists():
        raise ValueError("choose a fresh output directory")
    scenarios = args.scenarios.split(",")
    if any(s not in SCENARIOS for s in scenarios) or min(args.nsim, args.chunk, args.maxiter, args.basis_cache_size) < 1:
        raise ValueError("valid scenarios and positive sizes are required")
    out.mkdir(parents=True, exist_ok=True)
    bench.write_json(out / "run-config.json", dict(vars(args),
        target=f"Mean effective active parameter error <={args.error_limit}; median held-out G2 <= true-parameter median +10% absolute magnitude +1",
        scoring_note="Three held-out seeds; fixed training checkpoints, no scoring feedback",
        compared_features="Legacy versus improvement 1 only; full-horizon blocked correctness control",
        precision="fp32 spatial fields, fp64 RT, x64 enabled, default matmul",
        cache_note="Same reduced basis cache for all variants; library default unchanged"))
    deadline = time.monotonic() + args.budget_seconds
    rows = []
    micro_variants = args.micro_variants.split(",")
    if not micro_variants or any(v not in ("legacy", "blocked_full", "early_stop") for v in micro_variants):
        raise ValueError("valid micro variants required")
    jobs = [("micro", s, v, 0) for s in scenarios for v in micro_variants]
    if not args.pilot_only:
        jobs += [("data", s, "legacy", 0) for s in scenarios]
        for seed in range(args.starts):
            for scenario in scenarios:
                variants = ("legacy", "early_stop") if seed % 2 == 0 else ("early_stop", "legacy")
                for variant in variants:
                    jobs += [("fit", scenario, variant, seed), ("score", scenario, variant, seed)]
        if "baseline" in scenarios:
            for seed in range(args.starts, args.starts + args.baseline_extra_starts):
                variants = ("legacy", "early_stop") if seed % 2 == 0 else ("early_stop", "legacy")
                for variant in variants:
                    jobs += [("fit", "baseline", variant, seed), ("score", "baseline", variant, seed)]
    for action, scenario, variant, seed in jobs:
        remaining = deadline - time.monotonic()
        if remaining <= args.reserve_seconds:
            break
        name = f"{action}-{scenario}-{seed}-{variant}"
        path = out / f"{name}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--action", action,
                   "--scenario", scenario, "--variant", variant, "--start-seed", str(seed),
                   "--output", str(out), "--result-file", str(path), "--gpu-kind", args.gpu_kind]
        for field in ("nsim", "data_nsim", "score_nsim", "chunk", "sim_seeds", "repeats", "maxiter",
                      "starts", "start_spread", "fit_seconds_limit", "basis_cache_size", "checkpoint_every",
                      "max_score_checkpoints", "error_limit"):
            command += ["--" + field.replace("_", "-"), str(getattr(args, field))]
        if args.allow_cpu:
            command.append("--allow-cpu")
        fit_path = out / f"fit-{scenario}-{seed}-{variant}.json"
        if action == "score":
            if not fit_path.exists():
                rows.append(dict(action=action, scenario=scenario, variant=variant,
                                 start_seed=seed, status="skipped", reason="missing completed fit record"))
                continue
            command += ["--fit-file", str(fit_path)]
        print("RUN", name, "remaining", round(remaining), flush=True)
        try:
            with (out / f"{name}.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log,
                                         timeout=remaining - args.reserve_seconds)
            row = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(
                action=action, scenario=scenario, variant=variant, start_seed=seed,
                status="error", returncode=process.returncode)
        except subprocess.TimeoutExpired:
            row = dict(action=action, scenario=scenario, variant=variant,
                       start_seed=seed, status="timeout")
        rows.append(row)
        bench.write_json(out / "results.json", rows)
        bench.write_json(out / "summary.json", analyse(out, rows, args))
        print("DONE", name, row["status"], row.get("fit_seconds", ""), flush=True)
    summary = analyse(out, rows, args)
    summary["planned_cases"] = len(jobs)
    summary["recorded_cases"] = len(rows)
    summary["stopped_by_budget"] = len(rows) < len(jobs)
    summary["cpu_smoke_only"] = args.allow_cpu
    bench.write_json(out / "summary.json", summary)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--pilot-only", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--gpu-kind", default="3060")
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--micro-variants", default="legacy,blocked_full,early_stop")
    p.add_argument("--scenario", choices=SCENARIOS, default="baseline")
    p.add_argument("--action", choices=("micro", "data", "fit", "score"), default="micro")
    p.add_argument("--variant", choices=("legacy", "blocked_full", "early_stop"), default="legacy")
    p.add_argument("--result-file")
    p.add_argument("--fit-file")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--nsim", type=int, default=512)
    p.add_argument("--data-nsim", type=int, default=9000)
    p.add_argument("--score-nsim", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--sim-seeds", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument("--starts", type=int, default=2)
    p.add_argument("--baseline-extra-starts", type=int, default=0)
    p.add_argument("--start-spread", type=float, default=.2)
    p.add_argument("--basis-cache-size", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--max-score-checkpoints", type=int, default=8)
    p.add_argument("--error-limit", type=float, default=.05)
    p.add_argument("--budget-seconds", type=int, default=7200)
    p.add_argument("--fit-seconds-limit", type=int, default=450)
    p.add_argument("--reserve-seconds", type=int, default=180)
    return p


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".5")
    main(parser().parse_args())
