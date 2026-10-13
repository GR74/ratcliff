"""Deadline-limited, fresh-process comparison of time blocking/staged fitting.

Run on the GPU, not an asynchronous dispatch timer. Raw samples and logs go in
.benchmark-results (ignored by Git). The JSON/report explicitly distinguish
paired controls, the legacy implementation, and limited-budget pilot fits.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = ("legacy", "blocked_full", "early_stop", "staged", "combined")
TRUE = {
    "a": [200., 50., 50., 10., 4., 5., .7, 20., 10., 60.],
    "b": [200., 50., 10., 2., 12., 10., .5, 15., 10., 8., 14., 11., 9.],
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def metadata(jax):
    def command(args):
        try:
            p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return p.stdout.strip() if p.returncode == 0 else None
    packages = {}
    for name in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt",
                 "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    stats = jax.devices()[0].memory_stats()
    digest = hashlib.sha256()
    files = ["shared/blocked.py", "shared/staged_fit.py", "shared/benchmark_stats.py",
             "model_a/simulate.py", "model_a/objective.py", "model_a/fit.py",
             "model_b/simulate.py", "model_b/objective.py", "model_b/fit.py",
             "scripts/benchmark_speedups.py"]
    for path in files:
        digest.update(path.encode())
        digest.update((ROOT / path).read_bytes())
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "commit": command(["git", "rev-parse", "HEAD"]),
        "implementation_sha256": digest.hexdigest(), "hashed_files": files,
        "backend": jax.default_backend(),
        "python": sys.version, "packages": packages,
        "devices": [str(d) for d in jax.devices()],
        "device_kind": jax.devices()[0].device_kind,
        "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                               "--format=csv,noheader"]),
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "device_allocator_memory": stats,
        "memory_note": "Allocator peak includes warm-up and the entire worker process.",
    }


def config(variant, args):
    if variant == "legacy":
        return {"time_block_size": 0}
    return {"time_block_size": args.block_size,
            "stop_early": variant in ("early_stop", "combined"), "compact": True}


def schedule(args):
    from shared.staged_fit import TrialStage
    n = args.final_nsim
    a, b = min(512, n // 4), min(2000, n // 2)
    i = max(1, args.fit_iters // 4)
    return [TrialStage(a, i), TrialStage(b, i), TrialStage(n, args.fit_iters - 2 * i)]


def simulate_conditions(jax, model, params, nsim, seed, variant, args):
    import jax.numpy as jnp
    import numpy as np
    from shared import prng
    cfg = config(variant, args)
    rt_list, cat_list = [], []
    work = 0
    if model == "a":
        from model_a import objective, simulate
        p = objective.clamp(jnp.asarray(params))
        conditions = objective.COND_MAP
        for ci, (di, bi) in enumerate(conditions):
            key = prng.split_for_condition(jax.random.key(seed), ci)
            values = (key, p[0], p[1], p[bi], p[3], p[4], p[5], p[di], nsim,
                      args.chunk_a)
            if cfg["time_block_size"]:
                rt, cat, allocated = simulate.simulate_blocked_with_work(**dict(
                    zip(("key", "ter", "st", "cr", "crsd", "si", "sig", "av",
                         "nsim", "chunk_size"), values)), **cfg)
                work += int(allocated)
            else:
                rt, cat = simulate.simulate(*values)
                work += ((nsim + args.chunk_a - 1) // args.chunk_a) * args.chunk_a * 400
            rt_list.append(rt)
            cat_list.append(cat)
    else:
        from model_b import objective, simulate
        p = objective.clamp_b(jnp.asarray(params))
        for ci, (d1, d2, d3) in enumerate(objective.COND_MAP_B):
            key = prng.split_for_condition(jax.random.key(seed), ci)
            values = (key, p[0], p[1], p[2], p[3], p[d1], p[d2], p[d3],
                      p[4], float(p[5]), 6., nsim, args.chunk_b, True)
            if cfg["time_block_size"]:
                rt, cat, allocated = simulate.simulate_b_blocked_with_work(
                    *values, **cfg,
                )
                work += int(allocated)
            else:
                rt, cat = simulate.simulate_b(*values)
                work += ((nsim + args.chunk_b - 1) // args.chunk_b) * args.chunk_b * 400
            rt_list.append(rt)
            cat_list.append(cat)
    rt, cat = jnp.stack(rt_list), jnp.stack(cat_list)
    jax.block_until_ready((rt, cat))
    return np.asarray(rt), np.asarray(cat), work


def data_from_samples(rt, cat, categories):
    import numpy as np
    props, counts, quantiles = [], [], []
    for rts, cats in zip(rt, cat):
        counts.append([(cats == c).sum() for c in range(1, categories + 1)])
        props.append([(cats == c).mean() for c in range(1, categories + 1)])
        q = np.zeros((5, categories))
        for c in range(1, categories + 1):
            selected = rts[cats == c]
            if len(selected) >= 5:
                q[:, c - 1] = np.quantile(selected, [.1, .3, .5, .7, .9])
        quantiles.append(q)
    return {"prop": np.asarray(props), "count": np.asarray(counts),
            "quant": np.asarray(quantiles)}


def worker(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    if not args.allow_cpu and (jax.default_backend() != "gpu"
                              or "H100" not in jax.devices()[0].device_kind):
        raise RuntimeError(f"An H100 is required; got {jax.devices()}")
    out = Path(args.output)
    params = np.asarray(TRUE[args.model])
    categories = 3 if args.model == "a" else 5
    result = {"status": "completed", "action": args.action, "model": args.model,
              "variant": args.variant, "seed": args.seed,
              "settings": {"block_size": args.block_size, "chunk_a": args.chunk_a,
                           "chunk_b": args.chunk_b, "sim_nsim": args.sim_nsim,
                           "final_nsim": args.final_nsim, "fit_iters": args.fit_iters,
                           "matmul_precision": args.matmul_precision}}
    if args.action == "data":
        rt, cat, _ = simulate_conditions(jax, args.model, params, args.data_nsim,
                                         123, "legacy", args)
        np.savez_compressed(out / f"data-{args.model}.npz",
                            **data_from_samples(rt, cat, categories))
        result["data_seed"] = 123
        result["data_nsim"] = args.data_nsim
    elif args.action == "micro":
        started = time.perf_counter()
        simulate_conditions(jax, args.model, params, args.sim_nsim,
                             0, args.variant, args)
        result["compile_and_first_call_seconds"] = time.perf_counter() - started
        simulate_conditions(jax, args.model, params, args.sim_nsim,
                             0, args.variant, args)
        times = []
        seed_times = []
        for seed in range(args.sim_seeds):
            repeats = args.timing_repeats if seed == 0 else 1
            for _ in range(repeats):
                started = time.perf_counter()
                rt, cat, work = simulate_conditions(jax, args.model, params,
                                                    args.sim_nsim, seed,
                                                    args.variant, args)
                elapsed = time.perf_counter() - started
                if seed == 0:
                    times.append(elapsed)
            seed_times.append(elapsed)
            np.savez_compressed(out / f"samples-{args.model}-{args.variant}-{seed}.npz",
                                rt=rt, cat=cat)
        result.update({"timings_seconds": times, "seed_timings_seconds": seed_times,
                       "computed_trial_steps": work,
                       "nominal_full_trial_steps": rt.shape[0] * args.sim_nsim * 400})
    else:
        if args.model == "a":
            from model_a import fit, objective
            driver, evaluator = fit.fit_simplex, objective.fofs_new
            options = {"chunk_size": args.chunk_a}
        else:
            from model_b import fit, objective
            driver, evaluator = fit.fit_simplex_b, objective.fofs_b_new
            options = {"chunk_size": args.chunk_b, "use_kl": True}
        data = {k: jnp.asarray(v) for k, v in np.load(out / f"data-{args.model}.npz").items()}
        perturbation = np.random.default_rng(args.seed).uniform(.9, 1.1, len(params))
        x0 = params * perturbation
        cfg = config(args.variant, args)
        stages = schedule(args) if args.variant in ("staged", "combined") else None
        budgets = [s.nsim for s in stages] if stages else [args.final_nsim]
        started = time.perf_counter()
        for nsim in budgets:
            jax.block_until_ready(evaluator(jnp.asarray(x0), data, jax.random.key(321),
                                            nsim=nsim, **options, **cfg))
        result["compile_and_warmup_seconds"] = time.perf_counter() - started
        log_path = out / f"fit-progress-{args.model}-{args.variant}-{args.seed}.jsonl"
        def on_update(number, loss, current):
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"eval": number, "loss": loss,
                                         "elapsed": time.perf_counter() - fit_start}) + "\n")
        fit_start = time.perf_counter()
        fitted = driver(data, jax.random.key(321), x0,
                         nsim=args.final_nsim, maxiter=args.fit_iters, tol=1e-5,
                         trial_schedule=stages, on_update=on_update, **options, **cfg)
        wall = time.perf_counter() - fit_start
        estimate = np.asarray(fitted.params)
        active = np.asarray([i for i in range(len(params)) if i != 6])
        errors = np.abs(estimate[active] - params[active]) / np.abs(params[active])
        independent_loss = float(evaluator(
            fitted.params, data, jax.random.key(999), nsim=args.score_nsim,
            **options, time_block_size=0,
        ))
        result.update({"fit_seconds": wall, "params": estimate.tolist(),
                       "x0": x0.tolist(), "true_params": params.tolist(),
                       "active_indices": active.tolist(),
                       "mean_relative_parameter_error": float(errors.mean()),
                       "max_relative_parameter_error": float(errors.max()),
                       "within_7_percent": int((errors <= .07).sum()),
                       "independent_full_budget_loss": independent_loss,
                       "score_nsim": args.score_nsim, "score_seed": 999,
                       "fit_seed": 321, "n_iters": fitted.n_iters,
                       "n_evals": fitted.n_evals, "converged": fitted.converged,
                       "stages": fitted.stages})
    result["metadata"] = metadata(jax)
    return result


def analyse(out, rows, args):
    import numpy as np
    from shared.benchmark_stats import bootstrap_speedup, compare_distributions
    analysis = {"timing": {}, "distributions": {}, "fits": {},
                "scope": "Repeated synthetic-data screening; not human-data validation.",
                "fit_accuracy_margins": {"mean_relative_error_increase": .02,
                                         "independent_loss_increase_fraction": .10}}
    completed = [r for r in rows if r.get("status") == "completed"]
    for model in args.models.split(","):
        micro = {r["variant"]: r for r in completed
                 if r["model"] == model and r["action"] == "micro"}
        for reference, candidate in (("legacy", "early_stop"),
                                      ("blocked_full", "early_stop")):
            label = f"{model}:{reference}->{candidate}"
            if reference not in micro or candidate not in micro:
                continue
            analysis["timing"][label] = bootstrap_speedup(
                micro[reference]["timings_seconds"], micro[candidate]["timings_seconds"],
            )
            samples = []
            for variant in (reference, candidate):
                data = [np.load(out / f"samples-{model}-{variant}-{s}.npz")
                        for s in range(args.sim_seeds)]
                samples.append((np.concatenate([d["rt"] for d in data], axis=1),
                                np.concatenate([d["cat"] for d in data], axis=1)))
            analysis["distributions"][label] = compare_distributions(
                *samples[0], *samples[1], 3 if model == "a" else 5,
                paired=reference == "blocked_full",
            )
        fits = [r for r in completed if r["model"] == model and r["action"] == "fit"]
        by_variant = {v: {r["seed"]: r for r in fits if r["variant"] == v}
                      for v in VARIANTS}
        for reference, candidate in (("legacy", "combined"),
                                      ("legacy", "early_stop"),
                                      ("early_stop", "combined"),
                                      ("blocked_full", "early_stop"),
                                      ("blocked_full", "staged"),
                                      ("blocked_full", "combined")):
            common = sorted(set(by_variant[reference]) & set(by_variant[candidate]))
            if not common:
                continue
            a, b = ([by_variant[v][s] for s in common] for v in (reference, candidate))
            timing = bootstrap_speedup([r["fit_seconds"] for r in a],
                                       [r["fit_seconds"] for r in b], paired=True)
            differences = [rb["mean_relative_parameter_error"]
                           - ra["mean_relative_parameter_error"] for ra, rb in zip(a, b)]
            losses_ok = all(rb["independent_full_budget_loss"]
                            <= ra["independent_full_budget_loss"] * 1.10 + 1e-9
                            for ra, rb in zip(a, b))
            errors_ok = all(d <= .02 for d in differences)
            enough = len(common) >= 3
            faster = timing["bootstrap_95_ci"][0] > 1
            verdict = ("promising pilot improvement" if enough and faster and losses_ok and errors_ok
                       else "not established in this pilot")
            analysis["fits"][f"{model}:{reference}->{candidate}"] = {
                "timing": timing, "paired_start_seeds": common,
                "mean_relative_error_differences": differences,
                "accuracy_screen_pass": errors_ok and losses_ok,
                "verdict": verdict,
                "note": "Margins are provisional; fixed iteration budgets, not time-to-convergence.",
            }
    return analysis


def report(out, rows, analysis, args):
    title = "CPU smoke test (not H100 performance evidence)" if args.allow_cpu else "H100 speedup screening results"
    lines = [f"# {title}", "",
             "All comparisons use this run's GPU, software and synthetic observed data.",
             f"Final pilot budget: {args.final_nsim} trials/condition, {args.fit_iters} iterations.",
             "Initial compilation/warm-up is reported separately from timed fitting.", "",
             "| Model | Variant | Start seed | Fit seconds | Mean active error | Independent G2 | Converged |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        if r.get("status") == "completed" and r["action"] == "fit":
            lines.append(f"| {r['model'].upper()} | {r['variant']} | {r['seed']} | "
                         f"{r['fit_seconds']:.2f} | {100*r['mean_relative_parameter_error']:.2f}% | "
                         f"{r['independent_full_budget_loss']:.3f} | {r['converged']} |")
    lines += ["", "| Comparison | Median fit speedup | Bootstrap 95% interval | Pilot finding |",
              "|---|---:|---|---|"]
    for label, comparison in analysis["fits"].items():
        t = comparison["timing"]
        lo, hi = t["bootstrap_95_ci"]
        lines.append(f"| {label} | {t['median_speedup']:.2f}x | {lo:.2f}-{hi:.2f}x | "
                     f"{comparison['verdict']} |")
    lines += ["", "The comparison separates legacy code from the paired indexed-noise control.",
              "Distribution intervals and sparse-category limitations are in summary.json.",
              "A screening pass is not proof of equivalence across all datasets or parameters.",
              "Counts, seeds, GPU/driver, package versions, source commit, memory and warm-up",
              "measurements are retained in results.json. Interrupted cases are recorded too."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        try:
            result = worker(args)
        except Exception as error:
            import traceback
            traceback.print_exc()
            result = {"status": "error", "action": args.action, "model": args.model,
                      "variant": args.variant, "seed": args.seed, "error": str(error)}
        write_json(args.result_file, result)
        return 0 if result["status"] == "completed" else 1
    if (out / "results.json").exists():
        raise ValueError("choose a fresh output directory; existing results must not be mixed")
    if args.fit_iters < 4 or args.final_nsim < 4:
        raise ValueError("pilot fitting needs fit_iters >= 4 and final_nsim >= 4")
    if not set(args.models.split(",")) <= {"a", "b"}:
        raise ValueError("models must be a, b, or a,b")
    fit_variants = args.fit_variants.split(",")
    if not fit_variants or not set(fit_variants) <= set(VARIANTS):
        raise ValueError("fit variants must name supported simulation/fitting paths")
    if len(set(fit_variants)) != len(fit_variants):
        raise ValueError("fit variants must not be repeated")
    if min(args.sim_seeds, args.timing_repeats, args.recovery_seeds) < 1:
        raise ValueError("seed and timing counts must be positive")
    if min(args.sim_nsim, args.data_nsim, args.score_nsim,
           args.chunk_a, args.chunk_b, args.worker_timeout) < 1:
        raise ValueError("trial counts, chunk sizes and worker timeout must be positive")
    if not 1 <= args.block_size <= 400:
        raise ValueError("block size must be between 1 and 400")
    if not 0 < args.finish_reserve_seconds < args.budget_seconds:
        raise ValueError("budget must exceed the positive finish reserve")
    write_json(out / "run-config.json", vars(args))
    started = time.monotonic()
    deadline = time.monotonic() + args.budget_seconds
    rows = []
    base = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(out)]
    for name in ("sim_nsim", "sim_seeds", "timing_repeats", "data_nsim", "final_nsim",
                 "score_nsim", "fit_iters", "block_size", "chunk_a", "chunk_b",
                 "matmul_precision"):
        base += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if args.allow_cpu:
        base += ["--allow-cpu"]
    jobs = []
    models = args.models.split(",")
    for model in models:
        jobs += [("data", model, "legacy", 0)]
        if not args.skip_micro:
            jobs += [("micro", model, v, 0) for v in ("legacy", "blocked_full", "early_stop")]
    # Pair starting points and rotate variant order to reduce order confounding.
    for seed in range(args.recovery_seeds):
        offset = seed % len(fit_variants)
        variants = fit_variants[offset:] + fit_variants[:offset]
        for model in models:
            jobs += [("fit", model, v, seed) for v in variants]
    for action, model, variant, seed in jobs:
        remaining = deadline - time.monotonic()
        if remaining <= args.finish_reserve_seconds:
            print("Deadline reserve reached; no further GPU workers started.", flush=True)
            break
        name = f"{action}-{model}-{variant}-{seed}"
        result_file, log_file = out / f"{name}.json", out / f"{name}.log"
        result_file.unlink(missing_ok=True)
        if action == "fit" and not (out / f"data-{model}.npz").exists():
            rows.append({"status": "skipped", "action": action, "model": model,
                         "variant": variant, "seed": seed,
                         "error": "synthetic data worker did not complete"})
            write_json(out / "results.json", rows)
            continue
        command = base + ["--action", action, "--model", model, "--variant", variant,
                          "--seed", str(seed), "--result-file", str(result_file)]
        print(f"RUN {name} (remaining budget {remaining:.0f}s)", flush=True)
        try:
            with log_file.open("w", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log,
                    timeout=min(args.worker_timeout, remaining - args.finish_reserve_seconds))
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if process.returncode:
                print(f"ERROR {name}: {result.get('error')}", flush=True)
        except subprocess.TimeoutExpired:
            result = {"status": "timeout", "action": action, "model": model,
                      "variant": variant, "seed": seed}
        except (OSError, ValueError) as error:
            result = {"status": "error", "action": action, "model": model,
                      "variant": variant, "seed": seed,
                      "error": f"worker produced no readable result: {error}"}
        rows.append(result)
        write_json(out / "results.json", rows)
        if result.get("status") == "completed":
            print(f"DONE {name}: fit={result.get('fit_seconds', '-')}s", flush=True)
    analysis = analyse(out, rows, args)
    analysis["execution"] = {"planned_cases": len(jobs), "recorded_cases": len(rows),
                             "completed_cases": sum(r.get("status") == "completed" for r in rows),
                             "failed_or_timed_out_cases": sum(r.get("status") != "completed" for r in rows),
                             "stopped_by_budget": len(rows) < len(jobs),
                             "parent_elapsed_seconds": time.monotonic() - started,
                             "cpu_smoke_only": args.allow_cpu}
    write_json(out / "summary.json", analysis)
    report(out, rows, analysis, args)
    print(f"Results saved to {out}", flush=True)
    return 0 if any(r.get("status") == "completed" and r["action"] != "data"
                    for r in rows) else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=".benchmark-results/h100-speedups")
    p.add_argument("--models", default="a,b")
    p.add_argument("--fit-variants", default=",".join(VARIANTS))
    p.add_argument("--skip-micro", action="store_true", help="fit-only follow-up after simulator screening")
    p.add_argument("--budget-seconds", type=int, default=2700)
    p.add_argument("--finish-reserve-seconds", type=int, default=30)
    p.add_argument("--worker-timeout", type=int, default=480)
    p.add_argument("--sim-nsim", type=int, default=9000)
    p.add_argument("--sim-seeds", type=int, default=3)
    p.add_argument("--timing-repeats", type=int, default=7)
    p.add_argument("--data-nsim", type=int, default=1024)
    p.add_argument("--final-nsim", type=int, default=2048)
    p.add_argument("--score-nsim", type=int, default=9000)
    p.add_argument("--fit-iters", type=int, default=40)
    p.add_argument("--recovery-seeds", type=int, default=3)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--chunk-a", type=int, default=256)
    p.add_argument("--chunk-b", type=int, default=64)
    p.add_argument("--matmul-precision", choices=["default", "highest"], default="default")
    p.add_argument("--allow-cpu", action="store_true", help="local smoke only; not H100 evidence")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--action", choices=["data", "micro", "fit"], default="micro", help=argparse.SUPPRESS)
    p.add_argument("--model", choices=["a", "b"], default="a", help=argparse.SUPPRESS)
    p.add_argument("--variant", choices=VARIANTS, default="legacy", help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--result-file", help=argparse.SUPPRESS)
    return p


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.exit(main(parser().parse_args()))
