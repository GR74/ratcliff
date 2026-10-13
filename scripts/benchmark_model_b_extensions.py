"""One-at-a-time H100 comparisons for Model B improvements 3, 4 and 5.

Control is the existing blocked early-stop path with 13 search coordinates.
Use identical observed data, start seeds, indexed normal weights and precision.
Each worker starts fresh; compile/warm-up is separate from synchronized timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_speedups as bench

FEATURES = {
    "control": (),
    "active_only": ("--optimize-active-only",),
    "bucketed": ("--kl-bucketed",),
    "cumulative": ("--cumulative-kl",),
    "all_three": ("--optimize-active-only", "--kl-bucketed", "--cumulative-kl"),
}


def analyse(out, rows, args):
    import numpy as np
    from shared.benchmark_stats import bootstrap_speedup, compare_distributions

    complete = [r for r in rows if r.get("status") == "completed"]
    result = {"fits": {}, "simulators": {},
              "scope": "Synthetic Model B K-L screen, fixed iteration budget; no convergence claim."}
    micro = {r["experiment"]: r for r in complete if r["action"] == "micro"}
    fits = {v: {r["seed"]: r for r in complete
                if r["action"] == "fit" and r["experiment"] == v} for v in FEATURES}
    for variant in FEATURES:
        if variant == "control":
            continue
        if "control" in micro and variant in micro:
            samples = []
            for label in ("control", variant):
                rt, cat = [], []
                for seed in range(args.sim_seeds):
                    with np.load(out / label / f"samples-b-early_stop-{seed}.npz") as data:
                        rt.append(data["rt"]); cat.append(data["cat"])
                samples.append((np.concatenate(rt, axis=1), np.concatenate(cat, axis=1)))
            result["simulators"][variant] = {
                "timing": bootstrap_speedup(micro["control"]["timings_seconds"],
                                            micro[variant]["timings_seconds"]),
                "distributions": compare_distributions(*samples[0], *samples[1], 5, paired=True),
                "paired_trials": int(samples[0][0].size),
                "exact_rt_fraction": float((samples[0][0] == samples[1][0]).mean()),
                "exact_choice_fraction": float((samples[0][1] == samples[1][1]).mean()),
            }
        common = sorted(set(fits["control"]) & set(fits[variant]))
        if common:
            reference = [fits["control"][s] for s in common]
            candidate = [fits[variant][s] for s in common]
            timing = bootstrap_speedup([r["fit_seconds"] for r in reference],
                                       [r["fit_seconds"] for r in candidate], paired=True)
            accuracy = all(c["mean_relative_parameter_error"] <= r["mean_relative_parameter_error"] + .02
                           and c["independent_full_budget_loss"] <= r["independent_full_budget_loss"] * 1.10
                           for r, c in zip(reference, candidate))
            result["fits"][variant] = {"timing": timing, "paired_start_seeds": common,
                                       "accuracy_screen_pass": accuracy,
                                       "promising_screen": len(common) >= 3 and accuracy
                                           and timing["bootstrap_95_ci"][0] > 1}
    result["execution"] = {"planned_cases": 5 * (1 + args.recovery_seeds),
                           "completed_cases": len(complete), "recorded_cases": len(rows),
                           "cpu_smoke_only": args.allow_cpu}
    return result


def main(args):
    out, source = Path(args.output).resolve(), Path(args.data_source).resolve()
    if out.exists():
        raise ValueError("choose a fresh extension output directory")
    if min(args.nsim, args.sim_seeds, args.timing_repeats, args.recovery_seeds,
           args.fit_iters, args.chunk_b, args.worker_timeout) < 1:
        raise ValueError("counts must be positive")
    if not 0 < args.finish_reserve_seconds < args.budget_seconds:
        raise ValueError("budget must exceed reserve")
    if not source.is_file():
        raise ValueError("provide the baseline's data-b.npz")
    out.mkdir(parents=True)
    shutil.copyfile(source, out / "data-b.npz")
    bench.write_json(out / "run-config.json", {
        **vars(args), "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "features": FEATURES})
    for variant in FEATURES:
        folder = out / variant
        folder.mkdir()
        shutil.copyfile(source, folder / "data-b.npz")
    deadline = time.monotonic() + args.budget_seconds
    jobs = [("micro", v, 0) for v in FEATURES]
    labels = list(FEATURES)
    for seed in range(args.recovery_seeds):
        order = labels[seed % len(labels):] + labels[:seed % len(labels)]
        jobs += [("fit", v, seed) for v in order]
    rows = []
    base = [sys.executable, str(ROOT / "scripts/benchmark_speedups.py"), "--worker",
            "--model", "b", "--variant", "early_stop", "--sim-nsim", str(args.nsim),
            "--final-nsim", str(args.nsim), "--score-nsim", str(args.nsim),
            "--fit-iters", str(args.fit_iters), "--chunk-b", str(args.chunk_b),
            "--block-size", "32", "--sim-seeds", str(args.sim_seeds),
            "--timing-repeats", str(args.timing_repeats), "--matmul-precision", "default"]
    if args.allow_cpu:
        base += ["--allow-cpu"]
    for action, variant, seed in jobs:
        remaining = deadline - time.monotonic()
        if remaining <= args.finish_reserve_seconds:
            break
        name = f"{action}-{variant}-{seed}"
        path = out / variant / f"{name}.json"
        command = base + list(FEATURES[variant]) + ["--action", action,
                  "--seed", str(seed), "--output", str(out / variant), "--result-file", str(path)]
        print("RUN", name, "remaining", round(remaining), flush=True)
        try:
            with (out / variant / f"{name}.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log,
                    timeout=min(args.worker_timeout, remaining - args.finish_reserve_seconds))
            row = json.loads(path.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            row = {"status": "timeout", "action": action, "model": "b", "seed": seed}
        except (OSError, ValueError) as error:
            row = {"status": "error", "action": action, "model": "b", "seed": seed,
                   "error": str(error)}
        row["experiment"] = variant
        row["variant"] = variant
        rows.append(row)
        bench.write_json(out / "results.json", rows)
        print("DONE", name, row["status"], row.get("fit_seconds", ""), flush=True)
    summary = analyse(out, rows, args)
    summary["execution"]["stopped_by_budget"] = len(rows) < len(jobs)
    bench.write_json(out / "summary.json", summary)
    print("Saved", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nsim", type=int, default=9000)
    parser.add_argument("--sim-seeds", type=int, default=6)
    parser.add_argument("--timing-repeats", type=int, default=7)
    parser.add_argument("--recovery-seeds", type=int, default=3)
    parser.add_argument("--fit-iters", type=int, default=40)
    parser.add_argument("--chunk-b", type=int, default=64)
    parser.add_argument("--budget-seconds", type=int, default=2400)
    parser.add_argument("--worker-timeout", type=int, default=300)
    parser.add_argument("--finish-reserve-seconds", type=int, default=60)
    parser.add_argument("--allow-cpu", action="store_true")
    main(parser.parse_args())
