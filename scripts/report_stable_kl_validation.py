"""CPU-only progress/final report for the guarded 9000-trial repaired K-L run."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["JAX_PLATFORMS"] = "cpu"


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def write(path, value):
    from scripts.benchmark_speedups import write_json
    write_json(path, value)


def archive_raw(base, control, destination):
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "repaired-kl-9000-study.tar.gz"
    manifest_path = destination / "manifest.json"
    if archive.exists():
        manifest = read(manifest_path)
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
    else:
        files = [(p, "study/" + p.relative_to(base / "study").as_posix())
                 for p in sorted((base / "study").rglob("*")) if p.is_file()]
        names = ["active-study.json", "processes.json", "guard-state.json", "guard.log", "guard.err",
                 "run.log", "run.err", "source.tar", "setup-evidence.json", "launch.py", "prepare-run.py",
                 "keep-awake.py", "keep-awake-state.json", "runtime-environment.json"]
        files += [(control / name, "control/" + name) for name in names if (control / name).exists()]
        files += [(base / "checkout" / name, "source/" + name)
                  for name in ("benchmark-source.json", "source-manifest.json")]
        files.append((Path(__file__), "reporting/report_stable_kl_validation.py"))
        entries = []
        with tarfile.open(archive, "w:gz") as handle:
            for path, name in files:
                payload = path.read_bytes()
                item = tarfile.TarInfo(name)
                item.size = len(payload)
                handle.addfile(item, io.BytesIO(payload))
                entries.append(dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
        manifest = dict(archive=archive.name, bytes=archive.stat().st_size,
                        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=entries)
        write(manifest_path, manifest)
    with tarfile.open(archive) as handle:
        for item in manifest["files"]:
            assert hashlib.sha256(handle.extractfile(item["path"]).read()).hexdigest() == item["sha256"]
    return manifest


def main(args):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from model_b.objective import clamp_b
    from scripts.validate_early_stopping import meets_target

    control, out = args.control, args.output
    out.mkdir(parents=True, exist_ok=True)
    active = read(control / "active-study.json")
    base = args.base or Path(r"\\wsl.localhost\Ubuntu") / active["linux_base"].lstrip("/")
    guard = read(control / "guard-state.json", {})
    config = read(base / "study/run-config.json", {})
    rows = read(base / "study/results.json", [])
    planned = read(base / "study/planned-jobs.json", [])
    snapshot = read(base / "checkout/benchmark-source.json")
    if args.final:
        from types import SimpleNamespace
        from scripts.guard_laptop_validation import scan
        if scan(SimpleNamespace(distribution="Ubuntu", script=active["script"], output=active["output"])):
            raise RuntimeError("This benchmark is still running; use a progress report.")
        if guard.get("status") not in ("benchmark_processes_exited", "deadline_enforced") or guard.get("matching_pids"):
            raise RuntimeError("Wait for the independent guard to confirm this benchmark has exited.")
    assert config.get("nsim") == config.get("score_nsim") == 9000
    assert config.get("stable_kl_noise") is True
    assert snapshot["commit"] == active["source_commit"]
    source_manifest = base / "checkout/source-manifest.json"
    assert hashlib.sha256(source_manifest.read_bytes()).hexdigest() == snapshot["manifest_sha256"]
    source_files = read(source_manifest)
    for name, expected in source_files.items():
        assert hashlib.sha256((base / "checkout" / name).read_bytes()).hexdigest() == expected, name
    returned = []
    trajectories = []
    for variant in ("early_stop", "legacy"):
        name = f"baseline-1-{variant}"
        fit = read(base / "study" / f"fit-{name}.json", {})
        score = read(base / "study" / f"score-{name}.json", {})
        trace = base / "study" / f"trace-{name}.jsonl"
        records, partial_lines = [], 0
        if trace.exists():
            for line in trace.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    partial_lines += 1
        trajectory = []
        if records:
            params = np.asarray([r["params"] for r in records])
            effective = np.asarray(jax.vmap(clamp_b)(jnp.asarray(params)))
            truth = np.asarray([200., 50., 10., 2., 12., 10., .5, 15., 10., 8., 14., 11., 9.])
            indices = [i for i in range(13) if i != 6]
            errors = np.mean(abs(effective[:, indices] / truth[indices] - 1), axis=1)
            best = float("inf")
            for record, error in zip(records, errors):
                if record["loss"] < best:
                    best = record["loss"]
                    trajectory.append(dict(eval=record["eval"], elapsed=record["elapsed"],
                                           error=float(error), loss=best, params=record["params"]))
        final_error = fit.get("mean_active_effective_parameter_error")
        parameter_errors = []
        if fit:
            assert fit["settings"]["nsim"] == 9000 and fit["settings"]["stable_kl_noise"] is True
            assert fit["metadata"]["backend"] == "gpu"
            assert fit["metadata"]["commit"] == active["source_commit"]
            effective_final = np.asarray(clamp_b(jnp.asarray(fit["params"])))
            active_indices = [i for i in range(13) if i != 6]
            true_final = np.asarray(fit["true_params"])
            reproduced_error = float(np.mean(abs(effective_final[active_indices] / true_final[active_indices] - 1)))
            assert abs(reproduced_error-final_error) < 1e-10
            names = ["ter", "st", "cr", "crsd", "sis", "sig", "sv",
                     "av1_c1", "av2_c1", "av3_c1", "av1_c2", "av2_c2", "av3_c2"]
            parameter_errors = [dict(parameter=name, truth=float(true_final[i]),
                returned=float(fit["params"][i]), effective=float(effective_final[i]),
                absolute_relative_error=float(abs(effective_final[i] / true_final[i] - 1)),
                included_in_mean=i != 6) for i, name in enumerate(names)]
        if score:
            assert score["score_seeds"] == [999, 5001, 5002]
            assert score["settings"]["stable_kl_noise"] is True and score["score_nsim"] == 9000
            np.testing.assert_array_equal(score["final_params"], fit["params"])
            truth_median = float(np.median(score["truth_g2_by_seed"]))
            for checkpoint in score["scored_checkpoints"]:
                median = float(np.median(checkpoint["independent_g2_by_seed"]))
                assert abs(median - checkpoint["independent_g2_median"]) < 1e-10
                assert meets_target(checkpoint["mean_active_effective_parameter_error"], median, truth_median) == checkpoint["target_pass"]
        returned.append(dict(variant=variant, fit=fit, score=score,
            status=fit.get("status", "interrupted_without_returned_estimate" if args.final and records else
                           "running_or_pending" if not args.final else "missing"),
            returned_error=final_error, returned_error_below_5pct=final_error < .05 if final_error is not None else None,
            per_parameter_errors=parameter_errors,
            recorded_evaluations=len(records), partial_trace_lines=partial_lines,
            joint_target_verified=score.get("target_reached") if score.get("status") == "completed" else None,
            observed_time_to_quality=score.get("observed_time_to_target_seconds") if score.get("status") == "completed" else None))
        trajectories.append(dict(variant=variant, best_training_trajectory=trajectory))
    ratio = None
    if all(c["joint_target_verified"] is True and c["observed_time_to_quality"] is not None for c in returned):
        ratio = returned[1]["observed_time_to_quality"] / returned[0]["observed_time_to_quality"]
    micro = {}
    for row in rows:
        if row["action"] == "micro" and row["status"] == "completed":
            micro[row["variant"]] = row["timings_seconds"][0]
    if "legacy" in micro and "early_stop" in micro:
        micro["original_to_early_stop_ratio"] = micro["legacy"] / micro["early_stop"]
    paired_samples = [base / "study" / f"samples-baseline-{v}-0.npz" for v in ("blocked_full", "early_stop")]
    if all(p.exists() for p in paired_samples):
        with np.load(paired_samples[0]) as reference, np.load(paired_samples[1]) as candidate:
            assert reference["rt"].shape == candidate["rt"].shape == (2, 9000)
            micro["paired_trials"] = int(reference["rt"].size)
            micro["exact_rt_matches"] = int(np.count_nonzero(reference["rt"] == candidate["rt"]))
            micro["exact_choice_matches"] = int(np.count_nonzero(reference["cat"] == candidate["cat"]))
            assert micro["exact_rt_matches"] == micro["exact_choice_matches"] == micro["paired_trials"]
    job_key = lambda row: tuple(row[k] for k in ("action", "scenario", "variant", "start_seed"))
    inventory = dict(status_counts=dict(Counter(r["status"] for r in rows)),
        missing_jobs=[r for r in planned if job_key(r) not in {job_key(row) for row in rows}])
    archive = archive_raw(base, control, out / "raw") if args.final else None
    preflight_failed = any(r["action"] == "verify" and r["status"] != "completed" for r in rows)
    evidence = dict(updated_utc=datetime.now(timezone.utc).isoformat(), final=args.final,
        preflight_failed=preflight_failed,
        scope="One paired baseline start; repaired K-L in both arms, 9000 fitting and scoring trials per condition",
        active_study=active, guard=guard, configuration=config, source_snapshot=snapshot,
        runtime_environment=read(control / "runtime-environment.json", {}),
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_files_verified=len(source_files), recorded_jobs=rows, planned_jobs=planned,
        cases=returned, trajectories=trajectories, micro_pilot=micro, job_inventory=inventory,
        observed_time_to_quality_ratio=ratio, raw_archive=archive)
    write(out / "evidence.json", evidence)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    for path, color in zip(trajectories, ("#337f70", "#788696")):
        points = path["best_training_trajectory"]
        if points:
            ax.plot([p["elapsed"] / 60 for p in points], [p["error"] * 100 for p in points],
                    "o-", ms=3, label=path["variant"], color=color)
    ax.axhline(5, ls="--", color="#b04646", label="5% recovery target")
    ax.axhline(8.68792465972188, ls=":", color="#777777", label="Paired starting error")
    ax.set(xlabel="Training minutes excluding warm-up and independent scoring", ylabel="Mean active parameter error (%)",
           title="Repaired K-L fitting at 9000 trials per condition" + ("" if args.final else " — in progress"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(out / "recovery-progress.png", dpi=180)
    plt.close(fig)

    verification = next((r for r in rows if r["action"] == "verify" and r["status"] == "completed"), {})
    verification_text = (f'GPU preflight passed {verification["gpu_regression_tests"]} tests. The production nearby-width field change was '
                         f'{100*verification["nearby_width_relative_field_change"]:.6f}%; retained component sets identical: '
                         f'{verification["same_retained_component_set"]}.' if verification else
                         "GPU preflight has not passed in this attempt.")
    lines = ["# Repaired K L fitting on the RTX 3060", "",
        "**Verification failed before fitting; no recovery result.**" if preflight_failed else
        "**Final report.**" if args.final else "**In progress. No final fitting-speed or accuracy claim yet.**", "",
        "Both variants enable the mode-indexed noise repair. Every optimization evaluation uses 9,000 trials per condition. "
        "The observed data, 8.688% starting error and ordinary simplex settings are matched; other improvements are off.", "",
        verification_text, "",
        "Hard stop: **October 15, 2026 at 7:24:24 p.m. Eastern**. This includes setup/pilots and never extends the six-hour allowance.", "",
        "| Variant | Status | Returned parameter error | Training minutes | Converged | Joint target verified |",
        "|---|---|---:|---:|---|---|"]
    for case in returned:
        fit = case["fit"]
        error = f'{100*case["returned_error"]:.3f}%' if case["returned_error"] is not None else "Unavailable"
        duration = f'{fit["fit_seconds"] / 60:.2f}' if "fit_seconds" in fit else "Unavailable"
        lines.append(f'| {case["variant"]} | {case["status"]} | {error} | {duration} | {fit.get("converged", "Unavailable")} | {case["joint_target_verified"]} |')
    if args.final:
        stopped = datetime.fromisoformat(guard["last_checked_utc"]).astimezone(timezone(timedelta(hours=-4)))
        stopped_text = stopped.strftime("%B %d, %Y at %I:%M:%S %p Eastern")
        counts = ", ".join(f"{count} {status}" for status, count in inventory["status_counts"].items())
        lines += ["", f'Guard-confirmed exit: **{stopped_text}**; no matching benchmark processes remain. '
                  f'All {len(planned)} planned jobs have {len(rows)} result records ({counts}). '
                  f'Missing jobs: {len(inventory["missing_jobs"])}. The original fit reached its soft time cap; it did not converge. '
                  "Warm-up and independent scoring are excluded from the training-minute column.", ""]
    if "original_to_early_stop_ratio" in micro:
        lines += ["", f'The forward-simulation pilot took **{micro["legacy"]:.3f} s original versus {micro["early_stop"]:.3f} s early-stop**, '
                  f'a descriptive **{micro["original_to_early_stop_ratio"]:.2f}x** ratio at 9,000 trials per condition. '
                  "This is one timed call after two warm-ups, not a confidence interval or a fitting-speed estimate. "
                  f'The full-horizon blocked control and early stopping matched all **{micro.get("paired_trials", 0):,}** trial choices and RTs exactly. '
                  "The original and blocked implementations use different random layouts; the same root key does not make their samples identical.", ""]
    lines += ["", "![Recovery during optimization](recovery-progress.png)", "",
        "The plot follows strict improvements in training loss; recovery error does not guide the optimizer. "
        "The actual returned estimate is reported separately because tied losses can select another simplex vertex.", "",
        "Independent scoring uses the repaired original full-horizon simulator at 9,000 trials for keys 999, 5001 and 5002. "
        "A verified joint hit requires at most 5% mean active-parameter error and the predeclared prediction-quality margin. "
        "A low error alone, optimizer convergence, or a time cap is not a successful joint hit.", ""]
    for case in returned:
        score = case["score"]
        if score.get("status") != "completed":
            continue
        truth_median = float(np.median(score["truth_g2_by_seed"]))
        cutoff = truth_median + .1 * max(abs(truth_median), 1.) + 1.
        lines += [f'Independent scoring for **{case["variant"]}** (lower G2 means less prediction mismatch):', "",
                  "| Estimate | G2 key 999 | G2 key 5001 | G2 key 5002 | Median G2 | Joint gate |",
                  "|---|---:|---:|---:|---:|---|"]
        truth_values = " | ".join(f"{v:.3f}" for v in score["truth_g2_by_seed"])
        lines.append(f"| True parameters | {truth_values} | {truth_median:.3f} | Reference |")
        for checkpoint in score["scored_checkpoints"]:
            is_returned = np.array_equal(checkpoint["params"], score["final_params"])
            label = "Returned estimate" if is_returned else f'Checkpoint at evaluation {checkpoint["eval"]}'
            values = " | ".join(f"{v:.3f}" for v in checkpoint["independent_g2_by_seed"])
            lines.append(f'| {label} | {values} | {checkpoint["independent_g2_median"]:.3f} | {checkpoint["target_pass"]} |')
        lines += ["", f"The predeclared prediction cutoff is **{cutoff:.3f}**. Parameter recovery below 5% and prediction quality are separate checks. "
                  "The parameter error is a mean across twelve active parameters; individual errors may be higher. "
                  "This provisional gate is not a statistical significance test or a finalized scientific accuracy criterion.", ""]
    lines += [f"Observed original/early-stop time-to-quality ratio: **{ratio:.3f}x**." if ratio is not None else
        "**Time-to-quality ratio is not estimable: both target hits have not been verified.**", "",
        "This is one paired synthetic case on one laptop. Missing, interrupted and time-capped fits are retained. "
        "Scoring uses the same observations with independent simulator randomness, not held-out participant data.", "",
        "Relative to the earlier 512-trial study, both the noise assignment and optimization trial budget changed. "
        "This experiment does not separately estimate how much each change affects recovery.", "",
        f"[Predeclared plan]({Path(os.path.relpath(ROOT / 'docs/plans/2026-10-15-stable-kl-9000-rtx3060.md', out)).as_posix()}) · [Compact evidence](evidence.json)"]
    if all(c["per_parameter_errors"] for c in returned):
        lines += ["", "Returned effective parameters and absolute relative errors:", "",
                  "| Parameter | Truth | Improvement 1 | Error | Original | Error |",
                  "|---|---:|---:|---:|---:|---:|"]
        for candidate, original in zip(returned[0]["per_parameter_errors"], returned[1]["per_parameter_errors"]):
            name = candidate["parameter"] + (" (inactive; excluded)" if not candidate["included_in_mean"] else "")
            lines.append(f'| {name} | {candidate["truth"]:.6g} | {candidate["effective"]:.6g} | '
                         f'{100*candidate["absolute_relative_error"]:.3f}% | {original["effective"]:.6g} | '
                         f'{100*original["absolute_relative_error"]:.3f}% |')
    if archive:
        lines += ["", "[Verified raw archive manifest](raw/manifest.json). The archive includes source, observations, traces, scores, logs and guard evidence."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Final" if args.final else "Progress", "report written;", len(rows), "recorded jobs; verified source files:", len(source_files))
    for case in returned:
        print(case["variant"], case["status"], "returned error", case["returned_error"], "joint hit", case["joint_target_verified"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--control", type=Path, default=ROOT / ".benchmark-results/stable-kl-9000-3060/control")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/2026-10-15-stable-kl-9000-rtx3060")
    parser.add_argument("--final", action="store_true")
    main(parser.parse_args())
