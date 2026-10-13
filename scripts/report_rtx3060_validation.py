"""Publish compact laptop evidence and verified, small raw backups using CPU I/O."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def backup(base, destination, directories, controls, filename):
    files = [p for name in directories for p in sorted((base / "results" / name).rglob("*")) if p.is_file()]
    files += [base / "control" / name for name in controls if (base / "control" / name).is_file()]
    manifest = []
    with tarfile.open(destination / filename, "w:gz") as archive:
        for path in files:
            payload = path.read_bytes()
            name = path.relative_to(base).as_posix()
            manifest.append(dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    with tarfile.open(destination / filename, "r:gz") as archive:
        for record in manifest:
            payload = archive.extractfile(record["path"]).read()
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]
    path = destination / filename
    return dict(archive=filename, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), files=manifest)


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, out = Path(args.base), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    study = base / "results" / "study-512-six-hour"
    guard = read(base / "control" / "six-hour-guard-state.json", {})
    if args.final:
        from scripts.guard_laptop_validation import scan
        if (guard.get("status") not in ("benchmark_processes_exited", "deadline_enforced")
                or guard.get("matching_pids")):
            raise ValueError("the named GPU benchmark must be stopped before final backup")
        if scan(argparse.Namespace(distribution="Ubuntu", script=guard["script"], output=guard["output"])):
            raise ValueError("matching benchmark processes still running")

    runs = {name: read(base / "results" / name / "summary.json", {})
            for name in ("pilot-512", "pilot-9000", "study-128", "study-512-six-hour")}
    configs = {name: read(base / "results" / name / "run-config.json", {}) for name in runs}
    planned = read(study / "planned-jobs.json", [])
    rows = read(study / "results.json", [])
    cases = []
    for job in planned:
        if job["action"] != "fit":
            continue
        key = f'{job["scenario"]}-{job["start_seed"]}-{job["variant"]}'
        fitted = read(study / f"fit-{key}.json", {})
        scored = read(study / f"score-{key}.json", {})
        progress = read(study / f"fit-{key}.progress.json", {})
        if not fitted:
            fitted = dict(status="interrupted" if args.final and progress else
                          "not_completed" if args.final else "pending_or_running", progress=progress)
        cases.append(dict(**job, fit=fitted, score=scored or dict(status="unscored")))
    evidence = dict(updated_utc=datetime.now(timezone.utc).isoformat(), final=args.final,
        guard=guard, summaries=runs, configurations=configs, fitting_cases=cases,
        planned_job_count=len(planned), recorded_job_count=len(rows),
        prior_study_note="study-128 stopped during its first fit when the user revised the budget; no completed fit claim.")
    write(out / "evidence.json", evidence)

    manifest_path = raw / "manifest.json"
    manifest = read(manifest_path, [])
    if not any(r["archive"] == "pilots-and-interrupted-study.tar.gz" for r in manifest):
        manifest.append(backup(base, raw, ["pilot-512", "pilot-9000", "study-128"],
            ["gpu-verified.json", "wheel-manifest.json", "watchdog-smoke/state.json"],
            "pilots-and-interrupted-study.tar.gz"))
    if args.final:
        name = "six-hour-study.tar.gz"
        manifest = [r for r in manifest if r["archive"] != name]
        manifest.append(backup(base, raw, ["study-512-six-hour"],
            ["active-study.json", "six-hour-guard-state.json", "six-hour-run.log",
             "six-hour-run.err", "six-hour-guard.log", "six-hour-guard.err"], name))
    write(manifest_path, manifest)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, run, title in zip(axes, ("pilot-512", "pilot-9000"),
                             ("512 trials per condition", "9,000 trials per condition")):
        measured = runs[run]["micro"]["baseline"]
        values = [measured[v] for v in ("legacy", "early_stop")]
        bars = ax.bar(["Original", "Early stopping"], values, color=["#728094", "#26786c"], width=.58)
        ax.bar_label(bars, labels=[f"{v:.2f} s" for v in values], padding=5)
        ax.set_ylim(0, max(values) * 1.22)
        ax.set_ylabel("Median simulation time (seconds)")
        ax.set_title(f'{title}\n{measured["legacy_to_early_speedup"]:.2f}x faster')
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Measured on the RTX 3060 laptop GPU", fontsize=15)
    fig.text(.5, .01, "Model B K-L | chunk 8 | 3 warmed repetitions | one simulator seed | fitting quality evaluated separately", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(out / "simulation-pilots.png", dpi=160)
    plt.close(fig)

    lines = ["# RTX 3060 validation", "",
        f'Updated {evidence["updated_utc"]}. **Study {"finished/stopped" if args.final else "in progress"}.**', "",
        "The 9,000-trial pilot measured **90.85 s -> 8.60 s**, or **10.56x faster simulation**.",
        "This establishes a simulation result for this workload. Fitting to equal quality is a separate question.", "",
        "![Measured simulation times](simulation-pilots.png)", "",
        "| Pilot trials per condition | Original | Early stopping | Speedup |",
        "|---|---:|---:|---:|"]
    for run, n in (("pilot-512", 512), ("pilot-9000", 9000)):
        m = runs[run]["micro"]["baseline"]
        lines.append(f'| {n:,} | {m["legacy"]:.4f} s | {m["early_stop"]:.4f} s | {m["legacy_to_early_speedup"]:.2f}x |')
    lines += ["", "Both pilots used chunk 8, three warmed repetitions and simulator seed 0.",
        "In the 512-trial pilot, early stopping and its full-horizon blocked control produced exactly matching RTs and choices for all 1,024 trials.",
        "The laptop took about 23x longer than the H100 for the 9,000-trial workload; chunk sizes, drivers and CUDA dependencies differed.", "",
        "The smaller study-128 measured 8.35x/8.16x/7.22x simulation speedups across three settings. Its paired outputs matched exactly across all three simulator seeds. Some category RT screens lacked enough samples; these are unscored checks, not statistical equivalence proof.", "",
        "## Six-hour quality study", "",
        "Six hours total includes prior GPU pilots. Absolute stop: **October 13, 9:00 p.m. Eastern (October 14, 01:00 UTC)**.",
        "The benchmark and an independent Windows guard enforce the deadline. Keep the laptop powered, awake and the Codex app open for the follow-up.", "",
        "```text", "Three parameter settings + four paired starts", "                    |",
        "         Original vs early stopping", "            512 trials for fitting", "                    |",
        "      Save estimates and training times", "                    |",
        " Original simulator: 9,000 trials, three new random keys", "                    |",
        "       Compare recovery + prediction quality", "```", "",
        "Eight planned fits use at most 1,000 iterations and 30 minutes of training each. All other improvements remain disabled. Observations contain 9,000 trials per condition and were fingerprint-verified before reuse.",
        "The target remains <=5% mean error on effective active parameters AND independent-MC median G2 within the declared true-parameter margin. Score the first eligible saved checkpoint and final estimate, using keys 999/5001/5002. Scoring does not guide fitting. It uses the same observations with independent simulator randomness, not held-out human data.", "",
        "| Setting / start | Variant | Fit status | Converged | Recovery error | Target reached | Observed fit time to target |",
        "|---|---|---|---|---:|---|---:|"]
    for case in cases:
        f, s = case["fit"], case["score"]
        error = f.get("mean_active_effective_parameter_error")
        hit = s.get("target_reached")
        at = s.get("observed_time_to_target_seconds")
        lines.append(f'| {case["scenario"]} / {case["start_seed"]} | {case["variant"]} | {f["status"]} | {f.get("converged", "unknown")} | {f"{error:.2%}" if error is not None else "unmeasured"} | {hit if hit is not None else "unscored"} | {f"{at:.1f} s" if at is not None else "no measured hit"} |')
    lines += ["", "| Paired setting / start | Observed time-to-quality ratio (original / early stop) |",
              "|---|---|"]
    groups = {}
    for case in cases:
        groups.setdefault((case["scenario"], case["start_seed"]), {})[case["variant"]] = case["score"]
    for (scenario, seed), pair in groups.items():
        a, b = pair.get("legacy", {}), pair.get("early_stop", {})
        if (a.get("target_reached") and b.get("target_reached")
                and a.get("observed_time_to_target_seconds") is not None
                and b.get("observed_time_to_target_seconds", 0)):
            description = f'{a["observed_time_to_target_seconds"] / b["observed_time_to_target_seconds"]:.2f}x'
        else:
            description = "Not estimable: both target hits have not been verified"
        lines.append(f"| {scenario} / {seed} | {description} |")
    lines += ["", "A time limit or iteration limit is not convergence. A missing target hit is censored, not a speedup estimate. Training time to a scored target excludes later diagnostic scoring and warm-up, and is an upper bound at coarse saved checkpoints.",
        "This is a small synthetic study on one laptop. It does not settle all parameter settings, human-data validity, or fitting with 9,000 training trials.", "",
        "## Reproducibility and backup", "",
        "[Plan](../../plans/2026-10-13-rtx3060-early-stop-validation.md), [compact evidence](evidence.json), [archive/file SHA-256 manifest](raw/manifest.json). Raw archives retain settings, source snapshots, samples, logs, traces and completed/partial results.",
        "GPU JAX 0.10.2, NumPy 2.5.4, SciPy 1.18.1; x64 enabled, fp32 fields/fp64 RT; reduced basis cache 2 for both variants. Allocator limit 3 GiB on the 6 GiB GPU. Library defaults were unchanged.",
        "Before the study, 17 focused GPU tests passed; seven driver/guard tests passed after the scheduling changes, and the guard stopped its live smoke process.",
        "The H100 pod remains stopped. The offered $3.48 RunPod balance is reserved for a focused follow-up if needed.", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="D:/ratcliff-3060")
    parser.add_argument("--output", default="docs/results/2026-10-13-rtx3060")
    parser.add_argument("--final", action="store_true")
    main(parser.parse_args())
