"""Report the bounded FFT200 screen without promoting it to fitting qualification."""
import argparse
import json
from pathlib import Path

import numpy as np


def main(out):
    out = Path(out)
    run = json.loads((out / "screen.json").read_text())
    rng = np.random.default_rng(20261030)
    summary = {"status": run["status"], "covariance_widths": len(run["covariance"]),
               "covariance_worst": {}, "fft_comparisons": [], "kl_controls": []}
    for metric in ["max_target_error", "old_target_error", "max_reference_error",
                   "max_fp32_target_error", "max_fp32_reference_error", "demeaned_reference_error_upper_bound"]:
        summary["covariance_worst"][metric] = max(r[metric] for r in run["covariance"])
    timing_rows = []
    for phase, nsim in [("steady", 128), ("steady", 1024), ("steady", 9000), ("changing_sig", 1024)]:
        by_method = {}
        for name in ["fft199", "fft200"]:
            by_method[name] = {r["repetition"]: r["seconds"] for r in run["timings"]
                               if r["phase"] == phase and r["nsim"] == nsim and r["method"] == name}
        paired = sorted(set(by_method["fft199"]) & set(by_method["fft200"]))
        if not paired:
            continue
        a = np.array([by_method["fft199"][i] for i in paired])
        b = np.array([by_method["fft200"][i] for i in paired])
        indices = rng.integers(len(a), size=(10000, len(a)))
        ratios = np.median(b[indices], axis=1)/np.median(a[indices], axis=1)
        lo, hi = np.quantile(ratios, [.025, .975])
        ratio = np.median(b)/np.median(a)
        row = dict(phase=phase, nsim=nsim, pairs=len(a), baseline_median=float(np.median(a)),
                   candidate_median=float(np.median(b)), ratio=float(ratio), ci95=[float(lo), float(hi)],
                   preliminary_speed_target=bool(ratio <= .9 and hi < 1 and len(a) == 30))
        summary["fft_comparisons"].append(row)
        timing_rows.append(f"| {nsim:,} | {phase} | {len(a)} | {np.median(a):.3f} | {np.median(b):.3f} | {ratio:.3f} [{lo:.3f}, {hi:.3f}] |")
    control_rows = []
    for name in ["repaired_kl", "bucketed_kl"]:
        values = [r["seconds"] for r in run["timings"] if r["phase"] == "control_screen" and r["method"] == name]
        if values:
            summary["kl_controls"].append(dict(method=name, calls=len(values), median=float(np.median(values))))
            control_rows.append(f"| {name} | {len(values)} | {np.median(values):.3f} |")
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    field_rows = []
    for r in run["fields"]:
        field_rows.append(f"| {r['sig']:g} | {r['method']} | {r['count']:,} | {r['mean_spatial_variance']:.4f} | {r['maximum_p99']:.3f} | {r['real_imag_anchor_correlation']:+.4f} |")
    c = summary["covariance_worst"]
    validation_path = out / "validation.json"
    validation_text = ""
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        total = sum(r["passed"] for r in validation["runs"])
        validation_text = f"\n## Regression tests\n\n{total} focused CPU tests passed: covariance construction, flag propagation,\nexisting guarded-KL and blocked behavior, first-passage edge cases and the\ndefault FFT path. Exact commands and timings are in `validation.json`.\n"
    report = f"""# FFT200: initial correctness and speed screen

Status: **{run['status']}**. Elapsed {run['elapsed_seconds']/60:.2f} minutes on {', '.join(run['devices'])}.
The candidate is opt-in; defaults are unchanged. This screen does not establish
parameter recovery, behavioral equivalence margins or time-to-acceptable-fit improvement.

## What changed

The new `fft200=True, use_kl=False` option constructs a 200-by-320 covariance
embedding and samples its full spectrum. The decision grid stays 100 by 160.
The new Nyquist row/column are averaged under periodic inversion to maintain a
real symmetric covariance. All displacement covariances used by original-grid
pairs are retained. No positive spectral modes are intentionally dropped.

The option is available on `simulate_b`, `simulate_b_blocked`,
`simulate_b_blocked_with_work`, `fofs_b_new`, and `fit_simplex_b`. It rejects KL
combinations and sig outside [0.2,17]. The generic even builder requires its own
PSD audit if used with another grid or correlation domain.

## Covariance and integration

Audited **{len(run['covariance'])} widths**, including a 101-point domain sweep,
named values, perturbations and widths from the first 30 archived optimizer
requests. The raw records contain unclipped minimum eigenvalues and imaginary
spectrum residuals. All audited embeddings passed the declared PSD/covariance
checks. Float64 maximum errors over every required spatial lag:

- Candidate versus analytical covariance: {c['max_target_error']:.3e}.
- Old FFT versus analytical covariance: {c['old_target_error']:.3e}.
- Candidate versus old FFT: {c['max_reference_error']:.3e}.
- Bound after spatial demeaning: {c['demeaned_reference_error_upper_bound']:.3e}.
- Float32 spectrum-factor versus analytical covariance: {c['max_fp32_target_error']:.3e}.
- Float32 candidate versus old FFT: {c['max_fp32_reference_error']:.3e}.

The demeaning figure is a conservative bound of four times the maximum
pre-demeaning covariance-entry error, not a separately sampled statistic.
The float32 checks reconstruct covariance from float32 spectral square roots;
they do not bound all downstream floating-point accumulation error.

Public current-simulator outputs matched the candidate inserted into the frozen
harness for both full-horizon and blocked RT/category checks. This tests the
integration, not path identity between old and new random samplers. Different
embedding sizes consume different random draws.

## Complete objective timing

All arms use repaired source `322bd1f`'s blocking, random indexing, early stopping,
compaction and loss. Only the covariance builder changes between FFT199 and
FFT200. Timings include spectrum construction, RNG, simulation and scoring of
both experimental conditions. Same chunk 8, block 32, seed 20261029, archived
observations and truth parameters; no optimizer was run.

Three warm-ups precede timed comparisons; first calls and warm-ups are retained
separately. GPU calls are synchronized and baseline/candidate order alternates.
`steady` means repeated truth (sig=10); `changing_sig` cycles 5,8,10,12,15,17.
FFT spectra are rebuilt on each objective call, as in the baseline; this label
does not imply a new spectral cache. Both methods use the same caching behavior.

| Trials per condition | Regime | Pairs | FFT199 seconds | FFT200 seconds | Candidate/control [95% CI] |
| --- | --- | --- | --- | --- | --- |
""" + "\n".join(timing_rows) + """

Intervals are descriptive paired bootstrap intervals over 10,000 resamples of
whole timing pairs. They do not establish across-day or multi-machine speedups.
Thirty complete pairs, median ratio <=0.90 and upper interval <1 satisfy the
preliminary objective-speed target for that measured regime only.
Repeated timings reuse one simulation key and parameter case. Their intervals
measure timing variability, not variability across subjects or simulated paths;
changing the FFT shape also changes the generated paths and early-stop work.

## Existing engineering controls at 9,000 trials per condition

Both KL arms retain the frozen repaired mode-indexed random draws; `bucketed_kl`
also enables the existing zero-padding reduction. These are small, alternating
five-call control screens after three warm-ups, measured before the FFT timing
stage. Comparisons to the later FFT stage are descriptive, not paired confidence
intervals. The capped KL controls retain their known scientific approximation
limitations.

| Control | Timed calls | Median objective seconds |
| --- | --- | --- |
""" + "\n".join(control_rows) + """

## Random-field diagnostics

4,096 fields per method and sig, from independent method streams; each complex
FFT yields a real/imaginary pair. Statistics are exploratory and supplement the
deterministic covariance audit. Covariance at four directional lags and correlation
between successive independently generated pairs are also in `screen.json`.

| sig | Method | Fields | Mean within-field variance | Spatial maximum p99 | Real/imaginary anchor correlation |
| --- | --- | --- | --- | --- | --- |
""" + "\n".join(field_rows) + f"""

## Scope and remaining qualification

This is the authorized initial screen, not the full prospective protocol.
It does not yet measure low-support win equivalence, conditional RT quantiles,
continuous response distances, full optimizer-trace replay, peak device memory,
end-to-end fit time or cross-generator parameter recovery. Kernel components
were not separately profiled. The archived trace segment is saved for a later
replay; changing-width timings must not be called that replay.

Numerical covariance agreement supports matching the Gaussian law before
finite-precision first-passage calculations, but does not independently qualify
the original JAX model against the paper/Fortran. The inactive drift-variability
discrepancy remains outside this computational change.

## Reproducibility

JAX {run['jax']}, NumPy {run['numpy']}, production default matmul precision,
float64 covariance construction and float32 field sampling. Archived data hash,
source hashes, frozen manifest, candidate source, runner source and raw timings
are saved here. Pre-change current source is preserved under
`.benchmark-results/2026-10-25-fft200/source-before/`.

```sh
python scripts/screen_fft200.py --seconds 1200 --out docs/results/NEW-UNUSED-DIRECTORY
python -m scripts.report_fft200 --out docs/results/NEW-UNUSED-DIRECTORY
```

The runner refuses to overwrite its result directory and checks for other GPU
compute workloads before measurement. See
`docs/plans/2026-10-25-exact-noise-candidates.md` for subsequent gates.
"""
    (out / "README.md").write_text(report + validation_text, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/results/2026-10-25-fft200-screen")
    main(parser.parse_args().out)
