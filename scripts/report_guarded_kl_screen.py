"""Summarize bounded implementation checks, timing and fresh-seed fidelity."""
import json
from pathlib import Path

import numpy as np
from scripts.report_kl_winner_fidelity import paired_binary_interval


def main():
    out = Path("docs/results/2026-10-25-guarded-kl-screen")
    screen = json.loads((out / "screen.json").read_text())
    rng = np.random.default_rng(20261028)
    timing_rows = []
    timing_metrics = []
    for sig in [5., 10., 15.]:
        times = {}
        losses = {}
        for method in ["fft", "legacy_kl", "guarded_kl"]:
            items = [r for r in screen["timings"] if r["sig"] == sig and r["method"] == method and r["phase"] == "cache_miss"]
            if not items:
                continue
            times[method] = np.array([r["seconds"] for r in items])
            losses[method] = np.array([r["loss"] for r in items])
        if len(times) != 3 or len({len(t) for t in times.values()}) != 1:
            continue
        if sig == 5:
            np.testing.assert_array_equal(losses["fft"], losses["guarded_kl"])
        n = len(times["fft"])
        indices = rng.integers(n, size=(10000, n))
        ratios = {}
        for control in ["fft", "legacy_kl"]:
            samples = np.median(times["guarded_kl"][indices], axis=1)/np.median(times[control][indices], axis=1)
            ratio = np.median(times["guarded_kl"])/np.median(times[control])
            low, high = np.quantile(samples, [.025, .975])
            ratios[control] = [float(ratio), float(low), float(high)]
        med = {k: float(np.median(v)) for k, v in times.items()}
        r, lo, hi = ratios["fft"]
        timing_rows.append(f"| {sig:g} | {med['fft']:.3f} | {med['legacy_kl']:.3f} | {med['guarded_kl']:.3f} | {r:.3f} [{lo:.3f}, {hi:.3f}] |")
        timing_metrics.append({"sig": sig, "paired_repetitions": n, "medians_seconds": med, "candidate_control_ratios_95ci": ratios})
    speed_pass = all(r["candidate_control_ratios_95ci"]["fft"][0] <= .9 and
                     r["candidate_control_ratios_95ci"]["fft"][2] < 1 for r in timing_metrics)
    fidelity_dir = out / "fidelity"
    fidelity_meta = json.loads((fidelity_dir / "metadata.json").read_text())
    data = np.load(fidelity_dir / "trials.npz")
    mask = (data["baseline_drift"] <= .1*data["baseline_drift"].max()).ravel()
    fidelity_rows = []
    fidelity_metrics = []
    for sig in [5., 10., 15.]:
        for scenario in ["baseline", "weak_drift", "zero_drift"]:
            fft = data[f"sig{sig}_fft_{scenario}"]
            candidate = data[f"sig{sig}_guarded_kl_{scenario}"]
            a = (candidate[:, 3] == 1) & mask[candidate[:, 2].astype(int)]
            b = (fft[:, 3] == 1) & mask[fft[:, 2].astype(int)]
            low, high = paired_binary_interval(a, b)
            difference = 100*(a.mean()-b.mean())
            rt = float(np.mean(candidate[:, 0]-fft[:, 0]))
            if sig == 5:
                np.testing.assert_array_equal(candidate, fft)
            fidelity_rows.append(f"| {sig:g} | {scenario} | {difference:+.2f} [{low:+.2f}, {high:+.2f}] | {rt:+.2f} |")
            fidelity_metrics.append(dict(sig=sig, scenario=scenario, delta_low_support_pp=difference,
                                          ci95=[low, high], delta_mean_rt_ms=rt))
    (out / "summary.json").write_text(json.dumps({"timing": timing_metrics, "fidelity": fidelity_metrics,
                                                  "all_settings_speed_target_met": speed_pass}, indent=2)+"\n")
    (out / "README.md").write_text("""# Guarded KL implementation screen

The opt-in policy is implemented, but **not adopted as the default and not
qualified as an end-to-end fitting speedup**. It enforces a 99.9% spectral target
within 2,000 complex modes, otherwise dispatching to the existing FFT path.
Legacy behavior and defaults are preserved.

Enable with `use_kl=True, guarded_kl=True` on `simulate_b`,
`simulate_b_blocked`, `simulate_b_blocked_with_work`, `fofs_b_new`, or
`fit_simplex_b`. Query `grf_kl.guarded_kl_diagnostics(sig)` for required/retained
modes, requested/achieved variance, cap status and selected generator. Combining
this policy with bucketed or cumulative KL is rejected to isolate this change.

## Complete-objective cost

Seconds per two-condition objective, 128 trials per condition, chunk 4, time
block 16, early stopping and compaction enabled. Three warm-ups, then 30
alternating-order repetitions with basis and policy caches cleared for every
arm; compiled kernels remain warm. Costs include spectrum/basis construction,
transfers, simulation and scoring. The fixed synthetic scoring bins serve
timing only, not predictive qualification or parameter recovery.

| sig | FFT seconds | Legacy KL seconds | Guarded seconds | Guarded/FFT ratio [95% CI] |
| --- | --- | --- | --- | --- |
""" + "\n".join(timing_rows) + """

Intervals are descriptive paired bootstrap intervals over 10,000 whole-round
resamples, not across-day performance guarantees. Per-sig guarded/legacy ratios
and raw synchronized timings are in `summary.json` and `screen.json`.
First-call timings are retained separately; later settings reuse earlier
compilations, so these are not independent cold-process measurements.

**The speed adoption target is not met in this screen.** FFT has lower measured
median cost at all three tested settings. Guarded fallback at sig=5 improves both fidelity and cost
relative to legacy capped KL, but at sig=10 enforcing the higher target increases
complete-objective cost. At sig=15 guarded and legacy KL costs are similar.
The same padded matrix dimensions do not guarantee the same total cost: building
more active basis columns and selecting the spectrum also take time. Do not
attribute the measured difference to any one component without profiling it.
No larger fit was launched after this negative preliminary speed result.

The guarded fallback matches FFT exactly in full-horizon and blocked simulator
RT/category checks, unblocked objective checks, and all timed blocked objective
values at sig=5. The small-budget screen is insufficient for claims about
9,000-trial objectives or time to acceptable fit quality. Retaining an inaccurate
legacy approximation is not justified by its speed in any setting.

## Fresh-seed distribution screen

""" + f"{fidelity_meta['trials_per_cell']:,} paired trials per cell; seed {fidelity_meta['seed']}. " + """The full-grid
diagnostic samples the selected KL Gaussian law using masked FFTs; it is not a
production GEMM throughput measurement. Production-basis/field and decision
observer cross-checks passed at all three correlation settings. For fallback
cases it uses the full FFT law and exactly matches the paired reference.
The diagnostic uses highest matrix-multiplication precision for its numerical
cross-checks; timing uses the production default precision recorded in
`screen.json`. This screen does not qualify all finite-precision production
first-passage distributions. That remains part of the larger fidelity protocol.

Low support is the fixed baseline-drift mask at 10% of peak. Non-crossers are
excluded from winning numerators; all trials remain in the denominator. In the
zero-drift control this is the same geographic mask, not a claim that other
locations have nonzero drift. Mean RT is a supporting diagnostic, not a substitute
for conditional quantile qualification.

| sig | Drift condition | Low-support win difference, pp [95% CI] | Mean RT difference, ms |
| --- | --- | --- | --- |
""" + "\n".join(fidelity_rows) + """

Intervals are conservative paired pointwise intervals, not simultaneous
qualification bounds. This is an implementation screen, not the planned
4,096-trial preliminary study or 20-seed qualification suite. No end-to-end fit
or parameter-recovery claim is made. The original JAX/Fortran scientific
reference limitations remain unresolved by this computational change.

## Reproduce

```sh
python scripts/screen_guarded_kl.py --seconds 600
python scripts/check_kl_winner_fidelity.py --guarded --seed 20261027 --trials 1024 --out docs/results/2026-10-25-guarded-kl-screen/fidelity
python -m scripts.report_guarded_kl_screen
```

Hardware/runtime and simulation source hashes are in each manifest. No GPU
fidelity experiment ran concurrently with the timed objective screen. The
performance adoption gates and later fit experiments remain in
`docs/plans/2026-10-25-model-b-adaptive-kl-fidelity.md`.

Validation: 23 initial policy/legacy-extension tests, 24 follow-up tests and
one dispatch-boundary test (37 distinct tests total) passed. Coverage includes policy propagation
through fitting and both objective paths, blocked stopping, invalid option
combinations, and scientific diagnostic counts/intervals. GPU integration
checks and the fresh-seed field/decision-observer checks also passed.
""", encoding="utf-8")
    print(json.dumps(timing_metrics, indent=2))


if __name__ == "__main__":
    main()
