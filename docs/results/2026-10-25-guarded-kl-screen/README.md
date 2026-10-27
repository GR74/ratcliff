# Guarded KL implementation screen

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
| 5 | 0.413 | 1.565 | 0.426 | 1.033 [1.000, 1.089] |
| 10 | 0.387 | 1.156 | 1.522 | 3.938 [3.779, 4.171] |
| 15 | 0.383 | 0.915 | 0.916 | 2.395 [2.310, 2.570] |

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

1,024 paired trials per cell; seed 20261027. The full-grid
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
| 5 | baseline | +0.00 [-0.43, +0.43] | +0.00 |
| 5 | weak_drift | +0.00 [-0.43, +0.43] | +0.00 |
| 5 | zero_drift | +0.00 [-0.43, +0.43] | +0.00 |
| 10 | baseline | +0.59 [-0.97, +2.13] | +0.13 |
| 10 | weak_drift | +0.29 [-1.48, +2.06] | +0.19 |
| 10 | zero_drift | -0.10 [-1.87, +1.67] | +0.24 |
| 15 | baseline | +0.00 [-1.80, +1.80] | +0.38 |
| 15 | weak_drift | -0.78 [-2.56, +1.01] | +0.27 |
| 15 | zero_drift | -0.68 [-2.19, +0.84] | +0.50 |

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
