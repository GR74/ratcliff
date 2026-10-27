# FFT200 matched to the two archived RTX 4060 fits

Status: **complete_archived_screen**, phase **finished**. Updated: 2026-10-27T00:50:46.297232+00:00.
Detail: `{}`.

Only FFT200 is being fitted. Both previous fits are reused. Dataset SHA-256,
truth, exact start, fixed key 321, 9,000 trials per condition, chunk 8, all 13
parameters, Nelder-Mead/tolerances, 1,000 iterations, 135-minute soft cap,
checkpoint rules, frozen source and package versions have been checked.
FFT200 uses block 32 and early stopping like the improved KL comparator.

| Method | Status | Evaluations | Training minutes | Mean active recovery error |
| --- | --- | --- | --- | --- |
| legacy_full_horizon_kl | fit_time_limit | 84 | 135.92 | 8.10% |
| improved_kl | completed | 563 | 94.63 | 4.59% |
| fft200 | completed | 586 | 56.97 | 6.10% |

The prior "original" fit was full-horizon JAX **KL**, not FFT199 or Fortran.
Its cap means it is not a completed-fit timing reference. The improved KL fit
converged but its worst active error was 16.64%, above the 10% recovery screen.
The earlier fresh-data FFT200 run is interrupted and excluded from this comparison.

Final independent scores use the original keys 6001–6020. Existing KL samples
and scores are reused; only missing legacy scores and FFT200 predictions are
computed. FFT200 is scored natively and through the common KL scorer used for
the old fits. Report native and common results separately. Training warmup,
training, scoring and total wall time stay separate. This is an archived-case
screen, not behavioral equivalence or broad recovery qualification. Parameter
sv remains inert in this port, a known scientific limitation.
