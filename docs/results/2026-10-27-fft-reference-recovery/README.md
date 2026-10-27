# FFT-reference recovery comparison

Status: **complete_case_study**. Phase: **finished**. Completed fits: 3/3.
Updated UTC: 2026-10-27T06:07:31.082847+00:00. Detail: `{}`.

Reference: **corrected, Fortran-informed JAX bench model**, not the original
Fortran executable. sv, one-based geometry, MAXLOC ties, timeout handling and
binned qant1 scoring are corrected identically in all arms. Direct executable
qualification remains pending; float32 accumulation and different RNG remain
limitations. No silent fallback to the old port is allowed.

All arms fit the identical saved FFT199 dataset, seed 2026102701, 9,000 trials
per condition. Training seed 321, shared initial point, simplex and 135-minute
cap. All 13 parameters are active. Caps/failures remain visible. The older
KL-generated-data fits are not controls for this experiment.

| Method | Training minutes | Evaluations | Mean parameter error | Termination |
| --- | --- | --- | --- | --- |
| fft199 | 74.01 | 591 | 9.20% | Optimization terminated successfully. |
| fft200 | 66.40 | 630 | 7.68% | Optimization terminated successfully. |
| repaired_kl | 87.21 | 550 | 9.05% | Optimization terminated successfully. |

Final estimates are independently scored first, with 20 seeds using both native
and common FFT199 predictions, then time checkpoints. Generating truth and the
fitted FFT199 are separate references. Raw samples, traces, hashes, resource
telemetry and diagnostic samples are retained under `.benchmark-results/2026-10-27-fft-reference-recovery`.
This is one dataset and one start, not a general recovery or equivalence claim.
Prediction intervals describe Monte Carlo uncertainty conditional on estimates;
they are not recovery uncertainty. Engineering tolerances are not Fortran-calibrated.
