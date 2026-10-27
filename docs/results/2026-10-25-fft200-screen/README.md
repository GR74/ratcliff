# FFT200: initial correctness and speed screen

Status: **complete**. Elapsed 13.90 minutes on NVIDIA GeForce RTX 4060 Laptop GPU.
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

Audited **178 widths**, including a 101-point domain sweep,
named values, perturbations and widths from the first 30 archived optimizer
requests. The raw records contain unclipped minimum eigenvalues and imaginary
spectrum residuals. All audited embeddings passed the declared PSD/covariance
checks. Float64 maximum errors over every required spatial lag:

- Candidate versus analytical covariance: 7.994e-15.
- Old FFT versus analytical covariance: 7.994e-15.
- Candidate versus old FFT: 5.054e-15.
- Bound after spatial demeaning: 2.022e-14.
- Float32 spectrum-factor versus analytical covariance: 1.077e-07.
- Float32 candidate versus old FFT: 7.460e-08.

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
| 128 | steady | 30 | 0.439 | 0.409 | 0.931 [0.903, 0.948] |
| 1,024 | steady | 30 | 1.179 | 1.009 | 0.856 [0.832, 0.865] |
| 9,000 | steady | 30 | 7.886 | 6.546 | 0.830 [0.822, 0.838] |
| 1,024 | changing_sig | 30 | 1.173 | 1.055 | 0.900 [0.886, 0.919] |

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
| repaired_kl | 5 | 8.680 |
| bucketed_kl | 5 | 7.144 |

## Random-field diagnostics

4,096 fields per method and sig, from independent method streams; each complex
FFT yields a real/imaginary pair. Statistics are exploratory and supplement the
deterministic covariance audit. Covariance at four directional lags and correlation
between successive independently generated pairs are also in `screen.json`.

| sig | Method | Fields | Mean within-field variance | Spatial maximum p99 | Real/imaginary anchor correlation |
| --- | --- | --- | --- | --- | --- |
| 5 | fft199 | 4,096 | 1.0008 | 4.668 | -0.0095 |
| 5 | fft200 | 4,096 | 1.0008 | 4.755 | -0.0229 |
| 10 | fft199 | 4,096 | 1.0014 | 4.427 | +0.0396 |
| 10 | fft200 | 4,096 | 0.9989 | 4.426 | -0.0123 |
| 15 | fft199 | 4,096 | 0.9938 | 4.204 | -0.0078 |
| 15 | fft200 | 4,096 | 0.9979 | 4.228 | +0.0150 |

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

JAX 0.10.2, NumPy 2.5.4, production default matmul precision,
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

## Regression tests

49 focused CPU tests passed: covariance construction, flag propagation,
existing guarded-KL and blocked behavior, first-passage edge cases and the
default FFT path. Exact commands and timings are in `validation.json`.
