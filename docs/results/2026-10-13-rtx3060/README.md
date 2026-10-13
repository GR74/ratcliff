# RTX 3060 validation

Updated 2026-10-13T19:46:00.006528+00:00. **Study in progress.**

The 9,000-trial pilot measured **90.85 s -> 8.60 s**, or **10.56x faster simulation**.
This establishes a simulation result for this workload. Fitting to equal quality is a separate question.

![Measured simulation times](simulation-pilots.png)

| Pilot trials per condition | Original | Early stopping | Speedup |
|---|---:|---:|---:|
| 512 | 4.8816 s | 0.4853 s | 10.06x |
| 9,000 | 90.8538 s | 8.6001 s | 10.56x |

Both pilots used chunk 8, three warmed repetitions and simulator seed 0.
In the 512-trial pilot, early stopping and its full-horizon blocked control produced exactly matching RTs and choices for all 1,024 trials.
The laptop took about 23x longer than the H100 for the 9,000-trial workload; chunk sizes, drivers and CUDA dependencies differed.

The smaller study-128 measured 8.35x/8.16x/7.22x simulation speedups across three settings. Its paired outputs matched exactly across all three simulator seeds. Some category RT screens lacked enough samples; these are unscored checks, not statistical equivalence proof.

## Six-hour quality study

Six hours total includes prior GPU pilots. Absolute stop: **October 13, 9:00 p.m. Eastern (October 14, 01:00 UTC)**.
The benchmark and an independent Windows guard enforce the deadline. Keep the laptop powered, awake and the Codex app open for the follow-up.

```text
Three parameter settings + four paired starts
                    |
         Original vs early stopping
            512 trials for fitting
                    |
      Save estimates and training times
                    |
 Original simulator: 9,000 trials, three new random keys
                    |
       Compare recovery + prediction quality
```

Eight planned fits use at most 1,000 iterations and 30 minutes of training each. All other improvements remain disabled. Observations contain 9,000 trials per condition and were fingerprint-verified before reuse.
The target remains <=5% mean error on effective active parameters AND independent-MC median G2 within the declared true-parameter margin. Score the first eligible saved checkpoint and final estimate, using keys 999/5001/5002. Scoring does not guide fitting. It uses the same observations with independent simulator randomness, not held-out human data.

| Setting / start | Variant | Fit status | Converged | Recovery error | Target reached | Observed fit time to target |
|---|---|---|---|---:|---|---:|
| baseline / 0 | legacy | pending_or_running | unknown | unmeasured | unscored | no measured hit |
| baseline / 0 | early_stop | pending_or_running | unknown | unmeasured | unscored | no measured hit |
| lower_boundary / 0 | legacy | pending_or_running | unknown | unmeasured | unscored | no measured hit |
| lower_boundary / 0 | early_stop | pending_or_running | unknown | unmeasured | unscored | no measured hit |
| higher_boundary / 0 | legacy | pending_or_running | unknown | unmeasured | unscored | no measured hit |
| higher_boundary / 0 | early_stop | pending_or_running | unknown | unmeasured | unscored | no measured hit |
| baseline / 1 | early_stop | pending_or_running | unknown | unmeasured | unscored | no measured hit |
| baseline / 1 | legacy | pending_or_running | unknown | unmeasured | unscored | no measured hit |

| Paired setting / start | Observed time-to-quality ratio (original / early stop) |
|---|---|
| baseline / 0 | Not estimable: both target hits have not been verified |
| lower_boundary / 0 | Not estimable: both target hits have not been verified |
| higher_boundary / 0 | Not estimable: both target hits have not been verified |
| baseline / 1 | Not estimable: both target hits have not been verified |

A time limit or iteration limit is not convergence. A missing target hit is censored, not a speedup estimate. Training time to a scored target excludes later diagnostic scoring and warm-up, and is an upper bound at coarse saved checkpoints.
This is a small synthetic study on one laptop. It does not settle all parameter settings, human-data validity, or fitting with 9,000 training trials.

## Reproducibility and backup

[Plan](../../plans/2026-10-13-rtx3060-early-stop-validation.md), [compact evidence](evidence.json), [archive/file SHA-256 manifest](raw/manifest.json). Raw archives retain settings, source snapshots, samples, logs, traces and completed/partial results.
GPU JAX 0.10.2, NumPy 2.5.4, SciPy 1.18.1; x64 enabled, fp32 fields/fp64 RT; reduced basis cache 2 for both variants. Allocator limit 3 GiB on the 6 GiB GPU. Library defaults were unchanged.
Before the study, 17 focused GPU tests passed; seven driver/guard tests passed after the scheduling changes, and the guard stopped its live smoke process.
The H100 pod remains stopped. The offered $3.48 RunPod balance is reserved for a focused follow-up if needed.
