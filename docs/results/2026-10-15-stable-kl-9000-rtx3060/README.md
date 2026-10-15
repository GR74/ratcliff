# Repaired K L fitting on the RTX 3060

**Final report.**

Both variants enable the mode-indexed noise repair. Every optimization evaluation uses 9,000 trials per condition. The observed data, 8.688% starting error and ordinary simplex settings are matched; other improvements are off.

GPU preflight passed 10 tests. The production nearby-width field change was 0.000802%; retained component sets identical: True.

Hard stop: **October 15, 2026 at 7:24:24 p.m. Eastern**. This includes setup/pilots and never extends the six-hour allowance.

| Variant | Status | Returned parameter error | Training minutes | Converged | Joint target verified |
|---|---|---:|---:|---|---|
| early_stop | completed | 4.586% | 93.68 | True | False |
| legacy | fit_time_limit | 7.960% | 135.33 | False | False |

Guard-confirmed exit: **October 15, 2026 at 06:46:31 PM Eastern**; no matching benchmark processes remain. All 8 planned jobs have 8 result records (7 completed, 1 fit_time_limit). Missing jobs: 0. The original fit reached its soft time cap; it did not converge. Warm-up and independent scoring are excluded from the training-minute column.


The forward-simulation pilot took **93.893 s original versus 9.271 s early-stop**, a descriptive **10.13x** ratio at 9,000 trials per condition. This is one timed call after two warm-ups, not a confidence interval or a fitting-speed estimate. The full-horizon blocked control and early stopping matched all **18,000** trial choices and RTs exactly. The original and blocked implementations use different random layouts; the same root key does not make their samples identical.


![Recovery during optimization](recovery-progress.png)

The plot follows strict improvements in training loss; recovery error does not guide the optimizer. The actual returned estimate is reported separately because tied losses can select another simplex vertex.

Independent scoring uses the repaired original full-horizon simulator at 9,000 trials for keys 999, 5001 and 5002. A verified joint hit requires at most 5% mean active-parameter error and the predeclared prediction-quality margin. A low error alone, optimizer convergence, or a time cap is not a successful joint hit.

Independent scoring for **early_stop** (lower G2 means less prediction mismatch):

| Estimate | G2 key 999 | G2 key 5001 | G2 key 5002 | Median G2 | Joint gate |
|---|---:|---:|---:|---:|---|
| True parameters | 35.760 | 34.324 | 84.072 | 35.760 | Reference |
| Checkpoint at evaluation 225 | 74.595 | 48.135 | 86.070 | 74.595 | False |
| Returned estimate | 82.109 | 50.309 | 80.302 | 80.302 | False |

The predeclared prediction cutoff is **40.336**. Parameter recovery below 5% and prediction quality are separate checks. The parameter error is a mean across twelve active parameters; individual errors may be higher. This provisional gate is not a statistical significance test or a finalized scientific accuracy criterion.

Independent scoring for **legacy** (lower G2 means less prediction mismatch):

| Estimate | G2 key 999 | G2 key 5001 | G2 key 5002 | Median G2 | Joint gate |
|---|---:|---:|---:|---:|---|
| True parameters | 35.760 | 34.324 | 84.072 | 35.760 | Reference |
| Returned estimate | 143.199 | 144.960 | 169.270 | 144.960 | False |

The predeclared prediction cutoff is **40.336**. Parameter recovery below 5% and prediction quality are separate checks. The parameter error is a mean across twelve active parameters; individual errors may be higher. This provisional gate is not a statistical significance test or a finalized scientific accuracy criterion.

**Time-to-quality ratio is not estimable: both target hits have not been verified.**

This is one paired synthetic case on one laptop. Missing, interrupted and time-capped fits are retained. Scoring uses the same observations with independent simulator randomness, not held-out participant data.

Relative to the earlier 512-trial study, both the noise assignment and optimization trial budget changed. This experiment does not separately estimate how much each change affects recovery.

[Predeclared plan](../../plans/2026-10-15-stable-kl-9000-rtx3060.md) · [Compact evidence](evidence.json)

Returned effective parameters and absolute relative errors:

| Parameter | Truth | Improvement 1 | Error | Original | Error |
|---|---:|---:|---:|---:|---:|
| ter | 200 | 207.017 | 3.509% | 215.1 | 7.550% |
| st | 50 | 54.0586 | 8.117% | 57.9719 | 15.944% |
| cr | 10 | 9.5873 | 4.127% | 9.03625 | 9.638% |
| crsd | 2 | 2.3327 | 16.635% | 2.35439 | 17.720% |
| sis | 12 | 11.3598 | 5.335% | 11.1857 | 6.786% |
| sig | 10 | 9.90233 | 0.977% | 10.0218 | 0.218% |
| sv (inactive; excluded) | 0.5 | 0.468379 | 6.324% | 0.564811 | 12.962% |
| av1_c1 | 15 | 14.8975 | 0.683% | 14.1422 | 5.718% |
| av2_c1 | 10 | 10.2655 | 2.655% | 10.0071 | 0.071% |
| av3_c1 | 8 | 7.60133 | 4.983% | 6.46031 | 19.246% |
| av1_c2 | 14 | 14.1672 | 1.194% | 14.8547 | 6.105% |
| av2_c2 | 11 | 11.0883 | 0.803% | 10.8899 | 1.000% |
| av3_c2 | 9 | 8.45826 | 6.019% | 8.50325 | 5.519% |

[Verified raw archive manifest](raw/manifest.json). The archive includes source, observations, traces, scores, logs and guard evidence.
