# Repaired K L fitting on the RTX 3060

**Verification failed before fitting; no recovery result.**

Both variants enable the mode-indexed noise repair. Every optimization evaluation uses 9,000 trials per condition. The observed data, 8.688% starting error and ordinary simplex settings are matched; other improvements are off.

Hard stop: **October 15, 2026 at 7:24:24 p.m. Eastern**. This includes setup/pilots and never extends the six-hour allowance.

| Variant | Status | Returned parameter error | Training minutes | Converged | Joint target verified |
|---|---|---:|---:|---|---|
| early_stop | missing | Unavailable | Unavailable | Unavailable | None |
| legacy | missing | Unavailable | Unavailable | Unavailable | None |

![Recovery during optimization](recovery-progress.png)

The plot follows strict improvements in training loss; recovery error does not guide the optimizer. The actual returned estimate is reported separately because tied losses can select another simplex vertex.

Independent scoring uses the repaired original full-horizon simulator at 9,000 trials for keys 999, 5001 and 5002. A verified joint hit requires at most 5% mean active-parameter error and the predeclared prediction-quality margin. A low error alone, optimizer convergence, or a time cap is not a successful joint hit.

**Time-to-quality ratio is not estimable: both target hits have not been verified.**

This is one paired synthetic case on one laptop. Missing, interrupted and time-capped fits are retained. Scoring uses the same observations with independent simulator randomness, not held-out participant data.

[Predeclared plan](../../../plans/2026-10-15-stable-kl-9000-rtx3060.md) · [Compact evidence](evidence.json)

[Verified raw archive manifest](raw/manifest.json). The archive includes source, observations, traces, scores, logs and guard evidence.
