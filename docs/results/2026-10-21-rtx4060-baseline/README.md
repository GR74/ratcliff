# RTX 4060 Model B archived-case baseline

**Status: archived_case_screen_completed.** Updated 2026-10-21T22:08:56.935732+00:00.

This is the local RTX 4060 Laptop GPU baseline for repaired source `322bd1f`. It does not compare speed against the historical RTX 3060. The 27-fit qualification suite is deferred until the archived-case runtime has been reviewed.

[Study protocol](../../plans/2026-10-21-rtx4060-baseline.md) · [Compact evidence](evidence.json)

## Measurements

| Metric | Result |
| --- | --- |
| Pinned JAX GPU execution | Passed |
| Frozen Model B GPU regression tests | 10 passed |
| Warmed simulation, both conditions | Median 9.543 s; descriptive 95% interval [9.268, 9.806] s; 30 calls |
| Complete objective, changing widths | Median 11.302 s; descriptive 95% interval [11.099, 11.642] s; 30 calls |
| Full-horizon blocked control | RT exact fraction 1.000000; category exact fraction 1.000000 |
| Simulator first call (includes compilation) | 17.550 s |
| Objective first warm-up | 11.258 s; simulator already initialized |
| Fit status | completed; optimizer convergence True |
| Training time | 94.63 min |
| Fit initialization/warm-up | 15.916 s |
| Objective evaluations | 563 |
| Mean active relative error | 4.586% |
| Worst active relative error | 16.635% |
| Proposed recovery screen | Fail |
| Training after last strict improvement | 25.48 min |
| Fresh-seed G2, truth | Median 48.813; SD 9.637; 20 seeds |
| truth: mean of seedwise maximum probability errors (pp) | 1.130 |
| truth: mean of seedwise maximum conditional RT errors (ms) | 3.677 |
| Fresh-seed G2, fit | Median 71.623; SD 11.150; 20 seeds |
| fit: mean of seedwise maximum probability errors (pp) | 1.704 |
| fit: mean of seedwise maximum conditional RT errors (ms) | 3.993 |
| Spatial SW1, mean across 20 paired seeds | Condition 1: 1.413 pixels; condition 2: 1.171 pixels |
| Historical full-horizon G2, truth | Median 35.760; keys 999, 5001, 5002 |
| Historical full-horizon G2, fit | Median 80.302; keys 999, 5001, 5002 |

## Individual parameter recovery

| Parameter | Truth | Effective returned estimate | Signed error | Absolute relative error |
| --- | --- | --- | --- | --- |
| ter | 200 | 207.017 | +7.01735 | 3.509% |
| st | 50 | 54.0586 | +4.0586 | 8.117% |
| cr | 10 | 9.5873 | -0.4127 | 4.127% |
| crsd | 2 | 2.3327 | +0.332699 | 16.635% |
| sis | 12 | 11.3598 | -0.640187 | 5.335% |
| sig | 10 | 9.90233 | -0.0976679 | 0.977% |
| sv (inactive) | 0.5 | 0.468379 | -0.0316212 | Excluded |
| av1_c1 | 15 | 14.8975 | -0.102522 | 0.683% |
| av2_c1 | 10 | 10.2655 | +0.265541 | 2.655% |
| av3_c1 | 8 | 7.60133 | -0.398669 | 4.983% |
| av1_c2 | 14 | 14.1672 | +0.167171 | 1.194% |
| av2_c2 | 11 | 11.0883 | +0.0882787 | 0.803% |
| av3_c2 | 9 | 8.45826 | -0.541739 | 6.019% |

## Limits and remaining qualification

The simulator reference is the frozen JAX implementation. Exact blocked-control agreement does not independently establish scientific fidelity. The observed data are the archived synthetic observations; fresh simulator seeds are independent of optimization, not held-out participant data.

Prediction acceptance has not been calibrated, so this screen does not assert a joint-quality pass or time to acceptable quality. The 27-fit suite, parameter bias/RMSE across datasets, successful-fit rate and independent scientific-reference fidelity remain unmeasured. Spatial distances exclude no-crossing placeholders; their probability mass is recorded separately.

Raw timings, samples, trace, GPU telemetry, pinned dependencies and source hashes are under `.benchmark-results/2026-10-21-rtx4060-baseline/` in the repository. The measurement driver preserves source hashes and aborts on an observation-parity failure.
