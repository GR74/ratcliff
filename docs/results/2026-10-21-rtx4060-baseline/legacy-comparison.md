# RTX 4060 original versus improved Model B

Status: **stopped_by_user_after_fit**. Updated 2026-10-22T01:28:02.217519+00:00.

Original means the archived full-horizon JAX comparator at `322bd1f`, with the same noise repair as the improved version. No scientific or optimizer code is changed.

| Measurement | Original | Improved |
| --- | --- | --- |
| Paired simulation, median seconds | 96.426 | 8.669 |
| simulation speed ratio (original/improved) | 11.123x | Same-session 30 pairs |
| Fit status | fit_time_limit | completed |
| Converged | False | True |
| Training minutes | 135.92 | 94.63 |
| Evaluations | 84 | 563 |
| Mean active error | 8.098% | 4.586% |
| Worst active error | 19.546% | 16.635% |

Training uses the historical 135-minute soft cap, checked after each evaluation. A capped fit is not a completed fit; no time-to-quality speedup is established. Warm-up and scoring are reported separately in the evidence.

Timing intervals describe this session only. Objective pairs clear the basis cache symmetrically; these differ from the earlier unpaired timing suite. Common fresh-seed scoring reuses the improved run's saved truth samples. Independent scientific validation, prediction-threshold calibration and the 27-fit suite remain outstanding.

[Improved-run report](README.md) · [Comparison evidence](legacy-comparison-evidence.json)

The user stopped remaining objective timing and scoring after the simulation benchmark and fit finished. Partial objective timings are preserved; no completed objective-cost or independent-scoring result is claimed. BADS proceeds using the previously completed improved-run control.
