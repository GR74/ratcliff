# Model B: smaller K-L padding, one-change laptop screen

Status: implemented and queued for an idle GPU. Fifteen existing frozen padding/stable-noise GPU tests passed. The first launch correctly refused to overlap an unrelated `ag_news_dropout.py` process; that attempt's log is preserved under `.benchmark-results/2026-10-22-rtx4060-padding-screen/`. The active waiting runner uses `.benchmark-results/2026-10-22-rtx4060-padding-screen-queued/`; do not launch a duplicate. Its 20-minute measurement budget starts after the GPU becomes idle. The shared live report is `docs/results/2026-10-22-rtx4060-padding-screen/README.md`.

## Objective

Test the existing `kl_bucketed=True` option against the frozen improved simplex baseline on the RTX 4060. The aim is cheaper evaluations with preserved decisions, not a new approximation or optimizer. At `sig=10`, the same 1,325 retained complex modes can use capacity 1,536 instead of 2,000. Removing zero-padding may save computation; shape changes may also introduce compilation overhead. This option was explored on H100 previously and is not claimed as a novel method.

Keep source `322bd1f`, stable mode-indexed noise, 9,000 trials per condition, chunk 8, block 32, cache size 2, precision, retained-variance rule and 2,000-mode cap unchanged. Do not combine this with BADS, cumulative K-L, active-only optimization, different trial budgets or scientific corrections. Use the completed improved simplex data and objective as control; preserve the negative BADS experiment.

## Predeclared short screen

1. Verify frozen source hashes and confirm the GPU is idle. Use the existing pinned baseline environment and archive the runner and input hashes. Cap the entire screen at 20 minutes, checked between synchronized calls; retain partial evidence if capped.
2. Verify identical retained-mode count, variance, active basis columns and physical mode identities between padded and bucketed bases. Check widths 0.2, 8, 10, 10.000001, 12 and 17; the archived starting and fitted widths; and both sides of each capacity transition reachable between 0.2 and 17. Locate those transitions from the spectrum, not by fitting data.
3. Compare retained-mode random coefficients and instrumented first-passage outputs for the archived truth, archived start, returned simplex parameters, low-boundary and high-boundary probes, and the width-transition probes. Use 11 trials per condition (exercises padding), seeds 0 and 1, and preserve RT/category/step/location/crossing comparisons. Exporting existing kernel results is measurement-only. Check original RT/category agreement before relying on instrumentation.
4. Compare original simulator outputs at 9,000 trials per condition, seed 0, at the archived truth. Any first-passage or RT/category mismatch stops the exact-equivalence fast track and records a numerical-fidelity gap. This is not by itself proof of a scientifically important error. Do not launch a full fit after such a mismatch; a separately specified distributional validation would be needed.
5. If fidelity checks pass, perform three warm-ups per variant at the archived truth and 30 alternating, synchronized complete-objective pairs at the existing width-vector suite from 8 to 12. Clear the host basis cache before each arm to expose basis construction fairly. Include first encounters with new compiled bucket shapes in the recorded suite; report each timing, total suite cost, warm-up cost, capacities and objective equality. No separate additional simulator-only timing stage.
6. Report candidate/control median cost and descriptive paired-bootstrap 95% interval (10,000 whole-pair resamples, seed 2201). A performance pass requires a candidate/control median ratio at most 0.90, its upper interval endpoint below 1.0, and no increase in total suite time. All compared objective values and decision outputs must match for the exact fast track. Failure to show benefit is a negative screen, not a reason to change thresholds.

## Follow-through and boundaries

The short screen is the next experiment. It does not automatically launch another multi-hour fit or the 27-fit suite. If it passes, use its measured cost to schedule one matched simplex fit with only this flag changed, compare recovery and independent scores, and account for shape compilation and cache behavior across the full search. If it fails, retain the results and diagnose the specific failure before considering another change.

The improved simplex already misses the worst-parameter recovery target. Preserving its outputs establishes an engineering equivalence result only, not acceptable scientific recovery. An isolated speed gain cannot establish a time-to-quality gain.

Save active raw outputs under `.benchmark-results/2026-10-22-rtx4060-padding-screen-queued/` and an updating report under `docs/results/2026-10-22-rtx4060-padding-screen/`. Preserve all baseline source and results. No public API or parameter-meaning changes.
