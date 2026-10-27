# Model B: enforce KL fidelity with an FFT fallback

Status: first candidate implemented as an opt-in policy after the user's
implementation request. Existing defaults remain unchanged. The bounded local
integration, objective-cost and fresh-seed fidelity screen is recorded in
`docs/results/2026-10-25-guarded-kl-screen/`. This is not adoption or full
scorecard qualification; the larger fitting experiments below remain gated on
the preliminary evidence.

## Objective and evidence

The primary contribution sought is **less computational time and resource use to
obtain an acceptable scientific fit**. Behavioral fidelity and parameter recovery
are preservation constraints, not substitutes for a demonstrated speed benefit.
Report a fidelity repair without a speed benefit as a fidelity repair, not a
successful performance optimization. Distinguish computer wall time from the
reaction times predicted by the model.

Improve agreement with the full FFT simulator while preserving useful runtime
savings. Agreement means response probabilities, spatial response distributions
and reaction-time distributions at identical parameters, not trial-by-trial
winner identity or a correlation coefficient. Keep scientific parameters fixed.

The exploratory study in `docs/results/2026-10-25-kl-winner-fidelity/README.md`
found that the current 99% target silently hits its 2,000-complex-mode cap:

| sig | Current retained variance | Modes for 99.9% | Baseline mean RT difference: current / 99.9% |
| --- | --- | --- | --- |
| 5 | 75.29% | 7,399 | +21.23 / +0.20 ms |
| 10 | 99.00% | 1,850 | +1.10 / +0.21 ms |
| 15 | 99.00% | 821 | +1.43 / +0.21 ms |

At sig=5, low-support wins changed from 23.05% with FFT to 13.28% with current
KL. Uncapped 99.9% retention reduced the difference to +0.49 percentage points.
These observations motivate the candidate; they do not prove equivalence,
establish a runtime benefit, or validate the current JAX port against the paper.

## First candidate: a guarded 99.9% KL policy

1. Calculate the covariance spectrum and the mode count required to retain at
   least 99.9% variance **before allocating the dense KL basis**. Return requested
   and achieved variance, required and retained counts, cap-hit status, and the
   selected generator as explicit diagnostics. Include the policy in cache keys.
2. If the required count is at most 2,000, use KL with those modes, initially
   padded to the existing 2,000-mode capacity. At sig=10 this uses 1,850 active
   modes in the same matrix dimensions as today. Sampling cost may therefore be
   similar; measure initialization, RNG, compilation and complete-objective cost.
3. If the target cannot be met within 2,000 modes, use full FFT with the same
   drift, thresholds, variability, normalization, grid and time-step rules.
   Never silently claim the target was achieved or rescale the truncated field
   to hide missing variance. Do not change sig to avoid a fallback.
4. Preserve explicit legacy FFT and legacy KL modes for reproducibility. Add the
   guarded policy as an opt-in path and keep defaults unchanged until adoption.
   Use matching blocked/early-stop settings for both arms; do not inadvertently
   compare a blocked KL path against an unrelated full-horizon FFT path.

This bounds KL memory and avoids materializing a 7,399-mode basis at sig=5.
The 2,000-mode cutoff is an initial compute guard, not an established speed
break-even point. The fallback can cost more than today's inaccurate KL path;
fidelity takes priority, and any additional cost must be disclosed.

Likely implementation touchpoints: `model_b/grf_kl.py` for spectrum selection and
diagnostics; `model_b/simulate.py` for deterministic dispatch outside JIT;
`model_b/objective.py` for opt-in policy propagation and cost accounting. Extend
tests beside the affected package. Freeze the actual baseline source first;
do not mix the current checkout with historical repaired-source benchmarks.

## Correctness checks before performance measurements

- Check exact target selection and cap detection at sig=0.2, 5, 8, 10, 12, 15,
  and 17, plus archived truth/start/fitted values and values on both sides of the
  discovered 2,000-mode boundary. Confirm valid spectral embeddings or explicitly
  report how the historical implementation treats invalid embeddings.
- For fallback cases, require agreement with the selected FFT reference under
  identical random inputs and execution settings. Check crossing indicators,
  first-crossing steps, locations, categories and RTs, including no-crossing cases.
- For KL, verify its sampled covariance and the production basis against the
  selected spectral law. Preserve independent time increments, physical mode
  identities and stable random streams where supported. Generator switches may
  alter common-random-number coupling; document this and check objective behavior
  near dispatch boundaries before running fits. Equal seeds alone are not proof
  of equivalent paths.
- Reuse the existing winner-observer cross-checks. Confirm the new diagnostic
  outputs reproduce the actual production outputs before interpreting metrics.
- Keep the existing scientific-reference limitations explicit, including the
  current port's drift-variability discrepancy. Repairing scientific omissions
  is a separate versioned change, not part of this computational candidate.

## Fidelity experiment and acceptance

Compare three frozen arms: full FFT reference, current capped KL, guarded KL.
Use fresh seeds disjoint from the exploratory study. A preliminary 4,096-trial
screen per condition can detect gross discrepancies but cannot qualify adoption.
Freeze a manifest before running it: source, settings, seed lists, masks,
truths, grid/display mapping, numerical precision and all comparisons.

Cover both experimental difficulty conditions at their fixed parameters,
ordinary/quarter/zero-drift stress controls, boundary-height probes, and the
spectral/dispatch settings above. Define the primary qualification family before
looking at results; report additional stress probes as separately labeled checks.

For a qualifying run, use the canonical independent scorer and 20 fresh seeds
at 9,000 trials per experimental condition. Estimate runtime from the preliminary
screen before scheduling that larger run. If the fixed sample budget cannot
resolve the tolerances, report inconclusive; any expanded run needs a prospectively
versioned precision design rather than repeated testing until a pass.

Measure:

- Signed low-support winning-probability difference, using a fixed 10%-of-peak
  drift mask and 5%/20% sensitivity. Keep no-crossing trials separate from wins.
- Every A/B/C/D/T probability, conditional .1/.3/.5/.7/.9 RT quantile, region
  count and no-crossing fraction. Retain coordinates and first-crossing steps.
- Continuous spatial sliced Wasserstein-1 distance with 64 archived directions
  and the validated display-pixel mapping. Do not call grid cells display pixels.
- Achieved variance and mode counts, including fallback fraction over the frozen
  parameter suite. These diagnose fidelity; they do not replace behavioral checks.

Use the existing prospective engineering margins: 0.5 percentage points for
response probabilities (including the primary low-support diagnostic), 5 ms for
conditional RT quantiles, and 2 display pixels for spatial distance. These are
not scientifically established tolerances; lock their status before qualification.
Require simultaneous 95% uncertainty bounds entirely inside the margins.
Predeclare the comparison family size M and use Bonferroni-adjusted intervals:
exact discordance-probability bounds for paired binary differences, and paired
whole-trial bootstrap bounds for quantiles/spatial distances. Preserve seed
blocks and dependence between methods/conditions in resampling. Use upper
bounds for nonnegative distances. Validate bootstrap tail resolution for M;
if infeasible, revise the prospective procedure before the run, not afterward.
Sparse conditional quantiles or inadequate precision are inconclusive.

## Runtime experiment

Measure performance at three levels, with the same scientific inputs and matched
precision/trial budgets. Simulator-only speed is diagnostic; complete-objective
cost and time to independently verified fit quality determine the contribution.

| Level | Required measurements |
| --- | --- |
| Simulation | Synchronized wall time, trials per second, executed trial-steps, cold compilation, peak GPU memory and fallback frequency; report by sig and drift/boundary regime. |
| Complete objective | Wall time including basis selection/construction, cache behavior, transfers, simulation and loss; objective evaluations per second and cold versus warm costs. |
| End-to-end fitting | Time to independently verified acceptable quality, successful-fit fraction, evaluations to quality, total fitting/scoring cost, recovery and predictive quality at matched wall time. |

On the same RTX 4060 and frozen environment, use three warm-ups and 30 alternating,
synchronized complete-objective timing pairs across the fixed parameter-vector
suite. Include changing sig, cold/cache-miss basis construction, host transfers,
generator dispatch, loss evaluation and compilation. Report cold cost separately,
peak GPU memory, each parameter setting, total suite cost and fallback frequency.

Compare cost with both full FFT and current capped KL. The initial performance
target is at least 10% lower median complete-objective cost than full FFT, with
a paired 95% interval supporting an improvement and fidelity passing. In the
KL-eligible subset, separately test a proposed noninferiority limit of 10% extra
cost versus current capped KL. Report the fallback-region cost explicitly; do
not hide it in an average or claim faster runtime merely from fewer modes.

### End-to-end optimization experiment

After the short correctness/fidelity and timing screens, measure actual fitting
performance rather than extrapolating it from kernel throughput. Freeze the
optimizer, initial vectors, observations, scientific bounds, training/scoring
seeds, checkpoint cadence, trial budgets, stopping rules and precision. Change
only the simulator policy. Include full FFT and current capped KL controls;
label the capped control's known fidelity limitations in every comparison.

1. Run a matched-wall-time comparison, using a budget fixed from measured screen
   costs before launching fits. At identical elapsed checkpoints, independently
   score predictive fidelity and parameter recovery. Report objective evaluations
   completed and per-parameter errors, not just the training objective.
2. Run a matched-quality comparison: measure wall time to the first checkpoint
   independently meeting the frozen recovery and predictive criteria. Include
   initialization, compilation and required verification costs. Preserve capped
   or failed fits as censored runs and report successful-fit rates; never compute
   a speedup by comparing one successful run with an unsuccessful run's cap.
3. Begin with a clearly labeled single-case screen, then use the canonical
   three-truth, three-dataset, three-start design before claiming qualification.
   Generate recovery data with the FFT scientific comparator, not solely KL.
   Any unresolved discrepancy between the port and intended model remains a
   separate scientific qualification limitation.
4. Record convergence trajectories, objective-evaluation counts, invalid proposals,
   compilation/cache costs, policy-switch frequency, memory use and recovery.
   Investigate whether dispatch discontinuities increase evaluations enough to
   cancel per-evaluation savings. Do not switch optimizers or tune trial budgets
   in this same experiment.
5. A proposed end-to-end performance pass requires at least 10% lower median time
   to independently verified quality, a paired 95% interval supporting improvement,
   and preserved recovery/predictive acceptance. Predeclare the successful-fit-rate
   noninferiority margin and uncertainty method before qualification; lack of
   precision is inconclusive, not evidence of preserved success rate. Analyze
   the complete run set, including censored runs, rather than successful pairs alone.

Publish gains and losses against both controls by regime. Restoring fidelity may
cost time relative to capped KL at short correlation scales; quantify that cost
alongside savings against full FFT. Do not claim universal acceleration from an
average that hides slower settings. A faster objective with worse convergence,
recovery or predictive quality does not meet the end-to-end goal.

Use screen timings to estimate and explicitly schedule the larger fit experiment;
this plan amendment does not launch or authorize an unbounded compute run.

## Second candidate, only after isolating the first

If guarded KL passes fidelity, separately test existing mode-capacity buckets
(512, 1024, 1536, 2000), keeping the same retained modes and 99.9% target. This
may reduce padding at sig=15 (821 active modes can fit capacity 1,024), while
sig=10 still needs capacity 2,000. Preserve stable random coefficients; compare
extra compilations and cache costs over the whole objective suite.

Only if measurements justify it, study a larger KL capacity against FFT to locate
a hardware-specific break-even point. Select and freeze any dispatch policy
using calibration timings, then validate on a separate timing suite. Do not run
an adaptive timing competition inside every objective evaluation.

## Adoption and deliverables

Deliver a versioned report with before/after uncertainty plots, spatial maps,
region/RT tables, achieved retention and runtime/memory tables, raw observations,
source hashes and commands. Classify each gate as pass, fail or inconclusive.

After simulator fidelity and runtime gates pass, plan matched fitting and
cross-generator recovery: generate data with FFT, fit with the candidate, and
independently evaluate parameter-specific recovery and predictive quality. Check
whether dispatch boundaries disrupt optimization. Self-recovery with KL alone
is insufficient. Use the canonical multi-dataset/multi-start protocol before
claiming full scorecard qualification or time-to-quality improvement.

Do not change defaults merely because 99.9% variance is achieved. If behavioral
fidelity fails at that target, retain FFT as the control and prospectively test
stricter retention or more fallback; do not relax outcome tolerances to pass.
