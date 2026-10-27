# Model B: one-change BADS optimizer experiment

## Final disposition

Completed; **not adopted**. The equal-time screen missed recovery and flagged RT-quantile regression; improved G2, response proportions and spatial scores are preserved rather than summarized as universal prediction failure. The [final report](../results/2026-10-21-rtx4060-bads-screen/README.md) compares both laptop baselines, both BADS checkpoints and individual parameters. Historical launch notes below are retained for provenance. No further BADS tuning or replication is currently scheduled.

Status: **implemented; immediate launch authorized after stopping optional baseline stages**. On October 21, 2026, the user explicitly requested "Start bads", then "Lets stop it right now, and start bads" while the original-comparator objective benchmark was running. The original simulation benchmark and capped fit were already finished. Remaining objective timing and scoring were stopped and recorded as incomplete; the waiting BADS process was replaced by a new run using `--allow-stopped-baseline`. This explicit override requires the completed 30-pair simulation record and completed/capped fit and still refuses concurrent GPU compute. It does not change any candidate scientific settings or acceptance criteria. The 27-fit suite remains deferred.

Implementation: `scripts/run_model_b_bads_screen.py`, with analysis and budget controls in `scripts/model_b_bads_support.py`. PyBADS 1.5.1 and gpyreg 1.4.0 are installed in `/home/yash/ratcliff-rtx4060-bads/.venv`, preserving all pinned scientific dependency versions. Fifteen focused tests passed in that environment on CPU, including a real deterministic BADS toy fit. The immediate process performs GPU preflight before fitting; consult the live status for its outcome.

The active run's immutable runner copies, plan snapshot, dependency lock, source/input/package hashes, live status and subsequent samples are under `.benchmark-results/2026-10-21-rtx4060-bads-screen-started/`. The earlier waiting process is preserved under `.benchmark-results/2026-10-21-rtx4060-bads-screen/` with status `superseded_before_fitting`. The live report is [BADS screen](../results/2026-10-21-rtx4060-bads-screen/README.md). Check its status before asserting that fitting has started or results exist. Do not start a duplicate process.

## Decision and hypothesis

Replace Nelder-Mead with PyBADS for one archived-case fit of the **existing improved Model B simulator**. The hypothesis is that a different search strategy recovers the difficult parameters more accurately within the same elapsed fitting budget. The simulator's speed is not the proposed improvement.

The fixed-key objective is expensive, discontinuous and locally flat. The completed simplex fit spent approximately 25.48 minutes after its last strict loss improvement. BADS is a plausible alternative for this kind of derivative-free search, but neither its general benchmarks nor this diagnosis establishes a benefit for Model B.

The intervention is the BADS optimizer package, including its required internal scaling, initial design and stopping logic. It is not an isolated test of one BADS subroutine. Do not add a separately tuned parameter transformation, active-only fitting, restarts, an adaptive trial schedule, new random streams during training, or simulator changes.

## Frozen control and scientific contract

Follow [CLAUDE.md](../../CLAUDE.md), particularly its scientific preservation rule and canonical scorecard. Model B must still explain spatial decisions and RTs through the same accumulation and first-crossing process. All five response regions and both conditions remain in the objective. This experiment changes search behavior only; it cannot establish independent scientific fidelity of the port.

Use the completed **improved** RTX 4060 run as the optimizer control. The ongoing original/full-horizon run is contextual performance evidence, not this experiment's control.

| Control measurement | Recorded value |
| --- | --- |
| Frozen scientific source | `322bd1fb3893d394727f724947147b249eebb098` |
| Archived evidence commit | `cc2dbea7c8b75a742608787d7d6d5670a232d338` |
| Training duration | 5,677.777018686 seconds; 94.63 minutes |
| Initialization/warm-up | 15.915971603999878 seconds |
| Objective evaluations | 563 |
| Returned mean active relative parameter error | 4.586% |
| Returned worst active relative parameter error | 16.635%, `crsd` |
| Fresh-seed fitted median G2 | 71.623, keys 6001-6020 |
| Historical fitted median G2 | 80.302, keys 999, 5001, 5002 |
| Recovery screen | Failed: worst parameter exceeds 10% |

Read full-precision values from `.benchmark-results/2026-10-21-rtx4060-baseline/archived-case/`, not this rounded table. Preserve the existing [control report](../results/2026-10-21-rtx4060-baseline/README.md) and [baseline protocol](2026-10-21-rtx4060-baseline.md).

## Settings held fixed

| Item | Required setting |
| --- | --- |
| Hardware | Same RTX 4060 Laptop GPU through Ubuntu 24.04 WSL; serial GPU studies, external power |
| Runtime | Existing Python 3.12.3, JAX/JAXlib/CUDA plugin 0.10.2, NumPy 2.5.4 and SciPy 1.18.1; preserve all scientific dependency pins |
| Source | Verify every frozen source hash before and after the experiment |
| Data | Original archived `data-baseline.npz`; SHA-256 `841f8ba76a41d7a90575563bf408ac586c3811452fa37fd568cd7c14ee032cd2` |
| Starting vector | Copy the exact 13-element `x0` from the completed control JSON; do not regenerate from rounded values |
| Training randomness | Fixed JAX key 321 for every objective evaluation |
| Objective | Unmodified frozen `fofs_b_new`, including clamping, quantile bins, sparse-category rules and probability floors |
| Simulation | 9,000 trials per condition; K-L enabled; chunk 8; block 32; early stopping and compaction enabled; stable mode-indexed noise enabled |
| Other extensions | Bucketed basis, cumulative K-L, trial schedules and active-only optimization disabled |
| Basis cache | Two entries |
| Precision | Float32 fields, float64 RTs, JAX x64 enabled, default matrix-multiplication precision |
| Parameter interface | All 13 coordinates retained; the inert `sv` coordinate stays in the search and is excluded from the 12-parameter recovery screen |

Verify the data digest against the existing protocol and the file bytes before launch.

## Candidate configuration

1. Install PyBADS in a separate environment cloned from the frozen runtime. Use the documented 1.5 release family; record the exact resolved release, package hashes and dependency versions before any candidate evaluation. Freeze the complete resolved BADS options. If installation requires changing the scientific runtime, stop setup and resolve compatibility rather than silently changing the baseline environment.
2. Use one optimizer run with `random_seed=1701`, `uncertainty_handling=False` and `specify_target_noise=False`. Fixed simulator randomness makes repeat evaluations deterministic even though the objective is a finite-simulation approximation. Do not enable stochastic-objective handling merely because the underlying model is stochastic.
3. Pass the original raw 13-coordinate objective with no additional hard box constraints (`lower_bounds=None`, `upper_bounds=None`); the existing clamp continues to define effective parameters. Preserve its coupled restrictions on `st` and `crsd`. Log both raw and effective parameters.
4. Supply the required finite plausible bounds using a rule fixed before observing BADS results: for each positive archived starting coordinate, `plausible_lower = 0.5*x0` and `plausible_upper = 1.5*x0`. These are search scales, not hard scientific bounds. Verify positivity and ordering in preflight. Do not derive scales from the known truth, the simplex solution, or independent scores. This rule is a pilot configuration, not a claim that 90% of optima lie in that box.
5. Disable optional nonlinear scaling (`nonlinear_scaling=False`) to avoid an extra log-transform policy. BADS's required linear scaling based on its plausible bounds remains part of the optimizer intervention. Keep the documented search and termination defaults otherwise; do not force simplex tolerances onto a differently defined algorithm.
6. Use `max_fun_evals=6500` and `max_iter=2600` as explicit safety limits for 13 dimensions, alongside the stricter 8,100-second wall-time cap. Do not use previous simplex evaluations as BADS training data. Count all initial-design evaluations, repeat evaluations and internal search overhead in the candidate budget.
7. Treat nonfinite objective values or simulator failures as a recorded failed run. Do not silently insert a new penalty, change bounds, restart, or alter sparse-category behavior to rescue the candidate. Archive the offending raw/effective vector and error.

## Timing and checkpoint policy

- Perform a synchronized objective warm-up at the original starting vector, as in the control; record its duration separately. Keep basis-cache behavior consistent with the control.
- Start the training clock immediately before constructing the BADS search object, after common simulator initialization. Include constructor/setup work, initial design, CPU surrogate work, device transfers, simulation and result handling. Software installation time is a separately reported setup cost.
- Apply the same **8,100-second soft training cap**, checking before a new objective call, after each synchronized completed evaluation, and at supported optimizer callbacks. Preserve any overrun and unfinished state. An evaluation that finishes after a comparison cutoff is not available at that cutoff.
- Record every requested and completed evaluation with monotonic times, raw/effective parameters, fixed-key loss, validity, simulator-call count and cumulative training duration. Record internal cached reuses separately from actual simulator calls. Save GPU utilization, memory, temperature, power and clock throughout.
- Save candidates at 20, 40, 60 and 80 minutes; **5,677.777018686 seconds**; 110 minutes; and termination. Select the lowest training loss among evaluations completed by each cutoff, breaking exact ties by earliest completion. Apply the same selection rule to the archived simplex trace. Preserve each optimizer's native returned estimate separately, since tie policies can differ.
- The primary comparison is the matched **94.63-minute** checkpoint. If BADS terminates earlier, retain its available best candidate without restarting. The later 135-minute result is secondary and cannot rescue a failed primary result or be described as a same-cost improvement.
- Independent scoring happens after fitting and cannot influence continuation, candidate selection, settings or early stopping. Score only the primary and final candidate initially, deduplicating them if their effective parameter vectors match. Intermediate checkpoints remain diagnostic evidence, not additional opportunities to select a winner.

## Independent scoring and success criteria

Evaluate the primary candidate through the same frozen observable scorer as the control at **20 fresh-to-training keys, 6001 through 6020**, with 9,000 trials per condition. Reuse the hashed control and truth samples at those exact keys. These keys have already been used to describe the control; do not use the results to tune BADS. A later confirmatory experiment needs new untouched seeds as well as new datasets and starts.

Record all 12 active parameter errors, ten region-probability errors, 50 conditional-quantile errors, counts, spatial SW1 in each condition with the same 64 directions, and no-crossing fractions. Preserve missing/sparse quantiles as inconclusive. Compute recovery from clamped effective parameters. Parameter truth is diagnostic only and is never supplied to the optimizer.

For each seed compute five paired differences, candidate minus control (negative is better):

1. G2 against the fixed observations.
2. Maximum absolute region-probability error across the ten condition/region cells.
3. Maximum absolute conditional-RT-quantile error across the 50 cells.
4. Spatial SW1 against the saved truth sample in condition 1.
5. Spatial SW1 against the saved truth sample in condition 2.

Report mean paired differences and descriptive bootstrap intervals by resampling **whole matched seed pairs**, preserving both conditions and all cells together. Use 10,000 resamples with analysis RNG seed 1702. Use two-sided 99% intervals for each of these five outcomes (Bonferroni nominal family coverage 95%). Report raw values and medians as well. These intervals reflect simulator-seed variation for this dataset and selected fit, not optimizer repeatability or population-level recovery.

Classify the primary screen as follows:

| Classification | Predeclared interpretation |
| --- | --- |
| Promising for replication | Mean active relative error <=5%, worst active error <=10%; upper interval endpoint for paired G2 difference <0; upper endpoints for the other four summary differences <=0; all required quantiles estimable; no invalid-output or integrity failures |
| Recovery target missed | Either recovery threshold fails, regardless of a lower training loss or a better average prediction score |
| Predictive regression | An interval lies wholly above zero for any of the five outcomes; preserve the per-region evidence explaining it |
| Inconclusive | Recovery passes but prediction intervals overlap the decision boundary, necessary cells are sparse/missing, or quality/resource conditions prevent the declared comparison |
| Execution failure | Invalid evaluations, runtime incompatibility, interruption or other failure prevents a valid primary checkpoint; retain the attempt |

These classifications are evidence for a **pilot**, not automatic acceptance of the algorithm. Inspect all individual response cells and parameter errors even when aggregate criteria pass; flag any worsened cell rather than hiding it in an average. Report no-crossing mass separately. Do not retrospectively widen margins or add more seeds to turn an inconclusive screen into a pass. Changing the precision rule requires a separately versioned follow-up.

Score the primary and distinct final candidates with the historical full-horizon objective at keys **999, 5001 and 5002** as a separate diagnostic. Preserve its historical cutoff and results for traceability; do not substitute those three keys for the fresh-seed comparison.

The scorecard's absolute fitted-prediction acceptance threshold remains uncalibrated. This comparative pilot therefore makes **no claim of universal scientific accuracy, a jointly validated successful fit, or time to acceptable quality**. Its recovery thresholds and comparative guardrails are prospective engineering criteria. A scientific correction that activates `sv` is a different experiment.

## Implementation tasks when execution is requested

1. Verify the original-comparator baseline has finished; review its outcome and the improved control's final artifacts. Keep the GPU free of other studies.
2. Freeze a new experiment manifest containing all source/data/control/scorer hashes, the exact starting vector, BADS options and dependencies, seed lists, timings, candidate-selection rule and analysis rules above.
3. Add a standalone experimental runner under `scripts/` that calls the frozen objective. Do not edit the simulator, historical optimizer, public parameter interface, baseline environments or existing results.
4. Verify the wrapper against the original objective at identical vectors and keys; check that the raw/effective parameter logging preserves clamping. Use lightweight toy-function checks for callback timing, deadline behavior, failed evaluations and checkpoint selection before spending a full GPU fit.
5. Launch the single BADS run only after setup checks pass. Log partial progress atomically; never append a replacement fit to an interrupted trace. Measure all additional scoring time separately.
6. Produce `docs/results/<execution-date>-rtx4060-bads-screen/` with a concise comparison, machine-readable evidence, individual parameter/observable tables, loss-versus-time trace and complete cost accounting. Keep large samples and raw traces in a new ignored benchmark directory.
7. Report the primary classification before proposing anything else. If promising, plan replication across optimizer seeds, independent datasets, truths and starts. The existing 27-fit qualification remains a separate decision; one favorable fit cannot establish general superiority.

The candidate fit has a 135-minute ceiling, plus initialization, setup checks and scoring. Budget roughly three hours for the single-run screen, with extra time if setup exposes a problem. Use the same laptop and record thermal/power differences; elapsed-time comparisons across separate sessions are descriptive rather than a hardware-controlled replication.

## Sources

- [Canonical repository scorecard](../../CLAUDE.md).
- [Completed RTX 4060 improved baseline](../results/2026-10-21-rtx4060-baseline/README.md).
- [Frozen baseline and original-comparator protocol](2026-10-21-rtx4060-baseline.md).
- [PyBADS API](https://acerbilab.org/pybads/api/classes/bads.html): objective wrapper, bounds and deterministic/noisy configuration.
- [PyBADS options](https://acerbilab.org/pybads/api/options/bads_options.html): stopping settings and reproducible optimizer randomness.
- [PyBADS project and primary publications](https://github.com/acerbilab/pybads): methodological motivation; not evidence of Model B performance.
