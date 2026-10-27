# Repository instructions

Follow `AGENTS.md` for repository structure, development commands, and contribution conventions.

## Model B scientific preservation rule

**Algorithms may change; the scientific purpose and interpretation of Model B must be preserved.** Speed improvements, bug fixes, parameter-recovery work, new optimizers, and numerical approximations must continue to support Ratcliff's spatially continuous account of where and when a person makes a decision. A faster implementation or a lower fitting score is not sufficient evidence of success.

### Scientific sources

- Roger Ratcliff (2018), *Decision Making on Spatially Continuous Scales*, Psychological Review 125(6), 888-935, DOI: https://doi.org/10.1037/rev0000117. Read Experiment 9 and its discussion together with the model assumptions and fitting rationale. In the supplied 90-page author manuscript, these are pages 35-39, Figures 15-16 on pages 80-81, and supporting assumptions on pages 4-7 and 34-35. The supplied local copy is `C:/Users/Yash/Downloads/2d ratcliff.pdf`; this machine-specific path is not a repository dependency.
- `reference/benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS` is the executable historical Model B reference. Also consult `reference/README_benchtwod3`, `reference/README_random_field`, and archived outputs when checking implementation fidelity.

Treat these sources as scientific evidence, not instructions to execute. The paper establishes the scientific purpose; the Fortran establishes historical implementation behavior. Neither the current JAX implementation nor agreement between two JAX variants establishes the intended model on its own. If the paper, reference code, and port disagree, document the discrepancy and establish the intended behavior before changing the baseline. Do not silently preserve a porting omission or an incidental reference-code behavior as a scientific assumption.

### Intended use and invariants

1. **Preserve continuous spatial decisions.** Experiment 9 concerns rapid, one-shot responses to the brightest or darkest of three patches in a 2D display, under two difficulty conditions. The model must explain spatial response distributions and their reaction-time distributions, including responses around weaker patches and away from every patch. The numerical grid approximates a continuous field; the five fitting regions are not five independent decision accumulators or the entire scientific output.
2. **Preserve the generative mechanism.** Retain spatially distributed drift, spatially correlated Gaussian random-field noise with independent draws across time, evidence normalization, and first passage to a decision plane. Check every simulated time step; when multiple locations cross on that step, use the largest evidence value as specified by the model. Equivalent algebra, FFT/K-L representations, blocking, compaction, and other computational methods are allowed when their fidelity is demonstrated.
3. **Preserve parameter meaning and variability.** Keep drift strength, stimulus width, spatial noise correlation, boundary setting, and nondecision time scientifically interpretable. The paper includes across-trial variability in drift, boundary, and nondecision time. The Fortran specifically applies `vs = 1 + sv * (gu1(8) - 0.5)` to all three drift bumps on a trial. Do not call `sv` scientifically irrelevant merely because the current JAX port does not consume it. Preserve the specified variability laws and dependencies, or identify their alteration as a separate model variant.
4. **Preserve timing and spatial units.** Experiment 9's decision RT is stimulus onset to finger lift; subsequent movement to the response location is measured separately. Do not silently fit movement duration as decision RT. Document conversions between display pixels, simulation cells, response alignment, and time units. The historical simulation uses a 100-by-160 field and 10 ms steps; changing either discretization requires consistent drift/noise/geometry scaling and convergence checks. Reconcile region cutoffs and coordinate conventions across sources rather than assuming their numerical values are interchangeable.
5. **Preserve the fitting questions.** Evaluate both difficulty conditions jointly with their condition-specific drift strengths and shared parameters, with individual-subject fitting as the published baseline. Retain the A/B split around the strongest peak, C/D regions around weaker peaks, and T background/tail region in baseline comparisons. The paper motivates A/B separation to distinguish peak height from width. Assess region proportions and conditional RT quantiles at .1, .3, .5, .7, and .9, as well as spatial distributions. Optimizers, likelihood estimators, or objectives may change, but their connection to these observables and parameter interpretations must remain explicit and validated.

### Required evidence for changes

- State which scientific behavior is preserved, which computational method changes, and whether the proposal is an exact reformulation, numerical approximation, fitting-method change, or scientific model extension.
- Validate against an independent reference where possible, not only a control that shares the same implementation. Use paired-path checks when randomness permits; otherwise compare distributions with sampling uncertainty and declared tolerances.
- Cover the supported parameter range, rare/tail responses, leading and late RT quantiles, parameter recovery across datasets and starts, and numerical approximation error. Retained noise variance alone does not establish fidelity of spatial maxima or first-passage behavior. A single near-truth synthetic fit or mean parameter error does not establish recovery of every parameter.
- Make no-crossing trials, finite-horizon behavior, sparse categories, and invalid parameters explicit. Do not silently turn timeouts into genuine background responses, suppress tails, merge response regions, or restrict scientific parameter ranges merely to improve speed or scores.
- Report complete fitting cost alongside independent predictive quality and parameter-specific recovery. Preserve historical controls and disclose score changes; calibrate new acceptance criteria prospectively rather than relaxing them after observing failures.

### Experiment 9 limitations and research extensions

The paper reports excess predicted responses far from stimuli, misses in the leading RT edge, and some late-RT mismatches. These are scientific targets to investigate, not artifacts to hide through clipping, omitted categories, or revised scoring. It proposes stimulus-location-dependent noise and more controlled stimulus configurations as future research directions; these are hypotheses, not established baseline assumptions.

Such extensions may be explored as explicitly identified, separately validated model variants that retain the continuous 2D decision-making use case. Do not silently replace Model B with a discrete-choice race, a circular/hyperspherical diffusion model, or a predictor that loses the original parameter interpretation as part of a performance fix. Preserve a reproducible baseline and distinguish improved computation from changed scientific explanations.

## Canonical Model B comparison scorecard

Use this scorecard before and after each individual change. The primary outcomes are scientific predictive fidelity, parameter recovery, and elapsed time to acceptable quality. Simulator throughput is supporting evidence. Keep two references: the frozen implementation for engineering comparisons, and an independently validated implementation of the intended scientific model for fidelity checks. A justified scientific correction may change an incorrect baseline's output; document and version that correction separately.

### Current laptop baseline

The local RTX 4060 Laptop GPU study is documented in `docs/plans/2026-10-21-rtx4060-baseline.md`, with its updating report and compact evidence under `docs/results/2026-10-21-rtx4060-baseline/`. Check the recorded completion status before treating any measurement as final. Use measurements from this laptop for subsequent local speed comparisons; the RTX 3060 results below remain historical evidence. The user selected the archived-case screen first, with its measured runtime to be reviewed before the 27-fit qualification suite. The study freezes repaired source `322bd1f` and the original synthetic observations. Do not treat this single-case screen as full scorecard qualification.

After the improved screen completed, the user authorized the original full-horizon JAX comparator's simulation and fitting on the same RTX 4060. The follow-up protocol is in the same study plan; its updating report is `docs/results/2026-10-21-rtx4060-baseline/legacy-comparison.md`. Both variants retain the archived noise repair and 135-minute fitting cap. This is measurement of existing implementations, not implementation of new improvements. The 27-fit suite remains deferred. Do not substitute the blocked correctness control for the original performance comparator or claim a completed-fit speedup when a fit hits its cap.

The next single-change experiment is documented in `docs/plans/2026-10-21-model-b-bads-single-change.md`: replace simplex with BADS while retaining the frozen improved simulator and fixed training budget. The user requested "Start bads", then explicitly stopped the remaining original-comparator objective timing/scoring to start BADS immediately. The original simulation and capped fit are preserved; optional unfinished measurements must not be called complete. The immediate BADS run uses `.benchmark-results/2026-10-21-rtx4060-bads-screen-started/status.json`; the earlier waiting process is superseded. Check that status or `docs/results/2026-10-21-rtx4060-bads-screen/README.md` before claiming fitting has started or launching another process. Its primary endpoint is recovery and independent prediction at the control's exact 94.63-minute training duration; retain the original scorecard and report inconclusive results rather than changing criteria after observing them.

### Completed BADS screen and next candidate

The RTX 4060 BADS screen completed and is **not adopted**. At the matched 94.63-minute budget it returned 19.535% mean / 29.248% worst active-parameter error versus simplex's 4.586% / 16.635%. Independent median G2 improved (58.550 versus 71.623), as did region-probability and spatial summaries, but mean seedwise maximum RT-quantile error increased by 0.840 ms (paired descriptive 99% interval +0.525 to +1.215 ms). The 135-minute result did not rescue recovery (19.550% mean error). See `docs/results/2026-10-21-rtx4060-bads-screen/README.md` for the full comparison and individual parameters. Keep the frozen improved simplex as the engineering baseline; it is not scientifically qualified merely because BADS performed worse on recovery. Do not claim that BADS universally fails or that fixed-stream overfitting fully explains these results.

The next isolated laptop candidate is **K-L bucketed padding only**, described in `docs/plans/2026-10-22-model-b-padding-screen.md`. It enables the existing `kl_bucketed` option with stable mode-indexed noise and leaves retained modes, truncation, first-passage rules and simplex unchanged. Run a bounded fidelity/performance screen before considering another full fit. No BADS, optimizer tuning or additional model changes are part of that candidate.

Its runner is already queued behind an unrelated GPU workload. Check `.benchmark-results/2026-10-22-rtx4060-padding-screen-queued/status.json` or the report under `docs/results/2026-10-22-rtx4060-padding-screen/` before starting another process. Fifteen frozen GPU regression checks passed. The screen has a 20-minute measurement budget after GPU release and does not automatically start a full fit.

### Frozen historical comparison

The repaired implementation is commit `322bd1fb3893d394727f724947147b249eebb098`; its recorded results are at `cc2dbea7c8b75a742608787d7d6d5670a232d338`, under `docs/results/2026-10-15-stable-kl-9000-rtx3060/`. These commits are available locally but are not the current `main` implementation. Inspect them with `git show` or use an isolated checkout; do not substitute current `main` for the repaired baseline. The results directory contains the report, evidence, and a raw archive of source, observations, traces, scores, and environment records.

Historical hardware was an RTX 3060 Laptop GPU with 6 GB VRAM through WSL. The repaired early-stop run used 9,000 trials per condition, chunk size 8 and block size 32. These numbers describe one synthetic case, not a representative benchmark distribution:

| Historical measurement | Repaired early-stop baseline |
| --- | --- |
| Warmed simulation, both conditions | 9.271 s; one timed call after two warm-ups |
| Original full-horizon simulation pilot | 93.893 s; same limitation |
| Training time | 93.68 min; excludes warm-up and independent scoring |
| Objective evaluations | 563 |
| Starting mean absolute relative active-parameter error | 8.688% |
| Returned mean absolute relative active-parameter error | 4.586% |
| Worst returned active-parameter error | 16.635%, `crsd` |
| Returned independent median historical G2 | 80.302 |
| Truth independent median historical G2 | 35.760 |
| Historical joint prediction/recovery gate | Failed |
| Paired blocked-control choices and RTs | 18,000 / 18,000 exactly equal |
| Training after last strict loss improvement | Approximately 25.3 min |

Historical independent scoring used keys 999, 5001 and 5002. Its G2 cutoff of 40.336 is specific to that provisional experiment and is not a universal accuracy criterion. Agreement with the blocked control does not independently validate the scientific model. Historical recovery excluded inactive `sv`; a corrected implementation that activates it must include it and establish a separately identified baseline. Do not claim a time-to-quality speedup from these runs: the joint quality target was not verified.

### Required measurements

Report baseline, candidate, paired difference, uncertainty, and pass/fail/inconclusive status where applicable. Archive per-condition, per-region, per-parameter, and per-run values alongside aggregates.

| Outcome | Canonical measurement |
| --- | --- |
| Response-region fidelity | Absolute probability differences for A, B, C, D and T in each of the two conditions, expressed in percentage points; report all ten values, mean and maximum. |
| Low-support winning-probability fidelity | At identical parameters, report reference and candidate probabilities of a genuine first-passage win in a fixed low-drift mask, plus signed candidate-minus-reference difference in percentage points and uncertainty. Define the mask and cutoff prospectively; use the protocol below. This supplements the named response regions and continuous spatial metrics. |
| Conditional RT fidelity | Absolute errors in the .1, .3, .5, .7 and .9 RT quantiles for each condition and region, in milliseconds; report every estimable value, mean and maximum, and category counts. |
| Continuous spatial fidelity | Sliced Wasserstein-1 distance using 64 fixed, archived projection directions and the same coordinate mapping, in display pixels, separately by condition. Save response coordinates; region labels alone cannot establish spatial fidelity. |
| Parameter recovery | For each active parameter, signed error and absolute error for each fit; across independent datasets report signed bias and RMSE in parameter units. Also retain mean absolute relative error and worst absolute relative error. Predeclare absolute tolerances for parameters whose truth is zero or near zero. |
| Successful-fit rate | Number meeting both the recovery and independent prediction criteria divided by all attempted fits. Retain failures and capped runs in the denominator. Report uncertainty and results by truth and starting-point regime. |
| Time to quality | Elapsed time to the first independently verified qualifying checkpoint, including work needed to reach and verify it. Record checkpoint cadence. Also report training, initialization/compilation and scoring costs separately, and total run cost. Runs without a verified hit are censored, not successful. |
| Complete objective cost | Median wall time and a 95% confidence interval over a fixed, archived parameter-vector suite, including changing `sig`, basis construction, cache misses, transfers, simulation and loss evaluation. Also report cold-start cost separately. |
| Resource use and diagnostics | Peak GPU memory, compilation time, objective-evaluation count, invalid-evaluation count, no-crossing fraction by condition, and elapsed training time after the last strict improvement. Record retained-mode count and cap hits for approximation checks. |

Distinguish simulator fidelity at identical parameters from fitted predictions against observations. Use the same independent observable scorer for every optimizer, regardless of its training loss. Calibrate fitted-prediction acceptance against independent truth-generated data and simulator replicates with matched sample sizes; finite observed data are noisy. Do not apply same-parameter simulator-equivalence tolerances directly as a requirement to fit every observed bin perfectly. Record score variation across simulator seeds and trial budgets. Historical G2 may be retained for traceability but must not be the only quality measure.

### Low-support win diagnostic for simulator optimizations

Use **low-support winning-probability difference** as a canonical supplementary fidelity endpoint when changing the noise generator, numerical precision, time stepping, stopping logic, or other simulator computations. It measures agreement with the reference, not classification accuracy or whether a response is psychologically correct. Preserve parameter recovery as a separate endpoint; good self-recovery does not establish agreement with the reference simulator.

- Hold scientific parameters fixed between reference and candidate. Before inspecting outcomes, define a spatial mask from the reference's deterministic drift surface: the default diagnostic is drift at most 10% of its spatial maximum, before spatial demeaning. Record the drift surface, geometry, mask area, and whether the mask is condition-specific. Keep each mask identical across methods. Report 5% and 20% cutoff sensitivity; do not select the most favorable cutoff retrospectively. For zero-drift controls, explicitly use a predefined nonzero-drift reference mask as a geographic control, or label every location low drift; never conflate those definitions.
- Define `p_low = count(crossed_threshold AND winning_location_in_mask) / all_attempted_valid_trials`. Report the two probabilities, counts and `delta_low_pp = 100 * (p_low_candidate - p_low_reference)`. Keep no-crossing trials in the denominator but out of the win numerator, and report their fraction separately. Invalid simulations must be reported as failures rather than silently removed. Preserve winning coordinates, crossing indicators, crossing steps, RTs and named response categories so the endpoint can be audited.
- Report signed differences to distinguish excess from suppressed low-support wins; either can indicate a changed model. Include uncertainty appropriate to paired or independent sampling. Shared seeds alone do not establish paired paths across different generators: document the coupling and confirm its marginal distributions and temporal independence. For distributional approximations, individual-trial winner changes are not themselves failures if the relevant distributions agree.
- Evaluate this endpoint alongside A/B/C/D/T probabilities, conditional RT quantiles, continuous spatial fidelity, censoring, and achieved noise variance/mode counts. A low-support mask cannot reveal redistribution within the mask or among high-drift regions. Mean RT alone is insufficient. Include short-correlation settings where a mode cap binds, ordinary/weak/zero-drift controls, and the intended parameter range. Do not change scientific parameters merely to make an approximation pass.
- Predeclare an equivalence margin, confidence procedure, multiplicity correction and sampling-precision rule before a qualifying comparison. Apply the scorecard's simultaneous-bound requirement; an interval overlapping zero is not evidence of equivalence. The existing 0.5-percentage-point probability screen is a prospective engineering candidate, not an investigator-approved scientific guarantee. The exploratory report's shaded +/-1-point band is illustrative and does not replace that screen. Without a locked margin or adequate precision, mark equivalence inconclusive.
- The reproducible initial screen is `docs/results/2026-10-25-kl-winner-fidelity/README.md`, with `report.html`, raw observations, metrics and metadata beside it. Its runners are `scripts/check_kl_winner_fidelity.py` and `scripts/report_kl_winner_fidelity.py`; analysis checks are in `model_b/tests/test_winner_fidelity_diagnostics.py`. It used 1,024 paired trials per cell and pointwise intervals, not the full scorecard's qualification design. It compares the current JAX FFT law with truncated KL, not an independently established Fortran/paper baseline, and does not establish full scientific fidelity or parameter recovery. Reuse its instrumentation with a frozen prospective protocol rather than treating its exploratory results as canonical acceptance thresholds.

### Prospective screening tolerances

These are proposed engineering screening tolerances, not accuracy guarantees established by the paper. Lock them and the prediction-calibration procedure before running comparisons. Any revision requires a documented rationale and a new comparison version, never a retrospective relaxation to pass a candidate.

- Exact reformulations with identical arithmetic and common random inputs: 100% agreement in first-crossing step, winning location and response category. If floating-point ordering changes, use the distributional protocol and explain why exact agreement is not expected.
- Distributional simulator fidelity: response probabilities within 0.5 percentage points, conditional RT quantiles within 5 ms, and spatial sliced Wasserstein distance at most 2 display pixels.
- Synthetic recovery screening: mean absolute relative active-parameter error at most 5%, with no active parameter above 10%; substitute predeclared absolute tolerances where relative errors are ill-defined. Report the fraction of fits meeting this screen rather than only a pooled average.
- Worthwhile performance improvement: at least 10% lower median complete-objective cost or time to quality, with a paired 95% confidence interval supporting an improvement. Quality must also pass; report successful-fit rate alongside time to quality. This is an acceptance target, not a promised speedup.
- Zero silently accepted NaN/Inf results or invalid parameter outputs. Make invalid proposals and the handling of boundary proposals visible.

For distributional equivalence, require simultaneous uncertainty bounds to lie inside the declared tolerances (use an upper bound for nonnegative distances). Predeclare the confidence procedure, multiplicity handling and precision rule. Insufficient rare-category samples mean inconclusive, not pass; retain counts and unresolved quantiles. No-crossing trials must remain identifiable rather than being silently scored as ordinary T responses.

### Before/after protocol

1. Freeze source hashes, environment and dependency versions, hardware/driver, precision, data, truths, starting vectors, seed lists, simulator trial budgets, optimizer budgets and stopping rules. Save a machine-readable manifest with every run. Benchmark old and new source on the same hardware; the historical RTX 3060 timings are descriptive unless reproduced under comparable conditions.
2. Establish a fresh baseline before algorithm changes. For timing, use three warm-ups and 30 synchronized timed calls, alternate baseline/candidate order and retain raw timings. Separate cold initialization from warmed work; include a parameter-vector suite that exposes basis/cache costs rather than only repeating one cached width.
3. Screen on the archived synthetic case. Qualification then uses at least three truth settings, three independently generated datasets per truth, and three starts per dataset: 27 fits per variant. Include starts farther from truth and difficult parameter regimes. Freeze the actual settings in the study manifest before launching; this is a screening design, not proof over the entire supported parameter space.
4. Independently score with 20 fresh simulator seeds at 9,000 trials per condition, using matched seed sets for candidate/reference comparisons and streams disjoint from optimization. Expand sampling only under a predeclared precision rule. Preserve the historical three-key scores separately. Account for all independent scoring time.
5. Compare frozen baseline against baseline plus one change. After acceptance, preserve both the original and cumulative baselines. Keep trial budgets fixed for an algorithm comparison; study budget changes separately. For optimizers, report both equal-wall-time and matched-quality comparisons where feasible.
6. Record all failures, caps and interrupted runs. Do not drop unsuccessful fits when summarizing recovery or speed. Re-score saved checkpoints independently; parameter truth may diagnose recovery but must not guide the optimizer.
7. Before claiming full scorecard coverage, add spatial-coordinate capture, repeated synchronized timing, a common independent scorer and the expanded dataset/start/seed suite. Archive truth and fitted-baseline predictions before making changes. Existing archived samples contain only RTs and categories, and the existing three-key scoring harness does not implement this entire protocol.

This protocol specifies the required evidence; it does not authorize long GPU jobs, cloud rental, or reuse of an expired study's compute allowance. Lightweight checks may establish readiness first.
