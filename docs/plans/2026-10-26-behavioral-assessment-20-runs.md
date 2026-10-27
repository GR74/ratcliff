# Twenty-run behavioral assessment: FFT200 versus original FFT and KL

Status: **implemented; execution status is recorded separately in the live report**.
Requested October 26, 2026. This is a
new prospective behavioral study, not a continuation of the stopped timing
experiment and not a fitting study. Its numeric configuration is in
`2026-10-26-behavioral-assessment-20-runs.json` beside this document.

## Question and reference

At the same scientific parameters, how closely do FFT200 and the current
capped KL implementations reproduce the original FFT's response probabilities,
spatial responses and reaction-time distributions? Where do their discrepancies
occur, how large are they, and how precisely can we estimate them?

The reference is the original 199-by-319 full-spectrum FFT in frozen repaired
source 322bd1f, the JAX port of Ratcliff's model. This comparison does not newly
validate the entire port against Fortran or the paper. The existing inactive-sv
discrepancy is unchanged and must remain explicit. The FFT200 covariance builder
must match the hash of the successfully audited October 25 implementation.

Four arms:

1. Original FFT199: shared reference for the three contrasts.
2. FFT200: full-spectrum 200-by-320 embedding, same 100-by-160 decision grid.
3. Repaired KL: retained-variance target 0.99, maximum 2,000 modes, stable
   mode-indexed random draws, ordinary padding.
4. Bucketed KL: the same retained modes and repaired random indexing, with
   bucketed padding. Guarded KL and cumulative KL are disabled.

Hold blocking, first passage, parameter values, precision and normalization
fixed: chunk 8, block 32, x64 enabled, fp32 fields, default matmul precision,
early stopping and compaction enabled. Record actual mode counts and achieved
variance for both KL arms at each sig. Do not improve one arm during the study.

## What constitutes one run

One run is **one new independent seed block across every active method and
parameter case**, with 9,000 trials per experimental condition. It is not a
timing repetition of identical simulated paths.

Maximum 20 runs, yielding 180,000 trials per condition, case and method if all
runs complete. Both experimental conditions are always included. The maximum
four-arm, 17-case study generates 24,480,000 trials. The 4,096-trial pilot and
old fixed-seed timing replays are excluded from these new estimates.

Round indices are 0..19; method indices are FFT199=0, FFT200=1, repaired KL=2,
bucketed KL=3. Preserve the case order in the numeric JSON. The integer seed is
`2026102700 + 100000*method_index + 1000*case_index + round_index`, then use
the frozen `split_for_condition` function for condition keys. All streams are
fresh relative to the earlier pilot, qualification design and timing seed 321.
Method streams are independent; equal round labels do not create paired trials.
The common reference sample is reused across contrasts, so those contrasts are
dependent. The inference below does not require independence between contrasts.

## Main comparison and separately reported stress cases

The **main operating scope** is sig 5, 10 and 15 at ordinary drift, both
conditions: six method-comparison cells per candidate. Their numeric parameters
are copied unchanged from the prior pilot. No selection based on new outcomes.

Retain all 14 additional cases from the prior pilot as **secondary diagnostics**:
quarter-strength and zero drift, boundaries cr=5 and cr=20 with crsd=2 at each
sig, and the archived start and fitted parameter vectors. Thus the new study
still reports every one of the original 17 cases. The core/secondary separation
is a prospective change in inferential scope, not a retroactive pass for the
previous study. Passing the core does not qualify the stress cases or the full
parameter domain.

Compute every case in each complete round while its method remains active.
For ordinary and quarter-strength drift, the low-support mask is defined from
the deterministic drift profile before simulated evidence demeaning. A uniform
scaling of drift leaves the mask unchanged. For zero drift, reuse the archived
ordinary-drift geographic mask and label it a geographic control; do not call
all locations low-support or interpret its count as an evidence classification.

## Saved evidence and canonical endpoints

Save each trial's round, seed, condition, case, method, RT, category, crossing
flag, first-crossing step, winning position and display coordinates. Non-crossing
coordinates are missing, never genuine location zero. Save masks, drift maps,
region geometry, 64 projection directions and generated trial-step counts.
Keep raw trial identifiers so every summary can be rebuilt without resimulation.

At each reporting look, pool trials within each fixed method/case/condition;
do not average seed-level quantiles or pool different scientific parameters.
Report candidate-minus-reference signed differences, absolute differences and
intervals in physical units. Main endpoints per cell:

| Endpoint | Number | Proposed engineering tolerance |
| --- | ---: | --- |
| A/B/C/D/T response probabilities | 5 | +/-0.5 percentage points |
| Genuine low-support win probability, 10% cutoff | 1 | +/-0.5 percentage points |
| No-crossing probability | 1 | +/-0.5 percentage points |
| Conditional RT quantiles .1/.3/.5/.7/.9 in each region | 25 | +/-5 ms |
| 64-direction spatial sliced Wasserstein-1 | 1 | <=2 display pixels |

All trial counts enter probability denominators; non-crossings are excluded
from low-support win numerators. T retains the production fallback convention,
with non-crossing mass separately shown. RT quantiles and spatial distances
condition on genuine crossings. Report counts for every region and every
missing/unbounded quantile; sparse categories never silently pass.

Also report 5% and 20% low-support mask sensitivities, overall RT distributions,
joint region/RT histograms and the secondary stress-case endpoints. These are
diagnostic additions, not extra unregistered early-stop tests. Pointwise
intervals, if shown in supplements, are labeled descriptive and cannot support
family-wide or sequential claims.

The tolerances are the existing proposed engineering margins, not thresholds
scientifically established by Ratcliff's paper. This assessment proceeds with
its bounded 20-run budget even if it cannot certify every margin; it must still
report how close the observed behaviors are and which effects remain unresolved.
Do not repeat the earlier policy of aborting all behavioral sampling merely
because a pilot estimates that full-domain qualification needs millions of trials.

## Valid early stopping, fixed before collecting new trials

Make inferential decisions only after **5, 10, 15 and 20 complete runs**.
Progress between those looks is operational only. Allocate the family error
budget alpha=0.05 across the four looks with weights 0.05, 0.10, 0.20 and 0.65.
This gives per-look family budgets 0.0025, 0.005, 0.01 and 0.0325.

There are M = 3 candidates x 3 sig values x 2 conditions x 33 main endpoints
= **594 registered endpoints**. For endpoint j at look l use
`alpha_jl = 0.05 * weight_l / 594`.

- Binary differences: combine separate exact Clopper-Pearson intervals for the
  two proportions, each allocated alpha_jl/2. Subtract interval endpoints in
  the candidate-minus-reference direction. No paired-binomial formula.
- Conditional RT differences: use separate binomial order-statistic quantile
  intervals, each allocated alpha_jl/2, and subtract their endpoints. Extreme
  order indices produce unbounded intervals, not omitted endpoints.
- Spatial SW1: split alpha_jl equally between its upper and lower bounds.
  With D=2*sqrt(159^2+99^2) pixels and observed crossing counts nA,
  nB, use the upper bound
  `observed_SW1 + D*sqrt(0.5*(1/nA+1/nB)*log(2/alpha_jl))`.
  If either crossing count is zero, the bound is unbounded. It follows from
  bounded differences plus convexity as documented in the previous plan.
  For a material spatial rejection, use the registered lower bound from
  projected mean differences: fix all 64 directions, bound each method's mean
  projected coordinate by a two-sided Hoeffding interval at alpha_jl/(4*64),
  and average the lower
  bounds on absolute mean differences. Each mean difference is bounded above
  by its directional W1, hence this average lower-bounds SW1. This lower bound
  is conservative and may be zero even when empirical SW1 is appreciable.

Union bounds over the endpoints and scheduled looks give at least 95% joint
coverage for these inferential statements under the independent-trial model,
despite reuse of reference samples and overlap of cumulative looks. This is a
finite-look alpha-spending construction, not unadjusted repeated 95% intervals
and not a claim that a general confidence-sequence library has been implemented.
The rationale for protecting inference under optional stopping is discussed by
[Howard et al., 2021](https://arxiv.org/abs/1810.08240); the concrete four-look
allocation and conservative bounds above define this study's implementation.

At a look, each candidate's six core cells receives a status:

- **Equivalent within scope:** every registered endpoint's adjusted interval
  lies within its tolerance, including the spatial upper bound <=2 pixels.
- **Material mismatch:** at least one adjusted signed interval lies entirely
  beyond a tolerance, or the registered spatial lower bound exceeds 2 pixels.
- **Inconclusive:** otherwise. An interval containing zero is not a pass.

A candidate can stop collecting new trials early when **all six core cells are
decisive** (each equivalent or materially mismatched). This preserves useful
comparisons across sig, rather than stopping KL as soon as its already-known
sig=5 discrepancy reappears. Label the whole candidate equivalent only if all
six cells passed; any failed cell establishes a scoped mismatch. Keep the full
case-by-case pattern, including secondary cases that remain uncertain.

Other candidates continue up to 20 rounds. Stop the whole study when all three
candidates meet that decision rule or the 20-round/resource limit is reached.
Freeze a stopped candidate's estimates and reference prefix at its stopping
look; do not later attach an unadjusted larger reference sample. Reference
sampling continues as required by active candidates. Do not recycle alpha from
stopped arms. Interrupted partial rounds are saved but excluded from registered
look decisions until complete.

At 20 rounds, unresolved cells are **inconclusive**, with effect estimates and
intervals still reported. No automatic sample extension, relaxed margin or
discarding of rare RT categories. Early certainty is possible, not promised.

## Comparing FFT200 with KL

Primary contrasts remain each candidate versus FFT199. Show all three methods'
errors in the same units and reference coordinate system; include counts and
the sample budget of each estimate. Use the common collected-round prefix for
side-by-side diagnostic comparisons when arms stop at different looks.

For signed endpoints, a formal claim that FFT200 is closer than a KL arm needs
the adjusted intervals to imply that ordering: the largest absolute FFT200
error allowed by its interval must be smaller than the smallest absolute KL
error allowed by its interval. A difference in pass/fail labels alone is not a
test that one error is smaller. Spatial closeness requires the analogous valid
upper-versus-lower bound ordering; otherwise report point estimates as descriptive.
Never treat matching a few region counts as proof of matching spatial or RT laws.

## Workload, implementation requirements and deliverable

Run serially on the RTX 4060 in the pinned WSL environment; reject overlapping
GPU benchmark workloads. Preserve source and data hashes, exact numeric cases,
seed map, thresholds, geometry, dependency versions and all analysis code in a
new output directory. Verify observer parity before measurement. Test the
scheduled-look allocations, interval arithmetic, conditional-count handling,
spatial lower bound, early-stop decisions and interrupted-run resume behavior.

The previous 68 pilot batches took 288.5 sampling seconds at 4,096 trials per
condition. Linear extrapolation gives roughly **3.5 hours of sampling** for the
complete 20-round study, excluding compilation, summaries and disk writes.
Plan for **4-6 hours**, with a six-hour soft wall cap checked between calls.
This is an estimate, not an allocated cloud cost or completion guarantee.
Early stopping can shorten it. Record any capped result as incomplete.

Record synchronized simulator duration and work as secondary diagnostics, then
transfer/save outputs outside that timed section. Do not repeat the standalone
30-round speed benchmark. No optimizer runs, parameter recovery, cloud rental
or default changes are part of this planned assessment.

Proposed raw directory:
`.benchmark-results/2026-10-26-behavioral-assessment-20-runs/`.
Proposed report directory:
`docs/results/2026-10-26-behavioral-assessment-20-runs/`.

After each scheduled look publish:

1. A six-core-cell table per candidate with endpoint-level differences,
   adjusted uncertainty, sample counts and scoped pass/fail/inconclusive labels.
2. Low-support and named-region difference plots in percentage points.
3. Spatial response maps, difference maps, 64-direction SW1 and its bounds.
4. Conditional RT quantile plots and full RT diagnostic distributions.
5. Separate stress-case panels, including zero-drift geographic controls.
6. A cumulative-precision plot versus independent run count and a complete
   early-stop log. The final report must clearly distinguish numerical covariance
   agreement, observed behavioral closeness and formal scoped equivalence.

The next decision after this study is whether to run exploratory recovery or
target an unresolved behavior with a separately designed experiment. Neither
action is silently launched by this plan.

## Implementation and operation

The user subsequently requested implementation and launch. Run
`scripts/start_behavior_assessment.ps1` on Windows. The launcher temporarily
prevents automatic sleep while its WSL child runs and releases that request
when it exits. Use `-Resume` only to explicitly resume an interrupted study;
source, protocol, sample sizes and saved-batch checksums must still match.

`scripts/run_behavior_assessment.py` saves each independent batch atomically,
records source/dependency hashes, verifies the audited FFT200 builder and checks
observer parity against uninstrumented and full-horizon outputs before sampling.
Registered inference is in `scripts/behavior_assessment_stats.py`; scientific
figures are in `scripts/behavior_assessment_plots.py`. Completed-look JSON is
committed only after its CSV and figures are ready, so interrupted analysis can
be rebuilt from saved samples. No optimizer is called.

Focused preflight checks cover statistical allocation, seeds, spatial bounds,
stopping, missing categories, raw-output validation, corruption detection,
partial-file preservation, round accounting and plot rendering. The run's
`validation.json` records the completed checks and exact command.
