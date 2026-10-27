# FFT-reference recovery and optimization comparison

Status: implementation authorized October 26, 2026. The isolated reference and
runner are implemented; consult `docs/results/2026-10-27-fft-reference-recovery/`
for actual run status. This protocol uses a corrected **JAX FFT199** generator,
not execution of Ratcliff's original Fortran. Direct executable qualification
remains pending. The user's fidelity-first decision prohibits fallback to the
uncorrected frozen port, even if completing the work takes longer.

## Objective and limits of the answer

Choose faster computation while preserving reference-model behavior and
parameter interpretation. Improving the original model's recovery is optional,
not the purpose of the optimization. Separate three questions:

1. Absolute synthetic recovery: how far is each estimate from generating truth?
2. Reference agreement: how far are its estimates and predictions from an
   original-FFT fit on the identical dataset?
3. Cost: how much total compute is needed for those results, including failures,
   warmup, optimizer work and independent verification?

The previous FFT200/KL results are valid measurements on their archived
KL-generated dataset, but do not establish fidelity to original-FFT recovery.
Neither equal data nor synthetic data is inherently unfair; the generator must
match the scientific reference the experiment is intended to preserve. Do not
use those earlier fits as the controls for the new FFT-generated observations.

Three methods on one dataset provide a concrete case study, not population-wide
recovery bias, success rates, or a universal best method. An inconclusive result
is a possible valid outcome. A deadline cannot guarantee statistical certainty.

## Reference implementation and whether Fortran is necessary

Actual Fortran execution is not required to isolate FFT199 versus FFT200 versus
KL on one shared JAX model. FFT199 here means the original 199-by-319 embedding
and full spectrum, on the unchanged 100-by-160 decision grid. It is a reference
noise method, not automatically the complete original Ratcliff implementation.

Direct original-code evidence is needed before claiming end-to-end equivalence
to Ratcliff's fitting implementation. An independently validated port can later
stand in for that reference. Merely selecting FFT does not validate a port.
Fortran can run on CPU/MPI; it need not be ported to the 4060 to serve as the
scientific reference. Historical cluster timings are not same-hardware speedups.

Known discrepancies in the currently frozen port include inactive drift
variability sv. Commit 7c0543620711a7b56b0aa93588d0f044dbb29b7c, on
origin/feat/pass3-separable-cmaes, contains opt-in corrections for sv, the
Fortran binned/interpolated RT CDF and non-crossing handling. Its 'bench' and
'fit' modes target different Fortran programs; they must not be interchanged.
The source available in reference/benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS
defines the proposed 'bench' target. Neither the commit message nor its unit
tests alone prove full Fortran agreement. Do not import the entire branch:
it also contains unrelated computation/optimizer changes.

### Bounded preflight, before generating any study data

Reserve at most 90 minutes to inspect and isolate those corrections on a copy
of source 322bd1f, applied identically to all three arms. Preserve old results.
Do not change production defaults or fit an arm with a different model variant.

Audit sv's trial multiplier, spatial geometry and units, threshold and
nondecision variation, drift/noise centering, F1/F2 use, first crossing, tie
selection, horizon and fallback location, parameter bounds, and the objective.
Test the binned CDF against an independently transcribed qant1, explicit boundary
and non-crossing examples, and variability distributions. Recheck FFT200
covariance/normalization, KL mode counts and observer parity on this source.
Use archived Fortran trial outputs at their recorded parameters for distribution
checks where input provenance can be established; do not assert identical PRNG
paths or treat historical fitted parameters as generating truth.

The preflight output must lock ONE of these scopes before seeing fit outcomes:

- Preferred: shared, audited 'bench' semantics with sv active in all arms.
  Label it 'Fortran-informed JAX reference; direct executable qualification
  pending' unless independent original-code evidence actually supports more.
- If the correction cannot be validated within the preflight budget: stop before
  long fits and report the unresolved discrepancy. Do not fall back to frozen
  322bd1f with inactive sv. This supersedes the original draft fallback.
- If FFT200/FFT199 numerical checks or shared-arm consistency fail, do not start
  long fits. Report the defect and readiness limitation instead.

The compiler/MKL/MPI toolchain is currently absent. Installing and validating it
is a separate follow-up, not an unbounded prerequisite that consumes tonight's
whole fitting budget. No replacement of MKL RNG with a different RNG is hidden
under the label 'unchanged Fortran'.

## The three arms

| Arm | Noise | Purpose |
| --- | --- | --- |
| FFT199 | Full original 199-by-319 embedding | Reference fit and recovery baseline |
| FFT200 | Audited full-spectrum 200-by-320 embedding | Exact-covariance speed candidate |
| Repaired KL | Stable mode-indexed, existing 2,000-mode cap | Current approximation control |

No bucketed KL fourth arm, optimizer swap, trial-budget adaptation, or changes
to KL retention settings in this experiment. Record achieved retained variance
and mode cap hits at fitted and evaluated widths. Uniformly use blocked stopping
and the same fit driver so the comparison isolates noise generation. A separate
Fortran fit later checks the original optimizer as well as the model.

## Data choice

Use synthetic data as the primary overnight experiment: known generating truth
is required for absolute parameter recovery. Use real data later for reference
fit agreement and scientific applicability, not 'error from truth'.

Primary truth, fixed before outcomes:

    [ter, st, cr, crsd, sis, sig, sv, av1_c1, av2_c1, av3_c1,
     av1_c2, av2_c2, av3_c2]
    [200, 50, 10, 2, 12, 10, 0.5, 15, 10, 8, 14, 11, 9]

Generate ONE dataset with the locked FFT199 reference, seed 2026102701,
9,000 trials per condition, two conditions. All three arms consume the exact
same saved observations and summary arrays, verified by SHA-256. Preserve raw
RT, category, location, crossing step and crossing flag. Use the same declared
summary construction/objective conventions for every arm.

This sample size is a controlled recovery benchmark; it exceeds the counts in
individual real participant records and is not a claim about human-study
identifiability. Do not regenerate until a convenient dataset appears.

After locking the source, run small fixed-parameter diagnostic samples at sig
5, 10 and 15 (4,096 trials per condition per arm, keys 2026102710..2026102712).
These are descriptive scope checks, not extra full fits or equivalence gates.
Existing large behavioral results only apply to their frozen model; do not
transfer them to a corrected model without qualification.

## Matched fitting budget

- Keep 9,000 simulated trials per objective per condition, chunk 8, time block
  32, precision and PRNG conventions identical across arms.
- Common start: truth times NumPy default_rng(1).uniform(0.8, 1.2, 13), followed
  by the shared declared bounds. Save the exact vector. Same fixed training
  root key 321. Equal seed IDs do not imply matched paths across generators.
- Same frozen simplex driver, default simplex, maxiter 1000, xatol/fatol 1e-5,
  and 8,100-second soft training cap per arm, checked after each evaluation.
- Include all 13 coordinates, with sv active in every arm. There is no
  uncorrected-port fallback.
- Serial order: FFT199, FFT200, repaired KL. No competing GPU jobs. Record
  temperature, clocks, power and sampled memory. Between-arm thermal effects
  remain a limitation of a single fixed-order screen.
- Save every proposal, effective parameters, loss and synchronized evaluation
  time. Retain best-available checkpoints at 30, 60, 90 and 135 training minutes,
  plus the final returned estimate. Never credit an evaluation completed after
  a budget boundary to that earlier checkpoint.
- Never stop or choose an optimizer iterate using the known truth, validation
  seeds, or a favorable recovery result. Caps and failures remain in the table.

Re-run all three fits on these new observations. Reusing old KL fits would
reintroduce the mismatched-data problem. The additional required control is
FFT199 itself; a two-arm KL/FFT200 run cannot measure closeness to its fit.

## Independent scoring and interpretation

Generate an independent FFT199 truth reference of 180,000 trials per condition
with seed 2026102790. Validation keys are 2026102800 through 2026102819, disjoint
from data generation and training, at 9,000 trials per condition. Final estimates
are scored first; then the fixed time checkpoints, with identical coverage
across arms. Deduplicate identical estimates, with cached cost reported.

For every estimate retain two clearly labeled prediction views:

1. Native predictions: that arm's simulator using its fitted parameters. This
   measures the delivered fitted model's behavior.
2. Common-reference predictions: FFT199 using those same fitted parameters.
   This measures what the recovered parameters mean under the reference law.

Compare each with the independent truth reference, the fitted FFT199 reference,
and the observed summaries. Report A/B/C/D/T probabilities, RT quantiles
.1/.3/.5/.7/.9, no-crossing probability, continuous spatial distance, and
low-support probabilities on fixed truth masks at 5/10/20% of maximum drift.
Keep conditional-crossing and historical fallback-category summaries distinct.

Report parameter-specific signed and relative errors against truth and signed
differences from the FFT199 fit; mean/worst errors are supplementary. Spotlight
crsd and the six drifts without omitting other parameters. A single fit cannot
estimate recovery bias/RMSE across datasets, or a reliable recovery correlation.

Fresh scoring seeds quantify Monte Carlo uncertainty of predictions conditional
on a fitted estimate. They are NOT repeated independent fits or participants.
Do not bootstrap them to manufacture parameter-recovery confidence intervals.
Retain descriptive prediction intervals and sample counts. Original proposed
screens (5% mean / 10% worst recovery, 0.5 pp / 5 ms / 2 px behavior) remain
visible as uncalibrated engineering screens; do not loosen or relabel them as
Fortran-derived acceptance criteria. Broad simultaneous behavioral equivalence
may remain inconclusive at this budget.

A valid next-morning conclusion is a case-specific cost/quality comparison:
which arm is faster at matched training time, whether a speed gain accompanies
materially changed estimates/predictions, and whether the data support further
testing. Rank speed only alongside the quality vector. Faster convergence with
worse reference agreement is a tradeoff, not an accepted optimization. If no
arm dominates or uncertainty is too large, say that explicitly.

Do not demand that optimized code improve the reference's intrinsic recovery.
Conversely, copying the reference's parameter bias is not proof of ground truth.
To establish a baseline-derived acceptable change, follow this screen with
independent FFT199 datasets and repeated starts/streams, measure the reference's
own variability, and choose scientifically meaningful excess-error margins
before testing new candidate outcomes. One reference fit cannot calibrate that.

## Overnight budget and deliverables

Planning assumption: local review by 08:00 America/New_York on October 27,
2026, if launched on the evening of October 26. The actual launch must record
an absolute deadline and verify there is enough remaining time for the SAME
fit cap in every arm plus reserved verification. Do not grant early arms more
budget because late arms hit the morning deadline. If launched too late, lower
all caps prospectively or report a later ETA before any fit starts.

Budget envelope: preflight <=90 minutes; fitting <=6h45m plus cold warmups;
data generation, diagnostic samples, scoring and reporting <=2 hours initially.
That is roughly 10h15m at worst, with soft-cap overrun possible. Recent frozen
fits took 57 minutes (FFT200) and 95 minutes (KL); FFT199's full fit time is
unknown, and corrected semantics could change all these costs. Do not present
those numbers as a guaranteed completion estimate. Reserve final scoring for
all methods before optional checkpoint scoring; no 30-round timing campaign.

Morning report:

- One clearly labeled table of shared data/model settings, convergence/caps,
  cold time, training time, verification time, evaluations and recovery.
- Parameter plot with generating truth, FFT199 estimate, FFT200 estimate and KL
  estimate; no different-unit parameters pooled into a meaningless correlation.
- Region/RT/spatial prediction differences, with Monte Carlo uncertainty and
  unassessed or inconclusive checks visible.
- Cost-versus-quality checkpoint plots and a scoped recommendation: candidate
  worth advancing, observed distortion, or insufficient evidence.
- Explicit statement of whether Fortran executable agreement was established
  (expected: not yet), and identify the corrected JAX semantics used.

## Implementation amendment, October 26

The source audit also identified zero-based spatial coordinates in the old port
against one-based coordinates in the Fortran, and a different exact-tie MAXLOC
ordering. Both are corrected in the isolated source for every arm. Independent
geometry, CDF, crossing, timeout and covariance checks passed; numerical
precision and RNG differences remain disclosed. Production defaults and old
archives are unchanged. Runner: `scripts/run_fft_reference_recovery.py`;
preparer: `scripts/prepare_fft_reference.py`; preflight:
`scripts/fft_reference_preflight.py`; launcher:
`scripts/start_fft_reference_recovery.ps1`. The equal 135-minute caps remain;
completion may be later than the draft morning target. Final validation takes
priority over a deadline, and checkpoints are validated after all final fits.

## Real participants and original Fortran follow-up

Do not spend tonight's three-fit budget on real participant fits: their unknown
truth cannot answer the requested absolute-recovery question. Next, use the 16
two-condition records in data/twod3datanew for reference-fit agreement. Repeated
original Fortran starts are needed to distinguish optimizer variability from
implementation differences. Their summary-only format does not support invented
held-out trial splits. Keep the actual reference executable, geometry, sv,
objective and fit schedule explicit. The separate participant plan documents
the missing toolchain and disabled fitting calls.

## Sources

- Repository original Fortran and archived outputs under reference/.
- Shared-model correction: commit 7c05436; review and tests required, not assumed
  validated merely from its message.
- [Wilson and Collins (2019), Ten simple rules for the computational modeling
  of behavioral data](https://elifesciences.org/articles/49547): known-truth
  synthetic recovery and checks over scientifically relevant parameter ranges.
- [NIST, Comparing Instruments](https://www.nist.gov/publications/comparing-instruments):
  reference comparison and equivalence are questions distinct from merely
  detecting a statistically significant difference.
