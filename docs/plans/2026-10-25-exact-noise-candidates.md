# Model B: testing two structured noise generators

Status: Candidate A is implemented as the opt-in `fft200=True, use_kl=False`
option. Its initial correctness and objective-speed screen is recorded in
`docs/results/2026-10-25-fft200-screen/`. Candidate B remains prospective.
The initial screen does not complete the behavioral, recovery or fitting gates
below and does not change production defaults. The goal is lower time to
acceptable scientific fit quality while preserving parameter meaning, spatial
responses, and reaction-time distributions.

## 1. Freeze the experiment and two kinds of baseline

Use the same RTX 4060, pinned environment, precision, data, trial counts, chunk
and block sizes, first-passage logic, optimizer and stopping rules for all arms.
No concurrent GPU workloads. Record hardware/power settings, source copies and
hashes, dependencies, seeds, data hashes and exact commands in a manifest.
Current checkout HEAD at planning: `ccfd5b94b5a1e7ca3fb76a22fdfe1a1bb10a7c0a`;
uncommitted files also exist, so a commit ID alone is not a reproducible snapshot.

Maintain separate comparisons:

- **Fidelity reference:** full-spectrum 199-by-319 FFT in the selected frozen
  simulator. Audit it against the analytical covariance and historical reference
  first. The existing JAX port and its inactive drift-variability parameter are
  not independently qualified as the intended scientific model.
- **Engineering baseline:** repaired source `322bd1f` and its existing laptop
  harness. Include the separately tested `kl_bucketed=True` variant as a speed
  comparator because it already showed a benefit. Reproduce both controls on the
  same harness; do not compare a new blocked implementation with an old
  full-horizon implementation and attribute that difference to the sampler.
- **Candidate A:** the same harness with a newly constructed 200-by-320 circulant
  embedding and all spectral modes. The original decision grid remains 100 by 160.
- **Candidate B:** the same harness with a full-covariance Gaussian-derivative
  sampler using separable one-dimensional covariance factors.

Evaluate A and B separately against FFT and the engineering controls. Do not
combine them, change compaction, change the optimizer, or tune scientific sig to
help a candidate. A speed comparison with capped KL does not establish fidelity.

## 2. Mathematical and numerical gate: before large simulations

Test sig = 0.2, 1, 3, 5, 8, 10, 12, 15, 17, plus archived truth/start/fit values.
Also audit a 101-point uniform sig grid over [0.2, 17] and perturb each named
value by +/-1e-6 where within bounds. Record numerical defects, not just pass/fail.

Distinguish the intended analytical covariance C from the effective covariance
of the old FFT, which clips negative embedding eigenvalues. Reconstruct the
latter from its actual spectrum. If their discrepancy exceeds the numerical
budget, stop the exact-equivalence claim and resolve/version the scientific
baseline before adoption; do not silently change the model.

Candidate A:

- Rebuild the embedding from covariance displacements. Do not resize/interpolate
  the old spectrum or zero-pad an already sampled field.
- Verify periodic inversion symmetry, real spectrum, and every lag needed by
  pairs of original-grid locations: dx=-159..159, dy=-99..99. Explicitly check
  the new even-size Nyquist rows/columns and wrap-around conventions.
- Check positive semidefiniteness before any clipping. Materially negative
  eigenvalues reject this embedding. A larger size, such as 256 by 320, is a
  separately recorded candidate revision, not an undisclosed runtime change.
- Confirm FFT normalization and real/imaginary sample independence. A new shape
  must not change the noise variance or correlate successive time increments.

Candidate B:

- Independently derive and verify the covariance of
  eta=(s/2)*(F_x + F_y/2 + sqrt(3)*G_y/2), where F and G are independent Gaussian
  fields with covariance exp(-(dx^2+dy^2)/s^2), and s=sig.
- Factor the one-dimensional joint value/derivative covariance matrices and
  sample derivatives jointly. Finite differences of a sampled field are not this
  construction. Include cross-derivative covariance and independent F/G streams.
- Verify the assembled covariance operator against the analytical target. Use
  the separable algebra to avoid allocating a 16,000-square dense matrix. Audit
  edges, corners, center and 20 fixed random covariance-vector products as well
  as the component factors; the finite-factor numerical error can be nonstationary.
- Test conditioning over the entire sig audit suite. Do not add diagonal jitter,
  discard legitimate positive modes, or clip meaningful negative eigenvalues
  without declaring a new approximation. Roundoff-only clipping must be logged
  and remain within the covariance-error budget.

Proposed numerical budgets, fixed before execution: maximum absolute covariance
error 1e-10 in float64 and 2e-5 in production float32, relative to unit marginal
variance, for the checks above. These are engineering screens, not scientific
behavioral tolerances. Treat negative eigenvalues below
`-100 * eps * matrix_dimension * spectral_scale` as material; even smaller
clipping must pass the reconstructed covariance budget. Test the covariance
after the model's spatial demeaning, P*C*P, as well as before it.

Gate outcome: pass, reject, or unresolved reference discrepancy. No performance
claim is accepted for a candidate that fails this gate.

## 3. First-passage and finite-precision checks

Feed identical manufactured increment fields to the common decision code:
first-step crossings, crossing then falling within a block, simultaneous
crossings with distinct heights, tied maxima, final-step crossings, no crossings,
partial trial batches, and early stopping on/off. Require identical step,
position, category, crossing indicator and RT when arithmetic is identical.

For native random outputs, do not demand identical trajectories from different
factorizations or infer pairing merely from equal seeds. Keep trial-indexed
threshold and nondecision draws common where useful; use independently keyed
noise streams per method unless a mathematically valid coupling is implemented.
Document the coupling. Check repeatability, independence across trials/time,
and all production output/observer agreement before trusting diagnostic metrics.

Start with 4,096 independent fields per sig and method for variance, directional
covariances, temporal cross-covariances, spatial maxima and maximum locations.
These Monte Carlo checks supplement the covariance proof; they do not replace it.

## 4. Cheap performance screen before expensive qualification

At each stage stop for failure, invalid outputs, or the recorded runtime cap.
Use a 20-minute soft cap per candidate for this initial screen, checked between
synchronized calls. Save partial data and mark capped work incomplete.

Measure separately: factor/spectrum construction; RNG; field generation;
normalization/accumulation/crossing; complete two-condition objective. Report
peak device memory and actual generated trial-steps. Synchronize GPU results.

Use 3 warm-ups and 30 alternating baseline/candidate timing pairs where the cap
permits. Compare per-condition trial counts 128, 1,024, and 9,000. The 128-trial
case is a diagnostic only; the main objective screen is 9,000 trials per condition,
chunk 8 and block 32, matching the archived workload. Hold these settings constant
between methods; any implementation-specific tuning is a later experiment.

Distinguish three cache regimes:

1. Repeated sig: warm factors and executable, to measure steady-state sampling.
2. New sig: cached executable but uncached factors, including construction and transfers.
3. Realistic optimization trace: replay an immutable 30-vector segment from the
   archived simplex trace in original order, preserving repeated widths and
   changing drifts. Select the first 30 valid evaluated vectors before examining
   candidate timings, record the exact segment/hash, and start each replay with
   the same empty host-cache policy. Preserve normal cache behavior within it.

Replay-total cost is the primary objective endpoint. First-use compilation and
first-replay cost are reported separately and included later in fit-level cost.
For qualification repeat the full replay 30 paired times, including all its
internal evaluations; do not treat the 30 vectors as 30 independent full replays.
Use whole-replay paired bootstrap intervals, 10,000 resamples, and alternate order.
If the short cap prevents this, retain a preliminary result only.

Promotion target: at least 10% lower median realistic-replay cost, with a paired
95% interval supporting improvement, versus the fastest valid full-fidelity
control tested on the same harness. Also report all ratios versus existing KL
controls, without hiding their fidelity limitations. No individual sig or cache
regime may be omitted because it is slower. Higher memory or worse cold start
must be disclosed. No end-to-end speed claim follows from this gate alone.

## 5. Behavioral fidelity: fixed parameters, no fitting

Primary cells: sig 5, 10, 15, each with both experimental-condition drift vectors
and the frozen ordinary boundary/variability settings. Additional stress cells:
quarter-strength and zero drift, low/high boundaries, and the archived start/fit
vectors. Freeze their numeric values before simulation. For boundary probes use
cr=5 and cr=20 with crsd=2 unless the reference bounds require a documented change.

Initial diagnostic: 4,096 trials per cell and method, with fresh archived seeds.
Qualification: the canonical 20 fresh seeds x 9,000 trials per condition is a
minimum starting design, not a guarantee of adequate precision. Before the
qualification run, use the diagnostic to choose a fixed larger sample count if
needed for 90% power under the specified equivalence margins and multiplicity
correction. Archive that precision calculation and the independent final seed
bank. Do not inspect interim qualification outcomes and keep sampling until pass.

Record every trial's coordinates, crossing flag/step, response region and RT.
Measure all A/B/C/D/T probabilities; fixed-mask low-support win probabilities
(10% cutoff, 5%/20% sensitivity); per-region RT quantiles .1/.3/.5/.7/.9;
64-direction spatial sliced Wasserstein-1 distance; and no-crossing mass.
Audit display-coordinate conversion before interpreting pixel-distance thresholds.
Keep zero-drift geographic-mask controls distinct from an all-locations-low mask.

Use existing proposed engineering tolerances: +/-0.5 percentage points for
probabilities including the primary low-support endpoint, +/-5 ms for conditional
RT quantiles, and at most 2 display pixels for spatial distance. Predeclare the
primary family of both candidates, conditions, metrics and parameter cells.
Require simultaneous 95% uncertainty bounds inside tolerances. Use conservative
exact binary bounds and a prevalidated joint bootstrap procedure for quantiles
and spatial distances; archive resampling settings and adequate tail resolution.
Do not use unadjusted individual 95% intervals to declare a family-wide pass.
Sparse categories and insufficient precision are inconclusive, not passes.

These remain prospective engineering margins, not tolerances established by
Ratcliff's paper. A mathematically exact construction provides additional evidence
of distributional identity; the empirical tests catch implementation/precision
errors and cannot certify the entire parameter domain by themselves.

## 6. End-to-end optimization and parameter recovery

Only survivors of the covariance, behavioral and speed gates reach this stage.
Use the existing optimizer unchanged. Generate synthetic observations with the
audited FFT reference, not with the candidate being fitted. First run one matched
case; then use three truth settings (including sig 5, 10, 15), three independent
datasets per truth, and three starts per dataset: 27 fits per surviving method
and comparator. Freeze truths, starts and budgets before launching fits.

Use matched-wall-time checkpoints and matched-quality endpoints. Choose the
wall-time cap from the archived baseline and screen cost estimates before runs;
checkpoint every 5 training minutes and at termination. Independently score saved
checkpoints with streams disjoint from optimization, and charge verification time
to time-to-quality. Truth diagnoses recovery but must not steer the optimizer.

Keep the canonical recovery screen (mean active relative error <=5%, worst <=10%,
predeclared absolute tolerances near zero) and calibrate predictive acceptance on
independent truth replicates before candidate fits. Preserve inactive-sv limitations
and report every fitted parameter. Better behavior after refitting must not hide
a biased simulator. Parameter self-recovery alone cannot establish FFT equivalence.

Final contribution target: >=10% less time to independently verified acceptable
quality, a paired interval supporting improvement, and no material loss of
successful-fit rate or recovery/prediction quality. For this protocol propose a
5-percentage-point success-rate noninferiority margin, locked before fitting.
Twenty-seven fits may be insufficient to establish that margin; report inconclusive
and predesign a larger study rather than declaring success from similar point rates.
Retain failures and caps as censored runs; do not compare capped failure time with
successful completion time as a speedup. If neither baseline nor candidate reaches
acceptable quality, report no established time-to-quality improvement.

## Deliverable and decision

For A and B separately publish a gate table, source/seed manifests, covariance
error plots, winner maps, per-region probability/RT intervals, timing by component,
cache regime and trial count, memory, fitting trajectories and parameter errors.
Label outcomes: rejected, inconclusive, promising for larger qualification, or
qualified within the tested scope. Keep defaults unchanged until qualification.

Run A first because it is a smaller change; test B against the same frozen
controls afterward. Use cheap algebra/covariance and timing gates to avoid spending
hours fitting a sampler that changes the model or cannot beat the relevant control.
The 20-minute screen caps do not authorize the later multi-hour studies: estimate
and schedule those explicitly after the preliminary evidence is available.
