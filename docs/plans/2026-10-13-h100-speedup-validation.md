# Time blocking and staged fitting: H100 validation

Implement the two meeting proposals as optional paths. Existing simulation and
fitting defaults remain the control until H100 measurements support a change.

## Implemented options

- `simulate_blocked` / `simulate_b_blocked`: generate time blocks and check
  every intermediate step. Compact unfinished trials into static batch sizes.
- `stop_early=False`: the full-horizon control with the same indexed random
  inputs. Keep block size fixed in paired comparisons.
- `trial_schedule=[TrialStage(...), ...]`: restart the simplex at each larger
  trial budget, carrying parameters forward but recomputing all losses. The
  last stage optimizes at the requested full trial count.
- Keys depend on absolute trial ID and block ID, so increasing trial counts
  retains the earlier trials. Existing legacy noise layouts are unchanged.

## Comparison and budget

The user initially authorized one hour, then extended the rental limit to two
and finally three hours from deployment to test ideas 3–5. See
`2026-10-13-model-b-extension-tests.md` for the full plan and shutdown deadline.
The H100 NVL ran out of capacity; this experiment uses
one Secure Cloud H100 SXM (80 GB) deployed on October 13. Measure all comparisons
on the same GPU. Do not interpret differences from June's H100 records as
changes due only to this implementation.

Benchmark legacy full-path code as well as a paired 2-by-2 experiment:

| Simulation | Fitting |
|---|---|
| Indexed blocks, full horizon | Fixed trial budget |
| Indexed blocks, early stop | Fixed trial budget |
| Indexed blocks, full horizon | Increasing trial budgets |
| Indexed blocks, early stop | Increasing trial budgets |

First screen production-size simulator timings and distributions, then use
smaller but identical final fitting budgets for repeated pilot recovery. Run
larger recovery comparisons only when the remaining time allows them. The
runner must enforce its wall-clock deadline and checkpoint completed cases.

Record GPU model/driver, package versions, source commit, seeds, precision,
trial counts, chunk/block sizes, compile/warm-up time, steady-state timings,
peak device allocator memory when available, loss evaluations, total fitting
time, convergence flags, and error on the active parameters (exclude `sv`).

## Correctness and scientific checks

1. Fixed-seed blocked control versus early stop: choices, RTs, and work count.
2. Crossings between block endpoints; strict boundary; padded final block;
   empty/padded trials; slow/no-crossing trials; each model's existing fallback.
3. Independent legacy-versus-blocked distributions: per-category proportions
   and RT quantiles, with sampling uncertainty and multiple seeds.
4. Recovery on the same synthetic observed data and identical starting points.
   Score all fitted results on the same independent full-budget evaluation.
5. Report time and accuracy together. A lower-iteration or incomplete run is
   not evidence of faster convergence. Do not treat a nonsignificant difference
   as proof of statistical equivalence.

The first pass is a screening study, not a broad validation on human data.
Keep raw logs and arrays outside Git; publish compact results and the commands
needed to reproduce them after the GPU experiment completes.

## Running the first pass

On the H100 pod, use `bash scripts/run_h100_speedup_validation.sh`. The wrapper
caps installation, testing and benchmarking together at 55 minutes; reserve
five minutes to download results and stop GPU billing. Stopping
the benchmark process alone does not stop the pod. Use a fresh output folder
for each run so interrupted or old results cannot be mistaken for new ones.

The default pilot uses 2,048 final trials per condition and 40 total simplex
iterations for each of three starting points. The independent scoring budget
is 9,000 trials per condition. These are fixed iteration comparisons, not
completed fitting to convergence. Screening margins are provisional: category
proportions within one percentage point, RT quantiles within two percent, mean
active parameter error increasing by at most two percentage points, and
independent G2 increasing by at most ten percent. Sparse RT categories remain
unscored. Bootstrap intervals describe this pilot's timing samples, not all
machines or data. Inspect incomplete cases and convergence flags in the report.

With the extended rental, use a separate follow-up directory for selected
full-budget comparisons: `--fit-variants legacy,early_stop,combined
--data-nsim 9000 --final-nsim 9000 --fit-iters 40 --sim-seeds 12`. Keep three paired starts,
the same independent scoring seed, and an explicit deadline based on time left
in the two-hour rental. Finish as many complete paired comparisons as that
deadline allows; incomplete comparisons must remain labelled incomplete.

The follow-up simulator timer queues all conditions before one final wait,
including the work counters. The initial pilot read a counter after each
blocked condition, adding waits only to the blocked variants; retain that
pilot for provenance but use the corrected follow-up simulator timings.
Actual fitting measurements use neither of these work-counter waits.

Run cProfile and line_profiler in their own bounded window while benchmark
workers are stopped. Python profiles diagnose host work and synchronization;
they do not resolve individual GPU kernels or supply speedup evidence.
