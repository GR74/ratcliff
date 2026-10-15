# Two speedup ideas for Dr. Ratcliff

October 13, 2026

**Goal:** save computing time while preserving choices, reaction-time
distributions, and parameter recovery. These are experiments to test.

**1. Stop calculating when a simulated decision is finished**

Think of a runner crossing the finish line: their finishing time is already
known. Following them farther adds nothing.

The current GPU code calculates all 400 steps, then finds the first boundary
crossing. It reports the correct decision, but computes later steps too.

```text
Example: a trial finishes at step 40

Current: [steps 1-40: decision] [steps 41-400: unused]
Test:    [steps 1-20] [steps 21-40: decision] -> save result and stop
```

We could process time in small blocks, check every step within each block,
and continue only unfinished trials. The reference Fortran already exits its
evidence-update loop at the boundary. The experiment is to keep that saving
while retaining efficient GPU batching.

**Say:**

> "Your Fortran already stops at the boundary. Could the GPU version process
> time in blocks and stop finished trials, while keeping the GPU busy?"

**Ask:**

- How early do trials usually finish for the parameters you fit?
- Which settings or datasets should we test for slow or non-crossing trials?

**Tradeoff:** managing unfinished trials adds work. Fewer calculations might
not mean less total time on a GPU.

**2. Use fewer simulated trials early in fitting, then increase the number**

Think of testing recipes with small batches before making a large batch.
For each parameter guess, we simulate many decisions to estimate the model's
choice proportions and reaction-time distribution. More trials give a steadier
estimate, but cost more time.

```text
Early search       Refine promising guesses       Final refinement
  512 trials   ->        2,000 trials          ->      9,000 trials
Rough estimate         Clearer estimate             Careful estimate
```

These counts are examples to test, per condition. Keep all the participant's
recorded data in the comparison. Increase the simulated trial count as the
search progresses, and refine the parameters again with the full budget.

**Say:**

> "Could we use fewer simulated trials during the early parameter search,
> then increase the number for refinement and the final fit?"

**Ask:**

- How few trials could we use before rare errors or RT estimates get too noisy?
- What parameter-recovery and RT agreement would make this change acceptable?

**Tradeoff:** small batches can make a poor parameter guess look good by chance.

**Proposed first test:** compare each idea separately against the current code
on synthetic data with known parameters. Measure total fitting time and check
choices, RT distributions, and recovered parameters. Then test real data.

Code checks: [current GPU simulator](../../model_b/simulate.py),
[Fortran stopping rule](../../reference/benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS),
and [current fitting driver](../../model_b/fit.py).
