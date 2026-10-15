# 05 — Data, Fitting, and Validation

## The data files

The \`data/\` directory contains legacy aggregate inputs:

- \`twod24data\`: Model A, three categories and four condition rows per subject;
- \`twod3datanew\`: Model B, five categories and two condition rows per subject;
- \`parinp\`: legacy parameter/input material retained with the original workflow.

These are not ordinary trial-level tables. Each category block contains a response proportion, a count, five RT quantiles, and two unused extreme fields.

~~~mermaid
flowchart LR
    L["One condition line"] --> C1["Category 1 block"]
    L --> C2["Category 2 block"]
    L --> CN["...category N block"]
    C1 --> F["proportion | count | q1 q2 q3 q4 q5 | x1 x2"]
~~~

\`shared/data_io.py\` loads full local files into NumPy arrays. \`backend/parsers.py\` parses uploaded text and returns JSON lists for the web application.

## Canonical shapes

| Model | Proportions | Counts | Quantiles |
|---|---|---|---|
| Model A subject | \`(4, 3)\` | \`(4, 3)\` | \`(4, 5, 3)\` |
| Model B subject | \`(2, 5)\` | \`(2, 5)\` | \`(2, 5, 5)\` |

For quantiles, the final two axes mean quantile then category. Both happen to be length five in Model B, so code and documentation must preserve the semantic order even though the numeric shape looks symmetric.

The upload endpoint also accepts CSV columns \`rt,cat,condition\`. It aggregates trials into the same Model B summaries. Conditions must be 1–2 and categories 1–5. Categories with too few trials keep zero quantiles, so high-quality fitting data should contain adequate observations in each active category.

## The fitting loop

~~~mermaid
flowchart TD
    O["Observed proportions,<br/>counts, and RT quantiles"] --> X["Choose initial parameter vector x0"]
    X --> S["Simulate candidate parameters<br/>with fixed random key"]
    S --> A["Aggregate simulated categories<br/>and conditional RT bins"]
    A --> G["Compute G² loss"]
    G --> U["Optimizer proposes<br/>new parameters"]
    U --> S
    G --> Q{"Budget exhausted or<br/>converged?"}
    Q -- "Yes" --> R["Return parameters,<br/>loss, iterations, status"]
    Q -- "No" --> U
~~~

A fixed root random key makes the Monte Carlo objective deterministic for a candidate parameter vector. Without that, optimizer comparisons would be dominated by changing random noise.

## Why Nelder–Mead?

Nelder–Mead maintains a simplex of parameter vectors and compares their loss values. It does not require derivatives, making it appropriate for discrete first-passage simulators. It can be expensive because every function evaluation runs many simulated trials across all conditions.

Model A’s hybrid path reduces that cost by using a smooth surrogate for the coarse search. Model B retains the simpler and more defensible discrete simplex fit.

## Parameter recovery

A recovery test:

1. chooses known “true” parameters;
2. simulates synthetic observations;
3. starts fitting from a different vector;
4. checks whether the fitted values return near the truth.

Recovery is more informative than a low loss alone because multiple parameter combinations can sometimes mimic the same aggregate behavior. The frontend displays percentage error against its default synthetic truth, so that table should not be interpreted as ground truth for arbitrary uploaded human data.

## Validation layers

~~~mermaid
flowchart TB
    U["Unit tests<br/>shapes, bounds, parsers"] --> D["Determinism tests<br/>same key → same output"]
    D --> P["Implementation parity<br/>old JAX ↔ optimized JAX"]
    P --> S["Statistical parity<br/>FFT ↔ KL; JAX ↔ Fortran aggregates"]
    S --> R["Recovery tests<br/>known parameters"]
    R --> B["Hardware benchmarks<br/>wall time and memory"]
~~~

\`shared/validation.py\` compares category proportions with absolute tolerance and RT quantiles with relative tolerance. Exact trial equality is usually inappropriate across independently implemented random generators.

## Test commands

~~~powershell
pytest -m "not slow"
pytest model_a/tests -v
pytest model_b/tests -v
pytest backend/tests -v
pytest cognitive_society/tests -m "not slow" -v
.\scripts\smoke.ps1
~~~

Use \`-m slow\` or \`-m perf\` deliberately. Some Model B tests compile large JAX programs or allocate substantial memory.

## Common interpretation mistakes

- A preview with 128 trials is for visual intuition, not a publishable estimate.
- “Converged” means the optimizer met its stopping rule, not that parameters are uniquely correct.
- A lower approximate KL loss is not proof that the exact FFT model fits equally well; parity tests establish the acceptable approximation range.
- Benchmark comparisons must state \`nsim\`, chunk size, precision, GPU, simulator path, and whether compilation time was included.
