# 03 — Model A: The 1D Spatial Simulator

## Why Model A exists

Model A is the smaller bridge from the original Fortran to the modern architecture. It has 72 accumulator positions, 400 ten-millisecond steps, and three response categories. Its manageable size made it the right place to establish deterministic randomness, parity tests, vectorized simulation, objectives, and fitting before moving to Model B.

## One simulated trial

~~~mermaid
flowchart TB
    P["Parameters + JAX key"] --> K["Build 72 × 72 covariance<br/>and Cholesky factor"]
    P --> V["Build Gaussian drift bump<br/>centered at position 36"]
    K --> N["Generate 400 × 72<br/>correlated noise"]
    V --> I["Drift + noise"]
    N --> I
    I --> M["Demean each timestep"]
    M --> C["Cumulative sum over time"]
    C --> X["Find first timestep where<br/>max position crosses threshold"]
    X --> B["Map winning position<br/>to category 1, 2, or 3"]
    X --> R["RT = crossing step × 10 ms<br/>+ nondecision time"]
~~~

The mathematical simplification is crucial: demeaning the accumulator after every step is equivalent to demeaning every increment and then taking a cumulative sum. That removes a serial state update and exposes large matrix operations that JAX/XLA can optimize.

## Important files

### \`jax_port.py\`: faithful first port

This is copied from \`reference/twod24_jax.py\`. It keeps a per-timestep \`jax.lax.scan\` and serves as an immutable-style baseline for later parity checks. It documents a legacy ambiguity: the Fortran’s across-trial drift-variability argument appears inert because of a name mismatch, so \`SV_ACTIVE = False\` preserves observed behavior.

### \`simulate.py\`: optimized discrete simulator

The fast path pre-generates a whole \`(chunk, 400, 72)\` noise block, applies one batched Cholesky matrix multiplication, demeans, cumulatively sums, and locates the first crossing. Trials are processed in chunks to bound memory. \`nsim\` and \`chunk_size\` are static JIT arguments, so new size pairs cause a new compilation.

### \`objective.py\`: exact aggregate loss

\`fofs_new\` runs four conditions formed by two drifts crossed with two thresholds. It vectorizes simulation across conditions and computes a three-category G-squared loss.

### Smooth path

\`simulate_smooth.py\` replaces first-crossing and category decisions with continuous approximations:

- sigmoid/log-space weights approximate the first crossing;
- softmax over positions gives soft category membership;
- the result is differentiable with respect to parameters.

\`objective_smooth.py\` uses soft proportions and a smooth median-matching term. It is deliberately a biased optimization aid, not a replacement for the exact scientific objective.

### \`fit.py\`

- \`fit_simplex\`: SciPy Nelder–Mead on the exact discrete loss.
- \`fit_lbfgs_smooth\`: JAXopt L-BFGS on the smooth surrogate.
- \`fit_hybrid\`: smooth L-BFGS to find a useful basin, then exact simplex polishing.

## Parameter vector

Model A uses:

| Index | Name | Meaning |
|---:|---|---|
| 0 | \`ter\` | mean nondecision time |
| 1 | \`st\` | nondecision-time variability |
| 2 | \`a1\` | threshold for conditions 1–2 |
| 3 | \`sa\` | threshold variability |
| 4 | \`si\` | drift bump width |
| 5 | \`sig\` | spatial-noise correlation length |
| 6 | \`sv\` | legacy drift variability; preserved but inert in faithful path |
| 7–8 | \`drift1/2\` | drift amplitudes |
| 9 | \`a2\` | threshold for conditions 3–4 |

The \`clamp\` function enforces legacy parameter bounds before simulation.

## How Model A is verified

Tests cover finite outputs, shapes, deterministic keys, changed outputs for changed keys, parity between the scan and optimized implementations, objective parity, nonzero smooth gradients, synthetic fitting, and performance markers. Statistical parity matters more than identical random draws because vectorization changes how random numbers are consumed.
