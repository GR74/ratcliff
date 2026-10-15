# 11 — How the Repository Was Built

## The staged strategy

The project was not built as one large rewrite. The plans and completion notes show a gated sequence: preserve a reference, build a testable slice, verify it, then optimize or expand.

~~~mermaid
timeline
    title Repository evolution
    Stage 0–1 : package skeleton
              : deterministic PRNG and data loaders
              : faithful Model A smoke tests
    Stage 2   : Model A single-GEMM + cumulative-sum rewrite
              : parity against first JAX port
    Stage 3   : vectorized G² objective and simplex fitting
    Stage 3.5 : smooth differentiable surrogate
              : hybrid L-BFGS + simplex
    Stage 4   : Model B GRF, simulator, objective, fitting
    Stage 5   : H100 FFT benchmark and recovery
    Stage 6   : low-rank KL generator
              : cached, padded basis for stable JIT
    Stage 7   : device-aware wrapper, FastAPI, React, Docker
    Stage 8–9 : 3D field, trajectory, phase diagram
    Track B   : amortized SBI prototype and validation
~~~

## 1. Preserve the oracle

\`reference/\` retains Fortran sources, historical cluster outputs, and the first JAX reference. The first engineering goal was not speed; it was creating a reproducible baseline with deterministic keys, loaders, and statistical comparison tools.

## 2. Reformulate, then optimize

Model A exposed the key algebraic equivalence: repeated accumulator demeaning can be written as cumulative sums of demeaned increments. That allowed a scan-heavy algorithm to become matrix multiplication, cumulative sum, and reductions.

The project did not assume every vectorization would help everywhere. CPU performance notes record cases where the “faster” GPU structure was slower locally. This led to device-aware defaults rather than one universal chunk size.

## 3. Discover the gradient problem honestly

Automatic differentiation of the exact objective was structurally zero because first crossings and categories are discrete. Instead of hiding the result, the repository records it and implements a biased smooth surrogate only for coarse optimization, followed by the true objective.

## 4. Scale to the 2D scientific target

Model B ported the circulant Gaussian-random-field generator and five-zone accumulator. Tests were added for positive-definite embedding, field variance, deterministic simulation, objectives, and fitting before H100 benchmark work.

## 5. Reduce memory with KL

The original KL design expected very few modes, but empirical spectrum measurement showed roughly 1,325 complex modes were needed for 99% variance at \`sig=10\`. The implementation adapted rather than preserving the optimistic assumption. A first GPU run also revealed that changing basis width caused repeated compilation; caching and zero-padding fixed the shape and reduced objective time.

The recorded Stage 6 result is about 4.7 seconds per production simulator/objective call and a 20.1-minute synthetic recovery fit on one rented H100, with explicit caveats about comparison to the original MPI workflow.

## 6. Add a stable product boundary

\`model_b/api.py\` became the boundary between scientific arrays and ordinary JSON. FastAPI added uploads and background fits. React added an interactive parameter surface. Docker combined them into one deployable process. Later stages added the 3D evidence field, winning trajectory, and parameter sweeps.

## Paused research

The later Cognitive Society and collective-cognition extensions are preserved in
[the archive](../archive/cognitive_society/README.md). This guide focuses on the
simulators, fitting, validation, and web application.

## Design lessons encoded in the repository

- Keep a scientific reference before optimizing.
- Treat randomness as explicit input.
- Validate aggregate behavior, not just code execution.
- Separate CPU usability from GPU production defaults.
- Freeze shapes to get real JIT reuse.
- Record failed assumptions and null findings.
- Put scientific caveats next to headline results.

## Where to find the detailed record

\`docs/plans/\` contains designs and task-by-task implementation plans. \`docs/notes/\` contains measured results, failed attempts, and completion summaries. For exact historical numbers, prefer the relevant completion note over a plan, because plans describe intended behavior while notes record what actually happened.
