# Ratcliff Repository Learning Guide

This is a beginner-first tour of the active scientific simulators and web application. You do not need prior knowledge of diffusion models, JAX, FastAPI, or React.

## Suggested learning path

For a simpler illustrated lesson, start with [the DDM math guide](../docs/tutorials/DDM_MATH_VISUAL_GUIDE.md), then [how Gowrish sped up the models with JAX](../docs/tutorials/JAX_SPEEDUP_VISUAL_GUIDE.md).

1. [The big picture](00-overview.md) — what the project is trying to achieve.
2. [The science: diffusion decision models](01-ddm-foundations.md) — the core idea behind every simulator.
3. [Repository map](02-repository-map.md) — what every top-level folder is for.
4. [Model A](03-model-a.md) — the smaller 1D spatial model and its optimized JAX port.
5. [Model B](04-model-b.md) — the main 2D Gaussian-random-field model.
6. [Data, loss, fitting, and validation](05-data-fitting-validation.md).
7. [Backend and API](06-backend-api.md) — how Python becomes a web service.
8. [Frontend](07-frontend.md) — how the six interactive tabs work.
9. [Scientific experiments and inference](09-research-experiments.md) — parity, recovery, benchmarks, and SBI.
10. [Development and deployment](10-development-deployment.md).
11. [How the repository was built](11-build-history.md) and the [glossary](12-glossary.md).

## One-sentence mental model

The repository turns legacy scientific Fortran simulators into fast, testable JAX programs and exposes the main 2D simulator through a web app.

Cognitive Society research is paused in the [archive](../archive/cognitive_society/README.md), including its original learning-guide chapter and research documentation.

## Reading conventions

- “Model” means a mathematical process that generates choices and reaction times.
- “Simulation” means running that process many times with controlled randomness.
- “Fit” means searching for parameters whose simulations resemble observed data.
- Shapes such as \`(2, 5, 5)\` mean array dimensions: two conditions, five quantiles, five response categories.
- Benchmarks and experimental results are repository findings, not universal guarantees.
