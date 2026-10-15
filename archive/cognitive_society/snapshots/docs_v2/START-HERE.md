# Ratcliff Repository Learning Guide

This is a beginner-first tour of the whole repository. You do not need prior knowledge of diffusion models, JAX, FastAPI, React, or multi-agent systems.

## Suggested learning path

1. [The big picture](00-overview.md) — what the project is trying to achieve.
2. [The science: diffusion decision models](01-ddm-foundations.md) — the core idea behind every simulator.
3. [Repository map](02-repository-map.md) — what every top-level folder is for.
4. [Model A](03-model-a.md) — the smaller 1D spatial model and its optimized JAX port.
5. [Model B](04-model-b.md) — the main 2D Gaussian-random-field model.
6. [Data, loss, fitting, and validation](05-data-fitting-validation.md).
7. [Backend and API](06-backend-api.md) — how Python becomes a web service.
8. [Frontend](07-frontend.md) — how the six interactive tabs work.
9. [Cognitive Society](08-cognitive-society.md) — decision-making agents, trust, and networks.
10. [Research experiments](09-research-experiments.md) — what each experiment tests.
11. [Development and deployment](10-development-deployment.md).
12. [How the repository was built](11-build-history.md) and the [glossary](12-glossary.md).

## One-sentence mental model

The repository turns legacy scientific Fortran simulators into fast, testable JAX programs, exposes the main 2D simulator through a web app, and then reuses diffusion-style decision-making to study how groups of agents should combine confidence, trust, and evidence.

## Reading conventions

- “Model” means a mathematical process that generates choices and reaction times.
- “Simulation” means running that process many times with controlled randomness.
- “Fit” means searching for parameters whose simulations resemble observed data.
- Shapes such as \`(2, 5, 5)\` mean array dimensions: two conditions, five quantiles, five response categories.
- Benchmarks and experimental results are repository findings, not universal guarantees.
