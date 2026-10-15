# 00 — The Big Picture

## What problem does this repository solve?

Human choices are not just right or wrong; they also take time. A person may answer quickly because evidence is strong, slowly because a choice is difficult, or incorrectly because noise pushed internal evidence in the wrong direction. A diffusion decision model (DDM) represents that hidden process as noisy evidence accumulating until it reaches a decision threshold.

This repository began with Roger Ratcliff’s legacy Fortran implementations of spatially extended DDMs. Those programs were scientifically valuable but expensive to run and hard to integrate with modern tools. The project rebuilds them in Python/JAX so the same type of experiment can run on CPUs or GPUs, be tested automatically, fitted to data, and visualized in a browser.

## The two active parts

~~~mermaid
flowchart LR
    A["Legacy Fortran models<br/>scientific reference"] --> B["JAX Model A<br/>1D, 72 positions"]
    A --> C["JAX Model B<br/>2D, 100 × 160 field"]
    B --> D["Simulation, fitting,<br/>parity tests"]
    C --> D
    C --> E["FastAPI backend"]
    E --> F["React web app<br/>simulate, fit, predict, compare"]
~~~

### 1. Scientific computing core

- **Model A** is a one-dimensional spatial accumulator with 72 locations and three response categories. It is useful for validating optimization ideas and understanding the mechanics.
- **Model B** is the primary two-dimensional model. It accumulates a correlated Gaussian random field over a 100 × 160 grid and maps the winning location to one of five categories.
- Both generate reaction times and choices with Monte Carlo simulation.
- Fitting minimizes a G-squared discrepancy between simulated and observed response proportions and reaction-time quantiles.

### 2. Interactive application

The FastAPI backend wraps Model B in JSON-friendly functions. The React frontend lets a user:

- move parameter sliders and inspect reaction-time histograms;
- animate the two-dimensional evidence field;
- sweep two parameters into a phase diagram;
- upload data and run a background fit;
- predict from a 13-parameter fit;
- save and compare configurations in browser storage.

## What success looks like

The active project is trying to achieve three things:

1. **Faithfulness:** preserve the Fortran reference behavior statistically.
2. **Speed:** reformulate sequential simulation into GPU-friendly matrix operations.
3. **Usability:** expose the model through a local or hosted web application.

## Important boundaries

Monte Carlo estimates vary with sample size, and GPU benchmark numbers depend on hardware and configuration. Statistical parity, parameter recovery, and hardware-specific benchmarks support different claims and should be assessed separately.

## Archived research

Cognitive Society, collective-cognition experiments, and the cognitive digital twin research direction are paused in [archive/cognitive_society](../archive/cognitive_society/README.md). Their code, tests, and documentation are preserved for later resumption.
