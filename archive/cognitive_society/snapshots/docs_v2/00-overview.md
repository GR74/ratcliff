# 00 — The Big Picture

## What problem does this repository solve?

Human choices are not just right or wrong; they also take time. A person may answer quickly because evidence is strong, slowly because a choice is difficult, or incorrectly because noise pushed internal evidence in the wrong direction. A diffusion decision model (DDM) represents that hidden process as noisy evidence accumulating until it reaches a decision threshold.

This repository began with Roger Ratcliff’s legacy Fortran implementations of spatially extended DDMs. Those programs were scientifically valuable but expensive to run and hard to integrate with modern tools. The project rebuilds them in Python/JAX so the same type of experiment can run on CPUs or GPUs, be tested automatically, fitted to data, visualized in a browser, and extended into new research.

## The three projects inside one repository

~~~mermaid
flowchart LR
    A["Legacy Fortran models<br/>scientific reference"] --> B["JAX Model A<br/>1D, 72 positions"]
    A --> C["JAX Model B<br/>2D, 100 × 160 field"]
    B --> D["Simulation, fitting,<br/>parity tests"]
    C --> D
    C --> E["FastAPI backend"]
    E --> F["React web app<br/>simulate, fit, predict, compare"]
    B --> G["Decision-engine ideas"]
    C --> G
    G --> H["Cognitive Society<br/>agents, trust, networks, adversaries"]
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

### 3. Collective-cognition research

The \`cognitive_society/\` package asks a broader question: if every agent has a noisy decision process, how should agents communicate and decide together? It provides a fast NumPy DDM for large experiments and an adapter for the heavyweight spatial JAX model. Research modules study confidence, trust, cognitive maps of peer competence, adversarial clusters, learned deference, model routing, and label aggregation.

## What success looks like

The project is trying to achieve four things:

1. **Faithfulness:** preserve the Fortran reference behavior statistically.
2. **Speed:** reformulate sequential simulation into GPU-friendly matrix operations.
3. **Usability:** expose the model through a local or hosted web application.
4. **Research extensibility:** test explicit, reproducible hypotheses about collective cognition.

## Important boundaries

This is research software, not a clinical diagnostic tool. Monte Carlo estimates vary with sample size, and GPU benchmark numbers depend on hardware and configuration. The repository distinguishes established DDM ideas from its newer claims: the basic idea of social DDM agents is not novel; the proposed contribution is the combination of inferred cognitive maps, outcome-grounded trust, uncertainty-sensitive adaptation, and adversarial network tests.
