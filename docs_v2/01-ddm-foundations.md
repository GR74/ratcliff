# 01 — Diffusion Decision Models From Scratch

## Start with a simple choice

Imagine deciding whether a noisy image contains a face. Small pieces of evidence arrive over time. Some favor “face,” others favor “not face.” A DDM keeps a running evidence value:

\[
x_{t+\Delta t}=x_t+v\Delta t+\sigma\sqrt{\Delta t}\epsilon_t
\]

Here \(x_t\) is accumulated evidence, \(v\) is average drift, \(\sigma\) controls noise, and \(\epsilon_t\) is a fresh standard-normal value. A decision occurs when evidence reaches a boundary. The repository generalizes \(x\) from one number to a line or a 2D field.

~~~mermaid
flowchart LR
    S["Stimulus / evidence"] --> I["Noisy evidence increments"]
    I --> A["Accumulator grows over time"]
    A --> Q{"Any location<br/>above threshold?"}
    Q -- "No" --> I
    Q -- "Yes" --> P["Winning location"]
    P --> C["Response category"]
    P --> R["Reaction time"]
~~~

## The speed–accuracy tradeoff

A high threshold requires more evidence. Decisions become slower but usually more accurate. A low threshold produces quicker, less reliable decisions. This is the model’s speed–accuracy tradeoff.

Reaction time contains decision time plus nondecision time. The \`ter\` parameter represents encoding and motor delay outside evidence accumulation; \`st\` adds across-trial variation to that delay.

## What “spatially extended” means

In an ordinary two-choice DDM, evidence is a scalar. In Model A, the accumulator is a vector of 72 positions. In Model B, it is a 100 × 160 grid. Every timestep adds:

1. a structured drift pattern favoring particular locations;
2. spatially correlated random noise;
3. a demeaning operation keeping the field centered.

The first location to cross the threshold determines the category.

~~~mermaid
flowchart TB
    D["Drift profile<br/>signal"] --> ADD["Add"]
    N["Correlated random field<br/>noise"] --> ADD
    ADD --> M["Subtract spatial mean"]
    M --> U["Add to previous accumulator"]
    U --> T{"Maximum > threshold?"}
    T -- "No" --> D
    T -- "Yes" --> W["Read winning position and time"]
~~~

## Why correlated noise?

Nearby positions in a perceptual field should not fluctuate independently. A Gaussian process or Gaussian random field gives noise a covariance structure: nearby locations tend to move together, with correlation falling as distance grows. Model A uses a covariance matrix and Cholesky factor. Model B uses circulant embedding and FFTs, or a low-rank Karhunen–Loève approximation.

## Simulation versus fitting

**Forward simulation** starts with parameters and produces synthetic reaction times and categories. **Fitting** starts with observed behavior and repeatedly simulates candidate parameters until aggregate behavior matches. Because trials are stochastic, the fit compares category proportions and five conditional reaction-time quantiles rather than identical trial sequences.

## The G-squared objective

The loss functions compare observed bin counts with simulated expected counts using a likelihood-ratio statistic:

\[
G^2=2\sum_i O_i\log(O_i/E_i)
\]

Each sufficiently populated response category is divided into six RT bins by five observed quantiles. Sparse categories use a simpler lumped proportion term. Lower loss means a better match.

## Why gradients can fail

The exact simulator contains discrete first crossings, \`argmax\`, hard categories, and indicator comparisons. Tiny parameter changes often leave those outputs unchanged, so automatic differentiation sees zero gradient almost everywhere.

Model A therefore supports Nelder–Mead simplex on the true objective and a differentiable surrogate with soft crossing/category operations followed by simplex polishing. Model B currently uses simplex fitting.
