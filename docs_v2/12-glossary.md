# 12 — Glossary and Symbol Guide

## Scientific terms

| Term | Plain meaning |
|---|---|
| Accumulator | Running total of evidence over time |
| Boundary / threshold | Evidence level that triggers a decision |
| Category | Discrete response assigned from the winning position |
| Circulant embedding | Technique that turns a stationary covariance into an FFT-sampleable larger periodic matrix |
| Confidence calibration | Whether higher stated confidence actually predicts smaller error or higher accuracy |
| Covariance | How strongly two random quantities vary together |
| DDM | Diffusion decision model: noisy evidence accumulation to a boundary |
| Drift | Average direction and strength of evidence accumulation |
| FFT | Fast Fourier transform, used here to generate correlated spatial noise |
| First-passage time | First timestep at which evidence crosses a boundary |
| G-squared | Likelihood-ratio discrepancy between observed and predicted bins |
| Gaussian random field (GRF) | Spatially correlated random values over a grid |
| Identifiability | Whether behavior contains enough information to distinguish a parameter |
| Karhunen–Loève (KL) | Eigenmode expansion of a random field; truncation gives a low-rank approximation |
| Monte Carlo | Estimation by repeated random simulation |
| Nondecision time | Encoding/motor delay not spent accumulating evidence |
| Parity | Agreement between implementations, usually in aggregate statistics |
| Phase diagram | Heatmap showing behavior across two parameter dimensions |
| Reaction time (RT) | Total time from stimulus to response |
| Simulation-based inference (SBI) | Learning a posterior from simulated parameter/data pairs when likelihood is unavailable |

## Engineering terms

| Term | Plain meaning |
|---|---|
| API | Defined request/response boundary between software components |
| Chunk | Subset of trials processed together to control memory |
| CORS | Browser rule controlling which origins may call the API |
| FastAPI | Python HTTP framework used by the backend |
| JAX | NumPy-like system with accelerators, automatic differentiation, and compilation |
| JIT | Compile a function on first use for faster repeated execution |
| JSON | Text format used between frontend and backend |
| NumPy | CPU array library used for data handling and scientific utilities |
| Pydantic | Runtime validation for FastAPI request/response models |
| React | Component framework used by the browser UI |
| Static argument | Value known at JAX trace time; changing it may recompile |
| Tailwind CSS | Utility-class styling system |
| three.js | WebGL library used for the 3D field |
| TypeScript | JavaScript with static types |
| Vite | Frontend development server and bundler |
| XLA | Compiler used by JAX for CPU/GPU programs |
| Zustand | Small shared-state library used by the frontend |

## Model A symbols

| Name | Meaning |
|---|---|
| \`N=72\` | number of spatial positions |
| \`NSTEP=400\` | maximum accumulation steps |
| \`E=10 ms\` | duration of one step |
| \`cr/a\` | decision threshold |
| \`crsd/sa\` | threshold variability |
| \`si\` | drift bump width |
| \`sig\` | correlated-noise length scale |
| \`av\` | drift amplitude |

## Model B symbols

| Name | Meaning |
|---|---|
| \`N=100, M=160\` | field rows and columns |
| \`av1..av3\` | amplitudes of three drift bumps |
| \`sis\` | spatial width of drift bumps |
| \`sig\` | GRF correlation length |
| \`si=6\` | fixed response-zone width |
| \`LAM\` | square root of covariance spectrum |
| \`V_kl\` | materialized truncated KL basis |
| \`nsim\` | number of Monte Carlo trials |

## Array-shape reading example

\`quant.shape == (2, 5, 5)\` in Model B means:

~~~mermaid
flowchart LR
    A["Axis 0<br/>2 conditions"] --> B["Axis 1<br/>5 RT quantiles"]
    B --> C["Axis 2<br/>5 categories"]
~~~

Always confirm axis meaning from the function contract. Equal axis lengths can hide accidental transposes.
