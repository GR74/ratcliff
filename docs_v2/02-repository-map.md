# 02 — Repository Map

## Top-level structure

~~~mermaid
flowchart TB
    ROOT["repository root"]
    ROOT --> MA["model_a/<br/>1D JAX simulator"]
    ROOT --> MB["model_b/<br/>2D JAX simulator"]
    ROOT --> SH["shared/<br/>data, PRNG, validation"]
    ROOT --> BE["backend/<br/>FastAPI service"]
    ROOT --> FE["frontend/<br/>React application"]
    ROOT --> AR["archive/cognitive_society/<br/>paused research"]
    ROOT --> DA["data/<br/>observed fixtures"]
    ROOT --> RF["reference/<br/>Fortran oracle and archives"]
    ROOT --> SC["scripts/<br/>benchmarks and SBI"]
    ROOT --> DOC["docs/<br/>simulator and application plans/results"]
~~~

## Directory-by-directory

| Path | Purpose | Read first |
|---|---|---|
| \`model_a/\` | 72-position, three-category JAX DDM | \`simulate.py\`, \`objective.py\`, \`fit.py\` |
| \`model_b/\` | 100 × 160, five-category Gaussian-random-field DDM | \`simulate.py\`, \`grf.py\`, \`api.py\` |
| \`shared/\` | Deterministic JAX keys, data parsing, parity utilities | \`prng.py\`, \`data_io.py\` |
| \`backend/\` | HTTP API, uploads, jobs, optional GPU proxy | \`main.py\` |
| \`frontend/src/\` | React UI, API client, state, Plotly/three.js views | \`App.tsx\`, \`lib/api.ts\` |
| \`archive/cognitive_society/\` | Preserved paused research, excluded from active packages and tests | \`README.md\` |
| \`data/\` | Original aggregate input files | loaders in \`shared/data_io.py\` |
| \`reference/\` | Preserved implementations and historical outputs | Fortran and \`twod24_jax.py\` |
| \`scripts/\` | Smoke runners, H100 benchmarks, KL diagnostics, SBI | choose by task |
| \`docs/plans/\` | Detailed design and implementation rationale | stage-numbered files |
| \`docs/notes/\` | Completion reports, benchmark records, caveats | stage summaries |

## Root configuration

- \`pyproject.toml\` declares Python 3.11+, dependencies, optional groups, packages, pytest paths, and \`slow\`/\`perf\` markers.
- \`archive/\` is excluded from default pytest collection and the Docker build context; package discovery includes only the active Python packages.
- \`conftest.py\` enables JAX 64-bit mode before test imports. Numerical correctness depends on this ordering.
- \`Dockerfile\` builds the frontend with Node 20, installs the Python app in Python 3.12, and serves it on port 7860.
- \`HF_SPACE_README.md\` contains Hugging Face Space metadata.
- \`AGENTS.md\` is the concise contributor guide.

## Tests live beside packages

- \`shared/tests/\`: parsing, PRNG determinism, statistical comparisons;
- \`model_a/tests/\`: shapes, determinism, parity, performance, fitting;
- \`model_b/tests/\`: GRF validity, FFT/KL parity, objectives, wrappers;
- \`backend/tests/\`: endpoints, parsers, and job lifecycle;

## Source, reference, and generated output

Keep three categories separate:

1. **Source:** Python, TypeScript, configs, scripts, docs.
2. **Scientific reference:** Fortran programs, archived outputs, real fixtures.
3. **Generated output:** \`frontend/dist/\`, caches, logs, screenshots, benchmark text.

Reference code is not dead history. It defines the behavior the JAX versions must reproduce in aggregate.
