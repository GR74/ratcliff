# 09 — Scientific Experiments and Inference

The active experiments evaluate the simulators and parameter-fitting methods.
Record the random seed, dependency versions, input data, hardware, and whether
compilation time is included when reporting a result.

## Implementation and statistical parity

- Model A compares the optimized simulator against the initial JAX port.
- Model B checks the FFT and low-rank KL random-field paths and retained Fortran
  reference outputs.
- Compare response proportions and reaction-time quantiles; identical trial
  outcomes are not expected from independently implemented random generators.

See `shared/validation.py`, `model_a/tests/test_simulate_parity.py`, and
`model_b/tests/test_simulate_b_kl_parity.py` for the validation contracts.

## Parameter recovery and performance

Synthetic recovery experiments generate data from known parameters, fit from a
separate starting point, and compare the result with the generating values. A low
loss alone does not establish that every parameter is identifiable.

The H100 runners in `scripts/` retain the FFT and KL benchmark workflows. Their
historical results are documented in the Stage 4–6 completion notes under
`docs/notes/`; hardware-specific timings are not promises for a laptop.

## Simulation-based inference track

\`scripts/sbi_mvp.py\` generates Model B simulations, trains an SNPE neural posterior, and checks recovery/calibration. The recorded N=3,000 run reports good recovery for several location/drift parameters and weak recovery for variability parameters.

One especially important code-level caveat: the current Model B simulator does not consume the \`sv\` parameter even though the SBI prior includes a 13th slot. Consequently \`sv\` is structurally unidentifiable in the current implementation, not merely difficult to estimate. Any future SBI study should either activate a scientifically defined \`sv\` effect or remove that dimension.

## Reproduction commands

~~~powershell
pytest model_a/tests/test_simulate_parity.py -v
pytest model_b/tests/test_simulate_b_kl_parity.py -v
pytest -m "not slow"
.\scripts\smoke.ps1
~~~

The SBI prototype requires additional training dependencies and a suitable GPU
environment. Inspect `scripts/sbi_mvp.py` and its research design before running a
training job.

## Archived experiments

Cognitive Society, swarm, confidence pooling, AI routing, and crowd-label
experiments are paused in [the archive](../archive/cognitive_society/README.md).
The original version of this chapter is preserved in its `snapshots/docs_v2/`
directory.
