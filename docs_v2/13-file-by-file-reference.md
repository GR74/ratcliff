# 13 — File-by-File Reference

Use this chapter after the conceptual guides when you need to locate an implementation.

## Scientific core

| File | Responsibility |
|---|---|
| \`shared/prng.py\` | typed JAX root, condition, and trial keys |
| \`shared/data_io.py\` | Model A/B legacy aggregate loaders |
| \`shared/validation.py\` | proportion and RT-quantile parity reports |
| \`model_a/jax_port.py\` | first faithful scan-based JAX port |
| \`model_a/simulate.py\` | optimized discrete 72-position simulator |
| \`model_a/objective.py\` | vectorized four-condition exact G-squared |
| \`model_a/simulate_smooth.py\` | differentiable crossing/category surrogate |
| \`model_a/objective_smooth.py\` | differentiable surrogate loss |
| \`model_a/fit.py\` | simplex, smooth L-BFGS, and hybrid fits |
| \`model_b/grf.py\` | exact circulant-embedding GRF |
| \`model_b/grf_kl.py\` | cached low-rank spectral basis and sampler |
| \`model_b/simulate.py\` | FFT/KL five-category field simulator |
| \`model_b/objective.py\` | two-condition, 13-slot exact objective |
| \`model_b/fit.py\` | Nelder–Mead fit with progress callback |
| \`model_b/api.py\` | device-aware, JSON-friendly application wrapper |

## Web application

| File | Responsibility |
|---|---|
| \`backend/main.py\` | routes, Pydantic models, warm-up, static mount |
| \`backend/parsers.py\` | upload format detection and aggregation |
| \`backend/jobs.py\` | in-memory fit states and progress |
| \`backend/byo_gpu.py\` | generic remote GET/POST helpers |
| \`frontend/src/main.tsx\` | React entrypoint |
| \`frontend/src/App.tsx\` | lazy tab shell |
| \`frontend/src/store.ts\` | shared Zustand state |
| \`frontend/src/lib/types.ts\` | parameters and API contracts |
| \`frontend/src/lib/api.ts\` | typed fetch functions |
| \`frontend/src/lib/configs.ts\` | localStorage import/export |
| \`frontend/src/tabs/ForwardSimTab.tsx\` | debounced preview/full simulation |
| \`frontend/src/tabs/FieldTab.tsx\` | field generation and animation controls |
| \`frontend/src/tabs/PhaseTab.tsx\` | two-parameter sweep heatmap |
| \`frontend/src/tabs/FitTab.tsx\` | upload, start, and poll fit |
| \`frontend/src/tabs/PredictTab.tsx\` | 13-parameter condition predictions |
| \`frontend/src/tabs/CompareTab.tsx\` | simulate two saved configs |
| \`frontend/src/components/FieldView.tsx\` | three.js surface, trail, threshold |
| \`frontend/src/components/RTHistogram.tsx\` | category RT distributions |
| \`frontend/src/components/FitProgress.tsx\` | log-loss progress plot |
| \`frontend/src/components/RecoveryTable.tsx\` | parameter error table |
| \`frontend/src/components/ParamSliders.tsx\` | shared nine-parameter controls |
| \`frontend/src/components/ConfigSidebar.tsx\` | named browser configs |
| \`frontend/src/components/DataUpload.tsx\` | upload control |

## Infrastructure and retained evidence

| Path | Responsibility |
|---|---|
| \`scripts/smoke.ps1\`, \`smoke.sh\` | fixed scientific smoke suite |
| \`scripts/h100_*.py\`, \`h100_setup_and_run.sh\` | GPU setup and benchmarks |
| \`scripts/sbi_mvp.py\` | SNPE data generation, training, validation |
| \`reference/*.f*\` | legacy Fortran scientific oracle |
| \`reference/archive_outputs/\` | retained MPI outputs |
| \`data/\` | legacy observed inputs |
| \`docs/plans/\` | intended designs and task breakdowns |
| \`docs/notes/\` | measured completions, caveats, handoffs |
| \`archive/cognitive_society/\` | paused society code, tests, and research documentation |

## Test mapping

Every \`test_*.py\` mirrors a nearby module. Read tests when a docstring is not enough: they are executable examples of expected shapes, error handling, determinism, statistical tolerance, and scientific invariants.
