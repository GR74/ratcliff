# Stage 11 Research Design — Amortized Simulation-Based Inference for the 2D Diffusion Model

**Date:** 2026-06-05
**Status:** RESEARCH DESIGN (not implementation-ready; a research program)
**Scope:** Research amortized Bayesian parameter inference for the 2D spatial diffusion model using the fast K-L simulator. This is a research design; measured recovery and calibration are recorded separately in the validation note.

---

## 1. The gap this fills

The DDM field has moved to amortized / simulation-based inference (SBI):
- **LANs** — Likelihood Approximation Networks ([Fengler, Frank et al. 2021, eLife](https://elifesciences.org/articles/77220))
- **HSSM** — hierarchical Bayesian neurocognitive modeling on LANs ([Fengler et al. 2025](https://ski.clps.brown.edu/papers/Fengler_JoCN.pdf))
- **BayesFlow** — amortized Bayesian workflows ([Radev et al. 2023](https://arxiv.org/html/2306.16015))

**Every one of these targets the standard 1D sequential-sampling models. None targets the spatially-continuous / 2D GRF diffusion model.** That model has no analytic likelihood and the simplex fitting is slow precisely because each likelihood eval needs thousands of MC sims — exactly the case SBI was invented for. We are positioned to be first because **training an NPE requires a fast simulator to generate the training set, and that is what the K-L work built.**

## 2. Why NPE (not LAN) for this model

The 2D model already produces **sufficient summary statistics** in `model_b/objective.py`: per-condition category proportions (5) + RT quantiles per category (5×5 = 25) = ~30 numbers. That means we can skip the learned summary network and train a conditional normalizing flow directly:

    q(θ | s)   where  s = [props, quantiles]  (30-dim)  and  θ = 13 DDM params

At inference: feed a real subject's summary stats → sample the posterior over θ in milliseconds. No MCMC, no simplex, full uncertainty.

## 3. Pipeline

### 3.1 Prior + training-set generation (the expensive, GPU-bound part)

- Define a prior over the 13 params (uniform within `clamp_b` bounds, or weakly informative around Ratcliff's typical values).
- Sample N param sets θ_i. For each, run the K-L simulator at a realistic trial count (~500-1000) and compute summary stats s_i.
- **Vmap over param sets** so the GPU simulates B datasets per call. At B=256 param-sets × 1000 trials ≈ 256k trials/call ≈ ~2 min/batch on H100.
- Target N: ~200k datasets for a 13-param flow (≈ 1 day, ~$50 GPU). 1M for a tighter fit (≈ 6 days). **This is feasible only because of the K-L speedup — without it, months.**

### 3.2 Train the flow

- Conditional normalizing flow (nflows / flow-matching FMPE) or use BayesFlow directly.
- Input: 30-dim summary stats. Output: posterior over 13 params.
- Train on (s_i, θ_i) pairs. ~hours on one GPU once data exists.

### 3.3 Validation (do NOT skip — amortized nets fail silently under misspecification, [2024 robustness work](https://arxiv.org/pdf/2406.03154))

- **Simulation-Based Calibration (SBC):** ranks should be uniform.
- **Posterior predictive checks:** simulate from posterior, compare to held-out data.
- **Agreement with simplex:** NPE posterior mean should match the simplex point estimate on the same data, with calibrated credible intervals.
- **Recovery study:** the multi-seed recovery study from Stage 10 becomes the NPE validation set.

## 4. What this unlocks (the payoff)

- **Fitting 2D model: minutes → milliseconds**, with full posteriors instead of point estimates.
- **Hierarchical Bayesian 2D fits across many subjects** — previously infeasible, now a forward pass per subject.
- **Fast per-subject inference** for scientific model fitting.

## 5. Archived extensions

The original agentic application proposals and their strategy discussion are
preserved in `archive/cognitive_society/snapshots/docs/plans/2026-06-05-stage-11-sbi-research-design.md`.
The active scope is parameter inference and validation for Model B.

## 6. Honest feasibility + risks

- **Training-data generation is the real cost:** ~1-6 days of GPU (~$50-300). Feasible only because of K-L. This is the load-bearing dependency.
- **Misspecification:** amortized nets fail silently off-distribution. SBC + posterior-predictive checks are mandatory, not optional.
- **Scope discipline:** Focus on NPE for the 2D model and its recovery, calibration, and posterior-predictive validation.
- **Skip:** neural-operator GRF generation (high-risk, K-L is fast enough); full differentiable simulator (secondary to SBI for this model).

## 7. Recommended sequence

1. **Paper 1:** JAX/K-L methods paper (validation + writing) — the foundation. ~weeks.
2. **Paper 2:** NPE for the 2D spatial DDM (this design) — novel, field-aligned, feasible. ~2-3 months. Collaborators: Ratcliff + the Frank/Fengler (HSSM) group.

## 8. MVP (the smallest feasible first step)

Before committing GPU days: a **proof-of-concept on a coarse prior**. Generate ~20k datasets (a few hours GPU), train a small flow, run SBC. If the posterior is calibrated and agrees with the simplex on a handful of test cases, the approach is validated and worth scaling to the full training set. This de-risks the whole program for ~$10 of GPU.
