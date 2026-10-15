# Stage 11 Research Design — Amortized Simulation-Based Inference for the 2D Diffusion Model

**Date:** 2026-06-05
**Status:** RESEARCH DESIGN (not implementation-ready; a research program)
**Scope:** Train a neural posterior estimator (NPE) on the fast K-L simulator so the 2D spatial diffusion model can be fit in milliseconds with full Bayesian posteriors — and reuse that same network as the cognitive engine for agentic applications.

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
- **Real-time per-individual inference** — the bridge to agentic use (below).

## 5. Agentic extensions (the same network, three roles)

The trained NPE is a function `behavior → posterior over decision parameters`. That single artifact serves three applications:

### 5.A The experimenter-agent (amortized Bayesian experimental design)
An agent that, given the running posterior over a subject's params, picks the **next stimulus** to maximize expected information gain — adaptive psychophysics that fits a subject in far fewer trials. Grounded in:
- Deep Adaptive Design ([Foster et al.](https://arxiv.org/abs/2103.02438))
- [Amortized BED for Decision-Making, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf)
- [Amortised Experimental Design for User Models of Pointing, 2023](https://arxiv.org/pdf/2307.09878) — almost this exact idea for a spatial pointing task
Novel for the spatial DDM. Real methods paper.

### 5.B The self-modeling agent (online amortized inference = the Exo engine)
Each agent runs amortized inference on its own behavioral trace to maintain a **real-time posterior over its own DDM params** — its cognitive state (how decisive, how cautious). This is the Exo cognitive engine made rigorous: the per-individual fit is one forward pass, fast enough to be a runtime component.

### 5.C The other-modeling agent (amortized Theory of Mind)
An agent observes another agent's choices + RTs and infers the other's drift/threshold via the same NPE — computational Theory of Mind with a **real cognitive model as the generative model of the other agent**. In a multi-agent (A2A) system, agents maintain calibrated beliefs about each other's decisiveness/caution and adapt. Current ToM literature ([ToMCAT 2025](https://arxiv.org/pdf/2502.18438), [AutoToM 2025](https://arxiv.org/pdf/2502.15676), [ToM via Active Inference 2025](https://arxiv.org/pdf/2508.00401)) uses RL/planning models; **nobody has used a diffusion decision model as the ToM generative model.** Novel, and it is the Exo A2A vision made rigorous.

## 6. The unification (why this matters strategically)

Amortized inference is the bridge between the science and the product:
- **Science:** fast Bayesian fitting of the 2D model (Paper 2).
- **Exo self-modeling:** the same net = real-time per-user cognitive engine (5.B).
- **Exo A2A:** the same net = agents inferring each other (5.C).

**One artifact, three applications.** The methods paper and the product's core engine are the same network.

## 7. Honest feasibility + risks

- **Training-data generation is the real cost:** ~1-6 days of GPU (~$50-300). Feasible only because of K-L. This is the load-bearing dependency.
- **Misspecification:** amortized nets fail silently off-distribution. SBC + posterior-predictive checks are mandatory, not optional.
- **Scope discipline:** Paper 2 (NPE for the 2D model + validation) is the feasible anchor. The agentic extensions (5.A–5.C) are follow-ons, each its own project. Do NOT try to build the multi-agent society first.
- **Skip:** neural-operator GRF generation (high-risk, K-L is fast enough); full differentiable simulator (secondary to SBI for this model).

## 8. Recommended sequence

1. **Paper 1:** JAX/K-L methods paper (validation + writing) — the foundation. ~weeks.
2. **Paper 2:** NPE for the 2D spatial DDM (this design) — novel, field-aligned, feasible. ~2-3 months. Collaborators: Ratcliff + the Frank/Fengler (HSSM) group.
3. **Paper 3 / Exo:** one agentic extension — most fundable is 5.A (experimenter-agent) for academia, 5.B/5.C for Exo. Pick by track.

## 9. MVP (the smallest feasible first step)

Before committing GPU days: a **proof-of-concept on a coarse prior**. Generate ~20k datasets (a few hours GPU), train a small flow, run SBC. If the posterior is calibrated and agrees with the simplex on a handful of test cases, the approach is validated and worth scaling to the full training set. This de-risks the whole program for ~$10 of GPU.
