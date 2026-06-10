# Cognitive Society

A small multi-agent system where each agent's decision-making is a **diffusion
decision model** (DDM): noisy evidence accumulates to a threshold; choice +
reaction time fall out; parameters define a personality. Agents will communicate
over a custom trust-weighted protocol, infer each other's cognitive styles, and
adapt to novel problems.

This package is the **agentic layer** that grows out of the Ratcliff 2D-DDM
research codebase. It is a separate, scoped project — a bridge toward (not into)
the larger Exo system.

- **Roadmap:** `docs/plans/2026-06-05-OVERALL-PLAN-cognitive-society.md`
- **SBI research design (Track B):** `docs/plans/2026-06-05-stage-11-sbi-research-design.md`

---

## The core design decision: no compute bottleneck

Society logic (comms, trust, mapping, adaptation) runs on **lightweight 1D NumPy
DDM agents** — no JAX compile, no GPU, no waiting. The full validated **2D
spatial JAX model** (`model_b`, GPU-preferred) stays fully available as a
heavyweight engine, behind the *same interface*. So:

- **Prototype** the society on 1D agents — runs in seconds, iterate freely.
- **Scale / validate** on the 2D GPU engine — the real cognitive model for final
  results.

Both satisfy one contract (`DecisionEngine`), so the society code never knows or
cares which engine it's running. The only GPU-dependent piece is amortized
inference for the 2D model (Track B, async on RunPod) — cleanly isolated.

---

## Two-tier engine

| Engine | Module | Speed | Choices | Cognitive mapping |
|---|---|---|---|---|
| `DDMAgent` (1D) | `agent.py` | microseconds, NumPy | 2 | EZ-diffusion (closed form) |
| `SpatialDDMEngine` (2D) | `engine.py` | sec–min, JAX/GPU | 5 | amortized SBI (Track B) |

The `DecisionEngine` protocol (`engine.py`) is the contract: `n_choices`,
`decide(stimulus, rng) -> (choice, rt)`, `decide_batch(stimulus, n, rng)`.

---

## Modules

### `agent.py` — the 1D decision engine
- `DDMParams` — `boundary` (caution), `drift_scale`, `ndt`, `sigma`. `.style()`
  labels the personality (cautious / balanced / decisive).
- `DDMAgent` — `decide(evidence, rng) -> (choice, rt)` and a vectorized
  `decide_batch(evidence, n, rng)`. Pure NumPy, deterministic per seed.
- `make_population(styles)` — assemble a heterogeneous society.

### `engine.py` — the engine-agnostic interface
- `DecisionEngine` — the protocol both engines satisfy.
- `SpatialDDMEngine` — adapter wrapping `model_b.simulate_b`; maps the 2D
  simulator's (rt_ms, category 1..5) to the society contract (choice 0..4, rt
  seconds). Use `use_kl=True` on GPU for the K-L fast path.
- `is_decision_engine(obj)` — duck-type check.

### `ez_diffusion.py` — the cognitive-mapping engine (1D)
- `ez_recover(choices, rts, evidence, sigma)` — closed-form recovery of
  `drift`, `boundary`, `ndt` from observed behavior (Wagenmakers, van der Maas &
  Grasman, 2007). Instant; this is how an agent reads a peer's style.
- `recover_from_agent_observations(observations, sigma)` — pool several
  evidence levels into one robust style estimate.
- (For the 2D engine, mapping uses amortized SBI instead — see Track B.)

### `comms.py` — DDM-coupled communication + trust (checkpoint 2)
- `social_drift(trust_row, leanings, competences, social_gain)` — a peer's signed
  leaning, weighted by trust×competence and row-normalized, enters the agent's
  accumulator as a bounded social drift.
- `TrustModel` — per-agent trust over peers as a slow, bounded meta-accumulator;
  `set_prior_from_competence` seeds it from the cognitive map, `update` moves it
  on the realized OUTCOME (not on agreement — that's what avoids echo chambers).
- `leaning(choice)` — map a 2-choice decision {0,1} to a signed leaning {-1,+1}.

### `society.py` — the integrated society (checkpoints 3+4)
- `Society` — runs the private→social→outcome round loop; `build_cognitive_maps`
  infers peer competence and seeds trust; `round` adds uncertainty-gated caution
  and trust-weighted deference; `run` reports collective accuracy.
- `SocietyConfig` + `cfg_private` / `cfg_flat` / `cfg_outcome_trust` / `cfg_full` /
  `cfg_rl` — the named conditions the experiments compare.

### `rl.py` — a learned adaptive deference policy (checkpoint 4, RL)
- `DeferencePolicy` — a small linear-Gaussian REINFORCE policy that maps an agent's
  context (own uncertainty, mean trust in peers, peer consensus) to a deference
  multiplier, learning online from whether deferring led to a correct decision.
  Unlike the fixed gate it can learn to *stop* deferring when the whole group is
  untrusted. Pure NumPy, no GPU.

### `experiment.py` — the anchor robustness experiment
- `build_mixed(n_honest, n_adversary, seed)` — a population seeded with
  confidently-wrong adversaries.
- `run_experiment()` — compares the three conditions and prints the headline result.

### `demo.py` — runnable cognitive-mapping demo
- `python -m cognitive_society.demo` — a small society decides and reads each
  other's cognitive styles (speed–accuracy tradeoff + recovered-vs-true params).

---

## Build checkpoints (strict gating — each verified before the next)

1. **Agents decide** — `agent.py` + tests. ✅ done (11 tests).
2. **Agents talk** — custom trust-weighted A2A; *comms based on DDM
   interactions* (peer leaning enters the accumulator; trust as a meta-DDM).
   ✅ done (`comms.py`, 7 tests).
3. **Agents map each other** — `ez_diffusion.py` (1D) / SBI (2D), wired into the
   society loop to seed trust. ✅ done (`society.build_cognitive_maps`).
4. **Agents adapt** — each agent gauges its own uncertainty (how split its private
   decisions are) and raises its boundary / defers more to trusted peers under
   uncertainty. ✅ done (`society.round`, adaptive config; 6 society tests). A
   **learned RL deference policy** (`rl.py`) goes further: it adapts *how much* to
   defer online and can drop deference when the peer group turns hostile. ✅ done
   (`rl.py` + `rl_experiment.py`; 8 RL tests).

Status: checkpoints 1–4 implemented and tested on the 1D path (43 fast tests, no
GPU). SBI-fitted defaults for the 2D engine are Track B (in progress).

---

## Anchor research question

> Does cognitive-map-grounded, uncertainty-gated trust make a society robust to
> confidently-wrong / adversarial agents under novelty — where flat-broadcast or
> outcome-only-trust societies fail?

This is the question that depends on the *novel* machinery (DDM-based theory of
mind + uncertainty-triggered adaptation + cognitive-map-grounded trust), so the
existing collective-DDM literature can't claim the result. Build only what
answers it.

**Result** (`python -m cognitive_society.experiment`, 4 honest + 3 confidently-wrong
adversaries, 40 problems, 30 seeds):

| condition | collective accuracy |
|---|---|
| private (no social) | 68.2% |
| flat broadcast | 74.2% ± 5.8% |
| outcome-only trust | 93.8% ± 3.6% |
| cognitive-map + uncertainty-gated (**ours**) | **97.4% ± 2.1%** |

`flat (74.2%) > private (68.2%)` confirms social info genuinely helps, so `flat` is a
fair baseline (not a strawman). **Honest decomposition of the +23.3pt gain:** most of
it is plain trust-learning (`flat → outcome_trust` = **+19.7pt**); the *novel*
cognitive-map + uncertainty-gating layer adds a real but **modest +3.6pt**
(`outcome_trust → full`, winning in 22/30 seeds, paired Wilcoxon **p = 0.0003** — small
but significant). The full-vs-flat gap grows monotonically with the adversary fraction
— the signature of a genuine adversary-robustness mechanism. An adversarial code review
(independent 30-seed sweep + a blinded-map ablation) confirmed the result is **real,
not a ground-truth-leak artifact**. Reproduce: `python -m cognitive_society.experiment`.

### RL: learned deference (checkpoint 4)

`python -m cognitive_society.rl_experiment` stress-tests a *learned* deference policy
against the hand-tuned gate under a hostile regime shift (an all-honest society where a
majority then flip to confidently-wrong). Post-shift collective accuracy (12 seeds):
no-gating **79.7%** → fixed gate **85.9%** → learned gate (RL) **85.6%**. Honest read:
gating (fixed or learned) is worth **+6pt** here, and the REINFORCE policy *recovers that
entire benefit from reward alone* — matching the expert gate without being told the rule —
but does **not** beat it, and is noisier (±7.7%). The hand-tuned gate was already
near-optimal on this task; a learned policy's edge would show where the fixed rule is
mis-specified — which motivates richer, per-agent models (the decentralized-swarm direction).

---

## Honest novelty boundary

The *basic* society of diverse-threshold DDM agents observing each other's
choices is **already published** (e.g. *Diversity Improves Speed and Accuracy in
Social Networks*, 2020; the networked-DDM line). Our contribution is the
machinery layered on top:
1. a diffusion model used as the generative model inside computational theory of
   mind (inferring *decision dynamics*, not goals);
2. amortized-posterior uncertainty as the novelty/adaptation trigger;
3. trust grounded in an inferred cognitive map, not just outcome history.

The sandbox alone is not a contribution — the inference + trust + adaptation
machinery is.

---

## Running

```bash
# fast society tests (1D, no GPU) — a few seconds
pytest cognitive_society/tests/ -m "not slow" -v

# include the 2D-spatial engine check (invokes JAX; slower)
pytest cognitive_society/tests/ -v
```

No new dependencies beyond the repo's `numpy` (1D path) and the existing JAX
stack (only if you use the 2D `SpatialDDMEngine`).
