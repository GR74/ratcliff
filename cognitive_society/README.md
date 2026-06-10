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

---

## Build checkpoints (strict gating — each verified before the next)

1. **Agents decide** — `agent.py` + tests. ✅ done (11 tests, 1.65s).
2. **Agents talk** — custom trust-weighted A2A; *comms based on DDM
   interactions* (peer state enters the accumulator; trust as a meta-DDM).
   Design panel in progress.
3. **Agents map each other** — `ez_diffusion.py` (1D) / SBI (2D). ✅ engine done
   (7 tests, 0.25s); wiring into the society loop is checkpoint 3 proper.
4. **Agents adapt** — SBI-fitted + RL-tuned defaults; uncertainty-triggered
   adaptation to novel problems; uncertainty gates self-vs-social reliance.

Status: checkpoints 1 + the mapping/engine primitives are built and tested.

---

## Anchor research question

> Does cognitive-map-grounded, uncertainty-gated trust make a society robust to
> confidently-wrong / adversarial agents under novelty — where flat-broadcast or
> outcome-only-trust societies fail?

This is the question that depends on the *novel* machinery (DDM-based theory of
mind + uncertainty-triggered adaptation + cognitive-map-grounded trust), so the
existing collective-DDM literature can't claim the result. Build only what
answers it.

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
# fast society tests (1D, no GPU) — runs in ~2s
pytest cognitive_society/tests/ -m "not slow" -v

# include the 2D-spatial engine check (invokes JAX; slower)
pytest cognitive_society/tests/ -v
```

No new dependencies beyond the repo's `numpy` (1D path) and the existing JAX
stack (only if you use the 2D `SpatialDDMEngine`).
