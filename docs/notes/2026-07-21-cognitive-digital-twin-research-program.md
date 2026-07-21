# Cognitive Digital Twins — a grounded simulator of human belief and decision-making

**Date:** 2026-07-21
**Status:** Direction endorsed by Roger Ratcliff ("go full ahead"). Synthesized from the
Ratcliff meeting + the Simile/Percy-Liang "age of simulation" framing.
**Internal shorthand:** *mind model* (the dual of a world model).
**Outward name:** *cognitive digital twin*.

> **One line.** Build a grounded simulator of human belief and decision-making, where each
> person is a *validated cognitive model* (not an LLM), confidence is the universal currency,
> and the target application is **computational psychiatry at population scale**.

This document is the north-star architecture + the honest novelty boundary + the concrete
first result to build. It is deliberately explicit about what is *validated*, what is
*aspirational*, and what is *ours to own vs already in the literature*.

---

## 1. Motivation — the simulation frontier, and the gap

Percy Liang / Simile ("Simulation: The Next Frontier for AI", Mar 2026) argue AI's next era
after *prediction* and *reasoning* is **simulation**: high-fidelity, causal models of people and
societies you can run forward to answer "what if?" — interventions and counterfactuals (Pearl's
ladder). They name three unsolved problems: (1) **high-fidelity models of people** —
*"today's language models do not capture the nuances of human behavior"*; (2) **large-scale
efficiency**; (3) **trust/calibration** — *"calibrated probability estimates over the distribution
of outcomes."*

Their proposed fix is *more/better LLM data*. **Our thesis is different and, we argue, more
grounded: put a validated cognitive *mechanism* and a real *environment* underneath the persona,
and use the LLM as one layer, not the engine.** This is the classic **symbol-grounding**
(Harnad) / **embodied-cognition** critique of LLMs, answered constructively.

An LLM alone is a disembodied symbol-shuffler: it has **semantic** sense (language, knowledge)
but no **physical** sense (it isn't situated in a world with data) and no **mental/cognitive**
sense (it doesn't actually perceive evidence, accumulate it, decide, or form calibrated
confidence — it pattern-matches text *about* deciding). A grounded persona needs all three.

---

## 2. The architecture — three layers, one loop

A person is three grounded layers wrapped in a feedback loop:

| Layer | What it is | Grounds |
|---|---|---|
| **Physical** | environment / world data → the actual situation | *"in an actual world"* |
| **Cognitive (the engine)** | a **drift-diffusion model** — accumulate evidence → graded belief + confidence | *"an actual person"* — the mechanism |
| **Semantic** | an **LLM / knowledge representation** whose job is to *generate the evidence and priors* the DDM consumes | language, knowledge, context |

**The keystone insight.** The DDM's "evidence" was always a *placeholder*. Ratcliff feeds the
model an abstract drift rate; in a real person, evidence is *generated when the belief system
meets the world*. So the system is a loop, not a pipeline:

```
beliefs / priors ──▶ interpret the world ──▶ evidence (drift) ──▶ DDM accumulates
       ▲                                                              │
       └──────────── belief update ◀──── decision + confidence ◀──────┘
                                    ▲
             other agents' decisions + confidences  ──(more evidence)──┘
```

Two consequences fall out for free:
- **The LLM/knowledge layer gets a principled job:** *generate the evidence + priors*, not "talk."
- **Social interaction is captured automatically:** peers' decisions+confidences are simply *more
  evidence entering the same loop*. The society is not bolted on; social influence *is* evidence.

This loop is formally **active inference / predictive processing** (the Bayesian brain — Friston;
Clark) — but reached from the *validated-model* side (the DDM is the accumulation-to-commitment
layer of it), which is exactly why it is grounded rather than philosophical.

---

## 3. The engine — DDM, and confidence as the universal currency

Each agent is a drift-diffusion decision-maker (Ratcliff 1978; Ratcliff & McKoon 2008): the
continuous-time limit of the SPRT, i.e. *optimal* evidence integration. It accumulates noisy
evidence to a boundary → a choice, a reaction time, and — critically — a **confidence**.

**Confidence is read out of the diffusion, not added on** (Ratcliff & Starns RTCON2, 2013;
Pleskac & Busemeyer 2DSD, 2010; balance-of-evidence): fast, far-past-threshold decisions are more
confident, and confidence criteria partition the evidence axis into graded regions. This is what
"confidence time is boundaries" means.

**Confidence = precision = the universal weight.** One quantity appears at *every* junction of
the loop:
- how much your **prior** belief weighs (incoming precision),
- how sure your **decision** is (DDM readout),
- how much you weight **others** (wisdom-of-crowds pooling).

One variable, three roles. This is what unifies the confidence-DDM, wisdom-of-crowds, and
trust/competence threads into a single mechanism.

---

## 4. Wisdom of crowds — the first concrete, provable claim

Classic wisdom of crowds (Galton 1907; Surowiecki 2004) says *average* the crowd. The modern
refinement (Bahrami et al. "Optimally interacting minds", 2010; Becker, Brackbill & Centola,
PNAS 2017; the accuracy-weighting result) says the crowd gets *much* better if you **weight by
accuracy** — but you usually can't observe accuracy.

> **In a DDM society you can — because DDM confidence is calibrated to accuracy by construction.**
> So confidence-weighted aggregation is *provably-better* wisdom of crowds, not a heuristic: the
> weights track truth.

That is the headline of the first result (§8), and it is positioned directly against known theory
(Becker–Centola; the "social power ∝ accuracy" optimality result), so it does not float.

---

## 5. From binary to spectrum, and to n-dimensional

Ratcliff's steer: move from **binary** (positive/negative, 2-choice) to **spectrum** beliefs —
graded positions on a scale, with confidence = certainty of the position. The accumulator's
landing region *is* the graded belief; confidence *is* its certainty. Then generalize the
belief-position from a scalar to a **vector** → **n-dimensional belief spaces**. The mechanism is
unchanged; only the dimensionality of the accumulated quantity grows.

---

## 6. The application — computational psychiatry at population scale

This is the killer app and the reason the whole thing is grounded rather than a toy. Because the
architecture is the belief↔evidence↔confidence loop, psychiatric conditions become **specific,
named knobs** in that loop (this is an established field — Montague et al. 2012; Friston/Stephan;
Adams, Stephan, Brown, Frith & Friston 2013 "computational anatomy of psychosis"; Corlett &
Fletcher):

| Condition | Broken part of the loop |
|---|---|
| **Psychosis** | aberrant **precision** (evidence weighted wrong) |
| **Depression** | negative **priors** (belief system pre-biases the evidence) |
| **Anxiety** | intolerance of uncertainty (confidence dynamics) |
| **Social isolation (e.g. COVID)** | starve the loop of **social evidence** → beliefs drift, precision breaks |

**The experiment:** run a population of these agents, turn *one* knob, and watch collective belief
distort — polarization, echo chambers, loss of crowd wisdom. Grounded, calibrated, mechanistic —
and it directly fits a pre-med + computational profile.

---

## 7. Positioning & honest novelty boundary

**Every piece already exists:** the DDM (Ratcliff), confidence-DDM (Ratcliff & Starns; 2DSD),
wisdom of crowds (Galton; Bahrami; Becker–Centola), active inference (Friston; Clark),
computational psychiatry (Montague; Adams et al.), generative agents / digital twins of people
(Park et al. 2023; Park et al. "1,000 people", 2024; Simile), networked DDM (Srivastava & Leonard
2014), amortized SBI for DDMs (our Track B; Fengler et al. LANs 2021).

**What is ours to own is the fusion:**
> a *validated, calibrated cognitive process model* (the DDM) as the engine, **confidence-as-
> precision** as the universal weight, the belief↔evidence↔confidence loop closed as active
> inference, wrapped into a *groundable persona* (environment + LLM as evidence-generator), aimed
> at **population-scale mental-health simulation**.

Novelty lives in the **synthesis and the grounded implementation**, not in any single part. This
is a defensible, fundable, Ratcliff-backed place to stand — *provided* we keep claiming at that
resolution and do not overclaim any single ingredient as new. (This is the same honesty discipline
that carried the swarm work: report the boundary loudly.)

---

## 8. The build path (honest sequencing)

The buildable core is the **cognitive layer**; everything else wraps it. Do not build the
LLM/environment layers first — build the mind, with clean evidence-in / confidence-out interfaces
that the other layers plug into.

**Brick 1 — Confidence-weighted wisdom of crowds (NOW, days, pure NumPy, existing repo).**
- Extend the DDM agent to emit **(belief-position on a scale, confidence)**, confidence derived
  from the diffusion (decision time / evidence magnitude / confidence-criteria regions).
- Add **confidence-weighted pooling** to the society (alongside existing trust/competence).
- **Test** on spectrum-estimation problems (continuous truth): compare **simple average vs median
  vs confidence-weighted**. Claim: confidence-weighting wins, and — because DDM confidence tracks
  accuracy — it is principled, not a heuristic.
- **Then break calibration** (over/under-confident sub-populations) and watch the crowd fail →
  *this is already the computational-psychiatry hook.*

**Brick 2 — Close the loop.** Belief-update from decision+confidence; peers-as-evidence; the
active-inference formulation. Move from one-shot pooling to the iterated loop.

**Brick 3 — Spectrum → n-dimensional beliefs** (vector-valued accumulation).

**Brick 4 — The environment (physical) layer** — structured world/situation data feeding evidence.

**North star — The semantic layer.** LLM/knowledge as the evidence-and-prior generator → full
persona → a specific study (e.g., isolation → belief drift), i.e. a cognitive digital twin.

---

## 9. First-result spec (Brick 1, in detail)

**Files:** extend `cognitive_society/agent.py` (confidence readout) + `comms.py` /`society.py`
(confidence-weighted pooling); new experiment `cognitive_society/confidence_woc.py`; tests.

**Agent output:** `(belief ∈ [-1,1] or graded scale, confidence ∈ [0,1])`. Start simple:
confidence ∝ a monotone function of |accumulated evidence| and/or 1/decision-time, validated to be
**calibrated** (higher confidence ⇒ higher accuracy) before use.

**Aggregators to compare:** (a) simple mean (classic WoC), (b) median, (c) confidence-weighted
mean, (d) — later — confidence² / precision-weighted.

**Task:** a continuous latent truth per problem; each agent gets a noisy, drift-scaled read; the
crowd estimates the truth. Metric: error of the aggregate vs truth, across difficulty + crowd size.

**Predicted result (to be tested, reported honestly either way):** confidence-weighting reduces
aggregate error over simple mean/median, and the gap grows with heterogeneity in agent competence
(because confidence correctly demotes the weak/uncertain agents). Sanity control: with *homogeneous*
agents, all aggregators should coincide.

**The psychiatry follow-on (same file):** inject a sub-population with miscalibrated confidence
(gain > 1 = overconfident/"manic"; gain < 1 = underconfident/"depressive"; noise = "psychotic
precision") and show confidence-weighting *inverts* from asset to liability — the crowd is captured
by the overconfident-but-wrong. That is the population-mental-health result in miniature.

**Honest caveats to keep loud:** DDM-confidence-tracks-accuracy holds *only when calibrated* — so
the calibration check is load-bearing; toy substrate; the "provably-better" claim is conditional on
calibration; confidence readout choice (time vs magnitude vs criteria) is a modeling decision to
justify against Ratcliff & Starns.

---

## 10. Open questions / risks

- **Which confidence readout** matches human data best (RT-based vs balance-of-evidence vs 2DSD)? —
  ask Ratcliff; validate against his confidence-DDM.
- **Does confidence-weighting survive miscalibration**, and can the society *detect* miscalibration
  (meta-confidence / outcome-trust as a corrector)? — bridges to the earlier outcome-vs-agreement
  trust work.
- **Identifiability at n-dimensions** — our SBI (Track B) already shows some params are weakly
  identified; the spectrum/n-D extension must be checked with SBC.
- **Scope discipline** — the LLM/persona layer is the north star, not the first build; resist
  building it before the cognitive core is solid and calibrated.

---

## 11. Lineage / references

DDM: Ratcliff (1978); Ratcliff & McKoon (Neural Comp. 2008). Confidence-DDM: Ratcliff & Starns
(Psych Rev. 2013, RTCON2); Pleskac & Busemeyer (2010, 2DSD). Wisdom of crowds: Galton (1907);
Surowiecki (2004); Bahrami et al. (Science 2010); Becker, Brackbill & Centola (PNAS 2017). Active
inference / predictive processing: Friston (2010); Clark (2013); Parr, Pezzulo & Friston (2022).
Computational psychiatry: Montague et al. (2012); Adams, Stephan, Brown, Frith & Friston (2013);
Corlett & Fletcher. Simulation-of-humans / generative agents: Park et al. (2023); Park et al.
"1,000 people" (2024); Liang / Simile (2026). Networked DDM: Srivastava & Leonard (2014). Our prior
work: `cognitive_society/` (the society + swarm), Track B amortized SBI (`docs/notes/2026-06-10-
track-b-sbi-validation.md`), the 2D Ratcliff-diffusion port (`model_a`/`model_b`, the web interface).
