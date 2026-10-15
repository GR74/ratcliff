# Execution Spec — Papers 1 & 2 (Grounded Collective Cognition)

**Date:** 2026-07-21
**Basis:** the framework spine (`2026-07-21-grounded-collective-cognition-theory-spine.md`) + committed
code + real-data results. Tags: **[done]** / **[sim]** / **[real]** / **[needed]**.

Two papers off one framework: **Paper 1** introduces the framework and anchors it in **collective
computational psychiatry** (flagship). **Paper 2** applies it to **AI aggregation/routing** (fast,
timely companion). Diagnostics = a later med-AI follow-on; SBI methods + a unification perspective =
optional later.

---

# PAPER 1 — Flagship: a grounded theory of collective cognition and its disorders

**One-line thesis.** *Calibrated confidence is the currency of optimal collective cognition; its
miscalibration is the signature of collective failure; and whether failure is correctable depends on
whether outcomes are observable — validated on real humans, grounded in the diffusion/HGF model, and
applied to psychiatric collective pathology.*

**Target venues.** Nature Mental Health / Nature Human Behaviour / eLife / Computational Psychiatry
(interdisciplinary computational-behavioral).

## Section-by-section outline
1. **Intro** — collective cognition is everywhere; when it works vs fails is unanswered; LLM-based
   human simulation is ungrounded → the case for a *validated-cognition* theory.
2. **Framework** — confidence = precision; the **2×2** (calibration × outcome-observability); the six
   mechanisms. **[done — theory spine]**
3. **Theory / theorems** — the formal boundaries (see Theorem Spec). **[needed]**
4. **Human validation** — confidence is calibrated to accuracy and confidence-weighting beats naive
   pooling, on real people. **[real — Confidence Database]**
5. **Failure & defense** — the delusion cascade (aberrant precision); outcome-trust (corrects) vs
   diversity+robustness (contain); the inversion. **[sim]**
6. **ANCHOR — collective computational psychiatry** — HGF agents parametrized with *published clinical
   disorder parameters* → distinct collective phenotypes; + Ratcliff clinical validation. **[needed:
   HGF component + clinical data]**
7. **Generalization (brief)** — one figure: the same rule improves AI aggregation & medical decisions.
   **[real/sim — from Paper 2 material, condensed]**
8. **Discussion** — misinformation, hybrid human-AI, grounded-vs-LLM simulation, the MoE connection.
9. **Limits** — grounding assumptions; observability axis; scope of the generality claim.

## Figure specs
- **F1 — the 2×2** (schematic + which mechanism wins per cell). **[done conceptually]**
- **F2 — human calibration + WoC** (accuracy-by-confidence curve; confidence-weighted vs majority vs
  oracle across crowd size / across datasets). Source: `real_confidence_replication.py`,
  Double_2017 + replication. **[real]**
- **F3 — the inversion** (confidence-weighting best when calibrated → worst under aberrant precision;
  the dose-response; outcome-trust vs diversity+robustness defenses). Source: `belief_loop.py`,
  `confidence_woc.py`. **[sim]**
- **F4 — FLAGSHIP: disorder → collective phenotype** (each clinical parameter — aberrant precision,
  negative priors, starved social evidence, circumscribed miscalibration — mapped to its collective
  signature, with the defenses). Source: **HGF component (needed)**.
- **F5 — clinical validation** (individual calibration/HGF profile predicts the collective
  distortion). Source: **Ratcliff clinical data (needed)**.
- **F6 — generalization** (AI aggregation + medical decisions, condensed). Source: Paper 2 material.

## What's done vs needed
- **Done:** framework (F1), human validation (F2), the inversion + defenses (F3), all supporting sim.
- **Needed:** (a) HGF component → F4; (b) Ratcliff clinical data → F5; (c) the theorem (§3).

## Honest boundaries (state in-paper)
- The 2×2 generality is a *claim*; validate the cells you can, mark the rest to-be-tested.
- The collective-psychiatry effects are HGF-**simulation** (clinically-parametrized) until F5's clinical
  data; the observability axis is a *feature* of the theory, not a hidden caveat.

---

# PAPER 2 — AI companion: robust aggregation & routing under miscalibration

**One-line thesis.** *Confidence-weighted aggregation/routing is near-optimal when confidence is
calibrated, but inverts catastrophically under miscalibrated or colluding agents; outcome-grounding
(track record) is the robust fix — demonstrated on real crowdsourcing data and directly relevant to
RLHF label quality, model routing, and mixture-of-experts.*

**Target venues.** NeurIPS / ICML / ICLR (main or datasets-&-benchmarks / safety track) / TMLR.

## Section-by-section outline
1. **Intro** — pooling many models/annotators/agents is everywhere in modern AI (RLHF, MoE, ensembles,
   debate); when does confidence-based combination help vs hurt?
2. **Setup** — the confidence/precision aggregation rule + outcome-trust; a light, self-contained
   version of the framework (cite Paper 1 for the full theory).
3. **Label aggregation (RLHF)** — reliability-weighting beats majority on real annotators; a colluding
   wrong-label bloc catastrophically inverts agreement-based Dawid-Skene while outcome-grounding holds.
   **[real — `real_label_aggregation.py`, relevance-2]**
4. **Model routing** — confidence routing beats the best single model; one overconfident junk model
   collapses it; track-record routing is immune. **[sim — `model_routing.py`]**
5. **Mixture-of-experts connection** — self-gating by calibrated confidence; miscalibration/collapse as
   the router-robustness problem. **[sim — `confidence_woc_nd.py`]**
6. **Discussion** — RLHF data poisoning / annotation security; when to trust self-reported confidence;
   the calibration-gated hybrid.
7. **Limits** — semi-real adversarial injection (real workers + synthetic colluders); calibration
   assumptions.

## Figure specs
- **A1 — label aggregation, clean** (majority vs D-S vs track-record vs oracle on real relevance-2).
  **[real]**
- **A2 — the adversarial inversion** (dose-response: D-S → 0.00 past ~38% colluders, track-record →
  1.00). **[real+semi-real]**
- **A3 — model routing** (calibrated: confidence ≈ oracle ≫ best single; +junk model: confidence
  0.93→0.22, track-record robust). **[sim]**
- **A4 — the calibration trap, unified** (confidence-weighting: best when calibrated, worst under
  miscalibration; the 2×2 cell map for AI). **[sim]**

## What's done vs needed
- **Done:** A1–A4 material all exists and is committed + tested. Paper 2 is **write-ready** modulo
  polish + a light framework intro.
- **Needed (optional strengtheners):** a fully-real adversarial dataset (vs injection); the
  calibration-gated hybrid experiment; scale the routing to real LLM outputs.

## Sequencing note
Paper 2's data is **done** → it can finish first. If it does, either post Paper 1 as a preprint to
cite, or give Paper 2 a light self-contained framework intro (don't fully re-introduce the theory —
that's Paper 1's contribution).

---

# HGF COMPONENT SPEC (feeds Paper 1, F4/F6)

**Goal.** Replace/augment the toy accumulator with a **hierarchical Gaussian filter** agent so agents
carry clinically-validated belief-updating, enabling the disorder → collective-phenotype map.

**Reference.** Mathys et al. 2011 (Front. Hum. Neurosci.); Mathys et al. 2014; the TAPAS toolbox;
`pyhgf` (Python HGF) is a usable implementation. Disorder parameters: psychosis → aberrant
precision/high-level volatility (Powers, Corlett; Adams et al. 2013); anxiety → aberrant learning-rate/
uncertainty (de Berker); depression → negative priors.

**Interface (the plug).** The HGF agent must emit **(belief, precision)** per update — the *same*
interface `belief_loop.py` already consumes (belief + precision in/out). Nothing else changes.

**Deliverables / milestones.**
1. **Single-agent HGF** — reproduces standard HGF behavior on a volatility/probability-tracking task
   (sanity vs published HGF). **[M1]**
2. **Robustness** — drop HGF agents into `belief_loop.py` / `confidence_woc.py`; show the
   confidence-weighting / loop results **survive** with HGF agents (framework is model-agnostic). **[M2]**
3. **Disorder → collective phenotype (F4)** — parametrize sub-populations with published disorder
   parameters; produce the phenotype map: aberrant precision → mass overconfidence/delusion; negative
   priors → collective pessimism lock-in; starved social evidence → fragmentation; circumscribed
   miscalibration → monothematic collective delusion; each with the defenses. **[M3]**
4. **(Later) clinical hook** — HGF parameters fit to real clinical data (Ratcliff) → F5. **[M4]**

Scope: M1–M3 are **sim-based / clinically-parametrized** (no clinical data needed) → achievable now.

---

# THEOREM SPEC (feeds Paper 1, §3)

Prove (or characterize) the 2×2 boundaries:
1. **Optimality.** Confidence-weighting = the inverse-variance (BLUE) aggregator *iff* confidence ∝
   precision (calibration). (Empirically ≈ oracle; prove the equivalence + the exact condition.)
2. **Inversion threshold.** The fraction × severity of biased-overconfident agents at which
   confidence-weighting becomes *worse* than naive averaging (the top→bottom-row boundary).
3. **Correction bound.** With outcome-observation rate ρ, outcome-trust attains error ≤ f(ρ); the
   miscalibrated+observable cell is rescuable for ρ above a threshold.
4. **Containment bound.** Homeostatic diversity + robust-precision cap the achievable aggregate bias to
   g(cluster fraction, cap) with no outcomes (the danger-zone floor).

---

# DATA & CODE ASSETS (tagged)

**Real data (in hand):** Confidence Database (calibration + WoC) [real]; crowd-kit relevance-2
(aggregation + adversarial inversion) [real/semi-real].
**Real data (via collaboration):** clinical/decision data for F5 [needed].
**Code (committed, tested):** `confidence_woc`, `confidence_woc_ddm`, `belief_loop`, `confidence_woc_nd`,
`model_routing`, `label_aggregation`, `real_confidence_replication`, `real_label_aggregation`; plus the
swarm infra (`society/comms/topology/adversary/baselines/phase2–4`) and Track-B SBI (`model_a/model_b`).
**To build:** the HGF component; the theorem; (optional) fully-real adversarial dataset; calibration-
gated hybrid.

---

# SEQUENCING

1. **Paper 2 first** — data done; write + polish + light framework intro → submit (fast win).
2. **Paper 1 in parallel** — build HGF (M1–M3) → F4; draft framework + human-validation + inversion
   sections (all [done]/[real]); prove the theorem; slot Ratcliff clinical data → F5 when ready.
3. **Diagnostics (human+AI fusion)** — follow-on paper, once clinical data + a clear result exist.
4. **SBI methods paper + unification perspective** — optional, later.

**Key milestone for the flagship:** F4 (HGF disorder→phenotype, sim) makes Paper 1 *submittable* even
before F5; F5 (clinical) is what pushes it to the top venues.
