# Grounded Collective Cognition — theory spine + paper skeleton

**Date:** 2026-07-21
**Status:** Consolidation of Bricks 1–3 + the real-data validation into a single framework + paper plan.
**Companion:** the vision doc `2026-07-21-cognitive-digital-twin-research-program.md`; the mechanism
code is `cognitive_society/{confidence_woc, confidence_woc_ddm, belief_loop, confidence_woc_nd,
real_confidence_replication}.py`.
**Discipline:** every claim below is tagged **[real]** (validated on real human data),
**[sim]** (simulation only), or **[theory]** (to be proven / aspirational). State the theory only
as big as the evidence supports.

---

## 0. The one-sentence thesis
> **Calibrated confidence is the currency of optimal collective cognition; its miscalibration is the
> signature of collective failure; and whether that failure is *correctable* depends on whether
> outcomes are observable.** Built on a *validated* cognitive model (the diffusion / hierarchical
> Gaussian filter), not an LLM.

## 1. The central claim — the 2×2
Collective cognition is governed by two axes, and the framework *predicts* which mechanism wins in
each cell:

| | **outcomes observable** | **no outcomes** |
|---|---|---|
| **confidence calibrated** | **Optimal achievable** — confidence-weighting works *and* outcome-trust available *(diagnostics, forecasting)* | **Good but fragile** — works, no correction if it drifts *(expert panels)* |
| **miscalibrated** | **Rescuable** — outcome-trust *corrects* the overconfident *(diagnostics w/ bad readers; human+AI)* | **Danger zone** — confidence-weighting *inverts*; only *containment* left *(misinformation, echo chambers, mass delusion)* |

- The **top-left** is where you make a crowd near-optimal.
- The **bottom-right** is where societies fail (and where *only* diversity/robustness help).
- **Mental health / computational psychiatry** = the special case where the miscalibration is
  *endogenous and clinically parametrized* — a disorder is a specific arrow driving a population
  into the bottom row.

## 2. The mechanisms (one currency: confidence = precision = weight-on-evidence)
1. **Agent** — a drift-diffusion (or HGF) decision-maker emitting a belief **+ calibrated
   confidence**. The grounding. **[real: DDM validated 45y; HGF is the comp-psych standard]**
2. **Confidence-weighting** — pool by confidence; near-optimal because confidence tracks accuracy.
   **[real]**
3. **Delusion cascade** — biased + overconfident agents (aberrant precision) propagate a
   confident-but-wrong consensus; confidence-weighting is hijacked. **[sim]**
4. **Outcome-trust** — weight by *realized* accuracy from revealed outcomes; corrects error. Needs
   observable outcomes. **[sim; the real-data version is the diagnostics anchor]**
5. **Outcome-free defenses** — *homeostatic diversity* ("always doubters", from RSL) + *robust
   precision* (cap claimed confidence); together *contain* (~63% grip reduction) but don't *correct*.
   **[sim]**
6. **Structure** — the iterated **belief loop** (active inference) and **n-dimensional
   confidence-routing** (a self-gating **mixture of experts**: division of cognitive labor). **[sim]**

## 3. The evidence we already have
**[real] — validated on human data (Confidence Database, Rahnev et al.):**
- **Calibration replicates:** corr(confidence, accuracy) positive on every parsed dataset
  (Double +0.53, Pereira +0.72, Desender +0.17, Mei +0.20; mean +0.40). The premise holds in humans.
- **Confidence-weighting beats majority** on a real shared-item crowd (Double 2017, Raven's,
  33 subj × 20 items): majority 0.81 → confidence-weighted **0.87 (+5.8pt)**, matching the accuracy
  **oracle** *without knowing accuracy*. *Honest gap: the direct WoC test is still N=1 shared-item
  dataset — most confidence datasets are perceptual (no shared items). Needs more reasoning/decision
  datasets → the diagnostics anchor supplies this at high stakes.*

**[sim] — mechanism results (all tested, committed):**
- Brick 1: confidence-weighting captures **95%** of the oracle's gain; **outcome-trust** holds
  97–100% where confidence collapses under delusion; **real-DDM** grounding shows RT-confidence is
  *anti*-calibrated (fails) but the DDM **evidence-state** confidence is calibrated and best (93.2%).
- Brick 2: delusion propagates (honest RMSE 0.03→0.66); **outcome-trust cures** (→0.047); the
  **outcome-free stack** (robust-precision + homeostasis) contains (→0.24, ~63% cut, free on healthy
  crowds); precision double-counting overconfidence fixed by an information bound.
- Brick 3: **division of cognitive labor** — specialists beat the best individual by **87%** via
  confidence-routing; circumscribed (single-dimension) delusion isolated by per-dim outcome-trust.

## 4. The theorems to prove (upgrade "framework" → "theory") **[theory]**
1. **Optimality:** confidence-weighting = the inverse-variance (BLUE) aggregator *iff* confidence ∝
   precision (calibration). (Empirically ≈ oracle; prove the equivalence + the calibration condition.)
2. **Inversion threshold:** characterize the fraction × severity of biased-overconfident agents at
   which confidence-weighting becomes *worse* than naive averaging (the top→bottom-row boundary).
3. **Correction bound:** with outcome-observation rate ρ, outcome-trust achieves error ≤ f(ρ); the
   miscalibrated+observable cell is rescuable for ρ above a threshold.
4. **Containment bound:** homeostatic diversity + robust-precision cap the achievable aggregate bias
   to g(cluster fraction, cap) with *no* outcomes — quantifying the danger-zone floor.

## 5. Positioning / honest novelty boundary
- **vs LLM human-simulation (Simile/Park):** grounded, calibrated, mechanistic vs broad but
  ungrounded. Ours is the *domain-focused, trustworthy* alternative where grounding is decisive.
- **vs computational psychiatry:** a **level-shift** — the field models the *single brain*; we model
  the *collective* using the same clinically-validated disorder parameters (HGF). Novel.
- **vs mixture-of-experts:** a *self-gating* MoE where **calibrated confidence replaces the learned
  router**, decentralized.
- **vs wisdom of crowds / Bayesian committee:** the known parts are confidence-weighting and
  inverse-variance pooling; the contribution is the **unified 2×2 theory + the DDM/HGF grounding +
  real-data validation + the failure/defense characterization.** Claim only at that resolution.

## 6. Anchor domains (points in the 2×2)
- **Diagnostics (THE anchor — top-left + bottom-left; outcomes observable).** Multi-reader radiology
  /pathology, tumor boards, human+AI fusion. *Because outcomes are observable, the strong defense is
  deployable.* Real multi-reader datasets with per-reader confidence + biopsy ground truth exist →
  run the same test on clinicians. High impact, validatable, perfect fit.
- **Forecasting (top-left; clean public data).** Superforecasting / prediction markets — confidence
  -weighted analyst aggregation; outcomes resolve.
- **Misinformation / echo chambers (bottom-right; the danger zone).** Containment-only; platform
  design falls out of the defense results.
- **Collective computational psychiatry (the endogenous-miscalibration arrow).** The novel science
  prong; the flagship figure = disorder parameter → collective phenotype.

## 7. Paper skeleton
1. **Intro** — collective cognition is everywhere; when does it work vs fail; the grounding gap
   (LLM simulation can't answer it).
2. **Framework** — confidence-as-precision; the 2×2; the mechanisms.
3. **Theory** — the four theorems (§4).
4. **Human validation** — calibration + confidence-weighting across Confidence-Database datasets.
5. **Failure & defense** — the delusion inversion; outcome-trust (correct) vs diversity+robustness
   (contain); the cells.
6. **Anchor: diagnostics** — real multi-reader data; confidence/outcome-weighted aggregation reduces
   error (incl. human+AI fusion). *The high-impact result.*
7. **Discussion** — computational psychiatry level-shift; misinformation; grounded-vs-LLM simulation;
   the MoE connection.
8. **Limits** — grounding assumptions; observability requirement; scope of the generality claim.

## 8. Honest limitations (kept loud)
- The **general** 2×2 is a *claim*; we validate the cells we can (top-left on real humans; the
  danger-zone inversion in sim) and state the rest as to-be-tested.
- The novel *collective* effects are **[sim]** until real diagnostic / clinical / misinformation data
  confirm them — the diagnostics anchor is the first such test.
- Outcome-trust requires **observable outcomes**; the theory is explicit that this is the axis that
  decides correct-vs-contain, so it is a feature of the theory, not a hidden caveat.

## 9. Team & next steps
- **You:** framework + the human/diagnostics validation.
- **Friend (HGF):** swap DDM agents for hierarchical Gaussian filter agents → clinically-parametrized
  disorders → the collective-psychiatry prong; also robustness (result survives with HGF agents).
- **Ratcliff:** the DDM/confidence grounding, credibility, and access to clinical/decision data for
  the tier-defining collective validation.
- **Immediate:** (a) this spine [done]; (b) find a real multi-reader diagnostic dataset + run the
  aggregation test (the anchor); (c) more shared-item datasets to firm up the human WoC result;
  (d) start the inversion-threshold theorem.
