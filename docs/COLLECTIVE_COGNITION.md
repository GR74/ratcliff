# Collective Cognition — full documentation

A grounded, validated framework for **when many minds (or AI agents) are smarter together, when
they're dumber, and how to tell and fix which.** Built on a validated cognitive model (the diffusion
/ hierarchical Gaussian filter), not an LLM. This document catalogs every module, result, and doc.

**Design docs:** the vision (`notes/2026-07-21-cognitive-digital-twin-research-program.md`), the
framework spine (`notes/2026-07-21-grounded-collective-cognition-theory-spine.md`), the paper specs
(`notes/2026-07-21-papers-1-and-2-spec.md`).

Everything below is **pure NumPy**, runs on a laptop in seconds, is **committed + tested**, and is
tagged **[real]** (real human/LLM data) / **[sim]** (simulation) / **[semi-real]** / **[pending]**.

---

## 1. The theory (the 2×2)
Collective cognition is governed by two axes, which *predict* which mechanism wins:

| | **outcomes observable** | **no outcomes** |
|---|---|---|
| **confidence calibrated** | *optimal achievable* — confidence-weighting works + outcome-trust available | *good but fragile* |
| **miscalibrated** | *rescuable* — outcome-trust corrects | *danger zone* — confidence-weighting inverts; only containment |

**Currency:** confidence = precision = the weight on evidence. **Mechanisms:** (1) DDM/HGF agent →
belief + calibrated confidence; (2) confidence-weighting (near-optimal pooling); (3) delusion cascade
(aberrant precision); (4) outcome-trust (corrects, needs outcomes); (5) homeostatic diversity +
robust-precision (contain, no outcomes); (6) the iterated loop + n-D confidence-routing (MoE).

### The theorem (proven + numerically verified) [real math]
For a crowd where a fraction **f** are biased by **b** with reported weight **w_d** and the rest are
unbiased with weight **w_h**, the confidence-weighted estimator's bias is
`B_conf = b·f·w_d / ((1-f)·w_h + f·w_d)` vs uniform `B_unif = b·f`. Hence **confidence-weighting is
more biased than naive averaging iff `w_d > w_h`** (overconfidence), and as f→0 a tiny overconfident
minority is amplified by **κ = w_d/w_h**. Verified numerically (empirical bias matches to 3 digits).

---

## 2. Core mechanism modules (`cognitive_society/`)

### `confidence_woc.py` — Brick 1: confidence-weighted wisdom of crowds  [sim]
Spectrum estimation: agents accumulate noisy evidence about a continuous truth, emit (estimate,
confidence = precision). **Result:** confidence-weighting captures **~95% of the oracle's** gain over
simple mean *without knowing variances*; **outcome-trust** holds 97–100% where confidence collapses
under a spreading delusion; delusion dose-response. Run: `python -m cognitive_society.confidence_woc`.

### `confidence_woc_ddm.py` — Brick 1 grounded in the REAL DDM  [sim]
Uses the validated binary `DDMAgent`. **Result (honest):** raw **RT-confidence is anti-calibrated**
(pt-biserial −0.20) and *hurts* the crowd; the DDM's **evidence-state confidence** (internal decision
consistency) IS calibrated (+0.24) and gives the **best** crowd accuracy (93.2%, +3.6pt over
majority). Lesson: confidence in the real DDM is the balance of evidence, not the clock.

### `belief_loop.py` — Brick 2: the iterated belief↔evidence↔confidence loop  [sim]
Active-inference: agents on a Watts–Strogatz graph exchange beliefs + precisions and update via
precision-weighted Bayesian revision. **Results:** a deluded cluster **propagates** into a
confident-but-wrong echo chamber (honest RMSE 0.03→0.66); **outcome-trust cures** it (→0.047); naive
iteration **double-counts precision** (overconfidence) — fixed by an information bound; the
**outcome-free defense** (robust-precision + homeostatic diversity, the RSL set-point principle) cuts
the delusion's grip ~63% (0.66→0.24) with no outcomes and no cost on healthy crowds.

### `confidence_woc_nd.py` — Brick 3: n-dimensional beliefs + division of labor  [sim]
Belief is a vector; per-dimension confidence routes each dimension to its expert. **Result:** a crowd
of narrow specialists (each expert on 2 of 8 dims) **beats the best single generalist by 87%**
(0.058 vs 0.458 RMSE) — a **self-gating mixture-of-experts** where calibrated confidence is the
router. A *circumscribed* delusion (one dimension) is isolated per-dim by outcome-trust.

---

## 3. AI companion modules

### `model_routing.py` — confidence vs track-record model routing  [sim]
The LLM-router setting. **Result:** with calibrated specialist models, **confidence routing ≈ oracle
and beats the best single model** (0.93 vs 0.54); add ONE weak-but-overconfident "junk" model and
**confidence routing collapses** (0.93→0.22, worse than random) while **track-record routing is
immune** (0.93). The calibration trap for LLM routers.

### `label_aggregation.py` — RLHF / crowdsourcing label aggregation  [sim]
**Result:** with calibrated annotators, confidence-weighting beats majority (0.94 vs 0.81) ≈ oracle;
add confident-but-careless annotators and confidence-weighting is **poisoned below majority**
(0.94→0.65) while gold-grounded track-record stays robust. The training-data calibration trap.

### `router_bench.py` — real-LLM routing on RouterBench  [pending]
Validates model routing on **real frontier-LLM outputs** (RouterBench). Being produced by a separate
agent; lands in a follow-up commit.

---

## 4. Real-data analyses (the validation)

### `real_confidence_replication.py` — Confidence Database (Rahnev et al.)  [real]
Real human confidence + accuracy. **Result:** calibration **replicates** — corr(confidence, accuracy)
positive on every parsed dataset (mean **+0.40**); on a shared-item crowd (Double 2017, Raven's,
33 subj × 20 items) **confidence-weighting beats majority by +5.8pt** and matches the accuracy-oracle
*without knowing accuracy.* The premise, validated on real people.

### `real_label_aggregation.py` — crowd-kit `relevance-2` (7,138 real workers)  [real / semi-real]
**(1) Clean data:** reliability-weighting beats majority; agreement-based Dawid-Skene is strong
*absent adversaries* (exactly as the 2×2 predicts). **(2) Adversarial injection** (a colluding bloc
agrees on the WRONG label): agreement-based **Dawid-Skene catastrophically inverts** past ~38%
colluders (locks onto the false consensus → 0.000 accuracy), while **outcome-grounded track-record**
stays robust and even benefits (→1.000). The outcome-vs-agreement distinction (the swarm f-local
inversion) on real data with the field-standard tool = the **RLHF label-poisoning** problem. *(2) is
semi-real: real workers + synthetic perfectly-consistent colluders; the direction and the inversion
threshold are the finding.*

---

## 5. Related infrastructure (earlier work, in-repo)
- **Swarm** (`society, comms, topology, adversary, baselines, phase2–4`) — outcome-grounded trust
  isolates colluding clusters past the f-local bound where agreement-based SOTA inverts; per-victim
  AUC 1.00 vs agreement 0.00 at majority-capture; Phase-4 collective result (with an honest
  self-correction of an over-claim). See `notes/2026-06-10-phase2-swarm-result.md`.
- **Track-B SBI** (`model_a`, `model_b`, `scripts/sbi_mvp.py`) — amortized simulation-based inference
  for the 2D Ratcliff diffusion, validated at scale (N=3000, calibrated c2st≈0.5). The fast JAX
  simulator port. See `notes/2026-06-10-track-b-sbi-validation.md`.
- **Web interface** (`backend/`, `frontend/`) — the FastAPI + React app driving the 2D model.

---

## 6. Reproduce
```
python -m cognitive_society.confidence_woc            # Brick 1
python -m cognitive_society.confidence_woc_ddm        # Brick 1 (real DDM)
python -m cognitive_society.belief_loop               # Brick 2 (the loop + defenses)
python -m cognitive_society.confidence_woc_nd         # Brick 3 (division of labor / MoE)
python -m cognitive_society.model_routing             # AI: model routing
python -m cognitive_society.label_aggregation         # AI: label aggregation
python -m cognitive_society.real_confidence_replication   # REAL: Confidence Database
python -m cognitive_society.real_label_aggregation        # REAL: crowd-kit (needs: pip install crowd-kit)
```
Tests (this env has a broken pytest-plugin; disable autoload):
```
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; python -m pytest cognitive_society/tests -q
```

## 7. Honest boundaries (kept loud)
- The novel *collective* claims are **[sim]** / **[semi-real]** except the human premise; the
  distinctive collective validation (clinical) is future work.
- The individual mechanisms (confidence-weighting, WoC, MoE, active inference, Dawid-Skene) are
  **known**; the contribution is the **unified 2×2 + the grounding + real-data validation + the
  theorem + the failure/defense characterization.**
- Outcome-trust needs **observable outcomes** — that observability axis is a *feature* of the theory
  (it decides correct-vs-contain), not a hidden caveat.
