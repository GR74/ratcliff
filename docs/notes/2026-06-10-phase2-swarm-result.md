# Phase 2 — Decentralized swarm: the per-victim isolation result

**Date:** 2026-06-10
**Code:** `cognitive_society/{adversary,baselines,swarm_metrics,phase2_experiment,phase2_per_victim}.py`
**Reproduce:** `python -m cognitive_society.phase2_experiment` (headline harness) ·
`python -m cognitive_society.phase2_per_victim` (the demonstrated result)

This note records the Phase-2 result honestly, including the path from an apparent
null to a clean demonstration — because *how* we got there is the point.

---

## The claim under test

> **Outcome-grounded trust isolates a contiguous, confidently-wrong, mutually-agreeing
> adversary CLUSTER that AGREEMENT-based local trust (W-MSR, A-RepC, bounded-confidence)
> provably cannot — with no global controller, on a local topology.**

The decisive regime (pre-specified in `2026-06-10-swarm-coordination-research-synthesis.md`):
a *contiguous* cluster of confidently-wrong agents that *agree with each other*, sized so
that at least one honest node's neighbourhood is **adversary-majority** — past the
per-neighbourhood bound that median/agreement filters can survive.

## What was built (faithful, not strawman)

- `baselines.py` — **W-MSR** (LeBlanc/Zhang/Koutsoukos/Sundaram 2013: own-value trimming,
  keep-self, stubborn worst-case adversaries), **A-RepC** (exact sparsemax reputation from
  deviation-to-local-median loss), **cross-inhibition** (value-sensitive recruitment +
  inhibition), centralized-oracle ceiling, no-trust floor.
- `AgreementTrustModel` — the load-bearing ablation: our own pipeline with the trust
  *update target* swapped from resolved truth to the **observer-local** majority leaning.
  Observer-local **by design** — scoring agreement against the *global* majority would
  coincide with truth whenever adversaries are a global minority and make the comparison
  vacuous. Hand-audited as faithful.
- `swarm_metrics.py` — isolation metrics (suspect precision/recall, trust-vs-true-competence
  AUC, time-to-isolate). 13 Phase-2 tests; 68 fast tests green overall.

## The honest path: apparent null → clean result

The headline harness first printed **`FALSIFY/UNPROVEN`** — and that was correct *as the
verdict was specified*. Each step below removed a **measurement artifact**, not a result we
disliked; no parameter was tuned toward a target number.

| Step | What it removed | Outcome-vs-agreement gap |
|---|---|---|
| Global pooled-trust AUC | nothing (the artifact itself) | **+0.03** (looks null) |
| Measure **per captured victim** | the dilution by ~21 uncaptured honest observers | +0.20–0.40 |
| Strip the **competence prior** | the truth-derived prior propping up the agreement arm | +0.25–0.42 |
| **Dose-response** by capture severity | the mix of tie-captures with majority-captures | clean threshold |

Two structural facts made the global verdict the wrong lens:
1. **Collective accuracy can't separate the methods.** A cluster that is a *global minority*
   correctly cannot flip a healthy honest majority — every method sits at ~100%. Accuracy is
   the wrong yardstick; the claim is about **isolation**, not the collective vote.
2. **The global AUC averages over all honest observers.** With few captured victims among many
   uncaptured honest nodes, the average washes out a real per-victim effect.

## The result (per captured victim, no truth-prior on either arm)

**(A) Four arms** (K=24, deg=4):

| arm | per-victim AUC |
|---|---|
| outcome + prior | **1.00** |
| outcome, NO prior | **1.00** |
| agreement + prior | 0.80–0.91 |
| agreement, NO prior | 0.58–0.75 |

Outcome-trust isolates the cluster **perfectly, with or without the prior** — truth-grounding
needs no prop. The agreement arm is materially worse, and worse still once the truth-derived
prior is removed.

**(B) Dose-response** — per-victim AUC vs how completely the victim is surrounded:

| capture fraction | n | outcome AUC | agreement AUC |
|---|---|---|---|
| 0.50 (2/4, tie) | 32 | 1.00 | 0.71 |
| 0.60–0.67 (majority) | 6 | 1.00 | **0.00** |
| 0.75 (3/4, majority) | 3 | 1.00 | **0.00** |

**A clean threshold inversion.** Agreement-trust is merely degraded at a *tie*, but the instant
a victim is genuinely *majority*-captured it collapses to **AUC 0.00** — it ranks *every*
attacker above its honest neighbour, because the attackers **are** its local majority. Outcome
-trust holds at **1.00** at every severity. This is the load-bearing claim, demonstrated at the
resolution it actually lives at, as a dose-response (the whole curve) rather than a selected point.

## The mechanism, in one line

Agreement-based trust scores a peer by deviation from the *local consensus*; a colluding cluster
that locally exceeds the majority bound **is** the local consensus, so its members have zero
deviation (fully trusted) while the lone honest dissenter is the deviator (distrusted) — a full
inversion. Outcome-grounded trust scores a peer by whether it matched **resolved truth**, which is
independent of local agreement, so the confidently-wrong cluster accrues distrust no matter how
locally dominant it is.

## Honest caveats (kept loud)

- **Small decisive-n.** The fully-decisive (majority-captured) victims number in the single digits
  per run; the effect is *unambiguous* (0.00 vs 1.00, not noisy) but the sample of severely-captured
  victims is small. A larger seed/config sweep (`phase2_per_victim`) firms it up.
- **An ISOLATION result, not collective accuracy.** A minority cluster *correctly* doesn't flip the
  collective, so accuracy stays ~100% for everyone. The contribution is specifically *which peers a
  captured node trusts*, not the collective decision. Do **not** claim "our swarm is more accurate."
- **Regime-specific.** Contiguous cluster, small-world graph, low degree; the effect is fundamentally
  about *local majority capture*.
- **Toy substrate.** 1D NumPy DDM agents, not scaled. This is a mechanism demonstration.

## What it is, and isn't

It **is**: the first *demonstrated* novel claim of the swarm direction — outcome-grounded trust
isolates colluding clusters that faithful agreement-based SOTA (A-RepC/W-MSR) get inverted by,
shown mechanistically with a dose-response, faithful baselines, and a fair ablation.

It **isn't**: SOTA, validated at scale, or a collective-accuracy win. It is a narrow, honest,
defensible mechanism result.

## Phase 3 (scale) — first result

`python -m cognitive_society.phase3_scale` sweeps N at fixed degree (4) and fixed contiguous-cluster
fraction (25%), no truth-prior on either arm. The per-victim isolation doesn't just *hold* at scale —
the **advantage widens**:

| N | victims | outcome AUC | agreement AUC | gap |
|---|---|---|---|---|
| 24 | 4 | 1.00 | 0.75 | +0.25 |
| 48 | 5 | 1.00 | 0.55 | +0.45 |
| 96 | 8 | 1.00 | 0.55 | +0.45 |
| 144 | 9 | 1.00 | 0.44 | +0.56 |

**Outcome AUC is 1.00 at every N (perfectly scale-invariant)** — the isolation is *local*, so it's
scale-free by construction (no global controller). **Agreement AUC degrades as N grows** (a larger
contiguous cluster at fixed degree makes more *deeply*-captured victims, which fool agreement-trust
harder), so the outcome-vs-agreement gap grows +0.25 → +0.56. *Honest caveat:* victim counts are still
small (4–9/N), so the gap-grows-with-N trend is suggestive (more seeds to firm up); the robust claim is
the flat outcome = 1.00.

## Next

- **Firm up** the decisive-n (more seeds → tighten the gap-grows-with-N trend).
- **Collective scale (decisive):** the *tipping-fraction* test — does outcome-grounded trust raise the
  adversary-cluster fraction that flips the swarm's consensus vs agreement/no-trust, and does that margin
  hold/grow with N (Centola tipping point + Golub–Jackson influence→0)? This is where *collective* accuracy
  separates (near the tipping point, unlike the saturated minority-cluster regime).
- **Write-up / Exo bridge** once firmed.
