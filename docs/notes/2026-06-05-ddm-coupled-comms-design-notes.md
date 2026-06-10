# Design Notes — DDM-Coupled Communication + Trust (Cognitive Society checkpoint 2)

**Date:** 2026-06-05
**Source:** a 4-lane research/design panel (lit review, mechanism options, trust-as-meta-DDM, adversarial critic) + synthesis. These notes record the *why* behind the `comms.py` implementation and the honest novelty boundary for the eventual paper.

---

## What was built (and why it's the recommended design)

`comms.py` implements **trust-weighted drift injection on bounded peer leanings**:

    social_drift_i = social_gain * Σ_j  ŵ_ij * leaning_j        (ŵ = trust·competence, normalized)

where `leaning_j ∈ {−1,+1}` is peer j's *committed* choice direction, `ŵ` is row-normalized so the
weights sum to 1, and the whole term is capped by `social_gain`. Trust is a slow, bounded
meta-accumulator updated **per round against the realized outcome**.

The design panel independently recommended exactly this:

> "Ship mechanism (i) trust-weighted drift injection using **bounded peer leaning** (sign/confidence of
> peer choice), **not raw accumulator value**, with **row-normalized trust** and a **global gain cap**.
> It is unconditionally stable, cannot blow up, resists echo chambers when you inject committed leanings
> rather than live accumulators, and is honestly the cleanest novelty."

So the implementation is the literature-justified, provably-stable choice — not an accident.

## The three stability guarantees (all present in code)

The panel's stability analysis (drawn from Leaky Competing Accumulator theory and bounded-confidence
opinion dynamics) says naive *additive coupling on raw, unbounded accumulators* is the runaway-divergence
/ echo-chamber regime. The cure is three properties — all of which `comms.py` has:

1. **Bounded peer signal** — we inject `leaning ∈ {−1,+1}` (a committed, bounded stance), never a raw
   live accumulator. Coupling on a bounded signal cannot blow up.
2. **Row-normalized, gain-capped social drift** — `|social_drift| ≤ social_gain`, so social evidence
   can never dominate an agent's own private evidence. Round-based (not continuous within-trial), so
   there is no positive-feedback integration loop.
3. **Outcome-driven trust, not agreement-driven** — `TrustModel.update` keys on `sign(leaning) ×
   sign(outcome)` (did the peer match *ground truth*), **not** on whether the peer agreed with the
   observer. The panel flags agreement-driven trust as *the* mechanism that produces echo chambers
   (it is mathematically a bounded-confidence model → polarization). We avoid it by construction.

## Honest novelty boundary (for the paper)

Each ingredient exists separately — **do not claim coupled-DDM or social-DDM as novel:**

- **Peer accumulator → my accumulation** is Srivastava & Leonard, *Collective Decision-Making in Ideal
  Networks* (IEEE TCNS 2014; arXiv:1402.3634) — Laplacian-coupled DDM, but **fixed** adjacency weights,
  no trust.
- **Partner signal modulates my accumulator** is Esmaily, Hortensius, Bahrami et al., *Interpersonal
  alignment of neural evidence accumulation to social exchange of confidence* (eLife 2023) — but
  reported confidence, across-trials, **fixed** coupling, no trust.
- **Networked-DDM on committed stances** is arXiv:2408.12127 (2024) — but **unweighted** edges, OU leak.
- **Trust as slow reliability** is the *A race to belief* line (arXiv:2511.22617) and Bayesian-reliability
  work — but trust as a **Bayesian/Kalman precision**, never as a **diffusion-to-boundary meta-DDM**.

**The defensible, unclaimed contribution is the *combination*:**
> trust-gated drift injection of a peer's committed leaning into a first-order DDM, where the trust gate
> is itself a slower second-order DDM accumulating **outcome-verified** reliability, **seeded from a
> cognitive map** of inferred peer competence.

Claim *only* at that resolution.

## Honest caveats (reviewer-anticipating)

- **Ground-truth dependence.** Outcome-driven trust needs observable outcomes. Our experiment supplies
  them (truth = sign of evidence). In open-ended settings without verifiable outcomes, trust has nothing
  reliable to accumulate and would degrade toward agreement-driven (echo-chamber) dynamics. State this.
- **Optional refinement (not required for stability):** a Laplacian / mean-removing *difference* form
  `(g(peer) − x_i)` would make coupling self-cancel at consensus (Srivastava–Leonard structure). Our
  round-based, gain-capped additive-on-leanings form is already stable; the difference form is a possible
  v2 for a continuous within-trial variant.
- **Trust decay / exploration floor.** To avoid permanently locking a peer in/out, a small trust
  decay-to-prior or exploration noise is worth adding (panel suggestion). Currently trust is bounded but
  not decayed — a candidate improvement.

## Bottom line

The checkpoint-2 implementation is the literature-recommended, provably-stable mechanism, and it already
includes the two non-negotiable anti-pathology properties (bounded leanings + outcome-driven trust). The
novelty is the trust-as-meta-DDM gating + outcome-driven, cognitive-map-seeded trust — positioned
honestly against the coupled-DDM and social-DDM prior art above.
