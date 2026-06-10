# Research Synthesis — Decentralized DDM Swarm: the honest novelty boundary

**Date:** 2026-06-10
**Source:** 4-lane parallel literature workflow (swarm intelligence, networked-DDM / opinion
dynamics, distributed trust / Byzantine consensus, emergence-at-scale) + synthesis. Records
where a per-agent, locally-mapped, trust-coupled DDM swarm actually sits, what is already
published (so we never overclaim), and the one defensible claim + the experiment that proves it.

---

## Positioning (the 2×2 white space)

Four mature, surprisingly-disjoint literatures bound us:
- **Networked-DDM** (Srivastava–Leonard 2014) owns *{DDM substrate + FIXED Laplacian weights, honest agents, no adversary}*.
- **Distributed-SPRT** (CISPRT, distributed Chernoff) owns *{accumulator + provable N-scaling to the centralized error exponent}*; its Byzantine branch filters by instantaneous value-extremeness.
- **Opinion dynamics** (DeGroot/Friedkin–Johnsen, bounded-confidence, Bizyaeva–Franci–Leonard) owns *{adaptive/learned weights + saturation + gating}* — but on abstract opinions, weights driven by **agreement/homophily, never outcomes**.
- **Resilient-consensus / trust** (W-MSR; **A-RepC 2026**; Crowd-Vetting) owns *{local-only learned reputation that downweights adversaries, scalable}* — but trust is again **agreement-based** (deviation from neighborhood median).
- **Swarm-DDM** (SODM/aSODM; human-inspired DDM consensus 2024–25) owns *{per-agent DDM + social-drift speed-accuracy knob on a local topology}* — but benign, no trust, no competence inference.

Our cell — *a ground-truth-bound DDM whose social coupling weight is a **learned, OUTCOME-driven** trust meta-accumulator fed by inter-agent EZ-diffusion competence inference, gated by self-uncertainty, operating locally at scale against confident adversary **clusters*** — is occupied by no single line. **The integration is the position, not any ingredient.**

## ALREADY CLAIMED — do NOT claim (in isolation or rephrased)

1. "Collective decisions ARE a DDM/SPRT" + value-sensitivity from cross-inhibition (Marshall–Bogacz 2009; Pais 2013).
2. Per-agent accumulators coupled on a graph/Laplacian (Srivastava–Leonard 2014).
3. Social-drift / social-vs-personal weight as a speed-accuracy knob in a DDM swarm (SODM/aSODM 2024–25).
4. EZ-diffusion recovers drift from behavior — **single-agent recovery is solved** (Wagenmakers 2007).
5. Distributed accumulation scaling to the centralized error exponent with N (CISPRT).
6. Robust aggregation isolating a **bounded count** of Byzantines + the F-total/F-local/(r,s)-robustness machinery (W-MSR).
7. **Local-only learned per-neighbor reputation that downweights adversaries, no prior F, scalable — TAKEN by A-RepC (2026).** Our decentralized-local-map *goal* is not novel by itself.
8. Uncertainty/confidence-gated deference as a standalone resilience knob.
9. "Decentralization improves the crowd / less-is-more / local beats global" (Becker–Centola 2017; Talamali 2021).
10. Per-agent local suspect lists isolating adversaries with no controller (Crowd-Vetting).

## GENUINELY UNCLAIMED — defensible

- **(A) Trust grounded in realized OUTCOME, not AGREEMENT / median-deviation / homophily.** *Strongest.* Every robustness method above filters by instantaneous value-extremeness or agreement — so a **confidently-wrong, internally-consistent adversary cluster that agrees with itself** fools every median/consensus filter. Outcome-grounding is the one mechanism that can still flag it.
- **(B)** Inter-agent competence inference by inverting a generative cognitive model (EZ-diffusion) of a neighbor, *online, agent-on-agent, inside the decision loop*. (Novel use; rests on a solved primitive.)
- **(C)** Trust as a slow **two-timescale meta-accumulator** (a second outcome-driven DDM stacked on the fast decision DDM), not a scalar EWMA / simplex weight.
- **(D)** The four coupled into one per-agent local module. (Integration novelty — reviewers discount unless it yields a result the parts can't.)
- **(E)** Robustness specifically to **confident, mutually-reinforcing adversary CLUSTERS that locally exceed the W-MSR per-neighborhood bound** — exactly where agreement-based SOTA is weakest.

## The ONE claim (mechanism-class resolution)

> **Outcome-grounded trust isolates confident, internally-consistent adversary CLUSTERS that
> agreement-based local trust (A-RepC, W-MSR, bounded-confidence) provably cannot — with no
> global controller, on a local topology — and this advantage persists or grows as N scales.**

One sentence for a reviewer: *grounding trust in realized outcomes rather than agreement is
what lets a controller-free local swarm isolate a confidently-wrong colluding cluster.* The
EZ competence map + two-timescale meta-accumulator + self-uncertainty gate are the **mechanism**
("how"), not separate contributions. Do **not** claim "a swarm of DDMs," "decentralized
self-organizing consensus," or "social-drift coupling" — all taken.

## Recommended build — TOPOLOGY-FIRST (not scale-first)

Topology changes everything (spatiality breaks well-mixed assumptions; neighborhood size is the
dominant speed-accuracy knob). Do **not** assume the global-map 74→97% transfers.
- **Phase 0:** port the validated build to a fixed local topology, **all honest**, run the
  topology-only control across ring / random-regular / small-world / scale-free → prove residual
  collective bias ≈ 0 *before* any adversary.
- **Phase 1 (topology):** fix N≈50–100, add per-agent local maps, sweep mean degree + topology
  family; measure the global→local accuracy/isolation **gap** explicitly.
- **Phase 2 (adversary structure):** inject confident colluding **clusters** sized to locally
  exceed the W-MSR r-bound — the decisive regime.
- **Phase 3 (scale):** only after topology behaves, sweep N (50/100/500/1000+) at fixed degree;
  test Golub–Jackson (confidently-wrong influence → 0 as N grows) and whether the Centola tipping
  fraction **rises** with our mechanism.

Topology: **random-regular** primary (clean degree + (r,s)-robustness), small-world realistic,
scale-free to stress hub-capture (monitor trust-Gini — a de-facto controller emerging is the
"centralization-by-emergence" failure). Methodology: Reina/Valentini **micro-macro** (mean-field
ODE + master equation, validate vs agent sims) so scaling claims survive review.

## Headline experiment — clustered-confident-adversary stress test

Baselines on identical topology + scenario: (1) **centralized-oracle upper bound** (SPRT optimal),
(2) **no-trust lower bound** (Srivastava–Leonard uniform coupling), (3) **agreement-trust SOTA**
(A-RepC, W-MSR — the must-beats), (4) bounded-confidence gating, (5) cross-inhibition-only swarm
(cheap "antifragile" baseline), (6) **ours**. Cleanest isolation ablation: **outcome-trust vs an
agreement-trust variant of our OWN system.** Metrics: collective accuracy vs adversary fraction +
cluster size; adversary-isolation precision/recall + trust-vs-true-competence correlation +
time-to-isolate; Golub–Jackson realized cluster influence vs N (→0); Centola tipping fraction
(ours should raise it); Lorenz guard (accuracy must RISE, not just variance fall); speed-accuracy
vs SPRT ceiling; topology-only bias; trust-Gini; micro-macro agreement.

**Compelling result:** in the cluster-exceeds-r regime, ours sits near the centralized oracle while
A-RepC/W-MSR/bounded-confidence collapse toward the no-trust floor — and ablation attributes the gap
*specifically* to outcome-grounding (the agreement-trust variant of our own system also fails) —
and the advantage is flat-or-growing in N.

## Traps to avoid

- **Cognitive-layer-as-overhead:** cross-inhibition + median-trimming are strong cheap baselines; beating only naive averaging makes EZ+trust look like dead weight. Win *specifically* in the clustered-confident regime where they break.
- **Lorenz trap:** social drift on a topology can collapse diversity and manufacture false confidence with no accuracy gain; topology alone can distort perception. Run the all-honest control first; require accuracy (not just variance) to move.
- **Global→local non-transfer:** the 74→97% used a global shared map; per-agent local maps may isolate worse. Quantify the gap honestly — it's empirical, not given.
- **EZ cold-start:** EZ needs ~80 observations for a stable drift; bridge with the outcome meta-accumulator + outcome-validated second-hand trust (never raw agreement) until the map warms up.
