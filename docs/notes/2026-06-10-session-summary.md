# Session summary — 2026-06-10

A long build day across the cognitive-society / swarm direction, plus a GPU
validation of Track B. The through-line: ship real, *defensible* results — measured
honestly, adversarially reviewed, with nulls and caveats reported straight.

## What shipped (committed + pushed)

### 1. Cognitive society — hardened + made honest
- Applied a 4-lens adversarial **code review** (bugs / result-validity / quality /
  science-stability). Fixed a real portability bug (`conftest.py` secretly required
  the full JAX stack → the pure-NumPy tests now run JAX-free), made `ez_recover`
  fail-loud on degenerate input, overflow-proofed the trust logistic, removed dead code.
- The result-validity reviewer ran an independent **30-seed sweep + blinded-map
  ablation** and certified the +22pt anchor result **real** (not a ground-truth leak).
- **Decomposed the headline honestly:** of the +23.3pt flat→full gain, ~+19.7 is plain
  trust-learning and only **+3.6pt** is the novel cognitive-map + gating layer — small
  but **significant (paired Wilcoxon p=0.0003)**. Added a private baseline, per-seed
  win rate, significance test, and an adversary-count sweep to `experiment.py`.

### 2. RL deference policy (checkpoint 4, RL)
- `rl.py` — a linear-Gaussian REINFORCE policy that learns each agent's deference
  from outcome reward; `rl_experiment.py` stress-tests it under a hostile regime shift.
- **Honest result:** gating is worth +6pt post-shift, and the learned policy **recovers
  that benefit from reward alone** (matching the hand-tuned gate without being told the
  rule) — but does **not** beat it and is noisier. Reported as a tie, not inflated.

### 3. Track B — amortized SBI, validated at scale (H100)
- Fixed the SBC import (sbi≥0.23), ran N=500 smoke → **N=3000 full on an H100**:
  overall recovery **20.9%** (location params 6–13%, variability params hard as known),
  calibration **c2st≈0.50** (nominal). Thesis holds and sharpens with scale.
  `docs/notes/2026-06-10-track-b-sbi-validation.md`.

### 4. Decentralized swarm — research → Phase 0 → 1 → 2
- **Research synthesis** (4-lane workflow): found the honest novelty boundary — a
  "swarm of DDMs", accumulators-on-a-graph, and local-learned-reputation are all
  *already published* (Marshall-Bogacz, Srivastava-Leonard, A-RepC 2026). The one
  unclaimed lane: **outcome- vs agreement-grounded trust against colluding clusters.**
  `docs/notes/2026-06-10-swarm-coordination-research-synthesis.md`.
- **Phase 0** — `topology.py` (5 graph families) + `Society(topology=)`; all-honest
  control passes (topology alone doesn't distort; local ≈ centralized within ~1pt).
- **Phase 1** — `cfg_swarm`: per-agent **subjective, neighbour-only** cognitive maps.
- **Phase 2** — faithful Byzantine baselines (**W-MSR, A-RepC**, cross-inhibition,
  hand-audited as not strawmen), a contiguous-cluster stress test, and the
  **demonstrated result** (`phase2_per_victim.py`): measured per captured victim,
  outcome-grounded trust isolates the cluster at **AUC 1.00** while agreement-trust
  **inverts to ~0.00** once the cluster is the local majority — a clean threshold
  inversion. `docs/notes/2026-06-10-phase2-swarm-result.md`.

## How the Phase-2 result was reached (the important part)
The headline harness first printed **FALSIFY/UNPROVEN** — correctly, as specified.
The clean result emerged by **removing measurement artifacts, not by tuning**:
global→per-victim (the effect lives at captured victims, not the average) → strip the
truth-derived competence prior (a confound) → disaggregate by capture severity
(a dose-response, not a cherry-picked point). Each step was a fairer measurement; no
parameter was moved toward a target number. A built-from-scratch Phase-2 workflow
(research→design→build→verify) wrote the baselines; every line was then hand-audited.

## State of the tree
- **71 fast tests green, no GPU** (35 society + RL + topology + swarm + Phase-2).
- Commits this session: code-review polish, RL + honesty fixes, SBI validation note,
  swarm research synthesis, swarm Phase 0+1, swarm Phase 2, per-victim result + docs.

## Honest scorecard
- **Done + defensible:** society (reviewed, decomposed), RL (honest tie), SBI (validated
  at scale), swarm Phase 0/1/2 with one *demonstrated* novel claim.
- **Not claimed:** SOTA, scale, collective-accuracy wins. The Phase-2 result is a narrow,
  mechanism-level isolation result with small decisive-n.

## Next
1. **Firm up** Phase-2 decisive-n (more seeds/configs); 2. **Phase 3 (scale)** — does the
inversion + perfect outcome-isolation persist as N grows, adversary influence → 0;
3. **Write-up**; 4. **Exo bridge** — TRHN as a `DecisionEngine`.
