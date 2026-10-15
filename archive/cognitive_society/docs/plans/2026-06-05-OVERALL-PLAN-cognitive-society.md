# Overall Plan — From the Ratcliff DDM Codebase to a Cognitive Society

**Date:** 2026-06-05
**Status:** ROADMAP (research + build plan; honest novelty calls baked in)
**Author context:** wraps the Ratcliff 2D-DDM research project and grows a scoped agentic layer toward (but not into) Exo.

---

## 0. TL;DR

Three connected tracks, in dependency order:

1. **Track A — Methods paper (foundation).** Fast, validated JAX/K-L simulator for the 2D spatial diffusion model. Already ~80% done. Needs validation studies + writing.
2. **Track B — Amortized inference (the bridge).** Train a neural posterior estimator on the fast simulator so the 2D model fits in milliseconds with full posteriors. Novel (nobody's done SBI for the *spatial* DDM). Enables everything agentic.
3. **Track C — Cognitive Society (the agentic layer).** A small multi-agent system where each agent decides via a DDM, has SBI-fitted + RL-tuned default parameters, adapts to novelty via posterior uncertainty, infers others' parameters (DDM-based theory of mind), and communicates over a custom trust-weighted A2A protocol.

**The honest crux:** the *society of diverse DDM deciders* is already in the literature. Our novelty is the **inference + trust + adaptation machinery layered on top** — specifically (i) a diffusion model used as the generative model inside computational theory of mind, (ii) amortized-posterior uncertainty as the novelty/OOD trigger for adaptation, and (iii) trust grounded in an inferred cognitive map rather than raw outcome history. Track C is only worth building because of that machinery; the sandbox alone is not a contribution.

---

## 1. What the research says — established vs open (honest)

### Established (do NOT claim as novel)
- **Collective DDM with heterogeneous thresholds.** Networks where agents accumulate private evidence via a DDM and only observe each other's *choices*; cautious agents exploit hasty agents' early errors. ([Diversity Improves Speed and Accuracy in Social Networks, 2020](https://arxiv.org/pdf/2007.05629); network-DDM line.)
- **Threshold / speed-accuracy learning.** Humans and animals adapt the DDM boundary to maximize reward rate, quickly and without instruction ([Bogacz et al. 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC1808344/); [Simen et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2908414/); reward-rate optimality). RL-DDM for value-based choice ([Fontanesi/Pedersen/Frank 2019](https://link.springer.com/article/10.3758/s13423-018-1554-2)).
- **Amortized / simulation-based inference for standard DDMs.** LANs ([Fengler, Frank et al. 2021](https://elifesciences.org/articles/77220)), HSSM (2025), BayesFlow ([Radev et al. 2023](https://arxiv.org/html/2306.16015)). All target 1D sequential-sampling models.
- **Trust & reputation models.** FIRE, BETA, SPORAS; Byzantine-robust trust-weighted signal aggregation that down-weights low-trust signals without consensus ([2024-2025 trust-aggregation work](https://arxiv.org/html/2601.22168v1)).
- **Wisdom of crowds + diversity.** Page's diversity prediction theorem; social influence collapses diversity and erodes crowd wisdom ([Lorenz et al. 2011, PNAS](https://www.pnas.org/doi/10.1073/pnas.1008636108)).
- **Bayesian theory of mind / inverse planning.** Inferring goals/beliefs from behavior assuming rational planning (Baker & Tenenbaum; recent multi-agent ToM, AutoToM/ToMCAT 2025) — but with **RL/planning** generative models.
- **Amortized Bayesian experimental design.** Deep Adaptive Design ([Foster et al.](https://arxiv.org/abs/2103.02438)); [Amortized BED for decision-making, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf).
- **A2A protocol.** Google Agent2Agent (April 2025, donated to Linux Foundation June 2025): agent cards (capability discovery), task lifecycle (submitted→working→completed/failed), SSE streaming.
- **MARL trust/comms.** Trust-based social learning for protocol evolution ([TSLEC 2025](https://arxiv.org/pdf/2511.19562)); bottom-up reputation promotes cooperation ([2025](https://arxiv.org/pdf/2502.01971)).

### Open / genuinely novel for us
1. **SBI / amortized inference for the 2D SPATIAL diffusion model.** None exists. (Track B — the methods contribution.)
2. **A diffusion decision model as the generative model inside computational theory of mind.** ToM uses RL/planning; using a DDM to infer *another agent's decision dynamics* (how decisive/cautious, not what they want) appears unclaimed.
3. **Amortized-posterior uncertainty as the novelty/OOD trigger.** Using the SBI net's own miscalibration off-distribution as the "this is unfamiliar" signal that gates caution + social deference. The misspecification-detection literature ([2024](https://arxiv.org/pdf/2406.03154)) studies detecting this; *using it as an agent's adaptation trigger* is novel.
4. **The integrated architecture:** trust grounded in an inferred cognitive map + uncertainty gating self-vs-social reliance + SBI-fitted defaults. Pieces exist; the composition does not.

---

## 2. The precise novelty claims (what we can defend)

- **Claim 1 (Track B):** "First amortized neural posterior estimator for the spatially-continuous 2D diffusion model, enabling millisecond fits with calibrated posteriors and hierarchical inference previously infeasible." Strong, clean, field-aligned.
- **Claim 2 (Track C):** "Computational theory of mind in which agents infer each other's *decision dynamics* via a diffusion model, and use their own amortized-posterior uncertainty as an adaptation trigger." This is the agentic paper's spine.
- **Claim 3 (Track C, softer):** "Trust grounded in inferred cognitive profiles (not just outcome history) improves robustness to confidently-wrong / adversarial agents under novelty." A result, if the experiments show it.

Everything else (the society, threshold learning, diversity benefit) is *scaffolding built from known parts* — present it as such.

---

## 3. Track C architecture — Cognitive Society

Lightweight programmatic agents (no LLMs initially). Each agent = a DDM decision policy + a cognitive map of peers + trust weights + an adaptation controller.

### 3.1 The agent's mind (DDM)
Decision = accumulate noisy evidence to a threshold. Parameters: boundary `a` (caution), drift `v` (signal strength), non-decision time `t0`. Different params = personalities. The existing simulator is the engine.

### 3.2 Default parameters (SBI + RL)
- **SBI** fits each agent's baseline params from its behavior in familiar tasks (amortized; calibrated).
- **RL** tunes the baseline toward reward-rate-optimal speed-accuracy in the agent's normal task distribution. Use a *simple* learner (bandit / policy-gradient on `a`), grounded in Bogacz reward-rate theory — not deep RL.

### 3.3 Adaptation to novelty (the novel trigger)
On a novel/out-of-distribution problem, the SBI posterior over the agent's own params widens. That uncertainty spike → (a) raise boundary (gather more evidence), (b) up-weight trusted peers (social information when self-evidence is weak), (c) few-shot update from the default prior as new evidence arrives. As the problem becomes familiar, settle or consolidate.

### 3.4 Cognitive mapping (DDM-based theory of mind)
Each agent observes peers' choices + timing and infers their DDM params via the same amortized machinery → a map of "who's decisive, who's cautious, who hesitates." This is the sharpest novel piece.

### 3.5 Trust-weighted A2A (our own protocol)
- **Custom lightweight A2A**, borrowing concepts from Google A2A (capability/agent cards, task/message lifecycle) and FIPA speech acts — but minimal, in-process first.
- **Trust** = f(inferred cognitive map: is this peer competent/reliable?, outcome history: did listening to them help?). Updates from results; down-weights confidently-wrong/adversarial peers (Byzantine-robust pattern).
- **Uncertainty gates deference:** low self-confidence → weight trusted peers more.

---

## 4. Build sequence (four checkpoints — strict gating)

Each checkpoint must run + be verified before the next. This is the anti-sprawl discipline.

1. **Agents decide.** N programmatic DDM agents, no comms, no learning. Verify a single agent's RT/choice behavior is sane and parameters map to personalities.
2. **Agents talk.** Custom A2A bus; agents broadcast decisions; trust starts as a *running reliability estimate* (not RL yet). Verify messages flow + trust tracks reliability.
3. **Agents map each other.** Infer peers' DDM params from observed behavior; verify the cognitive map is accurate (recovered params match true params in sim).
4. **Agents adapt.** SBI-fitted defaults + RL threshold tuning + uncertainty-triggered adaptation + uncertainty-gated deference. Verify behavior improves with reward and agents get appropriately cautious under novelty.

(Trust upgrades from "dumb running estimate" to cognitive-map-grounded + RL only after checkpoint 3 works.)

---

## 5. Anchor research question (pick ONE, build only what answers it)

Ranked, sharpest first:

1. **Does cognitive-map-grounded, uncertainty-gated trust make a society robust to confidently-wrong / adversarial agents under novelty — where flat-broadcast or outcome-only-trust societies fail?** (Novel: ties trust→cognitive map→uncertainty. Defensible result.)
2. Does a society of SBI+RL threshold-learners reach a better *collective* speed-accuracy tradeoff under novelty than fixed-parameter agents? (Grounded; partly adjacent to known diversity results — frame carefully.)
3. Do agents correctly infer each other's decision styles from behavior alone, fast enough to be useful? (Validates the ToM machinery; a building block, not a headline.)

**Recommendation: Question 1.** It's the one that depends on *our* novel machinery (DDM-ToM + uncertainty trigger + grounded trust), so the result can't be claimed by the existing collective-DDM literature.

---

## 6. Feasibility, risks, what to skip

### Feasible
- Programmatic DDM agents + custom in-process A2A + simple trust: days.
- Cognitive mapping reuses the SBI/inference machinery (Track B).
- Simple RL on thresholds (bandit / reward-rate): grounded, tractable.

### Real risks (from the critic lens)
- **Sandbox-with-no-result.** Many moving parts → endless plots, no finding. *Mitigation:* one anchor question; build only what answers it.
- **The collective-DDM is already done.** *Mitigation:* anchor on Q1, which needs our novel machinery; cite and differentiate from the 2020 social-network-DDM work explicitly.
- **Amortized OOD detection can fail silently** ([2024 misspecification work](https://arxiv.org/pdf/2406.03154)). The uncertainty trigger may not flag novelty reliably. *Mitigation:* SBC + posterior-predictive checks; treat detector quality as a research sub-question (it's publishable).
- **RL finickiness.** *Mitigation:* simplest possible learner first; no deep RL until the skeleton works.
- **Naming honesty.** Track C is *applied multi-agent*, not core cognitive science, unless Q1 yields a real result. State it.

### Skip (scope traps)
- **Emergent communication** (learning *what* to send). Finicky, open-ended, defer indefinitely.
- **LLM-backed agents** until the programmatic skeleton works end-to-end.
- **Full Google A2A spec compliance** — borrow concepts, build minimal; meet the spec only if/when interop matters.
- **Neural-operator GRF / fully differentiable simulator** — Track-A distractions; K-L is fast enough.

---

## 7. Outputs / paper potential

- **Paper 1 (Track A):** JAX/K-L methods paper. Behavior Research Methods / J. Math Psych. Collaborators: Ratcliff.
- **Paper 2 (Track B):** Amortized inference for the 2D spatial DDM. J. Math Psych / Psychological Methods / ML-for-science workshop. Collaborators: Ratcliff + Frank/Fengler (HSSM group).
- **Paper 3 (Track C):** DDM-based theory of mind + uncertainty-gated trust in cognitive societies. A multi-agent / cognitive-systems venue. The agentic contribution.

---

## 8. Bridge to Exo (kept separate, noted)

Exo's TRHN engine is explicitly "a multi-dimensional DDM" (Bogacz noisy-Hopfield); its identity manifold gives per-person discovered axes; its decision parameters are currently *heuristic* (`from_manifold_position`). Track B's amortized inference is the principled replacement: fit persona decision parameters from behavior instead of guessing from style. Track C's trust + adaptation machinery is the scoped, scientific prototype of Exo's "adapt to unknown domains." **Same machinery, two homes.** Integration deferred by user; the architectural seam is documented so it's ready when wanted.

---

## 9. Immediate next move

Two cheap, decisive first steps (either order):

- **Track B MVP (~$10 GPU):** ~20k simulated datasets → small normalizing flow → SBC + agreement-with-simplex check. Validates the entire amortized-inference thesis (and the cognitive-mapping engine for Track C) for ~$10.
- **Track C checkpoint 1 (laptop, hours):** N programmatic DDM agents that decide. Zero new dependencies; pure use of the existing simulator. Verifies the agent abstraction before any comms/learning.

**Recommendation:** Track C checkpoint 1 first (free, immediate, no GPU), in parallel with scheduling the Track B MVP. They reinforce: the SBI net (Track B) is the cognitive-mapping engine (Track C checkpoint 3), so validating it early de-risks both.
