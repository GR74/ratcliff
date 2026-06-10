"""Checkpoints 3+4 — the integrated Cognitive Society.

Ties the pieces together:
  - agents decide (checkpoint 1)
  - DDM-coupled communication + trust (checkpoint 2, comms.py)
  - cognitive maps ground trust (checkpoint 3): agents infer peers' competence
    from observed behavior (accuracy at the mapping evidence levels; EZ-diffusion
    can recover full style) and seed trust from it
  - adaptation (checkpoint 4): each agent gauges its own uncertainty on a problem
    (how split its private decisions are); under high uncertainty it (a) raises
    its boundary (gathers more evidence) and (b) defers more to trusted peers
    (uncertainty gates self-vs-social reliance)

A round:
  1. PRIVATE: each agent samples its own decision distribution at this problem
     (a small internal batch) -> a leaning (majority) + a confidence (how
     decisive the split was). Low confidence = the problem is hard/novel for it.
  2. SOCIAL: each agent makes its final decision with a trust-weighted social
     drift from peers' leanings added to its accumulator, scaled up when its own
     confidence is low; and a raised boundary under uncertainty.
  3. OUTCOME: trust updates from whether each peer's leaning matched the truth.

Config flags let experiments compare conditions (flat broadcast vs outcome-only
trust vs cognitive-map-grounded + uncertainty-gated trust).
"""
from dataclasses import dataclass

import numpy as np

from cognitive_society.comms import TrustModel, leaning, social_drift


@dataclass
class SocietyConfig:
    social_gain: float = 0.6          # cap on social drift
    use_social: bool = True           # agents listen to peers at all
    use_trust_weights: bool = True    # weight peers by outcome-driven trust
    use_competence_prior: bool = True # seed trust + weight by inferred competence (cognitive map)
    adaptive: bool = True             # uncertainty gates caution + deference
    n_private: int = 21               # internal samples for the private leaning/conf (odd -> no ties)
    caution_gain: float = 0.6         # how much uncertainty raises the boundary
    social_base: float = 0.5          # baseline social-gain fraction (even when fully certain)
    social_uncertainty_scale: float = 1.0  # extra social-gain fraction per unit uncertainty
    map_evidence: tuple = (0.4, 0.6, 0.8)
    map_trials: int = 1500


# Named conditions for experiments.
def cfg_flat(**kw):
    """Flat broadcast: everyone weighted equally, no trust, no adaptation."""
    return SocietyConfig(use_social=True, use_trust_weights=False,
                         use_competence_prior=False, adaptive=False, **kw)


def cfg_outcome_trust(**kw):
    """Outcome-only trust: trust learns from results, but no cognitive-map head
    start and no uncertainty gating."""
    return SocietyConfig(use_social=True, use_trust_weights=True,
                         use_competence_prior=False, adaptive=False, **kw)


def cfg_full(**kw):
    """Our machinery: cognitive-map-grounded trust + uncertainty-gated deference."""
    return SocietyConfig(use_social=True, use_trust_weights=True,
                         use_competence_prior=True, adaptive=True, **kw)


class Society:
    def __init__(self, agents, config: SocietyConfig = None, sigma: float = 1.0,
                 rng_seed: int = 0):
        self.agents = agents
        self.K = len(agents)
        self.cfg = config or SocietyConfig()
        self.sigma = sigma
        self.rng = np.random.default_rng(rng_seed)
        self.trust = [TrustModel(self.K) for _ in range(self.K)]
        # competence[i, j] = how reliable observer i believes peer j is, in [0,1]
        self.competence = np.full((self.K, self.K), 0.5)
        self.mapped = False

    # ---- checkpoint 3: cognitive maps ground trust ----------------------
    def build_cognitive_maps(self):
        """Each agent observes the (public) behavior of every peer and infers a
        competence estimate, then seeds trust from it. Competence = inferred
        accuracy at the mapping evidence levels."""
        comp = np.zeros(self.K)
        for j, target in enumerate(self.agents):
            pcs = []
            for ev in self.cfg.map_evidence:
                ch, _ = target.decide_batch(ev, self.cfg.map_trials, self.rng)
                truth = 1 if ev > 0 else 0
                pcs.append((ch == truth).mean())
            comp[j] = float(np.mean(pcs))
            # Trust seeding needs only this scalar competence. Richer per-peer
            # style recovery (boundary/drift/ndt) is available in closed form via
            # ez_diffusion.recover_from_agent_observations if a downstream use
            # wants it — not run here so mapping stays cheap.
        for i in range(self.K):
            self.competence[i] = comp
            if self.cfg.use_competence_prior:
                self.trust[i].set_prior_from_competence(comp)
        self.mapped = True
        return comp

    # ---- a single decision round ----------------------------------------
    def _private(self, evidence):
        """Each agent's private leaning + confidence (how decisive its split)."""
        leanings = np.zeros(self.K, dtype=int)
        conf = np.zeros(self.K)
        for i, a in enumerate(self.agents):
            ch, _ = a.decide_batch(evidence, self.cfg.n_private, self.rng)
            mean = ch.mean()
            leanings[i] = leaning(1 if mean >= 0.5 else 0)
            conf[i] = abs(mean - 0.5) * 2.0   # 0 (split) .. 1 (unanimous)
        return leanings, conf

    def round(self, evidence):
        cfg = self.cfg
        truth_choice = 1 if evidence > 0 else 0
        outcome = 1 if truth_choice == 1 else -1

        leanings, conf = self._private(evidence)
        final = np.zeros(self.K, dtype=int)

        for i, a in enumerate(self.agents):
            mask = np.ones(self.K, dtype=bool)
            mask[i] = False
            # uncertainty gate: high when this agent's own decision was split.
            gate = (1.0 - conf[i]) if cfg.adaptive else 0.5
            # Adaptive cap: deference rises with the agent's own uncertainty. sg
            # spans [social_base, social_base + social_uncertainty_scale] x
            # social_gain (default [0.5, 1.5]x), so under high uncertainty social
            # drift can intentionally exceed weak private evidence — bounded,
            # per-round (not a within-trial feedback loop). See design notes.
            sg = (cfg.social_gain * (cfg.social_base + cfg.social_uncertainty_scale * gate)
                  if cfg.adaptive else cfg.social_gain)

            if cfg.use_social:
                t = self.trust[i].trust() if cfg.use_trust_weights else np.ones(self.K)
                comp = self.competence[i] if cfg.use_competence_prior else np.ones(self.K)
                sd = social_drift(t[mask], leanings[mask], comp[mask], sg)
            else:
                sd = 0.0

            bscale = 1.0 + (cfg.caution_gain * gate if cfg.adaptive else 0.0)
            ch, _ = a.decide(evidence, self.rng, extra_drift=sd, boundary_scale=bscale)
            final[i] = ch

        # Trust update from this round's outcome. Trust accumulators are size K
        # (indexed by global agent id); the self-entry updates harmlessly since
        # self-trust is masked out of the social drift. NOTE: every observer i is
        # updated with the same global leanings + outcome, so trust is currently a
        # shared, truth-driven (objective) reliability estimate replicated per
        # agent, not a per-observer subjective belief — a deliberate simplification
        # (per-observer subjective trust is a noted extension; see design notes).
        for i in range(self.K):
            self.trust[i].update(leanings, outcome)

        majority = 1 if final.mean() >= 0.5 else 0
        return {
            "final": final,
            "private_leanings": leanings,
            "majority": majority,
            "truth": truth_choice,
            "correct": majority == truth_choice,
        }

    def run(self, evidences, build_maps: bool = True):
        """Run a sequence of decision problems; return collective accuracy +
        per-round records. `evidences` is a list of signed evidence values."""
        if build_maps and self.cfg.use_competence_prior and not self.mapped:
            self.build_cognitive_maps()
        records = [self.round(ev) for ev in evidences]
        collective_acc = float(np.mean([r["correct"] for r in records]))
        return {"collective_accuracy": collective_acc, "records": records}
