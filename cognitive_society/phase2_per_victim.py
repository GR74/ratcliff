"""Phase 2 — the per-captured-victim isolation result (the demonstrated claim).

The headline `phase2_experiment.py` scores conditions on GLOBAL collective accuracy
and a GLOBAL pooled-trust AUC. Both *mask* the real effect:

  - Collective accuracy can't separate the methods at all: a contiguous cluster that
    is a GLOBAL MINORITY correctly cannot flip a healthy honest majority, so every
    method sits at ~100%. Accuracy is the wrong yardstick for this claim.
  - The global pooled-trust AUC averages over ALL honest observers; with only a few
    captured victims among many uncaptured honest nodes, the average washes the
    effect out (agreement AUC stayed ~0.97 globally).

The claim is about ISOLATION at the CAPTURED VICTIMS — honest nodes whose own
neighbourhood is adversary-majority. This experiment measures exactly there: for each
captured victim v, score v's OWN trust row over its neighbours (honest vs adversary),
with NO truth-derived competence prior on either arm (the prior is itself an outcome
signal, so it must be stripped for a fair outcome-vs-agreement comparison).

Two reads:
  (A) FOUR ARMS — outcome / agreement, each with and without the prior. Isolates the
      prior as a confound: outcome holds at AUC 1.0 either way; agreement is propped
      up by the prior and degrades sharply once it is removed.
  (B) DOSE-RESPONSE — per-victim AUC vs CAPTURE SEVERITY (adversary fraction of the
      victim's neighbourhood), no prior. The whole curve, not a selected point.

Demonstrated result (representative run): outcome-trust isolates the cluster at AUC
1.00 for every capture severity; agreement-trust is fine at a tie (~0.7) but INVERTS
to AUC ~0.0 once a victim is genuinely majority-captured (frac > 0.5) — it trusts its
attackers over its honest neighbour, because they ARE its local majority. That clean
threshold inversion is the load-bearing Phase-2 claim, demonstrated at the resolution
it lives at. Honest caveats: small decisive-n; an ISOLATION result, not a collective-
accuracy one; specific to the contiguous-majority-capture regime.

    python -m cognitive_society.phase2_per_victim
"""
import numpy as np

from cognitive_society.adversary import build_clustered_society, verify_fbound_exceeded
from cognitive_society.baselines import AgreementTrustModel
from cognitive_society.swarm_metrics import trust_auc
from cognitive_society.society import SocietyConfig
from cognitive_society.phase2_experiment import (
    make_evidences, DEFAULT_BUILDER, DEFAULT_TOPOLOGY_FN,
)


def _cfg(prior):
    """cfg_swarm-equivalent (local per-agent maps), with the competence prior on/off."""
    return SocietyConfig(use_social=True, use_trust_weights=True,
                         use_competence_prior=prior, adaptive=True, local_maps=True)


def build_and_run(K, deg, n_adv, seed, agreement, prior, n_problems=40):
    """Build a clustered society, optionally swap to agreement-trust, run the stream.

    agreement=True swaps soc.trust to observer-LOCAL AgreementTrustModel (update fed
    the round's local-majority leaning, not truth). prior=False removes the competence
    prior entirely (the only truth-derived signal besides the trust update), giving the
    clean outcome-vs-agreement update-rule comparison. Returns (soc, gt, victims).
    """
    soc, gt, S = build_clustered_society(
        K, deg, n_adv, seed, config=_cfg(prior),
        builder=DEFAULT_BUILDER, topology_fn=DEFAULT_TOPOLOGY_FN)
    if agreement:
        soc.trust = [AgreementTrustModel(soc.K, neighbour_mask=soc.adj[i])
                     for i in range(soc.K)]
    if soc.cfg.use_competence_prior and not soc.mapped:
        soc.build_cognitive_maps()
    for ev in make_evidences(n_problems, seed):
        soc.round(ev, learn=True)
    _, victims = verify_fbound_exceeded(soc.adj, S)
    return soc, np.asarray(gt), victims


def victim_aucs(soc, gt, victims):
    """Each captured victim's OWN trust over its neighbours, honest-vs-adversary AUC.

    AUC 1.0 = the victim ranks every honest neighbour above every adversary; 0.0 =
    full inversion (it trusts every attacker over its honest neighbour). Victims whose
    neighbours are all one class are skipped (AUC undefined)."""
    out = []
    for vrec in victims:
        v = int(vrec[0])
        nb = np.nonzero(soc.adj[v])[0]
        g = gt[nb]
        if (g == 0).any() and (g == 1).any():
            out.append((vrec, trust_auc(soc.trust[v].trust()[nb], g)))
    return out


ARMS = [
    ("outcome + prior", False, True),
    ("outcome, NO prior", False, False),
    ("agreement + prior", True, True),
    ("agreement, NO prior", True, False),
]


def four_arm_table(K, deg, n_adv, n_seeds):
    """(A) Per-victim AUC for all four arms, aggregated over seeds."""
    res = {name: [] for name, _, _ in ARMS}
    nv = 0
    for seed in range(n_seeds):
        _, _, v0 = build_and_run(K, deg, n_adv, seed, False, True)
        if not v0:
            continue
        nv += len(v0)
        for name, agr, pri in ARMS:
            soc, gt, vic = build_and_run(K, deg, n_adv, seed, agr, pri)
            res[name] += [a for _, a in victim_aucs(soc, gt, vic)]
    return res, nv


def dose_response(configs, n_adv_grid, n_seeds):
    """(B) Per-victim AUC vs capture severity (no prior on either arm)."""
    rec = []  # (capture_frac, outcome_auc, agreement_auc)
    for K, deg in configs:
        for n_adv in n_adv_grid:
            for seed in range(n_seeds):
                soc_o, gt, vic = build_and_run(K, deg, n_adv, seed, False, False)
                soc_a, _, _ = build_and_run(K, deg, n_adv, seed, True, False)
                oa = {int(r[0]): a for r, a in victim_aucs(soc_o, gt, vic)}
                aa = {int(r[0]): a for r, a in victim_aucs(soc_a, gt, vic)}
                for vrec in vic:
                    v = int(vrec[0])
                    if v in oa and v in aa:
                        rec.append((vrec[1] / vrec[2], oa[v], aa[v]))
    return np.array(rec) if rec else np.zeros((0, 3))


def main():
    print("=" * 70)
    print("PHASE 2 — per-captured-victim isolation (outcome vs agreement trust)")
    print("=" * 70)
    print("Measured AT the captured victims, NO truth-derived prior on either arm.\n")

    print("(A) FOUR ARMS — per-victim AUC (K=24, deg=4, n_adv=8)")
    res, nv = four_arm_table(24, 4, 8, n_seeds=8)
    for name, _, _ in ARMS:
        m = float(np.mean(res[name])) if res[name] else float("nan")
        print(f"    {name:22s} AUC = {m:.2f}")
    if res["outcome, NO prior"] and res["agreement, NO prior"]:
        gap = np.mean(res["outcome, NO prior"]) - np.mean(res["agreement, NO prior"])
        print(f"    >>> load-bearing (no-prior) gap = {gap:+.2f}  (captured victims: {nv})\n")

    print("(B) DOSE-RESPONSE — per-victim AUC vs capture severity (no prior)")
    rec = dose_response([(24, 4), (28, 4)], (8, 9, 10), n_seeds=8)
    print(f"    total measurable captured victims: {len(rec)}")
    print(f"    {'capture frac':18s} {'n':>4s} {'outcome AUC':>12s} {'agreement AUC':>14s}")
    for lo, hi, label in [(0.49, 0.51, "0.50 (tie)"),
                          (0.51, 0.74, "0.51-0.74 (maj)"),
                          (0.74, 1.01, ">=0.75 (strong maj)")]:
        m = (rec[:, 0] >= lo) & (rec[:, 0] <= hi)
        if m.any():
            print(f"    {label:18s} {int(m.sum()):4d} {rec[m, 1].mean():12.2f} {rec[m, 2].mean():14.2f}")
    if len(rec):
        maj = rec[rec[:, 0] > 0.5]
        print("\n  VERDICT (per-victim isolation, the claim's actual resolution):")
        print(f"    outcome-trust isolates at AUC {rec[:, 1].mean():.2f} across all severities")
        if len(maj):
            print(f"    agreement-trust INVERTS to AUC {maj[:, 2].mean():.2f} for majority-captured "
                  f"victims (frac>0.5, n={len(maj)})")
            supported = rec[:, 1].mean() >= 0.9 and maj[:, 2].mean() <= 0.4
            print(f"    [{'SUPPORTED' if supported else 'PARTIAL'}] outcome holds; agreement "
                  f"{'collapses' if supported else 'degrades'} where the cluster locally dominates.")
    print("=" * 70)


if __name__ == "__main__":
    main()
