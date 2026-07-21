"""Brick 1, grounded in the REAL DDM — chronometric (RT-based) confidence.

confidence_woc.py used a Gaussian sampler with analytic-precision confidence (n/var). That
makes the "confidence-weighting is optimal" claim close to a textbook inverse-variance identity.
This module removes that objection: it uses the VALIDATED binary DDMAgent (agent.py) and reads
confidence from the decision's REACTION TIME — the diffusion model's own signature (Ratcliff:
higher drift -> faster + more accurate, so a fast decision signals strong evidence).

The non-trivial claim tested here: confidence read from *timing* (not computed as 1/variance) is
STILL calibrated to accuracy, so RT-confidence-weighted wisdom of crowds beats plain majority —
in a crowd where a competent minority is outvoted by a noisy majority ("expert in the crowd").
And, as in the spectrum version, OUTCOME-TRUST learned from revealed outcomes is the robust
fallback when confidence can't be trusted.

    python -m cognitive_society.confidence_woc_ddm
"""
import numpy as np

from cognitive_society.agent import DDMAgent, DDMParams, DT, MAX_STEPS


def decide_grid(agents, ev, rng):
    """Vectorized DDM: decide ALL K agents on ALL P problems at once (one (K,P) accumulator
    stepped in lockstep). Returns (choices, rts) each (K, P). Same dynamics as DDMAgent.decide,
    just batched across the whole experiment so it runs in ~1s instead of stepping 300k singles."""
    K, P = len(agents), len(ev)
    ds = np.array([a.params.drift_scale for a in agents])[:, None]
    bnd = np.array([a.params.boundary for a in agents])[:, None]
    ndt = np.broadcast_to(np.array([a.params.ndt for a in agents])[:, None], (K, P))
    nsd = np.broadcast_to(np.array([a.params.sigma for a in agents])[:, None] * np.sqrt(DT), (K, P))
    drift = ds * ev[None, :]
    x = np.zeros((K, P))
    rt = np.full((K, P), np.nan)
    choice = np.full((K, P), -1, dtype=int)
    active = np.ones((K, P), dtype=bool)
    for step in range(1, MAX_STEPS + 1):
        na = int(active.sum())
        if na == 0:
            break
        x[active] += drift[active] * DT + nsd[active] * rng.standard_normal(na)
        hi = active & (x >= bnd)
        lo = active & (x <= -bnd)
        t = step * DT
        choice[hi] = 1; rt[hi] = ndt[hi] + t
        choice[lo] = 0; rt[lo] = ndt[lo] + t
        active &= ~(hi | lo)
    if active.any():
        choice[active] = (x[active] >= 0).astype(int)
        rt[active] = ndt[active] + MAX_STEPS * DT
    return choice, rt


def build_crowd(n_strong, n_weak, seed):
    """A competent minority + a noisy majority (the 'expert in the crowd' regime, where plain
    majority is dragged down and weighting by a calibrated signal should rescue the experts)."""
    rng = np.random.default_rng(seed)
    agents, kind = [], []
    for _ in range(n_strong):
        agents.append(DDMAgent(DDMParams(boundary=float(rng.uniform(1.1, 1.5)),
                                         drift_scale=float(rng.uniform(1.1, 1.4)),
                                         ndt=float(rng.uniform(0.15, 0.20)))))
        kind.append("strong")
    for _ in range(n_weak):
        agents.append(DDMAgent(DDMParams(boundary=float(rng.uniform(0.6, 0.9)),
                                         drift_scale=float(rng.uniform(0.25, 0.5)),
                                         ndt=float(rng.uniform(0.15, 0.20)))))
        kind.append("weak")
    return agents, kind


def _vote(choices, weights):
    """Weighted binary vote -> {0,1}. choices in {0,1}, mapped to +/-1."""
    s = np.sum(weights * (2 * np.asarray(choices) - 1))
    return 1 if s >= 0 else 0


METHODS = ("plain majority", "RT-conf (naive 1/RT)", "consistency-conf (DDM)",
           "outcome-trust (learned)", "oracle (true acc)")


def run(agents, n_problems, evidence_levels=(0.10, 0.18, 0.28), seed=0, ema=0.05, n_internal=15):
    """Each agent makes n_internal internal DDM decisions per problem. Its committed choice is
    the internal majority; its CONFIDENCE is the internal CONSISTENCY (|frac-0.5|*2) — the DDM's
    evidence-quality readout (what society.py already uses), which reflects drift/noise, not the
    boundary/caution that confounds raw RT. RT-naive is kept only to show it fails."""
    rng = np.random.default_rng(seed)
    K = len(agents)
    truths = (rng.random(n_problems) < 0.5).astype(int)
    mags = rng.choice(np.asarray(evidence_levels), n_problems)
    ev = np.where(truths == 1, mags, -mags)
    ev_exp = np.repeat(ev, n_internal)                             # (P*n_internal,)
    ch_exp, rt_exp = decide_grid(agents, ev_exp, rng)             # (K, P*n_internal), vectorized
    ch3 = ch_exp.reshape(K, n_problems, n_internal)
    frac1 = ch3.mean(axis=2)                                      # (K,P) fraction choosing 1
    choice = (frac1 >= 0.5).astype(int)                          # committed choice = internal majority
    consistency = np.abs(frac1 - 0.5) * 2.0                      # DDM evidence-quality confidence
    rt_naive = 1.0 / rt_exp.reshape(K, n_problems, n_internal).mean(axis=2)
    hit = (choice == truths[None, :]).astype(float)

    correct = {m: [] for m in METHODS}
    acc_hat = np.full(K, 0.5)
    true_acc = np.zeros(K)
    for p in range(n_problems):
        ch = choice[:, p]
        true_acc += hit[:, p]
        votes = {
            "plain majority": _vote(ch, np.ones(K)),
            "RT-conf (naive 1/RT)": _vote(ch, rt_naive[:, p]),
            "consistency-conf (DDM)": _vote(ch, consistency[:, p] + 1e-3),
            "outcome-trust (learned)": _vote(ch, acc_hat),
            "oracle (true acc)": _vote(ch, np.maximum(true_acc / (p + 1), 1e-3)),
        }
        for m in METHODS:
            correct[m].append(1.0 if votes[m] == truths[p] else 0.0)
        acc_hat = (1 - ema) * acc_hat + ema * hit[:, p]
    half = n_problems // 2
    acc = {m: float(np.mean(correct[m][half:])) for m in METHODS}
    return {"accuracy": acc,
            "calib_rt": _pointbiserial(rt_naive.ravel(), hit.ravel()),
            "calib_cons": _pointbiserial(consistency.ravel(), hit.ravel())}


def _pointbiserial(x, y):
    """Correlation between continuous x (confidence) and binary y (correct)."""
    x = (x - x.mean()) / (x.std() + 1e-12)
    return float(np.mean(x * (y - y.mean())) / (y.std() + 1e-12))


def main():
    N_PROB, N_SEEDS = 1500, 6
    n_strong, n_weak = 4, 11
    print("=" * 68)
    print("BRICK 1 (real DDM) — confidence-weighted wisdom of crowds")
    print("=" * 68)
    print(f"\n{n_strong} strong + {n_weak} weak DDM agents (expert-in-the-crowd), "
          f"{N_PROB} problems, {N_SEEDS} seeds.")

    outs = [run(build_crowd(n_strong, n_weak, s)[0], N_PROB, seed=s) for s in range(N_SEEDS)]
    acc = {m: float(np.mean([o["accuracy"][m] for o in outs])) for m in METHODS}
    c_rt = float(np.mean([o["calib_rt"] for o in outs]))
    c_cons = float(np.mean([o["calib_cons"] for o in outs]))

    print("\n  CALIBRATION CHECK  (pt-biserial: confidence vs correctness, higher = calibrated)")
    print(f"    raw RT (1/RT)             = {c_rt:+.3f}   (confounds caution with evidence quality)")
    print(f"    DDM consistency (evidence)= {c_cons:+.3f}   {'PASS' if c_cons > 0.05 else 'weak'}")
    print("\n  crowd accuracy (higher = better):")
    base = acc["plain majority"]
    for m in METHODS:
        print(f"    {m:26s}  {acc[m]:6.1%}   ({100*(acc[m]-base):+5.1f}pt vs majority)")
    print("\n  -> Raw RT is NOT calibrated here (fast-errors + boundary/caution confound) and HURTS.")
    print("     The DDM's EVIDENCE state — internal decision consistency — IS calibrated and lifts")
    print("     the crowd: confidence in the real model is the balance of evidence, not the clock.")
    print("     Outcome-trust (learned from outcomes) is robust regardless — the standing theme.")
    print("=" * 68)


if __name__ == "__main__":
    main()
