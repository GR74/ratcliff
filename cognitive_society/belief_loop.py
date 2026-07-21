"""Brick 2 — the belief<->evidence<->confidence LOOP (active inference), iterated on a network.

Direction: docs/notes/2026-07-21-cognitive-digital-twin-research-program.md.

Brick 1 was ONE-SHOT pooling. Brick 2 closes the loop: agents on a graph hold a continuous
belief b_i and a precision pi_i (confidence), and each round update by precision-weighted
Bayesian revision that folds in (a) their own private evidence and (b) neighbours' beliefs,
weighted by neighbours' communicated precision:

    b_i <- ( p_priv_i*e_priv_i + c * sum_j Weff_ij * b_j ) / ( p_priv_i + c * sum_j Weff_ij )
    pi_i <- p_priv_i + c * sum_j Weff_ij                        (precision accumulates)

This IS active inference / the Bayesian brain: priors + precision-weighted evidence -> posterior,
iterated so information propagates over the graph. Neighbours' outputs are just more evidence in
the loop, which is why social influence needs no separate machinery.

Gating (the effective neighbour weight Weff_ij = adj_ij * eff_j):
  - naive        : eff = 1        (plain DeGroot averaging; ignores confidence)
  - confidence   : eff = pi_j     (precision/confidence-weighted; the active-inference default)
  - outcome-trust: eff = learned realized precision (from revealed outcomes; immune to bad self-report)

Honest result: on a HEALTHY crowd, iterating does NOT improve accuracy and it massively inflates
precision (double-counting -> overconfidence; the known Golub-Jackson / DeGroot naive-social-
learning pathology). The loop's real consequence is PROPAGATION: a deluded cluster (shared false
belief + aberrant precision) spreads a confident-but-WRONG consensus (echo chamber) that one-shot
pooling can't form -- and outcome-trust defends by catching that their confidence doesn't predict
truth. So the loop is what makes both echo chambers AND their outcome-grounded cure possible.

    python -m cognitive_society.belief_loop
"""
import numpy as np

from cognitive_society import topology as topo


def private_read(theta, kappa, n_samples, rng, bias=0.0, conf_gain=1.0):
    """Private evidence about theta -> (estimate, precision). Same DDM-family read as Brick 1."""
    sigma = 1.0 / kappa
    s = theta + bias + rng.normal(0.0, sigma, n_samples)
    est = float(s.mean())
    prec = conf_gain * (n_samples / max(float(s.var(ddof=1)), 1e-9))
    return est, max(prec, 1e-9)


def belief_loop(kappas, adj, n_problems, T=6, coupling=0.5, gate="confidence",
                conf_gains=None, biases=None, n_samples=20, seed=0, ema=0.05,
                bound_precision=False, homeostatic=False, target_spread=0.35):
    """Iterated precision-weighted belief loop on graph `adj`. Returns collective + honest RMSE
    and the mean collective precision (to expose confident-but-wrong consensus).

    bound_precision: if True, cap each agent's precision at the total INDEPENDENT private evidence
    in the system (sum of private precisions). This is the information-theoretic ceiling: the loop
    can re-mix evidence but cannot manufacture more of it. Fixes the double-counting overconfidence
    (Golub-Jackson) of naive iterated updating."""
    K = len(kappas)
    cg = np.ones(K) if conf_gains is None else np.asarray(conf_gains)
    bs = np.zeros(K) if biases is None else np.asarray(biases)
    honest = (bs == 0) & (cg == 1)
    rng = np.random.default_rng(seed)
    A = adj.astype(float)
    mse_hat = np.ones(K)                       # outcome-trust: learned realized MSE per agent
    col_sq, hon_sq, col_prec, col_spr = [], [], [], []
    for _ in range(n_problems):
        theta = float(rng.uniform(-1.0, 1.0))
        e_priv = np.zeros(K); p_priv = np.zeros(K)
        for i in range(K):
            e_priv[i], p_priv[i] = private_read(theta, kappas[i], n_samples, rng, bs[i], cg[i])
        b, pi = e_priv.copy(), p_priv.copy()
        info_ceiling = float(p_priv.sum())    # total independent evidence in the system
        for _ in range(T):                    # the loop: propagate + precision-weighted update
            if gate == "naive":
                eff = np.ones(K)
            elif gate == "confidence":
                eff = pi
            else:                             # outcome-trust: realized precision replaces self-report
                eff = 1.0 / np.maximum(mse_hat, 1e-9)
            Weff = A * eff[None, :]
            if homeostatic:
                # RSL-style set-point control: as an agent's LOCAL consensus tightens below the
                # diversity set-point, it turns DOWN social coupling (holds its own view) -> a
                # "doubter" reflex that keeps the population from collapsing to unanimity, with
                # NO outcomes needed. (This is the belief-loop analog of RSL synaptic scaling.)
                deg = np.maximum(A.sum(1), 1.0)
                nb_mean = (A @ b) / deg
                local_spread = np.sqrt(np.maximum((A @ (b ** 2)) / deg - nb_mean ** 2, 0.0))
                c = coupling * np.minimum(local_spread / target_spread, 1.0)
            else:
                c = coupling
            num = p_priv * e_priv + c * (Weff @ b)
            den = p_priv + c * Weff.sum(1)
            b = num / den
            pi = np.minimum(den, info_ceiling) if bound_precision else den
        collective = float(np.sum(pi * b) / np.sum(pi))
        col_sq.append((collective - theta) ** 2)
        hon_sq.append(float(np.mean((b[honest] - theta) ** 2)) if honest.any() else np.nan)
        col_prec.append(float(pi.mean()))
        col_spr.append(float(np.std(b[honest])) if honest.any() else np.nan)
        mse_hat = (1 - ema) * mse_hat + ema * (b - theta) ** 2      # learn from revealed outcome
    half = n_problems // 2
    return {
        "collective_rmse": float(np.sqrt(np.mean(col_sq[half:]))),
        "honest_rmse": float(np.sqrt(np.nanmean(hon_sq[half:]))),
        "mean_precision": float(np.mean(col_prec[half:])),
        "belief_spread": float(np.nanmean(col_spr[half:])),
    }


def main():
    N_PROB, N_SEEDS = 1500, 6
    K, deg = 24, 4
    rng0 = np.random.default_rng(0)
    kappas = rng0.uniform(0.5, 3.0, K)
    adj = topo.watts_strogatz(K, deg, 0.3, np.random.default_rng(1))

    def avg(gate, T, conf_gains=None, biases=None, bound=False, homeo=False):
        outs = [belief_loop(kappas, adj, N_PROB, T=T, gate=gate, conf_gains=conf_gains,
                            biases=biases, bound_precision=bound, homeostatic=homeo, seed=s)
                for s in range(N_SEEDS)]
        return {k: float(np.mean([o[k] for o in outs])) for k in outs[0]}

    print("=" * 68)
    print("BRICK 2 — iterated belief<->evidence<->confidence loop (active inference)")
    print("=" * 68)
    print(f"\nK={K} agents on a Watts-Strogatz graph (deg={deg}), {N_PROB} problems, {N_SEEDS} seeds.")

    # --- Part A: the double-counting overconfidence, and the fix -----------------------
    print("\n  PART A — healthy crowd: iterating over-counts confidence, and the fix")
    print(f"    {'setup':>26s} {'RMSE T=6':>10s} {'precision T=6':>14s}")
    for label, gate, bound in (("confidence (naive loop)", "confidence", False),
                               ("confidence + info-bound", "confidence", True),
                               ("outcome-trust", "outcome-trust", False)):
        a = avg(gate, T=6, bound=bound)
        print(f"    {label:>26s} {a['collective_rmse']:>10.4f} {a['mean_precision']:>14.0f}")
    print("    -> the naive loop keeps accuracy fine but INFLATES precision ~100x (double-counting")
    print("       -> overconfidence; Golub-Jackson). Capping precision at the total independent")
    print("       evidence FIXES it: same accuracy, honest confidence. (Outcome-trust is grounded")
    print("       in realized accuracy, so it self-limits too.) Accuracy itself is already saturated")
    print("       on this small graph — iteration's real job is PROPAGATION (Part B).")

    # --- Part B: a deluded cluster propagates through the loop -------------------------
    print("\n  PART B — a deluded CLUSTER (contiguous, shared false belief + aberrant precision)")
    print("  deluded agents: private belief biased +0.8, confidence inflated x10")
    print(f"    {'#deluded':>9s} {'gate':>14s} {'honest RMSE':>12s} {'mean precision':>15s}")
    for n_bad in (0, 4, 8):
        idx = np.arange(n_bad)                 # contiguous on the ring lattice -> an echo chamber
        cg, bs = np.ones(K), np.zeros(K)
        cg[idx], bs[idx] = 10.0, 0.8
        for gate in ("confidence", "outcome-trust"):
            r = avg(gate, T=6, conf_gains=cg, biases=bs)
            print(f"    {n_bad:>7d}/{K} {gate:>14s} {r['honest_rmse']:>12.4f} {r['mean_precision']:>15.1f}")
    print("    -> confidence-gating lets the delusion PROPAGATE — honest agents' beliefs get dragged")
    print("       off-truth with HIGH precision (a confident-but-wrong consensus / echo chamber).")
    print("       Outcome-trust keeps honest agents accurate: it catches that the deluded cluster's")
    print("       confidence does not predict truth. (Caveat: needs observable outcomes.)")

    # --- Part C: HOMEOSTATIC diversity — the outcome-free defense (ported from RSL) ----
    print("\n  PART C — homeostatic diversity ('always doubters'): defend WITHOUT outcomes")
    print("  8/24 deluded cluster; RSL set-point principle ported to the belief loop")
    cg2, bs2 = np.ones(K), np.zeros(K)
    cg2[:8], bs2[:8] = 10.0, 0.8
    print(f"    {'setup':>32s} {'honest RMSE':>12s} {'belief spread':>14s}")
    for label, kw in (("confidence (no defense)", dict(gate="confidence")),
                      ("confidence + homeostasis", dict(gate="confidence", homeo=True)),
                      ("outcome-trust (needs outcomes)", dict(gate="outcome-trust"))):
        r = avg(T=6, conf_gains=cg2, biases=bs2, **kw)
        print(f"    {label:>32s} {r['honest_rmse']:>12.4f} {r['belief_spread']:>14.4f}")
    print("    -> homeostasis keeps a diversity 'gap' (doubters persist) and blunts the delusion")
    print("       with NO outcome signal — the RSL set-point principle: don't collapse to unanimity.")
    print("       Outcome-trust defends harder but needs observable outcomes; homeostasis is the")
    print("       outcome-FREE complement. Health = maintained doubt.")
    print("=" * 68)


if __name__ == "__main__":
    main()
