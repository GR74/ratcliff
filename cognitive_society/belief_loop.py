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
                conf_gains=None, biases=None, n_samples=20, seed=0, ema=0.05):
    """Iterated precision-weighted belief loop on graph `adj`. Returns collective + honest RMSE
    and the mean collective precision (to expose confident-but-wrong consensus)."""
    K = len(kappas)
    cg = np.ones(K) if conf_gains is None else np.asarray(conf_gains)
    bs = np.zeros(K) if biases is None else np.asarray(biases)
    honest = (bs == 0) & (cg == 1)
    rng = np.random.default_rng(seed)
    A = adj.astype(float)
    mse_hat = np.ones(K)                       # outcome-trust: learned realized MSE per agent
    col_sq, hon_sq, col_prec = [], [], []
    for _ in range(n_problems):
        theta = float(rng.uniform(-1.0, 1.0))
        e_priv = np.zeros(K); p_priv = np.zeros(K)
        for i in range(K):
            e_priv[i], p_priv[i] = private_read(theta, kappas[i], n_samples, rng, bs[i], cg[i])
        b, pi = e_priv.copy(), p_priv.copy()
        for _ in range(T):                    # the loop: propagate + precision-weighted update
            if gate == "naive":
                eff = np.ones(K)
            elif gate == "confidence":
                eff = pi
            else:                             # outcome-trust: realized precision replaces self-report
                eff = 1.0 / np.maximum(mse_hat, 1e-9)
            Weff = A * eff[None, :]
            num = p_priv * e_priv + coupling * (Weff @ b)
            den = p_priv + coupling * Weff.sum(1)
            b = num / den
            pi = den
        collective = float(np.sum(pi * b) / np.sum(pi))
        col_sq.append((collective - theta) ** 2)
        hon_sq.append(float(np.mean((b[honest] - theta) ** 2)) if honest.any() else np.nan)
        col_prec.append(float(pi.mean()))
        mse_hat = (1 - ema) * mse_hat + ema * (b - theta) ** 2      # learn from revealed outcome
    half = n_problems // 2
    return {
        "collective_rmse": float(np.sqrt(np.mean(col_sq[half:]))),
        "honest_rmse": float(np.sqrt(np.nanmean(hon_sq[half:]))),
        "mean_precision": float(np.mean(col_prec[half:])),
    }


def main():
    N_PROB, N_SEEDS = 1500, 6
    K, deg = 24, 4
    rng0 = np.random.default_rng(0)
    kappas = rng0.uniform(0.5, 3.0, K)
    adj = topo.watts_strogatz(K, deg, 0.3, np.random.default_rng(1))

    def avg(gate, T, conf_gains=None, biases=None):
        outs = [belief_loop(kappas, adj, N_PROB, T=T, gate=gate, conf_gains=conf_gains,
                            biases=biases, seed=s) for s in range(N_SEEDS)]
        return {k: float(np.mean([o[k] for o in outs])) for k in outs[0]}

    print("=" * 68)
    print("BRICK 2 — iterated belief<->evidence<->confidence loop (active inference)")
    print("=" * 68)
    print(f"\nK={K} agents on a Watts-Strogatz graph (deg={deg}), {N_PROB} problems, {N_SEEDS} seeds.")

    # --- Part A: iteration + gating on a healthy crowd ---------------------------------
    print("\n  PART A — healthy crowd: what does iterating the loop actually do?")
    print(f"    {'gate':>14s} {'RMSE T=1':>10s} {'RMSE T=6':>10s} {'precision T=6':>14s}")
    for gate in ("naive", "confidence"):
        a1, a6 = avg(gate, T=1), avg(gate, T=6)
        print(f"    {gate:>14s} {a1['collective_rmse']:>10.4f} {a6['collective_rmse']:>10.4f}"
              f" {a6['mean_precision']:>14.0f}")
    print("    -> HONEST NULL: iterating does NOT improve accuracy on a healthy crowd, and it")
    print("       INFLATES precision ~100-300x (double-counting -> overconfidence; the known")
    print("       Golub-Jackson / DeGroot naive-social-learning pathology). confidence ~ naive here.")
    print("       The loop's payoff is not accuracy on a healthy crowd — it is PROPAGATION (Part B).")

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
    print("=" * 68)


if __name__ == "__main__":
    main()
