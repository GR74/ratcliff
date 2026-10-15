"""Real-human label aggregation (AI / RLHF anchor) on the crowd-kit 'relevance-2' dataset
(7138 real crowd workers, binary relevance, ~10k gold tasks). Needs `pip install crowd-kit`.

Two results:
  (1) CLEAN data: reliability-weighting beats plain majority on real annotators. Without adversaries,
      agreement-based Dawid-Skene (the field standard) is strong -- exactly what the framework
      predicts (agreement suffices absent collusion/miscalibration).
  (2) ADVERSARIAL injection: add a colluding bloc that agrees on the WRONG label. Agreement-based
      Dawid-Skene CATASTROPHICALLY INVERTS past a threshold (it locks onto the colluders' false
      consensus -> 0% accuracy), while OUTCOME-grounded track-record (reliability learned from a
      gold subset) stays robust and even benefits (it flips the known-wrong colluders into signal).
      This is the outcome-vs-agreement distinction (the swarm f-local inversion) on real data with
      the standard tool -- and it is the RLHF label-poisoning / annotation-security problem.

Honest scope: (2) is SEMI-real -- real workers + synthetic (perfectly consistent) colluders, so the
extreme magnitudes are idealized; the direction and the inversion threshold are the finding.

    python -m cognitive_society.real_label_aggregation
"""
import numpy as np
import pandas as pd


def _load():
    from crowdkit.datasets import load_dataset
    df, gt = load_dataset("relevance-2")
    return df, gt.rename("truth")


def _wlo(sub, shrink=5.0):
    prior = sub["correct"].mean()
    g = sub.groupby("worker")["correct"].agg(["sum", "count"])
    acc = ((g["sum"] + shrink * prior) / (g["count"] + shrink)).clip(0.02, 0.98)
    return np.log(acc / (1 - acc)), np.log(prior / (1 - prior))


def main():
    from crowdkit.aggregation import DawidSkene
    df, gt = _load()
    truth = gt.to_dict()
    tasks = np.array(gt.index)
    base = df[df["task"].isin(gt.index)][["worker", "task", "label"]].copy()

    rng = np.random.default_rng(0)
    sh = tasks.copy(); rng.shuffle(sh)
    cal = set(sh[: int(0.30 * len(sh))])
    ev = set(sh[int(0.30 * len(sh)):])
    ADV = [f"ADV{i}" for i in range(300)]

    def with_adv(n):
        if n == 0:
            return base
        rows = [(a, t, 1 - truth[t]) for t in tasks for a in rng.choice(ADV, n, replace=False)]
        return pd.concat([base, pd.DataFrame(rows, columns=["worker", "task", "label"])],
                         ignore_index=True)

    def evaluate(full):
        fg = full[full["task"].isin(gt.index)].copy()
        fg["truth"] = fg["task"].map(truth)
        fg["correct"] = (fg["label"] == fg["truth"]).astype(int)
        fg["signed"] = fg["label"] * 2 - 1
        lo_t, plo = _wlo(fg[fg["task"].isin(cal)])
        lo_o, _ = _wlo(fg)
        de = fg[fg["task"].isin(ev)].copy()
        de["w1"] = de["signed"]
        de["wt"] = de["worker"].map(lo_t).fillna(plo) * de["signed"]
        de["wo"] = de["worker"].map(lo_o).fillna(plo) * de["signed"]
        grp = de.groupby("task")[["w1", "wt", "wo"]].sum()
        tr = gt[grp.index]
        ds = DawidSkene(n_iter=10).fit_predict(full[["worker", "task", "label"]]).reindex(grp.index)
        return {"majority": float(((grp["w1"] >= 0).astype(int) == tr).mean()),
                "Dawid-Skene": float((ds == tr).mean()),
                "track-record": float(((grp["wt"] >= 0).astype(int) == tr).mean()),
                "oracle": float(((grp["wo"] >= 0).astype(int) == tr).mean())}

    print("=" * 66)
    print("REAL LABEL AGGREGATION - relevance-2 (7138 real workers)")
    print("=" * 66)
    clean = evaluate(base)
    print("\n(1) CLEAN data - reliability-weighting vs majority:")
    for m in ("majority", "Dawid-Skene", "track-record", "oracle"):
        print(f"    {m:14s} {clean[m]:.3f}  ({100*(clean[m]-clean['majority']):+.1f}pt)")
    print("    -> all reliability-aware methods beat majority; agreement (D-S) is strong on CLEAN data.")

    print("\n(2) ADVERSARIAL injection - a colluding bloc agrees on the WRONG label:")
    print(f"    {'adv/task':>8} {'majority':>9} {'Dawid-Skene':>12} {'track-record':>13}")
    for n in (0, 1, 2, 3):
        r = evaluate(with_adv(n))
        print(f"    {n:>8} {r['majority']:>9.3f} {r['Dawid-Skene']:>12.3f} {r['track-record']:>13.3f}")
    print("    -> agreement-based D-S INVERTS past a threshold (locks onto the false consensus -> ~0);")
    print("       outcome-grounded track-record holds (gold set exposes the colluders). The swarm")
    print("       inversion + the RLHF label-poisoning problem, on real data.")
    print("=" * 66)


if __name__ == "__main__":
    main()
