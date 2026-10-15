"""AI companion 2 - trustworthy aggregation of noisy training labels (RLHF / crowdsourcing).

Many annotators label the same items with varying reliability; you aggregate into clean training
labels. Aggregators (log-odds weighted majority = the optimal combination of independent labelers):
  - majority      : unweighted vote (the standard baseline).
  - confidence    : weight by the annotator's SELF-REPORTED confidence on each item.
  - track-record  : weight by accuracy on a small GOLD set (outcome-grounded; outcomes observable).
  - oracle        : weight by true reliability (unattainable ceiling).

Results:
  (A) CALIBRATED annotators -> confidence-weighting beats majority and tracks the oracle: cleaner
      labels for free (you don't need gold).
  (B) Add confident-but-CARELESS annotators (chance accuracy, high self-reported confidence) ->
      confidence-weighting DEGRADES BELOW majority (it up-weights the loud-careless and poisons the
      labels), while track-record stays robust (the gold set reveals they're unreliable).
  => the training-data calibration trap: don't weight labels by self-reported confidence unless it's
     calibrated; use a gold set to learn who to trust. Directly relevant to RLHF preference data.

    python -m cognitive_society.label_aggregation
"""
import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


METHODS = ("majority", "confidence", "track-record", "oracle")


def run(skill, n_items=6000, gold_frac=0.1, conf_gains=None, seed=0):
    """skill[k] (logit): annotator competence. p_correct = sigmoid(skill - item_difficulty).
    conf_gains[k]: inflation of self-reported confidence (miscalibration). Returns per-method
    label accuracy on the non-gold items = training-data quality."""
    skill = np.asarray(skill, float)
    K = len(skill)
    cg = np.zeros(K) if conf_gains is None else np.asarray(conf_gains, float)
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, n_items) * 2 - 1                 # +/-1 true labels
    diff = rng.normal(0.0, 1.0, n_items)                       # per-item difficulty
    p_correct = _sig(skill[:, None] - diff[None, :])          # (K, n_items)
    correct = (rng.random((K, n_items)) < p_correct).astype(int)
    label = np.where(correct == 1, truth[None, :], -truth[None, :])
    conf = _sig((skill[:, None] + cg[:, None]) - diff[None, :])  # reported confidence (cg inflates)

    n_gold = int(gold_frac * n_items)
    gold, ev = slice(0, n_gold), slice(n_gold, n_items)
    track = np.clip((label[:, gold] == truth[None, gold]).mean(1), 0.01, 0.99)   # gold accuracy

    def agg(w):
        w = np.asarray(w, float)
        if w.ndim == 1:
            w = w[:, None]
        s = (w * label)[:, ev].sum(0)                          # log-odds weighted vote
        return np.where(s >= 0, 1, -1)

    out = {
        "majority": agg(np.ones(K)),
        "confidence": agg(_logit(conf)),
        "track-record": agg(_logit(track)),
        "oracle": agg(_logit(p_correct)),
    }
    return {m: float((v == truth[ev]).mean()) for m, v in out.items()}


def main():
    K, N, SEEDS = 10, 6000, 6
    print("=" * 60)
    print("LABEL AGGREGATION - clean training labels from noisy annotators")
    print("=" * 60)

    def agg(runs):
        return {m: float(np.mean([r[m] for r in runs])) for m in METHODS}

    A = agg([run(np.random.default_rng(s).uniform(0.3, 2.0, K), N, seed=s) for s in range(SEEDS)])
    print(f"\n(A) {K} CALIBRATED annotators (varying skill):")
    for m in METHODS:
        print(f"    {m:14s} label accuracy {A[m]:.3f}")
    print("    -> confidence-weighting beats majority and tracks the oracle: cleaner labels, no gold.")

    def build_B(s):
        rng = np.random.default_rng(s)
        skill = np.concatenate([rng.uniform(0.3, 2.0, K), np.zeros(5)])   # +5 careless (chance) annotators
        cg = np.concatenate([np.zeros(K), np.full(5, 4.0)])               # ...reporting HIGH confidence
        return run(skill, N, conf_gains=cg, seed=s)

    B = agg([build_B(s) for s in range(SEEDS)])
    print(f"\n(B) + 5 confident-but-CARELESS annotators (chance accuracy, loud confidence):")
    for m in METHODS:
        print(f"    {m:14s} label accuracy {B[m]:.3f}")
    print(f"    -> confidence-weighting POISONS the labels ({A['confidence']:.3f} -> {B['confidence']:.3f},"
          f" below majority {B['majority']:.3f}); track-record stays clean (gold set exposes them).")
    print("       The training-data calibration trap - directly relevant to RLHF preference labels.")
    print("=" * 60)


if __name__ == "__main__":
    main()
