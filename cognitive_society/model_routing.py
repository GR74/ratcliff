"""AI companion result — routing among multiple models/agents (the LLM-router setting).

Given several models (e.g. LLMs), each an expert on different task types, pick the best model per
query. Two strategies from the framework:
  - confidence routing : send the query to the model most CONFIDENT on it.
  - track-record routing: send it to the model with the best LEARNED accuracy on that task type
                          (outcome-grounded, from a dev set).

Results:
  (A) With CALIBRATED models, confidence routing routes each query to the right specialist and
      beats the best single (generalist) model -- the division-of-labor result, for models.
  (B) Add ONE weak-but-OVERCONFIDENT ("junk") model and confidence routing INVERTS -- it keeps
      picking the loud-wrong model. Track-record routing is immune: it learns the junk model is
      inaccurate and avoids it. => the calibration trap for LLM routers, made concrete:
      route-by-confidence is only safe if confidence is calibrated; else you need outcome-grounding.

    python -m cognitive_society.model_routing
"""
import numpy as np


def make_models(K, D, rng, expert_per=2, hi=3.0, lo=-1.0):
    """skill[k,t] (logit): high on a model's expert task-types, low elsewhere. Every type has >=1
    expert. p_correct = sigmoid(skill - query_difficulty)."""
    skill = np.full((K, D), lo)
    for k in range(K):
        for t in rng.choice(D, expert_per, replace=False):
            skill[k, t] = hi
    for t in range(D):
        if skill[:, t].max() < hi:
            skill[rng.integers(K), t] = hi
    return skill


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


METHODS = ("random", "best-single model", "confidence", "track-record", "oracle")


def run(skill, n_queries=8000, conf_gains=None, seed=0, ema=0.03):
    K, D = skill.shape
    cg = np.zeros(K) if conf_gains is None else np.asarray(conf_gains, float)
    rng = np.random.default_rng(seed)
    track = np.full((K, D), 0.5)                 # learned per-(model,type) accuracy (dev-set style)
    half = n_queries // 2
    picks = {m: [] for m in ("random", "confidence", "track-record", "oracle")}
    model_correct_all = np.zeros(K)              # overall accuracy (to pick the best single model)
    model_correct_sh = np.zeros(K); n_sh = 0     # second-half per-model accuracy
    for i in range(n_queries):
        t = rng.integers(D)
        d = rng.normal(0.0, 1.0)                  # query difficulty (shared across models)
        p_true = _sig(skill[:, t] - d)            # true correctness prob per model
        correct = (rng.random(K) < p_true).astype(float)
        conf = _sig((skill[:, t] + cg) - d)       # reported confidence (cg inflates it)
        if i >= half:
            picks["random"].append(correct[rng.integers(K)])
            picks["confidence"].append(correct[int(np.argmax(conf))])
            picks["track-record"].append(correct[int(np.argmax(track[:, t]))])
            picks["oracle"].append(correct[int(np.argmax(p_true))])
            model_correct_sh += correct; n_sh += 1
        track[:, t] = (1 - ema) * track[:, t] + ema * correct   # observe outcomes -> learn accuracy
        model_correct_all += correct
    acc = {m: float(np.mean(picks[m])) for m in picks}
    best = int(np.argmax(model_correct_all))
    acc["best-single model"] = float(model_correct_sh[best] / n_sh)
    return acc


def main():
    K, D, N, SEEDS = 8, 6, 8000, 5
    print("=" * 60)
    print("MODEL ROUTING - confidence vs track-record (the LLM-router setting)")
    print("=" * 60)

    def agg(runs):
        return {m: float(np.mean([r[m] for r in runs])) for m in METHODS}

    A = agg([run(make_models(K, D, np.random.default_rng(s)), N, seed=s) for s in range(SEEDS)])
    print(f"\n(A) {K} CALIBRATED specialist models (each expert on 2 of {D} task types):")
    for m in METHODS:
        print(f"    {m:22s} accuracy {A[m]:.3f}")
    print("    -> confidence routing sends each query to the right specialist and beats the best")
    print("       single (generalist) model -- division of labor, for models.")

    def build_B(s):
        rng = np.random.default_rng(s)
        skill = np.vstack([make_models(K, D, rng), np.full(D, -1.5)])   # + a weak 'junk' model
        cg = np.zeros(K + 1); cg[K] = 6.0                               # ...that shouts confidence
        return skill, cg

    Bruns = []
    for s in range(SEEDS):
        skill, cg = build_B(s)
        Bruns.append(run(skill, N, conf_gains=cg, seed=s))
    B = agg(Bruns)
    print(f"\n(B) + 1 weak-but-OVERCONFIDENT 'junk' model added to the ensemble:")
    for m in METHODS:
        print(f"    {m:22s} accuracy {B[m]:.3f}")
    print(f"    -> confidence routing COLLAPSES ({A['confidence']:.3f} -> {B['confidence']:.3f}): it")
    print("       keeps routing to the loud-wrong model. Track-record routing is immune (learns the")
    print("       junk model is inaccurate). The calibration trap for LLM routers, concretely:")
    print("       route-by-confidence is safe ONLY if confidence is calibrated -- else use outcomes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
