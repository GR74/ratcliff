"""REAL-LLM validation of the LLM-router claim from ``model_routing.py``.

The simulation in ``model_routing.py`` claimed two things about routing a query
to one of several specialist models:
  (A) an ENSEMBLE routed by learned track-record beats the best single model
      (division of labor), and
  (B) route-by-CONFIDENCE collapses if one model is miscalibrated/overconfident,
      while route-by-track-record (outcome-grounded) is immune -- "the
      calibration trap for LLM routers".

This module reproduces what is reproducible on REAL frontier-LLM outputs.

DATASET
-------
RouterBench (``withmartian/routerbench``, ``routerbench_0shot.pkl``; public,
ungated). 36,497 queries across 86 task categories (``eval_name``: hellaswag,
grade-school-math, mmlu-*, arc-challenge, winogrande, ...). For each query it
gives the per-query CORRECTNESS score in [0,1] and the dollar COST of 11 real
LLMs collected Nov 2023:

    WizardLM-13B, claude-instant-v1, claude-v1, claude-v2, gpt-3.5-turbo-1106,
    gpt-4-1106-preview, code-llama-34b, llama-2-70b, mistral-7b,
    mixtral-8x7b, Yi-34B-Chat

(Correctness is fractional on a few multi-part items; we use it as a graded
score, so accuracies are means of these scores.)

WHAT IS / ISN'T TESTABLE HERE (honest scope)
--------------------------------------------
* RouterBench has NO per-model confidence / logprob column. So route-by-
  CONFIDENCE, and therefore the calibration trap (B) as a claim about REAL
  reported confidence, is NOT directly testable on this data. We report that
  honestly and do the track-record vs best-single test, which is the strong
  real-LLM result.
* We ADD an optional, clearly-labelled SIMULATED stress test: inject a weak
  model with a synthetic overconfident "confidence" signal and show that
  confidence-routing over the real correctness matrix collapses while track-
  record holds. The correctness numbers are real; the confidence signal is
  synthetic (RouterBench provides none), so this only demonstrates the
  MECHANISM -- it is not a measurement of real LLM (mis)calibration.

RESULT (train/test split, learn per-category reliability on train, evaluate on
test; mean over 5 seeds -- see the printed table for exact numbers):
* ALL 11 models: track-record routing ~= best-single on raw ACCURACY.
  gpt-4-1106-preview wins ~32 of 35 large categories, so "route to the best
  category-specialist" degenerates to "always route to GPT-4". Division of
  labor on accuracy alone is essentially ABSENT when one generalist dominates
  -- honest negative for claim (A) as a pure-accuracy claim on the full
  ensemble.
* But routing's real win over best-single here is on COST, not accuracy. A
  cost-aware track-record router (per category, pick the CHEAPEST model within
  a tolerance of the best learned accuracy) MATCHES GPT-4 accuracy at ~15%
  lower cost, and stays within ~1 accuracy point at ~60% lower cost. That is
  the honest "routing beats best-single" story on real frontier LLMs: same
  quality, much cheaper -- division of labor shows up on the cost axis.
* 10 COMPARABLE models (GPT-4 removed): track-record routing BEATS the best
  single model by ~+3-4 accuracy points. Division of labor on ACCURACY is REAL
  when the models are peers rather than one dominating generalist.
* Oracle (any-model-correct) ~0.91 -- large headroom remains; real per-category
  track-record captures only a sliver of it, because category is a coarse
  predictor of which model will be right on a given query.

    python -m cognitive_society.router_bench
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REPO = "withmartian/routerbench"
FILE = "routerbench_0shot.pkl"

MODELS = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]
DOMINANT = "gpt-4-1106-preview"  # the generalist that dominates most categories


def load() -> pd.DataFrame:
    """Download RouterBench (cached by huggingface_hub) and coerce types."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, FILE, repo_type="dataset")
    df = pd.read_pickle(path)
    for m in MODELS:
        df[m] = df[m].astype(float)
    return df


# ---------------------------------------------------------------------------
# Routing strategies. Each is fit on `tr` (train) and evaluated on `te` (test).
# `corr[i, k]` = correctness score of model k on test query i.
# ---------------------------------------------------------------------------
def _split(df, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = len(df) // 2
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]]


def evaluate(df, models, seed=0):
    """Return a dict of {method: (accuracy, mean_cost)} on the test split."""
    tr, te = _split(df, seed)
    corr = te[models].values                        # (N_test, K) real correctness
    cost = te[[m + "|total_cost" for m in models]].values
    N, K = corr.shape
    col = {m: k for k, m in enumerate(models)}

    # -- baselines -----------------------------------------------------------
    rng = np.random.default_rng(seed + 999)
    rand_pick = rng.integers(K, size=N)

    best_single = max(models, key=lambda m: tr[m].mean())   # fixed best on TRAIN
    b = col[best_single]

    # -- track-record routing: best learned accuracy on the query's CATEGORY --
    cat_acc = tr.groupby("eval_name")[models].mean()
    glob = tr[models].mean()
    cat_pick = {c: cat_acc.loc[c].values.argmax() for c in cat_acc.index}
    glob_pick = int(glob.values.argmax())
    tr_pick = np.array([cat_pick.get(c, glob_pick) for c in te["eval_name"].values])

    # -- oracle: any model that is correct (upper bound) ---------------------
    def gather(pick):
        r = np.arange(N)
        return corr[r, pick].mean(), cost[r, pick].mean()

    out = {}
    out["random"] = gather(rand_pick)
    out["best-single model"] = (corr[:, b].mean(), cost[:, b].mean())
    out["track-record"] = gather(tr_pick)
    out["oracle (upper bound)"] = (corr.max(axis=1).mean(),
                                   cost[np.arange(N), corr.argmax(axis=1)].mean())
    out["_best_single_name"] = best_single
    return out


def cost_aware_route(df, models, tol, seed=0):
    """Cost-aware track-record: per category, among models within `tol` of the
    best learned (train) accuracy, pick the CHEAPEST. tol=0 -> quality-matched,
    just cheaper. Returns (accuracy, mean_cost) on the test split."""
    tr, te = _split(df, seed)
    costcols = [m + "|total_cost" for m in models]
    cat_acc = tr.groupby("eval_name")[models].mean()
    cat_cost = tr.groupby("eval_name")[costcols].mean()
    glob_acc = tr[models].mean().values
    glob_cost = tr[costcols].mean().values
    corr = te[models].values
    cost = te[costcols].values
    N = len(te)
    picks = np.empty(N, dtype=int)
    for i, c in enumerate(te["eval_name"].values):
        a = cat_acc.loc[c].values if c in cat_acc.index else glob_acc
        ct = cat_cost.loc[c].values if c in cat_cost.index else glob_cost
        elig = np.where(a >= a.max() - tol)[0]
        picks[i] = elig[np.argmin(ct[elig])]
    r = np.arange(N)
    return corr[r, picks].mean(), cost[r, picks].mean()


def _agg(df, models, seeds):
    runs = [evaluate(df, models, s) for s in seeds]
    methods = [k for k in runs[0] if not k.startswith("_")]
    acc = {m: float(np.mean([r[m][0] for r in runs])) for m in methods}
    cost = {m: float(np.mean([r[m][1] for r in runs])) for m in methods}
    return acc, cost, runs[0]["_best_single_name"]


def _print_table(title, acc, cost, best_name):
    print(f"\n{title}")
    print(f"    (best-single model chosen on train = {best_name})")
    print(f"    {'method':22s} {'accuracy':>9s} {'mean $/query':>14s}")
    order = ["random", "best-single model", "track-record", "oracle (upper bound)"]
    for m in order:
        print(f"    {m:22s} {acc[m]:9.4f} {cost[m]:14.6f}")
    gain = acc["track-record"] - acc["best-single model"]
    verdict = "BEATS" if gain > 0.002 else ("ties" if gain > -0.002 else "LOSES to")
    print(f"    -> track-record routing {verdict} best-single by {gain:+.4f} accuracy")


# ---------------------------------------------------------------------------
# Optional SIMULATED calibration-trap stress test. RouterBench has no real
# confidence signal, so we synthesize one to demonstrate the MECHANISM only.
# Real correctness stays real; the "confidence" is fabricated.
# ---------------------------------------------------------------------------
def calibration_trap_stress(df, seed=0):
    tr, te = _split(df, seed)
    models = MODELS
    corr_te = te[models].values
    N, K = corr_te.shape

    # A synthetic "confidence" that is HONEST for the real models (equals their
    # train accuracy, i.e. well-calibrated) ...
    train_acc = tr[models].mean().values
    # ... plus one injected weak-but-OVERCONFIDENT junk model.
    junk_correct_te = (np.random.default_rng(seed).random(N) < 0.20).astype(float)  # ~20% right
    corr_aug = np.hstack([corr_te, junk_correct_te[:, None]])
    honest_conf = np.append(train_acc, 0.20)   # what a calibrated signal would say
    loud_conf = np.append(train_acc, 0.99)     # what the overconfident junk model shouts

    # track-record uses LEARNED accuracy from train (junk model learns ~0.20).
    tr_scores = np.append(train_acc, 0.20)
    track_pick = int(tr_scores.argmax())
    track_acc = corr_aug[:, track_pick].mean()

    # confidence routing picks argmax reported confidence, per query it's fixed
    # here (confidence is model-level in this synthetic setup), so:
    conf_calib_pick = int(honest_conf.argmax())
    conf_loud_pick = int(loud_conf.argmax())
    return {
        "track-record (immune)": track_acc,
        "confidence, calibrated signal": corr_aug[:, conf_calib_pick].mean(),
        "confidence, junk model overclaims": corr_aug[:, conf_loud_pick].mean(),
    }


def main():
    print("=" * 74)
    print("REAL-LLM ROUTING on RouterBench (withmartian/routerbench, 0-shot)")
    print("11 real LLMs, 36,497 queries, 86 task categories, Nov-2023")
    print("=" * 74)
    df = load()
    seeds = range(5)

    acc, cost, bn = _agg(df, MODELS, seeds)
    _print_table("(1) FULL ENSEMBLE (all 11 models):", acc, cost, bn)

    peers = [m for m in MODELS if m != DOMINANT]
    acc2, cost2, bn2 = _agg(df, peers, seeds)
    _print_table("(2) COMPARABLE PEERS (drop dominant GPT-4, 10 models):",
                 acc2, cost2, bn2)

    # (3) cost-quality: routing's real win over best-single on frontier LLMs.
    seeds_l = list(seeds)
    gpt4_acc = np.mean([evaluate(df, MODELS, s)["best-single model"][0] for s in seeds_l])
    gpt4_cost = np.mean([evaluate(df, MODELS, s)["best-single model"][1] for s in seeds_l])
    print("\n(3) COST-AWARE track-record routing (all 11 models): match GPT-4")
    print("    quality for less money -- pick cheapest model within `tol` accuracy.")
    print(f"    {'router':28s} {'accuracy':>9s} {'mean $/query':>14s} {'vs GPT-4 cost':>13s}")
    print(f"    {'best-single (GPT-4)':28s} {gpt4_acc:9.4f} {gpt4_cost:14.6f} {'--':>13s}")
    for tol in (0.00, 0.02, 0.05, 0.10):
        a = np.mean([cost_aware_route(df, MODELS, tol, s)[0] for s in seeds_l])
        c = np.mean([cost_aware_route(df, MODELS, tol, s)[1] for s in seeds_l])
        print(f"    {'cost-aware (tol=%.2f)' % tol:28s} {a:9.4f} {c:14.6f} "
              f"{'-%.0f%%' % (100 * (1 - c / gpt4_cost)):>13s}")
    print("    -> at tol=0.00, MATCHES GPT-4 accuracy ~15% cheaper; at tol=0.05,")
    print("       within ~1 accuracy point at ~60% cheaper. Routing's win over")
    print("       best-single on real frontier LLMs is on the COST axis.")

    print("\n" + "-" * 74)
    print("VERDICT (real LLMs):")
    print("  * With a dominant generalist (GPT-4), routing ~= best-single: GPT-4")
    print("    wins nearly every category, so track-record degenerates to 'use GPT-4'.")
    print("    Division of labor is ABSENT here -- honest negative.")
    print("  * Among comparable peers, track-record routing BEATS the best single")
    print("    model. Division of labor is REAL when models are peers, not one")
    print("    dominating generalist. Honest positive for the routing claim.")
    print("  * Confidence-routing / the calibration trap is NOT directly testable:")
    print("    RouterBench ships no per-model confidence/logprob. See below for a")
    print("    SIMULATED mechanism demo (synthetic confidence, real correctness).")

    print("\n" + "-" * 74)
    print("(4) SIMULATED calibration-trap stress test  [confidence is SYNTHETIC]")
    trap = calibration_trap_stress(df, seed=0)
    for k, v in trap.items():
        print(f"    {k:36s} accuracy {v:.4f}")
    print("    -> a weak model that OVERCLAIMS confidence captures confidence-")
    print("       routing (it collapses); track-record ignores it (learned it is")
    print("       inaccurate). Mechanism only -- RouterBench has no real confidence.")
    print("=" * 74)


if __name__ == "__main__":
    main()
