"""Replicate the Brick-1 claim across multiple Confidence Database datasets (Rahnev et al.).

For each dataset:
  (1) CALIBRATION — corr(confidence, accuracy): is human confidence calibrated? (robust, universal)
  (2) WISDOM OF CROWDS (only where items are SHARED across subjects) — does confidence-weighted
      pooling beat majority vote, and approach the accuracy-oracle?

Downloads from OSF at runtime (needs internet + pandas). Reproducibility script for the direction
in docs/notes/2026-07-21-cognitive-digital-twin-research-program.md.

    python -m cognitive_society.real_confidence_replication
"""
import io
import urllib.request as u

import numpy as np
import pandas as pd

# (name, OSF download id) — a diverse set from the Confidence Database
DATASETS = [
    ("Double_2017 (Raven's)", "cs2wv"),
    ("Akdogan_2017", "k7u2p"),
    ("Palser_2018", "zyjxe"),
    ("Wang_2017", "8xftz"),
    ("Pereira_2018", "9qdup"),
    ("Desender_2014", "s9kz3"),
    ("Mei_2019", "tne9g"),
    ("Arbuzova_2021", "y7mgf"),
]


def fetch(osf_id):
    req = u.Request(f"https://osf.io/download/{osf_id}/", headers={"User-Agent": "research"})
    return pd.read_csv(io.BytesIO(u.urlopen(req, timeout=90).read()))


def prep(df):
    df = df.rename(columns={c: c.strip() for c in df.columns})
    need = {"Subj_idx", "Response", "Confidence"}
    if not need.issubset(df.columns):
        return None
    if "Accuracy" not in df.columns:
        if "Stimulus" in df.columns:
            df["Accuracy"] = (df["Stimulus"].astype(str) == df["Response"].astype(str)).astype(float)
        else:
            return None
    return df.dropna(subset=["Response", "Confidence", "Accuracy", "Subj_idx"])


def woc(df, crowd_K=9, boots=200, min_subj=8, min_items=6):
    """Item-level wisdom of crowds, only if items are shared across many subjects."""
    if "Stimulus" not in df.columns:
        return None
    per_item_subj = df.groupby("Stimulus")["Subj_idx"].nunique()
    shared = per_item_subj[per_item_subj >= min_subj].index
    if len(shared) < min_items:
        return None
    d = df[df["Stimulus"].isin(shared)]
    correct = {}
    for it in shared:
        ok = d[(d.Stimulus == it) & (d.Accuracy == 1)]
        if len(ok):
            correct[it] = ok["Response"].iloc[0]
    if len(correct) < min_items:
        return None
    subs = sorted(d["Subj_idx"].unique())
    sub_acc = d.groupby("Subj_idx")["Accuracy"].mean().to_dict()
    resp = {(r.Stimulus, r.Subj_idx): (r.Response, r.Confidence) for r in d.itertuples()}

    def vote(item, crowd, mode):
        tally = {}
        for s in crowd:
            rc = resp.get((item, s))
            if rc is None:
                continue
            opt, conf = rc
            w = 1.0 if mode == "maj" else (conf if mode == "conf" else sub_acc[s])
            tally[opt] = tally.get(opt, 0.0) + w
        return max(tally, key=tally.get) if tally else None

    rng = np.random.default_rng(0)
    K = min(crowd_K, len(subs))
    acc = {m: [] for m in ("maj", "conf", "oracle")}
    for _ in range(boots):
        crowd = list(rng.choice(subs, size=K, replace=False))
        for m in acc:
            acc[m].append(np.mean([vote(it, crowd, m) == correct[it] for it in correct]))
    return (len(correct), float(np.mean(acc["maj"])), float(np.mean(acc["conf"])),
            float(np.mean(acc["oracle"])))


def main():
    print("=" * 82)
    print("REPLICATION - confidence calibration + wisdom of crowds across Confidence Database")
    print("=" * 82)
    print(f"  {'dataset':26s} {'n_subj':>6} {'calib r':>8} | {'items':>5} {'maj':>6} {'conf':>6} {'oracle':>7}")
    cal_rs, woc_gains = [], []
    for name, osf_id in DATASETS:
        try:
            df = prep(fetch(osf_id))
        except Exception as e:
            print(f"  {name:26s}  fetch/parse failed: {str(e)[:30]}")
            continue
        if df is None or df["Confidence"].nunique() < 2 or df["Accuracy"].nunique() < 2:
            print(f"  {name:26s}  (no usable confidence/accuracy)")
            continue
        r = float(np.corrcoef(df["Confidence"], df["Accuracy"])[0, 1])
        if not np.isfinite(r):
            print(f"  {name:26s}  (degenerate confidence/accuracy)")
            continue
        cal_rs.append(r)
        w = woc(df)
        nsub = df["Subj_idx"].nunique()
        if w:
            nit, maj, conf, orc = w
            woc_gains.append(conf - maj)
            print(f"  {name:26s} {nsub:>6} {r:>+8.3f} | {nit:>5} {maj:>6.3f} {conf:>6.3f} {orc:>7.3f}")
        else:
            print(f"  {name:26s} {nsub:>6} {r:>+8.3f} |   (no shared items: calibration only)")
    print("-" * 82)
    if cal_rs:
        print(f"  CALIBRATION replicates: corr(confidence, accuracy) = {np.mean(cal_rs):+.3f} "
              f"mean over {len(cal_rs)} datasets, all positive: {all(x > 0 for x in cal_rs)}")
    if woc_gains:
        print(f"  WISDOM OF CROWDS: confidence-weighting beats majority by "
              f"{np.mean(woc_gains)*100:+.1f}pt mean over {len(woc_gains)} shared-item datasets, "
              f"all positive: {all(g > 0 for g in woc_gains)}")
    print("=" * 82)


if __name__ == "__main__":
    main()
