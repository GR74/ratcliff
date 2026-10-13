"""Plot completed extension cases; visibly retain censored fitting timings."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"control": "Early-stop\ncontrol", "active_only": "Fix unused\nsv", "bucketed": "Smaller K-L\nmatrices",
          "cumulative": "Cumulative\nweights", "all_three": "All three"}
COLORS = ["#65758b", "#8f8bc9", "#008b8b", "#db9525", "#3683d1"]


def plot(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((source / "results.json").read_text(encoding="utf-8"))
    complete = [r for r in rows if r["status"] == "completed"]
    recovery_path = source / "recovered-active-only-0-scored.json"
    recovery = (json.loads(recovery_path.read_text(encoding="utf-8"))
                if recovery_path.exists() else None)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [("micro", "timings_seconds", "Simulation (milliseconds)", 1000),
              ("fit", "fit_seconds", "Timed fitting (seconds)", 1),
              ("fit", "mean_relative_parameter_error", "Mean active parameter error (%)", 100),
              ("fit", "independent_full_budget_loss", "Independent G2 (lower is better)", 1)]
    for ax, (action, field, title, scale) in zip(axes.flat, panels):
        for i, variant in enumerate(LABELS):
            cases = [r for r in complete if r["action"] == action and r["experiment"] == variant]
            values = ([t * scale for r in cases for t in r[field]] if action == "micro"
                      else [r[field] * scale for r in cases])
            if values:
                ax.bar(i, np.median(values), color=COLORS[i], alpha=.8)
                ax.scatter(i + np.linspace(-.12, .12, len(values)), values,
                           color="#172233", s=22, zorder=3)
                ax.text(i, max(values), f" n={len(values)}", ha="center", va="bottom", fontsize=8)
        if recovery and field in ("mean_relative_parameter_error", "independent_full_budget_loss"):
            value = (recovery[field] if field == "mean_relative_parameter_error"
                     else next(s["g2"] for s in recovery["scores"] if s["seed"] == 999))
            ax.scatter([1.17], [value * scale], marker="D", facecolors="none",
                       edgecolors="#b23e39", s=45, zorder=4)
        if field == "fit_seconds":
            for r in rows:
                if r["status"] != "timeout" or r["action"] != "fit":
                    continue
                path = source / r["experiment"] / f"fit-progress-b-early_stop-{r['seed']}.jsonl"
                if path.exists():
                    trace = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
                    if trace:
                        lower = trace[-1]["elapsed"]
                        i = list(LABELS).index(r["experiment"])
                        ax.scatter([i + .17], [lower], marker="^", color="#b23e39", s=45, zorder=4)
                        ax.text(i + .17, lower, f" ≥{lower:.0f}s", fontsize=8, color="#b23e39", va="bottom")
        ax.set_title(title)
        ax.set_xticks(range(5), list(LABELS.values()), fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model B K-L extensions: speed and quality together\n"
                 "9,000 trials/condition, 40 iterations. Bars/dots = completed workers; triangles = time bounds; diamonds = recovered quality.",
                 fontsize=13)
    fig.savefig(output / "extensions-speed-and-quality.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.source, args.output)
