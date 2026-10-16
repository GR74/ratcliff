"""Figure: hard vs soft-count G2 along one-parameter slices (lab figure style)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Arial is a lab requirement; under WSL it lives in the Windows font folder.
for _font in ("/mnt/c/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arial.ttf"):
    if os.path.exists(_font):
        font_manager.fontManager.addfont(_font)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PALETTE = ["#191970", "#0000CD", "#1E90FF", "#008B8B", "#228B22", "#006400"]
LABELS = {"ter_ms": "ter (ms)", "st_ms": "st (ms)", "cr": "cr", "crsd": "crsd",
          "sig": "sig", "av1_c1": "av1, condition 1"}


def main(path=os.path.join(ROOT, "results", "pass3", "soft_g2_slices.json")):
    res = json.load(open(path))
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.direction": "out", "ytick.direction": "out",
                         "axes.grid": False})
    names = list(res["slices"])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        sl = res["slices"][name]
        for color, (tau, g2) in zip(PALETTE, sl["g2"].items()):
            label = "hard count" if float(tau) == 0 else f"soft, τ = {float(tau):g} ms"
            ax.plot(sl["grid"], g2, color=color, lw=1.0, label=label)
        ax.set_xlabel(LABELS.get(name, name))
        ax.set_ylabel("G²")
    axes.flat[0].legend(frameon=False, fontsize=7)
    s = res["settings"]
    fig.suptitle(f"Model B G² slices, subject {s['subject'] + 1}, "
                 f"nsim = {s['nsim']}/condition, fixed seed", fontsize=9)
    out = os.path.splitext(path)[0]
    for ext in (".png", ".pdf"):
        fig.savefig(out + ext, dpi=300)
    print(out + ".png")


if __name__ == "__main__":
    main(*sys.argv[1:])
