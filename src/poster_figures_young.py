#!/usr/bin/env python3
# FirstWords - board figure 11: does the screen work on toddlers / preschoolers?
# Same style as poster_figures_extra.py. Reads results/ellisweismer_scores.csv
# (written by src/ext_young_delay.py). One score per CHILD per visit: a child's tasks at
# a visit (e.g. examiner + parent play at 30 mo) are averaged, so no child counts twice.
#   fig11_young_children.png  delay-flag ROC-AUC (late talkers vs typical) at 30/42/54/66 mo
# USAGE: .venv/bin/python src/ext_young_delay.py && .venv/bin/python src/poster_figures_young.py
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "poster")
PURPLE, LILAC, DARK, GRAY, PALE, VERM = "#5a4fbf", "#9d94e0", "#2d2a61", "#a9a5c6", "#d9d5ee", "#D55E00"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 26, "axes.labelsize": 28, "xtick.labelsize": 24, "ytick.labelsize": 24,
    "legend.fontsize": 24, "text.color": DARK, "axes.labelcolor": DARK,
    "xtick.color": DARK, "ytick.color": DARK, "axes.edgecolor": DARK,
    "axes.linewidth": 2.5, "xtick.major.width": 2.5, "ytick.major.width": 2.5,
    "xtick.major.size": 9, "ytick.major.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.transparent": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})


def auc_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); v = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) == 2: v.append(roc_auc_score(y[s], p[s]))
    return roc_auc_score(y, p), np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(os.path.join(ROOT, "results", "ellisweismer_scores.csv"))
    per_child = d.groupby(["visit", "child_id"]).agg(y=("y", "first"), p=("p_lang", "mean"),
                                                    mluw=("mluw", "mean")).reset_index()
    rows = []
    for v, g in per_child.groupby("visit"):
        a, lo, hi = auc_ci(g.y.to_numpy(), g.p.to_numpy())
        rows.append((v, a, lo, hi, int(g.y.sum()), int((g.y == 0).sum())))
    r = pd.DataFrame(rows, columns=["visit", "auc", "lo", "hi", "n_lt", "n_td"])
    r.to_csv(os.path.join(ROOT, "results", "ellisweismer_by_visit.csv"), index=False)
    print(r.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(11.4, 7.5))
    ax.fill_between(r.visit, r.lo, r.hi, color=PALE, lw=0)
    ax.plot(r.visit, r.auc, color=PURPLE, lw=5, marker="o", ms=16, mec="white", mew=2.5)
    for _, row in r.iterrows():
        ax.text(row.visit, row.auc + 0.035, f"{row.auc:.2f}", ha="center", fontsize=24,
                fontweight="bold", color=DARK)
    ax.axhline(0.5, color=VERM, ls="--", lw=3)
    ax.text(27, 0.515, "coin flip (0.50)", color=VERM, fontsize=22, fontweight="bold", ha="left", va="bottom")
    ax.axvspan(48, 70, color=GRAY, alpha=0.18, lw=0)
    ax.text(69, 0.97, "training ages start here", ha="right", va="top", fontsize=20, color=DARK)
    ax.set_xlim(26, 70); ax.set_ylim(0.4, 1.0); ax.set_xticks(r.visit)
    ax.set_xticklabels([f"{v}" for v in r.visit])
    ax.set_xlabel("Child's age (months)"); ax.set_ylabel("ROC-AUC")
    fig.savefig(os.path.join(OUT, "fig11_young_children.png")); plt.close(fig)
    print("wrote results/poster/fig11_young_children.png")


if __name__ == "__main__":
    main()
