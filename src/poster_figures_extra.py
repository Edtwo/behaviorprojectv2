#!/usr/bin/env python3
# FirstWords - two extra board figures, same style as poster_figures.py.
#   fig9_asr_stress.png       transcription-error stress test (reads results/asr_stress.csv,
#                             written by src/ext_asr_stress.py)
#   fig10_validation_design.png  how testing worked: child-independent folds + outside labs
# USAGE: .venv/bin/python src/poster_figures_extra.py
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

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

# ---------------------------------------------------------------- fig 9: stress test
d = pd.read_csv(os.path.join(ROOT, "results", "asr_stress.csv"))
fig, ax = plt.subplots(figsize=(11.4, 7.5))
for m, c, ls, lab in [(0.0, PURPLE, "-", "all pauses caught"), (0.15, LILAC, "--", "15% of pauses missed"),
                      (0.30, DARK, ":", "30% of pauses missed")]:
    s = d[np.isclose(d["merge"], m)].sort_values("wer")
    ax.plot(s.wer * 100, s.auc, ls=ls, color=c, lw=5, marker="o", ms=14, mec="white", mew=2.5, label=lab)
ax.axhline(0.5, color=VERM, ls="--", lw=3)
ax.text(40, 0.515, "coin flip (0.50)", color=VERM, fontsize=22, fontweight="bold", ha="right", va="bottom")
ax.set_ylim(0.45, 1.0); ax.set_xlim(-2, 42); ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%"])
ax.set_xlabel("Words transcribed wrong"); ax.set_ylabel("ROC-AUC")
ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.16), frameon=False)
fig.savefig(os.path.join(OUT, "fig9_asr_stress.png")); plt.close(fig)
print("wrote results/poster/fig9_asr_stress.png")

# ---------------------------------------------------------------- fig 10: validation design
# drawn at its printed size on the board (~10.8 in wide) so every label prints at >= 24 pt
fig, ax = plt.subplots(figsize=(10.8, 3.9))
ax.set_xlim(0, 108); ax.set_ylim(0, 39); ax.axis("off")
for k in range(5):
    x0, test = 1 + k * 11.3, (k == 4)
    ax.add_patch(FancyBboxPatch((x0, 11), 9.6, 13, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=(VERM if test else PURPLE), ec=DARK, lw=2.5, alpha=0.9))
    for i in range(3):
        for j in range(2):
            ax.add_patch(Circle((x0 + 2.2 + i * 2.6, 15 + j * 5), 1.0, fc="white", ec="none"))
    ax.text(x0 + 4.8, 28, "test" if test else "train", ha="center", fontsize=24, fontweight="bold",
            color=(VERM if test else PURPLE))
ax.text(28.5, 2, "345 children, 5 rounds", ha="center", fontsize=24, color=DARK)
ax.add_patch(FancyArrowPatch((58, 17.5), (66, 17.5), arrowstyle="-|>", mutation_scale=34, lw=4, color=DARK))
ax.add_patch(FancyBboxPatch((68, 11), 39, 13, boxstyle="round,pad=0.3,rounding_size=1.2", fc=PALE, ec=DARK, lw=2.5))
ax.text(87.5, 18.5, "782 new children", ha="center", fontsize=26, fontweight="bold", color=DARK)
ax.text(87.5, 13, "2 other labs", ha="center", fontsize=24, color=DARK)
ax.text(87.5, 2, "never trained on", ha="center", fontsize=24, color=DARK)
fig.savefig(os.path.join(OUT, "fig10_validation_design.png")); plt.close(fig)
print("wrote results/poster/fig10_validation_design.png")

# ---------------------------------------------------------------- fig 11: a class of 100
# What the screen does for 100 children at 7% prevalence, using the STRICT operating point
# (cutoff chosen inside training folds; results/operating_point.csv from ext_operating_point.py).
op = pd.read_csv(os.path.join(ROOT, "results", "operating_point.csv")).iloc[0]
sens, spec, prev = op.sensitivity, op.specificity, 0.07
hit = round(100 * prev * sens); miss = round(100 * prev) - hit
fa = round(100 * (1 - prev) * (1 - spec)); clear = 100 - hit - miss - fa
cols = [VERM] * hit + ["white"] * miss + [LILAC] * fa + [PALE] * clear
edge = [VERM] * hit + [VERM] * miss + [LILAC] * fa + [GRAY] * clear
order = np.argsort(np.r_[np.zeros(hit + miss), np.ones(fa), 2 * np.ones(clear)], kind="stable")
fig, ax = plt.subplots(figsize=(5.8, 6.25))
ax.set_xlim(-0.6, 9.6); ax.set_ylim(-6.1, 9.6); ax.axis("off"); ax.set_aspect("equal")
for k in range(100):
    r, c = divmod(k, 10)
    ax.add_patch(Circle((c, 9 - r), 0.38, fc=cols[order[k]], ec=edge[order[k]], lw=2.5))
leg = [(VERM, VERM, f"{hit} with DLD: flagged"), ("white", VERM, f"{miss} with DLD: missed"),
       (LILAC, LILAC, f"{fa} typical: sent for a check"), (PALE, GRAY, f"{clear} typical: cleared")]
for i, (fc, ec, t) in enumerate(leg):
    y0 = -1.4 - i * 1.15
    ax.add_patch(Circle((0.1, y0), 0.38, fc=fc, ec=ec, lw=2.5))
    ax.text(0.8, y0, t, va="center", fontsize=22, color=DARK)
fig.savefig(os.path.join(OUT, "fig11_class_of_100.png")); plt.close(fig)
print(f"wrote results/poster/fig11_class_of_100.png  (flagged {hit}, missed {miss}, checked {fa}, cleared {clear}; "
      f"cleared children truly typical: {clear / (clear + miss):.0%})")
