#!/usr/bin/env python3
# FirstWords - board figure 13: does the recording path work on REAL toddler recordings?
# Same style as poster_figures.py. Reads results/asr_childspeech.csv (ISS-29, base.en) and
# results/voice_filter.csv (ISS-36). All three bars use Whisper base.en (the demo's model) and
# the FULL ~1-hour home session of each of 5 Providence children (~33 months).
#   fig13_real_recordings.png  language age from the recording minus language age from the
#                              human transcript (months): child-only audio / raw / raw + filter
# USAGE: .venv/bin/python src/ext_asr_childspeech.py && .venv/bin/python src/ext_voice_filter.py
#        && .venv/bin/python src/poster_figures_audio.py
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "poster")
PURPLE, LILAC, DARK, GRAY, VERM = "#5a4fbf", "#9d94e0", "#2d2a61", "#a9a5c6", "#D55E00"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 26, "axes.labelsize": 28, "xtick.labelsize": 24, "ytick.labelsize": 24,
    "text.color": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "axes.edgecolor": DARK, "axes.linewidth": 2.5, "xtick.major.width": 2.5, "ytick.major.width": 2.5,
    "xtick.major.size": 9, "ytick.major.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.transparent": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})


def main():
    a = pd.read_csv(os.path.join(ROOT, "results", "asr_childspeech.csv"))
    v = pd.read_csv(os.path.join(ROOT, "results", "voice_filter.csv"))
    gold = a[a.condition == "gold"].set_index("child").lang_age
    pipe = a[a.condition == "pipeline"].set_index("child").lang_age - gold
    s = v[v.window == "session"]
    vg = s[s.condition == "gold"].set_index("child").lang_age
    raw = s[s.condition == "raw_all"].set_index("child").lang_age - vg
    filt = s[s.condition == "voice_filter"].set_index("child").lang_age - vg
    groups = [("Raw home\nrecording", raw, GRAY), ("Raw recording +\nchild-voice filter", filt, PURPLE),
              ("Child's speech\nonly (best case)", pipe, LILAC)]
    rows = [(n.replace("\n", " "), d.mean(), d.abs().mean(), d.min(), d.max()) for n, d, _ in groups]
    pd.DataFrame(rows, columns=["condition", "mean", "mean_abs", "min", "max"]).to_csv(
        os.path.join(ROOT, "results", "real_recordings_summary.csv"), index=False)
    for r in rows: print(f"{r[0]:40s} mean {r[1]:+5.1f}  |{r[2]:4.1f}|  [{r[3]:+.1f}, {r[4]:+.1f}]")

    fig, ax = plt.subplots(figsize=(11.4, 7.5))
    rng = np.random.default_rng(1)
    for i, (n, d, col) in enumerate(groups):
        ax.bar(i, d.mean(), width=0.62, color=col, ec=DARK, lw=2, zorder=2)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=150, color="white", ec=DARK, lw=2.5, zorder=3)
        ax.text(i, max(d.mean(), d.max()) + 1.0, f"{d.mean():+.0f} mo", ha="center", va="bottom",
                fontsize=26, fontweight="bold", color=DARK)
    ax.axhline(0, color=VERM, ls="--", lw=3, zorder=1)
    ax.text(-0.5, -0.6, "0 = same as human transcript", color=VERM, fontsize=20, fontweight="bold", ha="left", va="top")
    ax.set_xticks(range(3)); ax.set_xticklabels([n for n, _, _ in groups], fontsize=22)
    ax.set_ylabel("Language age error (months)")
    ax.set_ylim(-4, 26); ax.set_xlim(-0.55, 2.55)
    fig.savefig(os.path.join(OUT, "fig13_real_recordings.png")); plt.close(fig)
    print("wrote results/poster/fig13_real_recordings.png")


if __name__ == "__main__":
    main()
