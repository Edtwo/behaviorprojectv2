#!/usr/bin/env python3
# FirstWords - poster figures. Recomputes every number from the same pipeline the
# analysis scripts use (no hard-coded results) and draws each board figure at its
# real print size, in the board palette, with poster-size fonts (>= 24 pt).
#
# Figures carry only data, axes and labels. The headline for each figure is written
# by the team on the board itself.
#
# USAGE:  .venv/bin/python src/poster_figures.py
# OUTPUT: results/poster/fig*.png (300 dpi, transparent) + results/poster/numbers.txt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
import stage2_level_estimator as s2
import stage3_delay_classifier as s3
import ext_percentiles as pct
import ext_crosscorpus as cc
import ext_shortsample as ss
import ext_benchmark as bm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "poster")
os.makedirs(OUT, exist_ok=True)

# board palette (matches the Canva board and the demo app)
PURPLE, LILAC, DARK = "#5a4fbf", "#9d94e0", "#2d2a61"
GRAY, PALE, VERM = "#a9a5c6", "#d9d5ee", "#D55E00"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 26, "axes.labelsize": 28, "xtick.labelsize": 24, "ytick.labelsize": 24,
    "legend.fontsize": 24, "text.color": DARK, "axes.labelcolor": DARK,
    "xtick.color": DARK, "ytick.color": DARK, "axes.edgecolor": DARK,
    "axes.linewidth": 2.5, "xtick.major.width": 2.5, "ytick.major.width": 2.5,
    "xtick.major.size": 9, "ytick.major.size": 9,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.transparent": True, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})
LOG = []
def note(s):
    print(s); LOG.append(s)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); note(f"  wrote results/poster/{name}")

def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))

def chance_line(ax, vertical=True, label=True, y_text=None):
    if vertical:
        ax.axvline(0.5, color=VERM, ls="--", lw=3, zorder=1)
        if label:
            ax.text(0.505, y_text, "coin flip (0.50)", color=VERM, fontsize=22,
                    ha="left", va="bottom", fontweight="bold")
    else:
        ax.axhline(0.5, color=VERM, ls="--", lw=3, zorder=1)

def dot_ci(ax, rows, xlim=(0.4, 1.0)):
    """Horizontal dot-and-interval plot. rows: list of (label, auc, lo, hi, color), top first."""
    n = len(rows)
    for i, (lab, a, lo, hi, c) in enumerate(rows):
        yy = n - 1 - i
        ax.plot([lo, hi], [yy, yy], color=c, lw=7, solid_capstyle="round", zorder=2)
        ax.scatter([a], [yy], s=900, color=c, edgecolor="white", linewidth=3, zorder=3)
        ax.text(hi + 0.012, yy, f"{a:.2f}", va="center", ha="left", fontsize=34,
                fontweight="bold", color=c)
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(*xlim); ax.set_ylim(-0.6, n - 0.4)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ROC-AUC  (0.5 = coin flip, 1.0 = perfect)")

# ---------------------------------------------------------------- data
enni = s3.load_enni()
y, groups = enni["y"].to_numpy(), enni["child_id"].to_numpy()
note(f"ENNI delay set: {len(enni)} transcripts, {enni['child_id'].nunique()} children, "
     f"{int(y.sum())} impaired transcripts")

# ---------------------------------------------------------------- Fig 1: artifact control (hero)
note("\nFig 1 - artifact control")
arms = [("Age alone", ["age_months"], GRAY),
        ("Transcript length\nalone", ["n_utts"], GRAY),
        ("Language alone", s3.LANG_FEATURES, LILAC),
        ("Language + age\n(our model)", s3.FEATURES, PURPLE)]
res1 = []
for lab, cols, c in arms:
    p = s3.oof_proba(lr, enni[cols].to_numpy(), y, groups)
    a = roc_auc_score(y, p); lo, hi = s3.boot_ci(y, p, groups, roc_auc_score)
    res1.append((lab, a, lo, hi, c))
    note(f"  {lab.replace(chr(10), ' '):28s} AUC {a:.3f} [{lo:.3f}-{hi:.3f}]")
    if cols == s3.FEATURES:
        p_full = p
fig, ax = plt.subplots(figsize=(15, 8.5))
ypos = np.arange(len(res1))[::-1]
for yy, (lab, a, lo, hi, c) in zip(ypos, res1):
    ax.barh(yy, a, height=0.66, color=c, zorder=2)
    ax.plot([lo, hi], [yy, yy], color=DARK, lw=3.5, zorder=3)
    for e in (lo, hi):
        ax.plot([e, e], [yy - 0.12, yy + 0.12], color=DARK, lw=3.5, zorder=3)
    ax.text(hi + 0.02, yy, f"{a:.2f}", va="center", ha="left", fontsize=40,
            fontweight="bold", color=PURPLE if c == PURPLE else DARK)
ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in res1])
ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, len(res1) - 0.25)
chance_line(ax, y_text=len(res1) - 0.55)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.set_xlabel("ROC-AUC  (0.5 = coin flip, 1.0 = perfect)")
save(fig, "fig1_artifact_control.png")

# ---------------------------------------------------------------- Fig 2: ROC + screening point
note("\nFig 2 - ROC")
fpr, tpr, thr = roc_curve(y, p_full)
j = int(np.argmin(np.abs(tpr - 0.90)))
sens, spec, (tp, fp, tn, fn) = s3.sens_spec_at(y, p_full, thr[j])
auc = roc_auc_score(y, p_full); lo, hi = res1[-1][2], res1[-1][3]
note(f"  AUC {auc:.3f} [{lo:.3f}-{hi:.3f}]; screening point sens {sens:.2f} ({tp} of {tp+fn}), "
     f"spec {spec:.2f}")
fig, ax = plt.subplots(figsize=(10, 9.5))
ax.fill_between(fpr, tpr, step="post", color=PALE, zorder=1)
ax.step(fpr, tpr, where="post", color=PURPLE, lw=5, zorder=3)
ax.plot([0, 1], [0, 1], color=VERM, ls="--", lw=3, zorder=2)
ax.text(0.80, 0.815, "coin flip", color=VERM, fontsize=22, fontweight="bold", rotation=45,
        ha="center", va="bottom", rotation_mode="anchor")
ax.scatter([1 - spec], [sens], s=700, color=VERM, edgecolor="white", linewidth=3, zorder=5)
ax.annotate(f"Screening point\nsensitivity {sens:.0%}\nspecificity {spec:.0%}",
            xy=(1 - spec, sens), xytext=(0.52, 0.44), fontsize=24, fontweight="bold",
            color=DARK, ha="left", va="top",
            arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.5, mutation_scale=28))
ax.text(0.97, 0.06, f"AUC {auc:.2f}\n[{lo:.2f}–{hi:.2f}]", ha="right", va="bottom",
        fontsize=34, fontweight="bold", color=PURPLE)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
ax.set_xlabel("False positive rate\n(1 − specificity)")
ax.set_ylabel("Sensitivity\n(impaired children caught)")
save(fig, "fig2_roc_screening.png")

# ---------------------------------------------------------------- Fig 3: growth chart
note("\nFig 3 - growth chart")
df = pd.read_csv(os.path.join(ROOT, "results", "features.csv"))
td = df[(df["group"] == "TD") & df["age_months"].notna()].dropna(subset=["mluw"])
sli = df[(df["group"] == "SLI") & df["age_months"].notna()].dropna(subset=["mluw"])
curves = pct.fit_percentile_curves(td)
pcts_td = np.array([pct.percentile_for(v, a, curves) for v, a in zip(td["mluw"], td["age_months"])])
pcts_sli = np.array([pct.percentile_for(v, a, curves) for v, a in zip(sli["mluw"], sli["age_months"])])
note(f"  typical transcripts below P10: {np.mean(pcts_td < 10):.1%} (target 10%); "
     f"impaired transcripts below P10: {np.mean(pcts_sli < 10):.1%}; below P25: {np.mean(pcts_sli < 25):.1%}")
fig, ax = plt.subplots(figsize=(13, 9))
ax.scatter(td["age_months"] / 12, td["mluw"], s=55, color=GRAY, alpha=0.45, edgecolor="none",
           label="Typically developing", zorder=1)
ax.scatter(sli["age_months"] / 12, sli["mluw"], s=110, color=VERM, alpha=0.85, edgecolor="white",
           linewidth=1, marker="D", label="Language impairment", zorder=4)
grid = np.linspace(td["age_months"].min(), td["age_months"].max(), 200)
styles = {0.10: (":", 3.5), 0.25: ("--", 3.5), 0.50: ("-", 5.5), 0.75: ("--", 3.5), 0.90: (":", 3.5)}
for q, m in curves.items():
    if q not in (0.10, 0.50, 0.90):   # three curves keep the chart readable from a distance
        continue
    ls, lw = styles[q]
    yv = m.predict(pct._design(grid))
    ax.plot(grid / 12, yv, ls=ls, lw=lw, color=PURPLE, zorder=3)
    ax.text(grid[-1] / 12 + 0.12, yv[-1], f"P{int(q * 100)}", color=PURPLE, fontsize=24,
            fontweight="bold", va="center", ha="left")
ax.set_xlim(1.3, 10.9); ax.set_ylim(0, None)
ax.set_xticks(range(2, 11))
ax.set_xlabel("Age (years)"); ax.set_ylabel("Sentence length\n(words per utterance)")
ax.legend(loc="upper left", frameon=False, markerscale=1.6, handletextpad=0.3)
save(fig, "fig3_growth_chart.png")

# ---------------------------------------------------------------- Fig 4: external validation
note("\nFig 4 - external validation (trained on ENNI, no retraining)")
full = lr().fit(enni[cc.DELAY_FEATURES], enni["y"])
lang = lr().fit(enni[cc.LANG_ONLY], enni["y"])
ext = {}
for name, path in cc.find_external_corpora():
    d = cc.extract_corpus(path).dropna(subset=cc.DELAY_FEATURES)
    ext[name] = d[d["group"].isin(["TD", "SLI"])].copy()
g = ext["Gillam"]; yg = (g["group"] == "SLI").astype(int).to_numpy()
pg = full.predict_proba(g[cc.DELAY_FEATURES])[:, 1]
ag = roc_auc_score(yg, pg); gl, gh = cc._auc_ci(yg, pg)
inr = (g["age_months"] <= enni["age_months"].max()).to_numpy()
ag_in = roc_auc_score(yg[inr], pg[inr]); gil, gih = cc._auc_ci(yg[inr], pg[inr])
note(f"  Gillam all: n={len(g)} ({int(yg.sum())} impaired), ages {g.age_months.min():.0f}-"
     f"{g.age_months.max():.0f} mo, AUC {ag:.3f} [{gl:.3f}-{gh:.3f}]")
note(f"  Gillam within ENNI age range: n={inr.sum()}, AUC {ag_in:.3f} [{gil:.3f}-{gih:.3f}]")
c4 = ext["Conti4"]; c4 = c4[c4["task"] == "frog"]
yc = (c4["group"] == "SLI").astype(int).to_numpy()
pc = lang.predict_proba(c4[cc.LANG_ONLY])[:, 1]
ac = roc_auc_score(yc, pc); cl, ch = cc._auc_ci(yc, pc)
note(f"  Conti4 frog story: n={len(c4)} ({int(yc.sum())} impaired), ages {c4.age_months.min():.0f}-"
     f"{c4.age_months.max():.0f} mo, language-only AUC {ac:.3f} [{cl:.3f}-{ch:.3f}]")
e_lab, e_a, e_lo, e_hi = res1[-1][0], res1[-1][1], res1[-1][2], res1[-1][3]
rows = [(f"ENNI, Canada\n{enni['child_id'].nunique()} children, ages 4–9\nheld-out children", e_a, e_lo, e_hi, PURPLE),
        (f"Gillam, USA\n{len(g)} children, ages 5–11\nnew lab, no retraining", ag, gl, gh, PURPLE),
        (f"Conti-Ramsden, UK\n{len(c4)} teens, ages 13–16\nnew lab and new ages", ac, cl, ch, PURPLE)]
fig, ax = plt.subplots(figsize=(14, 9))
dot_ci(ax, rows)
chance_line(ax, y_text=len(rows) - 0.55)
for t in ax.get_yticklabels():
    t.set_linespacing(1.3)
save(fig, "fig4_external_validation.png")

# ---------------------------------------------------------------- Fig 5: benchmark
note("\nFig 5 - benchmark vs black box")
d = bm.load_enni(); yb = d["y"].to_numpy(); gb = d["child_id"].to_numpy()
cols = lambda f: (lambda tr, te: (d.iloc[tr][f].to_numpy(), d.iloc[te][f].to_numpy()))
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_X(tr, te):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=3000)
    return vec.fit_transform(d.iloc[tr]["text"]), vec.transform(d.iloc[te]["text"])
bres = [("Our 4 explainable\nmeasures", bm.oof(lr, cols(bm.OURS), yb, gb), PURPLE),
        ("Black-box text model\n(3,000 word features)",
         bm.oof(lambda: LogisticRegression(class_weight="balanced", max_iter=2000), tfidf_X, yb, gb), GRAY),
        ("Simple rule\n(sentence length + age)", bm.oof(lr, cols(["mluw", "age_months"]), yb, gb), GRAY)]
rows = []
for lab, p, c in bres:
    a = roc_auc_score(yb, p); lo, hi = bm.ci(yb, p)
    rows.append((lab, a, lo, hi, c)); note(f"  {lab.replace(chr(10), ' '):44s} AUC {a:.3f} [{lo:.3f}-{hi:.3f}]")
fig, ax = plt.subplots(figsize=(14, 7.5))
dot_ci(ax, rows)
chance_line(ax, y_text=len(rows) - 0.55)
save(fig, "fig5_benchmark.png")

# ---------------------------------------------------------------- Fig 6: short samples
note("\nFig 6 - short samples")
sd = ss.collect()
def ss_proba(n):
    X, yy, gg = [], [], []
    for _, r in sd.iterrows():
        if pd.isna(r["age_months"]):
            continue
        m, t = ss.feats_at(r["utt_words"], n)
        if m is None or t is None or pd.isna(t):
            continue
        X.append([m, t, r["age_months"]]); yy.append(int(r["group"] == "SLI")); gg.append(r["child_id"])
    X, yy, gg = np.array(X), np.array(yy), np.array(gg)
    return yy, s3.oof_proba(lr, X, yy, gg), gg
NS = [10, 15, 20, 30, 40, 50, None]
pts = []
for n in NS:
    yy, pp, gg = ss_proba(n if n else 10**6)
    a = roc_auc_score(yy, pp); lo, hi = s3.boot_ci(yy, pp, gg, roc_auc_score)
    pts.append((a, lo, hi)); note(f"  N={'all' if n is None else n:>4}: AUC {a:.3f} [{lo:.3f}-{hi:.3f}]")
med = int(np.median([len(u) for u in sd["utt_words"]]))
fig, ax = plt.subplots(figsize=(13, 7.5))
xs = np.arange(len(NS))
ax.fill_between(xs, [p[1] for p in pts], [p[2] for p in pts], color=PALE, zorder=1)
ax.plot(xs, [p[0] for p in pts], "-o", color=PURPLE, lw=5, ms=20, mec="white", mew=3, zorder=3)
chance_line(ax, vertical=False)
ax.text(len(NS) - 1, 0.515, "coin flip (0.50)", color=VERM, fontsize=22, fontweight="bold",
        ha="right", va="bottom")
for x, (a, _, hi) in zip(xs, pts):
    ax.text(x, a + 0.022, f"{a:.2f}", ha="center", va="bottom", fontsize=24, fontweight="bold", color=PURPLE,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=4)
ax.set_xticks(xs); ax.set_xticklabels([str(n) for n in NS[:-1]] + [f"All\n(~{med})"])
ax.set_ylim(0.45, 1.0); ax.set_xlim(-0.4, len(NS) - 0.6)
ax.set_xlabel("Utterances used from each child")
ax.set_ylabel("ROC-AUC")
save(fig, "fig6_short_sample.png")

# ---------------------------------------------------------------- Fig 7: language age
note("\nFig 7 - language age estimate")
tdm = df[(df["group"] == "TD") & df["age_months"].notna()].dropna(subset=s2.FEATURES).copy()
tdm["pred"] = s2.oof_predict(lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                             tdm[s2.FEATURES].to_numpy(), tdm["age_months"].to_numpy(),
                             tdm["child_id"].to_numpy())
mae, r2 = s2.mae(tdm["age_months"], tdm["pred"]), s2.r2(tdm["age_months"], tdm["pred"])
note(f"  n={len(tdm)} transcripts, {tdm.child_id.nunique()} children; MAE {mae:.2f} mo, R2 {r2:.2f}")
fig, ax = plt.subplots(figsize=(10, 9.5))
ax.scatter(tdm["age_months"] / 12, tdm["pred"] / 12, s=70, color=PURPLE, alpha=0.45,
           edgecolor="none", zorder=2)
ax.plot([0, 11], [0, 11], color=DARK, ls="--", lw=3, zorder=3)
ax.text(0.04, 0.97, f"Typical error: {mae:.0f} months\nR² = {r2:.2f}", transform=ax.transAxes,
        ha="left", va="top", fontsize=30, fontweight="bold", color=PURPLE)
ax.text(0.97, 0.04, "dashed line =\nestimate matches\nreal age", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=22, color=DARK)
ax.set_xlim(1, 11); ax.set_ylim(0, 11); ax.set_aspect("equal")
ax.set_xticks(range(2, 11, 2)); ax.set_yticks(range(0, 11, 2))
ax.set_xlabel("Real age (years)"); ax.set_ylabel("Estimated language age (years)")
save(fig, "fig7_language_age.png")

# ---------------------------------------------------------------- Fig 8: who is in the data
note("\nFig 8 - data overview")
def span(d):
    return d["age_months"].min() / 12, d["age_months"].max() / 12
kids = lambda d: d["child_id"].nunique()
ennitd = df[(df.corpus == "ENNI") & (df.group == "TD") & df.age_months.notna()]
groups8 = [
    ("Typical development norms", PURPLE, [
        ("Bates (USA)", *span(df[df.corpus == "Bates"]), kids(df[df.corpus == "Bates"])),
        ("Gleason (USA)", *span(df[df.corpus == "Gleason"]), kids(df[df.corpus == "Gleason"])),
        ("Brown (USA)", *span(df[df.corpus == "Brown"]), kids(df[df.corpus == "Brown"])),
        ("ENNI typical (Canada)", *span(ennitd), kids(ennitd))]),
    ("Delay model training", LILAC, [
        ("ENNI (Canada)", *span(enni), enni["child_id"].nunique())]),
    ("Outside tests, no retraining", VERM, [
        ("Gillam (USA)", *span(g), len(g)),
        ("Conti-Ramsden (UK)", *span(c4), len(c4))]),
    ("Second languages", GRAY, [
        ("German (Szagun)", 15 / 12, 44 / 12, 22),
        ("Cantonese (HKU)", 29 / 12, 68 / 12, 70)]),
]
fig, ax = plt.subplots(figsize=(16, 10))
yy, ticks, labs = 0, [], []
for gname, c, items in groups8:
    ax.text(0.15, yy + 0.05, gname, fontsize=26, fontweight="bold", color=DARK, va="bottom")
    yy -= 0.85
    for lab, a0, a1, n in items:
        ax.plot([a0, a1], [yy, yy], color=c, lw=24, solid_capstyle="butt", zorder=2)
        ax.text(a1 + 0.2, yy, f"{n} children" + (f", {(df.corpus == 'Brown').sum()} sessions" if lab.startswith("Brown") else ""), va="center", ha="left", fontsize=24, color=DARK)
        ticks.append(yy); labs.append(lab); yy -= 0.85
        note(f"  {lab:24s} ages {a0:.1f}-{a1:.1f} yr, {n} children")
    yy -= 0.35
ax.set_yticks(ticks); ax.set_yticklabels(labs)
ax.set_ylim(yy + 0.4, 0.75); ax.set_xlim(0, 19.5)
ax.set_xticks(range(0, 17, 2)); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.spines["bottom"].set_bounds(0, 16)
ax.set_xlabel("Age (years)")
save(fig, "fig8_data_overview.png")

with open(os.path.join(OUT, "numbers.txt"), "w") as f:
    f.write("\n".join(LOG) + "\n")
print("\nnumbers: results/poster/numbers.txt")
