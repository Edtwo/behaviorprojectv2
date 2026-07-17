#!/usr/bin/env python3
# FirstWords - Stage 2: the CORE deliverable. A developmental-LEVEL estimator:
# predict a child's age (months) from transcript features on the combined TD data
# (Brown 18-62mo + ENNI-TD 48-120mo), validated CHILD-INDEPENDENTLY (GroupKFold by
# child), against a predict-the-mean baseline, with cluster-bootstrap CIs (resample
# CHILDREN, not files) and a calibration figure. Defines the "age-gap" score
# (predicted language age - chronological age), and previews it on ENNI-SLI
# (full artifact-controlled TD-vs-SLI classification is Stage 3, not claimed here).
#
# Per the 2026-07-16 audit: pylangacq's ipsyn() is NOT the published IPSyn scale,
# so it is used (and named) as a generic "syntax index" feature. TTR is length-
# confounded, so MATTR-50 (moving-average TTR, 50-word window) is added.
import os, re, statistics as st
import numpy as np
import pandas as pd
import pylangacq
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "childes")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
RNG = np.random.default_rng(42)

FEATURES = ["mluw", "mlum", "ttr", "mattr50", "syntax_index"]

def months(age):
    if age is None: return None
    m = re.search(r"(\d+);(\d+)(?:\.(\d+))?", str(age))
    if not m: return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return y*12 + mo + d/30.0

def mattr(words, window=50):
    """Moving-average type-token ratio: mean TTR over sliding 50-word windows.
    Unlike raw TTR this does not shrink as the transcript gets longer."""
    if len(words) < window:
        return len(set(words))/len(words) if words else None
    return st.mean(len(set(words[i:i+window]))/window
                   for i in range(len(words) - window + 1))

def extract(path, corpus, group, child_id_from):
    """Per-file feature rows for one corpus directory."""
    r = pylangacq.read_chat(path, strict=False)
    ages = [months(a) for a in r.ages()]
    mluw, mlum, ttr, ipsyn = r.mluw(), r.mlum(), r.ttr(), r.ipsyn()
    utts_by_file = r.utterances(by_file=True)
    rows = []
    for i, fp in enumerate(r.file_paths):
        chi = [u for u in utts_by_file[i] if u.participant == "CHI"]
        # words = tokens with a POS tag (punctuation has pos=""), lowercased
        words = [t.word.lower() for u in chi for t in u.tokens if t.pos]
        rows.append({
            "file": os.path.relpath(fp, DATA), "corpus": corpus, "group": group,
            "child_id": child_id_from(fp), "age_months": ages[i],
            "mluw": mluw[i], "mlum": mlum[i], "ttr": ttr[i],
            "mattr50": mattr(words), "syntax_index": ipsyn[i],
            "n_utts": len(chi),
        })
    return rows

def build_features():
    rows = []
    for child in sorted(os.listdir(os.path.join(DATA, "Brown"))):
        p = os.path.join(DATA, "Brown", child)
        if os.path.isdir(p):  # child_id = Brown child folder (longitudinal)
            rows += extract(p, "Brown", "TD", lambda fp, c=child: f"Brown:{c}")
    # ENNI: A/B are disjoint children (verified) -> child_id = file stem
    stem = lambda fp: "ENNI:" + os.path.splitext(os.path.basename(fp))[0]
    rows += extract(os.path.join(DATA, "ENNI", "TD"), "ENNI", "TD", stem)
    rows += extract(os.path.join(DATA, "ENNI", "SLI"), "ENNI", "SLI", stem)
    return pd.DataFrame(rows)

def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    ss = np.sum((y - y.mean())**2)
    return float(1 - np.sum((y - p)**2)/ss) if ss else float("nan")

def oof_predict(model_fn, X, y, groups, n_splits=5):
    """Out-of-fold predictions, GroupKFold by child (child-independent)."""
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = model_fn()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred

def cluster_bootstrap(df, pred_col, n=2000):
    """95% CI for MAE and R2, resampling CHILDREN with replacement (files from
    the same child are not independent, so a plain bootstrap would be too tight)."""
    by_child = {c: g for c, g in df.groupby("child_id")}
    kids = list(by_child)
    maes, r2s = [], []
    for _ in range(n):
        take = RNG.choice(kids, size=len(kids), replace=True)
        s = pd.concat([by_child[k] for k in take])
        maes.append(mae(s["age_months"], s[pred_col]))
        r2s.append(r2(s["age_months"], s[pred_col]))
    lo, hi = np.percentile(maes, [2.5, 97.5]); lo2, hi2 = np.percentile(r2s, [2.5, 97.5])
    return (lo, hi), (lo2, hi2)

def main():
    df = build_features()
    df.to_csv(os.path.join(RESULTS, "features.csv"), index=False)
    print(f"features.csv written: {len(df)} files, "
          f"{df['child_id'].nunique()} children, groups: {dict(df['group'].value_counts())}")

    td = df[(df["group"] == "TD") & df["age_months"].notna()].dropna(subset=FEATURES).copy()
    print(f"\nTD modeling set: {len(td)} files, {td['child_id'].nunique()} distinct children, "
          f"ages {td['age_months'].min():.0f}-{td['age_months'].max():.0f} mo "
          f"(Brown {sum(td['corpus']=='Brown')}, ENNI {sum(td['corpus']=='ENNI')})")

    X = td[FEATURES].to_numpy()
    y = td["age_months"].to_numpy()
    groups = td["child_id"].to_numpy()

    models = {
        "baseline (predict mean)": lambda: _Mean(),
        "ridge (linear)":          lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "grad-boosting":           lambda: HistGradientBoostingRegressor(random_state=0),
    }
    print("\n=== Child-independent CV (GroupKFold by child, 5 folds) ===")
    best_name, best_mae = None, np.inf
    for name, fn in models.items():
        td[f"pred::{name}"] = oof_predict(fn, X, y, groups)
        m, r = mae(y, td[f"pred::{name}"]), r2(y, td[f"pred::{name}"])
        (mlo, mhi), (rlo, rhi) = cluster_bootstrap(td, f"pred::{name}")
        print(f"  {name:24s} MAE = {m:5.2f} mo  [95% CI {mlo:.2f}..{mhi:.2f}]   "
              f"R2 = {r:5.2f} [{rlo:.2f}..{rhi:.2f}]")
        if m < best_mae and "baseline" not in name:
            best_name, best_mae = name, m
    td["pred"] = td[f"pred::{best_name}"]
    print(f"  -> best: {best_name}")

    print("\n=== Per-corpus breakdown (best model, out-of-fold) ===")
    for corpus, g in td.groupby("corpus"):
        print(f"  {corpus:6s} n={len(g):3d}  MAE = {mae(g['age_months'], g['pred']):5.2f} mo  "
              f"ages {g['age_months'].min():.0f}-{g['age_months'].max():.0f}")

    # ---- age-gap score: predicted language age - chronological age ----
    td["gap"] = td["pred"] - td["age_months"]
    print(f"\nTD age-gap (out-of-fold): mean {td['gap'].mean():+.2f} mo, sd {td['gap'].std():.2f}")

    # ---- SLI preview (Stage 3 teaser, NOT a validated claim yet): train on all
    # TD, apply to ENNI-SLI; compare gaps against ENNI-TD's own out-of-fold gaps
    # (same narrative task, mean-matched ages -> the fair within-task comparison).
    final = models[best_name]()
    final.fit(X, y)
    sli = df[(df["group"] == "SLI") & df["age_months"].notna()].dropna(subset=FEATURES).copy()
    sli["gap"] = final.predict(sli[FEATURES].to_numpy()) - sli["age_months"]
    enni_td = td[td["corpus"] == "ENNI"]
    d = (sli["gap"].mean() - enni_td["gap"].mean()) / enni_td["gap"].std()
    print(f"\n=== SLI preview (informal; Stage 3 will do this rigorously) ===")
    print(f"  ENNI-TD  gap: mean {enni_td['gap'].mean():+6.2f} mo (sd {enni_td['gap'].std():.2f}, n={len(enni_td)})")
    print(f"  ENNI-SLI gap: mean {sli['gap'].mean():+6.2f} mo (sd {sli['gap'].std():.2f}, n={len(sli)})")
    print(f"  -> SLI children's estimated language age lags TD peers by "
          f"{enni_td['gap'].mean() - sli['gap'].mean():.1f} months (Cohen's d ~ {abs(d):.2f})")

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(6, 6))
    for corpus, color in [("Brown", "tab:blue"), ("ENNI", "tab:orange")]:
        g = td[td["corpus"] == corpus]
        ax.scatter(g["age_months"], g["pred"], s=12, alpha=0.5, color=color, label=corpus)
    lim = [15, 125]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="chronological age (months)",
           ylabel="predicted language age (months)",
           title=f"Stage 2 level estimator - child-independent ({best_name})")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "stage2_calibration.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(-40, 41, 4)
    ax.hist(enni_td["gap"], bins=bins, alpha=0.6, density=True, label="ENNI TD (out-of-fold)")
    ax.hist(sli["gap"], bins=bins, alpha=0.6, density=True, label="ENNI SLI")
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="age gap: predicted language age - actual age (months)",
           ylabel="density", title="Age-gap score, TD vs SLI (same task) - preview")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "stage2_gap_preview.png"), dpi=150)
    print(f"\nfigures: results/stage2_calibration.png, results/stage2_gap_preview.png")

class _Mean:
    def fit(self, X, y): self.m = float(np.mean(y)); return self
    def predict(self, X): return np.full(len(X), self.m)

if __name__ == "__main__":
    main()
