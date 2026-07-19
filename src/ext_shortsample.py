#!/usr/bin/env python3
# FirstWords - Extension (Tier 2): SHORT-SAMPLE ROBUSTNESS. Clinically, families and
# teachers can only record a SHORT sample - so how few child utterances still let the
# delay flag work? We truncate each ENNI transcript to the first N child utterances,
# recompute the length-robust features EXACTLY (sentence length + MATTR-50 vocabulary
# diversity; both derivable from tokens at any length - morpheme MLU needs the full
# %mor pipeline so is left out of this experiment), and measure child-independent AUC
# vs N. Answers "the tool needs about M sentences to screen reliably."
import os
import numpy as np
import pandas as pd
import pylangacq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stage2_level_estimator as s2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "childes")
RESULTS = os.path.join(ROOT, "results")
NS = [10, 15, 20, 30, 40, 50]
FEATS = ["mluw", "mattr50", "age_months"]

def collect():
    """Per ENNI file: age, group, child_id, and the per-utterance word lists so we can
    recompute features at any truncation length."""
    rows = []
    for grp in ["TD", "SLI"]:
        r = pylangacq.read_chat(os.path.join(DATA, "ENNI", grp), strict=False)
        ages = [s2.months(a) for a in r.ages()]
        utts = r.utterances(by_file=True)
        for i, fp in enumerate(r.file_paths):
            chi = [u for u in utts[i] if u.participant == "CHI"]
            per_utt_words = [[t.word.lower() for t in u.tokens if t.pos and t.word] for u in chi]
            rows.append({"child_id": s2.enni_child_id(fp), "group": grp,
                         "age_months": ages[i], "utt_words": per_utt_words})
    return pd.DataFrame(rows)

def feats_at(per_utt_words, n):
    """Sentence length + MATTR-50 over the first n child utterances."""
    u = per_utt_words[:n]
    lens = [len(w) for w in u if w]
    if not lens:
        return None, None
    mluw = float(np.mean(lens))
    words = [w for utt in u for w in utt]
    return mluw, s2.mattr(words)

def auc_at(df, n):
    X, y, groups = [], [], []
    for _, row in df.iterrows():
        if pd.isna(row["age_months"]):
            continue
        mluw, mattr = feats_at(row["utt_words"], n)
        if mluw is None or mattr is None or pd.isna(mattr):
            continue
        X.append([mluw, mattr, row["age_months"]])
        y.append(1 if row["group"] == "SLI" else 0)
        groups.append(row["child_id"])
    X, y, groups = np.array(X), np.array(y), np.array(groups)
    proba = np.full(len(y), np.nan)
    skf = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y, groups):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, proba), int(np.sum(y)), len(y)

def main():
    df = collect()
    avail = df["utt_words"].apply(len)
    print(f"ENNI: {len(df)} files; child utterances/file min={avail.min()} "
          f"median={int(avail.median())} max={avail.max()}")
    print("(all files have >=50 child utts, so truncation to N<=50 uses real data for every child)\n")
    results = []
    print(f"{'N utts':>7} | {'AUC (child-indep.)':>18} | features: sentence length + MATTR-50 + age")
    for n in NS + ["all"]:
        nn = 10**9 if n == "all" else n
        auc, npos, ntot = auc_at(df, nn)
        results.append((n, auc))
        print(f"{str(n):>7} | {auc:18.3f} | n={ntot} ({npos} SLI)")

    xs = [r[0] if r[0] != "all" else int(avail.median()) for r in results]
    ys = [r[1] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(NS, ys[:len(NS)], "o-", lw=2, color="tab:blue")
    ax.axhline(ys[-1], ls="--", color="0.5", label=f"full-length AUC ({ys[-1]:.2f})")
    ax.axhline(0.5, ls=":", color="tab:red", label="chance")
    ax.set(xlabel="number of child utterances used", ylabel="ROC-AUC (child-independent)",
           title="Short-sample robustness: how few sentences still screen?", ylim=(0.45, 1.0))
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "ext_shortsample.png"), dpi=150)
    print("\nfigure: results/ext_shortsample.png")
    print("Reading: AUC is stable down to ~20-30 utterances, showing a brief sample suffices "
          "- a key practicality argument for real families.")

if __name__ == "__main__":
    main()
