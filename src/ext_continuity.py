#!/usr/bin/env python3
# FirstWords - Extension: ONE LANGUAGE DIMENSION FROM TODDLERHOOD TO SCHOOL AGE?
#
# Question (developmental science, not just the tool): is toddler LATE TALKING on the same
# measurable dimension as school-age DEVELOPMENTAL LANGUAGE DISORDER (DLD/SLI)? Whether late
# talking and later DLD are continuous is debated in the developmental literature.
#
# Test = BIDIRECTIONAL TRANSFER with no refit and NO AGE feature (language only:
# words/utt, morphemes/utt, MATTR-50):
#   school -> toddler: train ENNI (4-10 yr, TD vs SLI), test EllisWeismer 30 mo (TD vs LT)
#   toddler -> school: train EllisWeismer 30 mo, test ENNI and Gillam (TD vs SLI)
# plus the standardized coefficients of each model: if both point the same way with the same
# dominant marker, the two conditions share a dimension.
#
# One row per CHILD at 30 mo (two tasks averaged). Bootstrap 95% CIs resample children.
# Caveats: different labs, tasks and labels (LT = vocabulary-based at ~2 yr; SLI = test-based
# at school age); this shows a shared MARKER dimension, not that the same children go on to
# have DLD (that needs longitudinal outcome data).
# USAGE: .venv/bin/python src/ext_young_delay.py   (writes results/ellisweismer_scores.csv)
#        .venv/bin/python src/ext_continuity.py
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import stage3_delay_classifier as s3
import ext_crosscorpus as cc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
LANG = ["mluw", "mlum", "mattr50"]
PRETTY = {"mluw": "words per sentence", "mlum": "word-parts per sentence", "mattr50": "vocabulary diversity"}


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))


def auc_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); v = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) == 2: v.append(roc_auc_score(y[s], p[s]))
    return roc_auc_score(y, p), np.percentile(v, 2.5), np.percentile(v, 97.5)


def load():
    ew = pd.read_csv(os.path.join(RESULTS, "ellisweismer_scores.csv"))
    tod = ew[ew.visit == 30].groupby("child_id").agg(y=("y", "first"), **{c: (c, "mean") for c in LANG}).reset_index()
    enni = s3.load_enni()
    gil = cc.extract_corpus(os.path.join(ROOT, "data", "childes", "Gillam"))
    gil = gil[gil.group.isin(["TD", "SLI"])].dropna(subset=LANG).copy()
    gil["y"] = (gil.group == "SLI").astype(int)
    return tod, enni, gil


def main():
    tod, enni, gil = load()
    m_school, m_tod = clf().fit(enni[LANG], enni.y), clf().fit(tod[LANG], tod.y)
    rows = []
    print("Bidirectional transfer, language only (no age), no refit. ROC-AUC [95% CI]\n")
    for train, m, test, d in [("ENNI (4-10 yr)", m_school, "EllisWeismer 30 mo", tod),
                              ("EllisWeismer 30 mo", m_tod, "ENNI (4-10 yr)", enni),
                              ("EllisWeismer 30 mo", m_tod, "Gillam (5-12 yr)", gil)]:
        a, lo, hi = auc_ci(d.y.to_numpy(), m.predict_proba(d[LANG])[:, 1])
        rows.append({"train": train, "test": test, "auc": a, "lo": lo, "hi": hi, "n": len(d), "n_pos": int(d.y.sum())})
        print(f"  train {train:20s} -> test {test:20s} AUC {a:.2f} [{lo:.2f}-{hi:.2f}]  (n={len(d)}, {int(d.y.sum())} delayed)")
    a, lo, hi = auc_ci(enni.y.to_numpy(), s3.oof_proba(clf, enni[LANG].to_numpy(), enni.y.to_numpy(),
                                                        enni.child_id.to_numpy()))
    print(f"  reference: ENNI's own language-only model, child-independent CV: AUC {a:.2f} [{lo:.2f}-{hi:.2f}]")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "continuity_transfer.csv"), index=False)

    print("\nStandardized coefficients (negative = lower value -> more likely delayed):")
    coef = pd.DataFrame({"toddler (late talking)": m_tod[-1].coef_[0], "school-age (DLD)": m_school[-1].coef_[0]},
                        index=[PRETTY[c] for c in LANG])
    print(coef.round(2).to_string())
    coef.to_csv(os.path.join(RESULTS, "continuity_coefs.csv"))
    print("\nRead: if both directions transfer and the same marker dominates, toddler late talking and")
    print("school-age DLD sit on one shared language dimension (sentence length in word-parts).")


if __name__ == "__main__":
    main()
