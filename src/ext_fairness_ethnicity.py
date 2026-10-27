#!/usr/bin/env python3
# FirstWords - fairness across ethnic groups on the LOCKED external test set (Gillam).
#
# Question: language tests built on mainstream-English grammar can mistake dialect for
# disorder. Our full model counts MORPHEMES (word parts like "is", "-s", "-ed"), which are
# exactly the forms that differ in African American English and in English learners. The
# words-only model ignores them. Does the morpheme-based screen flag typically developing
# Black or Hispanic children more often than White children, and does the words-only screen
# reduce that gap?
#
# Design (no refitting on Gillam; Gillam was never used for training or tuning):
#   - train both screens on ENNI (as in the main results); set each cutoff on ENNI
#     out-of-fold predictions at >=90% sensitivity
#   - apply to every Gillam child; ethnicity from the transcript @Comment header
#   - per group: AUC, sensitivity, FALSE-ALARM RATE among typically developing children
#     (the bias that matters for a screen), with bootstrap / Wilson 95% intervals
#   - mechanism check: morphemes per word among typically developing children, by group
# Caveat: Gillam's impairment labels come from standardized testing, which can itself be
# biased; small groups give wide intervals.
#
# USAGE: .venv/bin/python src/ext_fairness_ethnicity.py
import os, sys, re, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve
import stage3_delay_classifier as s3
import ext_crosscorpus as cc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GILLAM = os.path.join(ROOT, "data", "childes", "Gillam")
MODELS = {"full (morphemes)": ["mluw", "mlum", "mattr50", "age_months"],
          "words-only": ["mluw", "mattr50", "age_months"]}
RNG = np.random.default_rng(7)

def ethnicity(fp):
    with open(fp, encoding="utf-8", errors="ignore") as f:
        m = re.search(r"@Comment:\s*Ethnic(?:ity)?\s*=\s*(.*)", f.read())
    e = (m.group(1).strip().lower() if m else "")
    if e.startswith(("european", "eurpean")): return "European American"
    if e.startswith("african american/") : return "Other / unknown"
    if e.startswith("african"): return "African American"
    if e.startswith("hispanic"): return "Hispanic American"
    return "Other / unknown"

def wilson(k, n):
    if n == 0: return (np.nan, np.nan)
    p, z = k / n, 1.96
    c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return c - h, c + h

def boot_auc(y, p, n=2000):
    v = []
    for _ in range(n):
        s = RNG.integers(0, len(y), len(y))
        if len(set(y[s])) == 2: v.append(roc_auc_score(y[s], p[s]))
    return np.percentile(v, [2.5, 97.5])

def main():
    enni = s3.load_enni(); y_e, g_e = enni["y"].to_numpy(), enni["child_id"].to_numpy()
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))

    files = {os.path.basename(f): f for f in glob.glob(os.path.join(GILLAM, "**", "*.cha"), recursive=True)}
    gil = cc.extract_corpus(GILLAM)
    gil = gil[gil.group.isin(["TD", "SLI"])].dropna(subset=MODELS["full (morphemes)"]).copy()
    gil["eth"] = gil["file"].map(lambda b: ethnicity(files[b]))
    gil["y"] = (gil.group == "SLI").astype(int)
    print(f"Gillam (locked external set): {len(gil)} children")
    print(gil.groupby("eth").y.agg(n="size", impaired="sum").to_string(), "\n")

    groups = ["European American", "African American", "Hispanic American", "Other / unknown"]
    out = []
    for name, feats in MODELS.items():
        p_oof = s3.oof_proba(lr, enni[feats].to_numpy(), y_e, g_e)
        fpr, tpr, thr = roc_curve(y_e, p_oof); cut = float(thr[int(np.argmax(tpr >= 0.90))])
        m = lr().fit(enni[feats], enni["y"])
        gil["p"] = m.predict_proba(gil[feats])[:, 1]
        print(f"=== {name}: ENNI cutoff {cut:.3f} (>=90% sensitivity on ENNI) ===")
        for grp in ["ALL"] + groups:
            d = gil if grp == "ALL" else gil[gil.eth == grp]
            y, p = d.y.to_numpy(), d.p.to_numpy()
            auc = roc_auc_score(y, p); lo, hi = boot_auc(y, p)
            td, sli = d[d.y == 0], d[d.y == 1]
            fa = int((td.p >= cut).sum()); fa_lo, fa_hi = wilson(fa, len(td))
            hit = int((sli.p >= cut).sum())
            out.append((name, grp, auc, fa / len(td)))
            print(f"  {grp:18s} n={len(d):3d} ({len(sli):3d} impaired)  AUC {auc:.2f} [{lo:.2f}-{hi:.2f}]  "
                  f"sensitivity {hit/len(sli):.2f}  FALSE ALARMS in typical kids {fa}/{len(td)} = "
                  f"{fa/len(td):.0%} [{fa_lo:.0%}-{fa_hi:.0%}]")
        # false-alarm gap vs European American, bootstrap 95% CI
        ea = gil[(gil.eth == "European American") & (gil.y == 0)].p.to_numpy() >= cut
        for grp in ["African American", "Hispanic American"]:
            o = gil[(gil.eth == grp) & (gil.y == 0)].p.to_numpy() >= cut
            diffs = [RNG.choice(o, len(o)).mean() - RNG.choice(ea, len(ea)).mean() for _ in range(4000)]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print(f"  false-alarm gap {grp} minus European American: {o.mean()-ea.mean():+.0%} [{lo:+.0%} to {hi:+.0%}]")
        print()

    # mechanism: morphemes per word in typically developing children (dialect marks fewer
    # inflections, so a dialect effect shows up here, not in plain word counts)
    td = gil[gil.y == 0].copy(); td["mpw"] = td.mlum / td.mluw
    print("Mechanism check, typically developing children only (age-matched means):")
    for grp in groups[:3]:
        d = td[td.eth == grp]
        print(f"  {grp:18s} n={len(d):3d}  age {d.age_months.mean():5.1f} mo  words/utt {d.mluw.mean():.2f}  "
              f"morphemes/utt {d.mlum.mean():.2f}  morphemes per word {d.mpw.mean():.3f}")

if __name__ == "__main__":
    main()
