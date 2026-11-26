#!/usr/bin/env python3
# FirstWords - Extension: CALIBRATE A SCREEN WITH TYPICAL CHILDREN ONLY.
#
# Problem (ISS-18, ISS-31): the ranking transfers to new labs and ages, but the cutoff does not.
# The ENNI cutoff catches only ~58% of impaired Gillam children, and flags 100% of toddlers.
# The usual fix needs DIAGNOSED children from the new setting, who are rare, slow and costly to
# find. Typical children are everywhere.
#
# Idea: in a new population, set the cutoff at a chosen percentile of the risk scores of
# TYPICAL children only (e.g. 85th -> ~15% false alarms by construction), using no diagnosed
# children at all. Then measure sensitivity on children NOT used to set it.
# Benchmark = the "label oracle": the best sensitivity any cutoff could reach at the same
# specificity using everyone's true labels (in-sample, so an UPPER bound).
#
# Cross-fit: 500 random splits; the cutoff uses half the typical children, evaluation uses
# all impaired children + the other half of the typical children.
# USAGE: .venv/bin/python src/ext_young_delay.py && .venv/bin/python src/ext_typical_only_calibration.py
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve
import stage3_delay_classifier as s3
import ext_crosscorpus as cc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
LANG = ["mluw", "mlum", "mattr50"]


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))


def typical_only(d, pct, reps=500, seed=0):
    rng = np.random.default_rng(seed); td = d.index[d.y == 0].to_numpy(); out = []
    for _ in range(reps):
        cal = rng.choice(td, len(td) // 2, replace=False); t = d.drop(cal)
        f = t.p >= np.percentile(d.loc[cal, "p"], pct)
        out.append(((f & (t.y == 1)).sum() / t.y.sum(), (~f & (t.y == 0)).sum() / (t.y == 0).sum()))
    return np.array(out)


def oracle_sens(d, spec):
    fpr, tpr, _ = roc_curve(d.y, d.p); ok = np.where(1 - fpr >= spec)[0]
    return float(tpr[ok[-1]]) if len(ok) else np.nan


def main():
    enni = s3.load_enni()
    full = clf().fit(enni[s3.FEATURES], enni.y); lang = clf().fit(enni[LANG], enni.y)
    fpr, tpr, thr = roc_curve(enni.y, s3.oof_proba(clf, enni[s3.FEATURES].to_numpy(), enni.y.to_numpy(),
                                                    enni.child_id.to_numpy()))
    enni_cut = thr[np.argmax(tpr >= 0.90)]

    gil = cc.extract_corpus(os.path.join(ROOT, "data", "childes", "Gillam"))
    gil = gil[gil.group.isin(["TD", "SLI"])].dropna(subset=s3.FEATURES).reset_index(drop=True)
    gil["y"] = (gil.group == "SLI").astype(int); gil["p"] = full.predict_proba(gil[s3.FEATURES])[:, 1]
    ew = pd.read_csv(os.path.join(RESULTS, "ellisweismer_scores.csv"))
    tod = ew[ew.visit == 30].groupby("child_id").agg(y=("y", "first"), **{c: (c, "mean") for c in LANG}).reset_index()
    tod["p"] = lang.predict_proba(tod[LANG])[:, 1]

    rows = []
    for name, d, cut in [("Gillam, 5-12 yr (full model)", gil, enni_cut), ("EllisWeismer, 30 mo (language-only)", tod, None)]:
        n_td = int((d.y == 0).sum())
        print(f"\n{name}: {len(d)} children ({int(d.y.sum())} delayed); cutoff set from ~{n_td // 2} typical children")
        if cut is not None:
            f = d.p >= cut
            print(f"  borrowed ENNI cutoff {cut:.2f}: sens {(f & (d.y == 1)).sum() / d.y.sum():.2f}  "
                  f"spec {(~f & (d.y == 0)).sum() / n_td:.2f}")
        for pct in (80, 85, 90):
            r = typical_only(d, pct)
            o = oracle_sens(d, pct / 100)
            rows.append({"population": name, "target_spec": pct / 100, "spec": r[:, 1].mean(), "sens": r[:, 0].mean(),
                         "sens_lo": np.percentile(r[:, 0], 2.5), "sens_hi": np.percentile(r[:, 0], 97.5), "oracle_sens": o})
            print(f"  typical-only, target spec {pct / 100:.2f}: achieved spec {r[:, 1].mean():.2f}  sens {r[:, 0].mean():.2f} "
                  f"[{np.percentile(r[:, 0], 2.5):.2f}-{np.percentile(r[:, 0], 97.5):.2f}]  | label oracle sens {o:.2f}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "typical_only_calibration.csv"), index=False)
    print("\nRead: with no diagnosed children, a new setting hits its chosen false-alarm rate and gets")
    print("about the sensitivity that full labels would allow. Sensitivity itself is limited by ranking.")


if __name__ == "__main__":
    main()
