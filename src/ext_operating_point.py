#!/usr/bin/env python3
# FirstWords - the screening operating point, done strictly.
#
#  1. NESTED cutoff on ENNI: the cutoff (target: catch 90% of impaired children) is chosen
#     ONLY from the training folds, then applied to the held-out fold. This is the honest
#     sensitivity / specificity (the plain version picks the cutoff on the same out-of-fold
#     predictions it reports on).
#  2. TRANSFER to outside labs: the model is never refit. We compare (a) the ENNI cutoff
#     used as-is with (b) LOCAL RECALIBRATION: the cutoff is set from a small sample of
#     typically developing local children (no diagnoses needed), and performance is
#     measured on all the OTHER children. Repeated 400 times with different samples.
#  3. REAL-WORLD rates: what the ENNI operating point means at 7% prevalence.
#
# USAGE: .venv/bin/python src/ext_operating_point.py     -> results/operating_point.csv
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
import stage3_delay_classifier as s3
import ext_crosscorpus as cc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = ["mluw", "mlum", "mattr50", "age_months"]
WORDS = ["mluw", "mattr50", "age_months"]
LANG = ["mluw", "mlum", "mattr50"]           # age-free, for out-of-age-range corpora
TARGET, PREVALENCE = 0.90, 0.07
N_LOCAL, LOCAL_QUANTILE, REPEATS = 30, 0.65, 400
RNG = np.random.default_rng(0)

def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))

def cutoff_for(y, p, target=TARGET):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr >= target))])

def wilson(k, n):
    p, z = k / n, 1.96
    c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return c - h, c + h

def nested(enni, feats):
    """Outer 5 folds for testing; inside each training set, 5 more folds pick the cutoff."""
    X, y, g = enni[feats].to_numpy(), enni["y"].to_numpy(), enni["child_id"].to_numpy()
    p, flag = np.zeros(len(y)), np.zeros(len(y), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        inner = np.zeros(len(tr))
        for itr, ite in StratifiedGroupKFold(5, shuffle=True, random_state=1).split(X[tr], y[tr], g[tr]):
            inner[ite] = lr().fit(X[tr][itr], y[tr][itr]).predict_proba(X[tr][ite])[:, 1]
        c = cutoff_for(y[tr], inner)
        p[te] = lr().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        flag[te] = p[te] >= c
    return y, p, flag

def local_recalibration(y, p):
    """Cutoff = the LOCAL_QUANTILE point of N_LOCAL randomly drawn typical children's scores;
    sensitivity / specificity measured on every child not used to set it."""
    td = np.where(y == 0)[0]; sens, spec = [], []
    for _ in range(REPEATS):
        cal = RNG.choice(td, N_LOCAL, replace=False)
        rest = np.setdiff1d(np.arange(len(y)), cal)
        c = np.quantile(p[cal], LOCAL_QUANTILE)
        sens.append(np.mean(p[rest][y[rest] == 1] >= c)); spec.append(np.mean(p[rest][y[rest] == 0] < c))
    return np.array(sens), np.array(spec)

def main():
    enni = s3.load_enni()
    rows = []
    print(f"1) ENNI, cutoff chosen only inside the training folds (target {TARGET:.0%} sensitivity)")
    for name, feats in [("full model", FULL), ("words-only", WORDS)]:
        y, p, flag = nested(enni, feats)
        k, n = int(flag[y == 1].sum()), int((y == 1).sum())
        lo, hi = wilson(k, n); spec = 1 - flag[y == 0].mean()
        print(f"   {name:11s} AUC {roc_auc_score(y, p):.3f}  catches {k}/{n} = {k/n:.0%} [{lo:.0%}-{hi:.0%}]  "
              f"clears {spec:.0%} of typical children")
        if name == "full model":
            sens_e, spec_e = k / n, spec
            rows.append(("ENNI (held-out children)", "cutoff from training folds", sens_e, lo, hi, spec_e))

    print(f"\n2) Outside labs, model never refit. Local recalibration = cutoff at the {LOCAL_QUANTILE:.0%} "
          f"point of {N_LOCAL} local typical children")
    y_e = enni["y"].to_numpy(); g_e = enni["child_id"].to_numpy()
    ext = {n: cc.extract_corpus(pth) for n, pth in cc.find_external_corpora()}
    for corpus, feats, label in [("Gillam", FULL, "Gillam, USA"), ("Conti4", LANG, "Conti-Ramsden, UK")]:
        d = ext[corpus].dropna(subset=FULL); d = d[d.group.isin(["TD", "SLI"])]
        if corpus == "Conti4":
            d = d[d.task == "frog"]
        y = (d.group == "SLI").astype(int).to_numpy()
        c_enni = cutoff_for(y_e, s3.oof_proba(lr, enni[feats].to_numpy(), y_e, g_e))
        p = lr().fit(enni[feats], y_e).predict_proba(d[feats])[:, 1]
        s0, sp0 = np.mean(p[y == 1] >= c_enni), np.mean(p[y == 0] < c_enni)
        s, sp = local_recalibration(y, p)
        print(f"   {label:18s} n={len(y)} ({int(y.sum())} impaired)  AUC {roc_auc_score(y, p):.2f}")
        print(f"      ENNI cutoff as-is:    catches {s0:.0%}, clears {sp0:.0%}")
        print(f"      local recalibration:  catches {s.mean():.0%} [{np.percentile(s, 2.5):.0%}-{np.percentile(s, 97.5):.0%}], "
              f"clears {sp.mean():.0%}")
        rows.append((label, "ENNI cutoff as-is", s0, np.nan, np.nan, sp0))
        rows.append((label, f"recalibrated with {N_LOCAL} local typical children", s.mean(),
                     np.percentile(s, 2.5), np.percentile(s, 97.5), sp.mean()))

    print(f"\n3) What the ENNI operating point means at {PREVALENCE:.0%} prevalence (sens {sens_e:.2f}, spec {spec_e:.2f})")
    tp, fn = PREVALENCE * sens_e, PREVALENCE * (1 - sens_e)
    tn, fp = (1 - PREVALENCE) * spec_e, (1 - PREVALENCE) * (1 - spec_e)
    print(f"   cleared ('not elevated'): {tn + fn:.0%} of children, and {tn / (tn + fn):.1%} of them are truly typical")
    print(f"   flagged for follow-up:    {tp + fp:.0%} of children, containing {sens_e:.0%} of all affected children "
          f"({tp / (tp + fp):.0%} of flagged children are affected)")
    out = pd.DataFrame(rows, columns=["dataset", "cutoff", "sensitivity", "lo", "hi", "specificity"])
    out.to_csv(os.path.join(ROOT, "results", "operating_point.csv"), index=False)
    print("\nwrote results/operating_point.csv")

if __name__ == "__main__":
    main()
