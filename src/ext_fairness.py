#!/usr/bin/env python3
# FirstWords - Extension (Tier 2): FAIRNESS / SUBGROUP AUDIT. A screening tool must
# work about equally well across groups, or it will over- or under-refer some children.
# Using child-independent out-of-fold predictions from the delay model (ENNI), we report
# AUC and sensitivity/specificity (at ONE global screening threshold) broken down by SEX
# and by AGE BAND, and flag any large disparity. Reported honestly incl. small-cell CIs.
import os, re
import numpy as np
import pandas as pd
import pylangacq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
import stage2_level_estimator as s2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "childes")
DELAY_FEATURES = ["mluw", "mlum", "mattr50", "age_months"]
RNG = np.random.default_rng(0)

def sex_of(fp):
    with open(fp, encoding="utf-8", errors="ignore") as f:
        m = re.search(r"\|CHI\|[^|]*\|(male|female)\|", f.read())
    return m.group(1) if m else "unknown"

def collect():
    rows = []
    for grp in ["TD", "SLI"]:
        r = pylangacq.read_chat(os.path.join(DATA, "ENNI", grp), strict=False)
        ages = [s2.months(a) for a in r.ages()]
        mluw, mlum, ttr, ipsyn = r.mluw(), r.mlum(), r.ttr(), r.ipsyn()
        utts = r.utterances(by_file=True)
        for i, fp in enumerate(r.file_paths):
            chi = [u for u in utts[i] if u.participant == "CHI"]
            words = [t.word.lower() for u in chi for t in u.tokens if t.pos and t.word]
            rows.append({"child_id": s2.enni_child_id(fp), "group": grp, "sex": sex_of(fp),
                         "age_months": ages[i], "mluw": mluw[i], "mlum": mlum[i],
                         "mattr50": s2.mattr(words)})
    df = pd.DataFrame(rows).dropna(subset=DELAY_FEATURES)
    df["y"] = (df["group"] == "SLI").astype(int)
    return df

def oof_proba(df):
    X, y, g = df[DELAY_FEATURES].to_numpy(), df["y"].to_numpy(), df["child_id"].to_numpy()
    proba = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    return proba

def metrics(y, p, thr):
    y, pred = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return auc, sens, spec

def report(df, by, thr):
    print(f"\n  by {by}:")
    print(f"    {'subgroup':<14}{'n':>5}{'nSLI':>6}{'AUC':>8}{'sens':>8}{'spec':>8}")
    for key, g in df.groupby(by, observed=True):
        auc, sens, spec = metrics(g["y"], g["proba"], thr)
        fmt = lambda v: f"{v:.2f}" if not np.isnan(v) else "  -"
        print(f"    {str(key):<14}{len(g):>5}{int(g['y'].sum()):>6}{fmt(auc):>8}{fmt(sens):>8}{fmt(spec):>8}")

def main():
    df = collect()
    df["proba"] = oof_proba(df)
    # global screening threshold ~90% sensitivity (same policy as Stage 3)
    fpr, tpr, thr = roc_curve(df["y"], df["proba"])
    t = thr[int(np.argmin(np.abs(tpr - 0.90)))]
    auc_all, sens_all, spec_all = metrics(df["y"], df["proba"], t)
    print(f"ENNI delay model - fairness audit (child-independent OOF)")
    print(f"  overall: n={len(df)}  AUC={auc_all:.3f}  at screening thr={t:.2f}: "
          f"sens={sens_all:.2f} spec={spec_all:.2f}")

    report(df, "sex", t)
    df["age_band"] = pd.cut(df["age_months"], [0, 72, 96, 999],
                            labels=["4-5y (<72mo)", "6-7y (72-95mo)", "8-10y (>=96mo)"])
    report(df, "age_band", t)

    # disparity flag
    sens_by_sex = [metrics(g["y"], g["proba"], t)[1]
                   for k, g in df[df["sex"] != "unknown"].groupby("sex", observed=True)]
    spread = np.nanmax(sens_by_sex) - np.nanmin(sens_by_sex)
    print(f"\n  sensitivity spread across sex = {spread:.2f} "
          f"({'acceptable (<0.15)' if spread < 0.15 else 'NOTABLE - investigate / report'})")
    print("  NOTE: some cells are small (SLI n=75 split across groups) -> subgroup estimates")
    print("  are noisy; report as an equity CHECK, not a precise per-group performance claim.")

if __name__ == "__main__":
    main()
