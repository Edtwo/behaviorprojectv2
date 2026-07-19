#!/usr/bin/env python3
# FirstWords - Stage 3: the HEADLINE deliverable (signal-gated). A language-DELAY
# classifier: typically-developing (TD) vs language-impaired (SLI) on ENNI, the
# SAME narrative task at matched mean age. This is the rigorous version of the
# Stage 2 SLI preview.
#
# Rigor / artifact controls (per handoff Lessons 4 & Section 0c):
#  - ENNI-ONLY. Brown is excluded: it is a different task (home conversation) and a
#    younger age range, so mixing it in would let the model detect task/age, not delay.
#  - Child-independent via StratifiedGroupKFold grouped on the header-derived child id
#    (birthdate+sex): ENNI's A/B folders can share a child (ISS-07), so we must NOT
#    split by file. Stratification preserves the 79/21 class ratio across folds.
#  - Length-robust features only (MLU-words/morphemes, MATTR-50). Raw TTR and the
#    syntax_index are EXCLUDED because they are transcript-length confounded (ISS-02/05).
#  - AGE is included as a covariate so the model cannot exploit residual age mixing
#    (ISS-04); we also report the AUC of age-alone and length-alone to prove those
#    artifacts do NOT carry the signal.
#  - Class imbalance (286 TD : 75 SLI) handled with class_weight='balanced' and
#    reported with balanced metrics (ROC-AUC, PR-AUC, balanced accuracy, sensitivity/
#    specificity), not raw accuracy. Cluster-bootstrap CIs resample children.
#  - Screening framing: we pick the operating threshold for HIGH SENSITIVITY (a
#    screen should rarely miss an affected child), and report the trade-off.
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
RNG = np.random.default_rng(42)

LANG_FEATURES = ["mluw", "mlum", "mattr50"]      # length-robust language markers
COVARIATES = ["age_months"]                       # control for residual age mixing
FEATURES = LANG_FEATURES + COVARIATES

def load_enni():
    csv = os.path.join(RESULTS, "features.csv")
    if not os.path.exists(csv):
        # regenerate features from raw transcripts (Stage 2 owns the extraction)
        import stage2_level_estimator as s2
        s2.build_features().to_csv(csv, index=False)
    df = pd.read_csv(csv)
    enni = df[df["corpus"] == "ENNI"].copy()
    enni = enni.dropna(subset=FEATURES + ["group"])
    enni["y"] = (enni["group"] == "SLI").astype(int)   # 1 = impaired (positive class)
    return enni

def oof_proba(model_fn, X, y, groups, n_splits=5):
    """Out-of-fold predicted P(SLI), StratifiedGroupKFold: no child crosses folds
    (grouped by header child id) and class ratio is preserved in each fold."""
    proba = np.full(len(y), np.nan)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y, groups):
        m = model_fn()
        m.fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    return proba

def boot_ci(y, p, groups, metric, n=2000):
    """Cluster bootstrap: resample CHILDREN with replacement (a child's A+B files are
    not independent), recompute the metric on their pooled files."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    by_child = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    kids = list(by_child)
    vals = []
    for _ in range(n):
        take = RNG.choice(kids, size=len(kids), replace=True)
        s = np.concatenate([by_child[k] for k in take])
        if len(np.unique(y[s])) < 2:   # need both classes for AUC
            continue
        vals.append(metric(y[s], p[s]))
    return np.percentile(vals, [2.5, 97.5])

def sens_spec_at(y, p, thr):
    y, pred = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec, (tp, fp, tn, fn)

def main():
    enni = load_enni()
    n_sli, n_td = int(enni["y"].sum()), int((1 - enni["y"]).sum())
    groups = enni["child_id"].to_numpy()
    print(f"ENNI classification set: {len(enni)} files, {enni['child_id'].nunique()} distinct "
          f"children  (TD {n_td}, SLI {n_sli} files; prevalence {n_sli/len(enni):.1%})")
    y = enni["y"].to_numpy()

    models = {
        "logreg (lang+age)":  lambda: make_pipeline(StandardScaler(),
                                LogisticRegression(class_weight="balanced", max_iter=1000)),
        "grad-boosting":      lambda: HistGradientBoostingClassifier(random_state=0),
    }

    print("\n=== Child-independent CV (StratifiedGroupKFold by child, 5 folds), out-of-fold ===")
    X_full = enni[FEATURES].to_numpy()
    best_name, best_auc, best_p = None, -1, None
    for name, fn in models.items():
        p = oof_proba(fn, X_full, y, groups)
        auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
        auc_ci = boot_ci(y, p, groups, roc_auc_score)
        ap_ci = boot_ci(y, p, groups, average_precision_score)
        print(f"  {name:20s} ROC-AUC = {auc:.3f} [{auc_ci[0]:.3f}..{auc_ci[1]:.3f}]   "
              f"PR-AUC = {ap:.3f} [{ap_ci[0]:.3f}..{ap_ci[1]:.3f}]")
        if auc > best_auc:
            best_name, best_auc, best_p = name, auc, p

    # ---- ARTIFACT CHECK (ISS-04): can age or transcript length ALONE classify? ----
    print("\n=== Artifact check - confounds ALONE should be near chance (0.5) ===")
    for label, cols in [("age_months only", ["age_months"]),
                        ("n_utts (length) only", ["n_utts"]),
                        ("language only (no age)", LANG_FEATURES)]:
        Xc = enni[cols].to_numpy()
        p = oof_proba(models["logreg (lang+age)"], Xc, y, groups)
        print(f"  {label:24s} ROC-AUC = {roc_auc_score(y, p):.3f}")

    # ---- operating point: screening wants HIGH SENSITIVITY ----
    print(f"\n=== Operating points (best model: {best_name}) ===")
    fpr, tpr, thr = roc_curve(y, best_p)
    # threshold nearest to 0.90 sensitivity
    target = 0.90
    j = int(np.argmin(np.abs(tpr - target)))
    for label, t in [("Youden-optimal", thr[int(np.argmax(tpr - fpr))]),
                     (f"sensitivity~{target:.0%} (screening)", thr[j])]:
        sens, spec, (tp, fp, tn, fn) = sens_spec_at(y, best_p, t)
        print(f"  {label:28s} thr={t:.2f}  sens={sens:.2f} spec={spec:.2f}  "
              f"(TP{tp} FP{fp} TN{tn} FN{fn})")

    # ---- figures: ROC + score distribution ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(fpr, tpr, lw=2, label=f"{best_name} (AUC {best_auc:.2f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(xlabel="false positive rate", ylabel="true positive rate (sensitivity)",
              title="Stage 3 delay classifier - ROC (child-independent)")
    ax[0].legend(loc="lower right")
    for lab, mask, c in [("TD", y == 0, "tab:blue"), ("SLI", y == 1, "tab:orange")]:
        ax[1].hist(best_p[mask], bins=np.linspace(0, 1, 21), alpha=0.6, density=True, label=lab, color=c)
    ax[1].set(xlabel="predicted P(language impairment)", ylabel="density",
              title="Out-of-fold risk scores, TD vs SLI")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "stage3_roc.png"), dpi=150)
    print(f"\nfigure: results/stage3_roc.png")

    # ---- interpretability: standardized logistic coefficients (which markers drive the flag) ----
    scaler = StandardScaler().fit(X_full)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(X_full), y)
    print("\n=== Drivers of the delay flag (standardized logistic coefficients; + => more SLI) ===")
    for f, c in sorted(zip(FEATURES, lr.coef_[0]), key=lambda kv: -abs(kv[1])):
        print(f"  {f:12s} {c:+.2f}")

if __name__ == "__main__":
    main()
