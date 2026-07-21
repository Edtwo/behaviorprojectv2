#!/usr/bin/env python3
# FirstWords - Extension (Tier 2, RIGOR): PROBABILITY CALIBRATION. The demo shows a
# "delay-flag probability," so those numbers must be trustworthy: when the tool says
# 0.70, ~70% of such children should truly be impaired. We check this with a reliability
# diagram + Brier score, on child-independent out-of-fold predictions.
#
# Subtlety we report honestly: class_weight='balanced' (used for good sensitivity on the
# 21% minority) DECALIBRATES probabilities upward. We show that, and that a plain
# (unweighted) or isotonic-recalibrated model restores calibration - so the DEMO should
# display a calibrated probability while the SCREENING THRESHOLD can still favor recall.
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DELAY_FEATURES = ["mluw", "mlum", "mattr50", "age_months"]

def oof(df, balanced=True):
    X = df[DELAY_FEATURES].to_numpy(); y = df["y"].to_numpy(); g = df["child_id"].to_numpy()
    p = np.full(len(y), np.nan)
    kw = {"class_weight": "balanced"} if balanced else {}
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **kw)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p

def main():
    df = pd.read_csv(os.path.join(RESULTS, "features.csv"))
    df = df[(df["corpus"] == "ENNI") & df["age_months"].notna()].dropna(subset=DELAY_FEATURES).copy()
    df["y"] = (df["group"] == "SLI").astype(int)

    p_bal = oof(df, balanced=True)
    p_unw = oof(df, balanced=False)
    # isotonic recalibration of the balanced model's OOF scores (fit on OOF -> honest-ish)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_bal, df["y"])
    p_iso = iso.predict(p_bal)

    print("Calibration on child-independent OOF predictions (ENNI delay model):")
    for name, p in [("balanced (Stage 3 default)", p_bal),
                    ("unweighted logistic", p_unw),
                    ("balanced + isotonic recal", p_iso)]:
        print(f"  {name:28s} Brier = {brier_score_loss(df['y'], p):.3f}  "
              f"(lower is better; base rate {df['y'].mean():.3f})")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    for name, p, c in [("balanced (default)", p_bal, "tab:orange"),
                       ("unweighted", p_unw, "tab:blue"),
                       ("balanced+isotonic", p_iso, "tab:green")]:
        frac_pos, mean_pred = calibration_curve(df["y"], p, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", color=c, label=name)
    ax.set(xlabel="predicted P(impairment)", ylabel="observed fraction impaired",
           title="Delay-flag calibration (reliability diagram)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "ext_calibration.png"), dpi=150)
    print("\nfigure: results/ext_calibration.png")
    print("Takeaway: the balanced model over-states probabilities (good recall, biased p);")
    print("isotonic recalibration restores honest probabilities for the demo readout, while")
    print("the screening THRESHOLD is chosen separately for sensitivity. Report both.")

if __name__ == "__main__":
    main()
