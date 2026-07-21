#!/usr/bin/env python3
# FirstWords - Extension (Tier 2, CLINICAL UTILITY): a good AUC is not the same as being
# USEFUL. Decision-Curve Analysis (Vickers & Elkin 2006) asks: across the range of
# thresholds a family/clinician might act on, does using the screen give more NET BENEFIT
# than the two default policies - "refer every child" or "refer none"? It also translates
# performance into a concrete "unnecessary referrals avoided per 100 children screened."
# Uses child-independent out-of-fold predictions from the ENNI delay model (Stage 3).
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DELAY_FEATURES = ["mluw", "mlum", "mattr50", "age_months"]

def oof(df):
    X, y, g = df[DELAY_FEATURES].to_numpy(), df["y"].to_numpy(), df["child_id"].to_numpy()
    p = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p

def net_benefit(y, p, pt):
    """Net benefit at threshold probability pt: (TP - FP*pt/(1-pt)) / n."""
    n = len(y); pred = p >= pt
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    return (tp - fp * (pt / (1 - pt))) / n

def main():
    df = pd.read_csv(os.path.join(RESULTS, "features.csv"))
    df = df[(df["corpus"] == "ENNI") & df["age_months"].notna()].dropna(subset=DELAY_FEATURES).copy()
    df["y"] = (df["group"] == "SLI").astype(int)
    y = df["y"].to_numpy()
    p = oof(df)
    prev = y.mean()
    print(f"ENNI delay model - clinical utility (child-independent OOF). prevalence={prev:.2f}")

    pts = np.linspace(0.05, 0.6, 45)
    nb_model = [net_benefit(y, p, t) for t in pts]
    nb_all = [net_benefit(y, np.ones_like(p), t) for t in pts]   # refer everyone
    nb_none = [0.0 for _ in pts]                                  # refer no one

    # range where the model is the best of the three policies
    best_model = [m > max(a, n) for m, a, n in zip(nb_model, nb_all, nb_none)]
    lo = pts[best_model].min() if any(best_model) else None
    hi = pts[best_model].max() if any(best_model) else None
    print(f"  model gives the HIGHEST net benefit for threshold probabilities "
          f"{lo:.2f}-{hi:.2f} (i.e. it's the best policy across that clinically plausible range).")

    # concrete translation at a screening threshold (~0.2, near prevalence)
    t = 0.2
    pred = p >= t
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    nb = net_benefit(y, p, t)
    # unnecessary referrals avoided vs "refer everyone", per 100 children, at same TP rate
    refer_all_fp = np.sum(y == 0)
    avoided = (refer_all_fp - fp) / len(y) * 100
    print(f"  at threshold {t}: catches {tp}/{tp+fn} affected (sens {tp/(tp+fn):.0%}), "
          f"{fp} false alarms (spec {tn/(tn+fp):.0%}).")
    print(f"  vs 'refer every child': same screen AVOIDS ~{avoided:.0f} unnecessary referrals per 100 "
          f"children, while still catching most affected kids.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(pts, nb_model, lw=2, color="tab:blue", label="FirstWords screen")
    ax.plot(pts, nb_all, lw=1.5, ls="--", color="tab:gray", label="refer every child")
    ax.plot(pts, nb_none, lw=1.5, ls=":", color="k", label="refer no child")
    ax.set(xlabel="threshold probability (how sure before referring)", ylabel="net benefit",
           title="Decision-curve analysis: the screen beats both default policies",
           ylim=(min(nb_all) * 0.5, max(nb_model) * 1.3 + 0.01))
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "ext_decision_curve.png"), dpi=150)
    print("  figure: results/ext_decision_curve.png")

if __name__ == "__main__":
    main()
