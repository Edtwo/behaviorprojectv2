#!/usr/bin/env python3
# FirstWords - Extension (Tier 1): NORM-REFERENCED PERCENTILES = the "growth chart
# for language" made literal. Quantile regression fits P10/25/50/75/90 curves of a
# language marker vs age on typically-developing children; a child's sample lands on
# a percentile, exactly like height-for-age on a pediatric growth chart.
#
# Deliverables: (1) a growth-chart figure with percentile bands + example children;
# (2) percentile_for() reusable by the Stage 5 demo -> "your child's sentence length
# is at the 6th percentile for age."
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
GROWTH_FEATURE = "mluw"   # sentence length (words/utt): the most interpretable marker
PRETTY = {"mluw": "sentence length (words per utterance)"}

def _design(age):
    """Quadratic in age: language growth decelerates, so age^2 captures the curve."""
    age = np.asarray(age, float)
    return np.column_stack([age, age**2])

def fit_percentile_curves(td, feature=GROWTH_FEATURE):
    g = td.dropna(subset=[feature, "age_months"])
    X, yv = _design(g["age_months"]), g[feature].to_numpy()
    return {q: QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(X, yv)
            for q in QUANTILES}

def percentile_for(value, age, curves):
    """Interpolate a child's percentile for their age from the fitted quantile curves.
    Returns a float in [0,100]; clamps to '<10' / '>90' outside the fitted bands."""
    preds = {q: float(m.predict(_design([age]))[0]) for q, m in curves.items()}
    qs = sorted(preds)                      # ascending quantiles
    vals = [preds[q] for q in qs]
    if value <= vals[0]:
        return max(1.0, 100 * qs[0] * value / vals[0]) if vals[0] > 0 else 1.0
    if value >= vals[-1]:
        return 100 * qs[-1]
    for i in range(len(qs) - 1):            # linear interp between bracketing quantiles
        if vals[i] <= value <= vals[i + 1]:
            frac = (value - vals[i]) / (vals[i + 1] - vals[i]) if vals[i + 1] > vals[i] else 0
            return 100 * (qs[i] + frac * (qs[i + 1] - qs[i]))
    return 50.0

def main():
    df = pd.read_csv(os.path.join(RESULTS, "features.csv"))
    td = df[(df["group"] == "TD") & df["age_months"].notna()]
    curves = fit_percentile_curves(td)

    # ---- growth-chart figure ----
    grid = np.linspace(td["age_months"].min(), td["age_months"].max(), 100)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(td["age_months"], td[GROWTH_FEATURE], s=8, alpha=0.25, color="0.6",
               label="typically-developing children", zorder=1)
    styles = {0.10: ":", 0.25: "--", 0.50: "-", 0.75: "--", 0.90: ":"}
    for q, m in curves.items():
        ax.plot(grid, m.predict(_design(grid)), styles[q], color="tab:blue", lw=2,
                zorder=2, label=f"P{int(q*100)}")
    # overlay example children (one clear SLI, one TD) if present
    sli = df[(df["group"] == "SLI") & df["age_months"].notna()].dropna(subset=[GROWTH_FEATURE])
    if len(sli):
        ex = sli.sort_values(GROWTH_FEATURE).iloc[0]
        pct = percentile_for(ex[GROWTH_FEATURE], ex["age_months"], curves)
        ax.scatter([ex["age_months"]], [ex[GROWTH_FEATURE]], s=140, color="tab:red",
                   marker="*", zorder=4,
                   label=f"example flagged child (P{pct:.0f} for age)")
    ax.set(xlabel="chronological age (months)",
           ylabel=PRETTY[GROWTH_FEATURE],
           title="Language growth chart: sentence length percentiles by age (TD norms)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "ext_growth_chart.png"), dpi=150)
    print("figure: results/ext_growth_chart.png")

    # ---- example percentile readouts ----
    print("\nExample percentile-for-age (sentence length):")
    for label, sub in [("clear TD child", td.dropna(subset=[GROWTH_FEATURE]).sort_values(GROWTH_FEATURE, ascending=False)),
                       ("SLI-group child", sli.sort_values(GROWTH_FEATURE))]:
        if len(sub):
            r = sub.iloc[0]
            pct = percentile_for(r[GROWTH_FEATURE], r["age_months"], curves)
            print(f"  {label:16s} age {r['age_months']:.0f} mo, {GROWTH_FEATURE}={r[GROWTH_FEATURE]:.2f} "
                  f"-> P{pct:.0f} for age")

    # ---- validity check: TD percentiles should be ~uniform (well-calibrated bands) ----
    tv = td.dropna(subset=[GROWTH_FEATURE])
    pcts = [percentile_for(v, a, curves) for v, a in zip(tv[GROWTH_FEATURE], tv["age_months"])]
    below10 = np.mean(np.array(pcts) < 10) * 100
    print(f"\nCalibration: {below10:.1f}% of TD children fall below P10 (target ~10%).")

if __name__ == "__main__":
    main()
