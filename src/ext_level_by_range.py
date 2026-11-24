#!/usr/bin/env python3
# FirstWords - how good is the language-age estimate inside each age range?
# The pooled R2 (0.81) spans 1.5-10 years across corpora; this reports it within ranges and
# checks whether the age gap (estimated minus real age) is biased by age.
# TAKEAWAY: the measures track age strongly across childhood (r = 0.90), but an individual
# "language age" at 4-10 is only good to about +-13 months and the gap regresses to the mean.
# -> on the board, lead with the age-normed GROWTH-CHART PERCENTILE, not a language-age number.
# USAGE: .venv/bin/python src/ext_level_by_range.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import stage2_level_estimator as s2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "results", "features.csv"))
td = df[(df.group == "TD") & df.age_months.notna()].dropna(subset=s2.FEATURES).copy()
td["pred"] = s2.oof_predict(lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                            td[s2.FEATURES].to_numpy(), td.age_months.to_numpy(), td.child_id.to_numpy())
print("Language-age estimate, typically developing children, child-independent")
for lab, d in [("all ages (1.5-10 yr)", td), ("ages 4-10 (ENNI)", td[td.corpus == "ENNI"]), ("under 4 yr", td[td.age_months < 48])]:
    err = d.pred - d.age_months
    r2 = 1 - np.sum(err**2) / np.sum((d.age_months - d.age_months.mean())**2)
    print(f"  {lab:22s} n={len(d):3d}  typical error {np.mean(np.abs(err)):4.1f} mo  R2 {r2:5.2f}  "
          f"r {np.corrcoef(d.pred, d.age_months)[0, 1]:.2f}")
e = td[td.corpus == "ENNI"]
print(f"  age bias of the gap at 4-10: slope {np.polyfit(e.age_months, e.pred - e.age_months, 1)[0]:+.2f} "
      f"months per month of age (0 = unbiased)")
