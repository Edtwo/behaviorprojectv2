#!/usr/bin/env python3
# FirstWords - Extension (interview prep): what happens in a REAL classroom at each cutoff?
#
# Judge question this answers: "At 7% prevalence, aren't most of your flags false alarms?"
# Yes at the 91%-sensitivity cutoff (PPV ~16%). This table shows the whole tradeoff so the
# team can explain the choice instead of defending one number.
#
# Method: child-independent out-of-fold risk scores from Stage 3 (same model, folds and seed).
# Sensitivity/specificity come from held-out ENNI children; PPV, NPV and "% of class flagged"
# are then re-weighted to a real-world prevalence (default 7%, Tomblin et al. 1997) with
# Bayes' rule, since ENNI's 21% SLI share is far above real life.
# Per 100 children at 7%: 7 truly affected, 93 not.
# LIMITATION: ENNI-only operating points; the cutoff does not transfer to other corpora
# without recalibration (ISS-18, ISS-31).
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import stage3_delay_classifier as s3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREVALENCE = 0.07


def main():
    enni = s3.load_enni()
    y = enni["y"].to_numpy()
    p = s3.oof_proba(lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
                     enni[s3.FEATURES].to_numpy(), y, enni["child_id"].to_numpy())
    rows = []
    for target in [0.95, 0.91, 0.85, 0.80, 0.70, 0.60, 0.50]:
        thr = np.sort(p[y == 1])[::-1][int(np.ceil(target * y.sum())) - 1]   # lowest cutoff keeping >= target sens
        sens = float(np.mean(p[y == 1] >= thr)); spec = float(np.mean(p[y == 0] < thr))
        tp, fp = sens * PREVALENCE, (1 - spec) * (1 - PREVALENCE)
        fn, tn = (1 - sens) * PREVALENCE, spec * (1 - PREVALENCE)
        rows.append({"cutoff": thr, "sensitivity": sens, "specificity": spec,
                     "PPV": tp / (tp + fp), "NPV": tn / (tn + fn), "pct_flagged": tp + fp,
                     "per100_caught": 100 * tp, "per100_missed": 100 * fn, "per100_false_alarms": 100 * fp})
    df = pd.DataFrame(rows).drop_duplicates("cutoff")
    df.to_csv(os.path.join(ROOT, "results", "operating_points.csv"), index=False)
    print(f"Operating points (held-out ENNI children; real-world prevalence {PREVALENCE:.0%})")
    print(f"Per 100 children: {100 * PREVALENCE:.0f} truly affected, {100 * (1 - PREVALENCE):.0f} not.\n")
    print(f"{'cutoff':>6} {'sens':>5} {'spec':>5} {'PPV':>5} {'NPV':>6} {'flagged':>8} "
          f"{'caught/7':>9} {'missed/7':>9} {'false alarms/93':>16}")
    for _, r in df.iterrows():
        print(f"{r.cutoff:6.2f} {r.sensitivity:5.2f} {r.specificity:5.2f} {r.PPV:5.0%} {r.NPV:6.1%} "
              f"{r.pct_flagged:8.0%} {r.per100_caught:9.1f} {r.per100_missed:9.1f} {r.per100_false_alarms:16.1f}")
    print("\nRead: every cutoff trades missed children for false alarms. A first-line screen keeps")
    print("misses low (a 'pass' is ~99% trustworthy); the false alarms are sent for a proper")
    print("evaluation, which is the normal design of two-step screening.")


if __name__ == "__main__":
    main()
