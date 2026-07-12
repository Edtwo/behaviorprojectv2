#!/usr/bin/env python3
# ALS-Forecast - STAGE 1 SIGNAL-GATE. RUN THIS BEFORE BUILDING ANYTHING.
# Run with the project venv:  .venv/bin/python stage1_signalgate.py
#
# THE QUESTION (make-or-break): do EARLY (<= ~3 months) clinical features predict a
# patient's FUTURE ALSFRS-R decline slope BETTER THAN a predict-the-mean baseline,
# evaluated PATIENT-INDEPENDENTLY (GroupKFold by patient - no identity leakage)?
#
# Expected to PASS: the DREAM ALS Prediction Prize (Kuffner et al., Nat Biotechnol 2015)
# already showed early-data -> progression prediction beats baselines and clinicians.
# We verify it on our own copy of PRO-ACT anyway. If it somehow FAILS, STOP and
# reconsider the target/features BEFORE writing any more code (the DopaLoop lesson).
#
# This gate uses ALSFRS longitudinal data ONLY (enough to test the core signal);
# richer features (FVC, labs, demographics, onset) get added in later stages.

import sys, glob, os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR   = "data/proact"
EARLY_DAYS = 91     # first ~3 months = "early window" (features come from here)
MIN_EARLY  = 2      # need >= 2 early points to estimate an early slope
MIN_LATE   = 2      # need >= 2 later points to estimate the FUTURE slope (target)
N_SPLITS   = 5

# ---- CONFIG: leave as None to auto-detect; override AFTER running explore_proact.py ----
ALSFRS_FILE = None   # e.g. "PROACT_ALSFRS.csv"
SUBJ_COL    = None   # e.g. "SubjectID"
TIME_COL    = None   # e.g. "ALSFRS_Delta"  (days from baseline)
SCORE_COL   = None   # e.g. "ALSFRS_R_Total" (prefer the -R total; else ALSFRS_Total)


def find_alsfrs_file():
    if ALSFRS_FILE:
        p = os.path.join(DATA_DIR, ALSFRS_FILE)
        if not os.path.isfile(p):
            sys.exit(f"Configured ALSFRS_FILE not found: {p}")
        return p
    files = glob.glob(os.path.join(DATA_DIR, "*.csv")) + glob.glob(os.path.join(DATA_DIR, "*.CSV"))
    if not files:
        sys.exit(f"No CSVs in {DATA_DIR}/. Download PRO-ACT and unzip there first.")
    cand = [f for f in files if "alsfrs" in os.path.basename(f).lower()]
    if not cand:
        sys.exit("No ALSFRS file auto-detected. Run explore_proact.py and set ALSFRS_FILE at top.")
    return sorted(cand)[0]


def detect_cols(df):
    cols = list(df.columns)
    subj = SUBJ_COL or next((c for c in cols if "subject" in c.lower()), None) \
                    or next((c for c in cols if c.lower().endswith("id")), None)
    time = TIME_COL or next((c for c in cols if "delta" in c.lower()), None) \
                    or next((c for c in cols if "day" in c.lower() or "time" in c.lower()), None)
    score = SCORE_COL
    if not score:
        score = next((c for c in cols if "alsfrs" in c.lower() and "r" in c.lower() and "total" in c.lower()), None)
        score = score or next((c for c in cols if "alsfrs" in c.lower() and "total" in c.lower()), None)
    return subj, time, score


def slope(ts, ys):
    ts = np.asarray(ts, float); ys = np.asarray(ys, float)
    if len(ts) < 2 or np.ptp(ts) == 0:
        return None
    return np.polyfit(ts, ys, 1)[0]   # units: score-points per day (negative = decline)


def main():
    path = find_alsfrs_file()
    df = pd.read_csv(path, low_memory=False)
    subj, time, score = detect_cols(df)
    print(f"ALSFRS file: {os.path.basename(path)}   shape={df.shape}")
    print(f"detected -> subject:'{subj}'  time:'{time}'  score:'{score}'")
    if not all([subj, time, score]):
        sys.exit("Could not detect required columns. Run explore_proact.py and set CONFIG at top.\n"
                 "(If the file has ALSFRS ITEMS but no precomputed total, we'll sum items in a follow-up.)")

    d = df[[subj, time, score]].copy()
    d.columns = ["subj", "t", "y"]
    d["t"] = pd.to_numeric(d["t"], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d = d.dropna()
    print(f"usable rows: {len(d)}   patients: {d.subj.nunique()}")
    print(f"score range: {d.y.min():.0f}..{d.y.max():.0f}   time range (days): {d.t.min():.0f}..{d.t.max():.0f}")

    # Build per-patient early features + FUTURE-slope target.
    rows = []
    for s, g in d.groupby("subj"):
        g = g.sort_values("t")
        early = g[g.t <= EARLY_DAYS]
        late  = g[g.t >  EARLY_DAYS]
        if len(early) < MIN_EARLY or len(late) < MIN_LATE:
            continue
        fut = slope(late.t.values, late.y.values)       # TARGET: future decline slope
        if fut is None:
            continue
        es = slope(early.t.values, early.y.values)       # early slope (a feature)
        rows.append(dict(
            subj=s,
            base=float(early.y.iloc[0]),                 # first early score
            last_early=float(early.y.iloc[-1]),          # last early score
            e_slope=float(es) if es is not None else 0.0,
            n_early=len(early),
            early_span=float(np.ptp(early.t.values)),
            future_slope=float(fut),
        ))
    F = pd.DataFrame(rows)
    print(f"\ncohort with early+late data: {len(F)} patients")
    if len(F) < 50:
        print("[!] small cohort - interpret with caution (report CIs).")
    if len(F) < N_SPLITS or F.empty:
        sys.exit("Too few patients with both early and late data to run the gate. "
                 "Inspect the data / adjust EARLY_DAYS.")

    F["future_slope_mo"] = F.future_slope * 30.0
    print(f"future decline slope (pts/month): mean={F.future_slope_mo.mean():.2f} "
          f"sd={F.future_slope_mo.std():.2f}  (a real spread = fast vs slow progressors to predict)")

    # ---- THE GATE: patient-independent prediction vs predict-mean baseline ----
    Xcols = ["base", "last_early", "e_slope", "n_early", "early_span"]
    X = F[Xcols].values
    y = F.future_slope.values
    groups = F.subj.values
    gkf = GroupKFold(n_splits=min(N_SPLITS, F.subj.nunique()))
    mae_model, mae_base = [], []
    preds = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups):
        m = GradientBoostingRegressor(random_state=0).fit(X[tr], y[tr])
        p = m.predict(X[te]); preds[te] = p
        mae_model.append(mean_absolute_error(y[te], p))
        mae_base.append(mean_absolute_error(y[te], np.full(len(te), y[tr].mean())))

    mm, mb = float(np.mean(mae_model)), float(np.mean(mae_base))
    r2 = r2_score(y, preds)
    improve = 100 * (mb - mm) / mb if mb > 0 else 0.0

    print("\n" + "=" * 70)
    print(" SIGNAL-GATE RESULT (patient-independent, GroupKFold)")
    print("=" * 70)
    print(f" model    MAE: {mm:.5f} pts/day  ({mm*30:.3f} pts/month)")
    print(f" baseline MAE: {mb:.5f} pts/day  ({mb*30:.3f} pts/month)   [predict-the-mean]")
    print(f" out-of-fold R2: {r2:.3f}")
    print(f" model beats baseline by {improve:.1f}% MAE")
    passed = (mm < mb) and (r2 > 0)
    print("\n GATE: " + ("PASS - early data predicts future ALS decline. Safe to build Stage 2+."
                          if passed else
                          "REVIEW - does NOT beat baseline. STOP; reconsider target/features BEFORE building."))
    print("=" * 70)


if __name__ == "__main__":
    main()
