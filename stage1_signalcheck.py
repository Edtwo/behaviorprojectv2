#!/usr/bin/env python3
# BEAR v2 / ParkinTrack - Stage 1: data verification + WITHIN-PERSON signal check.
# Pure standard library (no pandas needed) so it runs anywhere.
# Purpose: BEFORE building anything, confirm (a) the data is as expected, and
# (b) the project HEADLINE - within-person change monitoring - actually has signal:
#   does motor_UPDRS vary WITHIN a patient over time, and do voice features move with it?
# Also runs an integrity check: is motor_UPDRS just a linear interpolation of time
# per subject? (a known quirk of this dataset that we must be honest about).

import csv, math, statistics as st
from collections import defaultdict

PATH = "data/parkinsons_updrs.data"

# ----- load -----
rows = []
with open(PATH, newline="") as f:
    r = csv.DictReader(f)
    cols = r.fieldnames
    for row in r:
        rows.append({k: float(v) for k, v in row.items()})

print("=" * 70)
print(" STAGE 1 - VERIFICATION")
print("=" * 70)
print(f"rows: {len(rows)}   cols: {len(cols)}")
print(f"columns: {cols}")

# subjects
by_subj = defaultdict(list)
for row in rows:
    by_subj[int(row["subject#"])].append(row)
subjects = sorted(by_subj)
counts = [len(by_subj[s]) for s in subjects]
print(f"subjects: {len(subjects)}  (expect 42)")
print(f"recordings/subject: min={min(counts)} max={max(counts)} mean={sum(counts)/len(counts):.1f}")

tt = [row["test_time"] for row in rows]
print(f"test_time range (days): {min(tt):.1f} .. {max(tt):.1f}")
ages = sorted({int(row['age']) for row in rows})
sexes = sorted({int(row['sex']) for row in rows})
print(f"age range: {min(ages)}..{max(ages)}   sex codes: {sexes}")

# missing check
missing = sum(1 for row in rows for v in row.values() if v is None or (isinstance(v, float) and math.isnan(v)))
print(f"missing/NaN values: {missing}")

# feature columns (exclude ids/targets)
non_feat = {"subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS"}
feats = [c for c in cols if c not in non_feat]

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs)/n, sum(ys)/n
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sxx = sum((x-mx)**2 for x in xs)
    syy = sum((y-my)**2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx*syy)

# ----- within-person variation in the target (is there anything to track?) -----
print("\n" + "=" * 70)
print(" HEADLINE CHECK A: does motor_UPDRS vary WITHIN a patient over time?")
print("=" * 70)
within_sd = []
within_range = []
for s in subjects:
    vals = [r["motor_UPDRS"] for r in by_subj[s]]
    within_sd.append(st.pstdev(vals))
    within_range.append(max(vals) - min(vals))
overall_sd = st.pstdev([r["motor_UPDRS"] for r in rows])
between_sd = st.pstdev([sum(v["motor_UPDRS"] for v in by_subj[s])/len(by_subj[s]) for s in subjects])
print(f"motor_UPDRS overall SD: {overall_sd:.2f}")
print(f"between-subject SD (of per-subject means): {between_sd:.2f}")
print(f"within-subject SD: mean={sum(within_sd)/len(within_sd):.2f}  min={min(within_sd):.2f} max={max(within_sd):.2f}")
print(f"within-subject range (max-min): mean={sum(within_range)/len(within_range):.2f} days-span")
print("-> if within-subject SD is tiny vs between-subject SD, there is little within-person")
print("   change to monitor and the headline must be reconsidered.")

# ----- integrity: is motor_UPDRS ~ a linear function of time per subject? -----
print("\n" + "=" * 70)
print(" INTEGRITY CHECK: is motor_UPDRS just a linear trend in time per subject?")
print(" (this dataset's UPDRS may be interpolated between a few clinical visits)")
print("=" * 70)
r_updrs_time = []
for s in subjects:
    sub = sorted(by_subj[s], key=lambda r: r["test_time"])
    rr = pearson([r["test_time"] for r in sub], [r["motor_UPDRS"] for r in sub])
    if rr is not None:
        r_updrs_time.append(abs(rr))
print(f"|corr(motor_UPDRS, test_time)| within subject: mean={sum(r_updrs_time)/len(r_updrs_time):.3f} "
      f"median={st.median(r_updrs_time):.3f} min={min(r_updrs_time):.3f} max={max(r_updrs_time):.3f}")
print("-> if ~1.0 for most subjects, UPDRS is essentially a smooth line in time")
print("   (interpolated); within-person 'change' is then a linear trend, not noisy")
print("   per-visit measurement. We must report this honestly.")

# show one example subject's trajectory
ex = subjects[0]
sub = sorted(by_subj[ex], key=lambda r: r["test_time"])
print(f"\nExample subject {ex}: first 12 (test_time, motor_UPDRS):")
for r in sub[:12]:
    print(f"   t={r['test_time']:8.3f}   motor_UPDRS={r['motor_UPDRS']:7.3f}")

# ----- headline check B: do voice features track within-person UPDRS change? -----
print("\n" + "=" * 70)
print(" HEADLINE CHECK B: within a patient, do voice features move with motor_UPDRS?")
print(" (mean signed r and mean |r| across subjects, per feature)")
print("=" * 70)
feat_within = {}
for fcol in feats:
    rs = []
    for s in subjects:
        sub = by_subj[s]
        rr = pearson([r[fcol] for r in sub], [r["motor_UPDRS"] for r in sub])
        if rr is not None:
            rs.append(rr)
    if rs:
        feat_within[fcol] = (sum(rs)/len(rs), sum(abs(x) for x in rs)/len(rs))
ranked = sorted(feat_within.items(), key=lambda kv: kv[1][1], reverse=True)
print(f"{'feature':16s} {'mean r':>9s} {'mean|r|':>9s}")
for fcol, (mr, mabs) in ranked:
    print(f"{fcol:16s} {mr:>9.3f} {mabs:>9.3f}")

# ----- cross-person reference (pooled) -----
print("\n" + "=" * 70)
print(" CROSS-PERSON reference: pooled corr(feature, motor_UPDRS) across all rows")
print("=" * 70)
allu = [r["motor_UPDRS"] for r in rows]
cross = {fcol: pearson([r[fcol] for r in rows], allu) for fcol in feats}
for fcol, rr in sorted(cross.items(), key=lambda kv: abs(kv[1]) if kv[1] else 0, reverse=True):
    print(f"{fcol:16s} pooled r = {rr:.3f}")

print("\n" + "=" * 70)
print(" READ THIS: the headline (personalized within-person monitoring) needs")
print(" (A) real within-person UPDRS variation AND (B) voice features that track it.")
print(" The integrity check tells us whether that 'tracking' is really just following")
print(" an interpolated linear trend - which we must state honestly either way.")
print("=" * 70)
