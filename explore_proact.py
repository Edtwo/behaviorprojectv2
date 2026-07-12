#!/usr/bin/env python3
# ALS-Forecast - STEP 0: explore the PRO-ACT CSVs BEFORE writing any parsing.
# (The hard-won lesson: NEVER hardcode column names before inspecting the real files.)
# Run with the project venv:  .venv/bin/python explore_proact.py
#
# It finds every CSV in data/proact/, prints each file's shape, columns, dtypes,
# and a 3-row sample, and flags columns likely relevant to our task. Use its
# output to confirm the column names that stage1_signalgate.py auto-detects.

import sys, glob, os
import pandas as pd

DATA_DIR = "data/proact"
KEYS = ['subject', 'alsfrs', 'fvc', 'svc', 'delta', 'time', 'day', 'onset',
        'age', 'sex', 'vital', 'lab', 'death', 'survival', 'riluzole', 'diagnos']

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")) +
                   glob.glob(os.path.join(DATA_DIR, "*.CSV")))
    if not files:
        print(f"No CSVs found in {DATA_DIR}/ .")
        print("Register at PRO-ACT, download, and unzip the CSVs into that folder, then re-run.")
        sys.exit(1)

    print(f"Found {len(files)} CSV file(s) in {DATA_DIR}/\n")
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print(f"[!] {os.path.basename(f)}: could not read ({e})\n")
            continue
        print("=" * 74)
        print(f"FILE: {os.path.basename(f)}    shape = {df.shape[0]} rows x {df.shape[1]} cols")
        print("columns:", list(df.columns))
        rel = [c for c in df.columns if any(k in c.lower() for k in KEYS)]
        if rel:
            print("  >> likely-relevant columns:", rel)
        with pd.option_context('display.max_columns', 25, 'display.width', 220):
            print(df.head(3).to_string())
        print()
    print("=" * 74)
    print("NEXT: confirm the ALSFRS file + its subject-id / time(delta) / total-score")
    print("columns, then run:  .venv/bin/python stage1_signalgate.py")

if __name__ == "__main__":
    main()
