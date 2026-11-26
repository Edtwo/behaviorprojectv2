#!/usr/bin/env python3
# Checks a team member's own re-implementation of the delay classifier (see README.md here).
# USAGE: .venv/bin/python exercises/reimplement_classifier/check.py exercises/reimplement_classifier/<yourname>.py
import importlib.util, os, sys

REFERENCE, TOLERANCE = 0.861, 0.005

if len(sys.argv) != 2:
    raise SystemExit(__doc__ or "usage: check.py <your_file.py>")
path = sys.argv[1]
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(root)                                      # so results/features.csv resolves
spec = importlib.util.spec_from_file_location("attempt", path)
attempt = importlib.util.module_from_spec(spec); spec.loader.exec_module(attempt)
if not hasattr(attempt, "oof_auc"):
    raise SystemExit(f"{path} needs a function oof_auc() that returns the out-of-fold ROC-AUC")
auc = float(attempt.oof_auc())
print(f"your out-of-fold ROC-AUC: {auc:.3f}   (project: {REFERENCE:.3f})")
if abs(auc - REFERENCE) <= TOLERANCE:
    print("MATCH: you reproduced the project's headline result.")
else:
    print("Not matching yet. Check: ENNI rows only, the 4 features, balanced weights, "
          "StratifiedGroupKFold(5, shuffle=True, random_state=0) grouped by child_id.")
