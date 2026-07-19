#!/usr/bin/env python3
# FirstWords - Stage 5: the DEMO / TOOL. Paste (or point to) a child's CHAT transcript
# and get the full screening readout: estimated language age, age-gap, percentile
# ("growth chart" position), delay-flag probability, and the top language markers that
# are below age norm - with a plain-language, non-diagnostic recommendation.
#
# This is the judge-facing artifact. It wires together the validated pieces:
#   Stage 2 level estimator + Stage 4 age-normed readout + growth-chart percentiles.
# It trains the models once from results/features.csv (regenerating it if absent).
#
# USAGE:
#   .venv/bin/python src/stage5_demo.py --file path/to/child.cha
#   .venv/bin/python src/stage5_demo.py --demo          # runs built-in example children
#   .venv/bin/python src/stage5_demo.py                 # reads a CHAT transcript from stdin
import os, sys, argparse, tempfile
import numpy as np
import pandas as pd
import pylangacq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results")
import stage2_level_estimator as s2
import stage4_interpretability as s4
import ext_percentiles as pct
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DELAY_FEATURES = ["mluw", "mlum", "mattr50", "age_months"]

def _load_features():
    csv = os.path.join(RESULTS, "features.csv")
    if not os.path.exists(csv):
        s2.build_features().to_csv(csv, index=False)
    return pd.read_csv(csv)

class Screener:
    """Bundles the trained models + norms behind a single .screen(features) call."""
    def __init__(self):
        df = _load_features()
        td = df[(df["group"] == "TD") & df["age_months"].notna()]
        self.norms = s4.age_norms(td)
        self.curves = pct.fit_percentile_curves(td)
        lvl = td.dropna(subset=s4.LEVEL_FEATURES)
        self.level_model = s4._make_ridge().fit(lvl[s4.LEVEL_FEATURES], lvl["age_months"])
        enni = df[(df["corpus"] == "ENNI") & df["age_months"].notna()].dropna(subset=DELAY_FEATURES).copy()
        enni["y"] = (enni["group"] == "SLI").astype(int)
        self.delay_model = make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
        ).fit(enni[DELAY_FEATURES], enni["y"])

    def screen(self, row):
        text, info = s4.readout(row, self.level_model, self.delay_model, self.norms)
        p = pct.percentile_for(row["mluw"], row["age_months"], self.curves)
        return text, info, p

def features_from_chat(path):
    """Extract the model's features from a single CHAT (.cha) file with %mor tier."""
    r = pylangacq.read_chat(os.path.dirname(path) or ".", strict=False)
    # read_chat on a dir reads all files; filter to the requested one
    idx = [i for i, fp in enumerate(r.file_paths) if os.path.abspath(fp) == os.path.abspath(path)]
    if not idx:
        raise SystemExit(f"could not parse {path}")
    i = idx[0]
    age = s2.months(r.ages()[i])
    utts = r.utterances(by_file=True)[i]
    chi = [u for u in utts if u.participant == "CHI"]
    words = [t.word.lower() for u in chi for t in u.tokens if t.pos and t.word]
    return pd.Series({"age_months": age, "mluw": r.mluw()[i], "mlum": r.mlum()[i],
                      "ttr": r.ttr()[i], "mattr50": s2.mattr(words),
                      "syntax_index": r.ipsyn()[i], "n_utts": len(chi)})

BANNER = "=" * 64
def show(title, row, screener):
    text, info, p = screener.screen(row)
    print(f"\n{BANNER}\n  FirstWords screening readout - {title}\n{BANNER}")
    print(text)
    print(f"Sentence-length percentile for age : P{p:.0f}  "
          f"({'below the 10th percentile - notable' if p < 10 else 'within typical range'})")
    print("(Screening aid only - not a diagnosis. A flag means: consider a professional evaluation.)")

def run_examples(screener):
    df = _load_features()
    enni = df[(df["corpus"] == "ENNI") & df["age_months"].notna()].dropna(subset=DELAY_FEATURES)
    td = enni[enni["group"] == "TD"].sort_values("mlum", ascending=False).iloc[0]
    sli = enni[enni["group"] == "SLI"].sort_values("mlum").iloc[0]
    show("Example A (a typically-developing child)", td, screener)
    show("Example B (a child from the language-impaired group)", sli, screener)

def main():
    ap = argparse.ArgumentParser(description="FirstWords language screening demo")
    ap.add_argument("--file", help="path to a CHAT .cha transcript (needs the mor tier)")
    ap.add_argument("--demo", action="store_true", help="run built-in example children")
    args = ap.parse_args()
    screener = Screener()
    if args.demo or (not args.file and sys.stdin.isatty()):
        run_examples(screener); return
    if args.file:
        row = features_from_chat(args.file)
        show(os.path.basename(args.file), row, screener); return
    # read a CHAT transcript from stdin
    data = sys.stdin.read()
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "input.cha")
        open(fp, "w", encoding="utf-8").write(data)
        row = features_from_chat(fp)
        show("pasted transcript", row, screener)

if __name__ == "__main__":
    main()
