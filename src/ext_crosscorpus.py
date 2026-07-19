#!/usr/bin/env python3
# FirstWords - Extension (Tier 1, THE ceiling-raiser): CROSS-CORPUS GENERALIZATION.
# The strongest possible evidence a screening tool actually works is that it holds on
# data collected by DIFFERENT researchers. Two parts:
#
#  PART A (runs now): LEVEL-layer transfer. Train the age estimator on ONE corpus,
#    test on the OTHER (Brown home-conversation <-> ENNI narrative), in their
#    overlapping age band. Separates a systematic task offset (correctable) from a
#    loss of precision (the real generalization question).
#
#  PART B (needs a download): DELAY-layer transfer. Train the TD-vs-SLI classifier on
#    ENNI, TEST on a SECOND clinical corpus (e.g. Conti-Ramsden, Gillam). Labels are
#    read from the CHAT @ID header (field 6 = TD/SLI), so ANY corpus dropped into
#    data/childes/ is picked up automatically. If none is present, prints exactly what
#    to download.
import os, re, glob
import numpy as np
import pandas as pd
import pylangacq
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import stage2_level_estimator as s2   # reuse months(), mattr(), feature defs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "childes")
RESULTS = os.path.join(ROOT, "results")
KNOWN = {"Brown", "ENNI"}                      # corpora already used in the core
LEVEL_FEATURES = ["mluw", "mlum", "ttr", "mattr50", "syntax_index"]
DELAY_FEATURES = ["mluw", "mlum", "mattr50", "age_months"]

def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))

# ============================ PART A: level transfer ============================
def level_transfer():
    df = pd.read_csv(os.path.join(RESULTS, "features.csv"))
    df = df[df["group"] == "TD"].dropna(subset=LEVEL_FEATURES + ["age_months"])
    brown = df[df["corpus"] == "Brown"]; enni = df[df["corpus"] == "ENNI"]
    lo = max(brown["age_months"].min(), enni["age_months"].min())
    hi = min(brown["age_months"].max(), enni["age_months"].max())
    print(f"PART A - LEVEL cross-corpus transfer (overlap age band {lo:.0f}-{hi:.0f} mo)")
    print(f"  Brown n={len(brown)} (18-62 mo, home conversation); ENNI-TD n={len(enni)} (48-120 mo, narrative)")

    def in_band(d): return d[(d["age_months"] >= lo) & (d["age_months"] <= hi)]
    for train, test, trname, tename in [(brown, enni, "Brown", "ENNI-TD"),
                                        (enni, brown, "ENNI-TD", "Brown")]:
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
            train[LEVEL_FEATURES], train["age_months"])
        te = in_band(test)
        pred = m.predict(te[LEVEL_FEATURES])
        resid = pred - te["age_months"].to_numpy()
        bias = float(np.median(resid))              # systematic task offset
        raw_mae = mae(te["age_months"], pred)
        adj_mae = mae(te["age_months"], pred - bias)  # offset-corrected
        # within-corpus reference: 5-fold CV MAE on the TEST corpus, same band
        teb = in_band(test); refp = np.full(len(teb), np.nan)
        for tr, va in KFold(5, shuffle=True, random_state=0).split(teb):
            mm = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
                teb.iloc[tr][LEVEL_FEATURES], teb.iloc[tr]["age_months"])
            refp[va] = mm.predict(teb.iloc[va][LEVEL_FEATURES])
        print(f"  train {trname:8s} -> test {tename:8s} (n={len(te)}): "
              f"raw MAE {raw_mae:4.1f} mo | task-offset {bias:+5.1f} mo | "
              f"offset-corrected MAE {adj_mae:4.1f} mo | within-corpus ref MAE {mae(teb['age_months'], refp):4.1f}")
    print("  -> interpretation: a large offset with a SMALL offset-corrected MAE means the model "
          "transfers in RANK/precision but tasks have different absolute norms (expected & honest).")

# ============================ PART B: delay transfer ============================
def header_group_sex(fp):
    with open(fp, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    m = re.search(r"\|CHI\|[^|]*\|(male|female|unknown|)\|([A-Za-z]*)\|", txt)
    sex = m.group(1) if m else ""
    grp = (m.group(2) if m else "").upper()
    birth = re.search(r"Birth of CHI is ([\w-]+)", txt)
    cid = f"{birth.group(1)}:{sex}" if birth else os.path.basename(fp)
    return grp, sex, cid

def extract_corpus(path):
    """Generic per-file features + header-derived group for an arbitrary corpus dir."""
    r = pylangacq.read_chat(path, strict=False)
    ages = [s2.months(a) for a in r.ages()]
    mluw, mlum, ttr, ipsyn = r.mluw(), r.mlum(), r.ttr(), r.ipsyn()
    utts = r.utterances(by_file=True)
    rows = []
    for i, fp in enumerate(r.file_paths):
        chi = [u for u in utts[i] if u.participant == "CHI"]
        words = [t.word.lower() for u in chi for t in u.tokens if t.pos and t.word]
        grp, sex, cid = header_group_sex(fp)
        rows.append({"file": os.path.basename(fp), "group": grp, "sex": sex, "child_id": cid,
                     "age_months": ages[i], "mluw": mluw[i], "mlum": mlum[i], "ttr": ttr[i],
                     "mattr50": s2.mattr(words), "syntax_index": ipsyn[i], "n_utts": len(chi)})
    return pd.DataFrame(rows)

def find_external_corpora():
    out = []
    if not os.path.isdir(DATA): return out
    for name in sorted(os.listdir(DATA)):
        p = os.path.join(DATA, name)
        if os.path.isdir(p) and name not in KNOWN and glob.glob(os.path.join(p, "**/*.cha"), recursive=True):
            out.append((name, p))
    return out

def delay_transfer():
    print("\nPART B - DELAY cross-corpus transfer (train ENNI, test an external clinical corpus)")
    ext = find_external_corpora()
    if not ext:
        print("  No external corpus found. TO RUN THIS TEST, download a 2nd English clinical")
        print("  corpus with TD + SLI/DLD children from TalkBank (Clinical-Eng), e.g.:")
        print("    - Conti-Ramsden (Manchester)   - Gillam   - EllisWeismer")
        print("  Unzip into data/childes/<CorpusName>/ (any layout). Labels are read from the")
        print("  CHAT @ID header (TD/SLI), so no renaming is needed. Then re-run this script.")
        return
    # train delay model on ENNI (from features.csv)
    df = pd.read_csv(os.path.join(RESULTS, "features.csv"))
    enni = df[(df["corpus"] == "ENNI") & df["age_months"].notna()].dropna(subset=DELAY_FEATURES).copy()
    enni["y"] = (enni["group"] == "SLI").astype(int)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=1000)).fit(
        enni[DELAY_FEATURES], enni["y"])
    for name, p in ext:
        d = extract_corpus(p).dropna(subset=DELAY_FEATURES)
        d = d[d["group"].isin(["TD", "SLI"])]
        counts = dict(d["group"].value_counts())
        print(f"  {name}: {len(d)} labeled files {counts}")
        if {"TD", "SLI"} <= set(counts):
            y = (d["group"] == "SLI").astype(int)
            proba = model.predict_proba(d[DELAY_FEATURES])[:, 1]
            print(f"    -> ENNI-trained model on {name}: ROC-AUC = {roc_auc_score(y, proba):.3f} "
                  f"(external validation; no refit)")
        else:
            print(f"    -> need BOTH TD and SLI labels to score AUC; found only {set(counts)}")

def main():
    level_transfer()
    delay_transfer()

if __name__ == "__main__":
    main()
