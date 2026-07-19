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
from sklearn.model_selection import GroupKFold
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
        # within-corpus reference: child-independent CV MAE on the TEST corpus, same
        # band. MUST group by child_id - Brown has only 3 children, so a by-file split
        # would leak the same child across folds and understate the reference error.
        teb = in_band(test); refp = np.full(len(teb), np.nan)
        n_groups = teb["child_id"].nunique()
        for tr, va in GroupKFold(min(5, n_groups)).split(teb, groups=teb["child_id"]):
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

LANG_ONLY = ["mluw", "mlum", "mattr50"]        # age-free model (for out-of-age-range corpora)

def label_from_path(fp, root):
    """Read TD/SLI group and task from the relative folder path (e.g. Conti4 encodes
    both in folder names: TD-frog, SLI-spontaneous). Falls back to the @ID header."""
    rel = os.path.relpath(fp, root).upper()
    grp = "SLI" if re.search(r"\bSLI\b|DLD", rel) else ("TD" if re.search(r"\bTD\b|CONTROL|TYPICAL", rel) else "")
    task = next((t for t in ("FROG", "SPONTANEOUS", "NARRATIVE", "CONVERSATION") if t in rel), "all")
    if not grp:                                 # fall back to header @Types / @ID
        with open(fp, encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        if re.search(r"@Types:.*\bSLI\b", txt): grp = "SLI"
        elif re.search(r"@Types:.*\bTD\b", txt): grp = "TD"
    return grp, task.lower()

def extract_corpus(path):
    """Per-file features + group/task (from folder path) for an arbitrary corpus dir."""
    r = pylangacq.read_chat(path, strict=False)
    ages = [s2.months(a) for a in r.ages()]
    mluw, mlum, ttr, ipsyn = r.mluw(), r.mlum(), r.ttr(), r.ipsyn()
    utts = r.utterances(by_file=True)
    rows = []
    for i, fp in enumerate(r.file_paths):
        chi = [u for u in utts[i] if u.participant == "CHI"]
        words = [t.word.lower() for u in chi for t in u.tokens if t.pos and t.word]
        grp, task = label_from_path(fp, path)
        rows.append({"file": os.path.basename(fp), "task": task, "group": grp,
                     "child_id": os.path.splitext(os.path.basename(fp))[0],
                     "age_months": ages[i], "mluw": mluw[i], "mlum": mlum[i], "ttr": ttr[i],
                     "mattr50": s2.mattr(words), "syntax_index": ipsyn[i], "n_utts": len(chi)})
    return pd.DataFrame(rows)

def find_external_corpora():
    """Scan both data/childes/ and the data/ root for corpora other than Brown/ENNI."""
    roots, out = [DATA, os.path.dirname(DATA)], []
    seen = set()
    for base in roots:
        if not os.path.isdir(base): continue
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            if (os.path.isdir(p) and name not in KNOWN | {"childes"} and p not in seen
                    and glob.glob(os.path.join(p, "**/*.cha"), recursive=True)):
                out.append((name, p)); seen.add(p)
    return out

def _auc_ci(y, p, n=2000, seed=0):
    """Bootstrap 95% CI for AUC, resampling the external test children."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2: continue
        vals.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def delay_transfer():
    print("\nPART B - DELAY cross-corpus transfer (train ENNI, TEST external corpus; no refit)")
    ext = find_external_corpora()
    if not ext:
        print("  No external corpus found. Unzip a 2nd English clinical corpus (TD+SLI/DLD)")
        print("  into data/ or data/childes/. Labels are read from folder names or @Types. Re-run.")
        return
    df = pd.read_csv(os.path.join(RESULTS, "features.csv"))
    enni = df[(df["corpus"] == "ENNI") & df["age_months"].notna()].dropna(subset=DELAY_FEATURES).copy()
    enni["y"] = (enni["group"] == "SLI").astype(int)
    full = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(
        enni[DELAY_FEATURES], enni["y"])
    lang = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(
        enni[LANG_ONLY], enni["y"])
    enni_lo, enni_hi = enni["age_months"].min(), enni["age_months"].max()

    for name, p in ext:
        d = extract_corpus(p).dropna(subset=DELAY_FEATURES)
        d = d[d["group"].isin(["TD", "SLI"])]
        if d.empty:
            print(f"  {name}: no TD/SLI-labeled files detected - skipping."); continue
        a_lo, a_hi = d["age_months"].min(), d["age_months"].max()
        oor = a_lo > enni_hi or a_hi < enni_lo
        print(f"  {name}: {len(d)} files, ages {a_lo:.0f}-{a_hi:.0f} mo "
              f"(ENNI train range {enni_lo:.0f}-{enni_hi:.0f}{' -> OUT OF RANGE, use language-only' if oor else ''})")
        # report per TASK so each file is a distinct child within the comparison
        for task, g in d.groupby("task"):
            counts = dict(g["group"].value_counts())
            if {"TD", "SLI"} <= set(counts):
                y = (g["group"] == "SLI").astype(int).to_numpy()
                p_lang = lang.predict_proba(g[LANG_ONLY])[:, 1]
                auc_full = roc_auc_score(y, full.predict_proba(g[DELAY_FEATURES])[:, 1])
                auc_lang = roc_auc_score(y, p_lang)
                lo, hi = _auc_ci(y, p_lang)
                print(f"     task={task:12s} {counts}  ROC-AUC: language-only={auc_lang:.3f} "
                      f"[95% CI {lo:.3f}-{hi:.3f}]  full(+age)={auc_full:.3f}")
            else:
                print(f"     task={task:12s} {counts}  (need both TD & SLI - skipped)")
    print("  NOTE: Conti4 is adolescents (~13-16 yr) vs ENNI 4-10 yr -> language-only is the fair")
    print("  read (age is extrapolated in the full model). This is a HARD cross-age + cross-lab test.")

def main():
    level_transfer()
    delay_transfer()

if __name__ == "__main__":
    main()
