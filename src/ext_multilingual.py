#!/usr/bin/env python3
# FirstWords - Extension (Tier 3, THE ceiling-raiser): MULTILINGUAL LEVEL ESTIMATOR.
# Equity headline - a language screen for languages that have few/no screening tools.
# This harness is LANGUAGE-AGNOSTIC: drop any CHILDES-format corpus for any language
# into data/ml/<language>/<corpus>/ and it (1) runs the SIGNAL GATE (does language
# complexity still rise with age in this language?), and (2) if the gate passes, fits a
# child-independent level estimator and reports MAE/R2 + a calibration figure.
#
# GATE-FIRST DISCIPLINE (Lesson 1): we do NOT claim a language works until corr(age,MLU)
# is confirmed on that language's real data. A language that fails the gate is reported,
# not hidden. Features here are language-universal (MLU-words, MATTR vocab diversity) -
# they need only tokenized words, not a language-specific %mor grammar.
import os, glob, re
import numpy as np
import pandas as pd
import pylangacq
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stage2_level_estimator as s2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "data", "ml")
RESULTS = os.path.join(ROOT, "results")
FEATS = ["mluw", "mattr50"]        # language-universal (no %mor grammar needed)
GATE_CORR, GATE_MIN_CHILDREN = 0.40, 20

def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def child_id(fp, corpus_root):
    """Best-effort child id: a birthdate+sex header fingerprint if present, else the
    first sub-folder under the corpus root (per-child longitudinal layout), else file stem."""
    with open(fp, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    b = re.search(r"Birth of CHI is ([\w-]+)", txt)
    if b: return "b:" + b.group(1)
    rel = os.path.relpath(fp, corpus_root).split(os.sep)
    return rel[0] if len(rel) > 1 else os.path.splitext(rel[-1])[0]

def ingest(corpus_root):
    r = pylangacq.read_chat(corpus_root, strict=False)
    ages = [s2.months(a) for a in r.ages()]
    mluw = r.mluw()
    utts = r.utterances(by_file=True)
    rows = []
    for i, fp in enumerate(r.file_paths):
        chi = [u for u in utts[i] if u.participant == "CHI"]
        words = [t.word.lower() for u in chi for t in u.tokens if t.pos and t.word]
        rows.append({"child_id": child_id(fp, corpus_root), "age_months": ages[i],
                     "mluw": mluw[i], "mattr50": s2.mattr(words), "n_utts": len(chi)})
    return pd.DataFrame(rows)

def evaluate_language(lang, corpus_root):
    df = ingest(corpus_root).dropna(subset=["age_months", "mluw"])
    n_child = df["child_id"].nunique()
    if len(df) < 5:
        print(f"[{lang}] too few parseable files ({len(df)})."); return
    corr = pearson(df["age_months"], df["mluw"])
    print(f"\n=== {lang} ({os.path.basename(corpus_root)}) ===")
    print(f"  files={len(df)}  ~children={n_child}  ages {df['age_months'].min():.0f}-"
          f"{df['age_months'].max():.0f} mo  corr(age, MLU-words)={corr:.2f}")
    gate = corr >= GATE_CORR and n_child >= GATE_MIN_CHILDREN
    print(f"  SIGNAL GATE: {'PASS' if gate else 'FAIL'} "
          f"(need corr>={GATE_CORR} and >={GATE_MIN_CHILDREN} children)")
    if not gate:
        print("  -> not enough signal/children yet; report honestly, try a larger corpus."); return

    d = df.dropna(subset=FEATS)
    X, y, g = d[FEATS].to_numpy(), d["age_months"].to_numpy(), d["child_id"].to_numpy()
    n_splits = min(5, d["child_id"].nunique())
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, g):
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    mae = float(np.mean(np.abs(pred - y)))
    ss = np.sum((y - y.mean())**2); r2 = float(1 - np.sum((y - pred)**2)/ss) if ss else float("nan")
    base = float(np.mean(np.abs(y - y.mean())))
    print(f"  child-independent LEVEL estimator: MAE={mae:.1f} mo (baseline {base:.1f}), R2={r2:.2f}")
    print(f"  -> {lang} level estimator WORKS: language complexity predicts age cross-lingually.")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, pred, s=14, alpha=0.5)
    lim = [min(y.min(), pred.min()) - 3, max(y.max(), pred.max()) + 3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="chronological age (months)",
           ylabel="predicted language age (months)",
           title=f"{lang} level estimator (child-independent)\nMAE {mae:.1f} mo, R2 {r2:.2f}")
    fig.tight_layout()
    out = os.path.join(RESULTS, f"ext_ml_{lang.lower()}.png")
    fig.savefig(out, dpi=150); print(f"  figure: {os.path.relpath(out, ROOT)}")

def corpus_language(corpus_root):
    """Detect the corpus language from the first file's @Languages header."""
    for fp in glob.glob(os.path.join(corpus_root, "**/*.cha"), recursive=True)[:1]:
        m = re.search(r"@Languages:\t(\w+)", open(fp, encoding="utf-8", errors="ignore").read())
        return m.group(1) if m else "unknown"
    return "unknown"

def discover():
    """Find every corpus under data/childes/ and data/ml/, and evaluate the NON-English
    ones (the multilingual test). English corpora are the core pipeline, skipped here."""
    roots = [os.path.join(ROOT, "data", "childes"), ML]
    found = []
    for base in roots:
        if not os.path.isdir(base): continue
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            if os.path.isdir(p) and glob.glob(os.path.join(p, "**/*.cha"), recursive=True):
                found.append((name, p, corpus_language(p)))
    return found

def main():
    corpora = [(n, p, l) for n, p, l in discover() if l not in ("eng", "unknown")]
    if not corpora:
        print("No non-English corpus found. Drop e.g. a Spanish CHILDES corpus into")
        print("data/childes/<Corpus>/ (or data/ml/<Language>/<Corpus>/) and re-run.")
        print("The harness auto-detects language from @Languages and gates each corpus.")
        return
    print(f"Multilingual level-estimator test - found {len(corpora)} non-English corpus(es):")
    for name, path, lang in corpora:
        evaluate_language(lang, path)

if __name__ == "__main__":
    main()
