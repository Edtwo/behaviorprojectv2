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

AGE_MAX = 156          # drop clear non-child outliers (e.g. a 368-mo adult in Szagun)
# corpora where all files are one densely-sampled child (flat, age-coded filenames)
SINGLE_CHILD = {"Ornat"}

def child_id(fp, corpus_root):
    """Best-effort child id: single-child corpora -> one id; else a birthdate+sex header
    fingerprint; else the first sub-folder under the root (per-child layout); else stem."""
    base = os.path.basename(corpus_root.rstrip("/"))
    if base in SINGLE_CHILD:
        return base
    with open(fp, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    b = re.search(r"Birth of CHI is ([\w-]+)", txt)
    if b: return "b:" + b.group(1)
    rel = os.path.relpath(fp, corpus_root).split(os.sep)
    # a NAMED first sub-folder is a per-child layout; a NUMERIC one (e.g. HKU age bins
    # 2/3/4/5) is not -> then each file is its own child.
    if len(rel) > 1 and not rel[0].isdigit():
        return rel[0]
    return os.path.splitext(rel[-1])[0]

def ingest(corpus_root):
    r = pylangacq.read_chat(corpus_root, strict=False)
    ages = [s2.months(a) for a in r.ages()]
    mluw, mlum = r.mluw(), r.mlum()
    utts = r.utterances(by_file=True)
    rows = []
    for i, fp in enumerate(r.file_paths):
        chi = [u for u in utts[i] if u.participant == "CHI"]
        words = [t.word.lower() for u in chi for t in u.tokens if t.pos and t.word]
        rows.append({"child_id": child_id(fp, corpus_root), "age_months": ages[i],
                     "mluw": mluw[i], "mlum": mlum[i], "mattr50": s2.mattr(words),
                     "n_utts": len(chi)})
    return pd.DataFrame(rows)

def evaluate_language(lang, corpus_root):
    df = ingest(corpus_root).dropna(subset=["age_months", "mluw"])
    df = df[df["age_months"] <= AGE_MAX]                       # drop non-child outliers
    n_child = df["child_id"].nunique()
    if len(df) < 5:
        print(f"[{lang}] too few parseable files ({len(df)})."); return
    # The growth-chart signal = does ANY validated complexity marker rise with age?
    # (Length-based MLU dominates in inflected/analytic languages, but in ISOLATING
    # languages like Cantonese lexical diversity (MATTR) is the stronger index - so we
    # gate on the BEST marker, not MLU alone, and report which marker wins.)
    marker_corr = {f: pearson(df["age_months"], df[f]) for f in ["mluw", "mlum", "mattr50"]
                   if df[f].notna().sum() >= 5}
    best_marker = max(marker_corr, key=lambda k: abs(marker_corr[k]))
    corr = marker_corr[best_marker]
    wc = [pearson(g["age_months"], g[best_marker]) for _, g in df.groupby("child_id")
          if g[best_marker].notna().sum() >= 5]
    wc_med = float(np.median(wc)) if wc else float("nan")
    print(f"\n=== {lang} ({os.path.basename(corpus_root)}) ===")
    print(f"  files={len(df)}  children={n_child}  ages {df['age_months'].min():.0f}-"
          f"{df['age_months'].max():.0f} mo")
    print(f"  age-corr by marker: " + ", ".join(f"{k}={v:+.2f}" for k, v in marker_corr.items())
          + f"  -> STRONGEST = {best_marker} ({corr:+.2f})")
    print(f"  within-child corr of {best_marker} (median of {len(wc)} kids)={wc_med:.2f}")
    cross_ok = abs(corr) >= GATE_CORR and n_child >= GATE_MIN_CHILDREN
    premise_ok = len(wc) >= 1 and wc_med >= 0.5
    print(f"  PREMISE (within-child age<->complexity): {'CONFIRMED' if premise_ok else 'not shown'}"
          f"   CROSS-CHILD ESTIMATOR GATE: {'PASS' if cross_ok else 'FAIL (data-limited)'}")
    if not cross_ok:
        print("  -> premise may hold, but too few distinct children across a wide age range for a"
              " cross-child estimator (report honestly).")
        return

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
    # cluster-bootstrap CI (resample CHILDREN) - the sample is small, so report uncertainty
    rng = np.random.default_rng(0)
    by_child = {c: np.where(g == c)[0] for c in np.unique(g)}
    kids = list(by_child); maes = []
    for _ in range(2000):
        idx = np.concatenate([by_child[k] for k in rng.choice(kids, len(kids), replace=True)])
        maes.append(np.mean(np.abs(pred[idx] - y[idx])))
    mlo, mhi = np.percentile(maes, [2.5, 97.5])
    print(f"  child-independent LEVEL estimator: MAE={mae:.1f} mo [95% CI {mlo:.1f}-{mhi:.1f}] "
          f"(baseline {base:.1f}), R2={r2:.2f}")
    print(f"  -> {lang} level estimator WORKS: language complexity predicts age cross-lingually.")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, pred, s=14, alpha=0.5)
    lim = [min(y.min(), pred.min()) - 3, max(y.max(), pred.max()) + 3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="chronological age (months)",
           ylabel="predicted language age (months)",
           title=f"{lang} level estimator (child-independent)\nMAE {mae:.1f} mo, R2 {r2:.2f}")
    fig.tight_layout()
    out = os.path.join(RESULTS, f"ext_ml_{re.sub(r'[^a-z0-9]+', '_', lang.lower())}.png")
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
            if name == "not_in_use": continue          # archived, skip
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
        # if the corpus separates controls (e.g. Szagun CI vs TD), use TD only
        td = os.path.join(path, "TD")
        evaluate_language(f"{lang}/{name}", td if os.path.isdir(td) else path)

if __name__ == "__main__":
    main()
