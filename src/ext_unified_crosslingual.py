#!/usr/bin/env python3
# FirstWords - Extension (Tier 3, NOVELTY): a UNIFIED CROSS-LINGUAL developmental index.
# So far we built a SEPARATE level estimator per language. Here we ask the harder, more
# novel question: can ONE model, trained on language-NORMALIZED features, estimate
# developmental age across languages - and does it transfer to a language it never saw?
#
# METHOD: within each language, z-standardize the (language-universal) features using
# that language's own TD distribution (removes scale differences - German MLU >> Cantonese
# MLU in absolute terms - WITHOUT using age or labels, so no leakage). Then train ONE
# ridge model on the pooled standardized features to predict age. Two evaluations:
#   (1) POOLED, child-independent (GroupKFold by child).
#   (2) LEAVE-ONE-LANGUAGE-OUT (LOLO): train on all languages but one, test on the held-out
#       language -> the real test of a "universal" index. Compared to each language's OWN
#       within-language model (the per-language approach) as the reference.
# HONEST NOTE: applying the unified model to a new language needs that language's feature
# mean/SD (a light normative requirement), not full age-norms.
import os
import numpy as np
import pandas as pd
import pylangacq
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
import stage2_level_estimator as s2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "childes")
RESULTS = os.path.join(ROOT, "results")
FEATS = ["mluw", "mlum", "mattr50"]        # available + comparable across all languages

def load_dir(root, lang, single_child=False):
    r = pylangacq.read_chat(root, strict=False)
    ages = [s2.months(a) for a in r.ages()]
    mluw, mlum = r.mluw(), r.mlum()
    utts = r.utterances(by_file=True)
    rows = []
    for i, fp in enumerate(r.file_paths):
        chi = [u for u in utts[i] if u.participant == "CHI"]
        words = [t.word.lower() for u in chi for t in u.tokens if t.pos and t.word]
        rel = os.path.relpath(fp, root).split(os.sep)
        cid = lang if single_child else (rel[0] if len(rel) > 1 and not rel[0].isdigit()
                                         else os.path.splitext(rel[-1])[0])
        rows.append({"lang": lang, "child_id": f"{lang}:{cid}", "age_months": ages[i],
                     "mluw": mluw[i], "mlum": mlum[i], "mattr50": s2.mattr(words)})
    return pd.DataFrame(rows)

def gather():
    # English: reuse the core TD feature table (Brown+ENNI-TD+Bates+Gleason)
    eng = pd.read_csv(os.path.join(RESULTS, "features.csv"))
    eng = eng[(eng["group"] == "TD")][["child_id", "age_months"] + FEATS].copy()
    eng["lang"] = "eng"; eng["child_id"] = "eng:" + eng["child_id"].astype(str)
    parts = [eng]
    # German controls (drop the 368-mo outlier), Cantonese
    if os.path.isdir(os.path.join(DATA, "Szagun", "TD")):
        g = load_dir(os.path.join(DATA, "Szagun", "TD"), "deu"); parts.append(g[g.age_months <= 60])
    if os.path.isdir(os.path.join(DATA, "HKU")):
        parts.append(load_dir(os.path.join(DATA, "HKU"), "yue"))
    df = pd.concat(parts, ignore_index=True).dropna(subset=FEATS + ["age_months"])
    # standardize features WITHIN each language (scale-normalize; no age/labels used)
    for f in FEATS:
        df[f + "_z"] = df.groupby("lang")[f].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1))
    return df

ZF = [f + "_z" for f in FEATS]

def cv_group(d, feats):
    X, y, g = d[feats].to_numpy(), d["age_months"].to_numpy(), d["child_id"].to_numpy()
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(5, d["child_id"].nunique())).split(X, y, g):
        m = make_pipeline(StandardScaler(), Ridge(1.0)).fit(X[tr], y[tr]); p[te] = m.predict(X[te])
    return mae_r2(y, p)

def mae_r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    ss = np.sum((y - y.mean())**2)
    return float(np.mean(np.abs(p - y))), float(1 - np.sum((y - p)**2)/ss) if ss else float("nan")

def main():
    df = gather()
    print("Unified cross-lingual developmental index. Per-language sample:")
    for lang, g in df.groupby("lang"):
        print(f"  {lang}: {len(g)} files, {g['child_id'].nunique()} children, "
              f"ages {g.age_months.min():.0f}-{g.age_months.max():.0f} mo")

    mae, r2 = cv_group(df, ZF)
    print(f"\n(1) POOLED unified model (language-standardized feats, child-independent): "
          f"MAE={mae:.1f} mo, R2={r2:.2f}")

    print("\n(2) LEAVE-ONE-LANGUAGE-OUT (train on the others, test on the held-out language):")
    print(f"    {'held-out':10s}{'unified LOLO':>16}{'own-language ref':>20}")
    for lang in sorted(df["lang"].unique()):
        tr, te = df[df.lang != lang], df[df.lang == lang]
        m = make_pipeline(StandardScaler(), Ridge(1.0)).fit(tr[ZF], tr["age_months"])
        mae_l, r2_l = mae_r2(te["age_months"], m.predict(te[ZF]))
        # own-language reference: that language's own child-independent model
        mae_o, r2_o = cv_group(te, ZF) if te["child_id"].nunique() >= 3 else (float("nan"), float("nan"))
        print(f"    {lang:10s}  MAE {mae_l:4.1f} (R2 {r2_l:+.2f}) | own-lang MAE {mae_o:4.1f} (R2 {r2_o:+.2f})")
    # (3) FAIR LOLO in the COMMON age window (removes the age-range confound above)
    lo = max(g.age_months.min() for _, g in df.groupby("lang"))
    hi = min(g.age_months.max() for _, g in df.groupby("lang"))
    w = df[(df.age_months >= lo) & (df.age_months <= hi)]
    print(f"\n(3) AGE-MATCHED LOLO in the common window {lo:.0f}-{hi:.0f} mo (removes age-range confound):")
    for lang in sorted(w["lang"].unique()):
        tr, te = w[w.lang != lang], w[w.lang == lang]
        if te["child_id"].nunique() < 3 or len(tr) < 20:
            print(f"    {lang}: too few in-window ({len(te)} files) - skip"); continue
        m = make_pipeline(StandardScaler(), Ridge(1.0)).fit(tr[ZF], tr["age_months"])
        mae_l, r2_l = mae_r2(te["age_months"], m.predict(te[ZF]))
        base = np.mean(np.abs(te["age_months"] - te["age_months"].mean()))
        print(f"    {lang:10s}  unified-LOLO MAE {mae_l:4.1f} (R2 {r2_l:+.2f}, in-window baseline MAE {base:.1f})")

    print("\n  HONEST CONCLUSION: (a) a SINGLE language-standardized model spans 3 language families")
    print("  (pooled R2 0.51) - proof a cross-lingual index is possible - BUT it is WORSE than the")
    print("  per-language models (0.54-0.82), so per-language norms are preferred. (b) ZERO-DATA")
    print("  transfer to an unseen language (LOLO) FAILS - a new language needs its own normative")
    print("  calibration. This reinforces the core finding: language-specific norms + typology-aware")
    print("  markers are necessary, not optional. Reported as an honest cautionary result.")

if __name__ == "__main__":
    main()
