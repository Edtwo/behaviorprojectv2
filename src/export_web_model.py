#!/usr/bin/env python3
# FirstWords - export the words-only screen for the web app (web/model.json).
#
# The web app runs entirely in the browser, so it cannot run pylangacq. Instead it uses a
# small CHAT parser (cha_utterances below; web/app.js implements the SAME rules). To keep
# the web app honest, this script:
#   1. re-extracts sentence length (mluw) and vocabulary diversity (mattr50) from every
#      training transcript with that parser,
#   2. checks those values against the pylangacq features the paper results use,
#   3. trains the words-only models (the same recipe as audio_screen.ReducedScreener) on the
#      parser's own features, so web inference and training use identical feature code,
#   4. reports child-independent AUC and picks the screening cutoff (~90% sensitivity),
#   5. writes ONLY model parameters (no per-child data) to web/model.json.
#
# USAGE: .venv/bin/python src/export_web_model.py
import os, sys, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve
import stage2_level_estimator as s2
import stage3_delay_classifier as s3
import ext_percentiles as pct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "childes")
OUT = os.path.join(ROOT, "web", "model.json")
LEVEL = ["mluw", "mattr50"]
DELAY = ["mluw", "mattr50", "age_months"]
TARGET_SENS = 0.90

def cha_utterances(text, speaker="CHI"):
    """Word lists for one speaker's utterances in a CHAT transcript.
    Rules (mirrored exactly in web/app.js -> chaUtterances):
      - join continuation lines (start with a tab) onto the previous tier
      - drop retraced material: '<...> [/]', '<...> [//]', 'word [/]'
      - drop all other [...] codes, &-fillers/fragments, 0-omissions, xxx/yyy/www,
        pause markers, terminators, timing marks, and @-suffix codes (keep the word)
      - '(be)cause' -> 'because'; keep apostrophes; compounds 'ice+cream' stay one word"""
    lines, cur = [], None
    for raw in text.splitlines():
        if raw.startswith("\t") and cur is not None:
            cur += " " + raw.strip()
        else:
            if cur is not None:
                lines.append(cur)
            cur = raw
    if cur is not None:
        lines.append(cur)
    utts = []
    for ln in lines:
        if not ln.startswith(f"*{speaker}:"):
            continue
        s = ln.split(":", 1)[1]
        s = re.sub(r"\x15[^\x15]*\x15", " ", s)                 # timing bullets
        s = re.sub(r"<[^<>]*>\s*\[/{1,2}\]", " ", s)            # retraced group
        s = re.sub(r"\S+\s*\[/{1,2}\]", " ", s)                 # retraced word
        s = re.sub(r"\[[^\]]*\]", " ", s)                        # other codes
        s = s.replace("<", " ").replace(">", " ")
        words = []
        for tok in s.split():
            if tok.startswith(("&", "0", "+", "(.")) or tok in ("xxx", "yyy", "www"):
                continue
            tok = re.sub(r"@\S*$", "", tok)                      # word@s -> word
            tok = tok.replace("(", "").replace(")", "")
            tok = re.sub(r"[^A-Za-z'+_-]", "", tok).lower().strip("'-_")
            if re.search(r"[a-z]", tok) and tok not in ("xxx", "yyy", "www"):
                words.append(tok)
        if words:
            utts.append(words)
    return utts

def features(utts):
    lens = [len(u) for u in utts]
    words = [w for u in utts for w in u]
    if not lens:
        return None
    return {"mluw": float(np.mean(lens)), "mattr50": s2.mattr(words)}

def main():
    df = pd.read_csv(os.path.join(ROOT, "results", "features.csv"))
    rows = []
    for _, r in df.iterrows():
        with open(os.path.join(DATA, r["file"]), encoding="utf-8", errors="ignore") as f:
            ft = features(cha_utterances(f.read()))
        if ft:
            rows.append({**r.to_dict(), "w_mluw": ft["mluw"], "w_mattr50": ft["mattr50"]})
    d = pd.DataFrame(rows)

    print("Parser check vs pylangacq features (same transcripts):")
    for a, b in [("mluw", "w_mluw"), ("mattr50", "w_mattr50")]:
        ok = d[[a, b]].dropna()
        rel = (ok[b] - ok[a]).abs() / ok[a]
        print(f"  {a:8s} r = {ok[a].corr(ok[b]):.3f}, median |diff| = {rel.median():.1%}, "
              f"90th pct |diff| = {rel.quantile(.9):.1%}")

    d = d.drop(columns=["mluw", "mattr50"]).rename(columns={"w_mluw": "mluw", "w_mattr50": "mattr50"})
    td = d[(d.group == "TD") & d.age_months.notna()].dropna(subset=LEVEL)
    enni = d[(d.corpus == "ENNI") & d.age_months.notna()].dropna(subset=DELAY).copy()
    enni["y"] = (enni.group == "SLI").astype(int)
    y, g = enni["y"].to_numpy(), enni["child_id"].to_numpy()

    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    p = s3.oof_proba(lr, enni[DELAY].to_numpy(), y, g)
    auc = roc_auc_score(y, p); lo, hi = s3.boot_ci(y, p, g, roc_auc_score)
    fpr, tpr, thr = roc_curve(y, p)
    j = int(np.argmax(tpr >= TARGET_SENS))
    cut = float(thr[j])
    sens, spec, _ = s3.sens_spec_at(y, p, cut)
    print(f"\nWords-only delay screen, child-independent: AUC {auc:.3f} [{lo:.3f}-{hi:.3f}] "
          f"({len(enni)} transcripts, {enni.child_id.nunique()} children)")
    print(f"  screening cutoff {cut:.3f}: sensitivity {sens:.2f}, specificity {spec:.2f}")
    s45, sp45, _ = s3.sens_spec_at(y, p, 0.45)
    print(f"  (old demo cutoff 0.45: sensitivity {s45:.2f}, specificity {sp45:.2f})")

    level = make_pipeline(StandardScaler(), Ridge(1.0)).fit(td[LEVEL], td["age_months"])
    delay = lr().fit(enni[DELAY], enni["y"])
    curves = pct.fit_percentile_curves(td)
    norms = {f: np.polyfit(td["age_months"], td[f], 1).tolist() for f in LEVEL}
    resid = {f: float((td[f] - np.polyval(norms[f], td["age_months"])).std(ddof=1)) for f in LEVEL}

    def lin(pipe):
        sc, m = pipe[0], pipe[-1]
        return {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                "coef": np.ravel(m.coef_).tolist(), "intercept": float(np.ravel(m.intercept_)[0])}
    model = {
        "version": "web-words-only-1",
        "level_features": LEVEL, "delay_features": DELAY,
        "level": lin(level), "delay": lin(delay),
        "cutoff": round(cut, 4),
        "curves": {str(q): {"coef": m.coef_.tolist(), "intercept": float(m.intercept_)}
                   for q, m in curves.items()},
        "norms": norms, "resid_sd": resid,
        "flag_age_range_months": [48, 120],
        "trained_on": {"level_transcripts": int(len(td)), "level_children": int(td.child_id.nunique()),
                       "delay_transcripts": int(len(enni)), "delay_children": int(enni.child_id.nunique())},
        "validation": {"auc": round(auc, 3), "auc_ci": [round(lo, 3), round(hi, 3)],
                       "sensitivity": round(sens, 3), "specificity": round(spec, 3)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(model, f, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)} (parameters only, no per-child data)")

if __name__ == "__main__":
    main()
