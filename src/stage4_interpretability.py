#!/usr/bin/env python3
# FirstWords - Stage 4: interpretability -> an ACTIONABLE, clinician-style readout.
# Two complementary views (both go on the poster):
#   (1) GLOBAL drivers via SHAP on the two winning linear models (level ridge,
#       delay logistic) - "which language markers move the model, and which way."
#   (2) PER-CHILD readout that a family/clinician can act on: the child's estimated
#       language age, the age-gap, the delay-flag probability, and each marker
#       expressed as a z-score vs AGE-MATCHED typically-developing peers
#       ("morpheme MLU is 2.1 SD below age norm"). SHAP says what the model used;
#       the age-normed z tells a human whether it is high or low FOR THAT AGE.
# This module also exposes readout()/age_norms() for the Stage 5 demo to import.
import os
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

LEVEL_FEATURES = ["mluw", "mlum", "ttr", "mattr50", "syntax_index"]   # Stage 2
DELAY_FEATURES = ["mluw", "mlum", "mattr50", "age_months"]            # Stage 3
PRETTY = {"mluw": "sentence length (words/utt)", "mlum": "grammatical morphemes/utt",
          "ttr": "vocab diversity (TTR)", "mattr50": "vocab diversity (MATTR-50)",
          "syntax_index": "syntax complexity index", "age_months": "chronological age"}

def load():
    df = pd.read_csv(os.path.join(RESULTS, "features.csv"))
    if len(df) == 0:
        raise SystemExit("results/features.csv missing - run stage2 first")
    return df

# ---------- age-referenced norms (from TD only) ----------
def age_norms(td):
    """For each language feature, fit expected value as a linear function of age on
    TD children, and record the residual SD. z = (observed - expected(age)) / sd.
    This is what makes the readout 'below age norm' rather than 'below average'."""
    norms = {}
    for f in ["mluw", "mlum", "ttr", "mattr50", "syntax_index"]:
        g = td.dropna(subset=[f, "age_months"])
        b, a = np.polyfit(g["age_months"], g[f], 1)      # expected(age) = a + b*age
        resid = g[f] - (a + b * g["age_months"])
        norms[f] = (a, b, float(resid.std(ddof=1)))
    return norms

def feature_z(row, norms):
    z = {}
    for f, (a, b, sd) in norms.items():
        if pd.notna(row.get(f)) and sd > 0:
            z[f] = (row[f] - (a + b * row["age_months"])) / sd
    return z

# ---------- readout used by the demo ----------
def readout(row, level_model, delay_model, norms):
    """Human-readable screening readout for one child (a feature Series/dict)."""
    lvl = float(level_model.predict(pd.DataFrame([row])[LEVEL_FEATURES])[0])
    gap = lvl - row["age_months"]
    p = float(delay_model.predict_proba(pd.DataFrame([row])[DELAY_FEATURES])[0, 1])
    z = feature_z(row, norms)
    elevated = p >= 0.45
    # rank markers that are LOW for age (most negative z) as the actionable drivers
    low = sorted(((f, zz) for f, zz in z.items() if zz < -0.5), key=lambda kv: kv[1])
    lines = [
        f"Chronological age : {row['age_months']:.0f} months",
        f"Estimated language age : {lvl:.0f} months  (gap {gap:+.0f} mo vs age)",
        f"Delay-flag risk score : {p:.2f} (0-1, recall-tuned)  ->  {'ELEVATED' if elevated else 'not elevated'}",
    ]
    if elevated and low:
        lines.append("Markers below age norm (driving the flag):")
        for f, zz in low[:3]:
            lines.append(f"   - {PRETTY.get(f, f)}: {zz:+.1f} SD for age")
        lines.append("Suggested action: discuss a speech-language evaluation with a professional.")
    elif elevated:
        lines.append("Flag elevated but no single marker stands out - interpret with care.")
    elif low:
        lines.append(f"Overall within age norms; one marker mildly low "
                     f"({PRETTY.get(low[0][0], low[0][0])}: {low[0][1]:+.1f} SD).")
    else:
        lines.append("All language markers within or above age norm.")
    return "\n".join(lines), {"lang_age": lvl, "gap": gap, "p_delay": p, "z": z}

def shap_bar(model, X, feature_names, title, path, scaled=False):
    """SHAP global importance for a linear model (exact Shapley values)."""
    bg = X if len(X) <= 100 else shap.sample(X, 100, random_state=0)
    explainer = shap.LinearExplainer(model, bg)
    sv = explainer.shap_values(X)
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh([PRETTY.get(feature_names[i], feature_names[i]) for i in order], imp[order],
            color="tab:blue")
    ax.set(xlabel="mean |SHAP value| (impact on model output)", title=title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return {feature_names[i]: float(imp[i]) for i in order[::-1]}

def main():
    df = load()
    td = df[(df["group"] == "TD") & df["age_months"].notna()]
    norms = age_norms(td)

    # ---- refit the two winning models on full data (interpretation, not scoring) ----
    lvl_df = td.dropna(subset=LEVEL_FEATURES)
    level_model = _make_ridge().fit(lvl_df[LEVEL_FEATURES], lvl_df["age_months"])

    enni = df[(df["corpus"] == "ENNI") & df["age_months"].notna()].dropna(subset=DELAY_FEATURES).copy()
    enni["y"] = (enni["group"] == "SLI").astype(int)
    delay_model = _make_logreg().fit(enni[DELAY_FEATURES], enni["y"])

    # ---- (1) GLOBAL SHAP importance figures ----
    Xl = StandardScaler().fit_transform(lvl_df[LEVEL_FEATURES])
    lin_level = Ridge(alpha=1.0).fit(Xl, lvl_df["age_months"])
    imp_l = shap_bar(lin_level, Xl, LEVEL_FEATURES,
                     "Stage 4 - drivers of LANGUAGE-LEVEL estimate (SHAP)",
                     os.path.join(RESULTS, "stage4_shap_level.png"))

    Xd = StandardScaler().fit_transform(enni[DELAY_FEATURES])
    lin_delay = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xd, enni["y"])
    imp_d = shap_bar(lin_delay, Xd, DELAY_FEATURES,
                     "Stage 4 - drivers of DELAY flag (SHAP)",
                     os.path.join(RESULTS, "stage4_shap_delay.png"))

    print("Global SHAP importance - LEVEL estimate:")
    for f, v in sorted(imp_l.items(), key=lambda kv: -kv[1]):
        print(f"   {PRETTY.get(f,f):32s} {v:.3f}")
    print("Global SHAP importance - DELAY flag:")
    for f, v in sorted(imp_d.items(), key=lambda kv: -kv[1]):
        print(f"   {PRETTY.get(f,f):32s} {v:.3f}")
    print("figures: results/stage4_shap_level.png, results/stage4_shap_delay.png")

    # ---- (2) EXAMPLE per-child readouts (one clear TD, one clear SLI) ----
    print("\n" + "=" * 60 + "\n EXAMPLE READOUTS (what the demo will show)\n" + "=" * 60)
    sli = enni[enni["y"] == 1].sort_values("mlum")
    tdc = enni[enni["y"] == 0].sort_values("mlum", ascending=False)
    for label, row in [("a typically-developing child", tdc.iloc[0]),
                       ("a child who screens POSITIVE (from the SLI group)", sli.iloc[0])]:
        txt, _ = readout(row, level_model, delay_model, norms)
        print(f"\n--- Example: {label} ---\n{txt}")

def _make_ridge():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

def _make_logreg():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))

if __name__ == "__main__":
    main()
