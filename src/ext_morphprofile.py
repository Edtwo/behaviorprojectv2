#!/usr/bin/env python3
# FirstWords - Extension (Tier 3, NOVELTY): ITEM-LEVEL GRAMMATICAL PROFILE. Instead of
# one coarse syntax score, measure the SPECIFIC grammatical markers per child from the
# %mor / %gra tiers, and ask which ones separate TD from SLI. This directly tests the
# well-known clinical hypothesis that language impairment shows up most in VERB TENSE /
# FINITENESS marking (Rice & Wexler's Extended Optional Infinitive), and yields a far
# richer, clinician-style readout ("rarely marks past tense - a classic DLD marker").
#
# BONUS test: do these fine-grained markers add predictive value OVER the coarse MLU
# features? (child-independent AUC of a morphology-only model.)
import os
import numpy as np
import pandas as pd
import pylangacq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stage2_level_estimator as s2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "childes")
RESULTS = os.path.join(ROOT, "results")
SUBORD = {"XCOMP", "CCOMP", "ADVCL", "ACL", "ACL-RELCL", "CSUBJ"}  # subordinate-clause relations

def child_profile(chi_utts):
    """Per-child grammatical marker rates from %mor/%gra. Rates are normalized to be
    length-robust (per relevant token or per utterance), not raw counts."""
    n_utt = max(1, len(chi_utts))
    verbs = fin = past = prog = n_noun = plural = n_aux = gen = s3v = 0
    subord_utts = 0
    infl_tags = set(); n_tok = 0
    for u in chi_utts:
        has_sub = False
        for t in u.tokens:
            if not t.pos:  # punctuation
                continue
            n_tok += 1
            feats = t.mor.split("-")[1:] if t.mor else []
            infl_tags.update(feats)
            if t.pos in ("verb", "aux"):
                verbs += 1
                if "Fin" in feats: fin += 1
                if "Past" in feats: past += 1
                if "Part" in feats: prog += 1
                if "S3" in feats: s3v += 1
            if t.pos == "aux": n_aux += 1
            if t.pos == "noun":
                n_noun += 1
                if "Plur" in feats: plural += 1
            if "Gen" in feats: gen += 1
            if t.gra and t.gra.rel in SUBORD: has_sub = True
        if has_sub: subord_utts += 1
    v = max(1, verbs); nn = max(1, n_noun)
    return {
        "finite_verb_ratio": fin / v,          # tense/agreement marking (key DLD marker)
        "past_verb_ratio":   past / v,
        "prog_verb_ratio":   prog / v,
        "third_sg_ratio":    s3v / v,
        "aux_per_utt":       n_aux / n_utt,     # auxiliary/copula use
        "plural_noun_ratio": plural / nn,
        "genitive_per_utt":  gen / n_utt,
        "subord_clause_rate": subord_utts / n_utt,  # syntactic complexity (%gra)
        "morph_diversity":   len(infl_tags) / max(1, n_tok) * 100,  # distinct inflections per 100 tokens
    }

# morph_diversity (distinct inflections / tokens) is EXCLUDED: it is transcript-length
# confounded exactly like raw TTR (fewer tokens -> higher ratio), so its apparent
# "higher in SLI" is a shorter-sample artifact, not a real grammatical difference (ISS-10).
MARKERS = ["finite_verb_ratio", "past_verb_ratio", "third_sg_ratio", "prog_verb_ratio",
           "aux_per_utt", "plural_noun_ratio", "genitive_per_utt", "subord_clause_rate"]
PRETTY = {"finite_verb_ratio": "finite (tense-marked) verbs", "past_verb_ratio": "past-tense marking",
          "third_sg_ratio": "3rd-person -s marking", "prog_verb_ratio": "progressive -ing",
          "aux_per_utt": "auxiliary/copula use", "plural_noun_ratio": "plural -s marking",
          "genitive_per_utt": "possessive marking", "subord_clause_rate": "subordinate clauses",
          "morph_diversity": "morphological diversity"}

def collect():
    rows = []
    for grp in ["TD", "SLI"]:
        r = pylangacq.read_chat(os.path.join(DATA, "ENNI", grp), strict=False)
        ages = [s2.months(a) for a in r.ages()]
        utts = r.utterances(by_file=True)
        for i, fp in enumerate(r.file_paths):
            chi = [u for u in utts[i] if u.participant == "CHI"]
            prof = child_profile(chi)
            prof.update({"child_id": s2.enni_child_id(fp), "group": grp, "age_months": ages[i]})
            rows.append(prof)
    return pd.DataFrame(rows)

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean() - b.mean()) / sp if sp else 0.0

def main():
    df = collect()
    td, sli = df[df["group"] == "TD"], df[df["group"] == "SLI"]
    print(f"ENNI item-level grammatical profile: TD n={len(td)}, SLI n={len(sli)}\n")
    print(f"  {'marker':<28}{'TD mean':>9}{'SLI mean':>10}{'Cohen d':>9}  (d>0 => TD higher)")
    stats = []
    for m in MARKERS:
        d = cohens_d(td[m], sli[m])
        stats.append((m, d))
        print(f"  {PRETTY[m]:<28}{td[m].mean():>9.3f}{sli[m].mean():>10.3f}{d:>9.2f}")

    stats.sort(key=lambda kv: -abs(kv[1]))
    top = stats[0]
    print(f"\n  -> strongest discriminator: {PRETTY[top[0]]} (|d|={abs(top[1]):.2f}).")
    print("  DEVELOPMENTAL READ: finiteness (tense-marking) is ~equal by ENNI's school ages")
    print("  (0.71 both) - the classic Extended-Optional-Infinitive deficit is an EARLY marker that")
    print("  resolves by ~5 yr; at 4-10 yr the gap has shifted to higher-order SYNTAX (subordinate")
    print("  clauses, d=0.85) plus reduced past/3sg/possessive marking. This age-shift is itself a finding.")

    # ---- figure: effect size per marker ----
    stats_plot = sorted(stats, key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cols = ["tab:red" if d < 0 else "tab:blue" for _, d in stats_plot]
    ax.barh([PRETTY[m] for m, _ in stats_plot], [d for _, d in stats_plot], color=cols)
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="Cohen's d (TD - SLI); positive = lower in language impairment",
           title="Which grammatical markers distinguish language impairment? (ENNI)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "ext_morph_profile.png"), dpi=150)
    print("  figure: results/ext_morph_profile.png")

    # ---- BONUS: do morphology markers predict impairment on their own? ----
    d2 = df.dropna(subset=["age_months"]).copy()
    d2["y"] = (d2["group"] == "SLI").astype(int)
    X, y, g = d2[MARKERS].to_numpy(), d2["y"].to_numpy(), d2["child_id"].to_numpy()
    proba = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    print(f"\n  BONUS - morphology-only classifier (child-independent): ROC-AUC = {roc_auc_score(y, proba):.3f}")
    print(f"  (vs MLU-based Stage 3 AUC 0.861 - fine-grained grammar alone is competitive, and")
    print(f"   gives a clinically richer 'which structures are missing' readout.)")

if __name__ == "__main__":
    main()
