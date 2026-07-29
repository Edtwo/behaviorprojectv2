# FirstWords — Methods & Analysis Walkthrough
### A step-by-step technical companion for the team: what each analysis does, every methodological choice, and *why* — so anyone can explain it.

> Read `EXPLAINER.md` first for the concepts (what a model is, cross-validation, AUC, etc.).
> THIS document walks the actual **pipeline and analysis decisions** at a fine grain, script by script,
> and gives you an out-loud way to explain each one. When it says *"SAY:"* that's a sentence you can use
> with a teammate or judge.

---

## 0. The data flow (how the scripts connect)

```
CHILDES .cha transcripts
        │  parse + measure (pylangacq)  ── stage2_level_estimator.py::extract()
        ▼
results/features.csv   (one row per transcript: child_id, corpus, group, age, 4 features…)
        │
        ├── stage2_level_estimator.py   → level estimator (predict age)         → calibration + gap figures
        ├── stage3_delay_classifier.py  → delay classifier (TD vs SLI)          → ROC + artifact check
        ├── stage4_interpretability.py  → SHAP + age-normed readout             → drivers
        ├── stage5_demo.py              → the paste-a-transcript demo (reuses 2+3+4)
        └── ext_*.py                    → each extension reads features.csv or re-parses corpora
```
Everything downstream depends on `features.csv`, which Stage 2 regenerates. **Rule: if you change feature
extraction, re-run Stage 2 first, then everything else.**

*SAY:* "The pipeline is one feature table feeding a small set of interpretable models; every script is
child-independent and reproducible from that table."

---

## 1. Feature extraction — turning a transcript into numbers
**Where:** `stage2_level_estimator.py::extract()` and `mattr()`.

**Step by step:**
1. `pylangacq.read_chat(path, strict=False)` parses the CHAT files. **Why `strict=False`:** real transcripts
   contain quirks (e.g. a pause marker inside a word) that break strict parsing; we verified this.
2. Age comes from the CHAT header `Y;MM.DD`; `months()` converts it with a regex to a single number in
   months. **Why months:** a common continuous unit so all corpora are comparable.
3. The library computes `mluw()` (mean length of utterance in words), `mlum()` (in morphemes), `ttr()`
   (type-token ratio), `ipsyn()` (a syntax index) — each returns **one value per file**, computed on the
   *child's* utterances only.
4. We compute `mattr50` ourselves: `mattr()` slides a 50-word window across the child's words and averages
   the type-token ratio. **Why we don't just use `ttr`:** raw TTR shrinks as a transcript gets longer (a
   length artifact); MATTR does not. This is a deliberate, defensible fix.

**A methodological catch we handled (ISS-01):** the library's `ipsyn()` returns 13–36, but the *published*
IPSyn scale maxes near 112. So it is **not** standard IPSyn. We **renamed it `syntax_index`** and treat it
as a weak generic feature, never as "IPSyn." *SAY:* "We sanity-checked every library measure against known
norms and relabeled the one that didn't match its published scale."

**Child IDs (this is subtle and important):** each corpus names children differently, so `build_features()`
assigns `child_id` per corpus — Brown by folder, ENNI by a header fingerprint (birthdate+sex), Bates by a
normalized given-name (files repeat a child across task folders), Gleason by filename stem. **Why it
matters:** the child_id is what makes validation child-independent; get it wrong and you leak (see §3).

---

## 2. The signal gate — proving there's something to model
**Where:** `stage1_check.py`.

Before any modeling, we check two things on real data: (a) does language complexity rise with age in
typically-developing children? (Pearson correlation of age with each feature); (b) do impaired children
differ from typical ones at matched age? **Why first:** modeling a target with no real signal is how
projects fail (our handoff's "hard lessons"). Only because both checks passed did we build models.

*SAY:* "We refused to build a model until we'd shown, on held-out real data, that the signal exists — 
correlation of age with sentence length was 0.68 across children, and up to 0.90 within a child."

---

## 3. The level estimator — every choice explained
**Where:** `stage2_level_estimator.py::main()`.

- **Task:** regression — predict age (months) from the 4–5 features.
- **Model:** ridge regression inside a `StandardScaler` pipeline.
  - *Ridge vs plain linear:* ridge adds an L2 penalty (`alpha=1.0`) that shrinks the weights, reducing
    overfitting and improving generalization to new children.
  - *Why standardize:* puts features on the same scale so the penalty is fair and weights are comparable.
  - *Why linear at all:* interpretable (one signed weight per feature) and it **won** — we compared against
    gradient boosting (a nonlinear model) and it did not beat ridge by a meaningful margin, so we keep the
    simpler model. The code picks ridge unless another model beats it by >1 month MAE (a stated tie-break).
- **Validation:** `GroupKFold(n_splits=5)` grouped by `child_id`.
  - *What it does:* splits children (not files) into 5 folds; trains on 4, predicts the held-out 1, rotates;
    the pooled predictions are "out-of-fold."
  - *Why grouped:* some children have multiple recordings; a plain split would put one child on both sides
    and the model could memorize them (**leakage**). GroupKFold makes each child fully seen or fully unseen.
- **Baseline:** `_Mean` predicts the average age regardless of features. **Why:** a score is meaningless
  without the dumb-model comparison; we cut MAE from ~25 (baseline) to ~10.
- **Uncertainty:** `cluster_bootstrap()` resamples **children** (not files) 2000× and recomputes MAE/R².
  - *Why cluster (by child):* a child's files aren't independent; resampling files would give a falsely
    narrow interval. Resampling children is the honest version.
- **Outputs:** the age-gap score (`predicted_age − actual_age`), a calibration scatter (predicted vs actual),
  and an informal SLI preview.

*SAY:* "A regularized linear model, validated on children it never trained on, predicts developmental age to
within about ten months — less than half the error of guessing the mean — with a bootstrap confidence
interval computed by resampling children, not recordings."

---

## 4. The delay classifier — and the single most important analysis
**Where:** `stage3_delay_classifier.py`.

- **Task:** binary classification — typically-developing vs. language-impaired (SLI), on **ENNI only**.
  - *Why ENNI only:* ENNI has both groups doing the *same* narrative task at *matched* age. Mixing corpora
    would let the model detect task or age instead of impairment.
- **Model:** logistic regression, `class_weight='balanced'`.
  - *Why balanced:* only ~21% are SLI; balancing makes the model attend to the rare class.
  - *Why not accuracy:* always-guess-"typical" gives 79% accuracy and catches zero impaired kids. We report
    ROC-AUC, PR-AUC, sensitivity, specificity instead.
- **Validation:** `StratifiedGroupKFold` — child-independent (Group) *and* class-ratio-preserving (Stratified).
- **Metrics:** ROC-AUC 0.861 (ranking quality), PR-AUC 0.64 (baseline = prevalence 0.21), plus
  sensitivity/specificity at a chosen threshold. We pick the operating point for **high sensitivity** (a
  screen shouldn't miss affected children).

### ★ The artifact check (this is the crown jewel — know it cold)
The classic attack on transcript-based screening: *"you're detecting transcript length or the child's age,
not impairment."* We answer it directly by training the classifier on **only the suspected shortcut**:
- age alone → AUC **0.42** (below chance)
- transcript length alone → AUC **0.49** (chance)
- language features alone → AUC **0.78**

**Conclusion:** the signal is genuinely linguistic, not a confound. *SAY:* "We tried to predict impairment
from age alone and from transcript length alone — both failed at chance — so the model must be using real
language differences." (Subtlety worth knowing: age has the highest SHAP *importance* in the full model, yet
age *alone* can't classify — because age is the **reference frame** telling the model what to *expect*; the
flag fires when language falls short of the age-expected norm.)

---

## 5. Interpretability
**Where:** `stage4_interpretability.py`.

- **SHAP** (exact for our linear models) gives each feature's contribution to a prediction → the global
  "which markers matter" ranking.
- **Age-normed z-scores:** we fit each feature's expected value as a function of age on typical children,
  then express a child's score as standard deviations from that age norm → a human-readable readout
  ("morpheme-MLU is 2.2 SD below age norm"). SHAP says what the model used; the z-score says whether it's
  high or low *for the age*.

---

## 6. External validation — the strongest evidence
**Where:** `ext_crosscorpus.py`.

Train on ENNI, test on a **different lab's** corpus with **no retraining** (the model never saw it — a true
held-out / "locked-box" test). We report AUC with a bootstrap CI for both a full (with age) and a
language-only model, per task.
- **Gillam** (US, age-matched to ENNI): AUC **0.80–0.82** — the clean headline. We verified Gillam's TD/SLI
  groups are age-matched (98 vs 97 months), so it isn't an age artifact.
- **Conti-Ramsden 4** (UK, adolescents — out of our age range): AUC **0.70** — a stress test. Because it's
  out of range, we report the *language-only* model (the age feature would be extrapolated).

*SAY:* "The strongest evidence a tool generalizes is that it works, unchanged, on data from a different lab —
we did that twice."

---

## 7. The extensions (method per script)
- `ext_percentiles.py` — **quantile regression**: fit P10/25/50/75/90 curves of a marker vs age → a child's
  percentile, like a growth chart. Validated: ~10% of typical kids fall below P10.
- `ext_shortsample.py` — truncate transcripts to the first N utterances, recompute length-robust features,
  re-measure AUC vs N. Barely drops to ~10 utterances → "a brief sample suffices."
- `ext_fairness.py` — subgroup AUC/sensitivity/specificity by sex and age band (honest small-cell caveats).
- `ext_calibration.py` — **Brier score** + reliability diagram; isotonic recalibration fixes the recall-tuned
  model's overstated probabilities.
- `ext_morphprofile.py` — per-child grammatical marker rates from the `%mor`/`%gra` tiers; TD vs SLI with
  **Cohen's d**. Subordinate-clause rate is the strongest (d=0.85). We dropped one length-confounded marker.
- `ext_taskreliability.py` — do the same child's two task-samples agree? Out of range they don't (r=0.11) —
  an **honest negative** reported as a limitation.
- `ext_crosslingual_delay.py` — German TD vs cochlear-implant (language-delayed). The artifact check caught a
  group **age confound** (age-only 0.92), so we report the **age-matched** language-only AUC **0.90**.
- `ext_clinical_utility.py` — **decision-curve analysis** (net benefit vs "refer all"/"refer none"): the
  screen wins across the plausible threshold range → it's *useful*, not just accurate.
- `ext_multilingual.py` — the level estimator in German (MAE 3.1mo, R²0.79) and Cantonese (MAE 6.5mo). Gates
  each on the strongest marker with a bootstrap CI.
- `ext_unified_crosslingual.py` — one model on language-standardized features. Honest cautionary result:
  worse than per-language, and it doesn't transfer to an unseen language → per-language norms are necessary.
- `ext_typology.py` — **RETRACTED hypothesis** (a model of scientific honesty): with 3 languages it looked
  like the best marker tracks typology; adding Mandarin (a 2nd isolating language) **contradicted** it, so we
  retracted the claim (the effect is confounded with task). *SAY:* "We tested our own hypothesis with more
  languages, it failed, and we retracted it — that's the process working."

---

## 8. Reproducibility & integrity practices
- **Fixed seeds** (`random_state=0/42`) everywhere → every number reproduces on re-run.
- **Git** with small commits; **data is gitignored** (never commit minors' transcripts).
- **The Issue & Decision Log** (`PROJECT_HANDOFF.md` §0c, ISS-01…ISS-16): every confound/bug/retraction,
  dated, with evidence — this is the experimental-design paper trail.
- **Every stage is run and verified** before the next; we report the real numbers.

*SAY:* "The project's real strength isn't a clever algorithm — it's that every result is child-independent,
confound-controlled, externally validated, and honestly audited, including the hypothesis we tested and
retracted."

---

## 9. How to study this so you can explain it (for the team)
For each of §§1–8: (1) read the *SAY* line; (2) open the script and find the function named; (3) run it with
`.venv/bin/python src/<script>.py` and read the printed numbers; (4) change one thing (a threshold, a
feature) and see what moves — understanding what *breaks* a result is how you learn to defend it. If you can
run it, explain the *SAY* line in your own words, and answer "why this choice and not the alternative,"
you own that step.
