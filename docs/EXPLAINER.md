# FirstWords — The Complete Explainer
### Everything behind the project, taught from scratch: first the science, then the computer science of every step.

> Read this to *understand* the project deeply enough to defend any part of it. It assumes no
> prior linguistics or machine-learning background. Where a concept maps to a specific script,
> the file is named so you can connect the idea to the code. Pair this with `READING_LIST.md`.

---
# PART 1 — THE SCIENCE
---

## 1.1 The problem: language development, and when it goes wrong

Children learn to talk on a remarkably predictable schedule. A typical 18-month-old says single words;
a 3-year-old strings together short sentences; a 6-year-old tells a story with connected clauses. This
progression is one of the most reliable phenomena in all of developmental psychology — reliable enough
that we can treat "how a child talks" as a *measurable developmental signal*, the same way a pediatrician
treats height-for-age.

But not every child follows the schedule. **Developmental Language Disorder (DLD)** — historically called
**Specific Language Impairment (SLI)** — is a condition where a child's language is significantly behind
age expectations *without* an obvious cause like hearing loss, autism, or intellectual disability. It
affects roughly **7% of children** (Tomblin et al., 1997) — more common than autism — yet it is
dramatically **under-recognized and under-funded**. Left unaddressed, early language delay predicts later
difficulties with reading, school, employment, and social-emotional wellbeing.

The clinical catch: **early intervention works**, but screening is often *late, subjective, or gated
behind specialists* that many families can't easily reach. That gap — a common, consequential, treatable
condition that is nonetheless caught late — is the problem FirstWords addresses.

**The core idea (say it in one sentence):** *language complexity rises with age so reliably that we can
plot a child's talking against age norms — a "growth chart for language" — and flag children who look far
behind, so families can seek help sooner.*

## 1.2 What "language level" actually means — the four measurements

We never measure "language" directly; we measure specific, well-established properties of a transcript of
the child speaking. Four families of measures, each capturing a different dimension of development:

**(a) Sentence length — MLU (Mean Length of Utterance).** The average number of words (MLU-words) or
*morphemes* (MLU-morphemes) per utterance. A **morpheme** is the smallest meaningful unit of language:
"cats" = 2 morphemes (*cat* + plural *-s*); "walked" = 2 (*walk* + past *-ed*). MLU is *the* classic
developmental index — Roger Brown (1973) built his famous "stages" on it, and Miller & Chapman (1981)
showed it rises tightly with age. Longer utterances = more advanced. Crucially, MLU is a **per-utterance
average**, which (as we'll see) makes it robust to how long the transcript is.

**(b) Vocabulary diversity — how many *different* words the child uses.** The naive measure is **TTR
(type–token ratio)** = (number of distinct words) ÷ (total words). But TTR has a fatal flaw: it *shrinks*
as a transcript gets longer, purely for statistical reasons (you keep reusing common words like "the").
Richards (1987) documented this. The fix we use is **MATTR-50 (Moving-Average Type–Token Ratio)** — TTR
computed in a sliding 50-word window and averaged, which does **not** depend on transcript length
(Covington & McFall, 2010). Higher diversity = a richer vocabulary.

**(c) Grammatical/syntactic complexity.** Are the child's sentences grammatically sophisticated — do they
use subordinate clauses ("the boy *who was crying*"), tense marking, complex structures? The classic index
is **IPSyn** (Index of Productive Syntax; Scarborough, 1990). *(Important caveat, see §2.1: the software
index we compute is not on the published IPSyn scale, so we call it `syntax_index` and lean on it lightly.)*
For our deeper grammar analysis we compute **specific markers** (past-tense marking, plural, subordinate-
clause rate) directly from the parsed grammar.

**(d) The narrative/task itself.** *How* the sample is collected matters enormously. A child telling a
story (narrative) produces longer, more complex language than casual conversation. This is why we care so
much about comparing children **on the same task** — otherwise a difference in language could just be a
difference in what they were asked to do.

## 1.3 The developmental trajectory — the "growth curve" (and its plateau)

Plot any of these measures against age and you get a rising curve: steep in early childhood (roughly ages
1–5, when language explodes) and **flattening** in later childhood (by school age, sentence length has
largely plateaued — a 9-year-old and an 11-year-old have similar MLU). This has two consequences you must
understand, because they explain several of your results:

1. **The signal is strongest where the curve is steep (young children).** That's why your estimator is
   most accurate on toddlers (Bates corpus, ~2.6-month error) and why adding young children hardened the
   whole model.
2. **On the plateau, age is hard to predict from language** — which is exactly why the Cantonese *Shiro*
   corpus (all 6–11-year-olds) failed and *Szagun* (German toddlers) succeeded. It's the same reason a
   height growth chart is informative for a toddler but flat for an adult.

## 1.4 DLD/SLI — the condition, and how it reveals itself in language

At a matched age, children with DLD produce **shorter, simpler, less grammatically marked** utterances
than typically-developing peers. The most famous specific marker is **tense/agreement**: Rice & Wexler's
(1996) *Extended Optional Infinitive* hypothesis says young DLD children frequently *omit* tense markers
(saying "he walk" for "he walks/walked"). This is an **early-childhood** marker, though — by school age,
most DLD children have acquired basic tense marking, and their deficit **shifts to higher-order syntax**
(fewer subordinate clauses, less complex sentences). Your item-level analysis reproduced *exactly* this:
in the school-age ENNI data, tense-finiteness was equal between groups, but **subordinate-clause rate was
the strongest discriminator** (Cohen's *d* = 0.85). That age-shift is a genuine, textbook-consistent
finding — not a bug.

**Terminology note:** the field renamed "SLI" → "DLD" via the CATALISE consensus (Bishop et al., 2017).
Your corpora are labeled "SLI" (older term); say "DLD (historically SLI)" to sound current.

## 1.5 Cross-linguistic typology — why the best marker differs by language

Languages differ in *how* they package meaning:
- **Analytic** languages (like English) use word order and separate words.
- **Inflected/synthetic** languages (like German, Spanish) pack grammar into word endings.
- **Isolating** languages (like Cantonese) have very little word-ending morphology; meaning rides on
  word choice, particles, and classifiers.

This predicts something your data confirmed: **sentence-length measures (MLU) are a weaker developmental
index in isolating languages**, because length isn't where the complexity lives. In Cantonese, *vocabulary
diversity* (MATTR) tracked age far better than MLU (0.71 vs. 0.38). Klee et al. (2004) report the same
kind of pattern for Cantonese SLI. So the honest, sophisticated claim is: **the growth-chart *method* is
universal, but the *best marker* is typology-dependent** — length/morphemes for inflected languages,
lexical diversity for isolating ones. This is a real developmental-linguistics insight, and it's why your
multilingual harness gates on the *strongest* marker per language rather than assuming MLU.

## 1.6 The data — CHILDES, CHAT, and each corpus

**CHILDES** (Child Language Data Exchange System), part of **TalkBank** (MacWhinney, 2000), is the world's
largest open database of child-language transcripts. Transcripts are in the **CHAT** format: each line is
an utterance (`*CHI:` = child, `*MOT:` = mother), and specialized software (CLAN, or in our case the
Python library `pylangacq`) computes measures from them. Many transcripts include automatically-generated
**`%mor` (morphology)** and **`%gra` (grammatical-relations/dependency)** tiers (Sagae et al., 2010) — the
machinery that lets us count morphemes and detect subordinate clauses.

The corpora you use, and their role:

| Corpus | Language | Role | Why it matters |
|---|---|---|---|
| **Brown** | English | Core TD | Roger Brown's classic longitudinal data (3 children, 18–62 mo) |
| **ENNI** | English | Delay layer + core TD | TD vs. SLI on the **same narrative task at matched age** — the clean delay comparison |
| **Bates, Gleason** | English | Core TD | Many young children — fixed the toddler-range gap |
| **Gillam** | English | External delay validation | Independent US clinical corpus, **age-matched** to ENNI → clean external test |
| **Conti-Ramsden 4** | English (UK) | External delay validation | Adolescents, different task → **out-of-range stress test** |
| **Szagun (TD / CI)** | German | Multilingual level + cross-lingual delay | Toddlers, wide age; CI = cochlear-implant (language-delayed) group |
| **HKU-70** | Cantonese | Multilingual level | Isolating language → the typology finding |
| **Ornat, Marrero, FernAguado** | Spanish | Multilingual (premise) | Confirm the premise within-child; too few wide-age children for a full estimator |

*(Archived in `not_in_use/`: Shiro, BecaCESNo, DiezItza — Spanish corpora that were superseded.)*

**Ethics:** these are de-identified transcripts of minors, used under TalkBank terms. Never attempt
re-identification; the ISEF path is "human participants, pre-existing public dataset" and needs SRC
review + an adult sponsor.

---
# PART 2 — THE COMPUTER SCIENCE (every step, in order)
---

## 2.0 The governing philosophy (why the project is built the way it is)

Three principles, learned the hard way (see the project's `PROJECT_HANDOFF.md` "hard lessons"), drive
every technical decision:

1. **Signal first.** Before building anything, prove on *real, held-out* data that the target has
   predictable signal. Most failed projects skip this and build elaborate models on noise.
2. **Child-independent validation, always.** Never test a model on a child it trained on (that's
   *leakage*; it inflates results and collapses under judge questioning).
3. **Honesty over hype.** Report baselines, confidence intervals, and confounds; name your own limitations.

And one architectural choice: **simple, interpretable models done rigorously** beat complex black boxes.
A judge can understand ridge regression; they can't interrogate a neural net. Simplicity is a *strength*.

## 2.1 Parsing transcripts → features (`stage2_level_estimator.py::extract`)

Every stage starts by turning a CHAT transcript into a row of numbers. For each file we:
1. Parse it with `pylangacq.read_chat(path, strict=False)` (real transcripts break strict parsing).
2. Pull the child's age from the header (`Y;MM.DD` → months, via regex).
3. Ask the library for `mluw()`, `mlum()`, `ttr()`, `ipsyn()` (one value per file, computed on the
   child's utterances).
4. Compute `mattr50` ourselves from the child's word tokens (the length-robust vocabulary measure).

**A real bug we caught (a teaching moment):** the library's `ipsyn()` returns values of 13–36, but
published IPSyn maxes at ~112 and a typical 7-year-old scores ~90. So it is **not** the standard scale.
We didn't hide this — we **relabeled** it `syntax_index`, kept it as a weak feature, and documented it in
the issue log. *Lesson: always sanity-check that a library computes what you think it does against known
norms.* (This is issue **ISS-01** in the handoff.)

## 2.2 Stage 1 — the signal gate (`stage1_check.py`)

Before modeling, we simply checked: does language complexity correlate with age in the TD data, and do
SLI children differ from TD? We measured **Pearson correlation** *r* (a number from −1 to +1 measuring how
tightly two variables move together): corr(age, MLU) ≈ 0.68 pooled, and up to 0.90 *within* a single
child. And SLI children had ~1 standard deviation lower MLU at matched age. **Only because this passed did
we build anything.** This is the step that prevents building on noise.

## 2.3 Stage 2 — the level estimator (`stage2_level_estimator.py`)

**Goal:** predict a child's age (in months) from their language features. This is a **regression** problem
(predicting a continuous number).

**The model: Ridge regression.** Ordinary linear regression fits `age ≈ w₁·mluw + w₂·mlum + … + b`, choosing
the weights `w` that minimize squared error. **Ridge** adds a penalty on large weights (an "L2" penalty,
controlled by `alpha`) — this **regularization** discourages the model from over-relying on any one noisy
feature, which improves generalization to new children. We **standardize** features first (subtract mean,
divide by SD) so the penalty treats them fairly and so coefficients are comparable. We chose ridge because
it's **linear = interpretable** (each feature has one weight with a clear sign) and it **won** — it beat
gradient boosting (a complex nonlinear model). When the simple model wins, you keep it.

**Validation: GroupKFold by child.** To estimate honestly how the model does on *new* children, we use
**k-fold cross-validation**: split the data into 5 "folds," train on 4, test on the held-out 1, rotate,
and pool the predictions ("out-of-fold"). But a plain split could put a child's session A in training and
session B in testing → the model "memorizes" that child → **leakage** → fake-good numbers (Kaufman et al.,
2012). **GroupKFold** guarantees all of a child's files stay in the same fold, so we only ever test on
*genuinely unseen children*. This is the single most important rigor choice in the project.

**Baseline.** We always compare against "predict the mean age" (a model that ignores the features). Our
MAE (mean absolute error) is ~10 months vs. the baseline's ~25 — so the language features roughly halve the
error. Without a baseline, a number like "10 months" is meaningless.

**Uncertainty: cluster bootstrap.** A single MAE could be luck. The **bootstrap** re-computes the metric on
many resamples of the data to get a 95% confidence interval. Because a child's multiple files aren't
independent, we resample **children, not files** (a *cluster* bootstrap) — otherwise the interval would be
falsely narrow. Result: MAE 9.99 mo **[95% CI 9.11–11.26]**, R² 0.81.

- **R²** = fraction of the variance in age the model explains (1 = perfect, 0 = no better than the mean).
- **MAE** = average absolute error in months — the interpretable "how far off" number.

**The age-gap score.** We define `gap = predicted_language_age − actual_age`. For TD children it centers
near 0; for a delayed child it's strongly negative ("talks like a younger child"). This is the literal
"growth-chart position."

## 2.4 Stage 3 — the delay classifier (`stage3_delay_classifier.py`)

**Goal:** given a transcript, output typically-developing vs. language-impaired. This is **binary
classification**.

**The model: Logistic regression.** Like linear regression, but it squashes the output through a logistic
(sigmoid) function to produce a **probability** between 0 and 1. Interpretable (one weight per feature),
and it won against gradient boosting again.

**ENNI only, on purpose.** We train and evaluate on ENNI alone because ENNI has TD and SLI on the *same
task at matched age*. Mixing in other corpora would let the model cheat by detecting *task* or *age*
instead of impairment.

**Class imbalance.** Only ~21% of ENNI children are SLI. A naive model could get 79% "accuracy" by always
saying "TD" — useless. Two responses: (1) `class_weight='balanced'` makes the model pay proportionally more
attention to the rare class; (2) we **never report raw accuracy** — we use balanced metrics.

**Validation: StratifiedGroupKFold.** Child-independent (Group, as before) *and* Stratified — each fold
preserves the 79/21 class ratio so no fold accidentally has too few SLI children.

**Metrics (know each one):**
- **ROC-AUC** = the probability the model ranks a random SLI child above a random TD child. 0.5 = coin
  flip, 1.0 = perfect. Ours = **0.861**. AUC is *threshold-independent*, so it summarizes ranking quality.
- **PR-AUC** (precision–recall AUC) = better for imbalanced data (Saito & Rehmsmeier, 2015); baseline =
  prevalence (0.21), ours = 0.64.
- **Sensitivity** (recall) = fraction of truly-impaired children caught. **Specificity** = fraction of TD
  correctly cleared. A *screen* favors **high sensitivity** (don't miss affected kids), accepting more
  false alarms. At our screening threshold: 91% sensitivity.
- **Threshold / operating point.** The model outputs a probability; we pick a cutoff. Lower cutoff = catch
  more (higher sensitivity, lower specificity). The **Youden point** maximizes sensitivity+specificity−1.

**★ The artifact check — the most important idea in the whole project.** The classic attack on automatic
language screening is: *"you're not detecting impairment, you're detecting shorter transcripts or younger
kids."* We tested this head-on by training the classifier on **only age** (AUC 0.42) and **only transcript
length** (AUC 0.49) — both no better than chance. Meanwhile language features alone score 0.78. **So the
signal is genuinely linguistic, not a confound.** This is *confound control*, a graduate-level move, and
it's your best slide. (A subtlety: age has the highest *SHAP* importance in the full model, yet age *alone*
can't classify — because age is the **reference frame** that tells the model what MLU to *expect*; the flag
fires when language falls short of the age expectation. Age frames the norm; language triggers the flag.)

## 2.5 Stage 4 — interpretability (`stage4_interpretability.py`)

A screen that says "flagged" with no reason is a black box. Two complementary explanations:

- **SHAP (SHapley Additive exPlanations; Lundberg & Lee, 2017).** Borrowed from game theory: it fairly
  attributes a prediction to each feature (how much each pushed the output up or down). For our *linear*
  models these attributions are exact. This yields the global "which markers matter most" ranking.
- **Age-normed z-scores.** SHAP says what the *model* used; a *human* needs "is this high or low **for this
  age**?" So we fit, on TD children, each feature's expected value as a function of age, and express a
  child's score as a **z-score** (standard deviations from the age-expected mean). That turns a prediction
  into a clinician-style readout: *"morpheme-MLU is 2.2 SD below age norm."*

## 2.6 Stage 5 — the demo (`stage5_demo.py`)

The judge-facing tool. Paste (or point to) a transcript → it extracts features and prints: estimated
language age, age-gap, **growth-chart percentile**, delay risk score, the specific markers below age norm,
and a **non-diagnostic** recommendation. It reuses the exact validated pieces (`readout()`,
`percentile_for()`). This is the single biggest score-mover at a fair — judges remember a working tool.

## 2.7 The extensions — each one is a distinct CS/stats concept

- **Growth-chart percentiles (`ext_percentiles.py`).** **Quantile regression** fits not the mean but the
  10th/25th/…/90th percentile curves of a marker vs. age. A child then lands on a percentile, exactly like
  a pediatric height chart. Validation: ~10% of TD children fall below the P10 curve (as they should if the
  bands are honest).
- **Cross-corpus / external validation (`ext_crosscorpus.py`).** The gold standard for "does it
  generalize": train on one dataset, test on a *different lab's* dataset with **no retraining**. This
  measures robustness to **distribution shift**. Gillam (age-matched) → AUC 0.80–0.82; Conti4 (out-of-range)
  → 0.70. These double as your **locked-box** sets (never used for tuning).
- **Short-sample robustness (`ext_shortsample.py`).** Truncate each transcript to the first N utterances,
  recompute features, re-measure AUC. It barely drops down to ~10 utterances → "a brief sample is enough,"
  a key practicality argument.
- **Fairness / subgroup audit (`ext_fairness.py`).** Break performance down by sex and age band to check
  the screen isn't biased. Honest reporting of a modest sex gap (small subgroups → noisy).
- **Calibration (`ext_calibration.py`).** A probability is *calibrated* if "0.7" really means ~70% are
  affected. The **Brier score** measures this; a **reliability diagram** plots it. The recall-tuned model
  overstates probabilities; **isotonic regression** recalibrates them (Niculescu-Mizil & Caruana, 2005).
- **Item-level grammar profile (`ext_morphprofile.py`).** Instead of one syntax score, count *specific*
  grammatical markers (past tense, plural, subordinate clauses) from the `%mor`/`%gra` tiers, and compare
  TD vs. SLI with **Cohen's *d*** (a standardized effect size: difference in means ÷ pooled SD). Finding:
  subordinate-clause rate *d* = 0.85. *(We dropped one length-confounded marker — ISS-10 — the same TTR-
  length lesson.)*
- **Cross-task reliability (`ext_taskreliability.py`).** Using the same children on two tasks, we asked
  whether the flag *agrees* across samples. Out of range it didn't (r = 0.11) — an **honest negative** we
  reported as a limitation (scores compress out of the trained age range).
- **Multilingual level estimator (`ext_multilingual.py`).** The same pipeline in German and Cantonese,
  with the **gate on the strongest marker** per language (the typology insight). Reports within-child
  correlation (the premise test) and a child-independent estimator with a bootstrap CI.
- **★ Cross-lingual delay + a confound we caught (`ext_crosslingual_delay.py`).** German TD vs. cochlear-
  implant (language-delayed). The naive AUC was 0.99 — but the **artifact check flagged that age alone
  scored 0.92** (the impaired cohort was older). So we discarded the confounded number and reported the
  **age-matched, language-only AUC 0.90**. This is the artifact-control discipline from Stage 3, applied
  again — and a showcase of catching your own confound (ISS-15).
- **Decision-curve analysis (`ext_clinical_utility.py`; Vickers & Elkin, 2006).** Turns AUC into
  *usefulness*: across the range of "how sure before you refer" thresholds, does using the screen give more
  **net benefit** than "refer everyone" or "refer no one"? Ours wins across the whole plausible range.

## 2.8 Software practice & reproducibility (quietly important)

- **Fixed random seeds** everywhere → every number reproduces exactly on re-run.
- **Git** with small, message-clear commits; data is **gitignored** (never commit minors' transcripts).
- **The Issue & Decision Log** (handoff §0c): every confound/bug we found and fixed, dated, with evidence
  (ISS-01 … ISS-15). This *is* your experimental-design paper trail — the thing that proves rigor.
- Each stage is **run and verified** before the next; we report the *real* numbers, never hoped-for ones.

---
# PART 3 — Judge-question crosswalk (concept → where it lives → the source)
---

| If a judge asks… | The answer is… | Lives in | Reading |
|---|---|---|---|
| "How do you know it didn't just memorize kids?" | Child-independent GroupKFold — no child in both train & test | Stage 2/3 | Kaufman 2012 |
| "Isn't this just transcript length?" | Length-alone AUC = 0.49 (chance); MLU is per-utterance | Stage 3 artifact check | Richards 1987 |
| "Isn't it just age?" | Age-alone AUC = 0.42; age is the *norm reference*, not the signal | Stage 3 artifact check | — |
| "Does it work on new data?" | Gillam (diff. lab) AUC 0.82, no retraining | ext_crosscorpus | Steyerberg 2019 |
| "IPSyn of 22 at age 7 is wrong." | Correct — library index ≠ published scale; we relabeled it | ISS-01 | Scarborough 1990 |
| "What's novel?" | Not the ML — the honest, interpretable, *usable* tool + the typology finding | whole project | Gabani 2009 |
| "Is it actually useful, not just accurate?" | Decision curve beats refer-all / refer-none | ext_clinical_utility | Vickers 2006 |
| "Why subordinate clauses, not tense?" | Tense deficit resolves by school age; deficit shifts to syntax | ext_morphprofile | Rice & Wexler 1996 |
| "Why a different feature in Cantonese?" | Isolating language — diversity, not length, carries age | ext_multilingual | Klee 2004 |
| "Your German delay AUC 0.99 seems high." | It was an age confound; honest number is age-matched 0.90 | ext_crosslingual_delay | ISS-15 |
| "What are the limitations?" | (name them yourself: few young distinct children per language, CI≠SLI, screen not diagnosis, needs ASR for deployment) | handoff §9f | USPSTF 2006 |

---

**The through-line to remember:** the project's strength is not a fancy algorithm. It is a *real, verified
signal* + *child-independent validation* + *confound control* + *interpretability* + *a working tool* +
*honest self-audit*, applied to a common, consequential, under-served problem — and now shown to generalize
across labs and language families. Every number is defensible, and you found your own mistakes before a
judge could. That is what good science looks like.
