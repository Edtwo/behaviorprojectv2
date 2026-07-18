# FirstWords - PROJECT HANDOFF & COMPLETE PLAN
### An interpretable tool that estimates a child's language-development level (and flags likely delay) from a speech sample
_Self-contained record. Any agent (including a collaborator's agent) MUST read Sections 1-2 before touching the project._
_Updated 2026-07-16 (Stage 2 level estimator DONE + verified; audit + Issue Log 0c added). Category: ISEF Behavioral & Cognitive Sciences (Developmental Psychology)._

================================================================================
## 0. TL;DR
================================================================================
- **Project:** FirstWords - takes a short transcript of a young child's speech, estimates their **language-developmental level** vs. age norms, and **flags likely language delay**. An accessible, interpretable early-screening aid (early intervention greatly improves outcomes). Analogy for judges: a "growth chart for language."
- **True BEHA fit:** measures a developmental/cognitive construct (language acquisition) = Developmental Psychology. (We rejected an ALS idea because it was Translational-Medical, not BEHA.)
- **Two-layer, de-risked design:** (1) CORE = developmental-level estimator (textbook-certain signal -> cannot dead-null; a complete project by itself); (2) HEADLINE = delay flag (typically-developing vs. language-impaired), signal-gated with a clean fallback to the core.
- **STATUS (2026-07-14): data in hand + BOTH signals VERIFIED on real data.** Next = Stage 2 (build the level estimator), then Stage 3 (delay classifier). Nothing is a gamble now.
- **Data:** CHILDES (TalkBank). Downloaded: **Brown** (typically-developing) + **ENNI** (typically-developing AND language-impaired, same narrative task). Free TalkBank login required to download; de-identified.
- **Stack (verified, Python 3.14 venv):** pylangacq 0.23 (CHAT parser; built-in MLU/IPSyn/TTR/ages), pandas, numpy, scikit-learn, shap, matplotlib. See requirements.txt.
- **Honest ceiling:** genuine BEHA placement/finalist potential at Dallas Regional -> ISEF with strong execution + a live demo + the "helps kids" story. Not a guaranteed top prize; but it WILL complete and be a real, correctly-categorized, working tool.

================================================================================
## 0b. VERIFIED FACTS (established by running real checks - trust these)
================================================================================
### Signals (from `src/stage1_check.py` on the real data, 2026-07-14)
- **CORE signal CONFIRMED (Brown, typically-developing):** 214 transcripts, ages 18.0-62.4 months. Language complexity rises with age: corr(age, MLU-words)=0.68, MLU-morphemes=0.69, TTR=0.62, IPSyn=0.53. A multivariate model will do better. => the developmental-level estimator will work.
- **DELAY signal looks REAL (ENNI, TD vs SLI, AGES MATCHED ~85 mo):** TD n=286, SLI n=75. At the same age, language-impaired children have much lower MLU: words 6.90 (TD) vs 5.67 (SLI); morphemes 7.09 vs 5.81 - roughly a 1 SD gap on the SAME narrative task (so not an age or task artifact). TTR 0.37 vs 0.35 (small).
### 2026-07-16 audit (full re-run; every 0b number reproduced exactly; conclusions below - see Section 0c for the full issue write-ups & evidence)
- **`syntax_index` (pylangacq ipsyn) is NOT the published IPSyn scale** - keep as a feature, never call it "IPSyn." [ISS-01]
- **The IPSyn "anomaly" (SLI > TD) was a transcript-length artifact** - drop/length-correct it for TD-vs-SLI; MLU is length-robust and does the discriminating. [ISS-02]
- **ENNI A/B folders are DISJOINT children** -> each ENNI file = one distinct child (286 TD + 75 SLI); child-independent splits are trivial. [ISS-03]
- **ENNI ages span 48-120 mo, only MEAN-matched at ~85 mo** -> Stage 3 must include age as a covariate. One TD file lacks an age header (285/286 aged). [ISS-04, ISS-06]
- **Raw TTR is length-confounded** -> added MATTR-50 (`mattr50`), length-robust. [ISS-05]
- **Core signal is stronger WITHIN child than pooled:** Brown corr(age, MLUw) = 0.90 (Adam), 0.81 (Eve), 0.88 (Sarah) vs 0.68 pooled -> genuinely developmental, not a pooling artifact.
- **ENNI TD doubles as level-estimator data:** 286 distinct TD children at 48-120 mo, overlapping Brown at 48-62 mo -> extends the core across ages; task differs (narrative vs home conversation), so control/report corpus.
### Stage 2 RESULTS (2026-07-16, `src/stage2_level_estimator.py`, real out-of-fold numbers)
- TD modeling set: 499 files, 288 distinct children, 18-120 mo. Features: mluw, mlum, ttr, mattr50 (moving-average TTR, length-robust), syntax_index (pylangacq ipsyn, relabeled). GroupKFold by child, 5 folds; cluster bootstrap CIs resample CHILDREN.
- **Ridge (linear, interpretable) WINS: MAE 11.46 mo [95% CI 10.20..13.84], R2 0.74 [0.39..0.77]** vs predict-mean baseline MAE 27.77 [24.59..29.68] vs grad-boosting MAE 12.04. Per-corpus MAE: Brown 9.19, ENNI 13.17 (older ages harder - complexity growth flattens).
- Age-gap score defined = predicted language age - chronological age; TD out-of-fold gap mean -1.63 mo (sd 13.9) - roughly centered, as it should be.
- **SLI preview (informal, same task, vs ENNI-TD out-of-fold gaps): SLI mean gap -16.45 mo vs TD -0.90 mo -> SLI children's estimated language age lags ~15.6 months, Cohen's d ~ 0.98.** Judge-friendly line: "children with language impairment sound ~1.3 years younger than they are." Stage 3 must make this rigorous (age + length covariates, balanced metrics, CIs).
- Outputs: results/features.csv (575 files incl SLI), results/stage2_calibration.png, results/stage2_gap_preview.png.
### Access / tooling / parsing (all verified)
- CHILDES downloads require a FREE TalkBank login (no anonymous route). Corpora are per-language collections; clinical corpora live under "Clinical-Eng" (ENNI/Gillam/Conti-Ramsden); typically-developing English under "Eng-NA" (Brown, etc.). Clinical corpora may be a higher access tier.
- pylangacq API (rustling backend, v0.23): `read_chat(path, strict=False)` - MUST use strict=False (real CHAT files break strict parsing, e.g. "pause marker embedded in word"). Reader members: `n_files` and `file_paths` are ATTRIBUTES (no parens); `ages()` is a method with NO kwargs returning Age objects (parse "Y;MM.DD" -> months via regex); `mluw()/mlum()/ttr()/ipsyn()` are methods returning lists aligned to `file_paths`, computed for the target child. `mlum`/`ipsyn` need the %mor/%gra tiers (present in real CHILDES, absent in bare synthetic files).
- Feature meanings: MLU-words = mean words per utterance; MLU-morphemes = mean morphemes per utterance; TTR = type-token ratio (lexical diversity); IPSyn = Index of Productive Syntax (syntactic complexity).

================================================================================
## 0c. ISSUE & DECISION LOG (living record of the experimental design - UPDATE EVERY TIME AN ISSUE IS FOUND OR RESOLVED)
================================================================================
_Purpose: this is the project's methods lab-notebook for the ISEF board/paper. Every problem we hit,
every confound we caught, and every design choice we made in response goes here, dated, in the format
below. Judges reward researchers who FIND their own flaws and fix them - this section is the paper trail
that proves we did. Do NOT delete entries when resolved; mark them RESOLVED and keep the reasoning.
When an issue is closed, also fold the one-line conclusion into 0b (verified facts) so future agents trust it.
Format per entry: ID | date-found | STATUS (OPEN / RESOLVED / MONITORING) | what we saw | why it matters | what we did | evidence._

- **ISS-01 | 2026-07-16 | RESOLVED | pylangacq's `ipsyn()` is NOT the published IPSyn scale.**
  - Saw: values range 13-36 across every file in both corpora; ENNI TD 7-year-olds average ~22. Published IPSyn maxes at ~112 and TD 7-yr-olds score ~90+. Confirmed the default already uses the standard 100-utterance window (`ipsyn(n=100)` gives identical values), so it is not a settings mistake - the compiled backend computes a partial/nonstandard index.
  - Why it matters: presenting "IPSyn = 22 at age 7" to a knowledgeable BEHA judge who knows IPSyn norms would look like an error and damage credibility.
  - Did: kept the metric as a FEATURE (it still carries age signal, r=0.53 in Brown) but RELABELED it `syntax_index` everywhere (code, poster, paper) and will never call it "IPSyn (Scarborough 1990)." Documented the scale caveat.
  - Evidence: `.venv/bin/python -c "...ipsyn() vs ipsyn(n=100)..."`; Stage 2 feature list uses `syntax_index`.

- **ISS-02 | 2026-07-16 | RESOLVED | "IPSyn anomaly" - SLI scored slightly HIGHER than TD (22.56 vs 22.03), the wrong direction.**
  - Saw: counterintuitive since SLI = language-impaired. Investigated as a possible artifact.
  - Why it matters: an uncontrolled confound in a discriminating feature would undermine the whole delay-signal claim.
  - Did/found: it is a TRANSCRIPT-LENGTH artifact. `syntax_index` correlates with number of child utterances (r=0.37 TD, 0.50 SLI), 74% of ENNI files have <100 utterances, and SLI narratives are LONGER on average (93.6 vs 89.0 utts). Length-adjusting shrinks the gap (+0.53 -> +0.35); either way this feature does not separate the groups. CONCLUSION: drop or length-correct `syntax_index` for TD-vs-SLI; lean on MLU (length-robust: r with n_utts = 0.09-0.14). This validated our own artifact-control doctrine (Lesson 4) and is a poster-worthy "we caught a confound" panel.
  - Evidence: scratchpad length-confound diagnostic; pooled regression residuals by group.

- **ISS-03 | 2026-07-16 | RESOLVED | Are ENNI's A/B story-set folders the SAME children twice (which would leak across a train/test split)?**
  - Saw: ENNI is split into story-form A and story-form B folders per group; needed to know if a child appears in both.
  - Why it matters: if A and B are the same kids, a naive split would put one child on both sides = leakage = inflated results.
  - Did/found: file-name intersection between A and B is ZERO in both TD and SLI -> disjoint children -> each ENNI file is one distinct child (286 TD + 75 SLI). Child-independent splitting on ENNI is therefore trivial. Still used GroupKFold-by-child in Stage 2 for uniformity/safety.
  - Evidence: `comm -12` on the A/B file lists.

- **ISS-04 | 2026-07-16 | MONITORING | ENNI ages are only MEAN-matched (~85 mo), but individuals span 48-120 mo (sd~20).**
  - Saw: the "ages matched" claim is true for the group means/spread, not per child.
  - Why it matters: for the Stage 3 delay classifier, age is a confound if not handled - an older SLI child could be mistaken for delayed purely by age mixing.
  - Did/plan: include age as a covariate in Stage 3 and report age-band subgroups; in Stage 2 the age-gap score already subtracts chronological age so this is handled there. Left OPEN->MONITORING until Stage 3 implements it.
  - Evidence: ENNI age summaries (TD 48-120, SLI 50-118).

- **ISS-05 | 2026-07-16 | RESOLVED | Raw TTR (type-token ratio) shrinks as a transcript gets longer (length confound).**
  - Saw: known property of TTR; ENNI transcripts vary widely in length.
  - Why it matters: a length-confounded lexical-diversity feature would smuggle transcript length into the model.
  - Did: added MATTR-50 (moving-average TTR over sliding 50-word windows), which is length-robust, as `mattr50`; kept raw `ttr` too for comparison. Used in Stage 2.
  - Evidence: `mattr()` in `src/stage2_level_estimator.py`.

- **ISS-06 | 2026-07-16 | RESOLVED (noted) | One ENNI-TD file has no age header (285 of 286 aged).**
  - Saw: `ages()` returns None for one TD file.
  - Why it matters: silently dropping or mis-handling missing ages could bias the set.
  - Did: age-dependent code filters `age_months.notna()` explicitly (that one file is excluded from age modeling, kept for feature stats). Documented rather than hidden.
  - Evidence: Stage 2 "TD modeling set: ... 285 ENNI" vs 286 files.

================================================================================
## 1. HARD LESSONS - DO NOT REPEAT (paid for across DopaLoop, the idea-search, ParkinTrack, the ALS/category detour)
================================================================================
1. **Signal first, always.** Before building, verify on REAL data (child/subject-INDEPENDENT) that the target has predictable signal, measured honestly. This killed DopaLoop (asym->relapse, null AUC 0.30) and ParkinTrack (voice did NOT track within-person UPDRS). FirstWords passed this gate (Section 0b).
2. **Prefer targets whose signal is already known/published.** "Crowded with real signal" beats "novel with no signal." FirstWords' core (age<->language) is textbook-certain.
3. **Novelty != never-been-done.** ISEF-winnable = a specific angle + excellent execution + a usable TOOL + interpretability + clear impact + a demo + genuine understanding, judged vs other high schoolers. Chase workability + execution.
4. **Validate honestly:** child-INDEPENDENT splits (GroupKFold by child); baselines; calibration + bootstrap CIs; NEVER report in-sample/leaked numbers. Beware dataset ARTIFACTS (transcript length, #utterances, corpus source faking a "delay" signal) - control for them. (ENNI's matched ages + same task already remove the biggest confounds.)
5. **Scope discipline / signal-gate every added claim.** No unvalidatable bolt-ons. Depth on one validated output > breadth of hollow arms.
6. **Verify data ACCESS + category fit early.** (Enroll-HD needed a sponsor + months; ALS-Forecast was TMED not BEHA; CHILDES needs a free login; clinical corpora may be a higher tier.)
7. **Complexity belongs in the PROBLEM, not the architecture.** Simple, interpretable models done rigorously.
8. **Honesty over hype.** State limitations plainly; name them yourself (a credibility win).
9. **Winners build a usable thing for real people + a demo** (OncoNote/Harmony/ArabLexify/SuiSensor). ML is the engine, not the headline.

================================================================================
## 2. OPERATING RULES FOR ANY AGENT (incl. a collaborator's agent - repo is a git repo, collab with polaris242)
================================================================================
- Read Sections 0-1 before doing anything. The project is COMMITTED (FirstWords). Do NOT re-open the idea search; do NOT pivot unless a signal-gate fails AND it's documented here.
- Never build model code on an unverified target: confirm signal on real data, child-independent, vs a baseline, FIRST.
- Coordinate via git. Commit small with clear messages; keep this handoff + README current so the partner can resume. Read existing work before overwriting.
- One stage at a time; run it (use `.venv/bin/python`); verify the checkpoint; report the REAL numbers; only then proceed. Claude Code can run Python here - verify, don't just write and hope.
- If a gate fails, document it here honestly and adjust WITHIN the project (fall back to the developmental-level estimator) - don't silently paper over or abandon.
- **Maintain the Issue & Decision Log (Section 0c).** Every time you find a confound/bug/quirk or resolve one, add or update an ISS-## entry (dated, with evidence). This is our experimental-design paper trail for ISEF judges - never delete resolved entries, mark them RESOLVED; fold the one-line conclusion into 0b.
- Data ethics: CHILDES transcripts are de-identified minors' language data under TalkBank terms. Keep in `data/` (gitignored); DON'T commit raw data; follow TalkBank terms + citation rules; never attempt re-identification.
- ISEF: human-participants research on a pre-existing public dataset (minors) -> SRC review + forms + an adult sponsor/Designated Supervisor BEFORE formal work. Confirm current-year rules with the Dallas Regional SRC. (The author has a "Data Security during Fair Registration" screenshots folder for this.)

================================================================================
## 3. THE PROJECT, precisely
================================================================================
### Background
Language delay affects ~10-15% of toddlers; early intervention markedly improves outcomes, but screening is often late, subjective, or needs a specialist many families can't easily reach. Child language complexity rises with age extremely reliably (MLU, vocabulary diversity, syntactic complexity) - the bedrock signal.

### Target (two layers, deliberately)
- **CORE (de-risked, guaranteed):** estimate developmental LEVEL from a transcript - predict developmental age/stage from language features; quantify a child's gap from age-expected norms.
- **HEADLINE (signal-gated, looks viable per Section 0b):** a language-DELAY FLAG - classify typically-developing vs. language-impaired (ENNI TD vs SLI), child-independent, artifact-controlled. If it validates -> headline. If it weakens under rigor -> fall back to the CORE (still complete + useful). Anti-DopaLoop firewall.

### Why it's genuinely BEHA
Developmental Psychology / language acquisition = core Behavioral & Cognitive Sciences; developmental + accessible-clinical tools are judge-favored.

### Honest positioning (not novel; strong via execution)
Automatic language-level estimation and SLI detection from CHILDES exist in the literature (some with a known transcript-length ARTIFACT critique - which is why we control for it and why ENNI's matched design matters). Our contribution = an accessible, interpretable screening TOOL + rigorous child-independent, artifact-controlled validation + a demo. The ML is not the novelty; the honest, usable tool for real families is.

================================================================================
## 4. DATA - inventory (what's downloaded) + access + gaps
================================================================================
### In hand (in `data/childes/`, gitignored) - ~575 transcripts total
- **Brown/** (Adam, Eve, Sarah) - typically-developing, LONGITUDINAL. 214 files, ages 18-62 months. NOTE: only 3 distinct children (many sessions each) -> too few DISTINCT children for robust child-independent generalization on the toddler range by itself.
- **ENNI/TD/** and **ENNI/SLI/** - Edmonton Narrative Norms Instrument. TD 286 files, SLI 75 files, ages ~85 months (~7 yr), matched across groups, SAME narrative task. Many distinct children. This is the delay-layer gold: TD vs SLI on identical task at matched age.
### Recommended additional downloads (to strengthen the CORE across ages + more children)
- More Eng-NA typically-developing corpora with many children spanning ~18-72 months (e.g., a large multi-child corpus) - to give the level estimator enough DISTINCT children for honest child-independent CV across the young range. (Brown alone = 3 kids; add more.)
### Access
Free TalkBank account -> download corpus zips -> unzip into `data/childes/`. Clinical corpora (ENNI etc.) may require a higher access tier / permission request.
### ISEF paperwork
De-identified, public, pre-existing dataset of MINORS -> human-participants pre-existing-dataset path; needs SRC review + adult sponsor. Confirm with Dallas Regional SRC.

================================================================================
## 5. BUILD PLAN - stages with checkpoints (verify each BEFORE the next)
================================================================================
| Stage | What | Status / checkpoint |
|---|---|---|
| 0 | Access: register TalkBank; download Brown + ENNI (+ more TD corpora). ISEF SRC forms + sponsor. venv. | DONE (Brown+ENNI in hand; more TD corpora recommended). |
| 1 | Ingest + SIGNAL-GATE: parse transcripts; per-file features [child_id, corpus, group(TD/SLI), age_months, mluw, mlum, ttr, ipsyn, n_utterances, vocab...]; confirm core (age->complexity) + delay (TD vs SLI) signals, child-independent. | DONE - both signals confirmed (Section 0b). See src/stage1_check.py. |
| 2 | **CORE deliverable:** developmental-LEVEL estimator - predict developmental age/level from features on combined TD corpora; **GroupKFold by CHILD**; baselines (predict-mean) + bootstrap CIs + calibration; define an "age-expected vs observed" gap score. | DONE (see 0b Stage 2 RESULTS): Ridge MAE 11.46 mo, R2 0.74, child-independent, beats baseline (27.77). Age-gap score works; SLI preview d~0.98. More TD corpora in 18-48mo range still recommended (only 3 Brown children there). |
| 3 | **DELAY classifier + gate:** TD vs SLI (ENNI), child-independent, **artifact-controlled** (adjust/match for n_utterances/transcript length; MLU is per-utterance so robust; resolve the IPSyn anomaly). Balanced metrics (286 vs 75). | If a genuine delay signal survives rigor -> headline. Else -> fall back to Stage 2 core (documented). |
| 4 | Interpretability: SHAP -> which markers drive level/delay -> actionable readout ("flag: MLU + syntax below age norm"). | Ranked drivers + readout logic. |
| 5 | Deployable demo/tool: paste a child's transcript -> estimated level + age-gap + delay flag (if validated) + confidence + drivers. | Working demo (notebook / lightweight web app). |
| 6 | Rigor + board-ready summary: child-independent results, calibration, CIs, subgroup checks (age bands, sex, corpus), artifact controls, honest limitations, Claim-Evidence-Reasoning + poster. | Final results + poster. |

================================================================================
## 6. WHY IT WINS (map to real winners) + HONEST CEILING
================================================================================
- ArabLexify (BEHA placement): accessible screener for an underserved group -> FirstWords is an accessible language-delay screen. Same shape.
- Developmental + clinical + accessibility + interpretability = a resonant, judge-favored BEHA story.
- Honest ceiling: genuine BEHA placement/finalist at Dallas Regional -> ISEF with strong execution + demo + presentation. Not a guaranteed top prize. But it will COMPLETE and be real.

================================================================================
## 7. REACH GOALS (only after the core is solid; each signal-gated)
================================================================================
- Multilingual screening (CHILDES has 20+ languages) - equity/accessibility headline (screening where English tools don't exist). Strong ArabLexify-style angle.
- Speech -> transcript front-end (ASR on child speech) for "record and screen" accessibility - HARD (child ASR is poor); reach only.
- Longitudinal trajectory modeling (Brown/ENNI have repeated sessions).
- Do NOT add unvalidatable outputs; do NOT collect new human data.

================================================================================
## 8. REPO STRUCTURE + HOW TO RUN + RESTART
================================================================================
```
.
|- README.md               overview + quickstart
|- PROJECT_HANDOFF.md       THIS file (read first)
|- requirements.txt         pinned deps (verified py3.14)
|- src/
|   |- stage1_check.py            Stage 1: parse real data + verify core & delay signals
|   |- stage2_level_estimator.py  Stage 2: child-independent level estimator (DONE) + age-gap + figures
|   |- verify_pipeline.py         no-download smoke test of the CHAT->features pipeline
|- data/childes/            (gitignored) Brown/, ENNI/{TD,SLI}/, + more TD corpora
|- results/                 (gitignored contents) features.csv, stage2_calibration.png, stage2_gap_preview.png
|- .venv/                   (gitignored) Python 3.14 env
```
Run: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`, download corpora into `data/childes/`, then `.venv/bin/python src/stage1_check.py` and `.venv/bin/python src/stage2_level_estimator.py`.
### DATA LAYOUT a collaborator must reproduce (data/ is gitignored - each person downloads their own from TalkBank)
```
data/childes/
  Brown/{Adam,Eve,Sarah}/*.cha          # TD longitudinal, 3 children, 214 files, 18-62 mo
  ENNI/TD/{A,B}/*.cha                    # TD narrative, 286 distinct children, 48-120 mo
  ENNI/SLI/{A,B}/*.cha (+ SLI/A/0noaudio/)  # SLI narrative, 75 distinct children
```
Code resolves paths relative to repo root and reads whole corpus dirs recursively (`read_chat(dir, strict=False)`), so exact subfolders don't matter as long as Brown/ and ENNI/{TD,SLI}/ exist. If your numbers differ from Section 0b, your download differs - reconcile before building on it.
### Restart (fresh session or partner's agent)
- Read Sections 0, 0b, 0c, 1, 2 in that order. Committed project = FirstWords. Data + both signals verified; Stage 2 (level estimator) DONE and verified (0b Stage 2 RESULTS). Section 0c is the running issue log - skim it to see what's already been caught and resolved before re-investigating anything.
- NEXT concrete step = Stage 3 (delay classifier, TD vs SLI on ENNI, age + transcript-length covariates, balanced metrics, child-independent, CIs). Also still recommended: download 1-2 more Eng-NA TD corpora with many children at 18-48 mo (Brown alone is 3 children there) and re-run stage2.
- Keep the discipline that finally works: signal first, child-independent validation, baselines, honest CIs, artifact control, one checkpoint at a time, a usable tool + demo, honesty over hype.
- History/fallbacks: DopaLoop lives at /Users/mihir/DopaLoop (MATLAB; do not touch). The ParkinTrack (PD) and ALS-Forecast explorations were removed from this repo (kept only as LESSONS in Section 1) - ALS-Forecast on PRO-ACT is the strongest non-BEHA fallback (TMED); choices13k (decision-making) and WESAD (stress) are BEHA fallbacks with instant data.
