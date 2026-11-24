# FirstWords - PROJECT HANDOFF & COMPLETE PLAN
### An interpretable tool that estimates a child's language-development level (and flags likely delay) from a speech sample
_Self-contained record. Any agent (including a collaborator's agent) MUST read Sections 1-2 before touching the project._
_Updated 2026-10-26 (final board v2 built in Canva; browser web demo deployed; October robustness + fairness checks; READ SECTION 0d FIRST for the current state). Category: ISEF Behavioral & Cognitive Sciences (Developmental Psychology)._

================================================================================
## 0. TL;DR
================================================================================
- **Project:** FirstWords - takes a short transcript of a young child's speech, estimates their **language-developmental level** vs. age norms, and **flags likely language delay**. An accessible, interpretable early-screening aid (early intervention greatly improves outcomes). Analogy for judges: a "growth chart for language."
- **True BEHA fit:** measures a developmental/cognitive construct (language acquisition) = Developmental Psychology. (We rejected an ALS idea because it was Translational-Medical, not BEHA.)
- **Two-layer, de-risked design:** (1) CORE = developmental-level estimator (textbook-certain signal -> cannot dead-null; a complete project by itself); (2) HEADLINE = delay flag (typically-developing vs. language-impaired), signal-gated with a clean fallback to the core.
- **STATUS (2026-08): Stages 1-5 DONE; ~13 analyses; core HARDENED; TWO external validations; ACCESSIBLE RECORD-AND-SCREEN DEMO APP built; board restructured.** Core level estimator MAE 9.99 mo, R2 0.81 on **335 distinct children**. Delay flag ROC-AUC 0.861 (age/length proven NOT the signal); externally validates with NO refit on **Gillam (US, age-matched) AUC 0.80-0.82** + Conti4 (UK, out-of-range) 0.70 (held-out/locked-box). Interpretable features BEAT a black-box TF-IDF text model (0.86 vs 0.82) and a naive rule (`ext_benchmark.py`). Multilingual LEVEL works in German (MAE 3.1) + Cantonese (6.5); cross-lingual DELAY (German) age-matched AUC 0.90; clinical-utility decision curve positive. TYPOLOGY finding was TESTED + RETRACTED (Mandarin broke it; ISS-16) - do NOT headline it.
- **NEW - ACCESSIBILITY FRONT-END (`src/demo_server.py`, `src/audio_screen.py`): a working local demo app.** Open http://localhost:5050, drag a .cha transcript (full model) OR record a child talking (reduced model, faster-whisper transcribes), click Analyze, get the styled report. REMOVES the biggest limitation (needs a researcher-made transcript) - the equity headline. ENABLER: the reduced model (word features only, ASR-derivable) scores AUC 0.857 vs full 0.861. Clean-speech feature recovery within ~5-9%. HONEST: audio path is a PROTOTYPE; validating on real child recordings is data-limited (ENNI/Gillam share no audio) and real child ASR is hard - present as prototype + honest future work, NOT validated.
- Remaining = fill the board content/figures (Canva "Science fair poster board" already restructured into a numbered ISEF flow), poster polish, genuine command of the material + honest AI-use disclosure. RESEARCH is essentially complete + strong.
- **STATUS (2026-10-26) - SUPERSEDES THE LINES ABOVE: research is CLOSED (no more extensions); the board (Canva page 1) is fully laid out with figures and draft text that the team is rewriting; a browser-based WEB DEMO is deployed; the remaining work is the team's own board wording + interview preparation. See Section 0d.**
- **Data:** CHILDES (TalkBank). Downloaded: **Brown** (typically-developing) + **ENNI** (typically-developing AND language-impaired, same narrative task). Free TalkBank login required to download; de-identified.
- **Stack (verified, Python 3.14 venv):** pylangacq 0.23 (CHAT parser; built-in MLU/IPSyn/TTR/ages), pandas, numpy, scikit-learn, shap, matplotlib. See requirements.txt.
- **Honest ceiling:** genuine BEHA placement/finalist potential at Dallas Regional -> ISEF with strong execution + a live demo + the "helps kids" story. Not a guaranteed top prize; but it WILL complete and be a real, correctly-categorized, working tool.

================================================================================
## 0d. CURRENT STATE - OCTOBER 2026 (read this first; it supersedes older planning text below)
================================================================================
**Framing (decided by the authors).** The whole project is centered on ONE story: (1) Developmental Language Disorder is common (~7%, ~2 per classroom) and mostly missed; (2) screening is late, subjective and hard to reach, especially for marginalized and underserved families; (3) we built an accurate, explainable screen that needs only ~10 sentences or a recording; (4) we proved it is real and holds up. Everything on the board serves that story; side findings stay off the board and are kept for the interview. The authors explicitly REJECTED: a human-rater experiment, any human-subjects data collection, and a longitudinal-prediction extension. **Research is closed** - remaining points come from the team's own board wording and the interview (25 of 100 points).

**Board (Canva design "Science fair poster board", id DAHBa6WexdI).** Page 1 = the current board ("Board v2 layout", 48x36 in, 4608x3456 px). Page 2 = the older board draft. Page 3 = last year's sleep project (reference only, never print). Layout, 3 columns: [1 Problem, 2 Question & Hypotheses, 3 Data (fig8), 4 Methods (2 bullets + fig10 validation diagram)] | [At a Glance pipeline strip, 5 Key Results (hero fig1 + 4 big-number tiles + fig2 ROC + fig3 growth chart + fig7 language age), 6 Screen From a Recording (PROTOTYPE): demo screenshot + fig9 stress test + "0.857 vs 0.861" tile] | [7 Does It Hold Up (fig4 outside labs + fig5 black-box benchmark), 8 Conclusion, 9 Scope & Next Steps, Selected References]. The draft text (bullets) was AI-drafted; **the team must rewrite all board text in their own words** (ISEF bars AI-written posters - see Section 2). Open items flagged to the team: (a) the team's title "...Validated Across Labs, Ages, and Ethnic Groups" is CONTRADICTED by ISS-17 - drop "and Ethnic Groups" (suggested: "Detecting Developmental Language Disorder From Brief Speech Samples: An Accessible, Interpretable Screen Validated Across Labs and Ages"); (b) the team's data line says "400+ children" - use "1,100+ children across 6 public datasets, plus German and Cantonese"; (c) the core question text box appeared duplicated in one render; (d) optional extra visuals: a "height chart yes / language chart no" icon pair and a qualitative "today vs. FirstWords" access-path graphic for the Problem box, then cut that paragraph to ~3 lines. Display rules: no URL/QR code on the board, the word "abstract" cannot be a title, acknowledgments restricted, every figure needs a credit line.

**Board figures** (`results/poster/`, regenerate with `.venv/bin/python src/poster_figures.py` and `src/poster_figures_extra.py`; every number recomputed from the pipeline; `results/poster/numbers.txt` lists them): fig1 artifact control (hero), fig2 ROC + screening point, fig3 growth chart, fig4 outside-lab validation, fig5 benchmark, fig6 short sample (now OFF the board, replaced by fig9), fig7 language age, fig8 data overview, fig9 transcription stress test, fig10 validation-design diagram. All print-sized with labels >= 24 pt.

**What is ON the board vs. kept for the interview.** ON: prevalence/underdiagnosis problem; question + 4 hypotheses; 1,100+ children; 4 measures + 2 simple models + child-independent testing diagram; artifact control (0.86 vs age 0.42 / length 0.49); ROC 0.86 [0.81-0.91], 91% caught on held-out ENNI children; growth chart; language age MAE 10 mo (R2 0.81); 782 outside children (Gillam 0.80, Conti-Ramsden 0.70); as accurate as a 3,000-feature black box; 10 sentences enough (0.82); words-only model 0.857 vs 0.861; stress test; reliability 0.90 (in Methods). INTERVIEW ONLY (do not add to board): calibration/Brier, decision curve, SHAP, grammar-marker profile, sex/age subgroup check, ethnicity audit + fix test (ISS-17), cutoff transfer (ISS-18), real-world PPV/NPV, published-screener ranges, German/Cantonese details, self-caught data leak (ISS-07), retracted typology (ISS-16), Spanish data gap, failed unified cross-language model.

**Web demo (a prototype that runs in a browser on the laptop - do NOT call it a phone app).** Live at https://web-pearl-sigma-14.vercel.app (Vercel project "web", account mihirbr-6391; never put the URL on the board). Code in `web/` (index.html, app.js, model.json). Everything runs client-side: .cha transcripts are parsed in JS; typed sentences work; recordings are transcribed in the browser by Whisper base.en (transformers.js 3.8.1, onnx-community/whisper-base.en_timestamped, 8-bit, ~76 MB served from the site itself under web/models/ - fetch with `scripts/fetch_web_model.sh`, gitignored); nothing is uploaded. Model = words-only screen (sentence length + vocabulary diversity + age) exported by `src/export_web_model.py`, which trains on its OWN CHAT parser (mirrored exactly in app.js; agrees with pylangacq r=0.996 for mluw, 0.987 for mattr50) -> child-independent AUC 0.861 [0.81-0.91] on 345 ENNI children, cutoff 0.34 (sens 0.91, spec 0.62); delay flag shown only for ages 4-10; implausibly long utterances -> "could not screen". JS and Python outputs verified identical on the 4 demo transcripts. Record tab has two read-aloud demo scripts (Script A on track -> not flagged, language age ~79 mo at age 6; Script B short, simple, fully standard sentences -> flagged, ~50 mo), verified end-to-end with synthetic speech; normal adult speech is correctly refused. Deploy: `npx vercel@64.0.1 deploy web --prod` (after `scripts/fetch_web_model.sh`). Local test: serve the repo root (`.claude/launch.json` "web-test", port 8765) and open /web/index.html. Backup offline demo: `src/demo_server.py` (localhost:5050) - NOTE it still uses the full model and a 0.45 cutoff, so its flags can differ slightly from the web demo. Judging plan + pitch: `docs/PITCH.md`.

**October 2026 results (all reproducible; scripts in src/).**
- `ext_asr_stress.py` (CLEAR, on board): train on clean ENNI, test on corrupted held-out children. AUC 0.861 clean -> 0.846 with 40% of words wrong; 0.82-0.83 with 30% of sentence breaks also missed. Results in results/asr_stress.csv -> fig9.
- `ext_reliability.py` (CLEAR, on board): split-half (odd/even utterances) risk-score Spearman r 0.82, Spearman-Brown 0.90; same flag for 84% of children (kappa 0.67); first/second half 0.77 / 0.87 / 78% / 0.57; each half alone AUC 0.83-0.85.
- `ext_fairness_ethnicity.py` (FINDING, interview only): ISS-17.
- `ext_fairness_fix.py` (NOT CLEAR, interview only): dropping vocabulary diversity (sentence length + age only) keeps AUC (0.80 vs 0.81 on Gillam) and shrinks the ethnic false-alarm gap by 7-10 points in every comparison, but intervals are borderline (only Hispanic at 80% sensitivity excludes 0).
- Cutoff transfer: ISS-18. Real-world numbers at 7% prevalence (sens 0.91 / spec 0.65): PPV ~16%, NPV ~99%, ~39% of a class flagged - interview prep only.
- Published screeners (Berkman et al., 2015, Pediatrics; USPSTF evidence review): sensitivity 50-94%, specificity 45-96% across 23 studies -> ours is "comparable to published screeners" (safe one-liner).

================================================================================
## 0b. VERIFIED FACTS (established by running real checks - trust these)
================================================================================
### Signals (from `src/stage1_check.py` on the real data, 2026-07-14)
- **CORE signal CONFIRMED (Brown, typically-developing):** 214 transcripts, ages 18.0-62.4 months. Language complexity rises with age: corr(age, MLU-words)=0.68, MLU-morphemes=0.69, TTR=0.62, IPSyn=0.53. A multivariate model will do better. => the developmental-level estimator will work.
- **DELAY signal looks REAL (ENNI, TD vs SLI, AGES MATCHED ~85 mo):** TD n=286, SLI n=75. At the same age, language-impaired children have much lower MLU: words 6.90 (TD) vs 5.67 (SLI); morphemes 7.09 vs 5.81 - roughly a 1 SD gap on the SAME narrative task (so not an age or task artifact). TTR 0.37 vs 0.35 (small).
### 2026-07-16 audit (full re-run; every 0b number reproduced exactly; conclusions below - see Section 0c for the full issue write-ups & evidence)
- **`syntax_index` (pylangacq ipsyn) is NOT the published IPSyn scale** - keep as a feature, never call it "IPSyn." [ISS-01]
- **The IPSyn "anomaly" (SLI > TD) was a transcript-length artifact** - drop/length-correct it for TD-vs-SLI; MLU is length-robust and does the discriminating. [ISS-02]
- **ENNI A/B folders SHARE a few children** (~7; ISS-03's "disjoint" claim was wrong) -> child_id must come from the CHAT header fingerprint (birthdate+sex), not the filename; split with GroupKFold/StratifiedGroupKFold. Impact tiny but now airtight. [ISS-07 corrects ISS-03]
- **ENNI ages span 48-120 mo, only MEAN-matched at ~85 mo** -> include age as a covariate (done in Stage 3; age-alone AUC 0.42 = not the signal). One TD file lacks an age header (285/286 aged). [ISS-04, ISS-06]
- **Raw TTR is length-confounded** -> added MATTR-50 (`mattr50`), length-robust. [ISS-05]
- **Core signal is stronger WITHIN child than pooled:** Brown corr(age, MLUw) = 0.90 (Adam), 0.81 (Eve), 0.88 (Sarah) vs 0.68 pooled -> genuinely developmental, not a pooling artifact.
- **ENNI TD doubles as level-estimator data:** 286 distinct TD children at 48-120 mo, overlapping Brown at 48-62 mo -> extends the core across ages; task differs (narrative vs home conversation), so control/report corpus.
### Stage 2 RESULTS (updated 2026-07-20 after Move A: added Bates + Gleason toddler corpora)
- TD modeling set NOW: 669 files, **335 distinct children**, 18-120 mo (ENNI 285, Brown 214, Bates 100, Gleason 70). **Distinct children <48 mo went from 3 (Brown only) -> 48** = the old weak spot is FIXED. Features: mluw, mlum, ttr, mattr50, syntax_index. GroupKFold by child; cluster bootstrap CIs resample CHILDREN.
- **Ridge (interpretable primary): MAE 9.99 mo [95% CI 9.11..11.26], R2 0.81 [0.77..0.83]** (improved from 11.46 / 0.74 pre-Move-A); grad-boosting marginally better (MAE 9.34, R2 0.81); baseline 24.48. Per-corpus OOF MAE: Bates 2.6 (20-28mo), Brown 7.65, Gleason 7.43, ENNI 13.4 (older = flatter growth = harder, expected + cross-linguistically consistent).
- Age-gap score = predicted language age - chronological age; TD OOF gap ~centered.
- SLI preview (informal): SLI gap ~-16 mo vs TD ~0 -> ~1.3 yr lag, d~1.0. (Rigor = Stage 3.)
- Outputs: results/features.csv (746 files), stage2_calibration.png, stage2_gap_preview.png. NOTE: history pre-Move-A was MAE 11.19/R2 0.76 on 281 children - Move A is a real improvement, not a restatement.
### Stage 3 RESULTS (2026-07-18, `src/stage3_delay_classifier.py`, ENNI-only, child-independent)
- Set: 360 ENNI files, 345 distinct children, TD 285 vs SLI 75 (20.8% prevalence). ENNI-only (Brown excluded to avoid a task/age shortcut). Length-robust features (mluw, mlum, mattr50) + age covariate. StratifiedGroupKFold by header child_id (ISS-07).
- **Logistic regression WINS (interpretable): ROC-AUC 0.861 [95% CI 0.811..0.907], PR-AUC 0.639 [0.530..0.748]** (baseline prevalence PR-AUC = 0.21) vs grad-boosting AUC 0.793.
- **Artifact check (the money slide): age-alone AUC 0.42, transcript-length-alone AUC 0.49 (both ~chance) -> the signal is genuinely LINGUISTIC, not a confound.** Language-only 0.78; +age 0.86.
- Screening operating point (~90% sensitivity): thr 0.36 -> sens 0.91, spec 0.65 (misses 7 of 75 SLI). Youden point: sens 0.88, spec 0.74.
- Drivers (standardized logistic coefs; + => more likely impaired): mlum -1.39 (low morpheme MLU), age +1.35 (older but still low = flag), mattr50 -0.84 (low lexical diversity), mluw -0.38. All developmentally sensible.
- Output: results/stage3_roc.png (ROC + risk-score histograms). **DELAY HEADLINE SURVIVES RIGOR -> it IS the headline (not a fallback).**
### Stage 4 RESULTS (2026-07-18, `src/stage4_interpretability.py`, SHAP + age-normed readout)
- Global SHAP (exact, LinearExplainer) - LEVEL estimate: dominated by sentence length (words/utt), then MATTR-50 & morphemes/utt. DELAY flag: top |SHAP| = chronological age, then morphemes/utt, then MATTR-50.
- **IMPORTANT nuance (judge-ready):** age has the HIGHEST SHAP importance for the delay flag, but age ALONE cannot classify (AUC 0.42, ISS-04). Reconciliation: age is the NORM REFERENCE - it tells the model what MLU/vocab to EXPECT at that age; the flag fires when language falls short of that expectation. So SHAP-important != a shortcut; the two facts together are the story ("age frames the norm, language triggers the flag").
- Built the demo's engine: `readout(row, level_model, delay_model, norms)` returns language age, age-gap, delay probability, and each marker as a z-score vs AGE-MATCHED TD peers ("morphemes/utt -2.2 SD for age"). Example SLI child: chrono 64 mo, est. language age 38 mo, p=0.99, sentence length -2.3 SD -> "discuss an evaluation." Example TD child: within age norms, p=0.03.
- Outputs: results/stage4_shap_level.png, results/stage4_shap_delay.png. `age_norms()`/`readout()` are imported by Stage 5.
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

- **ISS-03 | 2026-07-16 | CORRECTED by ISS-07 | Are ENNI's A/B story-set folders the SAME children twice?**
  - Saw: ENNI is split into story-form A and B folders per group; needed to know if a child appears in both.
  - Why it matters: if A and B share children, a naive by-file split leaks a child across train/test = inflated results.
  - Did/found (INITIAL, later shown insufficient): file-NAME intersection between A and B is zero, so I concluded "disjoint children, each file = one child." **This method was wrong** - filenames are per-narrative, not per-child. See ISS-07 for the correct analysis and fix. Kept here (not deleted) as an honest record of a method error we caught ourselves.
  - Evidence: `comm -12` on the A/B file lists (necessary but insufficient test).

- **ISS-16 | 2026-07-20 | RESOLVED (self-retraction) | The "marker depends on language typology" finding did NOT replicate.**
  - Saw: with 3 languages, isolating Cantonese used vocabulary diversity while inflected English/German used sentence length - looked like a typology law, and we (briefly) reframed the poster around it.
  - Tested it (the right move): added Mandarin (2nd isolating) + Turkish (agglutinative). MANDARIN CONTRADICTS Cantonese (Mandarin is length-dominant, lex-dom -0.35). Two isolating languages disagree.
  - Diagnosis: the marker-dominance is CONFOUNDED WITH TASK (Cantonese=free-play->diversity; Mandarin/English=narrative->length), not typology, across cross-corpus data we can't disentangle. RETRACTED the typology claim; do NOT headline it. Kept the narrower defensible point (no universal-best marker -> per-language/task calibration needed). `src/ext_typology.py`. This is a credibility POINT (we tested and retracted our own hypothesis) - the opposite of overclaiming.
  - FOLLOW-UP CHECKED (so no one re-mines it): tested whether TASK cleanly drives marker-dominance within English (narrative ENNI/Gillam vs conversation Brown/Gleason, common 48-72mo). It does NOT - the two narrative corpora disagree (ENNI -0.29 vs Gillam +0.27) and the two conversation corpora disagree. CONCLUSION: marker-dominance is corpus-specific NOISE; there is NO robust "which marker dominates" finding (typology OR task) in this data. STOP chasing this angle - further mining = another retraction risk. The novelty is not here; do not manufacture it.
- **ISS-15 | 2026-07-20 | RESOLVED (rigor win) | Group AGE CONFOUND in the German cross-lingual delay test.**
  - Saw: German TD-vs-CI (cochlear-implant) delay classifier scored AUC 0.99 (full model) - suspiciously high. Artifact check: age-ALONE AUC 0.92 -> the CI cohort is systematically older (followed to later ages), so the classifier was mostly detecting AGE, not language.
  - Did: report the AGE-MATCHED (21-44 mo overlap) language-only AUC = 0.903 [0.837-0.957] as the honest headline (residual age-only 0.71 in that window; language clearly exceeds it). Never report the 0.99. This is the SAME artifact-control discipline as Stage 3 (age-alone must be checked), applied cross-lingually - and a poster-worthy example of catching our own confound. `src/ext_crosslingual_delay.py`.
- **ISS-14 | 2026-07-20 | RESOLVED (turned positive) | Multilingual: a DATA-match problem, not a language problem.**
  - Saw: Spanish cross-child estimator kept failing (Shiro plateau corr 0.23; BecaCESNo weak R2 0.23; pooled young corr 0.33) -> looked like the method doesn't transfer.
  - Diagnosis: the failures were DATA (age-plateau / narrow-age / task-heterogeneous), not the METHOD. Proof: within a well-sampled child the premise holds strongly in Spanish (Ornat r=0.77, Marrero/Idaira 0.99); and in GERMAN (Szagun, 22 controls, wide young 15-44 mo, consistent sampling) the full child-independent estimator REPRODUCES: MAE 3.1 mo, R2 0.79.
  - Resolution: multilingual = a genuine WIN (German) + an honest premise-confirmed-but-data-limited result (Spanish, which sharpens the equity story). Superseded Spanish corpora archived to not_in_use/. `src/ext_multilingual.py` now reports within-child corr + cross-child gate; discovery skips not_in_use/.
- **ISS-13 | 2026-07-20 | SUPERSEDED by ISS-14 | (was: "Spanish estimator weak")** - the weakness was a data-coverage artifact, not a method failure; see ISS-14.

- **ISS-12 | 2026-07-20 | RESOLVED | Bates encodes one child under several filenames across task/age conditions.**
  - Saw: child "amy" appears as amy.cha (Free20), amy28.cha (Free28), amy.cha (Snack28), amyst.cha (Story28). Using the raw filename stem as child_id would split one child into four -> child-leak across CV folds.
  - Did: `bates_child_id()` strips trailing digits + task suffixes (_snack/snack/st) to the base given-name, so a child's 20- and 28-month files group together. Gleason needed no fix (same stem across contexts). Verified: 48 distinct children <48mo (sane).
  - Evidence: build_features() in src/stage2_level_estimator.py.

- **ISS-11 | 2026-07-19 | FINDING (report as limitation) | Cross-task screening reliability is POOR out of the trained age range.**
  - Saw: applying the ENNI (4-10yr) flag to Conti4 adolescents (13-16yr) on their TWO tasks (frog narrative vs spontaneous), the per-child risk scores barely agree across tasks: Pearson r=0.11, Cohen's kappa=0.06, 53% flag agreement.
  - Why it matters: a screen must give a consistent answer from different samples of a child; this looks like it doesn't.
  - Diagnosis (honest, not fatal): scores are COMPRESSED out of range (frog p: mean 0.11, max 0.41) because at 13-16yr TD/SLI language converges - the RANKING still partly transfers (AUC 0.70, SLI mean p > TD) but absolute/consistent flagging collapses. Ties to ISS calibration: probabilities aren't trustworthy out of range. IMPLICATION/limitation: use a single standardized, age-appropriate task; measure true test-retest with REPEATED SAME-TASK samples in the 4-10 yr target range (future work). `src/ext_taskreliability.py`.

- **ISS-10 | 2026-07-19 | RESOLVED | Morphological-diversity marker was transcript-length confounded.**
  - Saw: in the item-level grammatical profile, `morph_diversity` (distinct inflection tags / tokens) appeared HIGHER in SLI (d=-0.65) - counterintuitive.
  - Why it matters: same class of artifact as raw TTR (fewer tokens -> higher type ratio); SLI transcripts are shorter, inflating it.
  - Did: EXCLUDED morph_diversity from the marker set and the morphology classifier; kept only length-robust per-verb/per-noun/per-utterance rates. Documented. Strongest honest discriminator is then subordinate-clause rate (d=0.85). `src/ext_morphprofile.py`.

- **ISS-09 | 2026-07-19 | RESOLVED | Leaky within-corpus reference baseline in the level-transfer comparison.**
  - Saw: the "within-corpus ref MAE" in ext_crosscorpus PART A used plain KFold (by file). Brown has only 3 children, so a by-file split leaks the same child across folds and understates the reference error (it read 3.2 mo).
  - Why it matters: an over-optimistic reference makes the cross-corpus transfer look worse by comparison; it's the same leakage class as ISS-07, in a context number rather than a headline.
  - Did: switched the reference to GroupKFold by child_id (min(5, n_children) folds). Brown ref corrected 3.2 -> 5.4 mo (ENNI ref 2.7 -> 2.6, ~unchanged, since ENNI has many distinct children). Conclusion IMPROVES: cross-corpus offset-corrected MAE (6.1) is only ~0.7 mo above the honest within-corpus ref (5.4).
  - Evidence: `GroupKFold` in `src/ext_crosscorpus.py` level_transfer(); re-run logs.

- **ISS-08 | 2026-07-19 | RESOLVED | Conti-Ramsden 4 encodes TD/SLI in FOLDER names, not the @ID header (unlike ENNI).**
  - Saw: the cross-corpus harness assumed the group label sits in the @ID header (true for ENNI). Conti4's @ID group field is empty; the label is in folder names (TD-frog/, SLI-spontaneous/) and the @Types tier.
  - Why it matters: without adapting, the external-validation test would silently find zero labels.
  - Did: `label_from_path()` reads group (SLI/DLD vs TD/control) AND task (frog/spontaneous) from the folder path, with an @Types header fallback; `find_external_corpora()` now scans BOTH data/ and data/childes/. Verified it recovers 99 TD + 19 SLI per task.
  - Evidence: PART B output in `src/ext_crosscorpus.py`.

- **ISS-07 | 2026-07-18 | RESOLVED | ENNI A/B folders DO share some children -> small child-leak in the by-file split (corrects ISS-03).**
  - Saw: building Stage 3, I remembered the ENNI instrument has each child tell BOTH story A and story B, so filename-disjointness (ISS-03) does not prove child-disjointness. Fingerprinting each file by its CHAT header (`Birth of CHI is ...` + sex) shows ~5 TD and ~2 SLI children appear in BOTH folders; one pair (402.cha in A, 404.cha in B) shares birthdate, sex AND exact age = unmistakably the same child. So 286 TD files map to ~279 distinct children, not 286.
  - Why it matters: Stage 2 (ENNI child_id = filename) and the first Stage 3 run (StratifiedKFold by file) each leaked ~7 children across folds -> mildly optimistic metrics.
  - Did: derive child_id from the header fingerprint (birthdate+sex; unique fallback if missing); Stage 2 now GroupKFold-by that; Stage 3 now StratifiedGroupKFold-by that. IMPACT WAS TINY (the honest, reassuring part): Stage 2 MAE 11.46->11.19 mo; Stage 3 ROC-AUC 0.860->0.861 - so the earlier numbers were not meaningfully inflated, but the validation is now airtight and defensible to a judge who knows ENNI's design.
  - Evidence: header-fingerprint overlap script (scratchpad); `enni_child_id()` in src/stage2_level_estimator.py; re-run logs.

- **ISS-04 | 2026-07-16 | RESOLVED | ENNI ages are only MEAN-matched (~85 mo), but individuals span 48-120 mo (sd~20).**
  - Saw: the "ages matched" claim is true for the group means/spread, not per child.
  - Why it matters: for the Stage 3 delay classifier, age is a confound if not handled - an older SLI child could be mistaken for delayed purely by age mixing.
  - Did: Stage 3 includes age_months as a covariate, and the artifact check proves age is NOT the signal (age-alone ROC-AUC = 0.42, i.e. below chance; length-alone = 0.49). The language features carry it (language-only AUC 0.78; language+age 0.86, where age acts as a legitimate age-norm reference, not a shortcut). Stage 2's age-gap score already subtracts chronological age.
  - Evidence: Stage 3 "Artifact check" block.

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

- **ISS-17 | 2026-10-24 | OPEN (known limitation, interview only) | The screen over-flags typically developing African American children on the outside (Gillam) data.**
  - Saw: Gillam records ethnicity in @Comment. Trained on ENNI, ENNI cutoff (90% sensitivity): false alarms among typical children = European American 14% (50/345), African American 31% (17/54), Hispanic 22% (10/45). Black-White gap +17 points [95% CI +5, +31]; Hispanic +8 [-4, +21]. AUC 0.83 / 0.72 / 0.77. Identical for the full and words-only models.
  - Why it matters: the project is framed around equity for marginalized children; the tool must not be described as working equally for all groups, and the title must not claim validation "across ethnic groups".
  - Did: tested the mechanism. My hypothesis (morpheme counting penalizes dialect) was WRONG: morphemes per word are equal across groups (1.028 vs 1.029). The gap is carried by VOCABULARY DIVERSITY (age-adjusted MATTR -0.53 SD for African American, -0.49 for Hispanic typical children; sentence length only -0.27 / -0.16, n.s.); a vocabulary-only screen is near chance for those groups (AUC 0.57 / 0.49). Training on the diverse Gillam sample itself did not shrink the gap. A sentence-length-only screen reduces the gap 7-10 points with equal AUC, but not conclusively (ext_fairness_fix.py). Caveats: Gillam labels come from standardized tests (possibly biased themselves); no SES data to separate dialect from opportunity.
  - Evidence: src/ext_fairness_ethnicity.py, src/ext_fairness_fix.py.

- **ISS-18 | 2026-10-24 | OPEN (known limitation, interview only) | The ENNI-tuned cutoff does not transfer to Gillam.**
  - Saw: Gillam AUC holds (0.80) but the ENNI 90%-sensitivity cutoff catches only ~58-60% of impaired Gillam children (false alarms 17%). Re-choosing the cutoff on Gillam (cross-fitted) reaches 80% sensitivity at ~34% false alarms.
  - Why it matters: the board's "catches 91%" is true for held-out ENNI children only; never apply it to the outside data. Ranking transfers; the cutoff needs local calibration (supports the "one norm doesn't fit all" point).
  - Evidence: ext_fairness_ethnicity.py output ("ALL" rows); scratch fix test.

- **ISS-19 | 2026-10-24 | RESOLVED | Demo Script B used dialect-overlapping forms.**
  - Saw: the first "delayed" read-aloud script used zero copula ("The boy sleeping") and similar forms that are normal in African American English, implicitly equating dialect with delay.
  - Did: rewrote Script B as short, repetitive, fully standard sentences; it still screens as delayed (language age ~50 mo at age 6, flagged) because the words-only model uses length + diversity, not grammar errors. Verified end-to-end in the web demo with synthetic speech.

- **ISS-20 | 2026-10-24 | RESOLVED | Web-demo feature parity with the Python pipeline.**
  - Saw: the browser cannot run pylangacq, so a JS CHAT parser could silently diverge from the training features.
  - Did: `src/export_web_model.py` trains the web model on its own parser (identical rules in app.js); parser vs pylangacq r=0.996 (mluw) / 0.987 (mattr50); words-only AUC unchanged at 0.861; JS == Python exactly on all 4 demo transcripts. Also fixed CHAT ages written as "5;02." (blank day) in the JS age parser.

- **ISS-21 | 2026-10-24 | RESOLVED | Browser speech model downloaded slowly / looked frozen.**
  - Saw: no dtype was set (could fetch ~290 MB full-precision files) and there was no progress display; a user's browser pulled from the Hugging Face CDN at ~0.25 MB/s.
  - Did: explicit 8-bit weights (~76 MB), a live download counter, early download when the Record tab opens, and the model served from the Vercel site itself (web/models/, ~5 MB/s; also avoids networks that block huggingface.co).

- **ISS-25 | 2026-11-24 | RESOLVED (consistency) | Offline backup demo used a different flag cutoff than the board and web demo.**
  - Saw: `demo_report.py` and `audio_screen.py` flagged at risk >= 0.45 (Stage 3's Youden point: sens 0.88, spec 0.74), while the board's "catches 91%" and the web demo use the ~90%-sensitivity screening point (full model 0.36; words-only web model 0.34). A judge comparing the backup demo to the board could see a mismatch.
  - Did: one named constant per path, `SCREEN_CUTOFF` = 0.36 (demo_report, full model) and 0.34 (audio_screen, words-only, same as web/model.json). To revert to Youden, set both to 0.45.
  - Evidence: `src/stage3_delay_classifier.py` operating-point lines (re-run 2026-11-24: AUC 0.862, thr 0.36 -> sens 0.91 spec 0.65; thr 0.45 -> sens 0.88 spec 0.74).

- **ISS-23 | 2026-11-24 | RESOLVED (new evidence) | The "early screen" had no delay validation under age 4.**
  - Saw: the delay flag was trained on ENNI (4-10 yr) and externally tested only on Gillam (5-12) and Conti4 (13-16), yet the project is pitched as an EARLY screen. Obvious judge question: "does it work on toddlers?"
  - Did: downloaded EllisWeismer (Clinical-Eng; late talkers LT vs TD, visits at 30/42/54/66 mo, 130 children, filename stem = child id across visits). `src/ext_young_delay.py` applies the ENNI-trained flag and the Stage 2 level estimator with NO refit; `src/poster_figures_young.py` averages each child's tasks per visit (one score per child) -> fig11.
  - Found: AUC 0.92 [0.87-0.96] at 30 mo, 0.78 at 42, 0.77 at 54, 0.55 (chance) at 66 (language-only model, per child). Within-EllisWeismer child-independent CV reaches about the same at 30 mo (0.89-0.92) -> the transferred model is near the ceiling. Level-gap score gives the same pattern (0.89-0.92 at 30 mo).
  - Caveats: LT is a vocabulary-based label at ~2 yr, NOT SLI/DLD, so 30-mo agreement is partly expected. The fall with age matches the literature (many late talkers catch up); the screen cannot tell which late talkers stay behind. Say "catches late talkers", never "predicts later disorder".
  - Evidence: results/ellisweismer_scores.csv, results/ellisweismer_by_visit.csv, results/poster/fig11_young_children.png.

- **ISS-22 | 2026-10-13 | RESOLVED | ISEF rule: generative AI may not write the poster/abstract/research plan.**
  - Saw: Society for Science "Rules for All Projects": AI may be used as a resource but must be cited; students cannot use generative AI to write the research plan, abstract or poster; inappropriate AI use can fail a project.
  - Did: the board's AI-drafted bullets are placeholders the team is rewriting in their own words; References includes an AI-assistance citation; figures carry a credit line. Keep this in mind for any future board/abstract text.

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
- **AI-USE INTEGRITY (critical - this repo was built with heavy AI-agent involvement).** This handoff and the code were produced with substantial AI assistance (an AI wrote most of the code and contributed to the statistical design). ISEF/affiliated fairs require HONEST disclosure of AI use AND that the work is genuinely the student's own. THE ONLY SAFE PATH IS HONESTY + REAL UNDERSTANDING: (1) fill out the fair's AI-use disclosure completely and truthfully; (2) the student must be able to EXPLAIN, DEFEND, RE-DERIVE, and ideally RE-IMPLEMENT every choice from their own understanding (use docs/EXPLAINER.md to LEARN it, then do independent work: re-run/verify analyses, extend something yourself, read every line). A rehearsed spin that misrepresents authorship is disqualifying if caught and is NOT worth it. Judges will ask "how much did you write vs. AI?" - the credible answer is an honest division of labor (student: conception, direction, decisions, data, interpretation; AI: implementation + methods support) PLUS demonstrated command of the material. Do not attempt to hide the AI involvement. **ISEF RULE (verified 2026-10, ISS-22): generative AI may NOT write the poster, abstract or research plan; AI used as a resource must be cited.** Any board/abstract text an agent drafts is a placeholder the team must rewrite in their own words.

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
### In hand (in `data/childes/`, gitignored). ACTIVE corpora + roles:
CORE TD level estimator (Stage 2): Brown + ENNI-TD + Bates + Gleason (335 distinct children, 18-120 mo).
DELAY layer (Stage 3): ENNI TD vs SLI (train). External delay validation (Section 10 PART B): Gillam + Conti4.
MULTILINGUAL (Section 10): Szagun(German) + Ornat/Marrero/FernAguado(Spanish).
- **Brown/** (Adam, Eve, Sarah) - TD longitudinal, 214 files, 18-62 mo, 3 distinct children. Core TD.
- **ENNI/{TD,SLI}/** - Edmonton Narrative Norms Instrument. TD 286 + SLI 75, ~4-10 yr, matched, SAME narrative task, many distinct children. Delay-layer GOLD + core TD.
- **Bates/** - English TD, ~28 children at 20 & 28 mo (folders=conditions; child_id via bates_child_id()). %mor. Core TD (Move A).
- **Gleason/** - English TD, ~24 children, 25-62 mo (folders=contexts; child_id=stem). %mor. Core TD (Move A).
- **Gillam/** - English clinical, 493 TD + 171 SLI, ages 5-12 yr, narrative, ~1 file/child (folders TD/SLI). External delay validation, AGE-MATCHED to ENNI (the CLEAN in-range test, AUC 0.80-0.82). Not in core train.
- **Conti4/** - Conti-Ramsden 4, British, 99 TD + 19 SLI adolescents 13-16 yr, frog+spontaneous. External delay validation, OUT-of-range stress test (AUC 0.70). Not in core train.
- **Szagun/{TD,CI}/** - GERMAN (deu), Szagun. TD = 22 normal-hearing controls, 15-44 mo, child-name folders, %mor. Multilingual WIN: level estimator MAE 3.1 mo, R2 0.79. (CI = cochlear-implant group, unused. One 368-mo outlier file auto-filtered.)
- **Ornat/** - SPANISH, 1 child (Maria) densely sampled 19-46 mo, 114 files (flat, age-coded names; SINGLE_CHILD in ext_multilingual). Multilingual premise: within-child corr 0.77.
- **Marrero/** - SPANISH, 3 children (folders), 27-59 mo. Multilingual premise: within-child up to 0.99.
- **FernAguado/** - SPANISH, 47 children (name folders), 34-50 mo (narrow!), longitudinal, multiple tasks. Multilingual: many kids but no age spread -> shows the cross-child DATA LIMITATION.
- **HKU/** - CANTONESE (yue), HKU-70. 70 TD children, ages 2-5 yr (folders 2/3/4/5 = AGE bins, so child_id=filename stem), free-play, %mor. Multilingual WIN (isolating language): LEVEL estimator MAE 6.5 mo, R2 0.55 via vocabulary diversity (MLU-words weak here).
- **EllisWeismer/{LT,TD}/<visit><task>/** - (added 2026-11) English late talkers vs TD, 130 children, visits 30/42/54/66 mo, tasks ec/pc/int/conv. Toddler delay validation (ISS-23). Not in core train. Download path: Clinical-Eng/EllisWeismer (the zip also holds a 0phon/ copy; skip it).
- **Providence/<Child>/*.cha** - (added 2026-11) English TD, 6 children ~1-3 yr, time-aligned transcripts. Audio for one ~33-mo session per child (5 children; Ethan's media is not served) extracted to `data/providence_audio/*.wav` (16 kHz mono) from media.talkbank.org/phon/Eng-NA/Providence/CHAT/<Child>/<session>.mp4. Real child-speech ASR test (ISS-24). Download path for transcripts: talkbank.org/data/phon/Eng-NA/Providence?f=zip (CHAT/ subfolder).
- **Szagun/CI/** - GERMAN cochlear-implant group (22 children, 21-89 mo), language-delayed. Used in the CROSS-LINGUAL DELAY test (TD vs CI, age-matched AUC 0.90). Not in the level-estimator set.
### ARCHIVED (`data/childes/not_in_use/`) - superseded/exploratory, NOT read by any script (discovery skips not_in_use/):
- **Shiro/** (Spanish, 113 kids, 6-11 yr) - age-plateau, gate failed. **BecaCESNo/** (Spanish, 70 kids, 3-12 yr) - heterogeneous, weak (R2 0.23). **DiezItza/** (Spanish, 20 kids, age 3) - redundant. All superseded by the within-child + German multilingual analysis; kept for provenance.
### Recommended additional downloads (to strengthen the CORE across ages + more children)
- **Move A (fix the young-children gap) - DONE 2026-07-20.** Added Bates + Gleason (both %mor present). Result: distinct children <48 mo went 3 -> 48; Stage 2 MAE 11.19 -> 9.99, R2 0.76 -> 0.81. Bates OOF MAE 2.6 mo. The core's biggest weakness is essentially removed. (Still could add more if a large continuous 18-48mo multi-child corpus appears, but the gap is now adequately covered.)
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
| 3 | **DELAY classifier + gate:** TD vs SLI (ENNI), child-independent, **artifact-controlled** (adjust/match for n_utterances/transcript length; MLU is per-utterance so robust; resolve the IPSyn anomaly). Balanced metrics (286 vs 75). | DONE (0b Stage 3 RESULTS): logreg ROC-AUC 0.861, child-independent, age/length proven NOT the signal. **Signal SURVIVED rigor -> delay flag is the headline.** |
| 4 | Interpretability: SHAP -> which markers drive level/delay -> actionable readout ("flag: MLU + syntax below age norm"). | DONE (0b Stage 4 RESULTS): SHAP global drivers + age-normed per-child readout (`readout()`), figures. |
| 5 | Deployable demo/tool: paste a child's transcript -> estimated level + age-gap + delay flag (if validated) + confidence + drivers. | DONE: `src/stage5_demo.py` (--demo / --file X.cha / stdin). Reuses Stage 4 readout + percentile. Prints language age, gap, percentile, flag prob, below-norm drivers, non-diagnostic recommendation. |
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
- Speech -> transcript front-end (ASR on child speech) for "record and screen" accessibility. **WORKING PROTOTYPE built 2026-08 (`src/audio_screen.py`, faster-whisper).** audio -> transcribe -> segment utterances by pause -> word features -> reduced model -> screen + styled HTML report. KEY ENABLER: the reduced model (sentence length + vocabulary + age, all ASR-derivable, no %mor) scores AUC 0.857 vs the full 0.861 - so dropping the grammar tiers costs ~nothing. Verified end-to-end on synthesized speech: with real utterance pauses it segments correctly (e.g. 42-mo child, 4 utts, MLU 2.5, language age 50 mo, not elevated); with run-on speech it HONESTLY flags "segmentation unreliable" instead of a fake number. OPEN PROBLEM (the real work): utterance segmentation from audio (Whisper word timestamps are contiguous; needs a proper voice-activity/silence-based segmenter). VALIDATION: DATA-LIMITED (checked 2026-08). Our clinical corpora ENNI and Gillam both list media "not available" - their audio was never shared publicly, so we CANNOT validate the audio->flag path on the exact clinical data. Synthetic feature-recovery check (TTS-read real child utterances -> transcribe -> compare): recovers MLU within ~5-9% and segments utterances well (gold 11.6 vs ASR 10.6 for a TD child; 3.8 vs 4.0 for an SLI child), so the plumbing/features/segmentation are SOUND on clear speech. UNTESTED (the hard part): real child-speech ASR error, which needs an audio+transcript corpus we don't have for the clinical data. FORWARD: either find an English child corpus WITH audio+transcripts (e.g. Providence, young ages) for a real feature-recovery study, or present the audio front-end as a demonstrated PROTOTYPE + the clean-speech recovery result + this honest data limitation. Do NOT claim the flag is validated on recorded speech.
- **DEMO APP (`src/demo_server.py`, Flask): DONE.** One local page (localhost, offline): drag a .cha transcript (full model) OR record/drop audio (reduced model), click Analyze, get the styled report. Run: `.venv/bin/python src/demo_server.py` then open http://localhost:5050 in Chrome (port 5050, not 5000 - macOS AirPlay Receiver owns 5000; override with PORT=...).
- **WEB DEMO (`web/`, 2026-10): DONE + deployed (https://web-pearl-sigma-14.vercel.app).** Browser-only prototype on the laptop (not a phone app): transcript / typed sentences / recording -> words-only screen. Details, deploy and test steps in Section 0d. This is now the PRIMARY judging demo; the Flask app above is the offline backup.
- Longitudinal trajectory modeling (Brown/ENNI have repeated sessions). (2026-10: rejected by the authors - "language problems predict later outcomes" is already established; not pursued.)
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
|   |- stage3..5, ext_*.py        delay classifier, interpretability, demo, extensions (see README)
|   |- poster_figures.py          board figures fig1-fig8 (results/poster/, print-sized, recomputed)
|   |- poster_figures_extra.py    board figures fig9 (stress test) + fig10 (validation diagram)
|   |- export_web_model.py        trains + exports the words-only web model -> web/model.json
|   |- ext_asr_stress.py          transcription-error stress test (-> results/asr_stress.csv)
|   |- ext_reliability.py         split-half reliability
|   |- ext_fairness_ethnicity.py  ethnicity fairness audit on Gillam (ISS-17)
|   |- ext_fairness_fix.py        does dropping vocabulary shrink the gap? (exploratory)
|- web/                     browser web demo (index.html, app.js, model.json; models/ gitignored)
|- scripts/fetch_web_model.sh  downloads the ~76 MB Whisper files into web/models/ before deploying
|- docs/                    EXPLAINER, METHODS_WALKTHROUGH, READING_LIST, PITCH (pitch + judging demo plan)
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
- Read Sections 0, 0b, 0c, 1, 2 in that order. Committed project = FirstWords. Data + both signals verified; Stage 2 (level estimator) AND Stage 3 (delay classifier) DONE and verified (0b Stage 2/3 RESULTS). Section 0c is the running issue log - skim it to see what's already been caught and resolved before re-investigating anything.
- Stages 1-5 + all Tier-1/Tier-2 extensions are DONE. The project is end-to-end complete and demoable. NEXT = POLISH, not new research: (1) build the poster from Section 9; (2) download 1-2 Eng-NA TD corpora with many children at 18-48 mo (Brown alone is 3 there) and re-run stage2 to firm up the young-age core; (3) optional: Gillam as an age-matched cross-corpus test to complement Conti4; run the live demo for practice. Do NOT add unvalidated arms (Lesson 5) - depth/polish now.
- Keep the discipline that finally works: signal first, child-independent validation, baselines, honest CIs, artifact control, one checkpoint at a time, a usable tool + demo, honesty over hype.
- History/fallbacks: DopaLoop lives at /Users/mihir/DopaLoop (MATLAB; do not touch). The ParkinTrack (PD) and ALS-Forecast explorations were removed from this repo (kept only as LESSONS in Section 1) - ALS-Forecast on PRO-ACT is the strongest non-BEHA fallback (TMED); choices13k (decision-making) and WESAD (stress) are BEHA fallbacks with instant data.

================================================================================
## 9. POSTER & PRESENTATION TALKING POINTS (LIVING - add to this as results land; polish later)
================================================================================
_Purpose: capture, as we go, exactly what to SAY and SHOW to judges so nothing is reconstructed from memory the week before the fair. Judging in BEHA weights Research Question, Design/Methodology, Execution, and Communication. This section is our communication bank. Refine wording later; capture substance now._

### 9a-0. ⚠ THE TYPOLOGY "DISCOVERY" DID NOT REPLICATE - DO NOT HEADLINE IT (updated 2026-07-20)
With only 3 languages the pattern looked clean (Cantonese isolating -> vocabulary diversity; English/German
-> sentence length). We then did the right thing and TESTED it by adding Mandarin (a 2nd isolating language)
and Turkish (agglutinative). **It broke:** Mandarin (also isolating) is LENGTH-dominant (lex-dom -0.35), the
OPPOSITE of Cantonese (+0.33). Two isolating languages disagree -> the marker-dominance is NOT a function of
morphological typology; it is CONFOUNDED WITH TASK (Cantonese = free-play -> diversity; Mandarin/English =
narrative -> length grows with age). See `src/ext_typology.py`. **Do NOT claim a typology law or lead the
poster with it** - a judge who asks "what about other isolating languages?" would break it. This is a
self-caught over-claim (log it; it's a credibility POINT that we tested and retracted our own hypothesis).
- HONEST positioning for the poster (what actually holds): (1) an accessible, INTERPRETABLE, rigorously
  validated language-delay screen with **artifact control** (age-/length-alone at chance) and **two
  independent external validations** (Gillam 0.82, Conti4 0.70) - this is the real, strong story; (2) the
  DELAY-detection approach generalizes to a 2nd language (German TD-vs-impaired, age-matched AUC 0.90) - the
  most novel DEFENSIBLE result; (3) a narrower methodological point that DOES hold: no single language
  feature is universally the best developmental marker across languages/tasks -> per-language/per-task
  calibration is necessary (supported by both ext_typology and the ext_unified failure).
- Creativity reality (honest): the core methods are not novel and the typology finding did not survive
  replication. Do not manufacture novelty. The score comes from rigor, execution, honesty, and a polished
  demoable tool - lead with those.

### 9a. The one-sentence pitch (tool framing - use as the SECONDARY line)
"FirstWords is a growth chart for a child's language: paste a short speech sample, and it estimates the child's language-developmental age and flags likely language delay - an accessible, interpretable early screen, since early intervention is what changes a child's life."

### 9b. The 60-second elevator version (the arc)
1. PROBLEM: 10-15% of toddlers have language delay; early intervention markedly improves outcomes; but screening is late, subjective, or gated behind specialists many families can't reach.
2. IDEA: language complexity (sentence length, vocabulary, grammar) rises with age extremely reliably - so a child's language can be scored against age norms, exactly like height on a growth chart.
3. WHAT I BUILT: two layers - (a) a level estimator that predicts language age from a transcript, and (b) a delay flag (typically-developing vs. language-impaired).
4. RESULTS: level estimate within ~11 months, child-independent; delay flag AUC 0.86 on the same task; and I proved the flag isn't cheating on age or transcript length.
5. WHY IT MATTERS: it's free, interpretable (it tells you WHICH markers are low), and it works on data the model has never seen - a real screening aid, not a black box.

### 9c. Headline claims -> evidence (Claim-Evidence-Reasoning; every number must be on the poster)
- CLAIM: Language level is predictable from a transcript. EVIDENCE: ridge MAE 11.2 mo, R2 0.76, child-independent (GroupKFold), vs predict-mean baseline 27.1 mo. REASONING: the model more than halves the error of guessing the average; validated on children never seen in training.
- CLAIM: The tool flags language delay. EVIDENCE: ROC-AUC 0.861 [0.81-0.91], child-independent (StratifiedGroupKFold), 91% sensitivity at a screening threshold. REASONING: strong separation on the SAME narrative task at matched age.
- CLAIM (the money slide): The delay signal is genuinely linguistic, not an artifact. EVIDENCE: age-alone AUC 0.42 and transcript-length-alone AUC 0.49 (both ~chance); language-only 0.78. REASONING: if the flag were just detecting older/longer transcripts, those confounds would classify - they don't.
- CLAIM: It's interpretable. EVIDENCE: standardized coefficients - low morpheme-MLU and low lexical diversity drive the flag, age acts as the norm reference. REASONING: a clinician-style readout ("grammar + sentence length below age norm"), not a black box.

### 9d-bis. CROSS-CORPUS GENERALIZATION (new headline evidence, 2026-07-19)
- CLAIM: the delay signal generalizes beyond our training data. EVIDENCE: ENNI-trained flag, NO refit, on Conti-Ramsden 4 (British adolescents, different lab/task/age): frog-narrative AUC 0.702 [0.566-0.823], both CIs exclude chance. REASONING: it transfers across country + lab + task + a 6-10 yr age gap; the drop from 0.86 is the honest cost of that shift, not a failure. This is the single most convincing "it actually works on new data" result - build a slide for it.

### 9d. "Money slides" to build (visuals that win)
1. Calibration scatter (results/stage2_calibration.png): predicted vs actual age hugging the diagonal, 18-120 mo.
2. Artifact-control bar chart: AUC of language vs age-alone vs length-alone - the "we controlled the confound" proof.
3. ROC + risk-score histograms (results/stage3_roc.png): clear TD vs SLI separation.
4. Cross-corpus generalization bar: in-domain ENNI AUC 0.86 vs external Conti4 AUC 0.70 (with CIs) - "works on a different lab's data."
5. Language growth chart (results/ext_growth_chart.png): percentile curves + a flagged child at P6 - the metaphor made literal.
6. Short-sample robustness (results/ext_shortsample.png): AUC ~flat from 10->all utterances - "a brief sample is enough."
6b. LIVE DEMO (Stage 5): paste a transcript -> language age + percentile + gap + flag + top drivers. The single biggest score-mover.
7. Accessibility / record-and-screen panel: screenshot of the demo app + the enabler ("word-only reduced model AUC 0.857 vs 0.861 -> a phone recording is enough") + honest "prototype" note. This is the equity headline.
7b. Benchmark panel (optional): interpretable 4-feature model (0.86) vs black-box TF-IDF text model (0.82) vs naive rule -> "interpretability costs no accuracy."
8. Decision-curve (results/ext_decision_curve.png): the screen beats "refer everyone" and "refer no one" - it's clinically USEFUL, not just accurate.
9. The Issue Log (Section 0c) as a "how I made this rigorous" panel: caught our own metric-scale error, a data-leak, and a cross-lingual age confound - and showed they didn't change the honest conclusions.

### 9e. Anticipated judge questions -> prepared answers (expand as we go)
- "Isn't this just measuring transcript length?" -> No; MLU is per-utterance (length-robust: r=0.09-0.14), and length-alone AUC is 0.49 (chance). Slide 2.
- "How do you know it's not just age?" -> Age-alone AUC 0.42; age enters only as the norm reference, and the language features carry the signal (language-only 0.78).
- "Did you test on unseen children?" -> Yes - all validation is child-independent (grouped by a child id derived from the CHAT header, after we found A/B share children; ISS-07).
- "What's novel?" -> Not the ML - the honest, artifact-controlled, interpretable, accessible TOOL. Automatic SLI detection exists in the literature; our contribution is a rigorous, usable screen with a demo and self-audited validation.
- "IPSyn of 22 at age 7 looks wrong." -> Correct catch - the library's index isn't on the published IPSyn scale, so we relabel it a generic 'syntax index' and don't rely on it for the delay flag (ISS-01/02).
- "What are the limits?" -> (self-name these, see 9f).
- "SHAP says age is the biggest driver of your flag - aren't you just detecting age?" -> No. Age has high SHAP importance because it's the norm REFERENCE (what MLU to expect at that age), but age ALONE classifies at AUC 0.42 (chance). The flag fires when language falls short of the age expectation - age frames the norm, language triggers the flag.

### 9f. Limitations to STATE OURSELVES (naming them first is a credibility win - Lesson 8)
- Few distinct young children (Brown = 3 kids under 48 mo) -> level estimate is thinnest in the toddler range; adding more TD corpora is in progress.
- Delay layer validated on ONE task/corpus (ENNI narratives, ~4-10 yr) -> cross-corpus/other-task generalization is the key open test (see Section 10).
- SLI is one clinical label; not a diagnosis - this is a SCREEN that suggests seeking a professional, not a clinical instrument.
- Transcripts are human-produced CHAT; a real deployment needs reliable child-speech transcription. We built a WORKING record-and-screen PROTOTYPE (demo app) that removes the transcript barrier, but its audio path is not yet validated on real child recordings (data-limited) - present as a prototype, name real-child ASR accuracy as the open validation.
- The syntax index is nonstandard (relabeled); percentile norms are corpus-derived, not clinical norms.
- Cross-TASK reliability was poor in an out-of-range stress test (ISS-11) -> the screen assumes a single standardized, age-appropriate task; true test-retest reliability (repeated same-task samples in 4-10yr) is future work.
- The delay-flag probability from the recall-tuned model is over-stated; report the isotonic-recalibrated probability for honesty (ext_calibration).
- Grammatical-deficit profile is descriptive/discriminative on ENNI (school-age); the early tense-marking deficit would need younger clinical data to show directly.
- (2026-10, interview only) Over-flags typically developing African American children on outside data (31% vs 14%), traced to vocabulary diversity (ISS-17); the ENNI cutoff does not transfer to Gillam (ISS-18). Do not claim equal performance across groups. Honest answer if asked: "we checked, found it, traced it to vocabulary rather than grammar, and sentence length looks fairer - that's our next step."

### 9g. Ethics / paperwork to show you handled (judges ask)
De-identified public minors' data (CHILDES/TalkBank terms); ISEF human-participants pre-existing-dataset path -> SRC review + adult sponsor BEFORE formal work; no re-identification; framed as a screen that points families to professionals, not a diagnosis.

### 9h. POSTER CONSTRUCTION GUIDE (build to ISEF-winner standard - e.g. the clean, discovery-led boards of Grand Award winners)
_(2026-10: the board has been BUILT - see Section 0d for its actual layout, figures, on-board vs interview-only content, and open items. The guide below is the original design reasoning; where it conflicts with 0d, follow 0d. Pitch + demo plan: docs/PITCH.md.)_
_A winning ISEF board is a MAGAZINE SPREAD, not a lab report: one clear story, read in 60 seconds from 3 feet away, with big figures doing the talking and minimal text. Judges decide in the first 15 seconds whether this looks like real science. Design for that._

**DESIGN PRINCIPLES (the difference between a winner board and a class-project board)**
- **One story, told with figures.** ~40% of the board should be VISUALS, ~30% white space, only ~30% text. If a figure can say it, do NOT write a paragraph.
- **Ruthless hierarchy.** Title (readable from 10 ft) > section headers (from 6 ft) > body (from 3 ft). Never a wall of text; max ~50 words per text block; use bullets, bold key numbers.
- **Visual system.** ONE accent color + neutrals (no rainbow). ONE clean sans-serif font (e.g. a single family, 2-3 sizes). Consistent figure style (same fonts/colors as the reuse of our matplotlib figures - regenerate them with a shared style). Align everything to a grid; generous margins.
- **Number-forward.** Every claim shows its number AND its uncertainty (CI) AND its comparison (baseline / chance). "AUC 0.86 [0.81-0.91] vs 0.50 chance," never a bare "0.86."
- **Self-contained flow.** A judge should be able to walk it left-to-right, top-to-bottom, with no guidance, and get: problem -> discovery -> how you validated it -> why it matters.

**LAYOUT (standard tri-fold, left->right reading order)**
1. **TOP BANNER (full width):** Title + your name/school + the one-line thesis. (NOTE: the typology "discovery" did NOT replicate - see 9a-0 - so do NOT title the board around it.) Title the board around the REAL, defensible contribution, e.g. "A Growth Chart for Language: An Interpretable, Externally-Validated Screen for Developmental Language Delay" (subtitle: rigor + accessibility). The honest strengths - not a retracted finding - lead.
2. **LEFT PANEL - THE SETUP (why + what):**
   - *Background / problem* (short): language delay is common (~7%, DLD > autism, under-recognized), early intervention works, screening is late/gated. 2-3 sentences + one small icon/graphic.
   - *The gap & question*: complexity rises with age (the "growth chart"); can we measure a child's language age from a transcript, flag delay, and does the method hold across languages? State the HYPOTHESIS as a testable question.
   - *Data & methods (compact)*: CHILDES/TalkBank; N (335 children core, 360 delay, + external Gillam/Conti4, + German/Cantonese); one small "methods at a glance" box: features (MLU, vocab diversity, syntax), interpretable models (ridge/logistic), CHILD-INDEPENDENT validation. Keep it tight - depth goes on the paper.
3. **CENTER PANEL - THE RESULTS (the biggest, most visual real estate; put the money slides here):**
   - **HEADLINE (top center) = THE ARTIFACT-CONTROL PANEL**: bar chart of language-only vs age-only vs length-only AUC (0.78 / 0.42 / 0.49) with the caption "the signal is real language, not age or transcript length." This is your strongest, most rigorous, most defensible result - make it the biggest figure. (The typology figure is RETRACTED - see 9a-0 - do not feature it as a finding.)
   - **The core results, compactly**: level estimator calibration scatter (MAE ~10 mo, R2 0.81) + delay ROC (AUC 0.86) + the two external validations (Gillam 0.82, Conti4 0.70) as a small "generalizes across labs" bar.
   - **The growth-chart percentile figure** + one **example readout** from the demo (a flagged child at P6 with drivers) - makes it concrete and human.
4. **RIGHT PANEL - MEANING + RIGOR (what it means + why to trust it):**
   - *Clinical utility*: the decision-curve figure ("beats refer-all / refer-none") - it's USEFUL, not just accurate.
   - *Rigor / self-audit box*: 1-2 lines on the issue log - "we found and fixed our own confounds (a data leak; a cross-lingual age confound)." A short, honest "Limitations" list (name them yourself - §9f).
   - *Conclusion*: restate the discovery + the tool, in 2-3 bullets, number-forward.
   - *Future work*: more typological extremes (Turkish, Mandarin), dialect/SES fairness, ASR front-end. *(Cite that Spanish was EXCLUDED from the typology axis on a pre-stated data-quality bar - honest.)*
   - *References* (small) + *Acknowledgments/AI-use disclosure* per fair rules (§2 integrity note).
5. **THE LIVE DEMO** sits ON THE TABLE in front of the board - the board points to it ("try it -> "). TWO forms: (a) the polished web app `src/demo_server.py` (run it, open http://localhost:5050) where you DRAG IN a held-out .cha transcript OR RECORD a child talking, then Analyze -> styled report; (b) fallback = pre-generated HTML reports in `demo/reports/` (double-click, no server). The RECORD-AND-SCREEN app is the closer and the accessibility story: "a family with a phone, no specialist needed." Frame the audio path honestly as a prototype. Use held-out Gillam transcripts (in the demo/ folder) so you can say "the model never saw this child." Only use children ~4-10 yr (delay-model range).

**FIGURE PREP (do this before printing):** regenerate every results/*.png used with a SHARED style (same font, same single accent color, large labels, no chart-junk, readable from 3 ft). A consistent figure look is 50% of a board looking "winner-level." Export at 300 DPI. Recommended headline figures, in priority order: ext_typology.png, the artifact-control bar, stage3_roc.png, stage2_calibration.png, ext_growth_chart.png, ext_decision_curve.png.

**RUTHLESS DON'Ts:** no full paragraphs of methods (that's the paper); no default-Excel charts; no more than ~6-7 figures total; no unexplained jargon in the title; no clutter/rainbow colors; nothing unreadable from 3 feet; never a number without its comparison/uncertainty.

**BEFORE THE FAIR:** (1) print a small draft, tape it up, stand 3 ft back - can a stranger get the story in 60 s? (2) rehearse the live demo COLD on an unseen transcript; (3) rehearse the 60-second arc (§9b) and the Q&A bank (§9e) out loud until they're yours; (4) complete the AI-use disclosure honestly (§2).

================================================================================
## 10. EXTENSION ROADMAP for the extra runway (~7-8 months) - EACH SIGNAL-GATED (do NOT bolt on unvalidated arms)
================================================================================
_The base (Stages 1-3) works and will complete. These are how to RAISE THE CEILING with the extra time. Same firewall as the two-layer core: every new arm must pass its own signal-gate FIRST, and a failure becomes a documented limitation, not a project-killer. Ordered by payoff-to-risk. Do Tier 1 before Tier 3._

### Tier 1 - highest payoff, moderate risk (these most increase winnability)
- **CROSS-CORPUS GENERALIZATION (the big one). `src/ext_crosscorpus.py` BUILT.**
  - PART A (LEVEL transfer) DONE + runnable now: train age estimator on one corpus, test on the other (Brown conversation <-> ENNI narrative), overlap band 48-62 mo. RESULT: transfers with raw MAE 5.5-7.2 mo and a small task offset (-2.7 to -4.2 mo); offset-corrected MAE 4.9-6.1 vs CHILD-INDEPENDENT within-corpus ref 2.6 (ENNI) / 5.4 (Brown). So train-ENNI->test-Brown (6.1) is only ~0.7 mo worse than Brown's own within-corpus ref (5.4) -> transfers in rank/precision, tasks differ only by a small absolute offset. Poster line: "the model still works across a different task/corpus, once you account for a small task offset."
  - PART B (DELAY transfer - the headline test) DONE. TWO independent external clinical corpora, trained on ENNI, NO refit (harness auto-finds any corpus with SLI labels; skips not_in_use/):
    - **GILLAM (US; 493 TD + 171 SLI; ages 5-12 yr; narrative; ~one/child) - the CLEAN, AGE-MATCHED test (overlaps ENNI 4-10 yr).** RESULT: **full(+age) AUC 0.803 [~0.77-0.84]; 0.823 restricted to ENNI's <=120 mo range**; language-only 0.778. This is the strong headline external validation - same task, matched age, different lab/country.
    - CONTI-RAMSDEN 4 (British; 99 TD + 19 SLI; adolescents 13-16 yr; frog narrative + conversation) - the HARD out-of-range stress test. Language-only (fair, since out of age range): frog AUC 0.702 [0.566-0.823], spontaneous 0.666 [0.533-0.784]; both CIs exclude 0.5.
  - INTERPRETATION (poster-ready): the delay flag generalizes to a DIFFERENT LAB with NO retraining - **AUC 0.80-0.82 in-range (Gillam), and still 0.70 even out-of-range across country+age (Conti4).** Two external validations, one clean + one stress-test, is a very strong generalization story. Gillam is the headline number; Conti4 shows graceful degradation under extreme shift.
- **NORM-REFERENCED PERCENTILES ("growth chart" made literal). `src/ext_percentiles.py` DONE.** Quantile regression (P10/25/50/75/90 of sentence length vs age on TD) -> a child lands on a percentile like a pediatric growth chart. VALIDATED: 10.4% of TD children fall below P10 (target 10% = well-calibrated bands); example flagged child at P6. Output: results/ext_growth_chart.png (a genuine money slide). `percentile_for()` reusable by the Stage 5 demo.

### Tier 2 - strong, lower risk (clinical realism + fairness) - BOTH DONE 2026-07-19
- **SHORT-SAMPLE ROBUSTNESS. `src/ext_shortsample.py` DONE.** Truncate each ENNI transcript to the first N child utterances, recompute the exactly-recomputable length-robust features (sentence length + MATTR-50; morpheme-MLU needs the full %mor pipeline so is omitted here) + age, child-independent AUC vs N. RESULT: AUC 0.818 (N=10), 0.829 (20), 0.850 (50), 0.862 (full) -> **barely degrades down to ~10-20 utterances.** Poster line: "a brief sample - about 10-20 sentences - is enough to screen." Fig results/ext_shortsample.png. (Aside: this 2-feature model already ~matches the 4-feature AUC, i.e. sentence length carries most of it.)
- **SUBGROUP / FAIRNESS AUDIT. `src/ext_fairness.py` DONE.** Child-independent OOF preds, AUC + sens/spec at one screening threshold, broken down by sex and age band. RESULT: overall AUC 0.860; by sex female 0.83 / male 0.89 (sensitivity spread 0.13, just under our 0.15 flag - a MODEST gap, likely from fewer SLI females n=29 vs 46 -> noisier); by age band 0.80-0.90 (weakest at 6-7y). HONEST framing: report as an equity CHECK, not precise per-group performance (small cells); name the female/male gap as a limitation + future-work item.
- **CLINICAL UTILITY / DECISION-CURVE ANALYSIS. `src/ext_clinical_utility.py` DONE.** Decision-curve analysis (Vickers & Elkin) on child-independent OOF preds: the screen gives the HIGHEST net benefit of the three policies across threshold probabilities 0.05-0.46 (the clinically plausible range) - i.e. it beats both "refer every child" and "refer none." Translation: at a screening threshold it catches ~95% of affected kids while avoiding ~37 unnecessary referrals per 100 children vs referring everyone. Reframes AUC as real-world value. Fig results/ext_decision_curve.png.
- **LOCKED-BOX / HELD-OUT VALIDATION note:** we did not pre-register a within-ENNI lockbox, BUT Gillam + Conti4 are TRUE held-out sets - trained only on ENNI, never used for any tuning/model-selection - so their AUC (Gillam 0.80-0.82, Conti4 0.70) IS the confirmatory locked-box result. Frame them that way on the poster (external held-out = stronger than an internal holdout).
- **PROBABILITY CALIBRATION. `src/ext_calibration.py` DONE.** Reliability diagram + Brier on OOF preds. RESULT: the balanced model (good recall) OVER-states probabilities (Brier 0.153); unweighted 0.113; balanced+isotonic recalibration 0.104. TAKEAWAY: for the demo, show an isotonic-RECALIBRATED probability (honest), but keep the recall-oriented screening THRESHOLD separate. Fig results/ext_calibration.png. (Demo currently shows the raw balanced score as a relative risk indicator - fine for a demo; use recalibrated on the poster.)

### Tier 3 - the ambitious reach (genuine novelty) - ITEM-LEVEL DONE 2026-07-19; others remain
- **ITEM-LEVEL GRAMMATICAL PROFILE. `src/ext_morphprofile.py` DONE (novelty).** Per-child rates of specific grammatical markers from %mor/%gra, TD vs SLI (ENNI). RESULT: strongest discriminator = SUBORDINATE-CLAUSE rate (Cohen's d=0.85; SLI 0.21 vs TD 0.30), then 3rd-sg -s (0.45), possessive (0.40), past-tense (0.32); finiteness is EQUAL by school age (0.71 both). DEVELOPMENTAL STORY (sophisticated + correct): the classic tense-marking/Extended-Optional-Infinitive deficit is an EARLY marker that resolves by ~5yr, so at ENNI's 4-10yr the gap has SHIFTED to higher-order syntax. Morphology-only classifier AUC 0.727 (child-independent) - competitive with, and clinically richer than, the MLU model. Dropped morph_diversity as length-confounded (ISS-10). Fig results/ext_morph_profile.png.
- **CROSS-TASK RELIABILITY. `src/ext_taskreliability.py` DONE - HONEST NEGATIVE (ISS-11).** On Conti4's two tasks, cross-task risk-score agreement is poor (r=0.11) - because out of the trained age range scores compress to noise. Reported as a limitation + future work (repeated same-task samples in target age). This is a CREDIBILITY win (we found and reported our own weak spot), not a failure.
- **MULTILINGUAL LEVEL ESTIMATOR (equity headline; ArabLexify shape). `src/ext_multilingual.py` DONE 2026-07-20 - GERMAN WORKS; Spanish premise confirmed but data-limited.**
  - Harness (`ext_multilingual.py`): auto-detects language from @Languages, scans data/childes/ (skips not_in_use/), prefers a TD/ subfolder for controls, filters age outliers (<=156mo), and reports for each non-English corpus: cross-sectional corr(age,MLU), MEDIAN WITHIN-CHILD corr (the PREMISE test), a cross-child estimator GATE, and (if it passes) a child-independent MAE/R2 + figure.
  - **GERMAN - Szagun controls (22 children, 15-44 mo): STRONG. best marker mlum (corr 0.87), within-child median 0.94, LEVEL estimator MAE 3.1 mo [95% CI 2.5-3.8], R2 0.79** (~English quality). Fig results/ext_ml_deu_szagun.png.
  - **CANTONESE - HKU-70 (70 children, 2-5 yr, isolating language, non-Indo-European): WORKS + a TYPOLOGY finding. Sentence LENGTH is a weak age index here (mluw corr 0.38) but VOCABULARY DIVERSITY is strong (mattr 0.71) -> LEVEL estimator MAE 6.5 mo [5.3-7.7], R2 0.55.** Fig results/ext_ml_yue_hku.png. INSIGHT: the growth-chart METHOD is universal across 3 language families (English Germanic-analytic, German Germanic-inflected, Cantonese Sino-Tibetan isolating), but the BEST MARKER is typology-dependent - length/morphemes for inflected languages, lexical diversity for isolating ones. (Harness now gates on the STRONGEST marker, not MLU alone - ext_multilingual.py.)
  - **CROSS-LINGUAL DELAY (the killer test - not just level, but the FLAG in another language). `src/ext_crosslingual_delay.py`.** German Szagun TD vs CI (cochlear-implant = language-delayed), within-German, child-independent. The artifact check CAUGHT a group age confound (CI cohort followed to older ages; age-only AUC 0.92 on full data -> full model's 0.99 is mostly age, DISCARDED). **Honest headline = AGE-MATCHED (21-44mo) language-only AUC 0.903 [0.837-0.957], vs residual age-only 0.71 -> language carries signal well beyond age.** So the DELAY-DETECTION approach generalizes to a 2nd language. CAVEATS: CI = hearing-related delay (not SLI); within-German model (English model NOT transferred - MLU scales differ); age-match imperfect (report language-vs-age side by side). This is also a RIGOR showcase: the tool caught its own confound.
  - **SPANISH - PREMISE CONFIRMED, cross-child estimator DATA-LIMITED.** Within-child age<->complexity is strong (Ornat r=0.77 over 1.6-3.8yr; Marrero/Idaira 0.99; FernAguado within-child median 0.62) = the growth-chart premise holds in Spanish too. BUT no Spanish corpus has many DISTINCT children across a wide age range (FernAguado = 47 kids but all age 3-4 -> no learnable slope; Ornat = 1 child; Marrero = 3), so a cross-child Spanish estimator FAILS the gate (pooled corr 0.33). This is a DATA-availability finding, not a method failure - and it SHARPENS the equity story: the developmental data/tools don't exist for Spanish-speaking children.
  - POSTER FRAMING (honest + strong): "The method is language-GENERAL: the age<->complexity premise holds within children in English, German, AND Spanish (within-child r=0.77-0.99), and a full child-independent estimator reproduces in German (MAE 3.1 mo, R2 0.79). Spanish confirms the premise but lacks the wide-age multi-child corpora needed to build a deployable norm - itself an equity gap this work highlights." (Earlier Shiro/BecaCESNo attempts - age-plateau / heterogeneous - are archived in data/childes/not_in_use/; superseded by the within-child + German analysis.)
  - **TYPOLOGY EXTENSION TESTED + RETRACTED (2026-07-20). Turkish (Aksu, 54 files/33 kids) + Mandarin (ZhouNarratives, 232 files) were downloaded and added to `ext_typology.py`.** Result: the typology hypothesis FAILED - Mandarin (isolating) is length-dominant, contradicting Cantonese (isolating). The effect is task-confounded, not typological (ISS-16). Do NOT pursue this as a novelty headline. The corpora stay (Turkish/Mandarin could support a within-language premise check or future work), but the "typology gradient" is a dead end with available cross-task data.
  - **THE KILLER EXTENSION (grand-award caliber, if data exists): extend the DELAY layer cross-linguistically.** So far only the LEVEL estimator is multilingual. If a clinical SLI corpus exists in a 2nd language (Cantonese DLD, or Szagun's cochlear-implant group as a language-impaired proxy), test whether the DELAY FLAG generalizes -> "the screen detects impairment across languages" is a genuinely novel, strong claim. Harder (needs per-language clinical data); highest payoff of any remaining research direction.
- **LONGITUDINAL TRAJECTORY (weak data - lowest priority).** Brown has only 3 children; not enough for honest growth-trajectory generalization. Skip unless a large longitudinal multi-child TD corpus is added.
- **NEW candidate extensions (2026-07-20, novelty-raising, using data in hand):**
  - **UNIFIED CROSS-LINGUAL MODEL - DONE 2026-07-20 (`src/ext_unified_crosslingual.py`) - HONEST CAUTIONARY RESULT.** Trained ONE ridge model on language-standardized features (z-score within each language, no age/labels used) across English+German+Cantonese. FINDINGS: (a) a single pooled model DOES span 3 language families (child-independent R2 0.51) - proof a cross-lingual index is *possible* - BUT it is WORSE than the per-language models (0.54-0.82). (b) LEAVE-ONE-LANGUAGE-OUT (train on 2 langs, test on the 3rd unseen) FAILS (R2 negative), and STILL fails when age-matched to the common 29-44mo window (deu R2 -4.18, eng -0.56, yue ~baseline) - so it's not just an age-range artifact; the feature->development mapping is genuinely language-specific. CONCLUSION (honest, reinforces the thesis): zero-data transfer to a new language doesn't work; **language-specific norms + typology-aware markers are NECESSARY, not optional.** Present as a rigorous cautionary result (we tested the ambitious unification and it didn't beat the simpler per-language approach) - scientific maturity, not a flashy win. Do NOT claim a "universal model."
  - **LLM BASELINE COMPARISON.** Benchmark the interpretable ridge/logistic model against a modern LLM-embedding classifier on the same child-independent splits. Expected/target story: the simple model MATCHES it while being interpretable + auditable ("we tried the fancy thing; the honest simple thing held up"). Timely, judge-resonant.
- **IMPACT FRAMING - LITERACY (add to intro/poster, evidence-based, INDIRECT).** Oral language is the foundation of reading (Simple View of Reading: reading comprehension = decoding x LANGUAGE comprehension); early language delay strongly predicts later reading difficulty (Catts, Fey, Tomblin & Zhang 2002). Frame as a downstream benefit of early screening, NOT a direct literacy claim (we screen spoken language, not reading).
- **UNDERSERVED-LANGUAGE / EQUITY (e.g. ARABIC).** Adding a language tied to underserved/low-resource regions strengthens the accessibility headline (ArabLexify archetype). HONEST REALITY: these languages usually LACK wide-age multi-child corpora - which is ITSELF the inequity/finding ("the data & tools don't exist for these children"). Gate first (does CHILDES have Arabic w/ enough children + age spread?); even within-child premise confirmation helps; frame the DATA GAP as the point. Do NOT add a language for optics without usable data.

### October 2026 - extensions CLOSED
Done: transcription stress test (clear, on board), split-half reliability (clear, on board), ethnicity fairness audit (finding, ISS-17, interview only), vocabulary-drop fix test (not conclusive), web demo. Rejected by the authors: human-rater experiment, any human-subjects work, longitudinal prediction. The only optional remaining study is a real child-audio check (needs a free TalkBank login for media; Providence has audio + transcripts) - not required. Do not add more extensions; spend the time on the team's own board wording and the interview.

### Honest strategic note (for the author)
The base already clears the bars that eliminate most projects (real verified signal, child-independent validation, correct category, a working tool). That gets you COMPETITIVE at the regional and a genuine shot at ISEF placement. Tier 1 (cross-corpus + percentiles) + a polished live demo is what pushes toward TOP awards - because "it generalizes to unseen data from a different lab" plus "here, try it live" is what separates a finalist from a winner. Add reach arms only on top of a finished, demoable core; never at its expense (Lesson 5).
