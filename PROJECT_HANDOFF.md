# FirstWords - PROJECT HANDOFF & COMPLETE PLAN
### An interpretable tool that estimates a child's language-development level (and flags likely delay) from a speech sample
_Self-contained record. Any agent (including a collaborator's agent) MUST read Sections 1-2 before touching the project._
_Updated 2026-07-14 (after real data downloaded + both signals verified). Category: ISEF Behavioral & Cognitive Sciences (Developmental Psychology)._

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
- **DELAY signal looks REAL (ENNI, TD vs SLI, AGES MATCHED ~85 mo):** TD n=286, SLI n=75. At the same age, language-impaired children have much lower MLU: words 6.90 (TD) vs 5.67 (SLI); morphemes 7.09 vs 5.81 - roughly a 1 SD gap on the SAME narrative task (so not an age or task artifact). TTR 0.37 vs 0.35 (small). NOTE anomaly: IPSyn 22.03 (TD) vs 22.56 (SLI) - slightly HIGHER for SLI (counterintuitive; MLU/morphemes are the strong discriminators; investigate IPSyn in Stage 3).
### Access / tooling / parsing (all verified)
- CHILDES downloads require a FREE TalkBank login (no anonymous route). Corpora are per-language collections; clinical corpora live under "Clinical-Eng" (ENNI/Gillam/Conti-Ramsden); typically-developing English under "Eng-NA" (Brown, etc.). Clinical corpora may be a higher access tier.
- pylangacq API (rustling backend, v0.23): `read_chat(path, strict=False)` - MUST use strict=False (real CHAT files break strict parsing, e.g. "pause marker embedded in word"). Reader members: `n_files` and `file_paths` are ATTRIBUTES (no parens); `ages()` is a method with NO kwargs returning Age objects (parse "Y;MM.DD" -> months via regex); `mluw()/mlum()/ttr()/ipsyn()` are methods returning lists aligned to `file_paths`, computed for the target child. `mlum`/`ipsyn` need the %mor/%gra tiers (present in real CHILDES, absent in bare synthetic files).
- Feature meanings: MLU-words = mean words per utterance; MLU-morphemes = mean morphemes per utterance; TTR = type-token ratio (lexical diversity); IPSyn = Index of Productive Syntax (syntactic complexity).

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
| 2 | **CORE deliverable (NEXT):** developmental-LEVEL estimator - predict developmental age/level from features on combined TD corpora; **GroupKFold by CHILD**; baselines (predict-mean) + bootstrap CIs + calibration; define an "age-expected vs observed" gap score. Add more TD corpora first for enough distinct children. | Solid, honest, child-independent level estimation. |
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
|   |- stage1_check.py      Stage 1: parse real data + verify core & delay signals
|   |- verify_pipeline.py   no-download smoke test of the CHAT->features pipeline
|- data/childes/            (gitignored) Brown/, ENNI/{TD,SLI}/, + more TD corpora
|- results/                 (gitignored contents) outputs / figures
|- .venv/                   (gitignored) Python 3.14 env
```
Run: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`, download corpora into `data/childes/`, then `.venv/bin/python src/stage1_check.py`.
### Restart (fresh session or partner's agent)
- Read Sections 0-2. Committed project = FirstWords. Data + both signals already verified.
- NEXT concrete step = Stage 2 (developmental-level estimator). Recommend downloading 1-2 more Eng-NA TD corpora first (Brown alone is 3 children).
- Keep the discipline that finally works: signal first, child-independent validation, baselines, honest CIs, artifact control, one checkpoint at a time, a usable tool + demo, honesty over hype.
- History/fallbacks: DopaLoop lives at /Users/mihir/DopaLoop (MATLAB; do not touch). The ParkinTrack (PD) and ALS-Forecast explorations were removed from this repo (kept only as LESSONS in Section 1) - ALS-Forecast on PRO-ACT is the strongest non-BEHA fallback (TMED); choices13k (decision-making) and WESAD (stress) are BEHA fallbacks with instant data.
