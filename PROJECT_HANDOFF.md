# BEAR v2 - PROJECT HANDOFF & COMPLETE PLAN
### Project: "FirstWords" - an interpretable tool that estimates a child's language-development level (and flags likely delay) from a short speech sample
_Self-contained. Any agent (including a collaborator's agent) MUST read Sections 1-2 before touching the project._
_Rewritten 2026-07-11. Category: ISEF Behavioral & Cognitive Sciences (Developmental Psychology) - a TRUE fit._

================================================================================
## 0. TL;DR
================================================================================
- **Project:** FirstWords - an accessible, interpretable ML tool that takes a short sample of a young child's speech (a transcript), estimates their **language-developmental level** (vs. age-expected norms), and **flags trajectories that look delayed** - an early-screening aid for language delay, where early intervention dramatically changes outcomes.
- **Why it's a TRUE Behavioral-Science project:** it measures a cognitive/developmental construct (language acquisition) - core Developmental Psychology. (Unlike the ALS idea, which was medical/TMED.)
- **Why it will COMPLETE (de-risked):** the core - "language complexity rises with age, and we can estimate developmental level from speech" - is one of the most established findings in developmental psycholinguistics (Brown's stages, replicated 60+ years). It cannot dead-null. Verified today: the full pipeline (parse -> features -> age signal) runs.
- **Winner shape:** an accessible, interpretable screening TOOL for real people (children) + a demo - the shape that wins BEHA (cf. ArabLexify: an accessible screener for an underserved group).
- **Data:** CHILDES (Child Language Data Exchange System) via TalkBank - tens of thousands of child-speech transcripts, age-labeled, 20+ languages, including typically-developing AND clinical/language-impaired corpora. FREE but requires a TalkBank login to download.
- **Stack:** Python venv (verified working, py3.14): pandas, numpy, scikit-learn, shap, + **pylangacq** (CHILDES parser, gives built-in MLU / TTR / IPSyn / age).
- **Status (2026-07-11):** COMMITTED. Tooling + pipeline + core-signal mechanics VERIFIED. Blocker: download real corpora (needs your free TalkBank login). Then Stage 1.
- **Honest ceiling:** a real shot at BEHA category placement / finalist with strong execution + the "helps kids" story. The one risk (the *delay-screening* headline) is signal-gated with a solid fallback (see below), so worst case still yields a complete, useful project.

================================================================================
## 0b. RESOURCE / PIPELINE VERIFICATION (done 2026-07-11 - "check before building")
================================================================================
- pylangacq 0.23 installs on Python 3.14; parses CHAT; provides ages() (as Age objects -> parse to months), and built-in measures mluw (MLU-words), mlum (MLU-morphemes), ttr (type-token ratio), ipsyn (Index of Productive Syntax). API note: some members are ATTRIBUTES not methods (n_files, file_paths); ages() takes NO kwargs and returns Age objects.
- mlum() and ipsyn() need the %mor/%gra tiers -> return 0 on bare/synthetic files, but WORK on real CHILDES (which is auto-tagged). Confirm tiers exist after download.
- Synthetic mechanics check: corr(age_months, MLU-words) = 0.97 by construction -> pipeline math is correct (see verify_childes_pipeline.py).
- ACCESS: CHILDES downloads route through a TalkBank auth wall (initAuthModals); no anonymous zip download. Free account required. Corpus zips live under media.talkbank.org / the TalkBank browser once logged in. => Stage 0 includes "register + download."

================================================================================
## 1. HARD LESSONS - DO NOT REPEAT (paid for across DopaLoop, the idea-search, ParkinTrack, and the ALS/category detour)
================================================================================
1. **Signal first, always.** Before building, verify on REAL data (patient/child-INDEPENDENT) that the target has predictable signal, measured honestly. This killed DopaLoop (asym->relapse, null AUC 0.30) and ParkinTrack (voice did NOT track within-person UPDRS) - both caught by a signal-gate, one too late, one on day one.
2. **Prefer targets whose signal is already known/published.** "Crowded with real signal" beats "novel with no signal." FirstWords' core (age<->language) is textbook-certain.
3. **Novelty != never-been-done.** ISEF-winnable = a specific angle + excellent execution + a usable TOOL + interpretability + clear impact + a demo + genuine understanding, judged vs other high schoolers. Chase workability + execution, not global-first novelty.
4. **Validate honestly:** child-INDEPENDENT splits (GroupKFold by child/corpus); baselines; calibration + bootstrap CIs; NEVER report in-sample/leaked numbers. Beware dataset ARTIFACTS (e.g., transcript-length differences between corpora faking a "delay" signal) - control for them.
5. **Scope discipline / signal-gate every added claim.** No unvalidatable bolt-ons. Depth on one validated output > breadth of hollow arms.
6. **Verify data ACCESS + category fit early.** (Enroll-HD needed a sponsor + months; ALS-Forecast was actually TMED not BEHA; CHILDES needs a free login.)
7. **Complexity belongs in the PROBLEM, not the architecture.** Simple, interpretable models done rigorously.
8. **Honesty over hype.** State limitations plainly; name them yourself (a credibility win).
9. **Winners build a usable thing for real people** + a demo (OncoNote/Harmony/ArabLexify/SuiSensor). ML is the engine, not the headline.

================================================================================
## 2. OPERATING RULES FOR ANY AGENT (incl. a collaborator's agent)
================================================================================
- Read Sections 0-1 before doing anything. The project is COMMITTED (FirstWords). Do not re-open the idea search; do not pivot unless a signal-gate fails AND it's documented.
- Never build model code on an unverified target: confirm signal on real data, child-independent, vs a baseline, FIRST.
- Coordinate via the git repo (`bearv2` is a git repo). Commit small with clear messages; keep this handoff + a progress log current so the partner can resume. Read existing work before overwriting.
- One stage at a time; run it; verify the checkpoint; report the REAL numbers; only then proceed.
- Environment: use the project venv (`.venv/bin/python`). Claude Code can run Python here - verify, don't just write and hope.
- If a gate fails, document it honestly here and adjust WITHIN the project (fall back to the developmental-level estimator) - don't silently paper over or abandon.
- Data ethics: CHILDES transcripts are de-identified minors' language data under TalkBank terms. Keep in `data/` (gitignored); don't commit raw data; follow TalkBank terms + citation rules; do not attempt re-identification.
- ISEF: human-participants research on a pre-existing public dataset (minors) -> SRC review + forms + a Designated Supervisor/adult sponsor BEFORE formal work. Confirm current-year rules with the fair's SRC (Dallas Regional).

================================================================================
## 3. THE PROJECT (FirstWords), precisely
================================================================================
### Background
- Language delay is common (~10-15% of toddlers) and early intervention markedly improves outcomes, but screening is often late, subjective, or requires a specialist. An accessible, objective, interpretable screen from a child's natural speech would help catch delay earlier - especially where specialists are scarce.
- Child language complexity increases with age extremely reliably (mean length of utterance, vocabulary diversity, syntactic complexity). This is the bedrock signal.

### The target (two layers, deliberately)
- **CORE (de-risked, guaranteed to complete):** estimate a child's language-developmental LEVEL from a transcript - i.e., predict developmental age / stage from language features - and quantify how far a given child sits from age-expected norms.
- **HEADLINE (signal-gated stretch):** turn that into a language-DELAY FLAG - classify typically-developing vs. language-impaired children (using CHILDES clinical corpora) child-independently, controlling for artifacts. If it validates -> the headline. If it's weak/artifact-driven -> fall back cleanly to the CORE (still a complete, useful, BEHA-appropriate tool). This structure is the anti-DopaLoop firewall.

### Why it's genuinely BEHA
Developmental Psychology / language acquisition = core Behavioral & Cognitive Sciences. Judges reward developmental + accessible-clinical tools.

### Honest positioning (not novel; strong via execution)
Automatic language-level estimation and SLI detection from CHILDES exist in the literature (some with a known artifact critique). Our contribution = an accessible, interpretable screening TOOL + rigorous child-independent, artifact-controlled validation + a demo. The ML is not the novelty; the honest, usable tool for real families is.

================================================================================
## 4. DATA (CHILDES) + ACCESS + PAPERWORK
================================================================================
- **CHILDES / TalkBank:** tens of thousands of child-speech transcripts (CHAT format), age-labeled, 20+ languages; auto-tagged with %mor/%gra tiers (enables MLU-morphemes + IPSyn).
- **Corpora to pull (English North America to start):**
  - Typically-developing, age-labeled: Brown (Adam/Eve/Sarah), and larger Eng-NA collections for age coverage/sample size.
  - Clinical / language-impaired (for the delay-screening layer): e.g., ENNI, Conti-Ramsden, Gillam narrative corpora (SLI vs TD). CONFIRM these are in your access tier and note collection differences (artifact risk).
- **Access:** free TalkBank account required to download (verified: no anonymous route). Register, download the corpus zips, unzip into `data/childes/`. Individual - no institutional signature.
- **ISEF paperwork:** de-identified, public, pre-existing dataset of MINORS -> human-participants pre-existing-dataset path; needs SRC review + adult sponsor. Simpler than collecting data, but minors' data may draw a bit more SRC attention - confirm with the Dallas Regional SRC.

================================================================================
## 5. BUILD PLAN - stages with checkpoints (verify each BEFORE the next)
================================================================================
| Stage | What | Checkpoint (must pass) |
|---|---|---|
| 0 | Register free TalkBank account; download TD + clinical corpora to `data/childes/`. Start ISEF SRC forms + line up an adult sponsor. venv ready (done). | Corpora on disk; %mor/%gra tiers present; paperwork in motion. |
| 1 | Ingest/verify: parse all transcripts (pylangacq); build a per-transcript table [child_id, corpus, group(TD/clinical), age_months, mluw, mlum, ttr, ipsyn, n_utterances, vocab_size, ...]. Report coverage (age range, N children, feature availability). | Clean feature table + verification printout. |
| 1b | **CORE SIGNAL-GATE:** child-independent (GroupKFold by child) - do language features predict developmental age BETTER than baseline? (Expected: strongly yes - textbook.) | Held-out age-prediction beats baseline. (If not, STOP - something's wrong with ingest.) |
| 2 | Build the developmental-LEVEL estimator: predict age / developmental stage from features, child-independent, with baselines + bootstrap CIs + calibration. Define an "age-expected vs observed" gap score. | Solid, honest level-estimation performance. This is the de-risked CORE deliverable. |
| 3 | **DELAY-SCREENING SIGNAL-GATE (the risky headline):** TD vs language-impaired classification, child-independent, **controlling for artifacts** (match/adjust for transcript length, #utterances; test cross-corpus). Does a genuine delay signal survive? | If YES -> delay screening is the headline. If NO/artifact -> FALL BACK to Stage 2 core (documented, not a failure). |
| 4 | Interpretability: SHAP -> which linguistic markers drive level/delay -> an actionable readout ("flag: MLU + syntactic complexity well below age norm"). | Ranked drivers + readout logic. |
| 5 | Deployable demo/tool: paste a child's transcript -> estimated developmental level + age-gap + delay flag (if validated) + confidence + drivers. | Working demo (notebook / lightweight web app). |
| 6 | Rigor + board-ready summary: child-independent results, calibration, CIs, subgroup checks (age bands, sex, corpus), artifact controls, honest limitations, Claim-Evidence-Reasoning + poster. | Final results + poster-ready summary. |

================================================================================
## 6. WHY IT WINS (map to real winners) + HONEST CEILING
================================================================================
- ArabLexify (BEHA placement): accessible screener for an underserved group -> FirstWords is an accessible language-delay screen. Same shape.
- Developmental + clinical + accessibility + interpretability = a resonant, judge-favored BEHA story ("could help catch language delay early, for families without easy specialist access").
- Honest ceiling: genuine BEHA placement/finalist potential at Dallas Regional -> ISEF with strong execution + demo + presentation. Not a guaranteed top prize. But it will COMPLETE and be a real, correctly-categorized, working tool.

================================================================================
## 7. REACH GOALS (only after the core is solid; each signal-gated)
================================================================================
- Multilingual screening (CHILDES has 20+ languages) - an equity/accessibility headline (screening where English tools don't exist). Strong ArabLexify-style angle.
- A speech -> transcript front-end (ASR on child speech) for true "record and screen" accessibility - HARD (child ASR is poor); keep as reach.
- Longitudinal trajectory modeling (many CHILDES children are recorded over time).
- Do NOT add unvalidatable outputs; do NOT collect new human data.

================================================================================
## 8. HOW TO RESTART / PICK UP
================================================================================
- Read Sections 0-2. Committed project = FirstWords (child language-development level + delay flag from CHILDES).
- If corpora not downloaded: blocker is the free TalkBank registration (author action). Everything else (venv, pipeline, plan) is ready.
- If data present: run Stage 1 ingest, then the Stage 1b core signal-gate, then Stage 2. Only pursue the delay headline after Stage 3's gate.
- Keep the discipline that finally works: signal first, child-independent validation, baselines, honest CIs, artifact control, one checkpoint at a time, a usable tool + demo, honesty over hype.
- Prior artifacts in this folder: verify_childes_pipeline.py (pipeline proof); DopaLoop lives at /Users/mihir/DopaLoop (MATLAB; do not touch); the ParkinTrack/ALS explorations are documented in git history + memory as evidence of the process working.
- FALLBACKS if FirstWords is ever blocked: ALS-Forecast on PRO-ACT (TMED category, proven DREAM signal) is the strongest non-BEHA option; the decision-making (choices13k) and stress (WESAD) ideas are BEHA alternatives with instant data.
