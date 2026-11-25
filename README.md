# FirstWords
_Science fair project - behavior research (collab with polaris242)._

An interpretable tool that estimates a young child's **language-developmental level** from a short
speech sample and **flags likely language delay** - an accessible early-screening aid, built on the
public CHILDES child-language database. ISEF project (Behavioral & Cognitive Sciences / Developmental
Psychology).

> A "growth chart for language": a pediatrician plots height against age norms; FirstWords plots a
> child's *talking* (sentence length, vocabulary, grammar) against age norms, and flags kids who look
> behind - so families can seek help sooner.

## Status (2026-07-18)
Both signals verified, and **Stages 2 (core level estimator) and 3 (delay classifier) are built and
validated child-independently**:
- **Core estimator** (Brown + ENNI-TD, 499 files, 281 distinct children, 18-120 mo): ridge regression
  predicts a child's age from transcript features with **MAE 11.2 months, R² 0.76, child-independent**
  (GroupKFold by child; predict-mean baseline MAE 27.1).
- **Delay classifier** (ENNI, TD vs SLI, same task): **ROC-AUC 0.861** [0.81–0.91], child-independent
  (StratifiedGroupKFold by child). Crucially, **age alone (AUC 0.42) and transcript length alone
  (0.49) are at/below chance** — the signal is genuinely linguistic, not an artifact. At a screening
  threshold: 91% sensitivity, 65% specificity.
- **Age-gap score**: TD children ≈ 0; SLI children's estimated language age lags ~16 months (d ≈ 1.0).

**Interpretability + demo + extensions (done):** SHAP drivers + an age-normed per-child readout
(Stage 4); a paste-a-transcript screening **demo** (Stage 5: `.venv/bin/python src/stage5_demo.py --demo`);
a language **growth chart** (percentile curves, well-calibrated); **cross-corpus external validation**
on Conti-Ramsden 4 (AUC 0.70, no refit); **short-sample robustness** (AUC 0.82 at just 10 utterances);
and a **fairness audit** (sex/age subgroups). See the handoff (Sections 0b, 9, 10).

**October 2026:** the screen holds up to transcription errors (AUC 0.85 with 40% of words wrong), gives
consistent answers (split-half reliability 0.90), and runs as a **browser-based web demo** (`web/`;
transcript, typed sentences, or a recording transcribed on-device). Current state, board and demo
details: PROJECT_HANDOFF.md **Section 0d**. Pitch + judging plan: [`docs/PITCH.md`](docs/PITCH.md).

## Learn the project
- [`docs/EXPLAINER.md`](docs/EXPLAINER.md) — teaches the whole project from scratch: the science
  (language development, DLD, the measures, cross-linguistic typology) then the computer science of
  every step (models, validation, metrics, confound control, each extension), plus a judge-question crosswalk.
- [`docs/READING_LIST.md`](docs/READING_LIST.md) — curated background papers by topic, with why-read notes.
- [`docs/METHODS_WALKTHROUGH.md`](docs/METHODS_WALKTHROUGH.md) — granular, code-level walkthrough of each analysis + how to explain it (for the team).

## Repository layout
```
.
├── README.md               # this file
├── PROJECT_HANDOFF.md      # FULL plan, all lessons, data facts, agent rules - READ THIS FIRST
├── requirements.txt        # pinned deps (verified on Python 3.14)
├── src/
│   ├── stage1_check.py             # Stage 1: parse real data + verify core & delay signals
│   ├── stage2_level_estimator.py   # Stage 2: child-independent developmental-level estimator
│   ├── stage3_delay_classifier.py  # Stage 3: TD-vs-SLI delay classifier (ENNI, artifact-controlled)
│   ├── stage4_interpretability.py  # Stage 4: SHAP drivers + age-normed per-child readout
│   ├── stage5_demo.py              # Stage 5: paste-a-transcript screening demo (the judge-facing tool)
│   ├── ext_percentiles.py          # Extension: language "growth chart" (percentile curves)
│   ├── ext_crosscorpus.py          # Extension: cross-corpus generalization (Brown<->ENNI + Conti4 external)
│   ├── ext_shortsample.py          # Extension: short-sample robustness (AUC vs transcript length)
│   ├── ext_fairness.py             # Extension: fairness/subgroup audit (sex, age band)
│   ├── ext_morphprofile.py         # Extension: item-level grammatical profile (which markers flag SLI)
│   ├── ext_calibration.py          # Extension: delay-flag probability calibration (reliability, Brier)
│   ├── ext_taskreliability.py      # Extension: cross-task screening reliability (Conti4 frog vs conversation)
│   ├── ext_multilingual.py         # Extension: multilingual level estimator (English/German/Cantonese)
│   ├── ext_crosslingual_delay.py   # Extension: cross-lingual delay detection (German TD vs impaired)
│   ├── ext_unified_crosslingual.py # Extension: unified cross-lingual model (honest: per-language norms needed)
│   ├── ext_typology.py             # Tested + RETRACTED: marker-vs-typology hypothesis (Mandarin broke it)
│   ├── ext_clinical_utility.py     # Extension: decision-curve analysis (net benefit vs default policies)
│   ├── ext_benchmark.py            # Benchmark: interpretable features vs TF-IDF text model + naive rule
│   ├── demo_report.py              # Visual HTML demo report for judges (offline, no server)
│   ├── demo_server.py              # Offline backup demo app (localhost:5050)
│   ├── audio_screen.py             # Record-and-screen prototype (faster-whisper)
│   ├── export_web_model.py         # Trains + exports the words-only model for the web demo
│   ├── ext_asr_stress.py           # Transcription-error stress test
│   ├── ext_reliability.py          # Split-half reliability
│   ├── ext_fairness_ethnicity.py   # Ethnicity fairness audit on Gillam (interview only)
│   ├── ext_fairness_fix.py         # Exploratory: does dropping vocabulary shrink the gap?
│   ├── poster_figures.py           # Board figures fig1-fig8 (print-sized, recomputed)
│   ├── poster_figures_extra.py     # Board figures fig9-fig10
│   ├── ext_young_delay.py          # Nov 2026: toddler validation on EllisWeismer (late talkers, 30-66 mo) + cutoff check
│   ├── poster_figures_young.py     # Nov 2026: fig11 (toddler AUC by age) + fig12 (both directions)
│   ├── ext_asr_childspeech.py      # Nov 2026: real child-speech ASR test (Providence home audio)
│   ├── ext_voice_filter.py         # Nov 2026: child-vs-adult voice filter from the Whisper encoder (ISS-31)
│   ├── ext_operating_points.py     # Nov 2026: false alarms vs misses per 100 children at each cutoff
│   ├── ext_continuity.py           # Nov 2026: toddler late talking <-> school-age DLD, transfer both ways (fig12)
│   ├── ext_typical_only_calibration.py # Nov 2026: set the cutoff in a new population from typical kids only
│   └── verify_pipeline.py          # no-download smoke test of the CHAT->features pipeline
├── web/                    # Browser web demo (index.html, app.js, model.json); web/models/ gitignored
├── scripts/                # fetch_web_model.sh (speech model files for the web demo)
├── docs/                   # EXPLAINER, METHODS_WALKTHROUGH, READING_LIST, PITCH, ISEF_SCORECARD, INTERVIEW_DRILL
├── data/                   # (gitignored) CHILDES corpora: Brown (TD), ENNI (TD+SLI)
└── results/                # (gitignored) outputs / figures
```

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Download CHILDES corpora (free TalkBank account) into data/childes/  (Brown, ENNI, ...)
.venv/bin/python src/stage1_check.py     # confirm the signals on your data
```

## Data
CHILDES via TalkBank (https://childes.talkbank.org) - free account required to download; de-identified;
used under TalkBank terms. Corpora are NOT committed (gitignored). See PROJECT_HANDOFF.md Section 4.
