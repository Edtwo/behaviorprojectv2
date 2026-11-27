# FirstWords
_ISEF project, Behavioral & Cognitive Sciences (Developmental Psychology)._

**Research question:** how little speech, and how little expertise, does it take to detect
Developmental Language Disorder (DLD) in school-age children? DLD affects about 7% of children
(about two per classroom) and is mostly missed; a full evaluation needs a specialist many families
cannot reach. We test whether a few simple, explainable measures of a short speech sample can screen
for it, using 1,100+ children from the public CHILDES database (included in `data/childes/`).

## Main results (all child-independent: no child is ever in both training and testing)
| Hypothesis | Result |
|---|---|
| The signal is language, not age or how much the child talks | ROC-AUC **0.86** [0.81-0.91] on 345 held-out ENNI children; age alone 0.42, transcript length alone 0.49 |
| It works as a screen | Cutoff chosen inside the training folds only: catches **91%** (68/75) of children with DLD, clears 64% of typical children; at 7% prevalence it clears 6 in 10 children, 98% of them correctly |
| About 10 sentences are enough | AUC 0.82 from 10 utterances; 0.85 even with 40% of words transcribed wrong |
| No grammar coding is needed | Words-only screen AUC 0.857 vs 0.861 with morpheme coding |
| It transfers to other labs | No retraining: Gillam (USA, 664 children) AUC 0.80; Conti-Ramsden (UK, 118 teens) 0.70. A cutoff recalibrated from 30 local typical children catches ~80% at Gillam |
| It gives the same answer twice | Split-half reliability 0.90 |

Also: as accurate as a 3,000-feature text model (0.82, overlapping intervals) while explaining every
decision; an age-normed growth chart (9.7% of typical children fall below the 10th percentile, target 10%);
an equity audit (vocabulary measures over-flag typically developing Black children, 31% vs 14%;
sentence length shows no significant gap); language measures also track age in German and Cantonese.
Limits we state: a screening aid, not a diagnosis; validated for ages 4-10; the recording path is a
prototype not yet tested on real child audio; an individual "language age" at 4-10 is only good to
about +-13 months, so the readout leads with the growth-chart percentile.

**Demo:** a browser-based prototype (`web/`) screens a transcript, typed sentences, or a recording
transcribed on the device. Full current state: PROJECT_HANDOFF.md **Section 0d**. Pitch and judging
plan: [`docs/PITCH.md`](docs/PITCH.md). Team ownership exercise: [`exercises/`](exercises/reimplement_classifier/README.md).

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
│   ├── ext_operating_point.py      # Strict cutoff (chosen in training folds), outside-lab recalibration
│   ├── ext_level_by_range.py       # Language-age accuracy within each age range
│   ├── ext_asr_stress.py           # Transcription-error stress test
│   ├── ext_reliability.py          # Split-half reliability
│   ├── ext_fairness_ethnicity.py   # Ethnicity fairness audit on Gillam (interview only)
│   ├── ext_fairness_fix.py         # Exploratory: does dropping vocabulary shrink the gap?
│   ├── poster_figures.py           # Board figures fig1-fig8 (print-sized, recomputed)
│   ├── poster_figures_extra.py     # Board figures fig9-fig10
│   ├── ext_young_delay.py          # Nov 2026: toddler validation on EllisWeismer (late talkers, 30-66 mo) + cutoff check
│   ├── poster_figures_young.py     # Nov 2026: fig14 (toddler AUC by age) + fig12 (both directions)
│   ├── ext_asr_childspeech.py      # Nov 2026: real child-speech ASR test (Providence home audio)
│   ├── ext_voice_filter.py         # Nov 2026: child-vs-adult voice filter from the Whisper encoder (ISS-36)
│   ├── export_voice_filter.py      # Nov 2026: writes models/voice_filter.json for the offline demo's --child-only option
│   ├── poster_figures_audio.py     # Nov 2026: fig13 (real home recordings: raw vs voice filter vs child-only)
│   ├── ext_operating_points.py     # Nov 2026: false alarms vs misses per 100 children at each cutoff
│   ├── ext_continuity.py           # Nov 2026: toddler late talking <-> school-age DLD, transfer both ways (fig12)
│   ├── ext_typical_only_calibration.py # Nov 2026: set the cutoff in a new population from typical kids only
│   └── verify_pipeline.py          # no-download smoke test of the CHAT->features pipeline
├── web/                    # Browser web demo (index.html, app.js, model.json); web/models/ gitignored
├── scripts/                # fetch_web_model.sh (speech model files for the web demo)
├── docs/                   # EXPLAINER, METHODS_WALKTHROUGH, READING_LIST, PITCH, ISEF_SCORECARD, BOARD_REVIEW, INTERVIEW_DRILL
├── data/childes/           # the public CHILDES corpora used (CC BY-NC-SA; see data/README.md)
├── exercises/              # ownership exercise: rebuild the classifier yourself
└── results/                # outputs (gitignored) except poster/ figures + summary CSVs
```

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# the CHILDES data is already in data/childes/ - no download needed
.venv/bin/python src/stage2_level_estimator.py   # builds results/features.csv (needed by most scripts)
.venv/bin/python src/stage3_delay_classifier.py  # headline delay result (AUC 0.861)
.venv/bin/python src/ext_operating_point.py      # strict cutoff + outside-lab recalibration
```

## Data
CHILDES via TalkBank (https://talkbank.org/childes/), de-identified child-language transcripts, used
under the CC BY-NC-SA license. The corpora the project uses are included in `data/childes/`; read
[`data/README.md`](data/README.md) for the license, required citations, and the rule against any
attempt to identify a child.
