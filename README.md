# FirstWords
_Science fair project - behavior research (collab with polaris242)._

An interpretable tool that estimates a young child's **language-developmental level** from a short
speech sample and **flags likely language delay** - an accessible early-screening aid, built on the
public CHILDES child-language database. ISEF project (Behavioral & Cognitive Sciences / Developmental
Psychology).

> A "growth chart for language": a pediatrician plots height against age norms; FirstWords plots a
> child's *talking* (sentence length, vocabulary, grammar) against age norms, and flags kids who look
> behind - so families can seek help sooner.

## Status (2026-07-16)
Both signals verified on real data, and **Stage 2 (the core level estimator) is built and validated**:
- **Core estimator** (Brown + ENNI-TD, 499 files, 288 distinct children, 18-120 mo): ridge regression
  predicts a child's age from transcript features with **MAE 11.5 months, R² 0.74, child-independent**
  (GroupKFold by child; predict-mean baseline MAE 27.8).
- **Age-gap score** (predicted language age − actual age): centered near 0 for typically-developing
  children; **language-impaired (SLI) children lag ~16 months** on the same task (Cohen's d ≈ 1.0) —
  informal preview; Stage 3 makes it rigorous.

Next: Stage 3 (delay classifier with age/length covariates). See the handoff.

## Repository layout
```
.
├── README.md               # this file
├── PROJECT_HANDOFF.md      # FULL plan, all lessons, data facts, agent rules - READ THIS FIRST
├── requirements.txt        # pinned deps (verified on Python 3.14)
├── src/
│   ├── stage1_check.py             # Stage 1: parse real data + verify core & delay signals
│   ├── stage2_level_estimator.py   # Stage 2: child-independent developmental-level estimator
│   └── verify_pipeline.py          # no-download smoke test of the CHAT->features pipeline
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
