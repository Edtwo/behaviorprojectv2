# FirstWords
_Science fair project - behavior research (collab with polaris242)._

An interpretable tool that estimates a young child's **language-developmental level** from a short
speech sample and **flags likely language delay** - an accessible early-screening aid, built on the
public CHILDES child-language database. ISEF project (Behavioral & Cognitive Sciences / Developmental
Psychology).

> A "growth chart for language": a pediatrician plots height against age norms; FirstWords plots a
> child's *talking* (sentence length, vocabulary, grammar) against age norms, and flags kids who look
> behind - so families can seek help sooner.

## Status (2026-07-14)
Data downloaded and BOTH signals verified on real data:
- **Core** (Brown, typically-developing, 214 files, ages 18-62 mo): language complexity rises with age
  (corr(age, MLU)=0.68) -> the developmental-level estimator will work.
- **Delay** (ENNI, TD vs SLI, matched age ~85 mo): language-impaired children have markedly lower MLU
  (5.67 vs 6.90, ~1 SD) on the same task -> the delay-screening headline looks viable.

Next: build Stage 2 (developmental-level estimator) then Stage 3 (delay signal-gate). See the handoff.

## Repository layout
```
.
├── README.md               # this file
├── PROJECT_HANDOFF.md      # FULL plan, all lessons, data facts, agent rules - READ THIS FIRST
├── requirements.txt        # pinned deps (verified on Python 3.14)
├── src/
│   ├── stage1_check.py     # Stage 1: parse real data + verify core & delay signals
│   └── verify_pipeline.py  # no-download smoke test of the CHAT->features pipeline
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
