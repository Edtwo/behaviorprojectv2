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

Next: Stage 4 (SHAP interpretability) and Stage 5 (the demo tool). See the handoff.

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
