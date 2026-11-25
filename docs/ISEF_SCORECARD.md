# FirstWords: ISEF Scorecard (November 2026)

An outside-style grade of the project against the ISEF Behavioral & Cognitive Sciences judging
rubric, written by an AI assistant (Claude) from the repo, the handoff and fresh re-runs. It is a
**coaching document for the team, not board text**. Scores are estimates; real judges vary a lot.

## Rubric scores

| Criterion (ISEF weight) | Before Nov 2026 | After Nov 2026 additions | Why |
|---|---|---|---|
| Research question (10) | 8 | 9 | Clear, important, testable. Gap was "early screen" framing with no data under age 4 for the delay flag. Toddler test closes it. |
| Design & methodology (15) | 13 | 14 | Child-independent splits, artifact controls, two locked-box outside labs, benchmark vs black box. Now also a cross-age test. |
| Execution: data, analysis, interpretation (20) | 16 | 18 | Unusually rigorous issue log (22 entries, two self-retractions). Biggest open hole was "audio path never tested on real child speech". Now tested. |
| Creativity (20) | 12 | 14 | Core method is known (Gabani et al., 2009 did ML on ENNI). Credit comes from the angles: artifact control, words-only ASR-robust model, private in-browser demo, fairness audit, and now "trained on 4-10 yr olds, works at 2.5 yr". |
| Presentation: poster (10) | 7 | 7 | Clean flow, but 10 figures is dense, and the text is still an AI draft that ISEF rules require the team to rewrite. |
| Presentation: interview (25) | 17 | 19 | Depends almost entirely on the team. Strong prepared answers exist. Main risk: AI did most of the code, so judges will test ownership. |
| **Total (100)** | **73** | **81** | |

Rough read: before, a strong regional/state project with an ISEF-level shot. After, the two most
likely judge "gotchas" have real answers, which matters most in the interview.

## Strengths judges will reward
- **Problem -> Solution -> Impact is tight.** ~7% prevalence, mostly missed, no "growth chart for language", a tool that needs ~10 sentences.
- **Validation is the real differentiator.** Never tested on a child it trained on; age-alone 0.42 and length-alone 0.49 prove the signal is language; 782 outside children without retraining.
- **Self-caught errors are documented** (ISS-07 data leak, ISS-15 age confound, ISS-16 retracted typology). Few high-school projects can show this.
- **Interpretable beats black box** (0.86 vs 0.82), with per-child "why" output.
- **Working demo** that a judge can touch.

## Weaknesses, ranked by points at risk

1. **Ownership of the work (interview, up to ~10 points).** Heavy AI involvement in code and stats. Judges ask "walk me through how you split the data" or "why logistic regression". If the team can't answer without notes, scores drop fast, and misrepresenting authorship can disqualify. *Fix: team only.* Re-run every script yourselves, read EXPLAINER.md, and be able to draw the validation diagram from memory.
2. **"Early screen" with no early data (fixed now, see below).** The flag was only validated at 4-16 yr.
3. **Recording path never tested on real child speech (now tested, see below: works on child-only audio, fails on raw home audio).**
4. **Real-world usefulness at the screening cutoff.** At 7% prevalence, PPV ~16% and ~39% of a class flagged. A sharp judge will ask "so most flags are false alarms?" *Answer:* yes, that is normal for a first-line screen tuned to miss few kids; NPV ~99% means a pass is trustworthy; published screeners have similar tradeoffs (sensitivity 50-94%, specificity 45-96%). *Added Nov 2026:* `src/ext_operating_points.py` prints the full tradeoff per 100 children (e.g. catch 6.4 of 7 with 36.5 false alarms, vs catch 3.5 of 7 with 7.8 false alarms), so you can explain the choice instead of defending one number.
5. **Fairness gap (ISS-17).** Over-flags typical African American children on Gillam (31% vs 14%). Already handled honestly; keep it interview-only and never claim equal performance.
6. **Prior work comparison.** A BEHA judge who knows Gabani et al. (2009) may ask how your accuracy compares. *Fix:* look up their reported numbers in the paper (do not quote from memory) and have one honest sentence ready: same corpus, different goal (interpretable, artifact-controlled, words-only, works from audio).
7. **Poster density.** Ten figures plus tiles. Judges spend ~1 minute on the board before talking. Consider dropping fig5 or fig9 to the interview binder.

## What was added in November 2026 (all reproducible, scripts in `src/`)

### A. Toddler/preschool validation: `ext_young_delay.py`, figure `fig11_young_children.png`
- Data: EllisWeismer (Clinical-Eng), 130 children, late talkers vs typical, seen at 30, 42, 54, 66 months.
- Test: the ENNI-trained delay flag (trained only on 4-10 yr olds), applied with **no retraining**. One score per child per visit.
- **Result: AUC 0.92 [0.87-0.96] at 30 months**, 0.78 at 42, 0.77 at 54, 0.55 (chance) at 66.
- Reference: a model trained on EllisWeismer itself gets about the same at 30 months (0.89-0.92), so the transferred model is near the ceiling.
- Interpretation (CER):
  - Claim: the screen catches late talkers at 2.5 years even though it never saw a child under 4.
  - Evidence: AUC 0.92 with no refit; late talkers average 1.6 words per sentence vs 2.5 for typical peers.
  - Reasoning: the same markers (sentence length, vocabulary diversity) separate groups at every age; the AUC falls with age because many late talkers catch up, which matches the developmental literature.
- **Cutoff check (ISS-26), important:** AUC measures ranking only. At the ENNI cutoff the flag fires for **100% of 30-month-olds**, typical ones included, because its risk scale is built for 4-10 yr olds. Set from ~34 local *typical* toddlers instead (no diagnosed kids needed), it catches **87%** of late talkers with **77%** specificity on children not used to set it. So: keep the demo's 4-10 yr limit; a toddler mode is a concrete next step, not a current feature.
- Honest caveats to say out loud:
  - "Late talker" is not the same label as DLD/SLI; it is identified by small vocabulary at about 2 years, so 30-month agreement is partly expected. Call it "catches late talkers", not "predicts later disorder".
  - The 54-66 month drop is a real limit: the screen cannot tell which late talkers will stay behind.

### B. Real child-speech ASR test: `ext_asr_childspeech.py`
- Data: Providence (Eng-NA), 5 typical children, one ~1-hour home recording each at ~33 months, with time-aligned human transcripts. Model: Whisper base.en (what the demos use), CPU.
- **Word level:** Whisper gets most toddler words wrong. Word error rate on the child's speech ~0.95 (0.73-1.21 per child).
- **Screen features on child-only audio** (child's own utterances, run through the real `transcribe()`): sentence length off by 11% on average, vocabulary diversity by 10%, language age by **3.8 months**. Close, despite wrong words, because sentence breaks come from pauses. Encouraging but fragile: 5 kids, one age, all typical.
- **Unfiltered home audio:** language age inflated by **~25 months**, because parents' speech is counted as the child's. The "could not screen" guard caught 1 of 5. Wrong direction for a screen: a delayed child would look older and be missed.
- Interpretation (CER):
  - Claim: the recording path is usable only on audio that is mostly the child.
  - Evidence: language age within ~4 months on child-only audio vs ~25 months off on raw home audio.
  - Reasoning: the features depend on counting the child's sentences and words, so any adult speech in the recording directly corrupts them; word identity matters less.
- Next step this points to: speaker separation, or a recording protocol where the adult stays quiet (a story retell, like ENNI's task). A large-v3 GPU run on the PC may lower word error (pending).
- Scoring effect: turns "untested on real kids" into a measured, honest limitation with a specific fix. That is worth more with judges than an untested claim.

## Recommended next moves (team decides)
1. **Board:** consider replacing fig6/fig9 with fig11 (toddler result). It directly answers "does an early screen work early?"
2. **Pitch:** done. The recording answer now uses the Providence numbers.
3. **Interview drill:** practice the 6 weakness questions above out loud with someone who has not seen the project.
4. **Do not add new analyses** after this. Research is closed; remaining points are in presentation.
