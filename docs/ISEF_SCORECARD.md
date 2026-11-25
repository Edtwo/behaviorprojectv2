# FirstWords: ISEF Scorecard (November 2026)

An outside-style grade of the project against the ISEF Behavioral & Cognitive Sciences judging
rubric, written by an AI assistant (Claude) from the repo, the handoff and fresh re-runs. It is a
**coaching document for the team, not board text**. Scores are estimates; real judges vary a lot.

## Rubric scores

| Criterion (ISEF weight) | Before Nov 2026 | After Nov 2026 additions | Why |
|---|---|---|---|
| Research question (10) | 8 | 9 | Clear, important, testable. Gap was "early screen" framing with no data under age 4. The toddler test shows the signal reaches 2.5 yr (ranking), and names what a toddler mode still needs (its own cutoff). |
| Design & methodology (15) | 13 | 14 | Child-independent splits, artifact controls, two locked-box outside labs, benchmark vs black box. Now also a cross-age test. |
| Execution: data, analysis, interpretation (20) | 16 | 19 | Unusually rigorous issue log (now 31 entries, two self-retractions, one failed gate). The recording path is tested on real toddler audio, and its main failure (parents' speech) now has a measured partial fix (ISS-31). |
| Creativity (20) | 12 | 17 | Core method is known (Gabani et al., 2009 did ML on ENNI). New credit: (1) a developmental-science finding: toddler late talking and school-age DLD share one language dimension, since the transfer works in BOTH directions (ISS-29, fig12); (2) a new deployable method: calibrate the cutoff for a new population with typical children only, matching full-label performance (ISS-30); (3) a child-voice filter made from the app's own speech model (ISS-31). Plus artifact control, words-only ASR-robust model, private in-browser demo, fairness audit. |
| Presentation: poster (10) | 7 | 7 | Clean flow, but dense, and the text is still an AI draft the team must rewrite. Board review (Nov 25) found 3 lines now contradicted by results (Section 6 "not yet tested", Section 9 "next: real recordings", the "30 typical kids" tile hides a 35% false-alarm cost). Fixes + fig13/fig12 swaps listed in BOARD_REVIEW.md: about 9 once the team applies them. |
| Presentation: interview (25) | 17 | 19 | Depends almost entirely on the team. Prepared answers now cover toddlers, false alarms and real recordings. Main risk: AI did most of the code, so judges will test ownership. |
| **Total (100)** | **73** | **85** | |

Rough read: before, a strong regional/state project with an ISEF-level shot. After, the two most
likely judge "gotchas" ("does it work on toddlers?", "does the recording work on real kids?") have
measured, honest answers, including where each one stops working. That matters most in the interview.

## Strengths judges will reward
- **Problem -> Solution -> Impact is tight.** ~7% prevalence, mostly missed, no "growth chart for language", a tool that needs ~10 sentences.
- **Validation is the real differentiator.** Never tested on a child it trained on; age-alone 0.42 and length-alone 0.49 prove the signal is language; 782 outside children without retraining.
- **Self-caught errors are documented** (ISS-07 data leak, ISS-15 age confound, ISS-16 retracted typology). Few high-school projects can show this.
- **Interpretable beats black box** (0.86 vs 0.82), with per-child "why" output.
- **Working demo** that a judge can touch.
- **Two original ideas (Nov 2026):** one shared language dimension from age 2 to school age, and calibrating a screen with typical children only. Both turn an engineering project into one that says something new.

## Weaknesses, ranked by points at risk

1. **Ownership of the work (interview, up to ~10 points).** Heavy AI involvement in code and stats. Judges ask "walk me through how you split the data" or "why logistic regression". If the team can't answer without notes, scores drop fast, and misrepresenting authorship can disqualify. *Fix: team only.* Re-run every script yourselves, read EXPLAINER.md, and be able to draw the validation diagram from memory.
2. **"Early screen" with no early data (partly fixed, see A).** The signal ranks late talkers at 2.5 yr (AUC 0.92), but the current cutoff flags every toddler. Never say "the flag works on toddlers".
3. **Recording path on real children (now tested, see B).** Works on child-only audio. Raw home audio counts parents' speech; a child-voice filter (ISS-31) fixes most of that on full recordings but not on short clips, and is not in the demo yet.
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
  - Claim: the screen's language markers rank late talkers below typical peers at 2.5 years, even though it never saw a child under 4.
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
- **Bigger model check (Whisper large-v3, same 5 children, cloud CPU):**

  | Measure | base.en | large-v3 |
  |---|---|---|
  | Word error rate, child-only | 0.95 | 0.84 (better for all 5 kids) |
  | Language-age error, child-only | 3.8 mo | 3.0 mo |
  | Language age, raw home audio | +25 mo (+15 without one broken child) | +10 mo |
  | Raw-audio results the guard caught | 1 of 5 | 0 of 5 (no wild failures) |

  - Read: a better recognizer helps a little on child-only audio and removes the one catastrophic raw-audio failure, but raw audio is still about 10 months too old for every child. Parents' speech remains the main problem, and better word accuracy does not fix it.
- **Child-voice filter (ISS-31, `ext_voice_filter.py`):** the app's own speech model also hears *who* is talking. Trained on 4 families, tested on the 5th, it tells child from adult sentences at AUC 0.94 (pitch: 0.73-0.84).
  - On full 1-hour raw recordings: language-age error **+11.7 -> +4.2 months**, 4 of 5 kids within 4 months. That is about as good as child-only audio (3.8).
  - On 15-minute clips it does not help: there Whisper glues speech into long run-on sentences, which a speaker filter cannot fix.
  - Also found: the small Whisper model gives different transcripts on repeat runs of the same audio, so single-run raw-audio numbers are noisy.
  - CER: Claim: speaker separation is doable with the model the app already ships. Evidence: +11.7 -> +4.2 mo on unseen families. Reasoning: the encoder's representation of a voice carries age information that pitch alone misses, because parents raise their pitch for toddlers but not their whole voice.
- Next step this points to: put the filter in the demo, and fix run-on segmentation on short recordings (or ask for longer ones).
- Scoring effect: turns "untested on real kids" into a measured, honest limitation with a specific fix. That is worth more with judges than an untested claim.

### C. Creativity: two original ideas (ISS-29, ISS-30)
- **One dimension from toddlers to school age** (`ext_continuity.py`, `fig12_both_directions.png`): with no age and no retraining, the school-age model ranks 2½-year-old late talkers at 0.92, and a model trained only on 2½-year-olds ranks school-age DLD at 0.77 (ENNI) and 0.78 (Gillam), as well as ENNI's own model (0.78). Both models lean hardest on sentence length in word-parts.
  - Claim: toddler late talking and school-age DLD sit on one shared, measurable language dimension.
  - Evidence: transfer in both directions, same dominant marker.
  - Reasoning: if they were unrelated conditions, a model of one would not rank the other; the school-age model also weights grammatical endings, matching what is known about DLD.
  - Caveat: shared marker dimension, not proof that the same children go on to DLD (needs longitudinal data). Find and cite the late-talker/DLD continuity papers yourselves.
- **Calibrate with typical kids only** (`ext_typical_only_calibration.py`): a new population sets its cutoff from typical children alone. On Gillam it hits the target false-alarm rate exactly and matches the sensitivity full labels would give (0.64 vs 0.64 at 80% specificity); on toddlers, ~34 typical kids are enough.
  - Why judges care: clinics rarely have diagnosed children to tune a tool, but typical children are everywhere. This is what makes the screen portable.

## Recommended next moves (team decides)
1. **Board:** follow `docs/BOARD_REVIEW.md`: fix the 3 outdated lines, swap fig9 -> fig13 (real recordings) and fig5 -> fig12 (both directions). Team writes all wording.
2. **Pitch:** done. The recording answer now uses the Providence numbers.
3. **Interview drill:** use `docs/INTERVIEW_DRILL.md` (28 questions, pointers to evidence, no scripted answers). Practice out loud with someone who has not seen the project.
4. **Demo check done:** every Gillam child matching the 4 demo transcripts' ages scores outside the 0.36-0.45 band, so the demo outcomes are unchanged by the new cutoff (ISS-25). Still open the 4 HTML reports once before judging.
5. **Do not add new analyses** after this. Research is closed; remaining points are in presentation.
