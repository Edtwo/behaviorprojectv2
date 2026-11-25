# FirstWords: Board Review (Nov 25, 2026)

Review of the Canva board ("Science fair poster board", page 1) against the current results.
**Coaching only: no board wording here.** ISEF bars AI-written poster text, so every fix below
says *what* is wrong and *why*; the team writes the words.

## 1. Must fix (accuracy: a judge can catch these)

| Where on the board | Problem | Evidence |
|---|---|---|
| Section 6, the PROTOTYPE note ("not yet tested on recorded child speech") | No longer true. It was tested on real home recordings of 5 toddlers, and a voice filter was built and tested. A judge who hears the pitch and then reads this line sees a contradiction. | ISS-24, ISS-31, fig13 |
| Section 9, "Next: real child recordings" | Done. The real next steps are now: put the child-voice filter in the demo, and fix sentence splitting on short recordings. | ISS-31 |
| Section 5 tile and Section 8 ("30 local typical kids ... catches 80%") | True for Gillam **only at about 35% false alarms** (specificity 0.65, range 0.47-0.81 with just 30 kids). That is the same false-alarm rate as the ENNI screening point, so it is fair, but the tile hides the cost. Know this cold, and consider showing the false-alarm side. At 20% false alarms, 30 typical kids catch 64%. | re-run 2026-11-25 (1,000 random draws of 30 typical Gillam children); ISS-30 |
| Title ("Validated Across Labs and Ages") | "Ages" is now backed by the toddler test (AUC 0.92 at 2½ years), but nothing on the board shows it. Either show fig11 or be ready to explain it in one sentence when a judge points at the title. | ISS-23, fig11 |

## 2. Figure swaps (stronger evidence, same or less space)

| Section | Now | Suggested | Why |
|---|---|---|---|
| 6 Screen from a recording | fig9 (simulated transcription errors) | **fig13 real recordings** (new, `src/poster_figures_audio.py`) | Real toddler audio beats a simulation. It shows the problem (raw: +12 months) and your fix (+4 months) in one picture, which is a strong problem → solution → impact story. Keep "0.85 with 40% wrong words" as a single number if you want it. |
| 7 Does it hold up? | fig4 + fig5 (black-box benchmark) | fig4 + **fig12 both directions** | fig12 is your most original result (creativity is 20 points). The black-box result already appears as a sentence in Section 8, so fig5 can move to the interview binder. |
| 5 Key results | 4 tiles + 3 figures | drop one tile | It is the densest block. The "10 utterances" tile repeats the Conclusion. |

Net: the same number of figures, with two simulated or duplicate items swapped for real-data and original results.

## 3. Layout and readability (from the board thumbnail)
- **Section 8 Conclusion** is the most text-heavy box. Each "Supported" line repeats numbers already shown in Section 5. Shorter lines that point back to the figures read faster.
- **Read the board from 6 feet away** (or zoom the Canva view to 15%). Any text you can't read at that size, a judge walking by can't either.
- **Credit line:** fig13 is made from TalkBank Providence data (Demuth et al., 2006). Add Providence and Ellis Weismer (if you use fig11/fig12) to the references, since the board's references don't list them yet. Find the exact citations on TalkBank's corpus pages yourselves.
- **Every figure needs its credit line** ("made by the team in Python"), as the board already does for the existing figures.

## 4. Files
- `results/poster/fig13_real_recordings.png`: new, print-sized, transparent background, same style as fig1-fig12.
- Regenerate it with `.venv/bin/python src/poster_figures_audio.py` (it needs `ext_asr_childspeech.py` and `ext_voice_filter.py` results).
