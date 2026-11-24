# FirstWords: the pitch

Draft talking points. Say them in your own words; don't memorize this word for word. Judges
want a conversation, not a recital. Every number here is checked against the code (see
results/poster/numbers.txt and web/model.json).

Shape: **problem → question → what we found → why it matters**, then the demo.

---

## The 30-second version (hallway, or a judge with no time)

- Seven in every hundred children have a language disorder. That's about two in every classroom, and most are never caught.
- A real evaluation needs a specialist, and many families can't reach one. So we asked: how little speech, and how little expertise, does it actually take to detect it?
- The answer: about ten sentences, scored on simple measures anyone can compute. On children it had never seen, it catches 9 in 10 kids with the disorder, and it works on children from other labs without retraining.

## The full pitch (about 2 minutes, then demo)

**1. The problem (20 s)**
- Developmental Language Disorder affects about 7% of children (Tomblin et al., 1997), more than autism.
- Most affected kids are missed. In that same study, only about 3 in 10 parents had been told their child had a problem. [verify the 29% figure in the paper before saying it]
- Untreated, it follows kids into reading and school, and early help works (Law et al., 2004).
- Height has a growth chart any pediatrician reads in seconds. Language has nothing like it, and a full evaluation needs a specialist.

**2. Our question (15 s)**
- How little speech, and how little expertise, does it take to detect it? We tested four hypotheses, each of which could have failed.

**3. What we found (45 s)** (point at the board as you go)
- *The signal is real language.* 0.86 on children the model never saw, where 0.5 is a coin flip. Age alone and transcript length alone score like coin flips (0.42, 0.49), so it isn't cheating.
- *It works as a screen.* We set the cutoff without ever looking at the test children, and it still catches 91%. In a class of 100, it clears 60 kids, and 98% of those really are typical, so specialists only need to see the rest.
- *Ten sentences are enough, and no grammar coding is needed.* 0.82 from ten sentences; words alone do as well as hand-coded grammar (0.857 vs 0.861); it holds at 0.85 even when 40% of the words are transcribed wrong.
- *It transfers.* 782 children from labs in the USA and the UK, no retraining: 0.80 and 0.70. And it adapts to a new place with just 30 typical local kids: that recalibration catches about 80%.

**4. Why it matters, and what's next (15 s)**
- A short, explainable screen means a teacher or pediatrician could get an early warning without a specialist, then send the right kids for a full evaluation.
- We also checked it for fairness: vocabulary measures over-flag typically developing Black children, sentence length doesn't, so our next version leans on sentence length.
- Next: real child recordings, and local norms for more communities and languages (language measures already track age in German and Cantonese).

**5. Hand off to the demo:** "Want to try it?"

---

## Lines for when a judge pushes

- **"Did you pick the cutoff on your test data?"** No. We choose it inside the training folds only, then apply it to children the model never saw: still 91% caught, 64% of typical children cleared.
- **"Your outside lab only caught about 60%."** At the original cutoff, yes; the ranking transfers (0.80) but every population needs its own reference point, like growth charts built for different countries. With 30 typical local kids it catches about 80%.
- **"How accurate is the language age?"** Across childhood the measures track age closely (r = 0.90), but for one 4-to-10-year-old it's only good to about a year, so the screen leads with the growth-chart percentile and treats language age as a rough guide.
- **"Isn't AUC 0.86 just okay?"** It's measured on children the model never saw, with confidence intervals. It held up on other labs' kids without retraining. A lot of high numbers you'll see at a fair come from testing on training data.
- **"Why not deep learning?"** We tested a 3,000-feature text model. It wasn't more accurate (0.82 vs 0.86), and a clinician can't see why it flags a child. Ours can.
- **"Does the recording version work on real kids?"** It's a prototype. The public clinical datasets don't share their audio, so we couldn't test that step on clinical recordings yet. That's our top next step. The words-only model itself is validated.
- **"Is this a diagnosis?"** No. It's a screen that says who should see a specialist sooner.
- **"Who did what / how did you use AI?"** Answer honestly and specifically, matching what you wrote in your references and notebook.

---

## The demo during judging

**Use the web demo on the laptop, in this order (about 60 seconds):**
1. **Drag in a typical child's transcript** (demo/transcripts/typical_1_age103mo.cha). The result is not elevated: 78th percentile for sentence length.
2. **Drag in an impaired child's transcript** (impaired_1_age62mo.cha). The result is ELEVATED, sentence length at about the 6th percentile, and the dot sits low on the growth chart. Say: "Neither child was ever used to train the model."
3. **Let the judge play.** Open "Type what the child said," hand them the keyboard, and have them type a few short sentences a 5-year-old might say. Judges remember what they touched.
4. **Recording: the judge reads a script.** Set the age to 6 years 0 months, open the Record tab, and open "No child here? Read a demo script aloud." Have the judge read Script A (on track; not flagged), then Script B (short, simple sentences; flagged, about the 10th percentile), pausing briefly after each sentence. Say: "Same voice both times. It measures what is said, not how it sounds." Don't let them just talk normally: adult sentences are about 15 words long and the demo will correctly refuse to screen them. Keep a pre-recorded clip of each script on the laptop in case the hall is too loud.

**Before judges arrive:**
- Open the web demo and run one recording once, so the speech model is already downloaded and cached.
- Bring a phone hotspot; fair Wi-Fi is unreliable. ISEF says you must supply your own internet if you need it.
- Charge the laptop. Many fairs don't give power for laptops.

**Backups, in order:**
1. The local version (no internet needed): `.venv/bin/python src/demo_server.py`, then open http://localhost:5050
2. The pre-made reports in demo/reports/: open any .html file in a browser.

**Display rule:** don't put the web address or a QR code on the board. ISEF bans URLs and QR codes on displays. Show the demo on the laptop instead.
