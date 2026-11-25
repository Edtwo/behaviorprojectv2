# FirstWords: the pitch

Draft talking points. Say them in your own words; don't memorize this word for word. Judges
want a conversation, not a recital. Every number here is checked against the code (see
results/poster/numbers.txt and web/model.json).

Shape: **problem → solution → effective → accessible → important**, then the demo.

---

## The 30-second version (hallway, or a judge with no time)

- Seven in every hundred children have a language disorder. That's about two in every classroom, and most are never caught.
- Pediatricians track height on a growth chart. Language has nothing like that.
- We built one: give FirstWords a few minutes of a child's speech and it estimates their language age and flags likely delay.
- On children it had never seen, it catches 9 out of 10 kids with a language disorder. It works on children from other labs and countries. Our working demo runs in an ordinary web browser.

## The full pitch (about 2 minutes, then demo)

**1. The problem (20 s)**
- Developmental Language Disorder affects about 7% of children (Tomblin et al., 1997). That's more common than autism.
- Most affected kids are missed. In that same study, only about 3 in 10 parents had been told their child had a problem. [verify the 29% figure in the paper before saying it]
- It matters because untreated language problems follow kids into reading and school, and early help works (Law et al., 2004).
- Why are they missed? Height has a growth chart that any pediatrician can read in seconds. Language has no simple equivalent. A full evaluation needs a speech-language pathologist, and many families can't get to one.

**2. Our solution (20 s)**
- FirstWords is a growth chart for language.
- It takes a short sample of a child's speech: a transcript, typed sentences, or a recording.
- It measures four simple things: how long the child's sentences are (in words and in word parts), how varied their vocabulary is, and their age.
- From those it gives three answers: an estimated language age, where the child sits on a growth chart, and a delay flag.
- It always shows *why*: "sentence length is 1.1 standard deviations below age norms." No black box.

**3. It's effective (30 s)** (point at the hero chart)
- We tested it the strict way: never on a child it trained on. It scores 0.86 out of 1.0, where 0.5 is a coin flip, and catches 91% of children with a language disorder.
- We checked it wasn't cheating. Age alone and transcript length alone score at coin-flip level (0.42 and 0.49). The signal is the child's language.
- Then we tested it on 782 children from two other labs, in the USA and the UK, without changing anything. It still worked: 0.80 and 0.70.
- And our four simple measures are as accurate as a black-box AI with 3,000 features, but ours can explain every decision.
- The signal even reaches younger than we trained. We trained only on 4- to 10-year-olds, then tested on 2½-year-olds from another lab: it still ranked late talkers below typical talkers at 0.92. (fig11) To actually flag toddlers it needs a toddler cutoff, set from typical toddlers; with one it catches about 87%.

**4. It's accessible (20 s)**
- About 10 sentences are enough to screen (0.82).
- A version using only the words, which is all an automatic transcript of a recording gives you, is just as accurate.
- It still scores 0.85 even when 40% of the words are transcribed wrong.
- So our demo screens from a recording: it transcribes the speech automatically in the browser and the recording never leaves the computer. It's a prototype, not a finished app, but it shows the path to screening without a specialist or a transcription service.

**5. Why it matters, and what's next (15 s)**
- A parent, teacher or pediatrician anywhere could get an early warning in minutes, then go to a specialist with a reason.
- It's a screening aid, not a diagnosis.
- Next: test the recording path on real child recordings, add more languages (it already works in German and Cantonese), and partner with clinicians.

**6. Hand off to the demo:** "Want to try it?"

---

## Lines for when a judge pushes

- **"Isn't AUC 0.86 just okay?"** It's measured on children the model never saw, with confidence intervals. It held up on other labs' kids without retraining. A lot of high numbers you'll see at a fair come from testing on training data.
- **"Why not deep learning?"** We tested a 3,000-feature text model. It wasn't more accurate (0.82 vs 0.86), and a clinician can't see why it flags a child. Ours can.
- **"Does the recording version work on real kids?"** It's a prototype. The public clinical datasets don't share their audio, so we couldn't test that step on clinical recordings yet. That's our top next step. The words-only model itself is validated.
- **"Does it work on toddlers, or just school-age kids?"** We trained on 4-10 year olds, then tested on 130 children from the Ellis Weismer study, with no retraining. At 2½ years it separates late talkers from typical talkers at 0.92. The score drops with age (0.78 at 3½, 0.55 at 5½) because many late talkers catch up, which is what the research literature says happens.
- **"So why does your demo only flag ages 4-10?"** Because the cutoff is tuned on 4-10 year olds. Used on toddlers it flags every child, since all toddlers have short sentences. The *ranking* still works (0.92), so the fix is a toddler cutoff. We tested setting it from about 35 typical toddlers only, no diagnosed kids needed: it caught 87% of late talkers and cleared 77% of typical ones, on children not used to set it. That's our next step, not a finished feature.
- **"So can it predict which toddlers will have a disorder later?"** No, and we don't claim that. "Late talker" is decided by vocabulary size at age 2, not a disorder diagnosis. By 5½ our screen can't tell which late talkers stayed behind. It catches who is behind *now*.
- **"Is this a diagnosis?"** No. It's a screen that says who should see a specialist sooner.
- **"Who did what / how did you use AI?"** Answer honestly and specifically, matching what you wrote in your references and notebook.

---

## The demo during judging

**Use the web demo on the laptop, in this order (about 60 seconds):**
1. **Drag in a typical child's transcript** (demo/transcripts/typical_1_age103mo.cha). The result is not elevated: language age about 104 months for a 103-month-old.
2. **Drag in an impaired child's transcript** (impaired_1_age62mo.cha). The result is ELEVATED, sentence length at about the 6th percentile, and the dot sits low on the growth chart. Say: "Neither child was ever used to train the model."
3. **Let the judge play.** Open "Type what the child said," hand them the keyboard, and have them type a few short sentences a 5-year-old might say. Judges remember what they touched.
4. **Recording: the judge reads a script.** Set the age to 6 years 0 months, open the Record tab, and open "No child here? Read a demo script aloud." Have the judge read Script A (on track; not flagged, language age about 79 months), then Script B (short, simple sentences; flagged, about 50 months), pausing briefly after each sentence. Say: "Same voice both times. It measures what is said, not how it sounds." Don't let them just talk normally: adult sentences are about 15 words long and the demo will correctly refuse to screen them. Keep a pre-recorded clip of each script on the laptop in case the hall is too loud.

**Before judges arrive:**
- Open the web demo and run one recording once, so the speech model is already downloaded and cached.
- Bring a phone hotspot; fair Wi-Fi is unreliable. ISEF says you must supply your own internet if you need it.
- Charge the laptop. Many fairs don't give power for laptops.

**Backups, in order:**
1. The local version (no internet needed): `.venv/bin/python src/demo_server.py`, then open http://localhost:5050
2. The pre-made reports in demo/reports/: open any .html file in a browser.

**Display rule:** don't put the web address or a QR code on the board. ISEF bans URLs and QR codes on displays. Show the demo on the laptop instead.
