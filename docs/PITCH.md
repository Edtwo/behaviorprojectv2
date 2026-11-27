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
- *It reaches younger than we trained.* Trained only on 4- to 10-year-olds, it still ranks 2½-year-old late talkers below typical talkers at 0.92 (fig14). To actually flag toddlers it needs its own cutoff, set from typical toddlers; with one it catches about 87%.

**4. Why it matters, and what's next (15 s)**
- A short, explainable screen means a teacher or pediatrician could get an early warning without a specialist, then send the right kids for a full evaluation.
- We also checked it for fairness: vocabulary measures over-flag typically developing Black children, sentence length doesn't, so our next version leans on sentence length.
- Next: put our child-voice filter into the web demo (it's already an experimental option in the offline one; on real home recordings it cut the parent-speech error from about 12 months to about 4), a toddler cutoff, and local norms for more communities and languages (language measures already track age in German and Cantonese).

**5. Hand off to the demo:** "Want to try it?"

---

## Lines for when a judge pushes

- **"Did you pick the cutoff on your test data?"** No. We choose it inside the training folds only, then apply it to children the model never saw: still 91% caught, 64% of typical children cleared.
- **"Your outside lab only caught about 60%."** At the original cutoff, yes; the ranking transfers (0.80) but every population needs its own reference point, like growth charts built for different countries. With 30 typical local kids it catches about 80%.
- **"How accurate is the language age?"** Across childhood the measures track age closely (r = 0.90), but for one 4-to-10-year-old it's only good to about a year, so the screen leads with the growth-chart percentile and treats language age as a rough guide.
- **"Isn't AUC 0.86 just okay?"** It's measured on children the model never saw, with confidence intervals. It held up on other labs' kids without retraining. A lot of high numbers you'll see at a fair come from testing on training data.
- **"Why not deep learning?"** We tested a 3,000-feature text model. It wasn't more accurate (0.82 vs 0.86), and a clinician can't see why it flags a child. Ours can.
- **"Does the recording version work on real kids?"** Partly, and we measured exactly where it breaks. We ran it on real home recordings of five 2½-year-olds that came with human transcripts. The speech recognizer gets most toddler words wrong, but our screen doesn't need the exact words, just how long the sentences are and how varied the vocabulary is. On the child's own speech, the estimated language age landed within about 4 months of the human-transcript answer. On the raw recording with parents talking, it was off by about 2 years, because it counted the parents' sentences as the child's. So we built a voice filter from the speech model's own internals that tells a child's voice from an adult's. Tested on families it never trained on, it brought the full-recording error from about 12 months down to about 4. It doesn't fix short clips yet. It's in our offline backup demo as an experimental switch, not in the web demo. It's still a prototype. (Only 5 typical children at one age, so we call this encouraging, not proven.)
- **"Why not just use a better speech model?"** We tested that. The largest Whisper model got fewer words wrong (84% vs 95% error) and brought the language-age estimate on the child's own speech from about 4 months off to about 3. But on raw home recordings it was still about 10 months too old for every child, because the parents' sentences are still counted. So word accuracy isn't the bottleneck; separating the child's voice from the parent's is. The demo keeps the small model because it runs fast in a browser.
- **"Does it work on toddlers, or just school-age kids?"** We trained on 4-10 year olds, then tested on 130 children from the Ellis Weismer study, with no retraining. At 2½ years it separates late talkers from typical talkers at 0.92. The score drops with age (0.78 at 3½, 0.55 at 5½) because many late talkers catch up, which is what the research literature says happens.
- **"So why does your demo only flag ages 4-10?"** Because the cutoff is tuned on 4-10 year olds. Used on toddlers it flags every child, since all toddlers have short sentences. The *ranking* still works (0.92), so the fix is a toddler cutoff. We tested setting it from about 35 typical toddlers only, no diagnosed kids needed: it caught 87% of late talkers and cleared 77% of typical ones, on children not used to set it. That's our next step, not a finished feature.
- **"So can it predict which toddlers will have a disorder later?"** No, and we don't claim that. "Late talker" is decided by vocabulary size at age 2, not a disorder diagnosis. By 5½ our screen can't tell which late talkers stayed behind. It catches who is behind *now*.
- **"In a real classroom, aren't most of your flags false alarms?"** Yes, at our cutoff, and that's on purpose. Out of 100 kids, about 7 have DLD. Our screen catches about 6 of the 7, and flags roughly 33 of the other 93 for a closer look (so it clears about 60 kids, matching the class-of-100 figure). A "pass" is right about 99% of the time. We could flip it: a stricter cutoff flags only about 8 typical kids, but then it misses half the kids who need help. For a first-step screen, missing a child is the worse mistake. (Strict cutoff from training folds: `src/ext_operating_point.py`, sens 0.91 / spec 0.64. Full tradeoff table at other cutoffs: `src/ext_operating_points.py`.)
- **"What's actually new here?"** Two things. First, a science finding: a model trained only on 2½-year-old late talkers spots school-age kids with DLD about as well as a model trained on those kids, and the reverse works too. That suggests late talking and DLD sit on one shared language dimension, mostly sentence length in word parts. Second, a method: a new clinic can set the cutoff using only typical kids, no diagnosed kids needed, and it does as well as if it had full diagnoses. (fig12; ISS-34, ISS-35)
- **"So late talkers grow up to have DLD?"** We can't say that. We showed the same *measure* separates both groups, not that the same children go on to DLD. That needs following kids over years.
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
