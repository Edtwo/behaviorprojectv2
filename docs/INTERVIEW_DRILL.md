# FirstWords: Interview Drill

Practice sheet for the 25-point interview. **No scripted answers on purpose.** Judges test whether
you understand the work, and AI wrote most of the code, so the answers have to come from you.

How to use it:
1. Have someone who hasn't seen the project read questions at random.
2. Answer out loud in under 45 seconds, no notes.
3. If you stall, open the pointer, learn it, and retry the next day.
4. You're ready when you can answer every **Level 1** and **Level 2** question cold.

Format: question -> what a strong answer must include -> where to check it.

## Level 1: the basics (every judge asks some of these)

1. **What problem are you solving, in one sentence?**
   Must include: who is affected, how common, why it's missed. -> PITCH.md section 1
2. **What does your tool take in, and what does it give back?**
   Must include: input (transcript / typed / recording), the 3 outputs. -> PITCH.md section 2
3. **What are the four measures, and why those?**
   Must include: what each one counts, why it changes with age. -> EXPLAINER.md, Stage 2 section
4. **What does an AUC of 0.86 mean?** Explain it to a 10-year-old.
   Must include: what 0.5 and 1.0 mean, what "ranking" means. -> fig2, EXPLAINER.md
5. **How do you know you didn't test on children you trained on?**
   Must include: splitting by child, not by file, and why that matters. -> fig10, ISS-07
6. **What's your data, and where did it come from?**
   Must include: CHILDES / TalkBank, de-identified, ~1,184 children, which corpora do what. -> fig8, Handoff section 4
7. **Is this a diagnosis?** -> PITCH.md "Lines for when a judge pushes"

## Level 2: methods (BEHA judges with a stats background)

8. **Why logistic regression instead of deep learning?**
   Must include: the black-box comparison numbers, interpretability. -> fig5, `src/ext_benchmark.py`
9. **How do you know the model isn't just detecting age or transcript length?**
   Must include: the three AUCs on fig1 and what "at chance" means. -> fig1, ISS-04
10. **Why does age have the biggest SHAP value if age alone can't classify?**
    Must include: age as the norm reference. -> Handoff 0b Stage 4, 9e
11. **What's the difference between your two models (level estimator vs delay flag)?** -> Handoff section 3
12. **Why does the ENNI cutoff not work on Gillam or on toddlers?**
    Must include: ranking vs cutoff, why a probability scale depends on the training group. -> ISS-18, ISS-31
13. **What does "91% sensitivity" cost you? How many false alarms per 100 kids?**
    Must include: the per-100 numbers, why you chose this end of the tradeoff. -> `src/ext_operating_points.py`
14. **How reliable is one sample from a child?**
    Must include: split-half result, what Spearman-Brown corrects for, and whether two *different tasks* agree (good at 30 mo, weak later). -> `src/ext_reliability.py`, ISS-11 follow-up
15. **Why MATTR instead of type-token ratio?**
    Must include: what goes wrong with TTR on longer transcripts. -> ISS-05

## Level 3: hard pushes (the judge who wants to see you think)

16. **What's the biggest weakness of your project?**
    Pick one honestly and give the evidence. -> ISEF_SCORECARD.md "Weaknesses"
17. **Does it work on toddlers?** Careful: ranking yes, flag no. -> ISS-28, ISS-31, fig14
18. **Does the recording version work on real children?**
    Must include: child-only vs raw home audio, why parents' speech breaks it, and what the voice filter fixed and didn't. -> ISS-29, ISS-36
19. **Is it fair across ethnic groups?**
    Must include: the Gillam gap, what you found the cause was (not grammar), what you'd do. -> ISS-17
20. **Name a result you were excited about that turned out to be wrong.** -> ISS-16 (typology), ISS-15 (German 0.99)
21. **Someone did ML on ENNI in 2009. What's new here?**
    Know Gabani et al.'s actual reported numbers (look them up; not in the repo). -> READING_LIST.md section D
22. **Why does the AUC fall from 30 to 66 months in the toddler test?** -> ISS-28
23. **If you had a year and a clinic partner, what would you do next?**
    Must include something concrete: speaker separation, a toddler cutoff from local typical kids, a prospective test.

27. **What's original about your project?** Know both ideas and their numbers. -> ISS-34 (fig12), ISS-35
28. **Does your both-directions result mean late talkers will develop DLD?** Must include: shared measure vs same children, what data would answer it. -> ISS-34

## Level 4: ownership (expect at least one of these)

24. **How much of this did you do, and how much did AI do?**
    Answer honestly and specifically; it must match your AI-use disclosure and references. -> Handoff section 2, ISS-22
25. **Walk me through how you'd re-run your main result from scratch.**
    Know the commands: venv, data folder, `stage2_level_estimator.py`, `stage3_delay_classifier.py`. -> README.md
26. **Change one thing in the model right now: what would you expect to happen if you removed age?**
    You should predict ~0.78 before checking. -> fig1

Self-check: if any Level 4 answer feels shaky, that's where to spend your practice time first.
