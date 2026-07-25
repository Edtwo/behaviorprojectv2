# FirstWords — Background Reading List
### The literature to know cold before the fairs

**How to use this:** you do NOT need to read every paper end-to-end. For each item, read the
**abstract + figures + discussion**, and take away the one idea noted after "→". The starred (★)
items are the ones you should actually be able to talk about if a judge asks. Everything else is
depth you can gesture at.

**Honesty note:** these citations are written from memory. Author lists / years may be slightly off —
**verify each on Google Scholar** before citing it on a poster or in a paper. Most are findable free via
Google Scholar, the authors' university pages, or ResearchGate; your school/public library likely has
institutional access to the journals (JSLHR = *Journal of Speech, Language, and Hearing Research*).

---

## A. The foundation: how children's language develops & how we measure it

- ★ **Brown, R. (1973). *A First Language: The Early Stages.* Harvard University Press.**
  → The origin of **MLU (mean length of utterance)** and "Brown's stages." The *Brown* corpus we use
  (Adam, Eve, Sarah) is literally his data. If you cite one classic, cite this.
- ★ **Miller, J. F., & Chapman, R. S. (1981). The relation between age and mean length of utterance in
  morphemes. *JSHR*, 24, 154–161.** → Establishes empirically that **MLU rises with age** — the
  bedrock signal your whole "growth chart" rests on.
- **Templin, M. (1957). *Certain Language Skills in Children.*** → Early large-sample developmental norms;
  the historical ancestor of the "language growth chart" idea.
- ★ **Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The moving-average type–token
  ratio (MATTR). *Journal of Quantitative Linguistics*, 17, 94–100.** → The exact lexical-diversity
  measure you use (`mattr50`) and **why** it's length-robust.
- ★ **Richards, B. (1987). Type/token ratios: what do they really tell us? *Journal of Child Language*,
  14, 201–209.** → Why raw **TTR is confounded by transcript length** — the reason you moved to MATTR.
  This is core to your artifact-control story.
- **Malvern, D., Richards, B., Chipere, N., & Durán, P. (2004). *Lexical Diversity and Language
  Development.*** → Book-length treatment of the TTR-length problem and the D/vocd fix. Depth for the
  lexical-diversity thread.
- **Scarborough, H. S. (1990). Index of Productive Syntax (IPSyn). *Applied Psycholinguistics*, 11, 1–22.**
  → Defines **IPSyn**. Read this so you can say precisely why the library's `ipsyn()` (which you relabeled
  `syntax_index`) is *not* the published 0–112 scale.

## B. Developmental Language Disorder (DLD / SLI) — the condition you screen for

- ★ **Leonard, L. B. (2014). *Children with Specific Language Impairment* (2nd ed.). MIT Press.**
  → The definitive book. Skim for: what SLI/DLD is, its markers, and that it's common but under-recognized.
- ★ **Bishop, D. V. M., et al. (2017). Phase 2 of the CATALISE study. *J. Child Psychology & Psychiatry.***
  → The consensus that renamed "SLI" → **Developmental Language Disorder (DLD)**. Know this so your
  terminology is current; use "DLD (historically SLI)."
- ★ **Tomblin, J. B., et al. (1997). Prevalence of specific language impairment in kindergarten children.
  *JSLHR*, 40, 1245–1260.** → The **~7% prevalence** figure — "more common than autism, less recognized."
  Your opening-slide statistic.
- ★ **Rice, M. L., & Wexler, K. (1996). Toward tense as a clinical marker of SLI. *JSLHR*, 39, 1239–1257.**
  → The **Extended Optional Infinitive** hypothesis: SLI children under-mark tense/agreement. This is
  exactly what your item-level grammar analysis tests — and why finiteness being *equal* by school age
  (with the deficit shifted to syntax) is a sophisticated, correct result.
- **Law, J., Garrett, Z., & Nye, C. (2004). The efficacy of treatment for children with developmental
  speech/language delay. *JSLHR*.** → Evidence that **early intervention works** — your "why it matters."
- **US Preventive Services Task Force / Nelson, H. D., et al. (2006). Screening for speech and language
  delay in preschool children. *Pediatrics*, 117, e298–e319.** → IMPORTANT counter-reading: the USPSTF
  found *insufficient evidence* to recommend universal screening. Know this so you can honestly frame your
  tool as an *accessible aid*, not a settled solution — and so a skeptical judge can't blindside you.
- **Marchman, V. A., & Fernald, A. (2008). Speed of word recognition and vocabulary knowledge in infancy
  predict later outcomes. *Developmental Science*, 11, F9–F16.** → Early language predicts later academic
  and cognitive outcomes — the stakes.

## C. The data & the tools (CHILDES / CHAT / narrative instruments)

- ★ **MacWhinney, B. (2000). *The CHILDES Project: Tools for Analyzing Talk.*** → The database, the CHAT
  transcription format, and the CLAN tools. You must be able to describe what CHILDES/TalkBank is.
- **Sagae, K., Davis, E., Lavie, A., MacWhinney, B., & Wintner, S. (2010). Automatic measurement of
  syntactic development in child language. *(ACL / Computer Speech & Language)*.** → How the `%mor`
  (morphology) and `%gra` (grammatical-relation) tiers you rely on are produced automatically. This is
  the machinery under your `pylangacq` features.
- **Ratner, N. B., & MacWhinney, B. (2016). Your laptop to the rescue: Using CHILDES … in clinical
  practice. *Seminars in Speech and Language*.** → Clinicians using language-sample analysis — the
  real-world frame for your tool.
- ★ **Schneider, P., Dubé, R. V., & Hayward, D. (2005). *The Edmonton Narrative Norms Instrument (ENNI).***
  → Your **main corpus**. Read the design: matched TD vs. language-impaired children on the *same*
  narrative task — the property that makes your delay comparison clean.
- **Gillam, R. B., & Pearson, N. A. (2004). *Test of Narrative Language (TNL).*** → Background for the
  **Gillam** corpus (your clean, age-matched external validation set).
- **Berman, R. A., & Slobin, D. I. (1994). *Relating Events in Narrative: A Crosslinguistic Developmental
  Study.*** → The "frog story" (*Frog, Where Are You?*) narrative method used by the **Conti-Ramsden**
  corpus (your out-of-range stress test).
- **Conti-Ramsden, G., & Botting, N. (1999). Classification of children with SLI. *JSLHR*.** → Background
  on the Manchester/Conti-Ramsden clinical cohort.

## D. Automatic language assessment & the transcript-length artifact (your direct prior work)

- ★ **Gabani, K., Solorio, T., Liu, Y., Bedore, L. M., & Peña, E. D. (2009). Exploring a corpus-based
  approach for detecting language impairment in children. *(NAACL / BioNLP)*.** → A canonical paper on
  **machine detection of SLI from transcripts**. Your project is the accessible, artifact-controlled,
  interpretable version of this line of work. Read it to place yourself in the literature honestly.
- **Hassanali, K.-N., Liu, Y., Iglesias, A., Solorio, T., & Dollaghan, C. (2014). Automatic generation of
  the Index of Productive Syntax. *(JSLHR / NLP venue)*.** → Automating IPSyn — relevant to your
  `syntax_index` caveats.
- **Prud'hommeaux, E., & Roark, B. (various, ~2011–2015).** Narrative-recall / graph-based language
  screening. → Another automatic-screening thread; shows the breadth of methods (and that yours is
  deliberately simpler + more interpretable).
- *(General principle, cite via §A Richards 1987 + the D/vocd literature)*: several automatic-assessment
  results can be **driven by transcript length** rather than true ability. Your **age-only / length-only
  artifact checks** (age-alone AUC 0.42, length-alone 0.49) are the direct answer to this critique — the
  single most important methodological point you have.

## E. Machine-learning methodology (so you can defend every modeling choice)

- ★ **James, G., Witten, D., Hastie, T., & Tibshirani, R. *An Introduction to Statistical Learning* (ISLR,
  free PDF).** → The best single source for a high-schooler: linear & ridge regression, logistic
  regression, cross-validation, the bias–variance trade-off. Read chs. 2, 3, 4, 5, 6.
- **Hastie, Tibshirani, & Friedman. *The Elements of Statistical Learning* (ESL, free PDF).** → The deeper
  version of the above, for reference.
- ★ **Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model
  selection. *BMC Bioinformatics*.** → Why choosing a model and estimating its error on the *same* data
  inflates results — background for doing CV honestly.
- ★ **Kaufman, S., Rosset, S., & Perlich, C. (2012). Leakage in data mining. *ACM TKDD*.** → **Data
  leakage** — exactly what your **child-independent (GroupKFold) validation** prevents. If a judge asks
  "how do you know it didn't memorize kids?", this is the concept.
- **Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot
  for imbalanced datasets. *PLoS ONE*.** → Why you report **PR-AUC** alongside ROC-AUC for the 21%-
  prevalence delay task.
- ★ **Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions (SHAP).
  *NeurIPS*.** → The interpretability method in Stage 4. Read the intro + the additive-feature-attribution
  idea.
- **Molnar, C. *Interpretable Machine Learning* (free online book).** → Plain-language interpretability
  (SHAP, permutation importance). Great companion to the SHAP paper.
- **Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning.
  *ICML*.** → **Calibration** (Platt/isotonic) — the basis of your calibration extension (Brier score,
  reliability diagram).
- ★ **Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis. *Medical Decision Making*, 26,
  565–574.** → The **net-benefit** method behind your clinical-utility figure. This is what turns "good
  AUC" into "actually useful."
- **Steyerberg, E. W. (2019). *Clinical Prediction Models* (2nd ed.).** → The field's methodology bible
  for medical prediction; skim for validation + calibration + utility.
- **Collins, G. S., Reitsma, J. B., Altman, D. G., & Moons, K. G. M. (2015). The TRIPOD statement.
  *Annals of Internal Medicine*.** → The reporting checklist for clinical prediction models. Structuring
  your paper around TRIPOD signals real methodological maturity.
- **Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap.*** → Where your **bootstrap
  confidence intervals** come from; the *cluster* bootstrap (resampling children) is the grouped variant.

## F. Cross-linguistic acquisition (for the multilingual aspect)

- ★ **Klee, T., Stokes, S. F., Wong, A. M.-Y., Fletcher, P., & Gavin, W. J. (2004). Utterance length and
  lexical diversity in Cantonese-speaking children with and without SLI. *JSLHR*, 47, 1396–1410.**
  → GOLD for you: shows that in **Cantonese**, lexical diversity and utterance length behave differently
  than in English SLI — directly supporting your finding that **vocabulary diversity, not MLU, is the key
  age marker in Cantonese** (an isolating language).
- **Slobin, D. I. (Ed.) (1985–1997). *The Crosslinguistic Study of Language Acquisition* (vols. 1–5).**
  → The reference work on how acquisition differs across languages/typologies — the frame for "the best
  marker depends on language type."
- **Fletcher, P., Leung, S. C. S., Stokes, S. F., & Weizman, Z. O. (various). Cantonese child language /
  the HKU corpus.** → Background on your **HKU-70** Cantonese data.
- **Szagun, G. (2001). Language acquisition in young German-speaking children with cochlear implants.
  *Audiology & Neurotology* (and related Szagun papers).** → Background on the **Szagun** corpus you use
  for the German level estimator and the cross-lingual delay (TD vs. cochlear-implant) test.
- *(Concept to know)* **MLU in *words* vs. *morphemes* is not equally valid across languages.** In
  richly-inflected or isolating languages, sentence-length measures can be weaker developmental indices —
  the reason your harness gates on the *strongest* marker per language rather than MLU alone.

---

## Suggested reading order (a 2–3 week plan)
1. **Week 1 — the science you're screening:** Brown (1973, skim) · Miller & Chapman (1981) · Leonard (2014,
   skim) · Tomblin (1997) · Rice & Wexler (1996) · CATALISE (2017).
2. **Week 2 — measurement & data:** Covington & McFall (2010) · Richards (1987) · MacWhinney (2000, skim) ·
   Schneider et al. (2005, ENNI) · Gabani et al. (2009).
3. **Week 3 — methods you must defend:** ISLR chs. 3–6 · Kaufman et al. (2012, leakage) · Lundberg & Lee
   (2017, SHAP) · Vickers & Elkin (2006, DCA) · Klee et al. (2004, Cantonese).

By the end you can answer, from the literature: *what you measure, why it rises with age, what DLD is, why
your validation can't be leaking, why length isn't the signal, why the method is interpretable and useful,
and why the marker differs across languages.* That is the full defensive perimeter of the project.
