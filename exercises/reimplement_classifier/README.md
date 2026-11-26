# Ownership exercise: rebuild the delay classifier yourself

**Goal:** each team member writes the core FirstWords delay classifier **from scratch, in their own
file, without copying from `src/`**, and gets the same answer as the project (ROC-AUC 0.861). Commit
your file under your own git name. This is how you prove, to yourself and to a judge, that you can
build and explain the project's central result. Keep notes of what you struggled with in your logbook.

## The task (about 40 lines of Python)

Create `exercises/reimplement_classifier/<yourname>.py` with a function `oof_auc()` that returns the
child-independent, out-of-fold ROC-AUC of the delay classifier on ENNI.

1. **Data.** Load `results/features.csv` (create it first with
   `.venv/bin/python src/stage2_level_estimator.py`). Keep rows where `corpus == "ENNI"`, drop rows
   missing any of the four features below, and make the label `y = 1` if `group == "SLI"`, else `0`.
2. **Features.** `mluw` (mean words per utterance), `mlum` (mean morphemes per utterance),
   `mattr50` (vocabulary diversity), `age_months`.
3. **Model.** Standardize the features, then logistic regression with balanced class weights
   (there are about 4 typical children for every impaired child).
4. **Validation.** 5-fold `StratifiedGroupKFold` with `shuffle=True, random_state=0`, grouped by
   `child_id`, so no child is ever in both training and testing. Collect each child's predicted
   probability from the fold where they were held out.
5. **Score.** ROC-AUC of those out-of-fold probabilities.

## Check your answer

```bash
.venv/bin/python exercises/reimplement_classifier/check.py exercises/reimplement_classifier/<yourname>.py
```

## Then make sure you can answer, in your own words

- Why group the folds by `child_id` instead of splitting by file? (Hint: ISS-07 in the handoff.)
- Why balanced class weights, and why report ROC-AUC instead of accuracy?
- What happens to the AUC if you use only `age_months`? Only the number of utterances? Why does that matter?
- What would change if you picked the screening cutoff on the same predictions you report on?
