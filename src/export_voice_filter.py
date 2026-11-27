#!/usr/bin/env python3
# FirstWords - export the child-vs-adult voice filter (ISS-36) as a small JSON model for the
# offline demo (audio_screen.py --child-only, demo_server.py checkbox).
# Trains ONE logistic regression on all 5 Providence families' human-timed utterances
# (Whisper base.en encoder mean, 512 numbers each), same settings as ext_voice_filter.py.
# The honest accuracy (AUC 0.94 on unseen families; full-session error +11.7 -> +4.2 mo) comes
# from ext_voice_filter.py's leave-one-family-out test, NOT from this exported model.
# Output: models/voice_filter.json (weights only; no audio, no transcripts, no child data).
# USAGE: .venv/bin/python src/ext_voice_filter.py && .venv/bin/python src/export_voice_filter.py
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "voice_filter_cache.json")
OUT = os.path.join(ROOT, "models", "voice_filter.json")
C = 0.01


def main():
    cache = json.load(open(CACHE))
    X = np.array([t[1] for fam in cache.values() for t in fam["train"]])
    y = np.array([t[0] for fam in cache.values() for t in fam["train"]])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=4000, class_weight="balanced").fit(sc.transform(X), y)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"what": "child (1) vs adult (0) utterance, Whisper base.en encoder mean (512)",
               "encoder": "base.en", "threshold": 0.5, "C": C, "n_utterances": int(len(y)),
               "n_families": len(cache), "validated_in": "src/ext_voice_filter.py (ISS-36)",
               "mean": np.round(sc.mean_, 6).tolist(), "scale": np.round(sc.scale_, 6).tolist(),
               "coef": np.round(lr.coef_[0], 6).tolist(), "intercept": float(lr.intercept_[0])},
              open(OUT, "w"))
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({len(y)} utterances, {len(cache)} families)")


if __name__ == "__main__":
    main()
