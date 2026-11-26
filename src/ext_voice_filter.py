#!/usr/bin/env python3
# FirstWords - Extension: CHILD-VOICE FILTER for raw home recordings (fixes ISS-29's open problem).
#
# Problem (ISS-29): on raw home audio the screen counts the PARENTS' sentences as the child's,
# so language age is inflated (+25 mo base.en, +10 mo large-v3) and a delayed child could be
# missed. A pitch-only filter failed its signal check (ISS-32: AUC 0.73-0.84).
#
# Idea: the speech model already in the app (Whisper base.en) has an audio ENCODER. Its internal
# representation of an utterance carries who is speaking (child vs adult voice) far better than
# pitch. Average the encoder output over the utterance's 20 ms frames (512 numbers), then a
# logistic regression says child vs adult. No new model, no new download.
#
# Validation is family-independent (leave-one-family-out): the classifier for each child is
# trained only on the OTHER 4 families' human-timed utterances.
#   Part 1  utterance-level AUC (child vs adult) on human-timed utterances.
#   Part 2  the real test: raw recording -> transcribe_timed() -> keep utterances the filter
#           calls "child" -> language age, vs the human transcript of the child. Also an ORACLE
#           filter (keep utterances that overlap the child's human time stamps) = best case.
# Data: Providence (5 TD children, ~33 mo, data/providence_audio + data/childes/Providence).
# USAGE: .venv/bin/python src/ext_voice_filter.py      (CPU, ~30-40 min first run; cached after)
import os, json
import numpy as np
import pandas as pd
import pylangacq
import ctranslate2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import audio_screen as asc
import ext_asr_childspeech as ac

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(RESULTS, "voice_filter_cache.json")
SR = ac.SR
N_TRAIN = 400          # utterances per speaker class per family for training
MIN_MS = 300
C = 0.01               # chosen on the Part-1 utterance AUC (0.01/0.05/0.2 all within 0.03)
WINDOWS = {"first15": 15 * 60, "session": None}


def embed(seg):
    """Mean of Whisper base.en encoder output over the utterance's frames (20 ms each)."""
    m = asc._get_model()
    seg = seg[:SR * 30]
    mel = m.feature_extractor(seg)
    mel = np.pad(mel, ((0, 0), (0, max(0, 3000 - mel.shape[1]))))[:, :3000]
    out = np.array(m.model.encode(ctranslate2.StorageView.from_array(
        np.ascontiguousarray(mel[None].astype(np.float32))), to_cpu=True))[0]
    k = max(1, min(1500, int(np.ceil(len(seg) / SR / 0.02))))
    return out[:k].mean(0)


def timed_utts(cha):
    """All human-timed utterances: (is_child, start_ms, end_ms)."""
    r = pylangacq.read_chat(cha, strict=False)
    return [(u.participant == "CHI", *u.time_marks) for u in r.utterances()
            if u.time_marks and u.time_marks[1] - u.time_marks[0] >= MIN_MS]


def overlap_child(s, e, spans):
    """Fraction of [s,e] (seconds) covered by child spans (ms)."""
    tot = 0.0
    for a, b in spans:
        tot += max(0.0, min(e, b / 1000) - max(s, a / 1000))
    return tot / max(e - s, 1e-6)


def main():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    save = lambda: json.dump(cache, open(CACHE, "w"))
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(ac.AUDIO) if f.endswith(".wav"))
    rng = np.random.default_rng(0)
    fam = {}
    for n in names:
        child, sess = n.split("_")
        cha = os.path.join(ac.CHA, child, f"{sess}.cha")
        audio = ac.read_wav(os.path.join(ac.AUDIO, n + ".wav"))
        U = timed_utts(cha)
        c = cache.setdefault(n, {})
        if "train" not in c:
            print(f"  embedding human-timed utterances: {child}", flush=True)
            ch = [u for u in U if u[0]]; ad = [u for u in U if not u[0]]
            pick = [ch[i] for i in rng.choice(len(ch), min(N_TRAIN, len(ch)), replace=False)] + \
                   [ad[i] for i in rng.choice(len(ad), min(N_TRAIN, len(ad)), replace=False)]
            c["train"] = [[int(isc), embed(audio[int(s * SR / 1000):int(e * SR / 1000)]).tolist()] for isc, s, e in pick]
            save()
        for w, secs in WINDOWS.items():
            if f"raw_{w}" not in c:
                print(f"  transcribing raw {w}: {child}", flush=True)
                a = audio if secs is None else audio[:secs * SR]
                p = ac.tmp_wav(n, a); utts = asc.transcribe_timed(p); os.remove(p)
                c[f"raw_{w}"] = [[t, s, e, embed(audio[int(s * SR):int(e * SR) + 1]).tolist()] for t, s, e in utts]
                save()
        gold, age = ac.gold_utts(cha)
        fam[child] = dict(n=n, age=age, gold=gold, child_spans=[(s, e) for isc, s, e in U if isc],
                          y=np.array([t[0] for t in c["train"]]), X=np.array([t[1] for t in c["train"]]))

    scr = asc.ReducedScreener(); tok = lambda u: __import__("re").findall(r"[a-z']+", u.lower())
    rows, aucs = [], []
    for child, f in fam.items():
        others = [g for g in fam if g != child]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000, class_weight="balanced")).fit(
            np.vstack([fam[g]["X"] for g in others]), np.concatenate([fam[g]["y"] for g in others]))
        auc = roc_auc_score(f["y"], clf.predict_proba(f["X"])[:, 1]); aucs.append(auc)
        for w, secs in WINDOWS.items():
            raw = cache[f["n"]][f"raw_{w}"]
            p = clf.predict_proba(np.array([r[3] for r in raw]))[:, 1]
            is_child = np.array([overlap_child(r[1], r[2], f["child_spans"]) >= 0.5 for r in raw])
            keep = p >= 0.5
            ref = [ws for s, _, ws in f["gold"] if secs is None or s / 1000 < secs]
            conds = {"gold": ref, "raw_all": [tok(r[0]) for r in raw],
                     "voice_filter": [tok(r[0]) for r, k in zip(raw, keep) if k],
                     "oracle_filter": [tok(r[0]) for r, k in zip(raw, is_child) if k]}
            for cond, lists in conds.items():
                ft = ac.feat_from_wordlists(lists)
                o = scr.screen({"mluw": ft["mluw"], "mattr50": ft["mattr50"]}, f["age"])
                rows.append(dict(child=child, window=w, condition=cond, age_months=f["age"], n_utts=ft["n_utts"],
                                 mluw=ft["mluw"], mattr50=ft["mattr50"], lang_age=o["lang_age"], p_delay=o["p_delay"],
                                 reliable=o["reliable"], utt_auc=auc,
                                 precision=float(is_child[keep].mean()) if cond == "voice_filter" else np.nan,
                                 recall=float(keep[is_child].mean()) if cond == "voice_filter" else np.nan))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, "voice_filter.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nPart 1: child-vs-adult utterance AUC (family never seen in training):",
          " ".join(f"{a:.3f}" for a in aucs), f"| mean {np.mean(aucs):.3f}")
    print("\nPart 2 per child:")
    print(df[["child", "window", "condition", "n_utts", "mluw", "mattr50", "lang_age", "p_delay", "precision", "recall"]]
          .round(2).to_string(index=False))
    print("\nSummary: language age minus the human-transcript language age (months), mean over children [range]:")
    for w in WINDOWS:
        g = df[(df.window == w) & (df.condition == "gold")].set_index("child")
        for cond in ["raw_all", "voice_filter", "oracle_filter"]:
            d = df[(df.window == w) & (df.condition == cond)].set_index("child")
            e = d.lang_age - g.lang_age
            extra = ""
            if cond == "voice_filter":
                extra = f"  precision {d.precision.mean():.2f}  recall {d.recall.mean():.2f}"
            print(f"  {w:8s} {cond:14s} {e.mean():+6.1f} (|{e.abs().mean():4.1f}|) [{e.min():+.1f} to {e.max():+.1f}]{extra}")


if __name__ == "__main__":
    main()
