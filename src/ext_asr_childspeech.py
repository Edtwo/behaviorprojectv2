#!/usr/bin/env python3
# FirstWords - Extension: REAL CHILD-SPEECH ASR VALIDATION (Providence, Eng-NA).
#
# Gap this closes: the record-and-screen path (src/audio_screen.py) was only checked on
# synthesized (TTS) speech. Its open risk is Whisper's error on REAL toddler speech.
# Providence = 6 children recorded at home with time-aligned human CHAT transcripts.
# We use one ~33-month session per child (5 children; Ethan's media file is not served).
#
# Audio: data/providence_audio/<Child>_<session>.wav (16 kHz mono, extracted from the
# TalkBank .mp4). Transcripts: data/childes/Providence/<Child>/<session>.cha.
#
# Three conditions, each compared to the human gold transcript of the SAME child speech:
#   1) ASR-ONLY:   each gold CHI utterance is cut from the audio by its CHAT time stamps and
#                  transcribed alone. Isolates recognition error on child speech.
#   2) PIPELINE:   the CHI clips are joined with 1.0 s silences and run through the real
#                  audio_screen.transcribe() (pause segmentation). Adds segmentation error.
#   3) RAW AUDIO:  first 15 min of the session, unfiltered, through transcribe(). Includes
#                  parent speech -> shows what happens WITHOUT speaker separation.
# Outputs: results/asr_childspeech.csv (per child x condition) + printed summary.
import os, re, sys, json, subprocess
import numpy as np
import pandas as pd
import pylangacq
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_screen as asc
import stage2_level_estimator as s2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(ROOT, "data", "providence_audio")
CHA = os.path.join(ROOT, "data", "childes", "Providence")
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(RESULTS, "asr_childspeech_cache.json")
SR = 16000
RAW_SECONDS = 15 * 60
SKIP = {"xxx", "yyy", "www"}


def gold_utts(cha):
    """CHI utterances with time marks -> (start_ms, end_ms, [words]). Drops unintelligible
    (xxx/yyy), fillers (&-uh), and punctuation, matching how audio_screen counts words."""
    r = pylangacq.read_chat(cha, strict=False)
    out = []
    for u in r.utterances():
        if u.participant != "CHI": continue
        if not u.time_marks: continue
        words = []
        for t in u.tokens:
            w = (t.word or "").lower()
            if w in SKIP or w.startswith("&") or not re.search(r"[a-z]", w): continue
            words += re.findall(r"[a-z']+", w.replace("_", " ").replace("+", " "))
        if words: out.append((u.time_marks[0], u.time_marks[1], words))
    age = s2.months(r.ages()[0])
    return out, age


def wer(ref, hyp):
    """Word error rate via edit distance."""
    d = np.arange(len(hyp) + 1)
    for i, rw in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / max(1, len(ref))


def feat_from_wordlists(lists):
    lens = [len(w) for w in lists if w]
    words = [x for w in lists for x in w]
    if not lens: return None
    return {"mluw": float(np.mean(lens)), "mattr50": s2.mattr(words), "n_utts": len(lens), "n_words": len(words)}


def read_wav(path):
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype("float32") / 32768.0


def tmp_wav(name, data):
    p = os.path.join(RESULTS, f"_tmp_{name}.wav")
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
    return p


def run_child(name, cache):
    child, sess = name.split("_")
    gold, age = gold_utts(os.path.join(CHA, child, f"{sess}.cha"))
    audio = read_wav(os.path.join(AUDIO, f"{name}.wav"))
    model = asc._get_model()
    key = name
    c = cache.setdefault(key, {})
    save = lambda: json.dump(cache, open(CACHE, "w"))
    if "asr_only" not in c:
        # 1) ASR-only, per gold utterance (pad 0.15 s each side)
        hyps = []
        for s, e, _ in gold:
            a, b = max(0, int((s / 1000 - .15) * SR)), min(len(audio), int((e / 1000 + .15) * SR))
            segs, _ = model.transcribe(audio[a:b], language="en", vad_filter=False,
                                       condition_on_previous_text=False, beam_size=5)
            hyps.append(" ".join(sg.text for sg in segs))
        c["asr_only"] = hyps; save()
    if "pipeline" not in c:
        # 2) pipeline: concatenated CHI clips + 1 s silences through transcribe()
        sil = np.zeros(SR, dtype="float32"); parts = []
        for s, e, _ in gold:
            parts += [audio[int(s / 1000 * SR):int(e / 1000 * SR)], sil]
        p = tmp_wav(name, np.concatenate(parts)); c["pipeline"] = asc.transcribe(p); os.remove(p); save()
    if "raw15" not in c:
        # 3) raw first 15 minutes (parents included)
        p = tmp_wav(name, audio[:RAW_SECONDS * SR]); c["raw15"] = asc.transcribe(p); os.remove(p); save()
    gold_lists = [w for _, _, w in gold]
    gold_raw15 = [w for s, _, w in gold if s / 1000 < RAW_SECONDS]
    tok = lambda u: re.findall(r"[a-z']+", u.lower())
    asr_lists = [tok(h) for h in c["asr_only"]]
    wers = [wer(g, h) for g, h in zip(gold_lists, asr_lists)]
    rows = []
    for cond, ref, hyp in [("gold", gold_lists, gold_lists), ("asr_only", gold_lists, asr_lists),
                           ("pipeline", gold_lists, [tok(u) for u in c["pipeline"]]),
                           ("gold_first15", gold_raw15, gold_raw15),
                           ("raw15_all_speakers", gold_raw15, [tok(u) for u in c["raw15"]])]:
        f = feat_from_wordlists(hyp)
        rows.append({"child": child, "session": sess, "age_months": age, "condition": cond, **(f or {}),
                     "wer_mean": float(np.mean(wers)) if cond == "asr_only" else np.nan,
                     "wer_pooled": (sum(w * len(g) for w, g in zip(wers, gold_lists)) /
                                    sum(len(g) for g in gold_lists)) if cond == "asr_only" else np.nan,
                     "dropped_utts": sum(1 for h in asr_lists if not h) if cond == "asr_only" else np.nan})
    return rows, age


def main():
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(AUDIO) if f.endswith(".wav"))
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    scr = asc.ReducedScreener()
    rows = []
    for n in names:
        print(f"  transcribing {n} ...", flush=True)
        r, age = run_child(n, cache)
        for row in r:
            if "mluw" in row and not pd.isna(row.get("mluw")):
                out = scr.screen({"mluw": row["mluw"], "mattr50": row["mattr50"]}, age)
                row.update(lang_age=out["lang_age"], p_delay=out["p_delay"], reliable=out["reliable"])
        rows += r
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, "asr_childspeech.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\nPer child (MLU-words / MATTR / language-age / delay prob):")
    print(df[["child", "age_months", "condition", "n_utts", "mluw", "mattr50", "lang_age", "p_delay",
              "wer_pooled", "dropped_utts"]].round(2).to_string(index=False))

    g = df[df.condition == "gold"].set_index("child")
    print("\nSummary vs gold (mean over children; signed % error, then mean |error|):")
    for cond, ref in [("asr_only", "gold"), ("pipeline", "gold"), ("raw15_all_speakers", "gold_first15")]:
        c = df[df.condition == cond].set_index("child"); r = df[df.condition == ref].set_index("child")
        e_mlu = 100 * (c.mluw - r.mluw) / r.mluw
        e_mat = 100 * (c.mattr50 - r.mattr50) / r.mattr50
        e_age = c.lang_age - r.lang_age
        print(f"  {cond:20s} MLU {e_mlu.mean():+6.1f}% (|{e_mlu.abs().mean():4.1f}|)  "
              f"MATTR {e_mat.mean():+6.1f}% (|{e_mat.abs().mean():4.1f}|)  "
              f"lang-age shift {e_age.mean():+5.1f} mo (|{e_age.abs().mean():4.1f}|)  "
              f"utts {c.n_utts.sum():.0f} vs gold {r.n_utts.sum():.0f}")
    a = df[df.condition == "asr_only"]
    print(f"\n  Word error rate on real ~33-mo child speech (pooled): {a.wer_pooled.mean():.2f}; "
          f"utterances Whisper returned EMPTY: {a.dropped_utts.sum():.0f} / {g.n_utts.sum():.0f}")


if __name__ == "__main__":
    main()
