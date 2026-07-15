#!/usr/bin/env python3
# FirstWords - verify the CHILDES pipeline MECHANICS on controlled synthetic CHAT
# files (no download / login needed). Confirms pylangacq parses CHAT, extracts ages,
# and that its built-in developmental measures (MLU, TTR, IPSyn) work and track age.
import os, math
import pylangacq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "results", "_synthetic_check")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# 3 tiny transcripts, CHI ages 1;06, 2;06, 3;06, with increasing child utterance
# length so the measures should increase with age (mechanics sanity check).
FILES = {
    "c1.cha": ("1;06.00", ["ball .", "no ."]),
    "c2.cha": ("2;06.00", ["more cookie .", "want ball ."]),
    "c3.cha": ("3;06.00", ["I want more cookie .", "the big ball is here ."]),
}
for fname, (age, chi_utts) in FILES.items():
    lines = ["@UTF8", "@Begin", "@Languages:\teng",
             "@Participants:\tCHI Target_Child , MOT Mother",
             f"@ID:\teng|Sample|CHI|{age}|male|||Target_Child|||",
             "@ID:\teng|Sample|MOT|||||Mother|||"]
    lines += [f"*CHI:\t{u}" for u in chi_utts]
    lines += ["*MOT:\tokay .", "@End"]
    with open(os.path.join(SAMPLE_DIR, fname), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

r = pylangacq.read_chat(SAMPLE_DIR)
print("n_files:", r.n_files)
print("file order:", [os.path.basename(p) for p in r.file_paths])

# ---- ages (this backend's ages() takes no kwargs; returns (y,m,d) tuples) ----
raw_ages = r.ages()
import re
def to_months(a):
    if a is None: return None
    m = re.search(r"(\d+);(\d+)(?:\.(\d+))?", str(a))   # parse "Y;MM.DD"
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        return y*12 + mo + d/30.0
    try: return float(a)
    except Exception: return None
ages_m = [to_months(a) for a in raw_ages]
print("ages raw:", raw_ages)
print("ages(months):", [round(x, 1) if x else x for x in ages_m])

# ---- built-in developmental measures; inspect return type/shape ----
def show(name):
    try:
        v = getattr(r, name)()
        print(f"{name}(): type={type(v).__name__}  value={v}")
        return v
    except Exception as e:
        print(f"{name}(): ERR {e}")
        return None
mluw = show("mluw")   # MLU in words
show("mlum")          # MLU in morphemes
show("ttr")           # type-token ratio (lexical diversity)
show("ipsyn")         # Index of Productive Syntax

# ---- correlate age vs MLU-words to confirm the measure tracks development ----
def as_list(v):
    if v is None: return None
    if isinstance(v, dict):  # keyed by file path -> align to file order
        return [v.get(p) for p in r.file_paths]
    return list(v)
mluw_list = as_list(mluw)
if mluw_list and all(x is not None for x in ages_m):
    xs, ys = ages_m, [float(x) for x in mluw_list]
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    corr = num/den if den else float("nan")
    print(f"\ncorr(age_months, MLU-words) = {corr:.3f}  (should be ~+1.0 by construction)")
    print("PIPELINE MECHANICS OK" if corr > 0.9 else "CHECK API/mechanics")
else:
    print("\nCould not align ages with MLU - inspect output above.")
