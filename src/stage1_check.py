#!/usr/bin/env python3
# FirstWords - Stage 1 check on the REAL downloaded data (Brown = typically-developing,
# ENNI = TD vs SLI). Confirms parsing, ages, developmental features, and previews
# BOTH signals: (A) core = age<->complexity within TD; (B) delay = TD vs SLI difference.
import os, re, math, statistics as st
import pylangacq

# Resolve data relative to the repo root so this runs from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "childes")

def months(age):
    if age is None: return None
    m = re.search(r"(\d+);(\d+)(?:\.(\d+))?", str(age))
    if not m: return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return y*12 + mo + d/30.0

def table(path):
    """Per-file developmental features for a corpus directory."""
    r = pylangacq.read_chat(path, strict=False)
    ages = [months(a) for a in r.ages()]
    feats = {"mluw": r.mluw(), "mlum": r.mlum(), "ttr": r.ttr(), "ipsyn": r.ipsyn()}
    rows = []
    for i, fp in enumerate(r.file_paths):
        row = {"file": os.path.relpath(fp, path), "age_months": ages[i] if i < len(ages) else None}
        for k, v in feats.items():
            row[k] = v[i] if i < len(v) else None
        rows.append(row)
    return r.n_files, rows

def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3: return None
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    return num/den if den else None

def summ(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return "n=0"
    return f"n={len(vals)} mean={st.mean(vals):.2f} sd={(st.pstdev(vals) if len(vals)>1 else 0):.2f}"

print("="*72); print(" A) BROWN (typically-developing) - CORE signal: age <-> complexity"); print("="*72)
n, rows = table(os.path.join(DATA, "Brown"))
print(f"n_files={n}")
ages = [r["age_months"] for r in rows]
print("age_months:", summ(ages), " range:",
      f"{min(a for a in ages if a):.1f}..{max(a for a in ages if a):.1f}")
for feat in ["mluw", "mlum", "ttr", "ipsyn"]:
    print(f"  {feat:6s}: {summ([r[feat] for r in rows])}   corr(age,{feat}) = "
          f"{pearson(ages, [r[feat] for r in rows])}")
print("\n sample rows:")
for r in rows[:4]:
    print("  ", {k: (round(v,2) if isinstance(v,(int,float)) else v) for k,v in r.items()})

print("\n"+"="*72); print(" B) ENNI - DELAY signal preview: TD vs SLI (same narrative task)"); print("="*72)
for grp in ["TD", "SLI"]:
    p = os.path.join(DATA, "ENNI", grp)
    if not os.path.isdir(p):
        print(f"  {grp}: folder not found at {p}"); continue
    n, rows = table(p)
    ages = [r["age_months"] for r in rows]
    print(f"\n {grp}: n_files={n}  age:", summ(ages))
    for feat in ["mluw", "mlum", "ttr", "ipsyn"]:
        print(f"    {feat:6s}: {summ([r[feat] for r in rows])}")
print("\n-> if SLI means are LOWER than TD on MLU/ipsyn at similar ages, the delay signal is real.")
