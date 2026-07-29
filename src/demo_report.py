#!/usr/bin/env python3
# FirstWords - the JUDGE-FACING VISUAL DEMO. Generates a clean, self-contained HTML report
# for a child's transcript: estimated language age, age-gap, growth-chart position, delay
# risk, and the specific markers below age norm - opens in any browser, no server needed
# (bulletproof for a fair; nothing to crash). Reuses the validated Stage 4/percentile engine.
#
# USAGE:
#   .venv/bin/python src/demo_report.py --demo          # builds reports for 2 example children
#   .venv/bin/python src/demo_report.py --file X.cha    # builds a report for one transcript
# Output: results/demo_<name>.html  (double-click to open).
import os, sys, argparse, base64, io, html
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stage5_demo as demo
import ext_percentiles as pct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

def growth_plot_b64(screener, row):
    """Growth-chart figure with THIS child marked - returned as a base64 PNG for the HTML."""
    df = demo._load_features(); td = df[(df["group"] == "TD") & df["age_months"].notna()]
    grid = np.linspace(18, max(td.age_months.max(), row["age_months"]) + 2, 120)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(td.age_months, td.mluw, s=6, alpha=0.18, color="#9bb", zorder=1)
    for q, style in zip([0.10, 0.50, 0.90], [":", "-", ":"]):
        ax.plot(grid, screener.curves[q].predict(pct._design(grid)), style, lw=2, color="#2b6cb0",
                zorder=2, label=f"P{int(q*100)} (typical)")
    ax.scatter([row["age_months"]], [row["mluw"]], s=260, marker="*", color="#e53e3e", zorder=4,
               edgecolor="white", linewidth=1.2, label="this child")
    ax.set(xlabel="age (months)", ylabel="sentence length (words/utterance)",
           title="Language growth chart")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.15)
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f4f6f9;margin:0;color:#1a202c}
.wrap{max-width:860px;margin:24px auto;background:#fff;border-radius:16px;box-shadow:0 6px 24px rgba(0,0,0,.08);overflow:hidden}
.head{background:linear-gradient(135deg,#2b6cb0,#1a365d);color:#fff;padding:22px 30px}
.head h1{margin:0;font-size:24px} .head p{margin:4px 0 0;opacity:.85;font-size:14px}
.body{padding:26px 30px;display:grid;grid-template-columns:1fr 1fr;gap:24px}
.card{background:#f7fafc;border:1px solid #e2e8f0;border-radius:12px;padding:18px}
.big{font-size:40px;font-weight:700;line-height:1} .sub{color:#4a5568;font-size:13px;margin-top:2px}
.flag{display:inline-block;padding:6px 14px;border-radius:999px;font-weight:700;font-size:15px}
.elev{background:#fed7d7;color:#c53030} .ok{background:#c6f6d5;color:#276749}
.drivers li{margin:5px 0} .full{grid-column:1/3}
img{width:100%;border-radius:10px;border:1px solid #e2e8f0}
.foot{padding:14px 30px;background:#edf2f7;color:#718096;font-size:12px;border-top:1px solid #e2e8f0}
table{border-collapse:collapse;width:100%;font-size:13px} td{padding:4px 0;border-bottom:1px solid #edf2f7}
"""

def build_html(title, screener, row):
    text, info, p = screener.screen(row)
    z = info["z"]; low = sorted(((f, v) for f, v in z.items() if v < -0.5), key=lambda kv: kv[1])[:3]
    elevated = p >= 0.45
    drivers = "".join(f"<li><b>{html.escape(demo.s4.PRETTY.get(f, f))}</b>: {v:+.1f} SD below age norm</li>"
                      for f, v in low) or "<li>All language markers within or above age norm.</li>"
    img = growth_plot_b64(screener, row)
    pct_txt = f"P{info.get('percentile', p):.0f}" if False else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>FirstWords - {html.escape(title)}</title>
<style>{CSS}</style></head><body><div class="wrap">
<div class="head"><h1>FirstWords - Language Screening Readout</h1>
<p>{html.escape(title)} &nbsp;|&nbsp; interpretable, research-validated screening aid (not a diagnosis)</p></div>
<div class="body">
 <div class="card"><div class="big">{info['lang_age']:.0f} mo</div>
   <div class="sub">estimated language age (chronological age {row['age_months']:.0f} mo)</div>
   <div class="sub" style="margin-top:10px;font-size:20px;font-weight:700;color:{'#c53030' if info['gap']<-6 else '#276749'}">
     gap {info['gap']:+.0f} months</div></div>
 <div class="card"><div class="sub" style="margin-bottom:8px">delay screen</div>
   <span class="flag {'elev' if elevated else 'ok'}">{'ELEVATED' if elevated else 'NOT ELEVATED'}</span>
   <div class="sub" style="margin-top:10px">risk score {p:.2f} (recall-tuned)</div></div>
 <div class="card full"><b>Markers below age norm</b><ul class="drivers">{drivers}</ul>
   {'<div class="sub">Suggested action: discuss a speech-language evaluation with a professional.</div>' if elevated else ''}</div>
 <div class="card full"><img src="data:image/png;base64,{img}"></div>
</div>
<div class="foot">Screening aid only - an ELEVATED flag means "consider a professional evaluation," not a diagnosis.
 Built on the public CHILDES database; validated child-independently with external held-out tests.</div>
</div></body></html>"""

def write(title, screener, row):
    out = os.path.join(RESULTS, f"demo_{title.split()[0].lower().replace('(','').replace(')','')}.html")
    open(out, "w", encoding="utf-8").write(build_html(title, screener, row))
    print(f"  wrote {os.path.relpath(out, ROOT)}  (double-click to open in a browser)")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--file"); ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(); sc = demo.Screener()
    if a.file:
        write(os.path.basename(a.file), sc, demo.features_from_chat(a.file)); return
    df = demo._load_features()
    enni = df[(df.corpus == "ENNI") & df.age_months.notna()].dropna(subset=demo.DELAY_FEATURES)
    write("Example-A typically-developing", sc, enni[enni.group == "TD"].sort_values("mlum", ascending=False).iloc[0])
    write("Example-B language-impaired", sc, enni[enni.group == "SLI"].sort_values("mlum").iloc[0])

if __name__ == "__main__":
    main()
