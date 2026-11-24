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

# Flag cutoff on the delay risk score. 0.36 = Stage 3's screening point (~90% sensitivity on
# held-out ENNI children: sens 0.91, spec 0.65), the number the board's "catches 91%" refers to.
# Was 0.45 (lower sensitivity) until 2026-11; see PROJECT_HANDOFF ISS-25.
SCREEN_CUTOFF = 0.36

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

def growth_plot_b64(screener, row):
    """Growth-chart figure with THIS child marked - returned as a base64 PNG for the HTML."""
    df = demo._load_features(); td = df[(df["group"] == "TD") & df["age_months"].notna()]
    grid = np.linspace(18, max(td.age_months.max(), row["age_months"]) + 2, 120)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    fig.patch.set_facecolor("#f2f0fb"); ax.set_facecolor("#ffffff")
    ax.scatter(td.age_months, td.mluw, s=7, alpha=0.22, color="#b3abd9", zorder=1)
    for q, style in zip([0.10, 0.50, 0.90], [":", "-", ":"]):
        ax.plot(grid, screener.curves[q].predict(pct._design(grid)), style, lw=2.2, color="#5a4fbf",
                zorder=2, label=f"P{int(q*100)} (typical)")
    ax.scatter([row["age_months"]], [row["mluw"]], s=300, marker="*", color="#d13b3b", zorder=4,
               edgecolor="#2d2a61", linewidth=1.3, label="this child")
    ax.set(xlabel="age (months)", ylabel="sentence length (words/utterance)")
    for s in ax.spines.values(): s.set_color("#2d2a61")
    ax.tick_params(colors="#2d2a61"); ax.xaxis.label.set_color("#241f52"); ax.yaxis.label.set_color("#241f52")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9); ax.grid(alpha=0.18, color="#b3abd9")
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130,
                                                        facecolor=fig.get_facecolor()); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

CSS = """
body{font-family:Arial,'Helvetica Neue',Helvetica,sans-serif;background:#e9e6f7;margin:0;color:#241f52}
.wrap{max-width:900px;margin:22px auto;background:#f2f0fb;border:3px solid #2d2a61;border-radius:6px;overflow:hidden}
.head{background:#7b6fce;color:#fff;padding:18px 26px;border-bottom:3px solid #2d2a61}
.head h1{margin:0;font-size:25px;font-weight:900;letter-spacing:.3px}
.head p{margin:6px 0 0;font-size:13px;color:#ece7ff}
.body{padding:20px 22px;display:grid;grid-template-columns:1fr 1fr;gap:16px}
.box{background:#dcd7f4;border:2px solid #2d2a61;border-radius:8px;overflow:hidden}
.label{background:#8478d0;color:#fff;font-weight:900;font-size:12px;letter-spacing:.7px;text-transform:uppercase;padding:7px 14px;border-bottom:2px solid #2d2a61}
.pad{padding:13px 16px}
.big{font-size:42px;font-weight:900;line-height:1;color:#241f52}
.sub{color:#3d3770;font-size:13px;margin-top:3px}
.flag{display:block;text-align:center;font-weight:900;font-size:20px;letter-spacing:1px;padding:13px;border-radius:6px}
.elev{background:#efb4b4;color:#7a1414;border:2px solid #7a1414}
.ok{background:#b4e0bd;color:#14532d;border:2px solid #14532d}
.drivers{margin:6px 0 0;padding-left:18px} .drivers li{margin:5px 0;font-size:14px}
.full{grid-column:1/3}
img{width:100%;border:2px solid #2d2a61;border-radius:6px;display:block}
.foot{padding:11px 26px;background:#cfc9ef;color:#3d3770;font-size:12px;border-top:2px solid #2d2a61}
"""

def build_html(title, screener, row):
    text, info, pctl = screener.screen(row)   # pctl = growth-chart percentile (0-100)
    p = info["p_delay"]                        # p = delay risk score (0-1) -> drives the flag
    z = info["z"]; low = sorted(((f, v) for f, v in z.items() if v < -0.5), key=lambda kv: kv[1])[:3]
    elevated = p >= SCREEN_CUTOFF
    drivers = "".join(f"<li><b>{html.escape(demo.s4.PRETTY.get(f, f))}</b>: {v:+.1f} SD below age norm</li>"
                      for f, v in low) or "<li>All language markers within or above age norm.</li>"
    img = growth_plot_b64(screener, row)
    gap_col = "#7a1414" if info['gap'] < -6 else "#14532d"
    action = ('<div class="sub" style="margin-top:8px">Suggested next step: discuss a speech-language '
              'evaluation with a professional.</div>') if elevated else ''
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>FirstWords: {html.escape(title)}</title>
<style>{CSS}</style></head><body><div class="wrap">
<div class="head"><h1>FirstWords: Language Screening Readout</h1>
<p>{html.escape(title)} &nbsp;&middot;&nbsp; an interpretable screening aid built on the CHILDES database (not a diagnosis)</p></div>
<div class="body">
 <div class="box"><div class="label">Estimated Language Age</div><div class="pad">
   <div class="big">{info['lang_age']:.0f} mo</div>
   <div class="sub">chronological age {row['age_months']:.0f} months</div>
   <div class="sub" style="margin-top:9px;font-size:19px;font-weight:900;color:{gap_col}">gap {info['gap']:+.0f} months</div></div></div>
 <div class="box"><div class="label">Delay Screen</div><div class="pad">
   <span class="flag {'elev' if elevated else 'ok'}">{'ELEVATED' if elevated else 'NOT ELEVATED'}</span>
   <div class="sub" style="margin-top:11px">risk score {p:.2f} (recall-tuned) &nbsp;&middot;&nbsp; sentence length at P{pctl:.0f} for age</div></div></div>
 <div class="box full"><div class="label">Markers Below Age Norm</div><div class="pad">
   <ul class="drivers">{drivers}</ul>{action}</div></div>
 <div class="box full"><div class="label">Language Growth Chart</div><div class="pad" style="padding:12px 12px 8px"><img src="data:image/png;base64,{img}"></div></div>
</div>
<div class="foot">Screening aid only. An ELEVATED flag means "consider a professional evaluation," not a diagnosis.
 Built on the public CHILDES database and validated on children the model never trained on.</div>
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
