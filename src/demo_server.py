#!/usr/bin/env python3
# FirstWords - the demo app. One local page (runs on your laptop, no internet). You either
# drag in a CHILDES transcript (.cha) OR record a child talking, then click Analyze and the
# screen appears. A transcript uses the full model; a recording uses the reduced model
# (sentence length + vocabulary + age) after automatic transcription.
#
# RUN IT:
#   .venv/bin/python src/demo_server.py
#   then open http://localhost:5000 in Chrome (microphone works on localhost).
#
# HONEST STATUS: the audio path is a prototype. Speech recognition makes errors and
# utterance segmentation from audio is unsolved. Present it as a working prototype.
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from flask import Flask, request, Response
import audio_screen as a
import stage5_demo as demo
import demo_report as dr

app = Flask(__name__)
_FULL = None      # full transcript model (stage5 Screener)
_REDUCED = None   # audio/reduced model

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".aac", ".aiff", ".aif", ".webm", ".ogg", ".opus", ".flac", ".caf")

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>FirstWords</title>
<style>
 *{box-sizing:border-box}
 body{font-family:Arial,Helvetica,sans-serif;background:#e9e6f7;margin:0;color:#241f52}
 .wrap{max-width:820px;margin:24px auto;background:#f2f0fb;border:3px solid #2d2a61;border-radius:8px;overflow:hidden}
 .head{background:#7b6fce;color:#fff;padding:18px 26px;border-bottom:3px solid #2d2a61}
 .head h1{margin:0;font-size:25px;font-weight:900}
 .head p{margin:6px 0 0;font-size:13px;color:#ece7ff}
 .body{padding:22px 26px}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 .card{border:2px solid #2d2a61;border-radius:8px;background:#dcd7f4;overflow:hidden}
 .card .label{background:#8478d0;color:#fff;font-weight:900;font-size:12px;letter-spacing:.6px;
   text-transform:uppercase;padding:7px 14px;border-bottom:2px solid #2d2a61}
 .card .pad{padding:16px;text-align:center}
 #drop{border:2px dashed #7b6fce;border-radius:8px;padding:22px 12px;background:#efedfb;cursor:pointer;font-size:14px;color:#3d3770}
 #drop.over{background:#ded9f7;border-color:#2d2a61}
 .btn{background:#7b6fce;color:#fff;border:2px solid #2d2a61;border-radius:6px;font-size:16px;font-weight:900;
   padding:11px 20px;cursor:pointer}
 .btn.recording{background:#d13b3b}
 .btn.big{width:100%;font-size:19px;padding:14px;margin-top:16px}
 .btn:disabled{opacity:.5;cursor:default}
 input[type=number]{font-size:16px;padding:6px;border:2px solid #2d2a61;border-radius:5px;width:60px}
 .agebar{margin-top:16px;font-weight:700;font-size:14px}
 #status{margin-top:12px;font-size:14px;color:#3d3770;min-height:20px}
 #chosen{margin-top:10px;font-size:13px;font-weight:700;color:#241f52}
 iframe{width:100%;height:660px;border:0;margin-top:6px;background:#e9e6f7}
</style></head><body><div class="wrap">
 <div class="head"><h1>FirstWords: Language Screening</h1>
  <p>Drag in a transcript, or record a child talking, then Analyze. A screening aid, not a diagnosis.</p></div>
 <div class="body">
  <div class="cols">
   <div class="card"><div class="label">Option 1 &middot; Transcript or audio file</div><div class="pad">
     <div id="drop">Drag a <b>.cha</b> transcript or an audio file here,<br>or click to choose one.</div>
     <input id="file" type="file" accept=".cha,audio/*" style="display:none">
   </div></div>
   <div class="card"><div class="label">Option 2 &middot; Record audio</div><div class="pad">
     <button id="rec" class="btn">Record</button>
     <div style="margin-top:10px;font-size:13px;color:#3d3770">Speak a few short sentences with small pauses.</div>
   </div></div>
  </div>
  <div class="agebar">Child's age (used for audio; read from the file for a transcript):
    <input id="yrs" type="number" min="1" max="12" value="5"> yrs
    <input id="mos" type="number" min="0" max="11" value="0"> mos</div>
  <div class="agebar"><label><input id="childonly" type="checkbox"> Recording has adults talking too: keep only the child's voice (experimental filter)</label></div>
  <div id="chosen"></div>
  <button id="go" class="btn big" disabled>Analyze</button>
  <div id="status">Choose a file or record, then click Analyze.</div>
 </div>
 <iframe id="result"></iframe>
</div>
<script>
let chosen=null, rec=null, chunks=[];
const drop=document.getElementById('drop'), file=document.getElementById('file'),
      go=document.getElementById('go'), status=document.getElementById('status'),
      chosenEl=document.getElementById('chosen'), frame=document.getElementById('result'),
      recBtn=document.getElementById('rec');
function ageMonths(){return (+document.getElementById('yrs').value)*12 + (+document.getElementById('mos').value);}
function setChosen(f, label){chosen=f; chosenEl.textContent='Selected: '+label; go.disabled=false;
  status.textContent='Ready. Click Analyze.';}
drop.onclick=()=>file.click();
file.onchange=e=>{if(e.target.files[0]) setChosen(e.target.files[0], e.target.files[0].name);};
['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('over');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('over');}));
drop.addEventListener('drop',e=>{if(e.dataTransfer.files[0]) setChosen(e.dataTransfer.files[0], e.dataTransfer.files[0].name);});
recBtn.onclick=async()=>{
 if(!rec || rec.state==='inactive'){
   let stream; try{stream=await navigator.mediaDevices.getUserMedia({audio:true});}
   catch(e){status.textContent='Microphone blocked. Allow it and reload.';return;}
   rec=new MediaRecorder(stream); chunks=[]; rec.ondataavailable=e=>chunks.push(e.data);
   rec.onstop=()=>{recBtn.classList.remove('recording');recBtn.textContent='Record';
     const blob=new Blob(chunks,{type:'audio/webm'});
     setChosen(new File([blob],'recording.webm',{type:'audio/webm'}),'recording (audio)');};
   rec.start(); recBtn.classList.add('recording'); recBtn.textContent='Stop';
   status.textContent='Recording. Speak, then click Stop.';
 } else {rec.stop(); rec.stream.getTracks().forEach(t=>t.stop());}
};
go.onclick=async()=>{
 if(!chosen) return;
 go.disabled=true; status.textContent='Analyzing (a few seconds)...';
 const fd=new FormData(); fd.append('file',chosen,chosen.name); fd.append('age',ageMonths()); fd.append('child_only',document.getElementById('childonly').checked?'1':'0');
 try{const resp=await fetch('/analyze',{method:'POST',body:fd}); frame.srcdoc=await resp.text();
   status.textContent='Done. Choose another, or record again.';}
 catch(e){status.textContent='Something went wrong. Try again.';}
 go.disabled=false;
};
</script></body></html>"""

@app.get("/")
def index():
    return PAGE

@app.post("/analyze")
def analyze():
    global _FULL, _REDUCED
    f = request.files.get("file")
    if not f:
        return "<p style='font-family:Arial'>No file received.</p>"
    name = f.filename or "input"
    ext = os.path.splitext(name)[1].lower()
    try:
        age = float(request.form.get("age") or 60)
    except ValueError:
        age = 60.0

    if ext == ".cha":                          # transcript -> full model
        if _FULL is None:
            _FULL = demo.Screener()
        d = tempfile.mkdtemp()
        path = os.path.join(d, "uploaded.cha")
        f.save(path)
        try:
            row = demo.features_from_chat(path)
            doc = dr.build_html(name, _FULL, row)
            return Response(doc, mimetype="text/html")
        except Exception:
            return ("<html><body style='font-family:Arial;padding:24px;color:#241f52'>"
                    "<b>Could not read that transcript.</b> It should be a CHILDES .cha file with the "
                    "%mor grammar tier.</body></html>")
        finally:
            try: os.remove(path); os.rmdir(d)
            except OSError: pass

    if ext in AUDIO_EXT or ext == "" or ext == ".webm":   # audio -> reduced model
        if _REDUCED is None:
            _REDUCED = a.ReducedScreener()
        tmp = tempfile.NamedTemporaryFile(suffix=ext or ".webm", delete=False)
        f.save(tmp.name)
        try:
            title = name
            if request.form.get("child_only") == "1":
                utts, k, n = a.child_only(tmp.name, a.transcribe_timed(tmp.name))
                title = f"{name} (child-voice filter: kept {k} of {n} sentences)"
                if k < a.MIN_CHILD_UTTS:
                    return ("<html><body style='font-family:Arial;padding:24px;color:#241f52'>"
                            f"<b>Only {k} of {n} sentences sound like the child.</b> About {a.MIN_CHILD_UTTS} "
                            "child sentences are needed to screen. Record longer, or turn the filter off if "
                            "only the child is talking. (The filter is experimental and was tested on full "
                            "home recordings, not short clips.)</body></html>")
            else:
                utts = a.transcribe(tmp.name)
            feat = a.features(utts)
            if not feat:
                return ("<html><body style='font-family:Arial;padding:24px;color:#241f52'>"
                        "<b>No speech detected.</b> Record again, closer to the microphone.</body></html>")
            doc, _ = a.build_report(title, utts, feat, age, screener=_REDUCED)
            return Response(doc, mimetype="text/html")
        finally:
            try: os.unlink(tmp.name)
            except OSError: pass

    return ("<html><body style='font-family:Arial;padding:24px;color:#241f52'>"
            f"<b>Unsupported file type ({ext}).</b> Use a .cha transcript or an audio file.</body></html>")

if __name__ == "__main__":
    # macOS uses port 5000 for AirPlay Receiver, so default to 5050 (override with PORT=...)
    port = int(os.environ.get("PORT", "5050"))
    print("FirstWords demo app.")
    print(f"Open http://localhost:{port} in Chrome. Drag in a .cha transcript or record audio, then Analyze.")
    print("The speech model loads on the first recording (a few extra seconds).")
    app.run(host="127.0.0.1", port=port, debug=False)
