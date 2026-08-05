#!/usr/bin/env python3
# FirstWords - RECORD-AND-SCREEN demo app. A small local web app that runs on your own
# laptop (no internet). Open it in a browser, enter the child's age, click Record, speak
# a few short sentences, click Stop. The browser records the audio, the app transcribes it
# (faster-whisper) and screens it, and shows the styled report. This is the accessibility
# point of the whole project: a family or teacher does not need a researcher-made
# transcript, just a phone or laptop microphone.
#
# RUN IT:
#   .venv/bin/python src/demo_server.py
#   then open http://localhost:5000 in Chrome (microphone works on localhost).
#
# HONEST STATUS: prototype. Speech recognition makes errors, especially for young or
# disordered speech, and utterance segmentation from audio is not solved. Present it as a
# working prototype plus a validation plan, not as validated.
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from flask import Flask, request, Response
import audio_screen as a

app = Flask(__name__)
SCREENER = None   # loaded lazily on first request (models + norms)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>FirstWords - Record and Screen</title>
<style>
 body{font-family:Arial,Helvetica,sans-serif;background:#e9e6f7;margin:0;color:#241f52}
 .wrap{max-width:760px;margin:26px auto;background:#f2f0fb;border:3px solid #2d2a61;border-radius:6px;overflow:hidden}
 .head{background:#7b6fce;color:#fff;padding:18px 26px;border-bottom:3px solid #2d2a61}
 .head h1{margin:0;font-size:24px;font-weight:900} .head p{margin:6px 0 0;font-size:13px;color:#ece7ff}
 .pad{padding:20px 26px}
 label{font-weight:700;font-size:14px;margin-right:8px}
 input{font-size:16px;padding:6px 8px;border:2px solid #2d2a61;border-radius:5px;width:66px}
 #rec{margin-top:18px;background:#7b6fce;color:#fff;border:2px solid #2d2a61;border-radius:6px;
   font-size:18px;font-weight:900;padding:14px 26px;cursor:pointer}
 #rec.recording{background:#d13b3b}
 #status{margin-top:12px;font-size:14px;color:#3d3770;min-height:20px}
 iframe{width:100%;height:640px;border:0;margin-top:8px}
</style></head><body><div class="wrap">
 <div class="head"><h1>FirstWords: Record and Screen</h1>
  <p>Enter the child's age, record a few short sentences, and see the language screen. Prototype, not a diagnosis.</p></div>
 <div class="pad">
   <label>Age:</label>
   <input id="yrs" type="number" min="1" max="12" value="5"> years
   <input id="mos" type="number" min="0" max="11" value="0"> months
   <div><button id="rec">Record</button></div>
   <div id="status">Click Record and allow microphone access.</div>
 </div>
 <iframe id="result"></iframe>
</div>
<script>
let rec=null, chunks=[];
const btn=document.getElementById('rec'), status=document.getElementById('status'), frame=document.getElementById('result');
function ageMonths(){return (+document.getElementById('yrs').value)*12 + (+document.getElementById('mos').value);}
btn.onclick=async()=>{
 if(!rec || rec.state==='inactive'){
   let stream;
   try{ stream=await navigator.mediaDevices.getUserMedia({audio:true}); }
   catch(e){ status.textContent='Microphone blocked. Allow it and reload.'; return; }
   rec=new MediaRecorder(stream); chunks=[];
   rec.ondataavailable=e=>chunks.push(e.data);
   rec.onstop=async()=>{
     btn.classList.remove('recording'); btn.textContent='Record';
     status.textContent='Transcribing and screening (a few seconds)...';
     const blob=new Blob(chunks,{type:'audio/webm'});
     const fd=new FormData(); fd.append('audio',blob,'recording.webm'); fd.append('age',ageMonths());
     const resp=await fetch('/screen',{method:'POST',body:fd});
     frame.srcdoc=await resp.text(); status.textContent='Done. Record again to try another sample.';
   };
   rec.start(); btn.classList.add('recording'); btn.textContent='Stop';
   status.textContent='Recording. Speak a few short sentences with small pauses, then click Stop.';
 } else {
   rec.stop(); rec.stream.getTracks().forEach(t=>t.stop()); status.textContent='Processing...';
 }
};
</script></body></html>"""

@app.get("/")
def index():
    return PAGE

@app.post("/screen")
def screen():
    global SCREENER
    if SCREENER is None:
        SCREENER = a.ReducedScreener()
    try:
        age = float(request.form.get("age") or 60)
    except ValueError:
        age = 60.0
    f = request.files.get("audio")
    if not f:
        return "<p style='font-family:sans-serif'>No audio received.</p>"
    tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    f.save(tmp.name)
    try:
        utts = a.transcribe(tmp.name)
        feat = a.features(utts)
        if not feat:
            return ("<html><body style='font-family:Arial;padding:24px;color:#241f52'>"
                    "<b>No speech detected.</b> Try recording again, closer to the microphone.</body></html>")
        doc, _ = a.build_report(f.filename or "recording", utts, feat, age, screener=SCREENER)
        return Response(doc, mimetype="text/html")
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass

if __name__ == "__main__":
    print("FirstWords record-and-screen app.")
    print("Open http://localhost:5000 in Chrome (microphone works on localhost).")
    print("Loading the speech model on the first recording takes a few extra seconds.")
    app.run(host="127.0.0.1", port=5000, debug=False)
