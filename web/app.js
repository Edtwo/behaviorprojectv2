// FirstWords web screen. Runs fully in the browser: transcripts are parsed here, audio is
// transcribed here (Whisper base.en via transformers.js), and the words-only model in
// model.json (exported by src/export_web_model.py) scores the child. Nothing is uploaded.
const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const ASR_MODEL = "onnx-community/whisper-base.en_timestamped";
const PAUSE = 0.5;   // seconds of silence that starts a new utterance (same as audio_screen.py)

let MODEL = null, asrPromise = null, busy = false, chosen = null, mode = "file", recorder = null, chunks = [];
const $ = id => document.getElementById(id);

// ---------------------------------------------------------------- CHAT parsing
// Same rules as cha_utterances() in src/export_web_model.py (the model was trained on them).
export function chaUtterances(text, speaker = "CHI") {
  const lines = [];
  let cur = null;
  for (const raw of text.split(/\r?\n/)) {
    if (raw.startsWith("\t") && cur !== null) cur += " " + raw.trim();
    else { if (cur !== null) lines.push(cur); cur = raw; }
  }
  if (cur !== null) lines.push(cur);
  const utts = [];
  for (const ln of lines) {
    if (!ln.startsWith(`*${speaker}:`)) continue;
    let s = ln.slice(ln.indexOf(":") + 1);
    s = s.replace(/\x15[^\x15]*\x15/g, " ")
         .replace(/<[^<>]*>\s*\[\/{1,2}\]/g, " ")
         .replace(/\S+\s*\[\/{1,2}\]/g, " ")
         .replace(/\[[^\]]*\]/g, " ")
         .replace(/[<>]/g, " ");
    const words = [];
    for (let tok of s.split(/\s+/)) {
      if (!tok) continue;
      if (tok.startsWith("&") || tok.startsWith("0") || tok.startsWith("+") || tok.startsWith("(.")
          || tok === "xxx" || tok === "yyy" || tok === "www") continue;
      tok = tok.replace(/@\S*$/, "").replace(/[()]/g, "");
      tok = tok.replace(/[^A-Za-z'+_-]/g, "").toLowerCase().replace(/^['_-]+|['_-]+$/g, "");
      if (/[a-z]/.test(tok) && !["xxx", "yyy", "www"].includes(tok)) words.push(tok);
    }
    if (words.length) utts.push(words);
  }
  return utts;
}

export function chaAgeMonths(text) {
  // ages look like 5;02.15, 5;02. or 5; (CHAT allows the month and day to be blank)
  const m = text.match(/\|CHI\|(\d+);(\d*)(?:\.(\d*))?\|/);
  return m ? (+m[1]) * 12 + (+(m[2] || 0)) + (m[3] ? +m[3] / 30 : 0) : null;
}

// Plain sentences (typed or transcribed): one utterance per entry.
export function plainUtterances(list) {
  return list.map(u => (u.toLowerCase().match(/[a-z][a-z'+-]*/g) || [])
                       .map(w => w.replace(/^['-]+|['-]+$/g, "")).filter(w => /[a-z]/.test(w)))
             .filter(w => w.length);
}

// ---------------------------------------------------------------- features + model
function mattr(words, win = 50) {
  if (!words.length) return null;
  if (words.length < win) return new Set(words).size / words.length;
  let tot = 0;
  for (let i = 0; i + win <= words.length; i++) tot += new Set(words.slice(i, i + win)).size / win;
  return tot / (words.length - win + 1);
}

export function features(utts) {
  const words = utts.flat();
  if (!utts.length) return null;
  return { mluw: words.length / utts.length, mattr50: mattr(words),
           n_utts: utts.length, n_words: words.length };
}

const lin = (m, xs) => m.intercept + xs.reduce((s, x, i) => s + m.coef[i] * (x - m.mean[i]) / m.scale[i], 0);
const curve = (q, age) => MODEL.curves[q].intercept + MODEL.curves[q].coef[0] * age + MODEL.curves[q].coef[1] * age * age;

function percentileFor(value, age) {
  const qs = Object.keys(MODEL.curves).map(Number).sort((a, b) => a - b);
  const v = qs.map(q => curve(String(q), age));
  if (value <= v[0]) return v[0] > 0 ? Math.max(1, 100 * qs[0] * value / v[0]) : 1;
  if (value >= v[v.length - 1]) return 100 * qs[qs.length - 1];
  for (let i = 0; i < qs.length - 1; i++)
    if (v[i] <= value && value <= v[i + 1]) {
      const f = v[i + 1] > v[i] ? (value - v[i]) / (v[i + 1] - v[i]) : 0;
      return 100 * (qs[i] + f * (qs[i + 1] - qs[i]));
    }
  return 50;
}

export function screen(feat, age) {
  const langAge = lin(MODEL.level, [feat.mluw, feat.mattr50]);
  const p = 1 / (1 + Math.exp(-lin(MODEL.delay, [feat.mluw, feat.mattr50, age])));
  const z = {};
  for (const f of MODEL.level_features) {
    const [b1, b0] = MODEL.norms[f];
    z[f] = (feat[f] - (b1 * age + b0)) / MODEL.resid_sd[f];
  }
  const [lo, hi] = MODEL.flag_age_range_months;
  return { langAge, gap: langAge - age, p, pctl: percentileFor(feat.mluw, age), z,
           inRange: age >= lo && age <= hi,
           reliable: feat.mluw <= 1.6 * curve("0.9", age),
           elevated: p >= MODEL.cutoff };
}

// ---------------------------------------------------------------- audio
async function decodeTo16k(blob) {
  const buf = await blob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const audio = await ctx.decodeAudioData(buf);
  ctx.close();
  const off = new OfflineAudioContext(1, Math.ceil(audio.duration * 16000), 16000);
  const src = off.createBufferSource();
  src.buffer = audio; src.connect(off.destination); src.start();
  return (await off.startRendering()).getChannelData(0);
}

// Download the speech model once (it is cached by the browser afterwards). Started early,
// when the Record tab opens or an audio file is chosen, so it is usually ready by Analyze.
function modelMsg(t) {
  $("modelStatus").textContent = t;
  if (busy) setStatus(t);
}
function loadASR() {
  if (!asrPromise) {
    asrPromise = (async () => {
      modelMsg("Getting the speech model ready (about 80 MB the first time)...");
      const { pipeline, env } = await import(TRANSFORMERS);
      // Serve the model from this site (web/models/, fetched by scripts/fetch_web_model.sh):
      // much faster than the Hugging Face CDN in some browsers, and still works on networks
      // that block huggingface.co. Falls back to Hugging Face if the local copy is missing.
      env.allowLocalModels = true;
      env.localModelPath = new URL("models/", location.href).href;
      // 8-bit weights (~80 MB, the same precision as the laptop app's int8 model). Without an
      // explicit dtype the browser can fetch the ~290 MB full-precision files.
      const files = {};
      const asr = await pipeline("automatic-speech-recognition", ASR_MODEL, {
        dtype: { encoder_model: "q8", decoder_model_merged: "q8" },
        device: "wasm",
        progress_callback: p => {
          if (p.status !== "progress" || !p.total) return;
          files[p.file] = [p.loaded, p.total];
          const [got, all] = Object.values(files).reduce((a, [l, t]) => [a[0] + l, a[1] + t], [0, 0]);
          modelMsg(`Downloading the speech model (first time only): ${Math.round(got / 1e6)} of ${Math.round(all / 1e6)} MB`);
        },
      });
      modelMsg("Speech model ready.");
      return asr;
    })();
    asrPromise.catch(e => { asrPromise = null; modelMsg("Could not load the speech model: " + (e.message || e) + ". Check the internet connection and try again."); });
  }
  return asrPromise;
}

async function transcribe(blob) {
  const asr = await loadASR();
  setStatus("Transcribing on this device...");
  const out = await asr(await decodeTo16k(blob), { return_timestamps: "word", chunk_length_s: 30, stride_length_s: 5 });
  const utts = []; let cur = [], prevEnd = null;
  for (const c of out.chunks || []) {
    const w = c.text.trim(); if (!w) continue;
    const [s, e] = c.timestamp;
    const gap = prevEnd === null ? 0 : s - prevEnd;
    const endsSentence = cur.length && /[.!?]$/.test(cur[cur.length - 1]);
    if (cur.length && (gap > PAUSE || endsSentence)) { utts.push(cur.join(" ")); cur = []; }
    cur.push(w); prevEnd = e ?? s;
  }
  if (cur.length) utts.push(cur.join(" "));
  return utts.filter(u => /[a-zA-Z]/.test(u));
}

// ---------------------------------------------------------------- report
const PRETTY = { mluw: "sentence length", mattr50: "vocabulary diversity" };
const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function growthSVG(mluw, age) {
  const W = 760, H = 380, L = 62, R = 50, T = 18, B = 54;
  const x = m => L + (m / 12 - 1.5) / 9 * (W - L - R), y = v => H - B - v / 11 * (H - T - B);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Language growth chart">`;
  for (let v = 0; v <= 10; v += 2) s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#d6d1f0"/><text x="${L - 10}" y="${y(v) + 5}" font-size="13" text-anchor="end" fill="#241f52">${v}</text>`;
  for (let a = 2; a <= 10; a++) s += `<text x="${x(a * 12)}" y="${H - B + 22}" font-size="13" text-anchor="middle" fill="#241f52">${a}</text>`;
  s += `<text x="${(L + W - R) / 2}" y="${H - 10}" font-size="14" text-anchor="middle" fill="#241f52">Age (years)</text>`;
  s += `<text transform="translate(18 ${(T + H - B) / 2}) rotate(-90)" font-size="14" text-anchor="middle" fill="#241f52">Words per utterance</text>`;
  for (const [q, dash, w] of [["0.1", "6 5", 2.5], ["0.5", "", 4], ["0.9", "6 5", 2.5]]) {
    let d = "";
    for (let m = 18; m <= 120; m += 2) d += (d ? "L" : "M") + x(m).toFixed(1) + " " + y(curve(q, m)).toFixed(1);
    s += `<path d="${d}" fill="none" stroke="#5a4fbf" stroke-width="${w}" stroke-dasharray="${dash}"/>`;
    s += `<text x="${x(120) + 6}" y="${y(curve(q, 120)) + 5}" font-size="13" font-weight="700" fill="#5a4fbf">P${Math.round(q * 100)}</text>`;
  }
  const ca = Math.min(Math.max(age, 18), 126), cv = Math.min(mluw, 10.8);
  s += `<circle cx="${x(ca)}" cy="${y(cv)}" r="11" fill="#D55E00" stroke="#fff" stroke-width="3"/>`;
  s += `<text x="${x(ca)}" y="${y(cv) - 18}" font-size="14" font-weight="700" text-anchor="middle" fill="#D55E00">this child</text>`;
  return s + "</svg>";
}

function report(title, utts, feat, age, isAudio) {
  const r = screen(feat, age);
  const lowCount = feat.n_utts < 10;
  let flagHTML;
  if (!r.inRange) flagHTML = `<span class="flag na">NOT SCORED</span><div class="sub">The delay flag is validated for ages 4 to 10 only.</div>`;
  else flagHTML = `<span class="flag ${r.elevated ? "elev" : "ok"}">${r.elevated ? "ELEVATED" : "NOT ELEVATED"}</span>
    <div class="sub">risk score ${r.p.toFixed(2)} (flag at ${MODEL.cutoff.toFixed(2)}, tuned to catch about 90% of children with language impairment)</div>`;
  const low = Object.entries(r.z).filter(([, v]) => v < -0.5).sort((a, b) => a[1] - b[1]);
  const drivers = low.length ? low.map(([f, v]) => `<li><b>${PRETTY[f]}</b>: ${Math.abs(v).toFixed(1)} SD below the age norm</li>`).join("")
                             : "<li>Sentence length and vocabulary are within age norms.</li>";
  const transcript = isAudio ? `<div class="card"><div class="label">Automatic transcript (${feat.n_utts} utterances, ${feat.n_words} words)</div>
      <div class="pad" style="font-size:14px">${utts.map(u => esc(u.join(" "))).join(". ")}.</div></div>` : "";
  if (!r.reliable) return transcript + `<div class="card"><div class="label">Could not screen this sample</div><div class="pad">
      The average sentence came out to ${feat.mluw.toFixed(0)} words, longer than any child this age would speak.
      Either an adult was talking normally, or the pauses between sentences were too short for the recorder to split them.
      Try again with a short pause after each sentence. For a demo without a child, read one of the scripts in the Record tab.</div></div>`;
  // Lead with the age-normed growth-chart position. "Language age" is shown as a rough guide
  // only: at ages 4-10 its typical error is about 13 months (src/ext_level_by_range.py).
  const pc = Math.round(r.pctl), ord = n => n + (n % 100 > 10 && n % 100 < 14 ? "th" : ["th", "st", "nd", "rd"][n % 10] || "th");
  const pcTxt = pc >= 90 ? "90th or above" : pc <= 1 ? "below the 1st" : ord(pc);
  const pcCol = pc < 10 ? "#c4410f" : "#241f52";
  return `${transcript}
   <div class="grid">
    <div class="card"><div class="label">Growth-chart position</div><div class="pad">
      <div class="big" style="color:${pcCol}">${pcTxt}</div><div class="sub">percentile for sentence length, compared with typical children of the same age (${age.toFixed(0)} months)</div>
      <div class="sub" style="margin-top:8px">Rough language age: about ${r.langAge.toFixed(0)} months (typical error &plusmn;13 months)</div></div></div>
    <div class="card"><div class="label">Delay screen</div><div class="pad">${flagHTML}</div></div>
   </div>
   <div class="card"><div class="label">Markers below age norm</div><div class="pad"><ul>${drivers}</ul>
     ${lowCount ? `<div class="sub">Only ${feat.n_utts} utterances: 10 or more gives a more reliable screen.</div>` : ""}</div></div>
   <div class="card"><div class="label">Language growth chart: ${esc(title)}</div><div class="pad">${growthSVG(feat.mluw, age)}</div></div>`;
}

// ---------------------------------------------------------------- UI
function setStatus(t) { $("status").textContent = t; }
function ageInput() { return (+$("yrs").value) * 12 + (+$("mos").value); }
function choose(f, label) {
  chosen = f; $("chosen").textContent = "Selected: " + label; setStatus("Ready. Tap Analyze.");
  if (!/\.cha$/i.test(f.name)) loadASR().catch(() => {});   // audio: start the model download now
}

export async function analyzeText(text, name, fallbackAge) {
  const utts = chaUtterances(text);
  const age = chaAgeMonths(text) ?? fallbackAge;
  const feat = features(utts);
  if (!feat) throw new Error("No child (*CHI) utterances found in this transcript.");
  return { html: report(name, utts, feat, age, false), feat, age, r: screen(feat, age) };
}

async function analyze() {
  $("go").disabled = true; $("result").innerHTML = ""; busy = true;
  try {
    let html;
    if (mode === "type") {
      const utts = plainUtterances($("typed").value.split(/\n+/));
      const feat = features(utts);
      if (!feat) throw new Error("Type at least one sentence.");
      html = report("typed sentences", utts, feat, ageInput(), false);
    } else {
      if (!chosen) throw new Error("Choose a file or make a recording first.");
      if (/\.cha$/i.test(chosen.name)) {
        html = (await analyzeText(await chosen.text(), chosen.name, ageInput())).html;
      } else {
        const sents = await transcribe(chosen);
        const utts = plainUtterances(sents), feat = features(utts);
        if (!feat) throw new Error("No speech was detected. Record again, closer to the microphone.");
        html = report(chosen.name, utts, feat, ageInput(), true);
      }
    }
    $("result").innerHTML = html;
    setStatus("Done.");
  } catch (e) { setStatus(e.message || String(e)); }
  $("go").disabled = false; busy = false;
}

function wire() {
  document.querySelectorAll(".tab").forEach(b => b.onclick = () => {
    mode = b.dataset.tab;
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x === b));
    for (const t of ["file", "type", "rec"]) $("tab-" + t).classList.toggle("hidden", t !== mode);
    if (mode === "type") $("chosen").textContent = "";
    if (mode === "rec") loadASR().catch(() => {});   // start the download while they record
  });
  const drop = $("drop"), file = $("file");
  drop.onclick = () => file.click();
  file.onchange = e => e.target.files[0] && choose(e.target.files[0], e.target.files[0].name);
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => e.dataTransfer.files[0] && choose(e.dataTransfer.files[0], e.dataTransfer.files[0].name));
  $("recBtn").onclick = async () => {
    if (!recorder || recorder.state === "inactive") {
      let stream;
      try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
      catch { setStatus("Microphone blocked. Allow microphone access and try again."); return; }
      recorder = new MediaRecorder(stream); chunks = [];
      recorder.ondataavailable = e => chunks.push(e.data);
      recorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        $("recBtn").classList.remove("rec"); $("recBtn").textContent = "Record";
        const type = recorder.mimeType || "audio/webm";
        choose(new File([new Blob(chunks, { type })], "recording", { type }), "recording");
      };
      recorder.start(); $("recBtn").classList.add("rec"); $("recBtn").textContent = "Stop";
      setStatus("Recording. Tap Stop when done.");
    } else recorder.stop();
  };
  $("go").onclick = analyze;
}

MODEL = await (await fetch("model.json")).json();
$("aucTxt").textContent = MODEL.validation.auc.toFixed(2);
wire();
window.FirstWords = { analyzeText, chaUtterances, plainUtterances, features, screen, transcribe, report };
