const $ = (id) => document.getElementById(id);
let sid = null;
let ws = null;
let lastE2e = null;
const entropyHist = { t: [], g: [], ch: [] };

$("btn-start").onclick = async () => {
  const body = {
    source: "brainflow",
    accelerated: $("accelerated").checked,
  };
  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  sid = data.session_id;
  $("sid").textContent = sid;
  $("btn-start").disabled = true;
  $("btn-stop").disabled = false;
  $("btn-summary").disabled = false;
  entropyHist.t = [];
  entropyHist.g = [];
  entropyHist.ch = [];
  connectWs();
};

$("btn-stop").onclick = async () => {
  if (!sid) return;
  const res = await fetch(`/api/sessions/${sid}/stop`, { method: "POST" });
  const summary = await res.json();
  $("summary").textContent = JSON.stringify(summary, null, 2);
  $("btn-start").disabled = false;
  $("btn-stop").disabled = true;
  if (ws) ws.close();
};

$("btn-summary").onclick = async () => {
  if (!sid) return;
  const res = await fetch(`/api/sessions/${sid}/summary`);
  $("summary").textContent = JSON.stringify(await res.json(), null, 2);
};

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/sessions/${sid}`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "status") {
      $("state").textContent = msg.state;
      $("state").style.color =
        msg.state === "OK" ? "#3dd6c6" : msg.state === "BACKPRESSURE" ? "#f5c542" : "#f07178";
      $("sps").textContent = (msg.throughput_sps || 0).toFixed(1);
      $("wins").textContent = msg.windows;
      $("drop").textContent = msg.dropped_chunks;
      $("q").textContent = `${msg.qsize}/${msg.qmax}`;
    } else if (msg.type === "entropy") {
      lastE2e = msg.e2e_ms;
      $("e2e").textContent = lastE2e == null ? "—" : `${lastE2e.toFixed(1)} ms`;
      entropyHist.t.push(msg.t_end);
      entropyHist.g.push(msg.global_smoothed);
      entropyHist.ch.push(msg.smoothed);
      if (entropyHist.t.length > 240) {
        entropyHist.t.shift();
        entropyHist.g.shift();
        entropyHist.ch.shift();
      }
      drawEntropy();
    } else if (msg.type === "waveform") {
      drawWave(msg);
    }
  };
}

function drawWave(msg) {
  const cv = $("wave");
  const ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height;
  ctx.fillStyle = "#0b1015";
  ctx.fillRect(0, 0, w, h);
  const samples = msg.samples;
  if (!samples || !samples.length || !samples[0].length) return;
  const nCh = samples.length;
  const n = samples[0].length;
  const rowH = h / nCh;
  ctx.strokeStyle = "#3dd6c6";
  ctx.lineWidth = 1;
  for (let ch = 0; ch < nCh; ch++) {
    const row = samples[ch];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = row[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    const span = Math.max(mx - mn, 1e-6);
    const y0 = rowH * ch;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w;
      const y = y0 + rowH * (1 - (row[i] - mn) / span) * 0.9 + rowH * 0.05;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = "#8b9aab";
    ctx.fillText(msg.channels[ch] || `ch${ch}`, 6, y0 + 12);
  }
}

function drawEntropy() {
  const cv = $("ent");
  const ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height;
  ctx.fillStyle = "#0b1015";
  ctx.fillRect(0, 0, w, h);
  const t = entropyHist.t;
  if (t.length < 2) return;
  const n = t.length;
  ctx.strokeStyle = "#f5c542";
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * w;
    const v = entropyHist.g[i];
    const y = h - Math.max(0, Math.min(1, v ?? 0)) * (h - 8) - 4;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  const last = entropyHist.ch[entropyHist.ch.length - 1] || [];
  ctx.fillStyle = "#8b9aab";
  ctx.fillText("global smoothed (yellow)  range [0,1]", 8, 14);
  last.forEach((v, i) => {
    if (v == null) return;
    ctx.fillText(`ch${i + 1}:${v.toFixed(3)}`, 8 + (i % 4) * 140, 28 + Math.floor(i / 4) * 12);
  });
}
