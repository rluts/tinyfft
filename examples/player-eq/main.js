import { TinyFft } from "tinyfft";

const FFT_SIZE = 2048;
const HOP = FFT_SIZE / 2;
const BANDS = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
const DB_RANGE = 12;
const RERENDER_DEBOUNCE_MS = 220;

const statusEl = document.getElementById("status");
const dropEl = document.getElementById("drop");
const fileEl = document.getElementById("file");
const infoEl = document.getElementById("info");
const playBtn = document.getElementById("play");
const seekEl = document.getElementById("seek");
const timeEl = document.getElementById("time");
const curveCanvas = document.getElementById("curve");
const curveCtx = curveCanvas.getContext("2d");
const bandsEl = document.getElementById("bands");
const resetBtn = document.getElementById("reset");
const bypassEl = document.getElementById("bypass");
const renderInfoEl = document.getElementById("render-info");

let fft = null;
let audioCtx = null;
let originalBuffer = null;
let processedBuffer = null;
let source = null;
let isPlaying = false;
let playStartedAt = 0;
let playOffset = 0;
let bandGainsDb = BANDS.map(() => 0);
let rerenderTimer = 0;
let rafHandle = 0;

buildBandUI();

async function init() {
  try {
    fft = await TinyFft.load();
    setStatus(
      `WASM ready · arena ${(fft.arenaCapacity / (1024 * 1024)).toFixed(1)} MiB · drop a file`,
    );
  } catch (e) {
    setStatus(`Failed to load WASM: ${e.message}`);
    throw e;
  }
  drawCurve();
}
init();

function setStatus(t) {
  statusEl.textContent = t;
}

function renderInfo(obj) {
  infoEl.innerHTML = Object.entries(obj)
    .map(([k, v]) => `<div>${k}: <b>${v}</b></div>`)
    .join("");
}

function buildBandUI() {
  bandsEl.innerHTML = "";
  BANDS.forEach((freq, i) => {
    const wrap = document.createElement("div");
    wrap.className = "band";
    wrap.innerHTML = `
      <span class="freq">${freq < 1000 ? freq + " Hz" : freq / 1000 + " kHz"}</span>
      <input type="range" min="${-DB_RANGE}" max="${DB_RANGE}" step="0.1" value="0" data-i="${i}" />
      <span class="db">0.0 dB</span>
    `;
    bandsEl.appendChild(wrap);
    const slider = wrap.querySelector("input");
    const readout = wrap.querySelector(".db");
    slider.addEventListener("input", () => {
      bandGainsDb[i] = parseFloat(slider.value);
      readout.textContent = `${bandGainsDb[i].toFixed(1)} dB`;
      drawCurve();
      scheduleReprocess();
    });
  });
}

resetBtn.addEventListener("click", () => {
  bandGainsDb = BANDS.map(() => 0);
  bandsEl.querySelectorAll("input").forEach((el, i) => {
    el.value = "0";
    el.parentElement.querySelector(".db").textContent = "0.0 dB";
  });
  drawCurve();
  scheduleReprocess();
});

bypassEl.addEventListener("change", () => {
  if (!originalBuffer) return;
  const wasPlaying = isPlaying;
  const offset = currentPosition();
  stopPlayback();
  if (wasPlaying) startPlayback(offset);
});

["dragenter", "dragover"].forEach((t) =>
  dropEl.addEventListener(t, (e) => {
    e.preventDefault();
    dropEl.classList.add("hover");
  }),
);
["dragleave", "drop"].forEach((t) =>
  dropEl.addEventListener(t, (e) => {
    e.preventDefault();
    dropEl.classList.remove("hover");
  }),
);
dropEl.addEventListener("drop", (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) handleFile(f);
});
fileEl.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) handleFile(f);
});

playBtn.addEventListener("click", () => {
  if (!processedBuffer) return;
  if (isPlaying) {
    const off = currentPosition();
    stopPlayback();
    playOffset = off;
  } else {
    startPlayback(playOffset);
  }
});

seekEl.addEventListener("input", () => {
  if (!processedBuffer) return;
  const t = (parseFloat(seekEl.value) / 100) * processedBuffer.duration;
  const wasPlaying = isPlaying;
  stopPlayback();
  playOffset = t;
  updateTimeDisplay(t);
  if (wasPlaying) startPlayback(t);
});

async function handleFile(file) {
  if (!fft) return setStatus("WASM not loaded yet");
  setStatus(`Decoding ${file.name}...`);
  stopPlayback();
  playOffset = 0;
  try {
    const buf = await file.arrayBuffer();
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") await audioCtx.resume();
    originalBuffer = await audioCtx.decodeAudioData(buf);
  } catch (e) {
    setStatus(`Decode failed: ${e.message}`);
    return;
  }

  renderInfo({
    file: file.name,
    duration: originalBuffer.duration.toFixed(2) + " s",
    "sample rate": originalBuffer.sampleRate + " Hz",
    channels: originalBuffer.numberOfChannels,
    "fft size": FFT_SIZE,
    hop: HOP,
  });
  await reprocess();
  playBtn.disabled = false;
  seekEl.disabled = false;
}

function scheduleReprocess() {
  if (!originalBuffer) return;
  if (rerenderTimer) clearTimeout(rerenderTimer);
  rerenderTimer = setTimeout(() => {
    rerenderTimer = 0;
    reprocess();
  }, RERENDER_DEBOUNCE_MS);
}

async function reprocess() {
  if (!originalBuffer) return;
  const sampleRate = originalBuffer.samplyeRate;
  const channels = originalBuffer.numberOfChannels;
  const len = originalBuffer.length;

  setStatus(`Processing ${channels} ch × ${len} samples through FFT EQ...`);
  await new Promise((r) => requestAnimationFrame(r));

  const t0 = performance.now();
  const gainCurve = buildGainCurve(FFT_SIZE, sampleRate, bandGainsDb);
  const out = audioCtx.createBuffer(channels, len, sampleRate);
  for (let ch = 0; ch < channels; ch++) {
    const inCh = originalBuffer.getChannelData(ch);
    const outCh = eqOverlapAdd(inCh, gainCurve, FFT_SIZE);
    out.copyToChannel(outCh, ch);
  }
  const t1 = performance.now();

  processedBuffer = out;
  renderInfoEl.textContent = `EQ render: ${(t1 - t0).toFixed(0)} ms`;
  setStatus(
    `Ready · ${channels} ch · ${(originalBuffer.duration).toFixed(2)} s · processed in ${(t1 - t0).toFixed(0)} ms`,
  );

  if (isPlaying) {
    const off = currentPosition();
    stopPlayback();
    startPlayback(off);
  } else {
    updateTimeDisplay(playOffset);
  }
}

function eqOverlapAdd(input, gain, N) {
  const hop = N / 2;
  const win = hannWindow(N);
  const output = new Float32Array(input.length);
  fft.reset();
  const plan = fft.plan1d(N);
  const view = plan.view;
  const halfN = N / 2;

  for (let off = 0; off + N <= input.length; off += hop) {
    for (let i = 0; i < N; i++) {
      view[2 * i] = input[off + i] * win[i];
      view[2 * i + 1] = 0;
    }
    plan.forward();

    for (let k = 0; k <= halfN; k++) {
      const g = gain[k];
      view[2 * k] *= g;
      view[2 * k + 1] *= g;
      if (k > 0 && k < halfN) {
        view[2 * (N - k)] *= g;
        view[2 * (N - k) + 1] *= g;
      }
    }

    plan.inverse();

    for (let i = 0; i < N; i++) {
      output[off + i] += view[2 * i];
    }
  }
  return output;
}

function hannWindow(N) {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
  }
  return w;
}

function buildGainCurve(N, sampleRate, dbPerBand) {
  const halfN = N / 2;
  const out = new Float32Array(halfN + 1);
  const logBands = BANDS.map(Math.log);

  for (let k = 0; k <= halfN; k++) {
    const f = (k * sampleRate) / N;
    let db;
    if (f <= 0) {
      db = dbPerBand[0];
    } else {
      const lf = Math.log(f);
      if (lf <= logBands[0]) db = dbPerBand[0];
      else if (lf >= logBands[logBands.length - 1]) db = dbPerBand[dbPerBand.length - 1];
      else {
        let i = 0;
        while (i < logBands.length - 1 && lf > logBands[i + 1]) i++;
        const t = (lf - logBands[i]) / (logBands[i + 1] - logBands[i]);
        db = dbPerBand[i] * (1 - t) + dbPerBand[i + 1] * t;
      }
    }
    out[k] = Math.pow(10, db / 20);
  }
  return out;
}

function drawCurve() {
  const dpr = window.devicePixelRatio || 1;
  const w = curveCanvas.clientWidth || 800;
  const h = curveCanvas.clientHeight || 100;
  curveCanvas.width = w * dpr;
  curveCanvas.height = h * dpr;
  curveCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  curveCtx.fillStyle = "#050505";
  curveCtx.fillRect(0, 0, w, h);

  const fMin = BANDS[0] / 1.5;
  const fMax = BANDS[BANDS.length - 1] * 1.2;
  const logMin = Math.log(fMin);
  const logMax = Math.log(fMax);
  const span = logMax - logMin;

  curveCtx.strokeStyle = "#1f1f24";
  curveCtx.lineWidth = 1;
  curveCtx.beginPath();
  curveCtx.moveTo(0, h / 2);
  curveCtx.lineTo(w, h / 2);
  curveCtx.stroke();

  curveCtx.strokeStyle = "#6c79ff";
  curveCtx.lineWidth = 2;
  curveCtx.beginPath();
  const samples = 256;
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const f = Math.exp(logMin + t * span);
    const db = interpolateDb(f, bandGainsDb);
    const x = t * w;
    const y = h / 2 - (db / DB_RANGE) * (h / 2 - 4);
    if (i === 0) curveCtx.moveTo(x, y);
    else curveCtx.lineTo(x, y);
  }
  curveCtx.stroke();

  curveCtx.fillStyle = "#6c79ff";
  for (let i = 0; i < BANDS.length; i++) {
    const t = (Math.log(BANDS[i]) - logMin) / span;
    const x = t * w;
    const db = bandGainsDb[i];
    const y = h / 2 - (db / DB_RANGE) * (h / 2 - 4);
    curveCtx.beginPath();
    curveCtx.arc(x, y, 3, 0, Math.PI * 2);
    curveCtx.fill();
  }
}

function interpolateDb(f, dbPerBand) {
  const logBands = BANDS.map(Math.log);
  const lf = Math.log(f);
  if (lf <= logBands[0]) return dbPerBand[0];
  if (lf >= logBands[logBands.length - 1]) return dbPerBand[dbPerBand.length - 1];
  let i = 0;
  while (i < logBands.length - 1 && lf > logBands[i + 1]) i++;
  const t = (lf - logBands[i]) / (logBands[i + 1] - logBands[i]);
  return dbPerBand[i] * (1 - t) + dbPerBand[i + 1] * t;
}

function startPlayback(offset) {
  if (!processedBuffer || !audioCtx) return;
  source = audioCtx.createBufferSource();
  source.buffer = bypassEl.checked ? originalBuffer : processedBuffer;
  source.connect(audioCtx.destination);
  source.onended = () => {
    if (!isPlaying) return;
    isPlaying = false;
    playOffset = 0;
    playBtn.textContent = "▶ Play";
    cancelAnimationFrame(rafHandle);
    seekEl.value = "0";
    updateTimeDisplay(0);
  };
  source.start(0, offset);
  playStartedAt = audioCtx.currentTime;
  playOffset = offset;
  isPlaying = true;
  playBtn.textContent = "❚❚ Pause";
  tickPlayhead();
}

function stopPlayback() {
  if (source) {
    try {
      source.onended = null;
      source.stop();
    } catch {}
    source.disconnect();
    source = null;
  }
  isPlaying = false;
  playBtn.textContent = "▶ Play";
  cancelAnimationFrame(rafHandle);
}

function currentPosition() {
  if (!isPlaying || !audioCtx || !processedBuffer) return playOffset;
  const elapsed = audioCtx.currentTime - playStartedAt;
  return Math.min(processedBuffer.duration, playOffset + elapsed);
}

function tickPlayhead() {
  if (!isPlaying || !processedBuffer) return;
  const t = currentPosition();
  seekEl.value = ((t / processedBuffer.duration) * 100).toFixed(3);
  updateTimeDisplay(t);
  rafHandle = requestAnimationFrame(tickPlayhead);
}

function updateTimeDisplay(t) {
  const dur = processedBuffer ? processedBuffer.duration : 0;
  timeEl.textContent = `${formatTime(t)} / ${formatTime(dur)}`;
}

function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

window.addEventListener("resize", drawCurve);
