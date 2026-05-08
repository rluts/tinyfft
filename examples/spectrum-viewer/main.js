import { TinyFft } from "../../dist/index.js";

const FFT_SIZE = 1024;
const MIN_HOP = 256;
const MAX_FRAMES = 4000;
const DYN_RANGE_DB = 80;

const statusEl = document.getElementById("status");
const dropEl = document.getElementById("drop");
const fileEl = document.getElementById("file");
const infoEl = document.getElementById("info");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const axisTop = document.getElementById("axis-top");
const axisBot = document.getElementById("axis-bot");
const axisRight = document.getElementById("axis-right");

let fft = null;
let lastResult = null;
let scaleMode = "linear";

document.querySelectorAll('input[name="scale"]').forEach((el) => {
  el.addEventListener("change", (e) => {
    scaleMode = e.target.value;
    if (lastResult) renderCurrent();
  });
});

async function loadFft() {
  try {
    fft = await TinyFft.load();
    setStatus(`WASM ready · arena ${(fft.arenaCapacity / (1024 * 1024)).toFixed(1)} MiB`);
  } catch (e) {
    setStatus(`Failed to load WASM: ${e.message}`);
    throw e;
  }
}

function setStatus(text) {
  statusEl.textContent = text;
}

function renderInfo(obj) {
  infoEl.innerHTML = Object.entries(obj)
    .map(([k, v]) => `<div>${k}: <b>${v}</b></div>`)
    .join("");
}

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

async function handleFile(file) {
  if (!fft) {
    setStatus("WASM not loaded yet");
    return;
  }
  setStatus(`Decoding ${file.name}...`);
  let audio;
  try {
    const buf = await file.arrayBuffer();
    const ctxAudio = new (window.AudioContext || window.webkitAudioContext)();
    audio = await ctxAudio.decodeAudioData(buf);
    ctxAudio.close();
  } catch (e) {
    setStatus(`Decode failed: ${e.message}`);
    return;
  }

  const samples = audio.getChannelData(0);
  const sampleRate = audio.sampleRate;
  const idealHop = Math.max(MIN_HOP, Math.ceil(samples.length / MAX_FRAMES));
  const hop = nextPowerOfTwoOrSelf(idealHop);

  setStatus(`Computing STFT (N=${FFT_SIZE}, hop=${hop})...`);
  await new Promise((r) => requestAnimationFrame(r));

  const t0 = performance.now();
  const { mags, numFrames, halfN } = stft(samples, FFT_SIZE, hop);
  const t1 = performance.now();

  renderInfo({
    file: file.name,
    duration: audio.duration.toFixed(2) + " s",
    "sample rate": sampleRate + " Hz",
    channels: audio.numberOfChannels,
    "fft size": FFT_SIZE,
    hop,
    frames: numFrames,
    "stft time": (t1 - t0).toFixed(1) + " ms",
  });
  setStatus(
    `Rendered ${numFrames}×${halfN} spectrogram in ${(t1 - t0).toFixed(1)} ms`,
  );

  lastResult = {
    mags,
    numFrames,
    halfN,
    sampleRate,
    duration: audio.duration,
    fftSize: FFT_SIZE,
  };
  renderCurrent();
}

function renderCurrent() {
  if (!lastResult) return;
  const { mags, numFrames, halfN, sampleRate, duration, fftSize } = lastResult;
  if (scaleMode === "log") {
    drawLogSpectrogram(mags, numFrames, halfN, sampleRate, fftSize);
  } else {
    drawSpectrogram(mags, numFrames, halfN);
  }
  drawAxes(duration, sampleRate, fftSize);
}

function nextPowerOfTwoOrSelf(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function hannWindow(N) {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
  }
  return w;
}

function stft(samples, N, hop) {
  const halfN = N >> 1;
  const window = hannWindow(N);
  const numFrames = Math.max(0, Math.floor((samples.length - N) / hop) + 1);
  const mags = new Float32Array(numFrames * halfN);

  fft.reset();
  const plan = fft.plan1d(N);
  const view = plan.view;

  for (let f = 0; f < numFrames; f++) {
    const off = f * hop;
    for (let i = 0; i < N; i++) {
      view[2 * i] = samples[off + i] * window[i];
      view[2 * i + 1] = 0;
    }
    plan.forward();
    const base = f * halfN;
    for (let k = 0; k < halfN; k++) {
      const re = view[2 * k];
      const im = view[2 * k + 1];
      mags[base + k] = Math.hypot(re, im);
    }
  }
  return { mags, numFrames, halfN };
}

const COLOR_STOPS = [
  [0, 0, 4],
  [80, 18, 123],
  [182, 54, 121],
  [251, 135, 97],
  [252, 253, 191],
];

function colormap(t) {
  if (t <= 0) return COLOR_STOPS[0];
  if (t >= 1) return COLOR_STOPS[COLOR_STOPS.length - 1];
  const x = t * (COLOR_STOPS.length - 1);
  const i = Math.floor(x);
  const frac = x - i;
  const a = COLOR_STOPS[i];
  const b = COLOR_STOPS[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ];
}

function drawSpectrogram(mags, numFrames, halfN) {
  canvas.width = numFrames;
  canvas.height = halfN;
  const img = ctx.createImageData(numFrames, halfN);

  let maxMag = 0;
  for (let i = 0; i < mags.length; i++) {
    if (mags[i] > maxMag) maxMag = mags[i];
  }
  const ref = maxMag || 1;
  const data = img.data;

  for (let f = 0; f < numFrames; f++) {
    for (let k = 0; k < halfN; k++) {
      const m = mags[f * halfN + k] / ref;
      const db = 20 * Math.log10(m + 1e-9);
      const norm = Math.max(0, Math.min(1, (db + DYN_RANGE_DB) / DYN_RANGE_DB));
      const [r, g, b] = colormap(norm);
      const y = halfN - 1 - k;
      const idx = (y * numFrames + f) * 4;
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function buildLogBinRanges(halfN, H, sampleRate, N) {
  const fMin = sampleRate / N;
  const fMax = sampleRate / 2;
  const logMin = Math.log(fMin);
  const logMax = Math.log(fMax);
  const span = logMax - logMin;
  const ranges = new Array(H);
  const denom = H - 1 || 1;

  const freqAt = (yEdge) => {
    const t = 1 - yEdge / denom;
    return Math.exp(logMin + t * span);
  };

  for (let y = 0; y < H; y++) {
    const fHi = freqAt(y - 0.5);
    const fLo = freqAt(y + 0.5);
    let lo = Math.max(1, Math.floor((fLo * N) / sampleRate));
    let hi = Math.min(halfN - 1, Math.ceil((fHi * N) / sampleRate));
    if (hi < lo) hi = lo;
    ranges[y] = [lo, hi];
  }
  return ranges;
}

function drawLogSpectrogram(mags, numFrames, halfN, sampleRate, N) {
  const H = halfN;
  canvas.width = numFrames;
  canvas.height = H;
  const img = ctx.createImageData(numFrames, H);

  let maxMag = 0;
  for (let i = 0; i < mags.length; i++) {
    if (mags[i] > maxMag) maxMag = mags[i];
  }
  const ref = maxMag || 1;
  const data = img.data;
  const ranges = buildLogBinRanges(halfN, H, sampleRate, N);

  for (let f = 0; f < numFrames; f++) {
    const base = f * halfN;
    for (let y = 0; y < H; y++) {
      const [lo, hi] = ranges[y];
      let m = 0;
      for (let k = lo; k <= hi; k++) {
        const v = mags[base + k];
        if (v > m) m = v;
      }
      m /= ref;
      const db = 20 * Math.log10(m + 1e-9);
      const norm = Math.max(0, Math.min(1, (db + DYN_RANGE_DB) / DYN_RANGE_DB));
      const [r, g, b] = colormap(norm);
      const idx = (y * numFrames + f) * 4;
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawAxes(duration, sampleRate, N) {
  const nyquist = sampleRate / 2;
  axisTop.textContent = `${formatHz(nyquist)} (Nyquist)`;
  axisBot.textContent = scaleMode === "log" ? `${formatHz(sampleRate / N)}` : `0 Hz`;
  axisRight.textContent = `${duration.toFixed(2)} s`;
}

function formatHz(hz) {
  if (hz >= 1000) return `${(hz / 1000).toFixed(1)} kHz`;
  return `${hz.toFixed(0)} Hz`;
}

loadFft();
