import { TinyFft } from "tinyfft";

// Single-partition overlap-add fast convolution.
//
// The impulse response (IR) is FFT'd once (zero-padded to the FFT size) and its
// spectrum cached. Each input block is transformed, multiplied bin-by-bin with
// the IR spectrum (complex multiply), inverse-transformed, and overlap-added
// into the output. FFT size = nextPow2(blockLen + irLen - 1).

const statusEl = document.getElementById("status");
const dropEl = document.getElementById("drop");
const fileEl = document.getElementById("file");
const infoEl = document.getElementById("info");
const irEl = document.getElementById("ir");
const wetEl = document.getElementById("wet");
const wetValEl = document.getElementById("wet-val");
const playBtn = document.getElementById("play");
const seekEl = document.getElementById("seek");
const timeEl = document.getElementById("time");
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
let rerenderTimer = 0;
let rafHandle = 0;

init();

async function init() {
  try {
    fft = await TinyFft.load();
    setStatus(`WASM ready · arena ${(fft.arenaCapacity / (1024 * 1024)).toFixed(1)} MiB · drop a file`);
  } catch (e) {
    setStatus(`Failed to load WASM: ${e.message}`);
    throw e;
  }
}

function setStatus(t) {
  statusEl.textContent = t;
}

function renderInfo(obj) {
  infoEl.innerHTML = Object.entries(obj)
    .map(([k, v]) => `<div>${k}: <b>${v}</b></div>`)
    .join("");
}

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

// --- Impulse response synthesis (built-in IRs) -----------------------------

function makeIR(kind, sampleRate) {
  const durations = { hall: 2.5, plate: 1.2, spring: 0.8 };
  const dur = durations[kind] ?? 1.5;
  const len = Math.floor(dur * sampleRate);
  const ir = new Float32Array(len);

  // Deterministic PRNG so IRs are stable across renders.
  let s = 0x2545f491;
  const rnd = () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 0xffffffff * 2 - 1;
  };

  if (kind === "spring") {
    // A few decaying, dispersive echoes for a "boingy" spring feel.
    const taps = [0.031, 0.057, 0.089, 0.121];
    for (let i = 0; i < len; i++) {
      const t = i / sampleRate;
      let v = rnd() * Math.exp(-t * 6) * 0.15;
      for (const tap of taps) {
        const d = Math.abs(t - tap);
        v += Math.exp(-d * 220) * Math.sin(2 * Math.PI * 90 * t) * Math.exp(-t * 3) * 0.5;
      }
      ir[i] = v;
    }
  } else {
    // Exponentially-decaying noise; "plate" decays faster / brighter.
    const decay = kind === "plate" ? 6.5 : 3.2;
    for (let i = 0; i < len; i++) {
      const t = i / sampleRate;
      ir[i] = rnd() * Math.exp(-t * decay);
    }
    // Short fade-in to avoid a click at t=0.
    const fade = Math.min(64, len);
    for (let i = 0; i < fade; i++) ir[i] *= i / fade;
  }

  // Normalize to unit energy so wet level is consistent across IRs.
  let energy = 0;
  for (let i = 0; i < len; i++) energy += ir[i] * ir[i];
  const norm = energy > 0 ? 1 / Math.sqrt(energy) : 1;
  for (let i = 0; i < len; i++) ir[i] *= norm;
  return ir;
}

// --- FFT convolution --------------------------------------------------------

function convolve(input, ir, N) {
  const irLen = ir.length;
  const blockLen = N - irLen + 1; // hop; guarantees no time aliasing
  const output = new Float32Array(input.length + irLen - 1);

  fft.reset();

  // FFT the IR once (zero-padded to N), cache its spectrum.
  const irPlan = fft.plan1d(N);
  irPlan.view.fill(0);
  for (let i = 0; i < irLen; i++) irPlan.view[2 * i] = ir[i];
  irPlan.forward();
  const irSpec = irPlan.view.slice(); // copy out; reused every block

  // Mark the arena so we can allocate the reusable block plan after the IR.
  const mark = fft.mark();
  const plan = fft.plan1d(N);
  const view = plan.view;

  for (let off = 0; off < input.length; off += blockLen) {
    const n = Math.min(blockLen, input.length - off);
    view.fill(0);
    for (let i = 0; i < n; i++) view[2 * i] = input[off + i];

    plan.forward();

    // Complex multiply: spectrum *= irSpec (bin by bin).
    for (let k = 0; k < N; k++) {
      const ar = view[2 * k];
      const ai = view[2 * k + 1];
      const br = irSpec[2 * k];
      const bi = irSpec[2 * k + 1];
      view[2 * k] = ar * br - ai * bi;
      view[2 * k + 1] = ar * bi + ai * br;
    }

    plan.inverse();

    const end = Math.min(N, output.length - off);
    for (let i = 0; i < end; i++) output[off + i] += view[2 * i];
  }

  fft.release(mark);
  return output;
}

// --- Rendering / playback ---------------------------------------------------

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
  }, 150);
}

async function reprocess() {
  if (!originalBuffer) return;
  const sampleRate = originalBuffer.sampleRate;
  const channels = originalBuffer.numberOfChannels;
  const wet = parseInt(wetEl.value, 10) / 100;
  const dry = 1 - wet;

  const ir = makeIR(irEl.value, sampleRate);
  const N = nextPow2(4096 + ir.length);

  setStatus(`Convolving (${channels} ch, IR ${ir.length} samples, N=${N})...`);
  await new Promise((r) => requestAnimationFrame(r));

  const t0 = performance.now();
  const outLen = originalBuffer.length + ir.length - 1;
  const out = audioCtx.createBuffer(channels, outLen, sampleRate);
  for (let ch = 0; ch < channels; ch++) {
    const inCh = originalBuffer.getChannelData(ch);
    const wetCh = convolve(inCh, ir, N);
    const outCh = out.getChannelData(ch);
    for (let i = 0; i < outLen; i++) {
      const dryS = i < inCh.length ? inCh[i] : 0;
      outCh[i] = dry * dryS + wet * wetCh[i];
    }
  }
  const t1 = performance.now();

  processedBuffer = out;
  const samples = originalBuffer.length * channels;
  const mps = (samples / ((t1 - t0) / 1000) / 1e6).toFixed(1);
  renderInfoEl.textContent = `render ${(t1 - t0).toFixed(0)} ms · ${mps} MSamples/s · FFT N=${N}`;
  setStatus(`Ready · ${channels} ch · ${originalBuffer.duration.toFixed(2)} s`);

  if (isPlaying) {
    const off = currentPosition();
    stopPlayback();
    startPlayback(off);
  } else {
    updateTimeDisplay(playOffset);
  }
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
  const dur = playbackDuration();
  if (!isPlaying || !audioCtx) return playOffset;
  return Math.min(dur, playOffset + (audioCtx.currentTime - playStartedAt));
}

function playbackDuration() {
  const b = bypassEl.checked ? originalBuffer : processedBuffer;
  return b ? b.duration : 0;
}

function tickPlayhead() {
  if (!isPlaying) return;
  const dur = playbackDuration();
  const t = currentPosition();
  seekEl.value = dur ? ((t / dur) * 100).toFixed(2) : "0";
  updateTimeDisplay(t);
  rafHandle = requestAnimationFrame(tickPlayhead);
}

function updateTimeDisplay(t) {
  timeEl.textContent = `${formatTime(t)} / ${formatTime(playbackDuration())}`;
}

function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

// --- UI wiring --------------------------------------------------------------

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

irEl.addEventListener("change", scheduleReprocess);
wetEl.addEventListener("input", () => {
  wetValEl.textContent = `${wetEl.value}%`;
  scheduleReprocess();
});

playBtn.addEventListener("click", () => {
  if (!processedBuffer) return;
  if (isPlaying) {
    playOffset = currentPosition();
    stopPlayback();
  } else {
    startPlayback(playOffset);
  }
});

seekEl.addEventListener("input", () => {
  if (!processedBuffer) return;
  const t = (parseFloat(seekEl.value) / 100) * playbackDuration();
  const wasPlaying = isPlaying;
  stopPlayback();
  playOffset = t;
  updateTimeDisplay(t);
  if (wasPlaying) startPlayback(t);
});

bypassEl.addEventListener("change", () => {
  if (!originalBuffer) return;
  const wasPlaying = isPlaying;
  const offset = currentPosition();
  stopPlayback();
  if (wasPlaying) startPlayback(offset);
});
