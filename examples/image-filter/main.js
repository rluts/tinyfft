import { TinyFft } from "../../dist/index.js";

const MAX_DIM = 512;

const statusEl = document.getElementById("status");
const dropEl = document.getElementById("drop");
const fileEl = document.getElementById("file");
const infoEl = document.getElementById("info");
const cutoffEl = document.getElementById("cutoff");
const cutoffValEl = document.getElementById("cutoff-val");
const maskEl = document.getElementById("mask");
const origCanvas = document.getElementById("orig");
const filteredCanvas = document.getElementById("filtered");

let fft = null;
let state = null;
let pendingFrame = 0;

async function loadFft() {
  try {
    fft = await TinyFft.load();
    setStatus(
      `WASM ready · arena ${(fft.arenaCapacity / (1024 * 1024)).toFixed(1)} MiB`,
    );
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

cutoffEl.addEventListener("input", () => {
  cutoffValEl.textContent = cutoffEl.value;
  scheduleFilter();
});
maskEl.addEventListener("change", scheduleFilter);

function scheduleFilter() {
  if (!state) return;
  if (pendingFrame) return;
  pendingFrame = requestAnimationFrame(() => {
    pendingFrame = 0;
    applyFilter();
  });
}

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

async function fileToImage(file) {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    await new Promise((res, rej) => {
      img.onload = () => res();
      img.onerror = () => rej(new Error("image decode failed"));
      img.src = url;
    });
    return img;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function imageToGray(img, maxDim) {
  let w = img.naturalWidth || img.width;
  let h = img.naturalHeight || img.height;
  if (Math.max(w, h) > maxDim) {
    const s = maxDim / Math.max(w, h);
    w = Math.max(1, Math.floor(w * s));
    h = Math.max(1, Math.floor(h * s));
  }
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const gray = new Uint8ClampedArray(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = data[4 * i];
    const g = data[4 * i + 1];
    const b = data[4 * i + 2];
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }
  return { width: w, height: h, gray };
}

function drawGray(canvas, gray, w, h) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    img.data[4 * i] = v;
    img.data[4 * i + 1] = v;
    img.data[4 * i + 2] = v;
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

async function handleFile(file) {
  if (!fft) {
    setStatus("WASM not loaded yet");
    return;
  }
  setStatus(`Decoding ${file.name}...`);
  let img;
  try {
    img = await fileToImage(file);
  } catch (e) {
    setStatus(`Decode failed: ${e.message}`);
    return;
  }

  const { width, height, gray } = imageToGray(img, MAX_DIM);
  drawGray(origCanvas, gray, width, height);

  const padW = nextPow2(width);
  const padH = nextPow2(height);

  fft.reset();
  let plan;
  try {
    plan = fft.plan2d(padW, padH);
  } catch (e) {
    setStatus(e.message);
    return;
  }

  plan.view.fill(0);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      plan.view[2 * (y * padW + x)] = gray[y * width + x] / 255 - 0.5;
    }
  }

  setStatus(`Forward FFT (${padW}×${padH})...`);
  await new Promise((r) => requestAnimationFrame(r));
  const t0 = performance.now();
  try {
    plan.forward();
  } catch (e) {
    setStatus(`fft_forward_2d failed: ${e.message}`);
    return;
  }
  const t1 = performance.now();

  const spectrum = plan.view.slice();

  const cutoffMax = Math.floor(Math.min(padW, padH) / 2);
  cutoffEl.max = cutoffMax;
  if (parseInt(cutoffEl.value, 10) > cutoffMax) {
    cutoffEl.value = Math.floor(cutoffMax / 4);
  }
  cutoffValEl.textContent = cutoffEl.value;

  state = {
    width,
    height,
    padW,
    padH,
    plan,
    spectrum,
    forwardMs: t1 - t0,
  };

  renderInfo({
    file: file.name,
    "image size": `${width}×${height}`,
    "padded size": `${padW}×${padH}`,
    "fwd 2D fft": (t1 - t0).toFixed(1) + " ms",
  });

  applyFilter();
}

function applyFilter() {
  if (!state || !fft) return;
  const { width, height, padW, padH, plan, spectrum, forwardMs } = state;
  const cutoff = parseInt(cutoffEl.value, 10);
  const mode = maskEl.value;

  const view = plan.view;
  const halfW = padW / 2;
  const halfH = padH / 2;
  const sigma = Math.max(cutoff, 0.5);
  const sigSq2 = 2 * sigma * sigma;

  for (let y = 0; y < padH; y++) {
    const fy = y <= halfH ? y : padH - y;
    const fy2 = fy * fy;
    const rowOff = y * padW * 2;
    for (let x = 0; x < padW; x++) {
      const fx = x <= halfW ? x : padW - x;
      const d2 = fx * fx + fy2;
      let m;
      if (mode === "ideal") {
        m = d2 > cutoff * cutoff ? 1 : 0;
      } else {
        m = 1 - Math.exp(-d2 / sigSq2);
      }
      const idx = rowOff + 2 * x;
      view[idx] = spectrum[idx] * m;
      view[idx + 1] = spectrum[idx + 1] * m;
    }
  }

  const t0 = performance.now();
  try {
    plan.inverse();
  } catch (e) {
    setStatus(`fft_inverse_2d failed: ${e.message}`);
    return;
  }
  const t1 = performance.now();

  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      out[y * width + x] = view[2 * (y * padW + x)];
    }
  }

  let mn = Infinity;
  let mx = -Infinity;
  for (let i = 0; i < out.length; i++) {
    const v = out[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const range = mx - mn || 1;

  const px = new Uint8ClampedArray(width * height);
  for (let i = 0; i < out.length; i++) {
    px[i] = ((out[i] - mn) / range) * 255;
  }
  drawGray(filteredCanvas, px, width, height);

  setStatus(
    `cutoff=${cutoff} · ${mode} · fwd ${forwardMs.toFixed(1)} ms · inv ${(t1 - t0).toFixed(1)} ms`,
  );
}

loadFft();
