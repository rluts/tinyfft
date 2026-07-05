import { TinyFft } from "tinyfft";

// Live in-browser benchmark: measures tinyfft throughput (MSamples/s) for 1D
// forward and round-trip across several sizes, using a reused plan (cached
// twiddle + digit-reversal tables) — the same hot-loop pattern the Node bench
// uses. Also reports the shipped wasm size and cold-load time.

const statusEl = document.getElementById("status");
const runBtn = document.getElementById("run");
const sizeEl = document.getElementById("wasm-size");
const coldEl = document.getElementById("cold-load");
const tableEl = document.getElementById("results");

const SIZES = [256, 1024, 4096, 16384, 65536];
const WARMUP_MS = 120;
const MEASURE_MS = 350;

let fft = null;

init();

async function init() {
  try {
    const wasmUrl = new URL("../lib/tinyfft/dist/tinyfft.wasm", import.meta.url);
    const resp = await fetch(wasmUrl);
    const bytes = await resp.arrayBuffer();
    sizeEl.textContent = `${(bytes.byteLength / 1024).toFixed(1)} KiB`;

    const t0 = performance.now();
    fft = await TinyFft.load(bytes);
    const cold = performance.now() - t0;
    coldEl.textContent = `${cold.toFixed(1)} ms`;

    setStatus("WASM ready — press Run benchmark");
    runBtn.disabled = false;
  } catch (e) {
    setStatus(`Failed to load WASM: ${e.message}`);
    throw e;
  }
}

function setStatus(t) {
  statusEl.textContent = t;
}

// A deterministic complex signal (interleaved [re, im, ...]).
function makeSignal(n) {
  const buf = new Float32Array(n * 2);
  let s = 0x9e3779b9 >>> 0;
  for (let i = 0; i < n * 2; i++) {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    buf[i] = ((s >>> 0) / 0xffffffff) * 2 - 1;
  }
  return buf;
}

// Time `fn` for a fixed budget; return operations/second.
function timeOps(fn, warmupMs = WARMUP_MS, measureMs = MEASURE_MS) {
  let end = performance.now() + warmupMs;
  while (performance.now() < end) fn();

  let ops = 0;
  const start = performance.now();
  end = start + measureMs;
  do {
    for (let i = 0; i < 8; i++) fn();
    ops += 8;
  } while (performance.now() < end);
  const elapsed = (performance.now() - start) / 1000;
  return ops / elapsed;
}

function makeRow(n) {
  const tr = document.createElement("tr");
  tr.innerHTML =
    `<td>${n.toLocaleString()}</td>` +
    `<td class="fwd">…</td>` +
    `<td class="rt">…</td>`;
  return tr;
}

async function run() {
  runBtn.disabled = true;
  tableEl.querySelector("tbody").innerHTML = "";
  setStatus("Benchmarking…");

  for (const n of SIZES) {
    const row = makeRow(n);
    tableEl.querySelector("tbody").appendChild(row);
    // Yield so the row paints before we block the thread measuring.
    await new Promise((r) => requestAnimationFrame(r));

    fft.reset();
    const plan = fft.plan1d(n);
    const signal = makeSignal(n);
    plan.view.set(signal);

    // Forward re-seeds the (destructive) input each iteration.
    const forward = () => {
      plan.view.set(signal);
      plan.forward();
    };
    // Round-trip is numerically stable; no re-seed needed.
    const roundTrip = () => {
      plan.forward();
      plan.inverse();
    };

    const fwd = (timeOps(forward) * n) / 1e6;
    row.querySelector(".fwd").textContent = fwd.toFixed(0);
    await new Promise((r) => requestAnimationFrame(r));

    const rt = (timeOps(roundTrip) * n) / 1e6;
    row.querySelector(".rt").textContent = rt.toFixed(0);
    await new Promise((r) => requestAnimationFrame(r));
  }

  setStatus("Done — numbers are MSamples/s (higher is better), this machine + browser.");
  runBtn.disabled = false;
}

runBtn.addEventListener("click", run);
