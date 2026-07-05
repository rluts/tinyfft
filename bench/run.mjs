// tinyfft benchmark suite.
//
// Compares tinyfft (SIMD + scalar wasm) against popular JS FFT libraries on
// 1D forward and round-trip throughput across sizes, plus artifact size and
// cold-load time. Pure Node, no browser required.
//
// Usage:
//   npm run bench          # uses whatever wasm variants are already built
//   npm run bench:full     # rebuild lib + scalar variant, then bench
//
// Methodology notes are printed at the end and documented in bench/README.md.

import { readFileSync, existsSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");

const SIZES = [256, 1024, 4096, 65536];
const WARMUP_MS = 150;
const MEASURE_MS = 500;

// --- helpers ----------------------------------------------------------------

function fmt(n, digits = 1) {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

// Time `fn` for a fixed budget; return operations/second.
function timeOps(fn, warmupMs = WARMUP_MS, measureMs = MEASURE_MS) {
  // Warmup (also lets the JIT / wasm tier up).
  let end = performance.now() + warmupMs;
  while (performance.now() < end) fn();

  // Measure in batches to amortize the clock read.
  let ops = 0;
  const start = performance.now();
  end = start + measureMs;
  do {
    for (let i = 0; i < 16; i++) fn();
    ops += 16;
  } while (performance.now() < end);
  const elapsed = (performance.now() - start) / 1000;
  return ops / elapsed;
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

// Reference DFT (only used at a small size to sanity-check each backend).
function referenceForward(interleaved) {
  const n = interleaved.length / 2;
  const out = new Float64Array(n * 2);
  for (let k = 0; k < n; k++) {
    let re = 0;
    let im = 0;
    for (let t = 0; t < n; t++) {
      const ang = (-2 * Math.PI * k * t) / n;
      const c = Math.cos(ang);
      const s = Math.sin(ang);
      const xr = interleaved[2 * t];
      const xi = interleaved[2 * t + 1];
      re += xr * c - xi * s;
      im += xr * s + xi * c;
    }
    out[2 * k] = re;
    out[2 * k + 1] = im;
  }
  return out;
}

function maxRelError(got, ref) {
  let maxErr = 0;
  let maxAbs = 0;
  for (let i = 0; i < ref.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(got[i] - ref[i]));
    maxAbs = Math.max(maxAbs, Math.abs(ref[i]));
  }
  return maxErr / (maxAbs || 1);
}

// --- backends ---------------------------------------------------------------
// Each backend exposes: { name, size, coldLoadMs, make(n) -> { forward(),
// roundTrip(), verify() } }. `make` returns closures that operate on internal
// buffers (no per-call allocation) so we measure the transform, not GC.

async function loadTinyFft(wasmPath, label) {
  const mod = await import(resolve(root, "dist/index.js"));
  const { TinyFft } = mod;
  const bytes = readFileSync(wasmPath);
  const t0 = performance.now();
  const fft = await TinyFft.load(bytes);
  const coldLoadMs = performance.now() - t0;
  const size = statSync(wasmPath).size;

  return {
    name: label,
    size,
    coldLoadMs,
    make(n) {
      fft.reset();
      const plan = fft.plan1d(n);
      const signal = makeSignal(n);
      plan.view.set(signal);
      return {
        // Forward is in-place, destructive and unnormalized, so re-seed the
        // input each iteration to stay numerically stable. This adds a 2N-float
        // copy that the out-of-place fft.js does not pay — see bench/README.md.
        forward() {
          plan.view.set(signal);
          plan.forward();
        },
        // Round-trip is numerically stable (ifft(fft(x)) ≈ x), so it needs no
        // re-seed: this measures the pure transform with no copy overhead.
        roundTrip() {
          plan.forward();
          plan.inverse();
        },
        verify() {
          plan.view.set(signal);
          plan.forward();
          return maxRelError(plan.view, referenceForward(signal));
        },
      };
    },
  };
}

async function loadFftJs() {
  let FFT;
  try {
    ({ default: FFT } = await import("fft.js"));
  } catch {
    return null;
  }
  const pkgPath = resolve(here, "node_modules/fft.js/package.json");
  const size = existsSync(pkgPath)
    ? bundleSize(resolve(here, "node_modules/fft.js/lib/fft.js"))
    : 0;
  return {
    name: "fft.js (pure JS)",
    size,
    coldLoadMs: 0,
    make(n) {
      const f = new FFT(n);
      const input = Array.from(makeSignal(n));
      const out = f.createComplexArray();
      const back = f.createComplexArray();
      return {
        forward() {
          f.transform(out, input);
        },
        roundTrip() {
          f.transform(out, input);
          f.inverseTransform(back, out);
        },
        verify() {
          f.transform(out, input);
          return maxRelError(out, referenceForward(Float32Array.from(input)));
        },
      };
    },
  };
}

function bundleSize(p) {
  try {
    return statSync(p).size;
  } catch {
    return 0;
  }
}

// --- run ---------------------------------------------------------------------

async function main() {
  const simdWasm = resolve(root, "dist/tinyfft.wasm");
  const scalarWasm = resolve(here, "vendor/tinyfft-scalar.wasm");

  const backends = [];
  if (existsSync(simdWasm)) {
    backends.push(await loadTinyFft(simdWasm, "tinyfft (SIMD)"));
  } else {
    console.error(`missing ${simdWasm} — run "npm run build:lib" first`);
    process.exit(1);
  }
  if (existsSync(scalarWasm)) {
    backends.push(await loadTinyFft(scalarWasm, "tinyfft (scalar)"));
  } else {
    console.warn(
      `note: ${scalarWasm} missing — run "npm run build:scalar" to include the scalar comparison`,
    );
  }
  const fftjs = await loadFftJs();
  if (fftjs) backends.push(fftjs);

  // Environment header.
  console.log(`\ntinyfft benchmark — ${new Date().toISOString()}`);
  console.log(`node ${process.version} · ${process.platform}/${process.arch}\n`);

  // Correctness sanity check at a small size.
  console.log("Correctness (max relative error vs naive DFT, N=256):");
  for (const b of backends) {
    const err = b.make(256).verify();
    const ok = err < 1e-2;
    console.log(`  ${b.name.padEnd(20)} ${err.toExponential(2)} ${ok ? "OK" : "FAIL"}`);
  }
  console.log("");

  // Size + cold-load table.
  console.log("Artifact size & cold-load:");
  console.log(`  ${"backend".padEnd(20)} ${"size".padStart(10)} ${"cold-load".padStart(12)}`);
  for (const b of backends) {
    const size = b.size ? `${fmt(b.size / 1024, 1)} KiB` : "n/a";
    const cold = b.coldLoadMs ? `${fmt(b.coldLoadMs, 2)} ms` : "n/a";
    console.log(`  ${b.name.padEnd(20)} ${size.padStart(10)} ${cold.padStart(12)}`);
  }
  console.log("");

  // Throughput tables (forward + round-trip), MSamples/s.
  for (const mode of ["forward", "roundTrip"]) {
    console.log(`Throughput — ${mode} (MSamples/s, higher is better):`);
    const header = ["backend", ...SIZES.map((n) => `N=${n}`)];
    console.log("  " + header.map((h, i) => (i === 0 ? h.padEnd(20) : h.padStart(12))).join(""));
    for (const b of backends) {
      const cells = [b.name.padEnd(20)];
      for (const n of SIZES) {
        const ctx = b.make(n);
        const opsPerSec = timeOps(ctx[mode]);
        const msamples = (opsPerSec * n) / 1e6;
        cells.push(fmt(msamples, 1).padStart(12));
      }
      console.log("  " + cells.join(""));
    }
    console.log("");
  }

  console.log(
    "Notes: f32 (tinyfft, fft.js is f64) · single-thread · warmup " +
      `${WARMUP_MS}ms, measure ${MEASURE_MS}ms per cell · in-place, no per-call alloc.`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
