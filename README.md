# tinyfft

[![CI](https://github.com/rluts/tinyfft/actions/workflows/ci.yml/badge.svg)](https://github.com/rluts/tinyfft/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/tinyfft.svg)](https://www.npmjs.com/package/tinyfft)

Tiny FFT for the browser and Node, written in `no_std` Rust and compiled to WebAssembly. **Radix-4 with wasm SIMD**, ~11 KB wasm shipped as a raw file. 1D and 2D, in-place, single-precision `f32`. Zero runtime dependencies.

**Live demos:** [WAV Spectrum Viewer](https://rluts.github.io/tinyfft/examples/spectrum-viewer/) · [FFT Image Filter](https://rluts.github.io/tinyfft/examples/image-filter/) · [Convolution Reverb](https://rluts.github.io/tinyfft/examples/convolution-reverb/)

```bash
npm install tinyfft
```

```ts
import { TinyFft, interleave, magnitudes } from "tinyfft";

const fft = await TinyFft.load();   // loads the co-located tinyfft.wasm

// One-shot 1D
const real = Float32Array.from({ length: 16 }, (_, i) => Math.sin(i));
const spectrum = fft.forward(interleave(real));
const mags = magnitudes(spectrum);

// Hot-loop 1D (no per-call allocations)
fft.reset();
const plan = fft.plan1d(1024);
for (const frame of frames) {
  for (let i = 0; i < 1024; i++) {
    plan.view[2 * i]     = frame[i];
    plan.view[2 * i + 1] = 0;
  }
  plan.forward();
  // read plan.view (interleaved [re, im]) — same Float32Array, mutated in place
}

// 2D
fft.reset();
const p2 = fft.plan2d(256, 256);
// fill p2.view, then:
p2.forward();
// ... mask in frequency domain ...
p2.inverse();
```

## API

| Symbol                              | Meaning                                                                                  |
| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| `TinyFft.load(source?)`             | Async factory. With no arg, loads the co-located `tinyfft.wasm` (fetch in the browser, file read in Node). Pass a `BufferSource`, `Response`, or `Promise<Response>` to load from elsewhere. |
| `fft.plan1d(n)` → `Plan1D`          | Allocates `n` complex samples in wasm memory. `n` must be a power of two.                |
| `fft.plan2d(width, height)` → `Plan2D` | Allocates `width × height` plus an equally-sized scratch buffer. Both dims power of two. |
| `plan.view` (`Float32Array`)        | Interleaved `[re, im, …]` view directly over wasm memory. Read/write in place.           |
| `plan.forward() / plan.inverse()`   | Run the transform. Inverse normalizes by `1/N` (1D) or `1/(W·H)` (2D).                   |
| `fft.forward(buf) / fft.inverse(buf)` | One-shot 1D. Returns a fresh `Float32Array` (copies out of wasm memory).               |
| `fft.forward2d(buf, w, h) / fft.inverse2d(...)` | One-shot 2D.                                                                  |
| `fft.reset()`                       | Resets the wasm bump arena. Invalidates any existing plans/views.                        |
| `fft.mark()` / `fft.release(mark)`  | Stack-style arena scoping: `mark()` records the arena position; `release(mark)` frees everything allocated since, LIFO. Finer-grained than `reset()`. |
| `fft.arenaCapacity` (number)        | Total bytes available in the arena (8 MiB by default).                                   |
| `interleave(real, imag?)`           | Helper: build `[re, im, …]` from real (and optional imag).                               |
| `interleaveInto(out, real, imag?)`  | Same, into a preallocated `out` (no allocation).                                         |
| `magnitudes(buf)`                   | Helper: per-bin `\|X[k]\|`.                                                               |
| `magnitudesInto(out, buf)`          | Same, into a preallocated `out` (no allocation).                                         |
| `FftError`                          | Thrown on wasm error codes (1 = not power of two, 2 = empty/null).                       |

Errors during plan creation come back as plain `Error` (e.g. arena exhaustion); errors from the wasm transform itself come back as `FftError`.

## Precision

All math is single-precision **`f32`**, chosen deliberately for small binary size and 4-lane wasm SIMD. There is no `f64` variant. Round-trip error (`ifft(fft(x)) ≈ x`) is typically within ~`1e-4`–`1e-3` relative, growing slowly with `N`. For scientific / high-dynamic-range work that needs double precision, use an `f64` FFT library instead.

## Normalization

The forward transform is **unnormalized**; the inverse divides by `1/N` (1D) or `1/(W·H)` (2D), applied per pass so 2D inverse is automatically `1/(W·H)`. Thus `ifft(fft(x)) ≈ x`.

## Memory model

The wasm module owns a fixed 8 MiB bump-allocated arena. Plans allocate from it. `fft.reset()` rewinds the bump pointer to zero (cheap), invalidating every plan. For finer control, use `fft.mark()` / `fft.release(mark)` to free allocations LIFO — e.g. cache one long-lived plan, then mark/release around temporary ones.

`plan.view` is a live view over wasm memory. The arena is fixed-size so the memory never grows in practice; if it ever did, the previous `Float32Array` would detach — always read `plan.view` fresh rather than caching it across operations that could grow memory.

For larger workloads, increase `ARENA_BYTES` in `src/lib.rs` and rebuild.

## Building from source

You need Rust with the `wasm32-unknown-unknown` target, Node ≥ 18, and (optionally) [binaryen](https://github.com/WebAssembly/binaryen)'s `wasm-opt` for the size-optimized build.

```bash
rustup target add wasm32-unknown-unknown
npm install
npm run build      # cargo build (SIMD) + tsc + wasm-opt -O3 into dist/
npm test           # cargo test + build + smoke test + vitest
```

wasm SIMD is enabled via `.cargo/config.toml` (`-C target-feature=+simd128`). `npm run build` produces `dist/`:

```
dist/index.js          # ESM entry
dist/index.d.ts        # types
dist/tinyfft.wasm      # raw wasm (~11 KB, radix-4 + SIMD, wasm-opt -O3)
```

If `wasm-opt` isn't installed the build still works and copies the unoptimized wasm (a few KB larger). Total tarball published to npm: ~13 KB.

## Examples

Three browser demos live under [examples/](examples). They build tinyfft from the local source (the examples `postinstall` runs `npm run build` in the repo root and copies `dist/index.js` + `dist/tinyfft.wasm` into `examples/lib/`), so demos always reflect your working tree — no published npm version needed.

Live: <https://rluts.github.io/tinyfft/>

| Demo | Source | Live |
| ---- | ------ | ---- |
| Spectrum viewer (1D STFT) — drop a WAV, see its spectrogram (linear or log freq, magma colormap). | [examples/spectrum-viewer/](examples/spectrum-viewer) | [demo](https://rluts.github.io/tinyfft/examples/spectrum-viewer/) |
| Image high-pass filter (2D) — drop an image, Gaussian or ideal cutoff, live slider. | [examples/image-filter/](examples/image-filter) | [demo](https://rluts.github.io/tinyfft/examples/image-filter/) |
| Convolution reverb (1D) — drop audio, convolve with an impulse response via FFT overlap-add, wet/dry mix. | [examples/convolution-reverb/](examples/convolution-reverb) | [demo](https://rluts.github.io/tinyfft/examples/convolution-reverb/) |

Run locally:

```bash
cd examples
npm install         # serve + tinyfft (postinstall copies the package into ./lib)
npm run dev         # static server on :3000
# open http://localhost:3000/spectrum-viewer/
# or   http://localhost:3000/image-filter/
# or   http://localhost:3000/convolution-reverb/
```

The live site is built and deployed by [.github/workflows/pages.yml](.github/workflows/pages.yml) on every push to `main`.

## Releasing

CI on every PR runs `cargo test`, builds the wasm + ts (with `wasm-opt`), runs the smoke + vitest tests, and verifies `npm pack`.

Tags `v*` trigger [.github/workflows/release.yml](.github/workflows/release.yml), which rebuilds, verifies the tag matches `package.json` `version`, and runs `npm publish`. Publishing uses npm **OIDC trusted publishing** (configured in the npm package settings for this repo/workflow) — no `NPM_TOKEN` secret required, and provenance is attached automatically.

```bash
npm version 0.2.1   # bumps package.json and creates a v0.2.1 tag
git push --follow-tags
```

## Algorithm

Iterative Cooley–Tukey **radix-4** over `f32`, with a single **radix-2** stage when `log2(N)` is odd (so all power-of-two sizes work). Input is reordered by a base-4 digit-reversal permutation, then combined with radix-4 butterflies (three twiddles `w, w², w³` plus a free `±j` rotation). On `wasm32` the butterfly runs on 128-bit SIMD (`v128`), processing two groups per iteration; a scalar path covers the remainder and host builds. 2D is row FFTs → blocked (cache-friendly) transpose → row FFTs → transpose back, reusing the 1D code. Inverse normalization is per-pass so 2D inverse is automatically `1/(W·H)`.

## License

[MIT](LICENSE)
