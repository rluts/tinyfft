# tinyfft

[![CI](https://github.com/rluts/tinyfft/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/tinyfft/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/tinyfft.svg)](https://www.npmjs.com/package/tinyfft)

Tiny FFT for the browser and Node, written in `no_std` Rust and compiled to WebAssembly. ~11 KB wasm, embedded as base64 in the JS bundle. 1D and 2D, in-place. Zero runtime dependencies.

**Live demos:** [WAV Spectrum Viewer](https://rluts.github.io/tinyfft/examples/spectrum-viewer/) · [FFT Image Filter](https://rluts.github.io/tinyfft/examples/image-filter/)

```bash
npm install tinyfft
```

```ts
import { TinyFft, interleave, magnitudes } from "tinyfft";

const fft = await TinyFft.load();   // wasm is bundled, no network fetch needed

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
| `TinyFft.load(source?)`             | Async factory. With no arg, uses bundled wasm. Pass a `BufferSource`, `Response`, or `Promise<Response>` to load from elsewhere. |
| `fft.plan1d(n)` → `Plan1D`          | Allocates `n` complex samples in wasm memory. `n` must be a power of two.                |
| `fft.plan2d(width, height)` → `Plan2D` | Allocates `width × height` plus an equally-sized scratch buffer. Both dims power of two. |
| `plan.view` (`Float32Array`)        | Interleaved `[re, im, …]` view directly over wasm memory. Read/write in place.           |
| `plan.forward() / plan.inverse()`   | Run the transform. Inverse normalizes by `1/N` (1D) or `1/(W·H)` (2D).                   |
| `fft.forward(buf) / fft.inverse(buf)` | One-shot 1D. Returns a fresh `Float32Array` (copies out of wasm memory).               |
| `fft.forward2d(buf, w, h) / fft.inverse2d(...)` | One-shot 2D.                                                                  |
| `fft.reset()`                       | Resets the wasm bump arena. Invalidates any existing plans/views.                        |
| `fft.arenaCapacity` (number)        | Total bytes available in the arena (8 MiB by default).                                   |
| `interleave(real, imag?)`           | Helper: build `[re, im, …]` from real (and optional imag).                               |
| `magnitudes(buf)`                   | Helper: per-bin `|X[k]|`.                                                                |
| `FftError`                          | Thrown on wasm error codes (1 = not power of two, 2 = empty/null).                       |

Errors during plan creation come back as plain `Error` (e.g. arena exhaustion); errors from the wasm transform itself come back as `FftError`.

## Memory model

The wasm module owns a fixed 8 MiB bump-allocated arena. Plans allocate from it. `fft.reset()` rewinds the bump pointer (cheap), invalidating every plan. There is no per-plan free — manage lifetimes by resetting between batches.

For larger workloads, increase `ARENA_BYTES` in `src/lib.rs` and rebuild.

## Building from source

You need Rust with the `wasm32-unknown-unknown` target and Node ≥ 18.

```bash
rustup target add wasm32-unknown-unknown
npm install
npm run build      # cargo build + base64-embed wasm + tsc
npm test           # cargo test + smoke test
```

`npm run build` produces `dist/`:

```
dist/index.js          # ESM entry (~6.5 KB)
dist/index.d.ts        # types
dist/wasm-bytes.js     # auto-generated base64 string (~15 KB)
```

Total tarball published to npm: ~21 KB.

## Examples

Three browser demos live under [examples/](examples). All consume the package via the npm library (`import "tinyfft"`).

Live: <https://rluts.github.io/tinyfft/>

| Demo | Source | Live |
| ---- | ------ | ---- |
| Spectrum viewer (1D STFT) — drop a WAV, see its spectrogram (linear or log freq, magma colormap). | [examples/spectrum-viewer/](examples/spectrum-viewer) | [demo](https://rluts.github.io/tinyfft/examples/spectrum-viewer/) |
| Image high-pass filter (2D) — drop an image, Gaussian or ideal cutoff, live slider. | [examples/image-filter/](examples/image-filter) | [demo](https://rluts.github.io/tinyfft/examples/image-filter/) |
| FFT equalizer player (1D OLA) — drop audio, play through a 10-band ISO octave EQ at ±12 dB. | [examples/player-eq/](examples/player-eq) | [demo](https://rluts.github.io/tinyfft/examples/player-eq/) |

Run locally:

```bash
cd examples
npm install         # serve + tinyfft (postinstall copies the package into ./lib)
npm run dev         # static server on :3000
# open http://localhost:3000/spectrum-viewer/
# or   http://localhost:3000/image-filter/
# or   http://localhost:3000/player-eq/
```

The live site is built and deployed by [.github/workflows/pages.yml](.github/workflows/pages.yml) on every push to `main`.

## Releasing

CI on every PR runs `cargo test`, builds the wasm + ts, runs the smoke test, and verifies `npm pack`.

Tags `v*` trigger [.github/workflows/release.yml](.github/workflows/release.yml), which rebuilds, verifies the tag matches `package.json` `version`, and runs `npm publish --provenance --access public`. Set the `NPM_TOKEN` secret in the repo for this to work.

```bash
npm version 0.1.1   # bumps package.json and creates a v0.1.1 tag
git push --follow-tags
```

## Algorithm

Iterative Cooley–Tukey radix-2 over `f32`. Trig per stage via the recurrence `w *= w_step` (one `cos`/`sin` per stage, not per butterfly). 2D is row FFTs → out-of-place transpose → row FFTs → transpose back, reusing the 1D code. Inverse normalization is per-pass so 2D inverse is automatically `1/(W·H)`.

## License

[MIT](LICENSE)
