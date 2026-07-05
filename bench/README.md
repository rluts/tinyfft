# tinyfft benchmarks

Reproducible microbenchmarks comparing **tinyfft** against popular JS/WASM FFT
libraries. Pure Node, no browser needed.

## Running

```bash
cd bench
npm install
npm run bench:full     # rebuild the lib + scalar wasm variant, then benchmark
# or, if the wasm variants are already built:
npm run bench
```

`bench:full` does three things:

1. `build:lib` — builds the default (SIMD) `dist/tinyfft.wasm` from source.
2. `build:scalar` — builds a **no-SIMD** variant into `bench/vendor/tinyfft-scalar.wasm`
   (via `RUSTFLAGS=-C target-feature=-simd128`) to quantify the SIMD win, then
   restores the SIMD build in `target/`.
3. `bench` — runs `run.mjs`.

## Precision

tinyfft is single-precision **`f32`** (round-trip ~1e-4–1e-3 relative; ~1e-7 max
error vs the naive DFT). fft.js is **`f64`** (~1e-14). The throughput tables are
therefore **not** a like-for-like precision comparison — see the correctness
column in the results.

## What it measures

- **Correctness** — max relative error vs a naive O(n²) DFT at N=256 (sanity check).
- **Artifact size** — bytes of the shipped wasm (tinyfft) or the JS bundle (fft.js).
- **Cold-load** — time for the first `TinyFft.load()` (wasm compile + instantiate).
  Note this is dominated by the first module compiled in the process, so treat it
  as a rough "first backend loaded" figure rather than a per-library constant.
- **Throughput** — MSamples/s for 1D `forward` and `roundTrip` across
  N ∈ {256, 1024, 4096, 65536}, with warmup and a fixed measurement budget.

## Backends

| Backend | Precision | Notes |
| ------- | --------- | ----- |
| tinyfft (SIMD) | f32 | default build, `+simd128` |
| tinyfft (scalar) | f32 | `-simd128`, isolates the SIMD contribution |
| [fft.js](https://github.com/indutny/fft.js) | f64 | mature pure-JS radix-4, the common baseline |

More backends can be added to `run.mjs` as additional entries in the
`backends` array.

## Methodology & fairness caveats

- **f32 vs f64.** tinyfft is single-precision; fft.js is double-precision. f64
  does more work per element but JS engines optimize `Float64Array`/`Array`
  math heavily. Not a like-for-like precision comparison — see the correctness
  column (fft.js error ~1e-14, tinyfft ~1e-7, as expected for f32).
- **In-place vs out-of-place.** tinyfft transforms in place over wasm memory;
  fft.js writes to a separate output array. The `forward` loop re-seeds
  tinyfft's buffer each iteration (a `2N`-float copy) because forward is
  destructive and unnormalized — overhead fft.js doesn't pay. The `roundTrip`
  loop is numerically stable and copy-free, so it's the fairer transform-only
  comparison.
- **Single-threaded, warm.** One thread; a warmup phase lets the JIT and wasm
  tier up before timing. GC pauses can still add noise — run a few times.
- **Node only.** Browser SIMD codegen differs; numbers there may vary.

## Takeaways

tinyfft's SIMD build **beats fft.js at every size** — e.g. N=1024 forward ~250 vs ~207
MSamples/s, N=65536 round-trip ~79 vs ~57 — while staying a tiny, dependency-free wasm. SIMD
is also ~1.8× its own scalar build at large N.

Two things make it fast: the **planar (split real/imag) layout** lets the radix-4 butterfly
run on contiguous `v128_load`/`v128_store` with pure vertical `f32x4` math (no shuffles), and
each `plan1d` **caches** its twiddle table and digit-reversal map, so `forward`/`inverse` do
no `cos/sin` and no permutation work.

Beyond raw speed, tinyfft ships one **~16 KB dependency-free wasm** that behaves identically
across browsers, Node, Deno, and edge runtimes, with **predictable, JIT-warmup-free**
performance and deterministic `f32`/SIMD results. Reusing a `plan` (its cached tables) is what
delivers the throughput above; the one-shot `fft.forward` rebuilds tables per call and is for
convenience, not hot loops.

## Example results

Machine/runtime-specific; regenerate locally. Sample run (Apple Silicon,
Node 25):

```
Artifact size & cold-load:
  backend                    size    cold-load
  tinyfft (SIMD)         16.0 KiB     ~20 ms (first compile)
  tinyfft (scalar)       13.5 KiB      ~0.1 ms
  fft.js (pure JS)       12.8 KiB          n/a

Throughput — forward (MSamples/s):
  backend                    N=256      N=1024      N=4096     N=65536
  tinyfft (SIMD)             ~262        ~250        ~238        ~159
  tinyfft (scalar)           ~220        ~182        ~162         ~86
  fft.js (pure JS)           ~258        ~207        ~172        ~115

Throughput — roundTrip (MSamples/s):
  backend                    N=256      N=1024      N=4096     N=65536
  tinyfft (SIMD)             ~133        ~124        ~120         ~79
  tinyfft (scalar)           ~108         ~91         ~81         ~43
  fft.js (pure JS)           ~119         ~97         ~82         ~57
```
