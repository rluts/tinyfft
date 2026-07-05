export const FFT_OK = 0;
export const FFT_ERR_NOT_POWER_OF_TWO = 1;
export const FFT_ERR_EMPTY_INPUT = 2;

export class FftError extends Error {
  readonly code: number;
  constructor(code: number) {
    super(
      code === FFT_ERR_NOT_POWER_OF_TWO
        ? "input length must be a power of two"
        : code === FFT_ERR_EMPTY_INPUT
          ? "input is empty or buffer is null"
          : `fft failed with code ${code}`,
    );
    this.name = "FftError";
    this.code = code;
  }
}

interface Exports {
  memory: WebAssembly.Memory;
  fft_forward(ptr: number, len: number): number;
  fft_inverse(ptr: number, len: number): number;
  fft_forward_2d(buf: number, scratch: number, w: number, h: number): number;
  fft_inverse_2d(buf: number, scratch: number, w: number, h: number): number;
  fft_alloc(bytes: number): number;
  fft_reset(): void;
  fft_mark(): number;
  fft_release(mark: number): void;
  fft_arena_capacity(): number;
  fft_plan_create(n: number): number;
  fft_plan_data(plan: number): number;
  fft_plan_forward(plan: number): number;
  fft_plan_inverse(plan: number): number;
}

function isPow2(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

export type WasmSource = BufferSource | Response | Promise<Response>;

export class TinyFft {
  private readonly _exports: Exports;

  private constructor(exports: Exports) {
    this._exports = exports;
  }

  static async load(source?: WasmSource): Promise<TinyFft> {
    const instance = await instantiate(source);
    return new TinyFft(instance.exports as unknown as Exports);
  }

  reset(): void {
    this._exports.fft_reset();
  }

  /**
   * Returns the current arena position (a "mark"). Allocate plans, then call
   * {@link release} with this mark to free everything allocated since, LIFO.
   * Cheaper and more granular than {@link reset}. Plans/views created after the
   * mark are invalidated by the matching `release`.
   */
  mark(): number {
    return this._exports.fft_mark();
  }

  /** Rewinds the arena to a previous {@link mark}, invalidating later plans. */
  release(mark: number): void {
    this._exports.fft_release(mark);
  }

  get arenaCapacity(): number {
    return this._exports.fft_arena_capacity();
  }

  plan1d(n: number): Plan1D {
    return new Plan1D(this._exports, n);
  }

  plan2d(width: number, height: number): Plan2D {
    return new Plan2D(this._exports, width, height);
  }

  forward(interleaved: Float32Array): Float32Array {
    return this.runOneShot1d(interleaved, false);
  }

  inverse(interleaved: Float32Array): Float32Array {
    return this.runOneShot1d(interleaved, true);
  }

  forward2d(interleaved: Float32Array, width: number, height: number): Float32Array {
    return this.runOneShot2d(interleaved, width, height, false);
  }

  inverse2d(interleaved: Float32Array, width: number, height: number): Float32Array {
    return this.runOneShot2d(interleaved, width, height, true);
  }

  private runOneShot1d(interleaved: Float32Array, inverse: boolean): Float32Array {
    if (interleaved.length % 2 !== 0) {
      throw new Error("buffer length must be even (interleaved [re, im] pairs)");
    }
    const n = interleaved.length / 2;
    this._exports.fft_reset();
    const plan = new Plan1D(this._exports, n);
    plan.view.set(interleaved);
    if (inverse) plan.inverse();
    else plan.forward();
    return plan.view.slice();
  }

  private runOneShot2d(
    interleaved: Float32Array,
    width: number,
    height: number,
    inverse: boolean,
  ): Float32Array {
    if (interleaved.length !== width * height * 2) {
      throw new Error(
        `buffer length ${interleaved.length} != 2*${width}*${height}`,
      );
    }
    this._exports.fft_reset();
    const plan = new Plan2D(this._exports, width, height);
    plan.view.set(interleaved);
    if (inverse) plan.inverse();
    else plan.forward();
    return plan.view.slice();
  }
}

export class Plan1D {
  readonly n: number;
  private readonly _exports: Exports;
  private readonly _plan: number;
  private readonly _ptr: number;
  private _view: Float32Array;
  private _buffer: ArrayBufferLike;

  /** @internal */
  constructor(exports: Exports, n: number) {
    if (!isPow2(n)) {
      throw new Error(`n=${n} is not a power of two`);
    }
    // A persistent plan caches the twiddle + digit-reversal tables in wasm,
    // built once here; forward/inverse then do no cos/sin or permutation work.
    const plan = exports.fft_plan_create(n);
    if (plan === 0) {
      throw new Error(
        `fft_plan_create failed for ${n} complex samples (arena ${exports.fft_arena_capacity()} bytes)`,
      );
    }
    this._exports = exports;
    this.n = n;
    this._plan = plan;
    this._ptr = exports.fft_plan_data(plan);
    this._buffer = exports.memory.buffer;
    this._view = new Float32Array(this._buffer, this._ptr, n * 2);
  }

  /**
   * Interleaved `[re, im, …]` view directly over wasm memory. Read/write in
   * place. Re-created transparently if wasm memory grew (which would detach the
   * previous view); prefer re-reading `plan.view` after any operation that could
   * grow memory rather than caching the array.
   */
  get view(): Float32Array {
    const buf = this._exports.memory.buffer;
    if (buf !== this._buffer || this._view.length === 0) {
      this._buffer = buf;
      this._view = new Float32Array(buf, this._ptr, this.n * 2);
    }
    return this._view;
  }

  forward(): void {
    const c = this._exports.fft_plan_forward(this._plan);
    if (c !== 0) throw new FftError(c);
  }

  inverse(): void {
    const c = this._exports.fft_plan_inverse(this._plan);
    if (c !== 0) throw new FftError(c);
  }
}

export class Plan2D {
  readonly width: number;
  readonly height: number;
  private readonly _exports: Exports;
  private readonly _bufPtr: number;
  private readonly _scratchPtr: number;
  private _view: Float32Array;
  private _buffer: ArrayBufferLike;

  /** @internal */
  constructor(exports: Exports, width: number, height: number) {
    if (!isPow2(width)) {
      throw new Error(`width=${width} is not a power of two`);
    }
    if (!isPow2(height)) {
      throw new Error(`height=${height} is not a power of two`);
    }
    const total = width * height;
    const bufPtr = exports.fft_alloc(total * 8);
    const scratchPtr = exports.fft_alloc(total * 8);
    if (bufPtr === 0 || scratchPtr === 0) {
      throw new Error(
        `fft_alloc failed for ${width}x${height} (need ${total * 16} bytes, arena ${exports.fft_arena_capacity()} bytes)`,
      );
    }
    this._exports = exports;
    this.width = width;
    this.height = height;
    this._bufPtr = bufPtr;
    this._scratchPtr = scratchPtr;
    this._buffer = exports.memory.buffer;
    this._view = new Float32Array(this._buffer, bufPtr, total * 2);
  }

  /** Interleaved `[re, im, …]` view over wasm memory. See {@link Plan1D.view}. */
  get view(): Float32Array {
    const buf = this._exports.memory.buffer;
    if (buf !== this._buffer || this._view.length === 0) {
      this._buffer = buf;
      this._view = new Float32Array(buf, this._bufPtr, this.width * this.height * 2);
    }
    return this._view;
  }

  forward(): void {
    const c = this._exports.fft_forward_2d(this._bufPtr, this._scratchPtr, this.width, this.height);
    if (c !== 0) throw new FftError(c);
  }

  inverse(): void {
    const c = this._exports.fft_inverse_2d(this._bufPtr, this._scratchPtr, this.width, this.height);
    if (c !== 0) throw new FftError(c);
  }
}

export function interleave(real: Float32Array, imag?: Float32Array): Float32Array {
  return interleaveInto(new Float32Array(real.length * 2), real, imag);
}

/**
 * Interleave into a preallocated buffer (no allocation). `out` must have length
 * `2 * real.length`. Returns `out`.
 */
export function interleaveInto(
  out: Float32Array,
  real: Float32Array,
  imag?: Float32Array,
): Float32Array {
  const n = real.length;
  if (out.length < n * 2) {
    throw new Error(`out length ${out.length} < 2*${n}`);
  }
  if (imag) {
    if (imag.length !== n) throw new Error("real and imag must have equal length");
    for (let i = 0; i < n; i++) {
      out[2 * i] = real[i] as number;
      out[2 * i + 1] = imag[i] as number;
    }
  } else {
    for (let i = 0; i < n; i++) {
      out[2 * i] = real[i] as number;
      out[2 * i + 1] = 0;
    }
  }
  return out;
}

export function magnitudes(interleaved: Float32Array): Float32Array {
  return magnitudesInto(new Float32Array(interleaved.length / 2), interleaved);
}

/**
 * Compute per-bin magnitudes into a preallocated buffer (no allocation). `out`
 * must have length `interleaved.length / 2`. Returns `out`.
 */
export function magnitudesInto(out: Float32Array, interleaved: Float32Array): Float32Array {
  const n = interleaved.length / 2;
  if (out.length < n) {
    throw new Error(`out length ${out.length} < ${n}`);
  }
  for (let i = 0; i < n; i++) {
    out[i] = Math.hypot(interleaved[2 * i] as number, interleaved[2 * i + 1] as number);
  }
  return out;
}

async function instantiate(source: WasmSource | undefined): Promise<WebAssembly.Instance> {
  // A Response (or a Promise of one) can be streamed directly.
  if (source instanceof Promise) {
    return instantiateResponse(source);
  }
  if (typeof Response !== "undefined" && source instanceof Response) {
    return instantiateResponse(Promise.resolve(source));
  }
  if (source !== undefined) {
    const bytes = source as BufferSource;
    return (await WebAssembly.instantiate(bytes)).instance;
  }
  // No source: load the co-located tinyfft.wasm.
  const url = new URL("./tinyfft.wasm", import.meta.url);
  if (url.protocol === "file:") {
    // Node: read the file from disk. `@ts-ignore` because we don't depend on
    // @types/node (keeps devDeps minimal); the modules exist at runtime.
    // @ts-ignore
    const fs = await import("node:fs/promises");
    // @ts-ignore
    const nodeUrl = await import("node:url");
    const bytes = await fs.readFile(nodeUrl.fileURLToPath(url));
    return (await WebAssembly.instantiate(bytes)).instance;
  }
  return instantiateResponse(Promise.resolve(fetch(url)));
}

async function instantiateResponse(resp: Promise<Response>): Promise<WebAssembly.Instance> {
  if (typeof WebAssembly.instantiateStreaming === "function") {
    try {
      return (await WebAssembly.instantiateStreaming(resp)).instance;
    } catch {
      // Fall back to buffered instantiation (e.g. wrong MIME type).
    }
  }
  const bytes = await (await resp).arrayBuffer();
  return (await WebAssembly.instantiate(bytes)).instance;
}
