import { WASM_BYTES_BASE64 } from "./wasm-bytes.js";

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
  fft_arena_capacity(): number;
}

export type WasmSource = BufferSource | Response | Promise<Response>;

export class TinyFft {
  private readonly _exports: Exports;

  private constructor(exports: Exports) {
    this._exports = exports;
  }

  static async load(source?: WasmSource): Promise<TinyFft> {
    const bytes = await resolveWasmBytes(source);
    const { instance } = await WebAssembly.instantiate(bytes);
    return new TinyFft(instance.exports as unknown as Exports);
  }

  reset(): void {
    this._exports.fft_reset();
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
  readonly view: Float32Array;
  private readonly _exports: Exports;
  private readonly _ptr: number;

  /** @internal */
  constructor(exports: Exports, n: number) {
    if (n <= 0 || (n & (n - 1)) !== 0) {
      throw new Error(`n=${n} is not a power of two`);
    }
    const ptr = exports.fft_alloc(n * 8);
    if (ptr === 0) {
      throw new Error(
        `fft_alloc failed for ${n} complex samples (${n * 8} bytes, arena ${exports.fft_arena_capacity()} bytes)`,
      );
    }
    this._exports = exports;
    this.n = n;
    this._ptr = ptr;
    this.view = new Float32Array(exports.memory.buffer, ptr, n * 2);
  }

  forward(): void {
    const c = this._exports.fft_forward(this._ptr, this.n);
    if (c !== 0) throw new FftError(c);
  }

  inverse(): void {
    const c = this._exports.fft_inverse(this._ptr, this.n);
    if (c !== 0) throw new FftError(c);
  }
}

export class Plan2D {
  readonly width: number;
  readonly height: number;
  readonly view: Float32Array;
  private readonly _exports: Exports;
  private readonly _bufPtr: number;
  private readonly _scratchPtr: number;

  /** @internal */
  constructor(exports: Exports, width: number, height: number) {
    if (width <= 0 || (width & (width - 1)) !== 0) {
      throw new Error(`width=${width} is not a power of two`);
    }
    if (height <= 0 || (height & (height - 1)) !== 0) {
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
    this.view = new Float32Array(exports.memory.buffer, bufPtr, total * 2);
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
  const n = real.length;
  const out = new Float32Array(n * 2);
  if (imag) {
    if (imag.length !== n) throw new Error("real and imag must have equal length");
    for (let i = 0; i < n; i++) {
      out[2 * i] = real[i] as number;
      out[2 * i + 1] = imag[i] as number;
    }
  } else {
    for (let i = 0; i < n; i++) out[2 * i] = real[i] as number;
  }
  return out;
}

export function magnitudes(interleaved: Float32Array): Float32Array {
  const n = interleaved.length / 2;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.hypot(interleaved[2 * i] as number, interleaved[2 * i + 1] as number);
  }
  return out;
}

async function resolveWasmBytes(source: WasmSource | undefined): Promise<ArrayBuffer> {
  if (source === undefined) {
    return decodeBase64(WASM_BYTES_BASE64);
  }
  if (source instanceof Promise) {
    return (await source).arrayBuffer();
  }
  if (typeof Response !== "undefined" && source instanceof Response) {
    return source.arrayBuffer();
  }
  if (source instanceof ArrayBuffer) {
    return source;
  }
  const view = source as ArrayBufferView;
  return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength) as ArrayBuffer;
}

declare const Buffer:
  | {
      from(input: string, encoding: string): {
        buffer: ArrayBufferLike;
        byteOffset: number;
        byteLength: number;
      };
    }
  | undefined;

function decodeBase64(b64: string): ArrayBuffer {
  if (typeof atob !== "undefined") {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out.buffer;
  }
  if (typeof Buffer !== "undefined") {
    const b = Buffer.from(b64, "base64");
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  }
  throw new Error("base64 decoder unavailable in this environment");
}
