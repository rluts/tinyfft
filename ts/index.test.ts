import { describe, it, expect, beforeAll } from "vitest";
import {
  TinyFft,
  FftError,
  interleave,
  interleaveInto,
  magnitudes,
  magnitudesInto,
} from "../dist/index.js";

// Naive DFT reference (f64) for correctness checks.
function naiveDft(re: number[], im: number[], inverse: boolean) {
  const n = re.length;
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  const sign = inverse ? 1 : -1;
  for (let k = 0; k < n; k++) {
    let sr = 0;
    let si = 0;
    for (let t = 0; t < n; t++) {
      const ang = (sign * 2 * Math.PI * k * t) / n;
      const c = Math.cos(ang);
      const s = Math.sin(ang);
      sr += re[t]! * c - im[t]! * s;
      si += re[t]! * s + im[t]! * c;
    }
    if (inverse) {
      sr /= n;
      si /= n;
    }
    outRe[k] = sr;
    outIm[k] = si;
  }
  return { outRe, outIm };
}

let fft: TinyFft;

beforeAll(async () => {
  fft = await TinyFft.load();
});

describe("helpers", () => {
  it("interleave real-only", () => {
    const out = interleave(Float32Array.from([1, 2, 3]));
    expect(Array.from(out)).toEqual([1, 0, 2, 0, 3, 0]);
  });

  it("interleave real + imag", () => {
    const out = interleave(Float32Array.from([1, 2]), Float32Array.from([9, 8]));
    expect(Array.from(out)).toEqual([1, 9, 2, 8]);
  });

  it("interleave mismatched lengths throws", () => {
    expect(() => interleave(Float32Array.from([1, 2]), Float32Array.from([1]))).toThrow();
  });

  it("interleaveInto writes into preallocated buffer, zeroes imag", () => {
    const out = new Float32Array(6).fill(7);
    const r = interleaveInto(out, Float32Array.from([1, 2, 3]));
    expect(r).toBe(out);
    expect(Array.from(out)).toEqual([1, 0, 2, 0, 3, 0]);
  });

  it("interleaveInto too-small buffer throws", () => {
    expect(() => interleaveInto(new Float32Array(2), Float32Array.from([1, 2]))).toThrow();
  });

  it("magnitudes computes hypot", () => {
    const m = magnitudes(Float32Array.from([3, 4, 0, 0, 1, 0]));
    expect(m[0]).toBeCloseTo(5, 6);
    expect(m[1]).toBeCloseTo(0, 6);
    expect(m[2]).toBeCloseTo(1, 6);
  });

  it("magnitudesInto too-small buffer throws", () => {
    expect(() => magnitudesInto(new Float32Array(1), Float32Array.from([3, 4, 0, 0]))).toThrow();
  });
});

describe("1D transform", () => {
  it("DC bin equals sum for a real ramp", () => {
    const real = Float32Array.from({ length: 8 }, (_, i) => i + 1);
    const out = fft.forward(interleave(real));
    expect(out[0]).toBeCloseTo(36, 3);
  });

  it("matches naive DFT (mixed-radix size 8)", () => {
    const n = 8;
    const re = Array.from({ length: n }, (_, i) => Math.sin(i * 0.3));
    const im = Array.from({ length: n }, (_, i) => Math.cos(i * 0.7));
    const inter = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      inter[2 * i] = re[i]!;
      inter[2 * i + 1] = im[i]!;
    }
    const out = fft.forward(inter);
    const { outRe, outIm } = naiveDft(re, im, false);
    for (let k = 0; k < n; k++) {
      expect(out[2 * k]).toBeCloseTo(outRe[k]!, 3);
      expect(out[2 * k + 1]).toBeCloseTo(outIm[k]!, 3);
    }
  });

  it("round-trips across sizes", () => {
    for (const n of [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]) {
      const orig = new Float32Array(n * 2);
      for (let i = 0; i < n * 2; i++) orig[i] = Math.sin(i * 0.13);
      fft.reset();
      const plan = fft.plan1d(n);
      plan.view.set(orig);
      plan.forward();
      plan.inverse();
      let maxErr = 0;
      for (let i = 0; i < n * 2; i++) {
        maxErr = Math.max(maxErr, Math.abs(plan.view[i]! - orig[i]!));
      }
      expect(maxErr).toBeLessThan(1e-3);
    }
  });

  it("one-shot forward equals plan-based forward", () => {
    const n = 64;
    const inter = new Float32Array(n * 2);
    for (let i = 0; i < n * 2; i++) inter[i] = Math.cos(i * 0.05);
    const oneShot = fft.forward(inter);

    fft.reset();
    const plan = fft.plan1d(n);
    plan.view.set(inter);
    plan.forward();
    for (let i = 0; i < n * 2; i++) {
      expect(oneShot[i]).toBeCloseTo(plan.view[i]!, 5);
    }
  });

  it("one-shot rejects odd-length buffer", () => {
    expect(() => fft.forward(new Float32Array(5))).toThrow();
  });

  it("plan1d rejects non-power-of-two", () => {
    expect(() => fft.plan1d(3)).toThrow(/power of two/);
  });
});

describe("2D transform", () => {
  it("DC bin equals sum for all-ones 4x4", () => {
    const inter = new Float32Array(4 * 4 * 2);
    for (let i = 0; i < 16; i++) inter[2 * i] = 1;
    const out = fft.forward2d(inter, 4, 4);
    expect(out[0]).toBeCloseTo(16, 3);
  });

  it("round-trips non-square", () => {
    const W = 8;
    const H = 4;
    const orig = new Float32Array(W * H * 2);
    for (let i = 0; i < orig.length; i++) orig[i] = Math.sin(i * 0.1);
    fft.reset();
    const plan = fft.plan2d(W, H);
    plan.view.set(orig);
    plan.forward();
    plan.inverse();
    let maxErr = 0;
    for (let i = 0; i < orig.length; i++) {
      maxErr = Math.max(maxErr, Math.abs(plan.view[i]! - orig[i]!));
    }
    expect(maxErr).toBeLessThan(1e-3);
  });

  it("plan2d rejects non-power-of-two dims", () => {
    expect(() => fft.plan2d(3, 4)).toThrow(/power of two/);
    expect(() => fft.plan2d(4, 6)).toThrow(/power of two/);
  });
});

describe("arena / lifetime", () => {
  it("mark + release frees allocations LIFO", () => {
    fft.reset();
    const m = fft.mark();
    // Allocate a chunk, then release back to the mark; a same-size plan should
    // reuse the same region (no arena growth / exhaustion).
    fft.plan1d(1024);
    fft.release(m);
    const before = fft.mark();
    fft.plan1d(1024);
    fft.release(m);
    const after = fft.mark();
    expect(before).toBe(m);
    expect(after).toBe(m);
  });

  it("throws on arena exhaustion", () => {
    fft.reset();
    // Arena is 8 MiB; a single plan far larger than that must fail to allocate.
    const tooBig = 1 << 24; // 16M complex = 128 MiB
    expect(() => fft.plan1d(tooBig)).toThrow(/fft_plan_create failed/);
  });

  it("reset lets a fresh plan reuse memory", () => {
    fft.reset();
    const p1 = fft.plan1d(256);
    p1.view[0] = 1;
    fft.reset();
    const p2 = fft.plan1d(256);
    // After reset the new plan starts at the arena base again.
    expect(p2.view.length).toBe(512);
  });
});

describe("FftError", () => {
  it("has a numeric code and name", () => {
    const e = new FftError(1);
    expect(e).toBeInstanceOf(Error);
    expect(e.name).toBe("FftError");
    expect(e.code).toBe(1);
  });
});
