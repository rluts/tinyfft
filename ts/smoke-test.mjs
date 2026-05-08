import { TinyFft, interleave, magnitudes } from "../dist/index.js";

const fft = await TinyFft.load();
console.log(`arena = ${(fft.arenaCapacity / (1024 * 1024)).toFixed(1)} MiB`);

// 1D: real ramp [1..8], DC bin should equal sum = 36
{
  const real = Float32Array.from({ length: 8 }, (_, i) => i + 1);
  const out = fft.forward(interleave(real));
  if (Math.abs(out[0] - 36) > 1e-3) {
    throw new Error(`1D DC bin: expected 36, got ${out[0]}`);
  }
  console.log("1D forward DC bin:", out[0].toFixed(2));
}

// 1D round-trip
{
  const original = interleave(
    Float32Array.from({ length: 16 }, (_, i) => Math.sin(i)),
    Float32Array.from({ length: 16 }, (_, i) => Math.cos(i)),
  );
  fft.reset();
  const plan = fft.plan1d(16);
  plan.view.set(original);
  plan.forward();
  plan.inverse();
  let maxErr = 0;
  for (let i = 0; i < original.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(plan.view[i] - original[i]));
  }
  if (maxErr > 1e-4) throw new Error(`1D round-trip err ${maxErr}`);
  console.log("1D round-trip max err:", maxErr.toExponential(2));
}

// 2D DC: 4x4 ones -> bin (0,0) = 16
{
  fft.reset();
  const plan = fft.plan2d(4, 4);
  for (let i = 0; i < 16; i++) {
    plan.view[2 * i] = 1;
    plan.view[2 * i + 1] = 0;
  }
  plan.forward();
  if (Math.abs(plan.view[0] - 16) > 1e-3) {
    throw new Error(`2D DC bin: expected 16, got ${plan.view[0]}`);
  }
  console.log("2D forward DC bin:", plan.view[0].toFixed(2));
}

// 2D round-trip on a non-square buffer
{
  const W = 8;
  const H = 4;
  const orig = new Float32Array(W * H * 2);
  for (let i = 0; i < W * H; i++) {
    orig[2 * i] = i * 0.1;
    orig[2 * i + 1] = -i * 0.05;
  }
  fft.reset();
  const plan = fft.plan2d(W, H);
  plan.view.set(orig);
  plan.forward();
  plan.inverse();
  let maxErr = 0;
  for (let i = 0; i < orig.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(plan.view[i] - orig[i]));
  }
  if (maxErr > 1e-3) throw new Error(`2D round-trip err ${maxErr}`);
  console.log("2D round-trip max err:", maxErr.toExponential(2));
}

// magnitudes helper
{
  const buf = new Float32Array([3, 4, 0, 0, 1, 0]);
  const mags = magnitudes(buf);
  if (Math.abs(mags[0] - 5) > 1e-6) throw new Error("magnitudes broken");
  console.log("magnitudes ok");
}

console.log("smoke ok");
