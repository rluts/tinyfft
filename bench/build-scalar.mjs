// Builds a SCALAR (no-SIMD) tinyfft.wasm variant into bench/vendor/ so the
// benchmark can quantify the SIMD speedup. The normal `npm run build` in the
// repo root produces the SIMD build (dist/tinyfft.wasm); this one disables
// simd128 via RUSTFLAGS so cargo ignores the workspace .cargo/config.toml flag.
import { execSync } from "node:child_process";
import { copyFileSync, mkdirSync, existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const vendor = resolve(here, "vendor");
mkdirSync(vendor, { recursive: true });

const scalarWasmSrc = resolve(
  root,
  "target/wasm32-unknown-unknown/release/tinyfft.wasm",
);
const scalarWasmOut = resolve(vendor, "tinyfft-scalar.wasm");

console.log("[scalar] cargo build with simd128 disabled");
// `-C target-feature=-simd128` overrides the workspace default (+simd128).
execSync("cargo build --release --target wasm32-unknown-unknown", {
  cwd: root,
  stdio: "inherit",
  env: {
    ...process.env,
    RUSTFLAGS: "-C target-feature=-simd128",
    // Ensure a clean, deterministic artifact even if a SIMD build was cached.
    CARGO_INCREMENTAL: "0",
  },
});

// The build output path is shared with the SIMD build, so touch a rebuild by
// removing the artifact first would be ideal; cargo rebuilds when RUSTFLAGS
// change, so the artifact here is the scalar one.
if (!existsSync(scalarWasmSrc)) {
  throw new Error(`scalar wasm not found at ${scalarWasmSrc}`);
}

let optimized = false;
try {
  execSync(
    `wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int "${scalarWasmSrc}" -o "${scalarWasmOut}"`,
    { cwd: root, stdio: "inherit" },
  );
  optimized = true;
} catch {
  copyFileSync(scalarWasmSrc, scalarWasmOut);
}

const size = readFileSync(scalarWasmOut).length;
console.log(
  `[scalar] vendor/tinyfft-scalar.wasm ${size} bytes${optimized ? " (wasm-opt -O3)" : ""}`,
);

// IMPORTANT: leave the shared target/ in the scalar state would break the SIMD
// dist. Rebuild the SIMD artifact so `dist/tinyfft.wasm` and target/ stay in
// sync with the default (SIMD) config for other consumers.
console.log("[scalar] restoring SIMD build in target/");
execSync("cargo build --release --target wasm32-unknown-unknown", {
  cwd: root,
  stdio: "inherit",
});
