import { execSync } from "node:child_process";
import { copyFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const wasmPath = resolve(root, "target/wasm32-unknown-unknown/release/tinyfft.wasm");
const distWasm = resolve(root, "dist/tinyfft.wasm");

console.log("[1/4] cargo build --release --target wasm32-unknown-unknown");
execSync("cargo build --release --target wasm32-unknown-unknown", {
  cwd: root,
  stdio: "inherit",
});

console.log("[2/4] tsc");
execSync("npx tsc", { cwd: root, stdio: "inherit" });

console.log("[3/4] optimize wasm (wasm-opt, if available)");
mkdirSync(resolve(root, "dist"), { recursive: true });
let optimized = false;
try {
  // The module uses SIMD (simd128); newer LLVM also emits bulk-memory and
  // non-trapping float-to-int. Enable them so wasm-opt's validator accepts it.
  const features = "--enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int";
  execSync(`wasm-opt -O3 ${features} "${wasmPath}" -o "${distWasm}"`, {
    cwd: root,
    stdio: "inherit",
  });
  optimized = true;
} catch {
  console.log("  wasm-opt not found; copying unoptimized wasm");
  copyFileSync(wasmPath, distWasm);
}

console.log("[4/4] done");
const size = readFileSync(distWasm).length;
console.log(`  dist/tinyfft.wasm ${size} bytes${optimized ? " (wasm-opt -O3)" : ""}`);
