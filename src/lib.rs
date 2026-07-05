#![no_std]
#![allow(clippy::missing_safety_doc)]

//! Tiny `no_std` FFT compiled to WebAssembly.
//!
//! Iterative Cooley–Tukey, **radix-4** with a single **radix-2** stage when
//! `log2(N)` is odd. Input length must be a power of two.
//!
//! **Precision:** all arithmetic is single-precision `f32`, chosen deliberately
//! for small binary size and 4-lane wasm SIMD. There is no `f64` variant.
//! Round-trip error (`ifft(fft(x)) ≈ x`) is typically within ~1e-4..1e-3
//! relative. For double-precision / high-dynamic-range work, use an `f64` FFT.
//!
//! **Normalization:** the forward transform is unnormalized; the inverse divides
//! by `1/N` (applied per pass, so 2D inverse is `1/(W·H)`).
//!
//! The engine works on a **planar (split real/imag) layout** internally: the
//! interleaved `[re, im]` buffer is de-interleaved, transformed, and
//! re-interleaved. On `wasm32` with `simd128` the radix-4 butterfly processes 4
//! consecutive butterflies per step with contiguous `v128` loads/stores and pure
//! vertical `f32x4` complex arithmetic; a scalar path covers the remainder and
//! host (test) builds.
//!
//! **Plans:** the `fft_plan_*` FFI builds an arena-backed [`Plan`] that caches
//! the all-stage twiddle table and the digit-reversal map **once**; repeated
//! `fft_plan_forward`/`fft_plan_inverse` calls then do no `cos/sin` and no
//! permutation computation (the reversal is fused into the de-interleave gather).
//! The one-shot `fft`/`ifft` rlib API and the 2D path build these tables per
//! call instead.

use core::f32::consts::PI;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl core::ops::Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
}

impl core::ops::Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

impl core::ops::Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

impl Complex {
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub const ZERO: Self = Self::new(0.0, 0.0);
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(u32)]
pub enum FftError {
    NotPowerOfTwo = 1,
    EmptyInput = 2,
}

#[inline]
fn is_pow2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

#[inline]
fn log2_pow2(mut n: usize) -> u32 {
    let mut k = 0u32;
    while n > 1 {
        n >>= 1;
        k += 1;
    }
    k
}

/// Reverse the digits of `i` interpreting it as a `bits`-bit number made of
/// base-4 digits, with an optional leading single (base-2) bit when `bits` is
/// odd. This is the permutation that makes the iterative radix-4 stages (plus a
/// single leading radix-2 stage for odd `bits`) produce natural output order.
#[inline]
fn digit_reverse_index(i: usize, bits: u32) -> usize {
    let mut x = i;
    let mut rev = 0usize;
    let mut remaining = bits;
    // When `bits` is odd the radix-2 stage runs first (finest DIT split) and
    // consumes the *least-significant* input bit. Peel it off; it becomes the
    // *most-significant* bit of the reversed index.
    let mut radix2_bit = 0usize;
    let has_radix2 = remaining & 1 == 1;
    if has_radix2 {
        radix2_bit = x & 0b1;
        x >>= 1;
        remaining -= 1;
    }
    // Reverse the base-4 digits (2 bits each), low end of `x` to the high end
    // of the base-4 portion of `rev`.
    while remaining >= 2 {
        rev = (rev << 2) | (x & 0b11);
        x >>= 2;
        remaining -= 2;
    }
    // Place the radix-2 bit as the most-significant bit.
    if has_radix2 {
        rev |= radix2_bit << (bits - 1);
    }
    rev
}

/// One radix-2 stage of size 2 on planar arrays (twiddle = 1: add/sub of pairs).
#[inline]
fn radix2_stage_size2_planar(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let mut i = 0;
    while i + 1 < n {
        let ar = re[i];
        let ai = im[i];
        let br = re[i + 1];
        let bi = im[i + 1];
        re[i] = ar + br;
        im[i] = ai + bi;
        re[i + 1] = ar - br;
        im[i + 1] = ai - bi;
        i += 2;
    }
}

/// Fill a per-stage twiddle table (planar) for radix-4 stage of length `size`,
/// using **forward** sign (`W = exp(-2πi/size)`). The inverse transform reuses
/// the same table with the imaginary parts negated at load time (conjugate).
///
/// For each `k in 0..quarter` writes `w1=W^k, w2=W^{2k}, w3=W^{3k}`. Layout:
/// `w1re[..], w1im[..], w2re[..], ...` — six contiguous `quarter`-length runs so
/// the SIMD path can `v128_load` 4 twiddles at once. Uses direct `cos/sin` (no
/// recurrence) to avoid drift.
fn fill_twiddles(tw: &mut [f32], quarter: usize, size: usize) {
    let base = -2.0 * PI / size as f32;
    let (w1re, rest) = tw.split_at_mut(quarter);
    let (w1im, rest) = rest.split_at_mut(quarter);
    let (w2re, rest) = rest.split_at_mut(quarter);
    let (w2im, rest) = rest.split_at_mut(quarter);
    let (w3re, w3im) = rest.split_at_mut(quarter);
    for k in 0..quarter {
        let a1 = base * k as f32;
        let a2 = a1 + a1;
        let a3 = a2 + a1;
        w1re[k] = libm::cosf(a1);
        w1im[k] = libm::sinf(a1);
        w2re[k] = libm::cosf(a2);
        w2im[k] = libm::sinf(a2);
        w3re[k] = libm::cosf(a3);
        w3im[k] = libm::sinf(a3);
    }
}

/// Number of radix-4 twiddle floats needed to cache **all** stages of an
/// `n`-point transform. Radix-4 stages have sizes 4,16,…(or 8,32,… after a
/// leading radix-2 stage); summing `6 * (size/4)` over them gives this total.
fn total_twiddle_len(n: usize) -> usize {
    let log2n = log2_pow2(n);
    let mut size = if log2n & 1 == 1 { 8 } else { 4 };
    let mut total = 0usize;
    while size <= n {
        total += 6 * (size / 4);
        size <<= 2;
    }
    total
}

/// Scalar radix-4 butterfly on planar arrays at indices `(i0,i1,i2,i3)` using
/// twiddles `(w1,w2,w3)` given as `(re,im)` scalars.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn radix4_bfly_planar_scalar(
    re: &mut [f32],
    im: &mut [f32],
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
    w1re: f32,
    w1im: f32,
    w2re: f32,
    w2im: f32,
    w3re: f32,
    w3im: f32,
    inverse: bool,
) {
    // Twiddles are stored with forward sign; inverse uses the conjugate.
    let s = if inverse { -1.0f32 } else { 1.0f32 };
    let w1im = s * w1im;
    let w2im = s * w2im;
    let w3im = s * w3im;

    let a0r = re[i0];
    let a0i = im[i0];
    // a_k = data[i_k] * w_k  (complex multiply)
    let a1r = re[i1] * w1re - im[i1] * w1im;
    let a1i = re[i1] * w1im + im[i1] * w1re;
    let a2r = re[i2] * w2re - im[i2] * w2im;
    let a2i = re[i2] * w2im + im[i2] * w2re;
    let a3r = re[i3] * w3re - im[i3] * w3im;
    let a3i = re[i3] * w3im + im[i3] * w3re;

    let t0r = a0r + a2r;
    let t0i = a0i + a2i;
    let t1r = a0r - a2r;
    let t1i = a0i - a2i;
    let t2r = a1r + a3r;
    let t2i = a1i + a3i;
    let t3r = a1r - a3r;
    let t3i = a1i - a3i;

    // t3 * (∓j): forward -j -> (im, -re); inverse +j -> (-im, re)
    let (jr, ji) = if inverse { (-t3i, t3r) } else { (t3i, -t3r) };

    re[i0] = t0r + t2r;
    im[i0] = t0i + t2i;
    re[i1] = t1r + jr;
    im[i1] = t1i + ji;
    re[i2] = t0r - t2r;
    im[i2] = t0i - t2i;
    re[i3] = t1r - jr;
    im[i3] = t1i - ji;
}

/// SIMD radix-4 butterfly: process **4 consecutive `k`** (butterfly indices)
/// within one group, i.e. contiguous lanes `i0+0..i0+3` etc. Twiddles are loaded
/// as vectors from the per-stage table. All loads/stores are contiguous
/// `v128_load`/`v128_store`; the complex multiply is pure vertical arithmetic
/// with no shuffles (planar layout).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn radix4_bfly_planar_simd4(
    re: &mut [f32],
    im: &mut [f32],
    i0: usize,
    quarter: usize,
    tw: &[f32],
    k: usize,
    inverse: bool,
) {
    use core::arch::wasm32::*;

    let i1 = i0 + quarter;
    let i2 = i1 + quarter;
    let i3 = i2 + quarter;

    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();

    #[inline(always)]
    unsafe fn ld(p: *const f32, i: usize) -> v128 {
        v128_load(p.add(i) as *const v128)
    }
    #[inline(always)]
    unsafe fn st(p: *mut f32, i: usize, v: v128) {
        v128_store(p.add(i) as *mut v128, v);
    }
    // complex multiply of (xr,xi) by (wr,wi), all vectors
    #[inline(always)]
    fn cmul(xr: v128, xi: v128, wr: v128, wi: v128) -> (v128, v128) {
        (
            f32x4_sub(f32x4_mul(xr, wr), f32x4_mul(xi, wi)),
            f32x4_add(f32x4_mul(xr, wi), f32x4_mul(xi, wr)),
        )
    }

    // twiddle table offsets: [w1re, w1im, w2re, w2im, w3re, w3im]. Table stores
    // forward twiddles; inverse uses the conjugate (negate imaginary parts).
    let twp = tw.as_ptr();
    let w1r = ld(twp, k);
    let w2r = ld(twp, 2 * quarter + k);
    let w3r = ld(twp, 4 * quarter + k);
    let mut w1i = ld(twp, quarter + k);
    let mut w2i = ld(twp, 3 * quarter + k);
    let mut w3i = ld(twp, 5 * quarter + k);
    if inverse {
        w1i = f32x4_neg(w1i);
        w2i = f32x4_neg(w2i);
        w3i = f32x4_neg(w3i);
    }

    let a0r = ld(rp, i0);
    let a0i = ld(ip, i0);
    let (a1r, a1i) = cmul(ld(rp, i1), ld(ip, i1), w1r, w1i);
    let (a2r, a2i) = cmul(ld(rp, i2), ld(ip, i2), w2r, w2i);
    let (a3r, a3i) = cmul(ld(rp, i3), ld(ip, i3), w3r, w3i);

    let t0r = f32x4_add(a0r, a2r);
    let t0i = f32x4_add(a0i, a2i);
    let t1r = f32x4_sub(a0r, a2r);
    let t1i = f32x4_sub(a0i, a2i);
    let t2r = f32x4_add(a1r, a3r);
    let t2i = f32x4_add(a1i, a3i);
    let t3r = f32x4_sub(a1r, a3r);
    let t3i = f32x4_sub(a1i, a3i);

    // t3 * (∓j): forward -> (t3i, -t3r); inverse -> (-t3i, t3r)
    let (jr, ji) = if inverse {
        (f32x4_neg(t3i), t3r)
    } else {
        (t3i, f32x4_neg(t3r))
    };

    st(rp, i0, f32x4_add(t0r, t2r));
    st(ip, i0, f32x4_add(t0i, t2i));
    st(rp, i1, f32x4_add(t1r, jr));
    st(ip, i1, f32x4_add(t1i, ji));
    st(rp, i2, f32x4_sub(t0r, t2r));
    st(ip, i2, f32x4_sub(t0i, t2i));
    st(rp, i3, f32x4_sub(t1r, jr));
    st(ip, i3, f32x4_sub(t1i, ji));
}

/// Planar radix-4 (+ optional leading radix-2) FFT engine over split arrays,
/// using a **precomputed all-stage twiddle table** `tw` (see [`fill_twiddles`] /
/// [`total_twiddle_len`]). Input `re`/`im` are assumed **already permuted** into
/// digit-reversed order (fused into the deinterleave step). No per-call `cos/sin`
/// and no per-call permutation.
fn fft_planar(re: &mut [f32], im: &mut [f32], tw: &[f32], inverse: bool) {
    let n = re.len();
    let log2n = log2_pow2(n);

    let mut size;
    if log2n & 1 == 1 {
        radix2_stage_size2_planar(re, im);
        size = 8;
    } else {
        size = 4;
    }

    // Walk the concatenated per-stage twiddle blocks.
    let mut tw_off = 0usize;
    while size <= n {
        let quarter = size / 4;
        let stage_tw = &tw[tw_off..tw_off + 6 * quarter];
        tw_off += 6 * quarter;

        let mut start = 0usize;
        while start < n {
            let base0 = start;
            let mut k = 0usize;

            // SIMD: 4 consecutive k at a time (contiguous lanes).
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            {
                while k + 4 <= quarter {
                    unsafe {
                        radix4_bfly_planar_simd4(re, im, base0 + k, quarter, stage_tw, k, inverse);
                    }
                    k += 4;
                }
            }

            // Scalar remainder (and the whole loop on non-SIMD targets).
            while k < quarter {
                let i0 = base0 + k;
                radix4_bfly_planar_scalar(
                    re,
                    im,
                    i0,
                    i0 + quarter,
                    i0 + 2 * quarter,
                    i0 + 3 * quarter,
                    stage_tw[k],
                    stage_tw[quarter + k],
                    stage_tw[2 * quarter + k],
                    stage_tw[3 * quarter + k],
                    stage_tw[4 * quarter + k],
                    stage_tw[5 * quarter + k],
                    inverse,
                );
                k += 1;
            }
            start += size;
        }
        size <<= 2;
    }

    if inverse {
        let inv_n = 1.0 / n as f32;
        for x in re.iter_mut() {
            *x *= inv_n;
        }
        for x in im.iter_mut() {
            *x *= inv_n;
        }
    }
}

/// Fill the concatenated all-stage twiddle table for an `n`-point transform.
/// Blocks are stored stage by stage (sizes 4,16,… or 8,32,… after a radix-2
/// stage), each of length `6 * (size/4)`; total is [`total_twiddle_len`].
fn build_twiddle_table(tw: &mut [f32], n: usize) {
    let log2n = log2_pow2(n);
    let mut size = if log2n & 1 == 1 { 8 } else { 4 };
    let mut off = 0usize;
    while size <= n {
        let quarter = size / 4;
        let len = 6 * quarter;
        fill_twiddles(&mut tw[off..off + len], quarter, size);
        off += len;
        size <<= 2;
    }
}

/// Fill a digit-reversal permutation map: `perm[i] = digit_reverse_index(i)`.
/// Applied via gather at deinterleave time (`re[i] = data[perm[i]].re`), which
/// fuses the reordering into a pass we already do and needs no cycle-walking.
fn build_perm(perm: &mut [u32], n: usize) {
    let bits = log2_pow2(n);
    for (i, p) in perm.iter_mut().enumerate() {
        *p = digit_reverse_index(i, bits) as u32;
    }
}

/// Run the planar engine over an interleaved buffer with **precomputed** tables.
/// Fuses the digit-reversal into the deinterleave gather (`re[i]=data[perm[i]]`),
/// runs the SoA engine on `re`/`im`, then re-interleaves.
fn fft_run_planar(
    data: &mut [Complex],
    re: &mut [f32],
    im: &mut [f32],
    tw: &[f32],
    perm: &[u32],
    inverse: bool,
) {
    let n = data.len();
    for i in 0..n {
        let src = perm[i] as usize;
        re[i] = data[src].re;
        im[i] = data[src].im;
    }
    fft_planar(re, im, tw, inverse);
    for i in 0..n {
        data[i] = Complex::new(re[i], im[i]);
    }
}

/// One-shot in-place FFT over an interleaved buffer, building the twiddle and
/// permutation tables on the fly. Used by the rlib `fft`/`ifft` API and the 2D
/// path. For repeated 1D transforms of the same size, use the persistent plan
/// API (`fft_plan_*`) which caches these tables.
///
/// Scratch: `wasm32` uses the static arena (single-threaded); host builds use a
/// per-call heap `Vec` so the multi-threaded test runner never shares the arena.
fn fft_in_place(data: &mut [Complex], inverse: bool) -> Result<(), FftError> {
    let n = data.len();
    if n == 0 {
        return Err(FftError::EmptyInput);
    }
    if !is_pow2(n) {
        return Err(FftError::NotPowerOfTwo);
    }
    if n == 1 {
        return Ok(());
    }

    let tw_len = total_twiddle_len(n);

    #[cfg(target_arch = "wasm32")]
    unsafe {
        let mark = fft_mark();
        let fptr = fft_alloc((2 * n + tw_len) * 4) as *mut f32;
        let pptr = fft_alloc(n * 4) as *mut u32;
        if fptr.is_null() || pptr.is_null() {
            fft_release(mark);
            return Err(FftError::EmptyInput);
        }
        let floats = core::slice::from_raw_parts_mut(fptr, 2 * n + tw_len);
        let perm = core::slice::from_raw_parts_mut(pptr, n);
        let (re, rest) = floats.split_at_mut(n);
        let (im, tw) = rest.split_at_mut(n);
        build_twiddle_table(tw, n);
        build_perm(perm, n);
        fft_run_planar(data, re, im, tw, perm, inverse);
        fft_release(mark);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        extern crate std;
        let mut floats = std::vec![0.0f32; 2 * n + tw_len];
        let mut perm = std::vec![0u32; n];
        let (re, rest) = floats.split_at_mut(n);
        let (im, tw) = rest.split_at_mut(n);
        build_twiddle_table(tw, n);
        build_perm(&mut perm, n);
        fft_run_planar(data, re, im, tw, &perm, inverse);
    }

    Ok(())
}

/// Performs an in-place Forward Fourier Transform.
///
/// # Errors
/// Returns `FftError::EmptyInput` if data is empty.
/// Returns `FftError::NotPowerOfTwo` if data length is not a power of two.
pub fn fft(data: &mut [Complex]) -> Result<(), FftError> {
    fft_in_place(data, false)
}

/// Performs an in-place Inverse Fourier Transform.
///
/// # Errors
/// Returns `FftError::EmptyInput` if data is empty.
/// Returns `FftError::NotPowerOfTwo` if data length is not a power of two.
pub fn ifft(data: &mut [Complex]) -> Result<(), FftError> {
    fft_in_place(data, true)
}

/// # Safety
/// `ptr` must point to a valid buffer of `len` `Complex` (or `2 * len` `f32`).
#[no_mangle]
pub unsafe extern "C" fn fft_forward(ptr: *mut f32, len: usize) -> u32 {
    run_fft(ptr, len, false)
}

/// # Safety
/// `ptr` must point to a valid buffer of `len` `Complex` (or `2 * len` `f32`).
#[no_mangle]
pub unsafe extern "C" fn fft_inverse(ptr: *mut f32, len: usize) -> u32 {
    run_fft(ptr, len, true)
}

unsafe fn run_fft(ptr: *mut f32, len: usize, inverse: bool) -> u32 {
    if ptr.is_null() {
        return FftError::EmptyInput as u32;
    }
    let buf = core::slice::from_raw_parts_mut(ptr as *mut Complex, len);
    match fft_in_place(buf, inverse) {
        Ok(()) => 0,
        Err(e) => e as u32,
    }
}

// --- Persistent 1D plan -----------------------------------------------------
//
// A `Plan` caches, in the arena, everything a repeated same-size 1D transform
// needs: the interleaved data buffer, planar re/im scratch, the all-stage
// twiddle table, and the digit-reversal permutation map. The tables are built
// **once** at creation, so `forward`/`inverse` do zero `cos/sin` and zero
// permutation computation — just the fused-gather deinterleave, the SoA
// butterflies, and re-interleave. This is the hot path used by the TS `Plan1D`.

/// Handles into the arena for one cached 1D plan. All pointers are stable for
/// the plan's lifetime (arena allocations are never moved; freed only by
/// `fft_reset`/`fft_release` past the plan's mark).
#[repr(C)]
pub struct Plan {
    n: usize,
    data: *mut f32, // interleaved [re,im,...], len 2n  (the public view)
    re: *mut f32,   // planar scratch, len n
    im: *mut f32,   // planar scratch, len n
    tw: *mut f32,   // all-stage twiddle table, len total_twiddle_len(n)
    perm: *mut u32, // digit-reversal map, len n
}

/// Creates a cached 1D plan for `n` (power of two). Returns a pointer to the
/// `Plan` (an opaque handle) or null on bad `n` / out of arena. The plan and its
/// buffers are allocated from the arena; free them with `fft_reset` or by
/// pairing `fft_mark`/`fft_release` around plan creation.
///
/// # Safety
/// Single-threaded use only. The returned handle is valid until the arena is
/// reset/released past it.
#[no_mangle]
pub unsafe extern "C" fn fft_plan_create(n: usize) -> *mut Plan {
    if !is_pow2(n) || n < 2 {
        return core::ptr::null_mut();
    }
    let tw_len = total_twiddle_len(n);

    let plan_ptr = fft_alloc(core::mem::size_of::<Plan>()) as *mut Plan;
    let data = fft_alloc(n * 8) as *mut f32;
    let re = fft_alloc(n * 4) as *mut f32;
    let im = fft_alloc(n * 4) as *mut f32;
    let tw = fft_alloc(tw_len * 4) as *mut f32;
    let perm = fft_alloc(n * 4) as *mut u32;
    if plan_ptr.is_null()
        || data.is_null()
        || re.is_null()
        || im.is_null()
        || tw.is_null()
        || perm.is_null()
    {
        return core::ptr::null_mut();
    }

    build_twiddle_table(core::slice::from_raw_parts_mut(tw, tw_len), n);
    build_perm(core::slice::from_raw_parts_mut(perm, n), n);

    plan_ptr.write(Plan {
        n,
        data,
        re,
        im,
        tw,
        perm,
    });
    plan_ptr
}

/// Returns the pointer to the plan's interleaved `[re,im,...]` data buffer
/// (length `2n` f32) that callers read/write in place.
///
/// # Safety
/// `plan` must be a valid handle from [`fft_plan_create`].
#[no_mangle]
pub unsafe extern "C" fn fft_plan_data(plan: *mut Plan) -> *mut f32 {
    if plan.is_null() {
        return core::ptr::null_mut();
    }
    (*plan).data
}

/// Runs a cached forward transform on the plan's data buffer.
///
/// # Safety
/// `plan` must be a valid handle from [`fft_plan_create`].
#[no_mangle]
pub unsafe extern "C" fn fft_plan_forward(plan: *mut Plan) -> u32 {
    run_plan(plan, false)
}

/// Runs a cached inverse transform on the plan's data buffer.
///
/// # Safety
/// `plan` must be a valid handle from [`fft_plan_create`].
#[no_mangle]
pub unsafe extern "C" fn fft_plan_inverse(plan: *mut Plan) -> u32 {
    run_plan(plan, true)
}

unsafe fn run_plan(plan: *mut Plan, inverse: bool) -> u32 {
    if plan.is_null() {
        return FftError::EmptyInput as u32;
    }
    let p = &*plan;
    let n = p.n;
    let data = core::slice::from_raw_parts_mut(p.data as *mut Complex, n);
    let re = core::slice::from_raw_parts_mut(p.re, n);
    let im = core::slice::from_raw_parts_mut(p.im, n);
    let tw = core::slice::from_raw_parts(p.tw, total_twiddle_len(n));
    let perm = core::slice::from_raw_parts(p.perm, n);
    fft_run_planar(data, re, im, tw, perm, inverse);
    0
}

/// Cache-friendly (blocked) out-of-place transpose. `src` is `rows x cols`
/// row-major; `dst` becomes `cols x rows` row-major.
fn transpose(src: &[Complex], dst: &mut [Complex], cols: usize, rows: usize) {
    // 16 complex = 128 bytes per tile row, a good fit for typical cache lines.
    const TILE: usize = 16;
    let mut r0 = 0;
    while r0 < rows {
        let r_end = core::cmp::min(r0 + TILE, rows);
        let mut c0 = 0;
        while c0 < cols {
            let c_end = core::cmp::min(c0 + TILE, cols);
            for r in r0..r_end {
                let off = r * cols;
                for c in c0..c_end {
                    dst[c * rows + r] = src[off + c];
                }
            }
            c0 += TILE;
        }
        r0 += TILE;
    }
}

fn fft_2d_in_place(
    buf: &mut [Complex],
    scratch: &mut [Complex],
    width: usize,
    height: usize,
    inverse: bool,
) -> Result<(), FftError> {
    if width == 0 || height == 0 {
        return Err(FftError::EmptyInput);
    }
    let total = width.checked_mul(height).ok_or(FftError::EmptyInput)?;
    if buf.len() < total || scratch.len() < total {
        return Err(FftError::EmptyInput);
    }
    if !is_pow2(width) || !is_pow2(height) {
        return Err(FftError::NotPowerOfTwo);
    }

    for r in 0..height {
        fft_in_place(&mut buf[r * width..(r + 1) * width], inverse)?;
    }

    transpose(&buf[..total], &mut scratch[..total], width, height);

    for r in 0..width {
        fft_in_place(&mut scratch[r * height..(r + 1) * height], inverse)?;
    }

    transpose(&scratch[..total], &mut buf[..total], height, width);

    Ok(())
}

pub fn fft_2d(
    buf: &mut [Complex],
    scratch: &mut [Complex],
    width: usize,
    height: usize,
) -> Result<(), FftError> {
    fft_2d_in_place(buf, scratch, width, height, false)
}

pub fn ifft_2d(
    buf: &mut [Complex],
    scratch: &mut [Complex],
    width: usize,
    height: usize,
) -> Result<(), FftError> {
    fft_2d_in_place(buf, scratch, width, height, true)
}

/// # Safety
/// `buf` and `scratch` must each point to valid buffers of at least
/// `width * height` `Complex`, and **must not overlap**. Passing overlapping
/// (or identical) pointers produces incorrect results.
#[no_mangle]
pub unsafe extern "C" fn fft_forward_2d(
    buf: *mut f32,
    scratch: *mut f32,
    width: usize,
    height: usize,
) -> u32 {
    run_fft_2d(buf, scratch, width, height, false)
}

/// # Safety
/// See [`fft_forward_2d`]. `buf` and `scratch` must not overlap.
#[no_mangle]
pub unsafe extern "C" fn fft_inverse_2d(
    buf: *mut f32,
    scratch: *mut f32,
    width: usize,
    height: usize,
) -> u32 {
    run_fft_2d(buf, scratch, width, height, true)
}

unsafe fn run_fft_2d(
    buf: *mut f32,
    scratch: *mut f32,
    width: usize,
    height: usize,
    inverse: bool,
) -> u32 {
    if buf.is_null() || scratch.is_null() {
        return FftError::EmptyInput as u32;
    }
    let total = match width.checked_mul(height) {
        Some(v) => v,
        None => return FftError::EmptyInput as u32,
    };
    let bs = core::slice::from_raw_parts_mut(buf as *mut Complex, total);
    let ss = core::slice::from_raw_parts_mut(scratch as *mut Complex, total);
    match fft_2d_in_place(bs, ss, width, height, inverse) {
        Ok(()) => 0,
        Err(e) => e as u32,
    }
}

const ARENA_BYTES: usize = 8 << 20;

#[repr(C, align(8))]
struct Arena([u8; ARENA_BYTES]);

static mut ARENA: Arena = Arena([0; ARENA_BYTES]);
static mut ARENA_OFFSET: usize = 0;

/// Allocates `bytes` from a static arena. Returns a null pointer if out of space.
///
/// # Safety
/// Single-threaded use only (wasm without threads). Returns a raw pointer into a
/// static mutable buffer; the caller must not use the pointer after `fft_reset`.
#[no_mangle]
pub unsafe extern "C" fn fft_alloc(bytes: usize) -> *mut u8 {
    let align = 8usize;
    let base = &raw mut ARENA as *mut u8;
    let offset = &raw mut ARENA_OFFSET;
    let cur = offset.read();
    let off = (cur + (align - 1)) & !(align - 1);
    let end = off.saturating_add(bytes);
    if end > ARENA_BYTES {
        return core::ptr::null_mut();
    }
    offset.write(end);
    base.add(off)
}

/// Resets the arena allocator's offset to 0.
///
/// # Safety
/// Single-threaded use only. Invalidates all previously allocated pointers.
#[no_mangle]
pub unsafe extern "C" fn fft_reset() {
    let offset = &raw mut ARENA_OFFSET;
    offset.write(0);
}

/// Returns the current arena offset (a "mark"). Pair with [`fft_release`] to
/// free everything allocated since the mark, LIFO/stack-style.
///
/// # Safety
/// Single-threaded use only.
#[no_mangle]
pub unsafe extern "C" fn fft_mark() -> usize {
    let offset = &raw mut ARENA_OFFSET;
    offset.read()
}

/// Rewinds the arena to a previous `mark` from [`fft_mark`], invalidating every
/// allocation made after that mark. Marks larger than the current offset are
/// ignored (no-op) to avoid growing the used region.
///
/// # Safety
/// Single-threaded use only. Invalidates pointers allocated after `mark`.
#[no_mangle]
pub unsafe extern "C" fn fft_release(mark: usize) {
    let offset = &raw mut ARENA_OFFSET;
    let cur = offset.read();
    if mark <= cur {
        offset.write(mark);
    }
}

#[no_mangle]
pub unsafe extern "C" fn fft_arena_capacity() -> usize {
    ARENA_BYTES
}

// Tests (only built for host targets, not wasm32)
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;
    use std::vec::Vec;

    fn approx_eq(a: Complex, b: Complex, eps: f32) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    /// Reference O(n^2) DFT for correctness checks.
    fn dft_naive(input: &[Complex], inverse: bool) -> Vec<Complex> {
        let n = input.len();
        let sign = if inverse { 1.0f32 } else { -1.0f32 };
        let mut out = vec![Complex::ZERO; n];
        for (k, o) in out.iter_mut().enumerate() {
            let mut acc = Complex::ZERO;
            for (t, x) in input.iter().enumerate() {
                let ang = sign * 2.0 * PI * (k as f32) * (t as f32) / n as f32;
                let w = Complex::new(libm::cosf(ang), libm::sinf(ang));
                acc = acc + *x * w;
            }
            if inverse {
                acc.re /= n as f32;
                acc.im /= n as f32;
            }
            *o = acc;
        }
        out
    }

    #[test]
    fn rejects_non_power_of_two() {
        let mut data = vec![Complex::ZERO; 3];
        assert_eq!(fft(&mut data), Err(FftError::NotPowerOfTwo));
    }

    #[test]
    fn rejects_empty() {
        let mut data: Vec<Complex> = Vec::new();
        assert_eq!(fft(&mut data), Err(FftError::EmptyInput));
    }

    #[test]
    fn dc_signal() {
        let mut data = vec![Complex::new(1.0, 0.0); 8];
        fft(&mut data).unwrap();
        assert!(approx_eq(data[0], Complex::new(8.0, 0.0), 1e-5));
        for c in &data[1..] {
            assert!(approx_eq(*c, Complex::ZERO, 1e-5));
        }
    }

    #[test]
    fn impulse_signal() {
        let mut data = vec![Complex::ZERO; 8];
        data[0] = Complex::new(1.0, 0.0);
        fft(&mut data).unwrap();
        for c in &data {
            assert!(approx_eq(*c, Complex::new(1.0, 0.0), 1e-5));
        }
    }

    #[test]
    fn round_trip() {
        let original: Vec<Complex> = (0..16)
            .map(|i| Complex::new(i as f32, (i as f32) * 0.5))
            .collect();
        let mut data = original.clone();
        fft(&mut data).unwrap();
        ifft(&mut data).unwrap();
        for (a, b) in data.iter().zip(original.iter()) {
            assert!(approx_eq(*a, *b, 1e-4), "mismatch: {:?} vs {:?}", a, b);
        }
    }

    #[test]
    fn single_sample() {
        let mut data = vec![Complex::new(2.5, -1.0)];
        fft(&mut data).unwrap();
        assert_eq!(data[0], Complex::new(2.5, -1.0));
    }

    #[test]
    fn matches_naive_dft_all_sizes() {
        // Cover both power-of-four (4, 16, 64) and non-power-of-four (2, 8, 32) sizes
        // to exercise the radix-2 cleanup stage.
        for &n in &[2usize, 4, 8, 16, 32, 64, 128] {
            let input: Vec<Complex> = (0..n)
                .map(|i| {
                    let f = i as f32;
                    Complex::new(
                        libm::sinf(f * 0.3) + 0.5 * f,
                        libm::cosf(f * 0.7) - 0.25 * f,
                    )
                })
                .collect();
            let reference = dft_naive(&input, false);
            let mut got = input.clone();
            fft(&mut got).unwrap();
            for (k, (a, b)) in got.iter().zip(reference.iter()).enumerate() {
                let eps = 1e-2 * (n as f32);
                assert!(
                    approx_eq(*a, *b, eps),
                    "n={} k={} mismatch: {:?} vs {:?}",
                    n,
                    k,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn round_trip_all_sizes() {
        for &n in &[2usize, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let original: Vec<Complex> = (0..n)
                .map(|i| Complex::new(libm::sinf(i as f32 * 0.11), libm::cosf(i as f32 * 0.07)))
                .collect();
            let mut data = original.clone();
            fft(&mut data).unwrap();
            ifft(&mut data).unwrap();
            for (a, b) in data.iter().zip(original.iter()) {
                assert!(
                    approx_eq(*a, *b, 1e-3),
                    "n={} mismatch: {:?} vs {:?}",
                    n,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn large_n_round_trip() {
        let n = 1 << 16;
        let original: Vec<Complex> = (0..n)
            .map(|i| Complex::new(libm::sinf(i as f32 * 0.001), 0.0))
            .collect();
        let mut data = original.clone();
        fft(&mut data).unwrap();
        ifft(&mut data).unwrap();
        let mut max_err = 0.0f32;
        for (a, b) in data.iter().zip(original.iter()) {
            max_err = max_err.max((a.re - b.re).abs()).max((a.im - b.im).abs());
        }
        assert!(max_err < 1e-3, "large-N round-trip err {}", max_err);
    }

    #[test]
    fn fft_2d_dc_signal() {
        let w = 4;
        let h = 4;
        let mut buf = vec![Complex::new(1.0, 0.0); w * h];
        let mut scratch = vec![Complex::ZERO; w * h];
        fft_2d(&mut buf, &mut scratch, w, h).unwrap();
        assert!(approx_eq(buf[0], Complex::new((w * h) as f32, 0.0), 1e-4));
        for c in &buf[1..] {
            assert!(approx_eq(*c, Complex::ZERO, 1e-4));
        }
    }

    #[test]
    fn fft_2d_round_trip_non_square() {
        let w = 8;
        let h = 4;
        let original: Vec<Complex> = (0..(w * h))
            .map(|i| Complex::new(i as f32 * 0.1, (i as f32) * -0.05))
            .collect();
        let mut buf = original.clone();
        let mut scratch = vec![Complex::ZERO; w * h];
        fft_2d(&mut buf, &mut scratch, w, h).unwrap();
        ifft_2d(&mut buf, &mut scratch, w, h).unwrap();
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!(approx_eq(*a, *b, 1e-3), "mismatch: {:?} vs {:?}", a, b);
        }
    }

    #[test]
    fn fft_2d_rejects_non_power_of_two() {
        let mut buf = vec![Complex::ZERO; 12];
        let mut scratch = vec![Complex::ZERO; 12];
        assert_eq!(
            fft_2d(&mut buf, &mut scratch, 3, 4),
            Err(FftError::NotPowerOfTwo)
        );
    }

    #[test]
    fn fft_2d_impulse() {
        let w = 4;
        let h = 4;
        let mut buf = vec![Complex::ZERO; w * h];
        buf[0] = Complex::new(1.0, 0.0);
        let mut scratch = vec![Complex::ZERO; w * h];
        fft_2d(&mut buf, &mut scratch, w, h).unwrap();
        for c in &buf {
            assert!(approx_eq(*c, Complex::new(1.0, 0.0), 1e-4));
        }
    }

    /// Larger non-square 2D transform to exercise the blocked transpose across
    /// multiple tiles (dimensions > TILE=16).
    #[test]
    fn fft_2d_round_trip_large_non_square() {
        let w = 64;
        let h = 32;
        let original: Vec<Complex> = (0..(w * h))
            .map(|i| Complex::new(libm::sinf(i as f32 * 0.01), libm::cosf(i as f32 * 0.013)))
            .collect();
        let mut buf = original.clone();
        let mut scratch = vec![Complex::ZERO; w * h];
        fft_2d(&mut buf, &mut scratch, w, h).unwrap();
        ifft_2d(&mut buf, &mut scratch, w, h).unwrap();
        let mut max_err = 0.0f32;
        for (a, b) in buf.iter().zip(original.iter()) {
            max_err = max_err.max((a.re - b.re).abs()).max((a.im - b.im).abs());
        }
        assert!(max_err < 1e-3, "2D large round-trip err {}", max_err);
    }

    /// Property-style: random inputs, many sizes, `ifft(fft(x)) ~= x`.
    #[test]
    fn random_round_trip_property() {
        // Small xorshift PRNG (no deps).
        let mut state = 0x9e3779b9u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for log2 in 1u32..=12 {
            let n = 1usize << log2;
            let original: Vec<Complex> = (0..n).map(|_| Complex::new(next(), next())).collect();
            let mut data = original.clone();
            fft(&mut data).unwrap();
            ifft(&mut data).unwrap();
            let mut max_err = 0.0f32;
            for (a, b) in data.iter().zip(original.iter()) {
                max_err = max_err.max((a.re - b.re).abs()).max((a.im - b.im).abs());
            }
            assert!(max_err < 1e-3, "n={} random round-trip err {}", n, max_err);
        }
    }

    /// Compare against the naive DFT for a random signal at a mixed-radix size.
    #[test]
    fn random_matches_naive_dft() {
        let mut state = 0x12345678u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for &n in &[8usize, 32, 128] {
            let input: Vec<Complex> = (0..n).map(|_| Complex::new(next(), next())).collect();
            let reference = dft_naive(&input, false);
            let mut got = input.clone();
            fft(&mut got).unwrap();
            for (k, (a, b)) in got.iter().zip(reference.iter()).enumerate() {
                let eps = 1e-3 * (n as f32);
                assert!(
                    approx_eq(*a, *b, eps),
                    "n={} k={} mismatch: {:?} vs {:?}",
                    n,
                    k,
                    a,
                    b
                );
            }
        }
    }
}
