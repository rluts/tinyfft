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
//! On `wasm32` with `simd128` enabled the butterfly loops use `core::arch::wasm32`
//! intrinsics; on other targets (host tests) an equivalent scalar path is used.

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

    /// Multiply by `-j` (i.e. rotate by -90°): `(re, im) -> (im, -re)`.
    #[inline]
    fn mul_neg_j(self) -> Self {
        Self::new(self.im, -self.re)
    }

    /// Multiply by `+j` (i.e. rotate by +90°): `(re, im) -> (-im, re)`.
    #[inline]
    fn mul_pos_j(self) -> Self {
        Self::new(-self.im, self.re)
    }
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

/// Apply the base-4 digit-reversal permutation in place.
///
/// Unlike plain bit-reversal, this permutation is **not** an involution, so a
/// naive swap loop is incorrect. We walk each permutation cycle instead, which
/// needs no scratch buffer.
fn digit_reverse_permute(data: &mut [Complex]) {
    let n = data.len();
    let bits = log2_pow2(n);
    for i in 0..n {
        let mut j = digit_reverse_index(i, bits);
        // Rotate the cycle starting at `i`, but only once per cycle: act when
        // `i` is the smallest index in its cycle.
        let mut is_min = true;
        while j != i {
            if j < i {
                is_min = false;
                break;
            }
            j = digit_reverse_index(j, bits);
        }
        if !is_min {
            continue;
        }
        // Move elements around the cycle.
        let tmp = data[i];
        let mut cur = i;
        loop {
            let next = digit_reverse_index(cur, bits);
            if next == i {
                data[cur] = tmp;
                break;
            }
            data[cur] = data[next];
            cur = next;
        }
    }
}

/// One radix-2 stage of size 2 (used first when `log2(N)` is odd).
///
/// Twiddle is always 1, so this is a pure add/sub over adjacent pairs.
#[inline]
fn radix2_stage_size2(data: &mut [Complex]) {
    let n = data.len();
    let mut i = 0;
    while i + 1 < n {
        let a = data[i];
        let b = data[i + 1];
        data[i] = a + b;
        data[i + 1] = a - b;
        i += 2;
    }
}

/// Scalar radix-4 butterfly for one group at base index `i0` (the `+k` offset is
/// already folded into `i0`). `quarter = size / 4` is the sub-transform length.
#[inline(always)]
fn radix4_butterfly_scalar(
    data: &mut [Complex],
    i0: usize,
    quarter: usize,
    w1: Complex,
    w2: Complex,
    w3: Complex,
    inverse: bool,
) {
    let i1 = i0 + quarter;
    let i2 = i1 + quarter;
    let i3 = i2 + quarter;

    let a0 = data[i0];
    let a1 = data[i1] * w1;
    let a2 = data[i2] * w2;
    let a3 = data[i3] * w3;

    let t0 = a0 + a2;
    let t1 = a0 - a2;
    let t2 = a1 + a3;
    let t3 = a1 - a3;

    let t3r = if inverse {
        t3.mul_pos_j()
    } else {
        t3.mul_neg_j()
    };

    data[i0] = t0 + t2;
    data[i1] = t1 + t3r;
    data[i2] = t0 - t2;
    data[i3] = t1 - t3r;
}

/// SIMD radix-4 butterfly processing **two** groups at once. The two groups
/// (`start` and `start + size`) share the same twiddles, so each `f32x4` lane
/// pair holds `[re, im]` for the two groups at the same position.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
#[allow(clippy::too_many_arguments)]
fn radix4_butterfly_pair_simd(
    data: &mut [Complex],
    start: usize,
    size: usize,
    quarter: usize,
    k: usize,
    w1: Complex,
    w2: Complex,
    w3: Complex,
    inverse: bool,
) {
    use core::arch::wasm32::*;

    // Indices for group A (start) and group B (start + size).
    let a_i0 = start + k;
    let a_i1 = a_i0 + quarter;
    let a_i2 = a_i1 + quarter;
    let a_i3 = a_i2 + quarter;
    let b_i0 = a_i0 + size;
    let b_i1 = a_i1 + size;
    let b_i2 = a_i2 + size;
    let b_i3 = a_i3 + size;

    // Load two complex values into one v128 as [re_a, im_a, re_b, im_b].
    #[inline(always)]
    fn load2(data: &[Complex], ia: usize, ib: usize) -> v128 {
        f32x4(data[ia].re, data[ia].im, data[ib].re, data[ib].im)
    }
    #[inline(always)]
    fn store2(data: &mut [Complex], ia: usize, ib: usize, v: v128) {
        data[ia] = Complex::new(f32x4_extract_lane::<0>(v), f32x4_extract_lane::<1>(v));
        data[ib] = Complex::new(f32x4_extract_lane::<2>(v), f32x4_extract_lane::<3>(v));
    }

    // Complex multiply of packed [re,im,re,im] by a broadcast scalar twiddle w.
    // (re + im i)(wr + wi i) = (re*wr - im*wi) + (re*wi + im*wr) i
    // With v = [re,im,re,im], swapped = [im,re,im,re]:
    //   p = v * wr_splat        = [re*wr, im*wr, ...]
    //   q = swapped * wi_splat  = [im*wi, re*wi, ...]
    //   result = p + q * [-1, +1, -1, +1]
    //          = [re*wr - im*wi, im*wr + re*wi, ...]
    #[inline(always)]
    fn cmul_broadcast(v: v128, w: Complex) -> v128 {
        let swapped = i32x4_shuffle::<1, 0, 3, 2>(v, v);
        let p = f32x4_mul(v, f32x4_splat(w.re));
        let q = f32x4_mul(swapped, f32x4_splat(w.im));
        let q_signed = f32x4_mul(q, f32x4(-1.0, 1.0, -1.0, 1.0));
        f32x4_add(p, q_signed)
    }

    let a0 = load2(data, a_i0, b_i0);
    let a1 = cmul_broadcast(load2(data, a_i1, b_i1), w1);
    let a2 = cmul_broadcast(load2(data, a_i2, b_i2), w2);
    let a3 = cmul_broadcast(load2(data, a_i3, b_i3), w3);

    let t0 = f32x4_add(a0, a2);
    let t1 = f32x4_sub(a0, a2);
    let t2 = f32x4_add(a1, a3);
    let t3 = f32x4_sub(a1, a3);

    // Multiply t3 by -j (forward) or +j (inverse), per packed complex.
    // -j*(re+im i) = im - re i  => [im, -re]
    // +j*(re+im i) = -im + re i => [-im, re]
    let t3_swap = i32x4_shuffle::<1, 0, 3, 2>(t3, t3); // [im, re, im, re]
    let t3r = if inverse {
        // [-im, re, -im, re] = t3_swap * [-1, 1, -1, 1]
        f32x4_mul(t3_swap, f32x4(-1.0, 1.0, -1.0, 1.0))
    } else {
        // [im, -re, im, -re] = t3_swap * [1, -1, 1, -1]
        f32x4_mul(t3_swap, f32x4(1.0, -1.0, 1.0, -1.0))
    };

    store2(data, a_i0, b_i0, f32x4_add(t0, t2));
    store2(data, a_i1, b_i1, f32x4_add(t1, t3r));
    store2(data, a_i2, b_i2, f32x4_sub(t0, t2));
    store2(data, a_i3, b_i3, f32x4_sub(t1, t3r));
}

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

    digit_reverse_permute(data);

    let log2n = log2_pow2(n);
    let sign = if inverse { 1.0f32 } else { -1.0f32 };

    // If log2(N) is odd, do a single radix-2 stage of size 2 first so the
    // remaining stages are all radix-4.
    //
    // - even log2(N): radix-4 stages combine sizes 4, 16, 64, ...
    // - odd  log2(N): after the size-2 stage, radix-4 stages combine sizes
    //   8, 32, 128, ... (each takes size/4-length sub-transforms).
    let mut size;
    if log2n & 1 == 1 {
        radix2_stage_size2(data);
        size = 8;
    } else {
        size = 4;
    }

    // Radix-4 stages. `size` is the transform length handled by this stage
    // (a power of four here). Each stage combines 4 sub-transforms of length
    // `quarter = size / 4`.
    //
    // Loop order is `k` (twiddle index) outer, groups inner: every group at a
    // given `k` shares the same twiddles `w1,w2,w3`, which lets the SIMD path
    // process two groups per iteration with a broadcast twiddle.
    while size <= n {
        let quarter = size / 4;
        let theta = sign * 2.0 * PI / size as f32;
        let w1_step = Complex::new(libm::cosf(theta), libm::sinf(theta));

        let mut w1 = Complex::new(1.0, 0.0);
        for k in 0..quarter {
            let w2 = w1 * w1;
            let w3 = w2 * w1;

            let mut start = 0usize;
            // SIMD: handle two groups at a time (same k => same twiddles).
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            {
                while start + size <= n && (start + size) + size <= n {
                    radix4_butterfly_pair_simd(data, start, size, quarter, k, w1, w2, w3, inverse);
                    start += 2 * size;
                }
            }
            // Scalar tail (and the whole loop on non-SIMD targets).
            while start + size <= n {
                radix4_butterfly_scalar(data, start + k, quarter, w1, w2, w3, inverse);
                start += size;
            }

            w1 = w1 * w1_step;
        }
        size <<= 2;
    }

    if inverse {
        let inv_n = 1.0 / n as f32;
        for c in data.iter_mut() {
            c.re *= inv_n;
            c.im *= inv_n;
        }
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
