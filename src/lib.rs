#![no_std]
#![allow(clippy::missing_safety_doc)]

use core::f32::consts::PI;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
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

fn bit_reverse_permute(data: &mut [Complex]) {
    let n = data.len();
    let bits = log2_pow2(n);
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (32 - bits);
        let j = j as usize;
        if j > i {
            data.swap(i, j);
        }
    }
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

    bit_reverse_permute(data);

    let sign = if inverse { 1.0f32 } else { -1.0f32 };

    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let theta = sign * 2.0 * PI / size as f32;
        let w_step = Complex::new(libm::cosf(theta), libm::sinf(theta));

        for start in (0..n).step_by(size) {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let i = start + k;
                let j = i + half;
                let t = w * data[j];
                data[j] = data[i] - t;
                data[i] = data[i] + t;
                w = w * w_step;
            }
        }
        size <<= 1;
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

fn transpose(src: &[Complex], dst: &mut [Complex], cols: usize, rows: usize) {
    for r in 0..rows {
        let off = r * cols;
        for c in 0..cols {
            dst[c * rows + r] = src[off + c];
        }
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

pub fn fft_2d(buf: &mut [Complex], scratch: &mut [Complex], width: usize, height: usize) -> Result<(), FftError> {
    fft_2d_in_place(buf, scratch, width, height, false)
}

pub fn ifft_2d(buf: &mut [Complex], scratch: &mut [Complex], width: usize, height: usize) -> Result<(), FftError> {
    fft_2d_in_place(buf, scratch, width, height, true)
}

#[no_mangle]
pub unsafe extern "C" fn fft_forward_2d(
    buf: *mut f32,
    scratch: *mut f32,
    width: usize,
    height: usize,
) -> u32 {
    run_fft_2d(buf, scratch, width, height, false)
}

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
/// This is unsafe because it returns a raw pointer to a static mutable buffer.
/// The caller must ensure that the returned pointer is not used after `fft_reset` is called.
#[no_mangle]
pub unsafe extern "C" fn fft_alloc(bytes: usize) -> *mut u8 {
    let align = 8usize;
    let off = (ARENA_OFFSET + (align - 1)) & !(align - 1);
    let end = off.checked_add(bytes).unwrap_or(usize::MAX);
    if end > ARENA_BYTES {
        return core::ptr::null_mut();
    }
    ARENA_OFFSET = end;
    let base = &raw mut ARENA as *mut u8;
    base.add(off)
}

/// Resets the arena allocator's offset to 0.
///
/// # Safety
/// This is unsafe because it invalidates all previously allocated pointers from `fft_alloc`.
#[no_mangle]
pub unsafe extern "C" fn fft_reset() {
    ARENA_OFFSET = 0;
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
}
