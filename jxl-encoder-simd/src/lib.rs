// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated primitives for jxl_encoder.
//!
//! This crate wraps platform-specific SIMD intrinsics behind safe public functions.
//! The main encoder crate (`jxl_encoder`) maintains `#![forbid(unsafe_code)]` and
//! calls into these safe wrappers.
//!
//! Uses [archmage](https://docs.rs/archmage) for token-based SIMD dispatch
//! and [magetypes](https://docs.rs/magetypes) for cross-platform vector types.
//!
//! # Direct variant access
//!
//! Each kernel is available in three forms:
//! - A dispatching function (e.g. `dct_8x8`) that picks the best at runtime
//! - Concrete `_avx2(token, ...)` / `_neon(token, ...)` / `_scalar(...)` variants
//!
//! For hot loops, callers should summon a token once, then call the concrete
//! variant directly from an `#[arcane]` function so LLVM can inline across the
//! target-feature boundary.

#![cfg_attr(not(feature = "unsafe-performance"), forbid(unsafe_code))]
#![cfg_attr(feature = "unsafe-performance", deny(unsafe_code))]
// Numerical SIMD/DSP code: range loops and many-parameter kernels are natural.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
#![no_std]
extern crate alloc;

/// Return an uninitialized `[f32; N]` scratch buffer (unsafe-performance path).
///
/// # Safety
/// Caller must write every element before reading it. All call sites are DCT/IDCT
/// scratch arrays that are immediately filled by copy_from_slice, transpose, or
/// gather_col before any read occurs.
#[cfg(feature = "unsafe-performance")]
#[allow(unsafe_code, clippy::uninit_assumed_init)]
#[inline(always)]
pub(crate) fn scratch_buf<const N: usize>() -> [f32; N] {
    // SAFETY: All call sites write every element via copy_from_slice, transpose,
    // or gather_col before any read. f32 has no trap representations on IEEE 754.
    unsafe { core::mem::MaybeUninit::<[f32; N]>::uninit().assume_init() }
}

/// Return a zero-initialized `[f32; N]` scratch buffer (safe default path).
#[cfg(not(feature = "unsafe-performance"))]
#[inline(always)]
pub(crate) fn scratch_buf<const N: usize>() -> [f32; N] {
    [0.0f32; N]
}

/// Allocate a `Vec<f32>` of length `n` without zeroing (unsafe-performance path).
///
/// # Safety
/// Caller must write every element before reading it. Intended for output buffers
/// that are immediately overwritten by IDCT, EPF, gaborish, or similar operations.
#[cfg(feature = "unsafe-performance")]
#[allow(unsafe_code, clippy::uninit_vec)]
#[inline]
pub fn vec_f32_dirty(n: usize) -> alloc::vec::Vec<f32> {
    let mut v = alloc::vec::Vec::with_capacity(n);
    // SAFETY: f32 has no trap representations on IEEE 754. Caller must write all
    // elements before reading. Length is within the allocated capacity.
    unsafe { v.set_len(n) };
    v
}

/// Allocate a zero-initialized `Vec<f32>` of length `n` (safe default path).
#[cfg(not(feature = "unsafe-performance"))]
#[inline]
pub fn vec_f32_dirty(n: usize) -> alloc::vec::Vec<f32> {
    alloc::vec![0.0f32; n]
}

/// Slice from offset without bounds check (unsafe-performance path).
///
/// # Safety
/// Caller must ensure `offset <= s.len()`.
#[cfg(all(feature = "unsafe-performance", target_arch = "x86_64"))]
#[inline(always)]
#[allow(unsafe_code)]
pub(crate) fn slice_from(s: &[f32], offset: usize) -> &[f32] {
    debug_assert!(offset <= s.len());
    // SAFETY: Caller guarantees offset <= s.len(); debug_assert checks in debug builds.
    unsafe { s.get_unchecked(offset..) }
}

/// Slice from offset with bounds check (safe default path).
#[cfg(all(not(feature = "unsafe-performance"), target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn slice_from(s: &[f32], offset: usize) -> &[f32] {
    &s[offset..]
}

/// Load 8 floats at offset — no bounds checks (unsafe-performance path).
///
/// Bypasses both slice-from bounds check AND `f32x8::from_slice`'s internal
/// `[..8]` bounds check by using `_mm256_loadu_ps` directly.
///
/// # Safety
/// Caller must ensure `offset + 8 <= s.len()`.
#[cfg(all(feature = "unsafe-performance", target_arch = "x86_64"))]
#[inline(always)]
#[allow(unsafe_code)]
pub(crate) fn load_f32x8(
    token: archmage::X64V3Token,
    s: &[f32],
    offset: usize,
) -> magetypes::simd::f32x8 {
    use magetypes::simd::f32x8;
    debug_assert!(
        offset + 8 <= s.len(),
        "load_f32x8: offset={offset}, len={}",
        s.len()
    );
    // SAFETY: Caller guarantees offset + 8 <= s.len(); debug_assert checks in debug builds.
    unsafe {
        let ptr = s.as_ptr().add(offset);
        f32x8::from_m256(token, core::arch::x86_64::_mm256_loadu_ps(ptr))
    }
}

/// Load 8 floats at offset — with bounds checks (safe default path).
#[cfg(all(not(feature = "unsafe-performance"), target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn load_f32x8(
    token: archmage::X64V3Token,
    s: &[f32],
    offset: usize,
) -> magetypes::simd::f32x8 {
    use magetypes::simd::f32x8;
    f32x8::from_slice(token, &s[offset..])
}

/// Store 8 floats at offset — no bounds checks (unsafe-performance path).
///
/// Bypasses slice bounds check and `try_into().unwrap()` by using
/// `_mm256_storeu_ps` directly.
///
/// # Safety
/// Caller must ensure `offset + 8 <= s.len()`.
#[cfg(all(feature = "unsafe-performance", target_arch = "x86_64"))]
#[inline(always)]
#[allow(unsafe_code)]
pub(crate) fn store_f32x8(s: &mut [f32], offset: usize, v: magetypes::simd::f32x8) {
    debug_assert!(
        offset + 8 <= s.len(),
        "store_f32x8: offset={offset}, len={}",
        s.len()
    );
    // SAFETY: Caller guarantees offset + 8 <= s.len(); debug_assert checks in debug builds.
    unsafe {
        let ptr = s.as_mut_ptr().add(offset);
        core::arch::x86_64::_mm256_storeu_ps(ptr, v.raw());
    }
}

/// Store 8 floats at offset — with bounds checks (safe default path).
#[cfg(all(not(feature = "unsafe-performance"), target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn store_f32x8(s: &mut [f32], offset: usize, v: magetypes::simd::f32x8) {
    let out: &mut [f32; 8] = (&mut s[offset..offset + 8]).try_into().unwrap();
    v.store(out);
}

/// Load column `j` from 8 consecutive rows starting at `base_row` with given stride.
///
/// Unsafe-performance path: uses unchecked indexing (validated by debug_assert).
/// Safe path: uses bounds-checked indexing.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[cfg_attr(feature = "unsafe-performance", allow(unsafe_code))]
pub(crate) fn gather_col_strided(
    token: archmage::X64V3Token,
    data: &[f32],
    base_row: usize,
    j: usize,
    stride: usize,
) -> magetypes::simd::f32x8 {
    #[cfg(feature = "unsafe-performance")]
    {
        debug_assert!(
            (base_row + 7) * stride + j < data.len(),
            "gather_col_strided OOB: base_row={base_row}, j={j}, stride={stride}, len={}",
            data.len()
        );
        // SAFETY: Caller guarantees (base_row + 7) * stride + j < data.len().
        // All lower indices are within bounds since base_row + r <= base_row + 7.
        unsafe {
            let arr = [
                *data.get_unchecked(base_row * stride + j),
                *data.get_unchecked((base_row + 1) * stride + j),
                *data.get_unchecked((base_row + 2) * stride + j),
                *data.get_unchecked((base_row + 3) * stride + j),
                *data.get_unchecked((base_row + 4) * stride + j),
                *data.get_unchecked((base_row + 5) * stride + j),
                *data.get_unchecked((base_row + 6) * stride + j),
                *data.get_unchecked((base_row + 7) * stride + j),
            ];
            magetypes::simd::f32x8::from_array(token, arr)
        }
    }
    #[cfg(not(feature = "unsafe-performance"))]
    magetypes::simd::f32x8::from_array(
        token,
        [
            data[base_row * stride + j],
            data[(base_row + 1) * stride + j],
            data[(base_row + 2) * stride + j],
            data[(base_row + 3) * stride + j],
            data[(base_row + 4) * stride + j],
            data[(base_row + 5) * stride + j],
            data[(base_row + 6) * stride + j],
            data[(base_row + 7) * stride + j],
        ],
    )
}

/// Store f32x8 lanes back to column `j` of 8 consecutive rows with given stride.
///
/// Unsafe-performance path: uses unchecked indexing (validated by debug_assert).
/// Safe path: uses bounds-checked indexing.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[cfg_attr(feature = "unsafe-performance", allow(unsafe_code))]
pub(crate) fn scatter_col_strided(
    v: magetypes::simd::f32x8,
    data: &mut [f32],
    base_row: usize,
    j: usize,
    stride: usize,
) {
    let mut lane = [0.0f32; 8];
    v.store(&mut lane);
    #[cfg(feature = "unsafe-performance")]
    {
        debug_assert!(
            (base_row + 7) * stride + j < data.len(),
            "scatter_col_strided OOB: base_row={base_row}, j={j}, stride={stride}, len={}",
            data.len()
        );
        // SAFETY: Caller guarantees (base_row + 7) * stride + j < data.len().
        unsafe {
            for (r, &val) in lane.iter().enumerate() {
                *data.get_unchecked_mut((base_row + r) * stride + j) = val;
            }
        }
    }
    #[cfg(not(feature = "unsafe-performance"))]
    for (r, &val) in lane.iter().enumerate() {
        data[(base_row + r) * stride + j] = val;
    }
}

mod adaptive_quant;
mod block_l2;
mod cfl;
mod dct16;
mod dct32;
mod dct4;
mod dct64;
mod dct8;
mod dequant;
mod entropy;
mod epf;
mod fused_dct8;
mod gab;
mod gaborish5x5;
mod idct16;
mod idct32;
mod idct64;
mod mask1x1;
mod noise;
mod pixel_loss;
mod quantize;
mod transpose;
mod xyb;

// Re-export archmage token types so callers don't need a direct archmage dependency
#[cfg(target_arch = "aarch64")]
pub use archmage::NeonToken;
pub use archmage::SimdToken;
#[cfg(target_arch = "wasm32")]
pub use archmage::Wasm128Token;
#[cfg(target_arch = "x86_64")]
pub use archmage::X64V3Token;

// --- Dispatching functions (runtime auto-select) ---

pub use adaptive_quant::{compute_pre_erosion, per_block_modulations};
pub use block_l2::compute_block_l2_errors;
pub use cfl::find_best_multiplier as cfl_find_best_multiplier;
pub use cfl::find_best_multiplier_newton as cfl_find_best_multiplier_newton;
pub use cfl::{NEWTON_EPS_DEFAULT, NEWTON_MAX_ITERS_DEFAULT};
pub use dct4::{
    dct_4x4_full, dct_4x8_full, dct_8x4_full, idct_4x4_full, idct_4x8_full, idct_8x4_full,
};
pub use dct8::{dct_8x8, idct_8x8};
pub use dct16::{dct_8x16, dct_16x8, dct_16x16};
pub use dct32::{dct_16x32, dct_32x16, dct_32x32};
pub use dct64::{dct_32x64, dct_64x32, dct_64x64};
pub use dequant::dequant_block_dct8;
pub use entropy::{
    EntropyCoeffResult, entropy_estimate_coeffs, fast_log2f, fast_pow2f, fast_powf,
    shannon_entropy_bits,
};
pub use epf::{epf_step1, epf_step2, pad_plane};
pub use fused_dct8::fused_dct8_entropy;
pub use gab::gab_smooth_channel;
pub use gaborish5x5::gaborish_5x5_channel;
pub use idct16::{idct_8x16, idct_16x8, idct_16x16};
pub use idct32::{idct_16x32, idct_32x16, idct_32x32};
pub use idct64::{idct_32x64, idct_64x32, idct_64x64};
pub use mask1x1::compute_mask1x1;
pub use noise::denoise_channel;
pub use pixel_loss::pixel_domain_loss;
pub use quantize::{quantize_block_dct8, quantize_block_large};
pub use transpose::transpose_8x8;
pub use xyb::{linear_rgb_to_xyb_batch, xyb_to_linear_rgb_batch, xyb_to_linear_rgb_planar};

// --- Scalar variants (no token needed) ---

pub use adaptive_quant::{compute_pre_erosion_scalar, per_block_modulations_scalar};
pub use block_l2::compute_block_l2_errors_scalar;
pub use cfl::find_best_multiplier_newton_scalar as cfl_find_best_multiplier_newton_scalar;
pub use cfl::find_best_multiplier_scalar as cfl_find_best_multiplier_scalar;
pub use dct4::{
    dct_4x4_full_scalar, dct_4x8_full_scalar, dct_8x4_full_scalar, idct_4x4_full_scalar,
    idct_4x8_full_scalar, idct_8x4_full_scalar,
};
pub use dct8::{dct_8x8_scalar, idct_8x8_scalar};
pub use dct16::{dct_8x16_scalar, dct_16x8_scalar, dct_16x16_scalar};
pub use dct32::{dct_16x32_scalar, dct_32x16_scalar, dct_32x32_scalar};
pub use dct64::{dct_32x64_scalar, dct_64x32_scalar, dct_64x64_scalar};
pub use dequant::dequant_dct8_scalar;
pub use entropy::{entropy_coeffs_scalar, shannon_entropy_scalar};
pub use epf::{epf_step1_scalar, epf_step2_scalar};
pub use fused_dct8::fused_dct8_entropy_fallback;
pub use gab::gab_smooth_scalar;
pub use gaborish5x5::gaborish_5x5_scalar;
pub use idct16::{idct_8x16_scalar, idct_16x8_scalar, idct_16x16_scalar};
pub use idct32::{idct_16x32_scalar, idct_32x16_scalar, idct_32x32_scalar};
pub use idct64::{idct_32x64_scalar, idct_64x32_scalar, idct_64x64_scalar};
pub use mask1x1::compute_mask1x1_scalar;
pub use noise::denoise_channel_scalar;
pub use pixel_loss::pixel_domain_loss_scalar;
pub use quantize::{quantize_dct8_scalar, quantize_large_scalar};
// transpose has no separate scalar — the dispatching fn IS the scalar fallback
pub use xyb::{forward_xyb_scalar, inverse_xyb_planar_scalar, inverse_xyb_scalar};

// --- AVX2 variants (require X64V3Token) ---

#[cfg(target_arch = "x86_64")]
pub use adaptive_quant::{compute_pre_erosion_avx2, per_block_modulations_avx2};
#[cfg(target_arch = "x86_64")]
pub use block_l2::compute_block_l2_errors_avx2;
#[cfg(target_arch = "x86_64")]
pub use cfl::find_best_multiplier_avx2 as cfl_find_best_multiplier_avx2;
#[cfg(target_arch = "x86_64")]
pub use dct4::{
    dct_4x4_full_avx2, dct_4x8_full_avx2, dct_8x4_full_avx2, idct_4x4_full_avx2,
    idct_4x8_full_avx2, idct_8x4_full_avx2,
};
#[cfg(target_arch = "x86_64")]
pub use dct8::{dct_8x8_avx2, idct_8x8_avx2};
#[cfg(target_arch = "x86_64")]
pub use dct16::{dct_8x16_avx2, dct_16x8_avx2, dct_16x16_avx2};
#[cfg(target_arch = "x86_64")]
pub use dct32::{dct_16x32_avx2, dct_32x16_avx2, dct_32x32_avx2};
#[cfg(target_arch = "x86_64")]
pub use dct64::{dct_32x64_avx2, dct_64x32_avx2, dct_64x64_avx2};
#[cfg(target_arch = "x86_64")]
pub use dequant::dequant_dct8_avx2;
#[cfg(target_arch = "x86_64")]
pub use entropy::{entropy_coeffs_avx2, shannon_entropy_avx2};
#[cfg(target_arch = "x86_64")]
pub use epf::{epf_step1_avx2, epf_step2_avx2};
#[cfg(target_arch = "x86_64")]
pub use fused_dct8::fused_dct8_entropy_avx2;
#[cfg(target_arch = "x86_64")]
pub use gab::gab_smooth_avx2;
#[cfg(target_arch = "x86_64")]
pub use gaborish5x5::gaborish_5x5_avx2;
#[cfg(target_arch = "x86_64")]
pub use idct16::{idct_8x16_avx2, idct_16x8_avx2, idct_16x16_avx2};
#[cfg(target_arch = "x86_64")]
pub use idct32::{idct_16x32_avx2, idct_32x16_avx2, idct_32x32_avx2};
#[cfg(target_arch = "x86_64")]
pub use idct64::{idct_32x64_avx2, idct_64x32_avx2, idct_64x64_avx2};
#[cfg(target_arch = "x86_64")]
pub use mask1x1::compute_mask1x1_avx2;
#[cfg(target_arch = "x86_64")]
pub use noise::denoise_channel_avx2;
#[cfg(target_arch = "x86_64")]
pub use pixel_loss::pixel_domain_loss_avx2;
#[cfg(target_arch = "x86_64")]
pub use quantize::{quantize_dct8_avx2, quantize_large_avx2};
#[cfg(target_arch = "x86_64")]
pub use transpose::transpose_8x8_avx2;
#[cfg(target_arch = "x86_64")]
pub use xyb::{forward_xyb_avx2, inverse_xyb_avx2, inverse_xyb_planar_avx2};

// --- NEON variants (require NeonToken) ---

#[cfg(target_arch = "aarch64")]
pub use adaptive_quant::{compute_pre_erosion_neon, per_block_modulations_neon};
#[cfg(target_arch = "aarch64")]
pub use block_l2::compute_block_l2_errors_neon;
#[cfg(target_arch = "aarch64")]
pub use cfl::find_best_multiplier_neon as cfl_find_best_multiplier_neon;
#[cfg(target_arch = "aarch64")]
pub use dct8::{dct_8x8_neon, idct_8x8_neon};
#[cfg(target_arch = "aarch64")]
pub use dct16::{dct_8x16_neon, dct_16x8_neon, dct_16x16_neon};
#[cfg(target_arch = "aarch64")]
pub use dequant::dequant_dct8_neon;
#[cfg(target_arch = "aarch64")]
pub use entropy::{entropy_coeffs_neon, shannon_entropy_neon};
#[cfg(target_arch = "aarch64")]
pub use epf::{epf_step1_neon, epf_step2_neon};
#[cfg(target_arch = "aarch64")]
pub use gab::gab_smooth_neon;
#[cfg(target_arch = "aarch64")]
pub use gaborish5x5::gaborish_5x5_neon;
#[cfg(target_arch = "aarch64")]
pub use idct16::{idct_8x16_neon, idct_16x8_neon, idct_16x16_neon};
#[cfg(target_arch = "aarch64")]
pub use mask1x1::compute_mask1x1_neon;
#[cfg(target_arch = "aarch64")]
pub use noise::denoise_channel_neon;
#[cfg(target_arch = "aarch64")]
pub use pixel_loss::pixel_domain_loss_neon;
#[cfg(target_arch = "aarch64")]
pub use quantize::{quantize_dct8_neon, quantize_large_neon};
#[cfg(target_arch = "aarch64")]
pub use transpose::transpose_8x8_neon;
#[cfg(target_arch = "aarch64")]
pub use xyb::{forward_xyb_neon, inverse_xyb_neon, inverse_xyb_planar_neon};

// --- WASM SIMD128 variants (require Wasm128Token) ---

#[cfg(target_arch = "wasm32")]
pub use adaptive_quant::{compute_pre_erosion_wasm128, per_block_modulations_wasm128};
#[cfg(target_arch = "wasm32")]
pub use block_l2::compute_block_l2_errors_wasm128;
#[cfg(target_arch = "wasm32")]
pub use cfl::find_best_multiplier_wasm128 as cfl_find_best_multiplier_wasm128;
#[cfg(target_arch = "wasm32")]
pub use dct8::{dct_8x8_wasm128, idct_8x8_wasm128};
#[cfg(target_arch = "wasm32")]
pub use dct16::{dct_8x16_wasm128, dct_16x8_wasm128, dct_16x16_wasm128};
#[cfg(target_arch = "wasm32")]
pub use dequant::dequant_dct8_wasm128;
#[cfg(target_arch = "wasm32")]
pub use entropy::{entropy_coeffs_wasm128, shannon_entropy_wasm128};
#[cfg(target_arch = "wasm32")]
pub use epf::{epf_step1_wasm128, epf_step2_wasm128};
#[cfg(target_arch = "wasm32")]
pub use idct16::{idct_8x16_wasm128, idct_16x8_wasm128, idct_16x16_wasm128};
#[cfg(target_arch = "wasm32")]
pub use mask1x1::compute_mask1x1_wasm128;
#[cfg(target_arch = "wasm32")]
pub use noise::denoise_channel_wasm128;
#[cfg(target_arch = "wasm32")]
pub use pixel_loss::pixel_domain_loss_wasm128;
#[cfg(target_arch = "wasm32")]
pub use quantize::{quantize_dct8_wasm128, quantize_large_wasm128};
#[cfg(target_arch = "wasm32")]
pub use xyb::{forward_xyb_wasm128, inverse_xyb_planar_wasm128, inverse_xyb_wasm128};
