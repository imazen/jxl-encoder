// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

#![forbid(unsafe_code)]
#![no_std]
extern crate alloc;

mod block_l2;
mod dct16;
mod dct8;
mod dequant;
mod entropy;
mod epf;
mod gab;
mod gaborish5x5;
mod idct16;
mod mask1x1;
mod pixel_loss;
mod quantize;
mod transpose;
mod xyb;

// Re-export archmage token types so callers don't need a direct archmage dependency
#[cfg(target_arch = "aarch64")]
pub use archmage::NeonToken;
pub use archmage::SimdToken;
#[cfg(target_arch = "x86_64")]
pub use archmage::X64V3Token;

// --- Dispatching functions (runtime auto-select) ---

pub use block_l2::compute_block_l2_errors;
pub use dct8::{dct_8x8, idct_8x8};
pub use dct16::{dct_8x16, dct_16x8, dct_16x16};
pub use dequant::dequant_block_dct8;
pub use entropy::{EntropyCoeffResult, entropy_estimate_coeffs};
pub use epf::{epf_step1, epf_step2};
pub use gab::gab_smooth_channel;
pub use gaborish5x5::gaborish_5x5_channel;
pub use idct16::{idct_8x16, idct_16x8, idct_16x16};
pub use mask1x1::compute_mask1x1;
pub use pixel_loss::pixel_domain_loss;
pub use quantize::quantize_block_dct8;
pub use transpose::transpose_8x8;
pub use xyb::{linear_rgb_to_xyb_batch, xyb_to_linear_rgb_batch, xyb_to_linear_rgb_planar};

// --- Scalar variants (no token needed) ---

pub use block_l2::compute_block_l2_errors_scalar;
pub use dct8::{dct_8x8_scalar, idct_8x8_scalar};
pub use dct16::{dct_8x16_scalar, dct_16x8_scalar, dct_16x16_scalar};
pub use dequant::dequant_dct8_scalar;
pub use entropy::entropy_coeffs_scalar;
pub use epf::{epf_step1_scalar, epf_step2_scalar};
pub use gab::gab_smooth_scalar;
pub use gaborish5x5::gaborish_5x5_scalar;
pub use idct16::{idct_8x16_scalar, idct_16x8_scalar, idct_16x16_scalar};
pub use mask1x1::compute_mask1x1_scalar;
pub use pixel_loss::pixel_domain_loss_scalar;
pub use quantize::quantize_dct8_scalar;
// transpose has no separate scalar — the dispatching fn IS the scalar fallback
pub use xyb::{forward_xyb_scalar, inverse_xyb_planar_scalar, inverse_xyb_scalar};

// --- AVX2 variants (require X64V3Token) ---

#[cfg(target_arch = "x86_64")]
pub use block_l2::compute_block_l2_errors_avx2;
#[cfg(target_arch = "x86_64")]
pub use dct8::{dct_8x8_avx2, idct_8x8_avx2};
#[cfg(target_arch = "x86_64")]
pub use dct16::{dct_8x16_avx2, dct_16x8_avx2, dct_16x16_avx2};
#[cfg(target_arch = "x86_64")]
pub use dequant::dequant_dct8_avx2;
#[cfg(target_arch = "x86_64")]
pub use entropy::entropy_coeffs_avx2;
#[cfg(target_arch = "x86_64")]
pub use epf::{epf_step1_avx2, epf_step2_avx2};
#[cfg(target_arch = "x86_64")]
pub use gab::gab_smooth_avx2;
#[cfg(target_arch = "x86_64")]
pub use gaborish5x5::gaborish_5x5_avx2;
#[cfg(target_arch = "x86_64")]
pub use idct16::{idct_8x16_avx2, idct_16x8_avx2, idct_16x16_avx2};
#[cfg(target_arch = "x86_64")]
pub use mask1x1::compute_mask1x1_avx2;
#[cfg(target_arch = "x86_64")]
pub use pixel_loss::pixel_domain_loss_avx2;
#[cfg(target_arch = "x86_64")]
pub use quantize::quantize_dct8_avx2;
#[cfg(target_arch = "x86_64")]
pub use transpose::transpose_8x8_avx2;
#[cfg(target_arch = "x86_64")]
pub use xyb::{forward_xyb_avx2, inverse_xyb_avx2, inverse_xyb_planar_avx2};

// --- NEON variants (require NeonToken) ---

#[cfg(target_arch = "aarch64")]
pub use block_l2::compute_block_l2_errors_neon;
#[cfg(target_arch = "aarch64")]
pub use dct8::{dct_8x8_neon, idct_8x8_neon};
#[cfg(target_arch = "aarch64")]
pub use dct16::{dct_8x16_neon, dct_16x8_neon, dct_16x16_neon};
#[cfg(target_arch = "aarch64")]
pub use dequant::dequant_dct8_neon;
#[cfg(target_arch = "aarch64")]
pub use entropy::entropy_coeffs_neon;
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
pub use pixel_loss::pixel_domain_loss_neon;
#[cfg(target_arch = "aarch64")]
pub use quantize::quantize_dct8_neon;
#[cfg(target_arch = "aarch64")]
pub use transpose::transpose_8x8_neon;
#[cfg(target_arch = "aarch64")]
pub use xyb::{forward_xyb_neon, inverse_xyb_neon, inverse_xyb_planar_neon};
