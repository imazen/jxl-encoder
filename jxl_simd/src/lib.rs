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

#![forbid(unsafe_code)]
#![no_std]
extern crate alloc;

mod adaptive_quant;
mod block_l2;
mod cfl;
mod dct16;
mod dct32;
mod dct64;
mod dct8;
mod dequant;
mod entropy;
mod epf;
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
pub use dct8::{dct_8x8, idct_8x8};
pub use dct16::{dct_8x16, dct_16x8, dct_16x16};
pub use dct32::{dct_16x32, dct_32x16, dct_32x32};
pub use dct64::{dct_32x64, dct_64x32, dct_64x64};
pub use dequant::dequant_block_dct8;
pub use entropy::{EntropyCoeffResult, entropy_estimate_coeffs};
pub use epf::{epf_step1, epf_step2, pad_plane};
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
pub use dct8::{dct_8x8_scalar, idct_8x8_scalar};
pub use dct16::{dct_8x16_scalar, dct_16x8_scalar, dct_16x16_scalar};
pub use dct32::{dct_16x32_scalar, dct_32x16_scalar, dct_32x32_scalar};
pub use dct64::{dct_32x64_scalar, dct_64x32_scalar, dct_64x64_scalar};
pub use dequant::dequant_dct8_scalar;
pub use entropy::entropy_coeffs_scalar;
pub use epf::{epf_step1_scalar, epf_step2_scalar};
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
pub use cfl::find_best_multiplier_wasm128 as cfl_find_best_multiplier_wasm128;
#[cfg(target_arch = "wasm32")]
pub use noise::denoise_channel_wasm128;
