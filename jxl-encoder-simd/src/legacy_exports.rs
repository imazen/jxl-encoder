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
pub use entropy::{entropy_coeffs_avx2, shannon_entropy_impl_v3 as shannon_entropy_avx2};
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
pub use sanitize::{is_finite_plane_avx2, sanitize_finite_avx2};
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
pub use entropy::{entropy_coeffs_neon, shannon_entropy_impl_neon as shannon_entropy_neon};
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
pub use sanitize::{is_finite_plane_neon, sanitize_finite_neon};
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
pub use entropy::{
    entropy_coeffs_wasm128, shannon_entropy_impl_wasm128 as shannon_entropy_wasm128,
};
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
