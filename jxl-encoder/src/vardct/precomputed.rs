// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Precomputed encoder state for iterative rate control.
//!
//! This module holds cached computations that don't change between rate control
//! iterations, allowing ~50% time savings per iteration.

use super::ac_strategy::AcStrategyMap;
use super::chroma_from_luma::CflMap;
use super::common::*;
use super::noise::NoiseParams;

/// Precomputed encoder state that can be reused across rate control iterations.
///
/// These computations are independent of the quant field scaling and don't need
/// to be recomputed when adjusting quantization:
/// - XYB color conversion
/// - Gaborish pre-filter
/// - CfL map
/// - Noise params
/// - Float quant field (pre-scaling)
/// - Masking field
/// - Per-pixel mask (for pixel-domain loss)
/// - AC strategy map
pub struct EncoderPrecomputed {
    /// Original image width in pixels.
    pub width: usize,
    /// Original image height in pixels.
    pub height: usize,
    /// Number of 8x8 blocks in x direction.
    pub xsize_blocks: usize,
    /// Number of 8x8 blocks in y direction.
    pub ysize_blocks: usize,
    /// Padded width (rounded up to block boundary).
    pub padded_width: usize,
    /// Padded height (rounded up to block boundary).
    pub padded_height: usize,

    /// XYB X channel (after gaborish if enabled), padded.
    pub xyb_x: Vec<f32>,
    /// XYB Y channel (after gaborish if enabled), padded.
    pub xyb_y: Vec<f32>,
    /// XYB B channel (after gaborish if enabled), padded.
    pub xyb_b: Vec<f32>,

    /// Original linear RGB data (for butteraugli comparison).
    pub linear_rgb: Vec<f32>,

    /// Chroma-from-luma map.
    pub cfl_map: CflMap,
    /// Noise parameters (if noise synthesis enabled).
    pub noise_params: Option<NoiseParams>,
    /// Float quant field (before scaling by inv_scale).
    pub quant_field_float: Vec<f32>,
    /// Masking field for AC strategy selection.
    pub masking: Vec<f32>,
    /// Per-pixel mask for pixel-domain loss (if enabled).
    pub mask1x1: Option<Vec<f32>>,
    /// AC strategy map.
    pub ac_strategy: AcStrategyMap,

    /// Whether gaborish was applied.
    pub gaborish_enabled: bool,
    /// Distance used for initial quant field computation.
    pub base_distance: f32,
    /// X channel pixel chromacity (max gradient of pre-gaborish XYB X).
    pub chromacity_x_pixelized: u32,
    /// B channel pixel chromacity (from pre-gaborish XYB Y/B).
    pub chromacity_b_pixelized: u32,
}

impl EncoderPrecomputed {
    /// Compute precomputed state from linear RGB input.
    ///
    /// This performs all computations that are independent of the final
    /// quant field scaling:
    /// - XYB conversion with edge-replicated padding
    /// - Gaborish inverse (if enabled)
    /// - Noise estimation and optional denoising (if enabled)
    /// - Float quant field and masking
    /// - CfL map
    /// - Per-pixel mask (if pixel-domain loss enabled)
    /// - AC strategy selection
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        distance: f32,
        cfl_enabled: bool,
        ac_strategy_enabled: bool,
        pixel_domain_loss: bool,
        enable_noise: bool,
        enable_denoise: bool,
        enable_gaborish: bool,
        force_strategy: Option<u8>,
        profile: &crate::effort::EffortProfile,
        color_encoding: Option<&crate::headers::color_encoding::ColorEncoding>,
    ) -> Self {
        use super::ac_strategy::compute_ac_strategy;
        use super::adaptive_quant::{compute_mask1x1, compute_quant_field_float};
        use super::chroma_from_luma::compute_cfl_map;
        use super::gaborish::gaborish_inverse;
        use super::noise::{denoise_xyb, estimate_noise_params, noise_quality_coef};

        assert_eq!(linear_rgb.len(), width * height * 3);

        // Calculate dimensions
        let xsize_blocks = div_ceil(width, BLOCK_DIM);
        let ysize_blocks = div_ceil(height, BLOCK_DIM);
        let padded_width = xsize_blocks * BLOCK_DIM;
        let padded_height = ysize_blocks * BLOCK_DIM;

        // Convert to XYB with edge-replicated padding
        let (mut xyb_x, mut xyb_y, mut xyb_b) = convert_to_xyb_padded(
            width,
            height,
            padded_width,
            padded_height,
            linear_rgb,
            color_encoding,
        );

        // Estimate noise parameters (if enabled)
        let noise_params = if enable_noise {
            let quality_coef = noise_quality_coef(distance);
            let params = estimate_noise_params(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                quality_coef,
            );

            // Apply denoising pre-filter if enabled
            if enable_denoise && let Some(ref p) = params {
                denoise_xyb(
                    &mut xyb_x,
                    &mut xyb_y,
                    &mut xyb_b,
                    padded_width,
                    padded_height,
                    p,
                    quality_coef,
                );
            }

            params
        } else {
            None
        };

        // Compute pixel chromacity stats BEFORE gaborish (matching libjxl's pipeline order).
        // Gated at effort >= 7 to skip the full-image gradient scan at low effort.
        let (chromacity_x_pixelized, chromacity_b_pixelized) = if profile.chromacity_adjustment {
            let pixel_stats = super::frame::PixelStatsForChromacityAdjustment::calc(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
            );
            (
                pixel_stats.how_much_is_x_channel_pixelized(),
                pixel_stats.how_much_is_b_channel_pixelized(),
            )
        } else {
            (0, 0)
        };

        // Apply gaborish inverse (5x5 sharpening) before adaptive quant
        if enable_gaborish {
            gaborish_inverse(
                &mut xyb_x,
                &mut xyb_y,
                &mut xyb_b,
                padded_width,
                padded_height,
            );
        }

        // Compute adaptive per-block quantization field and masking
        // When gaborish is off, scale distance by 0.62 for the quant field
        let distance_for_iqf = if enable_gaborish {
            distance
        } else {
            distance * 0.62
        };

        let (quant_field_float, masking) = compute_quant_field_float(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            padded_width,
            padded_height,
            xsize_blocks,
            ysize_blocks,
            distance_for_iqf,
            profile.k_ac_quant,
        );

        // Compute CfL map
        let cfl_map = if cfl_enabled {
            compute_cfl_map(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                profile.cfl_newton,
                profile.cfl_newton_eps,
                profile.cfl_newton_max_iters,
            )
        } else {
            CflMap::zeros(
                div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS),
                div_ceil(ysize_blocks, TILE_DIM_IN_BLOCKS),
            )
        };

        // Compute per-pixel mask for pixel-domain loss
        let mask1x1 = if ac_strategy_enabled && pixel_domain_loss {
            Some(compute_mask1x1(&xyb_y, padded_width, padded_height))
        } else {
            None
        };

        // Compute AC strategy
        let ac_strategy = if let Some(forced) = force_strategy {
            AcStrategyMap::force_strategy(xsize_blocks, ysize_blocks, forced)
        } else if !ac_strategy_enabled {
            AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks)
        } else {
            compute_ac_strategy(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                distance,
                &quant_field_float,
                &masking,
                &cfl_map,
                mask1x1.as_deref(),
                padded_width,
                profile,
            )
        };

        // CfL pass 2 refinement happens in encoder.rs after the butteraugli loop
        // produces the final quant_field. No refinement here — pass 1 values from
        // compute_cfl_map are sufficient for initial AC strategy selection.

        Self {
            width,
            height,
            xsize_blocks,
            ysize_blocks,
            padded_width,
            padded_height,
            xyb_x,
            xyb_y,
            xyb_b,
            linear_rgb: linear_rgb.to_vec(),
            cfl_map,
            noise_params,
            quant_field_float,
            masking,
            mask1x1,
            ac_strategy,
            gaborish_enabled: enable_gaborish,
            base_distance: distance,
            chromacity_x_pixelized,
            chromacity_b_pixelized,
        }
    }
}

/// Convert linear RGB to XYB color space with padding to block boundaries.
///
/// If `primaries` is non-sRGB, applies a 3x3 matrix to convert to sRGB primaries
/// before the XYB transform (the opsin matrix is defined for sRGB/BT.709).
fn convert_to_xyb_padded(
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
    linear_rgb: &[f32],
    color_encoding: Option<&crate::headers::color_encoding::ColorEncoding>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    use super::xyb::primaries_to_srgb_matrix;
    use crate::color::xyb::linear_rgb_to_xyb;

    let primaries_matrix = color_encoding.and_then(primaries_to_srgb_matrix);

    let padded_n = padded_width * padded_height;
    // Output planes are fully overwritten below: rows 0..height by the per-row
    // conversion + right-edge pad, rows height..padded_height by the bottom-pad
    // loop. Safe to dirty-initialize.
    let mut xyb_x = jxl_simd::vec_f32_dirty(padded_n);
    let mut xyb_y = jxl_simd::vec_f32_dirty(padded_n);
    let mut xyb_b = jxl_simd::vec_f32_dirty(padded_n);

    // Scratch buffers for deinterleaving + optional matrix transform. These
    // are written in full every row before being read.
    let mut row_r = jxl_simd::vec_f32_dirty(width);
    let mut row_g = jxl_simd::vec_f32_dirty(width);
    let mut row_b = jxl_simd::vec_f32_dirty(width);

    // Convert the actual image pixels
    for y in 0..height {
        let src_row = y * width;
        for x in 0..width {
            let si = (src_row + x) * 3;
            row_r[x] = linear_rgb[si];
            row_g[x] = linear_rgb[si + 1];
            row_b[x] = linear_rgb[si + 2];
        }

        if let Some(ref m) = primaries_matrix {
            super::xyb::apply_matrix_3x3(&mut row_r, &mut row_g, &mut row_b, m);
        }

        let dst_row = y * padded_width;
        for x in 0..width {
            let (xv, yv, bv) = linear_rgb_to_xyb(row_r[x], row_g[x], row_b[x]);
            xyb_x[dst_row + x] = xv;
            xyb_y[dst_row + x] = yv;
            xyb_b[dst_row + x] = bv;
        }

        // Pad right edge with last pixel value
        if padded_width > width {
            let last_x_idx = y * padded_width + (width - 1);
            let last_x = xyb_x[last_x_idx];
            let last_y = xyb_y[last_x_idx];
            let last_b = xyb_b[last_x_idx];
            for x in width..padded_width {
                let dst_idx = y * padded_width + x;
                xyb_x[dst_idx] = last_x;
                xyb_y[dst_idx] = last_y;
                xyb_b[dst_idx] = last_b;
            }
        }
    }

    // Pad bottom rows by copying the last row
    if padded_height > height {
        let last_row_start = (height - 1) * padded_width;
        for y in height..padded_height {
            let dst_row_start = y * padded_width;
            for x in 0..padded_width {
                xyb_x[dst_row_start + x] = xyb_x[last_row_start + x];
                xyb_y[dst_row_start + x] = xyb_y[last_row_start + x];
                xyb_b[dst_row_start + x] = xyb_b[last_row_start + x];
            }
        }
    }

    (xyb_x, xyb_y, xyb_b)
}
