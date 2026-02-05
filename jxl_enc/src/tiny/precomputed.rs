// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
        let (mut xyb_x, mut xyb_y, mut xyb_b) =
            convert_to_xyb_padded(width, height, padded_width, padded_height, linear_rgb);

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
            )
        };

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
        }
    }
}

/// Convert linear RGB to XYB color space with padding to block boundaries.
fn convert_to_xyb_padded(
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
    linear_rgb: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    use crate::color::xyb::linear_rgb_to_xyb;

    let padded_n = padded_width * padded_height;
    let mut xyb_x = vec![0.0f32; padded_n];
    let mut xyb_y = vec![0.0f32; padded_n];
    let mut xyb_b = vec![0.0f32; padded_n];

    // Convert the actual image pixels
    for y in 0..height {
        for x in 0..width {
            let src_idx = y * width + x;
            let dst_idx = y * padded_width + x;
            let r = linear_rgb[src_idx * 3];
            let g = linear_rgb[src_idx * 3 + 1];
            let b = linear_rgb[src_idx * 3 + 2];
            let (xv, yv, bv) = linear_rgb_to_xyb(r, g, b);
            xyb_x[dst_idx] = xv;
            xyb_y[dst_idx] = yv;
            xyb_b[dst_idx] = bv;
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
