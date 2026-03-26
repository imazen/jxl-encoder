// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Iterative rate control for improved distance targeting.
//!
//! This module implements libjxl-style rate control: encode → decode → measure
//! butteraugli → adjust quant field → repeat. Typically converges in 2-4
//! iterations with 5% tolerance of target distance.

use super::VarDctEncoder;
use super::adaptive_quant::quantize_quant_field;
use super::frame::DistanceParams;
use super::precomputed::EncoderPrecomputed;
use super::tile_distmap::{TileDistMap, compute_butteraugli_diffmap};
use crate::debug_rect;
use crate::error::Result;

/// Configuration for iterative rate control.
#[derive(Debug, Clone)]
pub struct RateControlConfig {
    /// Maximum number of iterations (default: 3).
    pub max_iterations: usize,
    /// Target tolerance as fraction (default: 0.05 = 5%).
    /// Converges when 95th percentile distance <= target * (1 + tolerance).
    pub tolerance: f32,
    /// Minimum quant field value (default: 1).
    pub qf_min: u8,
    /// Maximum quant field value (default: 255).
    pub qf_max: u8,
}

impl Default for RateControlConfig {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            tolerance: 0.05,
            qf_min: 1,
            qf_max: 255,
        }
    }
}

/// Result of a single rate control iteration.
pub struct IterationResult {
    /// Encoded JXL bytes.
    pub encoded: Vec<u8>,
    /// Maximum butteraugli distance across all tiles.
    pub max_distance: f32,
    /// 95th percentile butteraugli distance.
    pub p95_distance: f32,
    /// Mean butteraugli distance.
    pub mean_distance: f32,
    /// Fraction of tiles exceeding target distance.
    pub fraction_exceeding: f32,
}

/// Encode with iterative rate control.
///
/// This function:
/// 1. Computes precomputed state (XYB, CfL, masking, AC strategy)
/// 2. Initializes quant field from float values
/// 3. Loops: encode → decode → butteraugli → adjust quant field
/// 4. Returns when converged or max iterations reached
///
/// Returns the final encoded bytes and iteration count.
pub fn encode_with_rate_control(
    encoder: &VarDctEncoder,
    precomputed: &EncoderPrecomputed,
    config: &RateControlConfig,
) -> Result<(Vec<u8>, usize)> {
    let target = encoder.distance;

    // Compute distance params from effort profile
    let params = DistanceParams::compute_for_profile(target, &encoder.profile);

    // Initialize quant field
    let mut quant_field = quantize_quant_field(&precomputed.quant_field_float, params.inv_scale);
    let initial_qf = quant_field.clone();

    // Compute bounds for quant field adjustment
    let qf_lower = config.qf_min as f32;
    let qf_upper = config.qf_max as f32;

    let mut best_encoded: Option<Vec<u8>> = None;
    let mut best_p95: f32 = f32::MAX;

    for iter in 0..=config.max_iterations {
        // Encode with current quant field
        let encoded = encode_iteration(encoder, precomputed, &quant_field)?;

        // On last iteration or if decoding fails, return what we have
        if iter == config.max_iterations {
            return Ok((encoded, iter));
        }

        // Decode and measure quality
        let decoded = match decode_jxl_to_linear(&encoded) {
            Some(d) => d,
            None => {
                // Decode failed - return encoded data anyway
                debug_rect!(
                    "rate_ctrl/warn",
                    0,
                    0,
                    precomputed.width,
                    precomputed.height,
                    "decode failed on iteration {}",
                    iter
                );
                return Ok((encoded, iter));
            }
        };

        // Compute butteraugli diffmap
        let diffmap = compute_butteraugli_diffmap(
            &precomputed.linear_rgb,
            &decoded,
            precomputed.width,
            precomputed.height,
        );

        // Compute tile distances
        let tile_dist = TileDistMap::from_diffmap(
            &diffmap,
            precomputed.width,
            precomputed.height,
            &precomputed.ac_strategy,
        );

        let p95_dist = tile_dist.percentile_95();

        #[cfg(feature = "debug-tokens")]
        {
            let max_dist = tile_dist.max();
            let mean_dist = tile_dist.mean();
            let frac_exceed = tile_dist.fraction_exceeding(target);
            eprintln!(
                "[rate_control] iter {}: max={:.3}, p95={:.3}, mean={:.3}, exceed={:.1}%",
                iter,
                max_dist,
                p95_dist,
                mean_dist,
                frac_exceed * 100.0
            );
        }

        // Track best result
        if p95_dist < best_p95 {
            best_p95 = p95_dist;
            best_encoded = Some(encoded.clone());
        }

        // Check convergence: 95th percentile within tolerance of target
        let target_with_tolerance = target * (1.0 + config.tolerance);
        if p95_dist <= target_with_tolerance {
            return Ok((encoded, iter));
        }

        // Adjust quant field based on tile distances
        adjust_quant_field(
            &mut quant_field,
            &tile_dist,
            &initial_qf,
            target,
            iter,
            qf_lower,
            qf_upper,
            precomputed.xsize_blocks,
        );
    }

    // Should not reach here, but return best result if we do
    Ok((best_encoded.unwrap_or_default(), config.max_iterations))
}

/// Encode a single iteration with the given quant field.
fn encode_iteration(
    encoder: &VarDctEncoder,
    precomputed: &EncoderPrecomputed,
    quant_field: &[u8],
) -> Result<Vec<u8>> {
    // Create a modified encoder state and encode
    // This is essentially the encode_from_precomputed method
    encoder.encode_from_precomputed(precomputed, quant_field)
}

/// Adjust quant field based on butteraugli tile distances.
///
/// For tiles exceeding target, increases quant value proportionally.
/// Blends with initial quant field on iteration 1 to prevent oscillation.
#[allow(clippy::too_many_arguments)]
fn adjust_quant_field(
    quant_field: &mut [u8],
    tile_dist: &TileDistMap,
    initial_qf: &[u8],
    target: f32,
    iter: usize,
    qf_lower: f32,
    qf_upper: f32,
    xsize_blocks: usize,
) {
    for by in 0..tile_dist.ysize_blocks {
        for bx in 0..tile_dist.xsize_blocks {
            let idx = by * xsize_blocks + bx;
            let dist = tile_dist.get(bx, by);

            if dist <= target {
                // This tile is within target, no adjustment needed
                continue;
            }

            let ratio = dist / target;
            let old = quant_field[idx] as f32;

            // Decrease quant value (lower = more bits = better quality)
            // quant_field controls quantization strength, so LOWER values = finer quantization
            // To improve quality, we divide by ratio (or equivalently, multiply by 1/ratio)
            let mut new_val = old / ratio;

            // Ensure at least some change when ratio > 1
            if ratio > 1.0 && (new_val as u8) == (old as u8) {
                new_val = old - 1.0;
            }

            // On iteration 1, blend with initial to prevent oscillation
            if iter == 1 {
                let init = initial_qf[idx] as f32;
                new_val = 0.6 * new_val + 0.4 * init;
            }

            // Clamp to valid range
            new_val = new_val.clamp(qf_lower, qf_upper);
            quant_field[idx] = new_val as u8;
        }
    }
}

/// Decode JXL to linear RGB.
///
/// Returns None if decoding fails.
fn decode_jxl_to_linear(data: &[u8]) -> Option<Vec<f32>> {
    // Use jxl-oxide for decoding as it's available as a dev dependency
    // In production, this could use jxl-rs or any other decoder
    use std::io::Cursor;

    let cursor = Cursor::new(data);
    let mut img = match jxl_oxide::JxlImage::builder().read(cursor) {
        Ok(img) => img,
        Err(_) => return None,
    };

    // Request linear sRGB output for butteraugli comparison
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));

    let render = match img.render_frame(0) {
        Ok(r) => r,
        Err(_) => return None,
    };

    let buf = render.image_all_channels();
    let pixels = buf.buf().to_vec();

    Some(pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_control_config_default() {
        let config = RateControlConfig::default();
        assert_eq!(config.max_iterations, 3);
        assert!((config.tolerance - 0.05).abs() < 0.001);
        assert_eq!(config.qf_min, 1);
        assert_eq!(config.qf_max, 255);
    }
}
