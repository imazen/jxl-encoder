// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SSIMULACRA2-based quantization loop for iterative quality refinement.
//!
//! Alternative to the butteraugli loop: uses per-block linear RGB RMSE for
//! spatial adjustment and full-image SSIMULACRA2 for global quality tracking.
//!
//! Advantages over butteraugli loop:
//! - Faster: SSIM2 + per-block RMSE is cheaper than butteraugli diffmap
//! - SSIM2 is a well-calibrated perceptual metric with a simple 0-100 scale
//!
//! The spatial error map uses per-block RMSE in linear RGB (weighted by
//! luminance to approximate perceptual importance). This is cruder than
//! butteraugli's per-pixel HVS model but the adaptive quant field already
//! captures most perceptual weighting.

use super::ac_strategy::AcStrategyMap;
use super::adaptive_quant::quantize_quant_field;
use super::chroma_from_luma::CflMap;
use super::common::*;
use super::encoder::VarDctEncoder;
use super::frame::DistanceParams;
use crate::debug_rect;

/// Map butteraugli distance to approximate target SSIM2 score.
/// Based on measured data from quality_compare (CID22, 41 images × 9 distances).
fn distance_to_target_ssim2(distance: f32) -> f64 {
    // Piecewise linear fit to measured data:
    // d=0.25→94, d=0.5→91, d=1.0→87, d=1.5→83, d=2.0→79,
    // d=2.5→76, d=3.0→72, d=4.0→67, d=5.0→61
    let d = distance as f64;
    if d <= 0.25 {
        94.0
    } else if d <= 0.5 {
        94.0 - (d - 0.25) * 12.0 // 94→91
    } else if d <= 1.0 {
        91.0 - (d - 0.5) * 8.0 // 91→87
    } else if d <= 2.0 {
        87.0 - (d - 1.0) * 8.0 // 87→79
    } else if d <= 3.0 {
        79.0 - (d - 2.0) * 7.0 // 79→72
    } else if d <= 5.0 {
        72.0 - (d - 3.0) * 5.5 // 72→61
    } else {
        (61.0 - (d - 5.0) * 4.0).max(20.0)
    }
}

impl VarDctEncoder {
    /// SSIM2 quantization loop: iteratively refines per-block quant_field
    /// using SSIMULACRA2 for global quality assessment and per-block linear
    /// RGB RMSE for spatial error distribution.
    ///
    /// Same structure as butteraugli_refine_quant_field but faster per iteration.
    #[cfg(feature = "ssim2-loop")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ssim2_refine_quant_field(
        &self,
        linear_rgb: &[f32],
        width: usize,
        height: usize,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize,
        padded_height: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        initial_params: &DistanceParams,
        quant_field: &mut [u8],
        quant_field_float: &mut [f32],
        initial_quant_field_float: &[f32],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        patches_data: Option<&super::patches::PatchesData>,
        splines_data: Option<&super::splines::SplinesData>,
    ) -> DistanceParams {
        use super::epf;
        use super::reconstruct::{gab_smooth, reconstruct_xyb, xyb_to_linear_rgb_planar};

        let target_distance = self.distance;
        let _target_ssim2 = distance_to_target_ssim2(target_distance);
        let num_blocks = xsize_blocks * ysize_blocks;
        let padded_pixels = padded_width * padded_height;

        // Precompute SSIM2 reference from source image (linear RGB → interleaved [f32; 3]).
        let n = width * height;
        let mut source_rgb3: Vec<[f32; 3]> = Vec::with_capacity(n);
        for i in 0..n {
            source_rgb3.push([
                linear_rgb[i * 3],
                linear_rgb[i * 3 + 1],
                linear_rgb[i * 3 + 2],
            ]);
        }
        let source_img = imgref::Img::new(source_rgb3, width, height);
        let ssim2_ref = match fast_ssim2::Ssimulacra2Reference::new(source_img.as_ref()) {
            Ok(r) => r,
            Err(_) => return initial_params.clone(),
        };

        // Pre-deinterleave original linear RGB for per-block error computation.
        let mut orig_r = vec![0.0f32; n];
        let mut orig_g = vec![0.0f32; n];
        let mut orig_b = vec![0.0f32; n];
        for i in 0..n {
            orig_r[i] = linear_rgb[i * 3];
            orig_g[i] = linear_rgb[i * 3 + 1];
            orig_b[i] = linear_rgb[i * 3 + 2];
        }

        // Deviation bounds (same as butteraugli loop).
        let initial_qf_min = initial_quant_field_float
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or(0.01)
            .max(1e-6);
        let initial_qf_max = initial_quant_field_float
            .iter()
            .copied()
            .reduce(f32::max)
            .unwrap_or(1.0);
        let initial_qf_ratio = initial_qf_max / initial_qf_min;
        let qf_max_deviation_low = (250.0f32 / initial_qf_ratio).sqrt();
        let asymmetry = 2.0f32.min(qf_max_deviation_low);
        let qf_lower = initial_qf_min / (asymmetry * qf_max_deviation_low);
        let qf_higher = initial_qf_max * (qf_max_deviation_low / asymmetry);

        // Pre-allocate buffers
        let sharpness = vec![4u8; num_blocks];
        let mut tile_dist = vec![0.0f32; num_blocks];
        let mut recon_r = vec![0.0f32; padded_pixels];
        let mut recon_g = vec![0.0f32; padded_pixels];
        let mut recon_b = vec![0.0f32; padded_pixels];
        let mut transform_out = super::transform::TransformOutput::new(xsize_blocks, ysize_blocks);

        let iters = self.ssim2_iters as usize;
        let mut current_params;

        for iter in 0..iters + 1 {
            // Step 1: SetQuantField — recompute global_scale from float field
            current_params =
                DistanceParams::compute_from_quant_field(target_distance, quant_field_float);
            current_params.x_qm_scale = initial_params.x_qm_scale;
            current_params.b_qm_scale = initial_params.b_qm_scale;
            current_params.epf_iters = initial_params.epf_iters;

            let qf_vec = quantize_quant_field(quant_field_float, current_params.inv_scale);
            quant_field.copy_from_slice(&qf_vec);

            // Step 2: Transform and quantize
            self.transform_and_quantize_into(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                xsize_blocks,
                ysize_blocks,
                &current_params,
                quant_field,
                cfl_map,
                ac_strategy,
                &mut transform_out,
            );

            // Step 3: Reconstruct XYB → linear RGB
            let mut planes = reconstruct_xyb(
                &transform_out.quant_dc,
                &transform_out.quant_ac,
                &current_params,
                quant_field,
                cfl_map,
                ac_strategy,
                xsize_blocks,
                ysize_blocks,
            );

            if self.enable_gaborish {
                gab_smooth(&mut planes, padded_width, padded_height);
            }

            if current_params.epf_iters > 0 {
                epf::apply_epf(
                    &mut planes,
                    quant_field,
                    &sharpness,
                    current_params.scale,
                    current_params.epf_iters,
                    xsize_blocks,
                    ysize_blocks,
                    padded_width,
                    padded_height,
                );
            }

            if let Some(pd) = patches_data {
                super::patches::add_patches(&mut planes, padded_width, pd);
            }

            if let Some(sd) = splines_data {
                super::splines::add_splines(&mut planes, padded_width, width, height, sd);
            }

            // Step 4: Convert reconstructed XYB to planar linear RGB
            xyb_to_linear_rgb_planar(
                &planes[0],
                &planes[1],
                &planes[2],
                &mut recon_r,
                &mut recon_g,
                &mut recon_b,
                padded_pixels,
            );

            // Step 5: Compute full-image SSIM2 (using precomputed reference)
            let mut recon_rgb3: Vec<[f32; 3]> = Vec::with_capacity(n);
            for y in 0..height {
                for x in 0..width {
                    let pi = y * padded_width + x;
                    recon_rgb3.push([recon_r[pi], recon_g[pi], recon_b[pi]]);
                }
            }
            let recon_img = imgref::Img::new(recon_rgb3, width, height);
            let ssim2_score = ssim2_ref.compare(recon_img.as_ref()).unwrap_or(100.0);

            // Step 6: Compute per-block tile distance from linear RGB RMSE.
            // Weight by luminance (0.2126R + 0.7152G + 0.0722B) so bright-channel
            // errors dominate, approximating perceptual importance.
            tile_dist.fill(0.0);
            for by in 0..ysize_blocks {
                for bx in 0..xsize_blocks {
                    if !ac_strategy.is_first(bx, by) {
                        continue;
                    }
                    let covered_x = ac_strategy.covered_blocks_x(bx, by);
                    let covered_y = ac_strategy.covered_blocks_y(bx, by);
                    let px_start_x = bx * BLOCK_DIM;
                    let px_start_y = by * BLOCK_DIM;
                    let px_end_x = ((bx + covered_x) * BLOCK_DIM).min(width);
                    let px_end_y = ((by + covered_y) * BLOCK_DIM).min(height);
                    if px_start_x >= width || px_start_y >= height {
                        continue;
                    }

                    let mut err_sum = 0.0f64;
                    let mut pixels = 0.0f64;
                    for py in px_start_y..px_end_y {
                        for px in px_start_x..px_end_x {
                            let oi = py * width + px;
                            let ri = py * padded_width + px;
                            let dr = (recon_r[ri] - orig_r[oi]) as f64;
                            let dg = (recon_g[ri] - orig_g[oi]) as f64;
                            let db = (recon_b[ri] - orig_b[oi]) as f64;
                            // Luminance-weighted MSE
                            let weighted = 0.2126 * dr * dr + 0.7152 * dg * dg + 0.0722 * db * db;
                            err_sum += weighted;
                            pixels += 1.0;
                        }
                    }
                    if pixels == 0.0 {
                        pixels = 1.0;
                    }
                    let rmse = (err_sum / pixels).sqrt() as f32;

                    // Scale RMSE to butteraugli-like distance units.
                    // Empirical calibration: RMSE ~0.005 in linear RGB ≈ butteraugli 1.0
                    // for typical photographic content. This makes the adjustment logic
                    // (which uses tile_dist/target_distance) work with similar dynamics.
                    let scaled_dist = rmse * 200.0;

                    for sy in 0..covered_y {
                        for sx in 0..covered_x {
                            tile_dist[(by + sy) * xsize_blocks + (bx + sx)] = scaled_dist;
                        }
                    }
                }
            }

            // Log per-iteration summary
            {
                let qf_min = quant_field_float
                    .iter()
                    .copied()
                    .reduce(f32::min)
                    .unwrap_or(0.0);
                let qf_max = quant_field_float
                    .iter()
                    .copied()
                    .reduce(f32::max)
                    .unwrap_or(0.0);
                let qf_sum: f64 = quant_field_float.iter().map(|&v| v as f64).sum();
                let qf_avg = qf_sum / quant_field_float.len() as f64;
                let td_max = tile_dist.iter().copied().reduce(f32::max).unwrap_or(0.0);
                let bad_blocks = tile_dist.iter().filter(|&&d| d > target_distance).count();
                debug_rect!(
                    "ssim2/iter",
                    0,
                    0,
                    width,
                    height,
                    "iter={}/{} ssim2={:.2} target_d={:.3} gs={} qf_avg={:.4} qf=[{:.4};{:.4}] td_max={:.2} bad={}",
                    iter,
                    iters,
                    ssim2_score,
                    target_distance,
                    current_params.global_scale,
                    qf_avg,
                    qf_min,
                    qf_max,
                    td_max,
                    bad_blocks
                );
            }

            // Last iteration is compare-only
            if iter == iters {
                break;
            }

            // Step 7: kOriginalComparisonRound = 1 (same as butteraugli loop)
            const K_ORIGINAL_COMPARISON_ROUND: usize = 1;
            if iter == K_ORIGINAL_COMPARISON_ROUND {
                const K_INIT_MUL: f64 = 0.6;
                const K_ONE_MINUS_INIT_MUL: f64 = 1.0 - K_INIT_MUL;
                for bi in 0..num_blocks {
                    let init_qf = initial_quant_field_float[bi] as f64;
                    let cur_qf = quant_field_float[bi] as f64;
                    let clamp_val = K_ONE_MINUS_INIT_MUL * cur_qf + K_INIT_MUL * init_qf;
                    if cur_qf < clamp_val {
                        let mut v = clamp_val as f32;
                        if v > qf_higher {
                            v = qf_higher;
                        }
                        if v < qf_lower {
                            v = qf_lower;
                        }
                        quant_field_float[bi] = v;
                    }
                }
            }

            // Step 8: Adjust float quant_field based on tile distances
            let cur_pow: f64 = if iter < 2 { 0.2 } else { 0.0 };

            let inv_global_scale = current_params.inv_scale;
            let quantizer_scale = current_params.scale;

            if cur_pow == 0.0 {
                for bi in 0..num_blocks {
                    let diff = tile_dist[bi] / target_distance;
                    if diff > 1.0 {
                        let old = quant_field_float[bi];
                        quant_field_float[bi] = old * diff;
                        let qf_old = (old * inv_global_scale + 0.5).floor() as i32;
                        let qf_new =
                            (quant_field_float[bi] * inv_global_scale + 0.5).floor() as i32;
                        if qf_old == qf_new {
                            quant_field_float[bi] = old + quantizer_scale;
                        }
                    }
                    if quant_field_float[bi] > qf_higher {
                        quant_field_float[bi] = qf_higher;
                    }
                    if quant_field_float[bi] < qf_lower {
                        quant_field_float[bi] = qf_lower;
                    }
                }
            } else {
                for bi in 0..num_blocks {
                    let diff = tile_dist[bi] / target_distance;
                    if diff <= 1.0 {
                        quant_field_float[bi] *= (diff as f64).powf(cur_pow) as f32;
                    } else {
                        let old = quant_field_float[bi];
                        quant_field_float[bi] = old * diff;
                        let qf_old = (old * inv_global_scale + 0.5).floor() as i32;
                        let qf_new =
                            (quant_field_float[bi] * inv_global_scale + 0.5).floor() as i32;
                        if qf_old == qf_new {
                            quant_field_float[bi] = old + quantizer_scale;
                        }
                    }
                    if quant_field_float[bi] > qf_higher {
                        quant_field_float[bi] = qf_higher;
                    }
                    if quant_field_float[bi] < qf_lower {
                        quant_field_float[bi] = qf_lower;
                    }
                }
            }
        }

        // Final SetQuantField
        let mut final_params =
            DistanceParams::compute_from_quant_field(target_distance, quant_field_float);
        final_params.x_qm_scale = initial_params.x_qm_scale;
        final_params.b_qm_scale = initial_params.b_qm_scale;
        final_params.epf_iters = initial_params.epf_iters;

        let qf_vec = quantize_quant_field(quant_field_float, final_params.inv_scale);
        quant_field.copy_from_slice(&qf_vec);

        final_params
    }
}
