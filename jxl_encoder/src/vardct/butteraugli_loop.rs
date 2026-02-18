// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Butteraugli quantization loop for iterative quality refinement.
//!
//! Iteratively refines per-block quant_field by measuring perceptual distance
//! (butteraugli) between the original and reconstructed image.

#![allow(dead_code)]

use super::ac_strategy::AcStrategyMap;
use super::chroma_from_luma::CflMap;
use super::common::*;
use super::encoder::VarDctEncoder;
use super::frame::DistanceParams;
use crate::debug_rect;

impl VarDctEncoder {
    /// Butteraugli quantization loop: iteratively refines per-block quant_field
    /// by measuring perceptual distance (butteraugli) between the original image
    /// and the reconstruction from quantized coefficients.
    ///
    /// Algorithm (libjxl FindBestQuantization):
    /// For each iteration:
    ///   1. transform_and_quantize with current quant_field
    ///   2. reconstruct XYB → apply gab → EPF → XYB-to-linear
    ///   3. butteraugli(original_linear, reconstructed_linear) → per-block distmap
    ///   4. For blocks where distmap > target: increase quant (qf *= distmap/target)
    ///      For blocks where distmap < target: decrease quant (qf *= distmap/target)
    ///   5. Clamp and constrain (don't diverge too far from initial)
    ///
    /// AC strategy is FIXED throughout — only quant_field changes.
    #[cfg(feature = "butteraugli-loop")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn butteraugli_refine_quant_field(
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
        params: &DistanceParams,
        quant_field: &mut [u8],
        initial_quant_field: &[u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        patches_data: Option<&super::patches::PatchesData>,
        splines_data: Option<&super::splines::SplinesData>,
    ) {
        use super::epf;
        use super::reconstruct::{gab_smooth, reconstruct_xyb, xyb_to_linear_rgb_planar};

        let target_distance = self.distance;
        let num_blocks = xsize_blocks * ysize_blocks;
        let padded_pixels = padded_width * padded_height;

        // Precompute butteraugli reference from original image ONCE.
        // This saves ~40-50% of butteraugli time per iteration by caching
        // the XYB conversion and frequency decomposition of the reference.
        let butteraugli_params = butteraugli::ButteraugliParams::new()
            .with_intensity_target(80.0)
            .with_compute_diffmap(true);
        let reference = match butteraugli::ButteraugliReference::new_linear(
            linear_rgb,
            width,
            height,
            butteraugli_params,
        ) {
            Ok(r) => r,
            Err(_) => return, // Bail on error (e.g., image too small)
        };

        // Work in f32 during the loop for precision (libjxl uses float quant_field).
        // Converting to u8 each iteration loses ~0.5-1.5 per value, accumulating over iters.
        let mut qf_float: Vec<f32> = quant_field.iter().map(|&v| v as f32).collect();
        let initial_qf_float: Vec<f32> = initial_quant_field.iter().map(|&v| v as f32).collect();

        // Compute qf_lower/qf_higher deviation bounds (matching libjxl lines 968-976).
        // These prevent the quant field from diverging too far from the initial field,
        // avoiding oscillation and wild over/under-quantization.
        let initial_qf_min = initial_qf_float
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or(1.0)
            .max(1.0);
        let initial_qf_max = initial_qf_float
            .iter()
            .copied()
            .reduce(f32::max)
            .unwrap_or(255.0);
        let initial_qf_ratio = initial_qf_max / initial_qf_min;
        let qf_max_deviation_low = (250.0 / initial_qf_ratio).sqrt();
        let asymmetry = 2.0f32.min(qf_max_deviation_low);
        let qf_lower = (initial_qf_min / (asymmetry * qf_max_deviation_low)).max(1.0);
        let qf_higher = (initial_qf_max * (qf_max_deviation_low / asymmetry)).min(255.0);

        // Pre-allocate buffers reused across butteraugli iterations
        let mut qf_copy = vec![0u8; quant_field.len()];
        let sharpness = vec![4u8; num_blocks];
        let mut tile_dist = vec![0.0f32; num_blocks];
        // Planar reconstruction buffers (padded dimensions, reused across iterations)
        let mut recon_r = vec![0.0f32; padded_pixels];
        let mut recon_g = vec![0.0f32; padded_pixels];
        let mut recon_b = vec![0.0f32; padded_pixels];
        let mut transform_out = super::transform::TransformOutput::new(xsize_blocks, ysize_blocks);

        for iter in 0..self.butteraugli_iters {
            // Step 1: Quantize with current quant_field (convert float→u8 for quantizer)
            for (dst, &src) in qf_copy.iter_mut().zip(qf_float.iter()) {
                *dst = (src.round() as u8).clamp(1, 255);
            }
            self.transform_and_quantize_into(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                xsize_blocks,
                ysize_blocks,
                params,
                &mut qf_copy,
                cfl_map,
                ac_strategy,
                &mut transform_out,
            );

            // Step 2: Reconstruct XYB from quantized coefficients
            let mut planes = reconstruct_xyb(
                &transform_out.quant_dc,
                &transform_out.quant_ac,
                params,
                &qf_copy,
                cfl_map,
                ac_strategy,
                xsize_blocks,
                ysize_blocks,
            );

            // Apply gaborish smooth if enabled
            if self.enable_gaborish {
                gab_smooth(&mut planes, padded_width, padded_height);
            }

            // Apply EPF if active
            if params.epf_iters > 0 {
                epf::apply_epf(
                    &mut planes,
                    &qf_copy,
                    &sharpness,
                    params.scale,
                    params.epf_iters,
                    xsize_blocks,
                    ysize_blocks,
                    padded_width,
                    padded_height,
                );
            }

            // Step 2b: Add patches back (decoder applies patches after gab+EPF via blend kAdd)
            if let Some(pd) = patches_data {
                super::patches::add_patches(&mut planes, padded_width, pd);
            }

            // Step 2c: Add splines back (decoder adds splines after patches)
            if let Some(sd) = splines_data {
                super::splines::add_splines(&mut planes, padded_width, width, height, sd);
            }

            // Step 3: Convert reconstructed XYB to planar linear RGB (in-place, no interleave)
            xyb_to_linear_rgb_planar(
                &planes[0],
                &planes[1],
                &planes[2],
                &mut recon_r,
                &mut recon_g,
                &mut recon_b,
                padded_pixels,
            );

            // Step 4: Compare against precomputed reference using planar API.
            // Pass padded buffers with stride=padded_width; butteraugli reads only
            // width pixels per row, skipping the padding — no crop copy needed.
            let result =
                match reference.compare_linear_planar(&recon_r, &recon_g, &recon_b, padded_width) {
                    Ok(r) => r,
                    Err(_) => return,
                };

            let diffmap = match result.diffmap {
                Some(dm) => dm,
                None => return,
            };

            // Step 5: Compute per-block tile distance (16th-power norm, matching libjxl)
            // libjxl uses TileDistMap with 16th-norm and kTileNorm=1.2 scaling
            const K_TILE_NORM: f32 = 1.2;
            let diffmap_buf = diffmap.buf();
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
                    let mut dist_norm = 0.0f64;
                    let mut pixels = 0.0f64;
                    for py in px_start_y..px_end_y {
                        for px in px_start_x..px_end_x {
                            let v = diffmap_buf[py * width + px] as f64;
                            // v^16 (16th-power norm)
                            let v2 = v * v;
                            let v4 = v2 * v2;
                            let v8 = v4 * v4;
                            let v16 = v8 * v8;
                            dist_norm += v16;
                            pixels += 1.0;
                        }
                    }
                    if pixels == 0.0 {
                        pixels = 1.0;
                    }
                    // x^(1/16) = sqrt(sqrt(sqrt(sqrt(x))))
                    let td = K_TILE_NORM * (dist_norm / pixels).sqrt().sqrt().sqrt().sqrt() as f32;
                    // Fill all sub-blocks of this transform
                    for sy in 0..covered_y {
                        for sx in 0..covered_x {
                            tile_dist[(by + sy) * xsize_blocks + (bx + sx)] = td;
                        }
                    }
                }
            }

            // Step 6: Constrain and adjust quant_field based on tile distances.
            //
            // Convention: higher qf = finer quantization = better quality (same as libjxl).
            // quantize_coeff_ac: val = coef * inv_weight * qac * qm_mul
            // Higher qac (from higher qf) → larger quantized int → more precision.
            //
            // libjxl order: constrain toward initial (kOriginalComparisonRound=1),
            // THEN adjust based on tile distances. Both phases enforce qf_lower/qf_higher.

            // kOriginalComparisonRound = 1: constrain toward initial BEFORE adjustment.
            // Prevents oscillation by keeping qf from diverging too far from initial.
            if iter == 1 {
                const K_INIT_MUL: f64 = 0.6;
                const K_ONE_MINUS_INIT_MUL: f64 = 1.0 - K_INIT_MUL;
                for bi in 0..num_blocks {
                    let init_qf = initial_qf_float[bi] as f64;
                    let cur_qf = qf_float[bi] as f64;
                    let clamp_val = K_ONE_MINUS_INIT_MUL * cur_qf + K_INIT_MUL * init_qf;
                    if cur_qf < clamp_val {
                        qf_float[bi] = (clamp_val as f32).clamp(qf_lower, qf_higher);
                    }
                }
            }

            // Adjust quant_field based on tile distances.
            // ASYMMETRIC: aggressively fix bad blocks, barely touch good blocks.
            // kPow = [0.2, 0.2, 0, 0, ...] — only iters 0-1 touch good blocks, gently.
            let cur_pow: f64 = if iter < 2 {
                0.2 + (target_distance as f64 - 1.0) * 0.0 // kPowMod[0..1] = 0
            } else {
                0.0
            };

            for bi in 0..num_blocks {
                let diff = tile_dist[bi] / target_distance;
                let old_qf = qf_float[bi];

                if diff <= 1.0 {
                    // Quality is good enough — save bits by reducing precision.
                    if cur_pow != 0.0 {
                        // diff < 1 → pow(diff, 0.2) < 1 → qf decreases slightly.
                        qf_float[bi] = old_qf * (diff as f64).powf(cur_pow) as f32;
                    }
                    // cur_pow == 0: don't touch good blocks on later iterations
                } else {
                    // Quality too bad — aggressively improve by increasing qf.
                    qf_float[bi] = old_qf * diff;
                    // Ensure at least 1 integer step change (matching libjxl's rounding check)
                    if qf_float[bi].round() as u8 == old_qf.round() as u8 {
                        qf_float[bi] = old_qf + 1.0;
                    }
                }
                // Enforce deviation bounds after every adjustment (matching libjxl)
                qf_float[bi] = qf_float[bi].clamp(qf_lower, qf_higher);
            }

            // Log per-iteration summary
            let qf_min = qf_float.iter().copied().reduce(f32::min).unwrap_or(1.0);
            let qf_max = qf_float.iter().copied().reduce(f32::max).unwrap_or(255.0);
            let qf_sum: f64 = qf_float.iter().map(|&v| v as f64).sum();
            let qf_avg = qf_sum / qf_float.len() as f64;
            let td_max = tile_dist.iter().copied().reduce(f32::max).unwrap_or(0.0);
            let bad_blocks = tile_dist.iter().filter(|&&d| d > target_distance).count();
            debug_rect!(
                "bfly/iter",
                0,
                0,
                width,
                height,
                "iter={} score={:.3} target={:.3} qf_avg={:.1} qf=[{:.0};{:.0}] td_max={:.2} bad_blocks={}",
                iter,
                result.score,
                target_distance,
                qf_avg,
                qf_min,
                qf_max,
                td_max,
                bad_blocks
            );
        }

        // Convert float quant_field back to u8 for final encoding
        for (dst, &src) in quant_field.iter_mut().zip(qf_float.iter()) {
            *dst = (src.round() as u8).clamp(1, 255);
        }
    }
}
