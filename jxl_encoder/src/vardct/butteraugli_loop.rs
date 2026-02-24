// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Butteraugli quantization loop for iterative quality refinement.
//!
//! Iteratively refines per-block quant_field by measuring perceptual distance
//! (butteraugli) between the original and reconstructed image.
//!
//! Matches libjxl's FindBestQuantization (enc_adaptive_quantization.cc:929-1115):
//! - Works in float quant field domain (values ~0.3-1.5), NOT integer (1-255)
//! - Recomputes global_scale each iteration via SetQuantField (median/MAD)
//! - Returns final DistanceParams for use in CfL pass 2 and encoding

use super::ac_strategy::AcStrategyMap;
use super::adaptive_quant::quantize_quant_field;
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
    /// **Float-domain operation** (matching libjxl FindBestQuantization):
    /// The quant field is maintained in float domain (~0.3-1.5 range). Each
    /// iteration recomputes global_scale from the float field's median/MAD
    /// (matching libjxl's SetQuantField), then converts to u8 for quantization.
    ///
    /// Algorithm:
    /// For each iteration:
    ///   1. SetQuantField: recompute global_scale from float field, convert to u8
    ///   2. transform_and_quantize with current quant_field and new params
    ///   3. reconstruct XYB → apply gab → EPF → XYB-to-linear
    ///   4. butteraugli(original_linear, reconstructed_linear) → per-block distmap
    ///   5. Adjust float quant_field based on tile distances
    ///   6. Enforce deviation bounds from initial field
    ///
    /// AC strategy is FIXED throughout — only quant_field changes.
    /// Returns the final DistanceParams (with recomputed global_scale).
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
        let num_blocks = xsize_blocks * ysize_blocks;
        let padded_pixels = padded_width * padded_height;

        // Precompute butteraugli reference from original image ONCE.
        // Deinterleave to planar to avoid interleave round-trip inside the crate.
        let n = width * height;
        let mut ref_r = vec![0.0f32; n];
        let mut ref_g = vec![0.0f32; n];
        let mut ref_b = vec![0.0f32; n];
        for i in 0..n {
            ref_r[i] = linear_rgb[i * 3];
            ref_g[i] = linear_rgb[i * 3 + 1];
            ref_b[i] = linear_rgb[i * 3 + 2];
        }
        let butteraugli_params = butteraugli::ButteraugliParams::new()
            .with_intensity_target(80.0)
            .with_compute_diffmap(true);
        let reference = match butteraugli::ButteraugliReference::new_linear_planar(
            &ref_r,
            &ref_g,
            &ref_b,
            width,
            height,
            width,
            butteraugli_params,
        ) {
            Ok(r) => r,
            Err(_) => return initial_params.clone(),
        };
        // Free the planar buffers — reference data is already precomputed inside.
        drop(ref_r);
        drop(ref_g);
        drop(ref_b);

        // Compute deviation bounds from the FLOAT initial field (libjxl lines 968-976).
        // These prevent the quant field from diverging too far from the initial field.
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

        // Pre-allocate buffers reused across iterations
        let sharpness = vec![4u8; num_blocks];
        let mut tile_dist = vec![0.0f32; num_blocks];
        let mut recon_r = vec![0.0f32; padded_pixels];
        let mut recon_g = vec![0.0f32; padded_pixels];
        let mut recon_b = vec![0.0f32; padded_pixels];
        let mut transform_out = super::transform::TransformOutput::new(xsize_blocks, ysize_blocks);

        let iters = self.butteraugli_iters as usize;
        let mut current_params;

        // Loop runs iters+1 times (matching libjxl: last iteration is compare-only).
        // i=0..iters-1: SetQuantField + roundtrip + compare + adjust
        // i=iters: SetQuantField + roundtrip + compare + break
        for iter in 0..iters + 1 {
            // Step 1: SetQuantField — recompute global_scale from float field,
            // then convert float → u8.
            // (libjxl: quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field))
            current_params =
                DistanceParams::compute_from_quant_field(target_distance, quant_field_float);
            // Preserve chromacity adjustments and EPF from initial params
            current_params.x_qm_scale = initial_params.x_qm_scale;
            current_params.b_qm_scale = initial_params.b_qm_scale;
            current_params.epf_iters = initial_params.epf_iters;

            // Convert float → u8 with current params' inv_scale
            // (libjxl: SetQuantFieldRect: ClampVal(row_qf[x] * inv_global_scale_ + 0.5f, 1, 255))
            let qf_vec = quantize_quant_field(quant_field_float, current_params.inv_scale);
            quant_field.copy_from_slice(&qf_vec);

            // Step 2: Transform and quantize with current params
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

            // Step 3: Reconstruct XYB from quantized coefficients
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

            // Step 5: Butteraugli comparison
            let result =
                match reference.compare_linear_planar(&recon_r, &recon_g, &recon_b, padded_width) {
                    Ok(r) => r,
                    Err(_) => return current_params,
                };

            let diffmap = match result.diffmap {
                Some(dm) => dm,
                None => return current_params,
            };

            // Step 6: Compute per-block tile distance (16th-power norm, matching libjxl TileDistMap)
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
                    let td = K_TILE_NORM * (dist_norm / pixels).sqrt().sqrt().sqrt().sqrt() as f32;
                    for sy in 0..covered_y {
                        for sx in 0..covered_x {
                            tile_dist[(by + sy) * xsize_blocks + (bx + sx)] = td;
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
                    "bfly/iter",
                    0,
                    0,
                    width,
                    height,
                    "iter={}/{} score={:.3} target={:.3} gs={} qf_avg={:.4} qf=[{:.4};{:.4}] td_max={:.2} bad={}",
                    iter,
                    iters,
                    result.score,
                    target_distance,
                    current_params.global_scale,
                    qf_avg,
                    qf_min,
                    qf_max,
                    td_max,
                    bad_blocks
                );
            }

            // Last iteration is compare-only (libjxl: if (i == iters) break;)
            if iter == iters {
                break;
            }

            // Step 7: kOriginalComparisonRound = 1: constrain toward initial BEFORE adjustment.
            // Prevents oscillation by keeping qf from diverging too far from initial.
            // (libjxl enc_adaptive_quantization.cc:1039-1057)
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

            // Step 8: Adjust float quant_field based on tile distances.
            // (libjxl enc_adaptive_quantization.cc:1059-1110)
            //
            // kPow controls how aggressively to reduce quality of good blocks:
            // iters 0-1: pow=0.2 (gently reduce good blocks to save bits)
            // iters 2+: pow=0.0 (only fix bad blocks)
            let cur_pow: f64 = if iter < 2 {
                0.2 + (target_distance as f64 - 1.0) * 0.0 // kPowMod[0..1] = 0
            } else {
                0.0
            };

            // InvGlobalScale and Scale from current iteration's params
            // (these change per iteration as global_scale is recomputed)
            let inv_global_scale = current_params.inv_scale; // = 65536 / global_scale
            let quantizer_scale = current_params.scale; // = global_scale / 65536

            if cur_pow == 0.0 {
                // Only adjust bad blocks (diff > 1.0)
                // (libjxl enc_adaptive_quantization.cc:1066-1086)
                for bi in 0..num_blocks {
                    let diff = tile_dist[bi] / target_distance;
                    if diff > 1.0 {
                        let old = quant_field_float[bi];
                        quant_field_float[bi] = old * diff;
                        // Minimum step check: if rounding to integer quant produces
                        // the same value, bump by one quantizer step
                        // (libjxl: if (fi == pi) row_q[x] = old + quantizer.Scale())
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
                // Adjust both directions (libjxl enc_adaptive_quantization.cc:1087-1110)
                for bi in 0..num_blocks {
                    let diff = tile_dist[bi] / target_distance;
                    if diff <= 1.0 {
                        // Good quality: reduce precision to save bits
                        quant_field_float[bi] *= (diff as f64).powf(cur_pow) as f32;
                    } else {
                        // Bad quality: increase precision
                        let old = quant_field_float[bi];
                        quant_field_float[bi] = old * diff;
                        // Minimum step check
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

        // Final SetQuantField: compute definitive params from final float field
        // (libjxl enc_adaptive_quantization.cc:1112-1113)
        let mut final_params =
            DistanceParams::compute_from_quant_field(target_distance, quant_field_float);
        final_params.x_qm_scale = initial_params.x_qm_scale;
        final_params.b_qm_scale = initial_params.b_qm_scale;
        final_params.epf_iters = initial_params.epf_iters;

        // Convert final float → u8 with definitive params
        let qf_vec = quantize_quant_field(quant_field_float, final_params.inv_scale);
        quant_field.copy_from_slice(&qf_vec);

        final_params
    }
}
