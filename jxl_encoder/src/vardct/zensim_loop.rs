// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Zensim-based quantization loop for iterative quality refinement.
//!
//! Alternative to the butteraugli loop: uses zensim's psychovisual metric for
//! both global quality tracking and per-pixel spatial error map (diffmap).
//!
//! Uses planar linear RGB input directly — no RGBA interleaving overhead.
//! The diffmap includes SSIM + edge artifact + MSE features for comprehensive
//! spatial error detection (blocking, ringing, detail loss).
//!
//! Per-tile distance calibration uses `approx_butteraugli()` — a trained mapping
//! from zensim scores to butteraugli-compatible distance units.

use super::ac_strategy::AcStrategyMap;
use super::adaptive_quant::quantize_quant_field;
use super::chroma_from_luma::CflMap;
use super::common::*;
use super::encoder::VarDctEncoder;
use super::frame::DistanceParams;
use crate::debug_rect;

/// Tunable parameters for the zensim loop, readable from environment variables.
/// Allows parameter sweeps without recompilation.
#[derive(Debug, Clone)]
struct ZensimParams {
    // Diffmap options
    masking_strength: Option<f32>, // ZENSIM_MASKING (f32 or "none")
    sqrt: bool,                    // ZENSIM_SQRT (0/1)
    include_hf: bool,              // ZENSIM_HF (0/1)
    include_edge_mse: bool,        // ZENSIM_EDGE_MSE (0/1)
    // Tile aggregation
    norm_power: f32,     // ZENSIM_NORM (2.0=L2, 4.0=L4, etc.)
    spatial_weight: f32, // ZENSIM_SPATIAL_W (0.0-1.0)
    ratio_max: f32,      // ZENSIM_RATIO_MAX (1.0-10.0)
    // Redistribution
    alpha_base: f32, // ZENSIM_ALPHA (0.01-1.0)
    factor_max: f32, // ZENSIM_FACTOR_MAX (1.01-2.0)
}

impl ZensimParams {
    fn from_env() -> Self {
        Self {
            // Defaults tuned by parameter sweep (2026-03-08, 20 images validated).
            // masking=8,sqrt=0: preserves diffmap dynamic range for redistribution.
            // L2/sw=0.6/rmax=3.0: best quality gain across diverse images.
            //   → +1.262 SSIM2 at +1.10% size (e7-zen4 vs e7, 20-img avg)
            //   → +0.000 SSIM2 at -4.61% size (e8-zen4 vs e7, 20-img avg)
            // L6/sw=1.0/rmax=2.0 tested: lower size cost but -11% less quality
            // gain and e8-zen4 regression (-0.114 SSIM2). Overfits on 4-img sample.
            masking_strength: Self::env_masking("ZENSIM_MASKING", Some(8.0)),
            sqrt: Self::env_bool("ZENSIM_SQRT", false),
            include_hf: Self::env_bool("ZENSIM_HF", true),
            include_edge_mse: Self::env_bool("ZENSIM_EDGE_MSE", true),
            norm_power: Self::env_f32("ZENSIM_NORM", 2.0),
            spatial_weight: Self::env_f32("ZENSIM_SPATIAL_W", 0.6),
            ratio_max: Self::env_f32("ZENSIM_RATIO_MAX", 3.0),
            alpha_base: Self::env_f32("ZENSIM_ALPHA", 0.25),
            factor_max: Self::env_f32("ZENSIM_FACTOR_MAX", 1.15),
        }
    }

    fn env_f32(name: &str, default: f32) -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    fn env_bool(name: &str, default: bool) -> bool {
        match std::env::var(name).ok().as_deref() {
            Some("0" | "false" | "no") => false,
            Some("1" | "true" | "yes") => true,
            _ => default,
        }
    }

    fn env_masking(name: &str, default: Option<f32>) -> Option<f32> {
        match std::env::var(name).ok().as_deref() {
            Some("none" | "0" | "off") => None,
            Some(s) => s.parse().ok().map(Some).unwrap_or(default),
            None => default,
        }
    }
}

/// Compute per-block tile distance from a zensim diffmap.
///
/// For each transform in the strategy map, computes the L4 norm of diffmap
/// pixels in the covered region (emphasizing worst-case pixels within each
/// tile), then normalizes so the global average tile distance matches
/// `anchor_dist`.
///
/// The anchor_dist is typically `target_distance` (budget-neutral redistribution)
/// with an optional small correction from the measured quality. This avoids
/// the bias in `approx_butteraugli()` which caused +55% file size inflation
/// when used directly as the anchor.
///
/// L4 norm is a compromise: mean would ignore spatial peaks (missing bad pixels),
/// L16 (as butteraugli uses) would be too aggressive for SSIM-based diffmap
/// values which already have 11×11 spatial smoothing.
#[allow(clippy::too_many_arguments)]
fn compute_tile_dist(
    diffmap: &[f32],
    width: usize,
    height: usize,
    ac_strategy: &AcStrategyMap,
    xsize_blocks: usize,
    ysize_blocks: usize,
    anchor_dist: f32,
    tile_dist: &mut [f32],
    params: &ZensimParams,
) {
    tile_dist.fill(0.0);

    let norm_power = params.norm_power as f64;
    let inv_power = 1.0 / norm_power;

    let mut sum_raw = 0.0f64;
    let mut n_tiles = 0u32;
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

            let mut norm_sum = 0.0f64;
            let mut pixels = 0.0f64;
            for py in px_start_y..px_end_y {
                for px in px_start_x..px_end_x {
                    let v = diffmap[py * width + px] as f64;
                    if !v.is_finite() {
                        continue;
                    }
                    norm_sum += v.abs().powf(norm_power);
                    pixels += 1.0;
                }
            }
            if pixels == 0.0 {
                pixels = 1.0;
            }
            // Lp norm: (sum(|v|^p) / n)^(1/p)
            let tile_norm = (norm_sum / pixels).powf(inv_power) as f32;

            for sy in 0..covered_y {
                for sx in 0..covered_x {
                    tile_dist[(by + sy) * xsize_blocks + (bx + sx)] = tile_norm;
                }
            }

            sum_raw += tile_norm as f64;
            n_tiles += 1;
        }
    }

    if n_tiles == 0 || sum_raw < 1e-12 {
        tile_dist.fill(anchor_dist);
        return;
    }
    let avg_raw = (sum_raw / n_tiles as f64) as f32;
    let spatial_weight = params.spatial_weight;
    let ratio_max = params.ratio_max;
    for td in tile_dist.iter_mut() {
        let ratio = if avg_raw > 1e-10 {
            (*td / avg_raw).min(ratio_max)
        } else {
            1.0
        };
        let blended = 1.0 - spatial_weight + spatial_weight * ratio;
        *td = anchor_dist * blended;
    }
}

/// Refine AC strategy by splitting large transforms with high perceptual error.
///
/// Scans the strategy map for multi-block transforms (DCT16+) where the
/// tile distance exceeds `split_threshold * target_distance`. Those blocks
/// are split one level down (e.g., DCT32x32 → four DCT16x16).
///
/// Returns the number of splits performed.
///
/// Currently disabled: splitting large transforms with high error causes size
/// inflation (+5-12%) without proportional quality gain. Kept for future
/// re-enablement once RD impact is characterized.
#[allow(dead_code)]
fn refine_strategy_from_diffmap(
    ac_strategy: &mut AcStrategyMap,
    tile_dist: &[f32],
    xsize_blocks: usize,
    ysize_blocks: usize,
    target_distance: f32,
    split_threshold: f32,
) -> usize {
    let threshold = split_threshold * target_distance;
    let mut splits = 0;

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            if !ac_strategy.is_first(bx, by) {
                continue;
            }
            let cx = ac_strategy.covered_blocks_x(bx, by);
            let cy = ac_strategy.covered_blocks_y(bx, by);
            // Only consider multi-block transforms
            if cx <= 1 && cy <= 1 {
                continue;
            }
            let td = tile_dist[by * xsize_blocks + bx];
            if td > threshold && ac_strategy.split_one_level(bx, by) {
                splits += 1;
            }
        }
    }

    splits
}

impl VarDctEncoder {
    /// Zensim quantization loop: iteratively refines per-block quant_field
    /// using zensim's psychovisual metric for both global quality and spatial
    /// error distribution.
    ///
    /// Same structure as butteraugli_refine_quant_field. The spatial error map
    /// comes from zensim's diffmap (per-pixel SSIM + edge + MSE error in XYB
    /// space, weighted across channels). The global quality is tracked via
    /// `approx_butteraugli()` which maps zensim scores to butteraugli-compatible
    /// distance units.
    #[cfg(feature = "zensim-loop")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn zensim_refine_quant_field(
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
        ac_strategy: &mut AcStrategyMap,
        patches_data: Option<&super::patches::PatchesData>,
        splines_data: Option<&super::splines::SplinesData>,
    ) -> DistanceParams {
        use super::epf;
        use super::reconstruct::{gab_smooth, reconstruct_xyb, xyb_to_linear_rgb_planar};

        let target_distance = self.distance;
        let num_blocks = xsize_blocks * ysize_blocks;
        let padded_pixels = padded_width * padded_height;
        let n = width * height;

        // Read tunable parameters from environment variables.
        // Defaults match the benchmark-validated configuration.
        let params = ZensimParams::from_env();

        let (src_r, src_g, src_b) = deinterleave_rgb(linear_rgb, n);
        let z = zensim::Zensim::new(zensim::ZensimProfile::latest()).with_parallel(false);
        let precomputed = match z.precompute_reference_linear_planar(
            [&src_r, &src_g, &src_b],
            width,
            height,
            width,
        ) {
            Ok(r) => r,
            Err(_) => return initial_params.clone(),
        };
        drop((src_r, src_g, src_b));

        let diffmap_opts = zensim::DiffmapOptions {
            weighting: zensim::DiffmapWeighting::Trained,
            include_edge_mse: params.include_edge_mse,
            include_hf: params.include_hf,
            masking_strength: params.masking_strength,
            sqrt: params.sqrt,
        };

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

        let iters = self.zensim_iters as usize;
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

            // Step 5: Zensim comparison → global score + per-pixel diffmap.
            // Pass planar linear RGB directly — no RGBA interleaving, no byte cast.
            let dm_result = match z.compute_with_ref_and_diffmap_linear_planar(
                &precomputed,
                [&recon_r, &recon_g, &recon_b],
                width,
                height,
                padded_width, // stride = padded_width (encoder pads rows for SIMD)
                diffmap_opts,
            ) {
                Ok(r) => r,
                Err(_) => return current_params,
            };

            let zensim_score = dm_result.score();
            let diffmap = dm_result.diffmap();

            // Step 6: Compute per-block error from zensim diffmap.
            // Use L4 norm per tile to capture spatial error distribution.
            let measured_dist = dm_result.result().approx_butteraugli() as f32;
            compute_tile_dist(
                diffmap,
                width,
                height,
                ac_strategy,
                xsize_blocks,
                ysize_blocks,
                target_distance,
                &mut tile_dist,
                &params,
            );

            // Strategy refinement disabled: splitting large transforms with high
            // error causes size inflation (+5-12%) without proportional quality gain.
            // Pure quant redistribution is more size-neutral.
            // TODO: re-enable with better thresholds once RD impact is characterized.

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
                debug_rect!(
                    "zensim/iter",
                    0,
                    0,
                    width,
                    height,
                    "iter={}/{} zensim={:.2} bfly≈{:.3} target={:.3} gs={} qf_avg={:.4} qf=[{:.4};{:.4}] td_max={:.2}",
                    iter,
                    iters,
                    zensim_score,
                    measured_dist,
                    target_distance,
                    current_params.global_scale,
                    qf_avg,
                    qf_min,
                    qf_max,
                    td_max
                );
            }

            // Last iteration is compare-only
            if iter == iters {
                break;
            }

            // Step 7: Symmetric sum-preserving redistribution of quant_field.
            //
            // Tiles with above-average error get lower quant_field (more bits),
            // tiles with below-average error get higher quant_field (fewer bits).
            // Adaptive K_ALPHA scales by 1/(1+CV). Factor clamped per iteration.
            let td_sum: f64 = tile_dist.iter().map(|&v| v as f64).sum();
            let td_avg = (td_sum / num_blocks as f64) as f32;

            if td_avg > 1e-10 {
                let td_var: f64 = tile_dist
                    .iter()
                    .map(|&v| {
                        let d = v as f64 - td_avg as f64;
                        d * d
                    })
                    .sum::<f64>()
                    / num_blocks as f64;
                let td_cv = (td_var.sqrt() / td_avg as f64) as f32;
                let k_alpha = params.alpha_base / (1.0 + td_cv);
                let factor_max = params.factor_max;

                let qf_sum_before: f64 = quant_field_float.iter().map(|&v| v as f64).sum();

                for bi in 0..num_blocks {
                    let ratio = tile_dist[bi] / td_avg;
                    let factor =
                        (1.0 + k_alpha * (ratio - 1.0)).clamp(1.0 / factor_max, factor_max);
                    quant_field_float[bi] *= factor;

                    if quant_field_float[bi] > qf_higher {
                        quant_field_float[bi] = qf_higher;
                    }
                    if quant_field_float[bi] < qf_lower {
                        quant_field_float[bi] = qf_lower;
                    }
                }

                // Renormalize: preserve original sum to control size growth.
                let qf_sum_after: f64 = quant_field_float.iter().map(|&v| v as f64).sum();
                if qf_sum_after > 1e-10 {
                    let scale = (qf_sum_before / qf_sum_after) as f32;
                    for v in quant_field_float.iter_mut() {
                        *v *= scale;
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

/// Deinterleave `[R, G, B, R, G, B, ...]` into three separate channel buffers.
fn deinterleave_rgb(rgb: &[f32], n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for i in 0..n {
        r[i] = rgb[i * 3];
        g[i] = rgb[i * 3 + 1];
        b[i] = rgb[i * 3 + 2];
    }
    (r, g, b)
}
