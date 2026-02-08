// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Edge-Preserving Filter (EPF) for decoder-side reconstruction.
//!
//! The EPF is a bilateral filter that smooths flat regions while preserving edges.
//! It operates on XYB pixel data after IDCT and gaborish smooth.
//!
//! Three filter steps, controlled by `epf_iters`:
//! - epf_iters=1: Step 2 only (lightest)
//! - epf_iters=2: Step 1 + Step 2
//! - epf_iters=3: Step 0 + Step 1 + Step 2 (heaviest)

use super::ac_strategy::AcStrategyMap;
use super::chroma_from_luma::CflMap;
use super::common::BLOCK_DIM;
use super::frame::DistanceParams;
use super::reconstruct::{gab_smooth, reconstruct_xyb};

/// Constants from libjxl epf.h
const K_INV_SIGMA_NUM: f32 = -1.171_572_9;

/// Default EPF parameters from libjxl loop_filter.cc
const EPF_QUANT_MUL: f32 = 0.46;
const EPF_PASS0_SIGMA_SCALE: f32 = 0.9;
const EPF_PASS2_SIGMA_SCALE: f32 = 6.5;
const EPF_BORDER_SAD_MUL: f32 = 2.0 / 3.0;

/// Channel importance weights for SAD computation
const EPF_CHANNEL_SCALE: [f32; 3] = [40.0, 5.0, 3.5];

/// Default sharpness LUT: epf_sharp_lut[i] = i / 7.0
const EPF_SHARP_LUT: [f32; 8] = [
    0.0,
    1.0 / 7.0,
    2.0 / 7.0,
    3.0 / 7.0,
    4.0 / 7.0,
    5.0 / 7.0,
    6.0 / 7.0,
    1.0,
];

/// Compute the inverse sigma map for EPF filtering.
///
/// Returns a 2D map of inv_sigma values, one per 8x8 block.
/// `inv_sigma = 1 / sigma` where `sigma = epf_quant_mul / (quant_scale * raw_quant * K_INV_SIGMA_NUM) * sharp_lut[sharpness]`
///
/// The sigma stays negative (K_INV_SIGMA_NUM is negative), so inv_sigma is negative.
pub(crate) fn compute_inv_sigma_map(
    quant_field: &[u8],
    sharpness_map: &[u8],
    quant_scale: f32,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> Vec<f32> {
    let mut inv_sigma = vec![0.0f32; xsize_blocks * ysize_blocks];

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let idx = by * xsize_blocks + bx;
            let raw_quant = quant_field[idx] as f32;
            let sharpness = sharpness_map[idx].min(7) as usize;

            let sigma_quant = EPF_QUANT_MUL / (quant_scale * raw_quant * K_INV_SIGMA_NUM);
            let sigma = sigma_quant * EPF_SHARP_LUT[sharpness];

            // Sigma should be negative (K_INV_SIGMA_NUM < 0), clamp to avoid div-by-zero
            if sigma.abs() > 1e-10 {
                inv_sigma[idx] = 1.0 / sigma;
            }
            // If sigma ~= 0, inv_sigma stays 0 -> filter has no effect (all weights = 1)
        }
    }

    inv_sigma
}

/// EPF weight function: w = max(0, sad * inv_sigma + 1)
#[inline(always)]
fn epf_weight(sad: f32, inv_sigma: f32) -> f32 {
    (sad * inv_sigma + 1.0).max(0.0)
}

/// Get pixel value with clamped bounds
#[inline(always)]
fn get_pixel(plane: &[f32], x: isize, y: isize, width: usize, height: usize) -> f32 {
    let cx = x.clamp(0, width as isize - 1) as usize;
    let cy = y.clamp(0, height as isize - 1) as usize;
    plane[cy * width + cx]
}

/// Compute the SAD (sum of absolute differences) for a 3x3 plus pattern.
///
/// Compares the 5-pixel plus pattern centered at (cx, cy) with the same pattern
/// at (nx, ny), using channel importance weights.
fn sad_3x3_plus(
    planes: &[Vec<f32>; 3],
    cx: isize,
    cy: isize,
    nx: isize,
    ny: isize,
    width: usize,
    height: usize,
) -> f32 {
    // Plus pattern offsets: center, up, left, right, down
    const PLUS: [(isize, isize); 5] = [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)];

    let mut sad = 0.0f32;
    for &(dy, dx) in &PLUS {
        for c in 0..3 {
            let cv = get_pixel(&planes[c], cx + dx, cy + dy, width, height);
            let nv = get_pixel(&planes[c], nx + dx, ny + dy, width, height);
            sad += (cv - nv).abs() * EPF_CHANNEL_SCALE[c];
        }
    }
    sad
}

/// Get the border SAD multiplier for a pixel position within a block.
///
/// Pixels at block edges (first/last row or column of an 8x8 block) use a
/// reduced multiplier to avoid filtering across block boundaries where
/// quantization may cause artificial edges.
#[inline(always)]
fn border_mul(px: usize, py: usize) -> f32 {
    let at_border_x = px.is_multiple_of(BLOCK_DIM) || px % BLOCK_DIM == BLOCK_DIM - 1;
    let at_border_y = py.is_multiple_of(BLOCK_DIM) || py % BLOCK_DIM == BLOCK_DIM - 1;
    if at_border_x || at_border_y {
        EPF_BORDER_SAD_MUL
    } else {
        1.0
    }
}

/// Apply EPF Step 0: 5x5 plus kernel with 3x3-plus SAD.
///
/// This is the heaviest filter step, using 12 neighbor positions with
/// multi-point SAD comparison.
fn epf_step0(
    planes: &[Vec<f32>; 3],
    inv_sigma: &[f32],
    xsize_blocks: usize,
    width: usize,
    height: usize,
) -> [Vec<f32>; 3] {
    let base_sm = EPF_PASS0_SIGMA_SCALE * 1.65;

    // 12 neighbor offsets for the 5x5 plus pattern
    const NEIGHBORS: [(isize, isize); 12] = [
        (-2, 0),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -2),
        (0, -1),
        (0, 1),
        (0, 2),
        (1, -1),
        (1, 0),
        (1, 1),
        (2, 0),
    ];

    let mut output = [
        vec![0.0f32; width * height],
        vec![0.0f32; width * height],
        vec![0.0f32; width * height],
    ];

    for py in 0..height {
        let by = py / BLOCK_DIM;
        for px in 0..width {
            let bx = px / BLOCK_DIM;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];

            if is == 0.0 {
                // No filtering
                for c in 0..3 {
                    output[c][py * width + px] = planes[c][py * width + px];
                }
                continue;
            }

            let sm = base_sm * border_mul(px, py);
            let eff_inv_sigma = is * sm;

            let cx = px as isize;
            let cy = py as isize;

            let mut total_weight = 1.0f32;
            let mut sums = [0.0f32; 3];
            for c in 0..3 {
                sums[c] = planes[c][py * width + px];
            }

            for &(dy, dx) in &NEIGHBORS {
                let nx = cx + dx;
                let ny = cy + dy;
                let sad = sad_3x3_plus(planes, cx, cy, nx, ny, width, height);
                let w = epf_weight(sad, eff_inv_sigma);
                total_weight += w;
                for c in 0..3 {
                    sums[c] += w * get_pixel(&planes[c], nx, ny, width, height);
                }
            }

            let inv_tw = 1.0 / total_weight;
            for c in 0..3 {
                output[c][py * width + px] = sums[c] * inv_tw;
            }
        }
    }

    output
}

/// Apply the full EPF pipeline to XYB pixel planes.
///
/// `epf_iters` controls filter strength:
/// - 0: no filtering
/// - 1: Step 2 only (lightest)
/// - 2: Step 1 + Step 2
/// - 3: Step 0 + Step 1 + Step 2 (heaviest)
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_epf(
    planes: &mut [Vec<f32>; 3],
    quant_field: &[u8],
    sharpness_map: &[u8],
    quant_scale: f32,
    epf_iters: u32,
    xsize_blocks: usize,
    ysize_blocks: usize,
    width: usize,
    height: usize,
) {
    if epf_iters == 0 {
        return;
    }

    let inv_sigma = compute_inv_sigma_map(
        quant_field,
        sharpness_map,
        quant_scale,
        xsize_blocks,
        ysize_blocks,
    );

    // Step 0: heavy 5x5 plus (only at epf_iters >= 3)
    if epf_iters >= 3 {
        let result = epf_step0(planes, &inv_sigma, xsize_blocks, width, height);
        *planes = result;
    }

    let n = width * height;

    // Step 1: medium 3x3 cross with multi-point SAD (only at epf_iters >= 2)
    // Uses SIMD-accelerated kernel from jxl_simd.
    if epf_iters >= 2 {
        let mut out_x = vec![0.0f32; n];
        let mut out_y = vec![0.0f32; n];
        let mut out_b = vec![0.0f32; n];
        jxl_simd::epf_step1(
            &planes[0],
            &planes[1],
            &planes[2],
            &mut out_x,
            &mut out_y,
            &mut out_b,
            &inv_sigma,
            xsize_blocks,
            width,
            height,
            1.65, // sigma_scale for step 1
            EPF_BORDER_SAD_MUL,
        );
        planes[0] = out_x;
        planes[1] = out_y;
        planes[2] = out_b;
    }

    // Step 2: light 3x3 cross with single-pixel SAD (always runs when epf_iters >= 1)
    // Uses SIMD-accelerated kernel from jxl_simd.
    {
        let mut out_x = vec![0.0f32; n];
        let mut out_y = vec![0.0f32; n];
        let mut out_b = vec![0.0f32; n];
        jxl_simd::epf_step2(
            &planes[0],
            &planes[1],
            &planes[2],
            &mut out_x,
            &mut out_y,
            &mut out_b,
            &inv_sigma,
            xsize_blocks,
            width,
            height,
            EPF_PASS2_SIGMA_SCALE * 1.65, // sigma_scale for step 2
            EPF_BORDER_SAD_MUL,
        );
        planes[0] = out_x;
        planes[1] = out_y;
        planes[2] = out_b;
    }
}

/// Compute per-block masked L2 distance between original and reconstructed XYB.
///
/// Channel weights: X=12.34, Y=1.0, B=0.2 (from libjxl ComputeBlockL2Distance).
fn compute_block_l2_errors(
    original: [&[f32]; 3],
    reconstructed: [&[f32]; 3],
    mask1x1: &[f32],
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> Vec<f32> {
    const CHANNEL_WEIGHTS: [f32; 3] = [12.339_445, 1.0, 0.2];
    let padded_width = xsize_blocks * BLOCK_DIM;
    let nblocks = xsize_blocks * ysize_blocks;
    let mut errors = vec![0.0f32; nblocks];

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let block_idx = by * xsize_blocks + bx;
            let mut total_err = 0.0f32;

            for py in 0..BLOCK_DIM {
                for px in 0..BLOCK_DIM {
                    let y = by * BLOCK_DIM + py;
                    let x = bx * BLOCK_DIM + px;
                    let pixel_idx = y * padded_width + x;
                    let mask = mask1x1[pixel_idx];
                    let mask_sq = mask * mask;

                    for c in 0..3 {
                        let diff = original[c][pixel_idx] - reconstructed[c][pixel_idx];
                        total_err += CHANNEL_WEIGHTS[c] * mask_sq * diff * diff;
                    }
                }
            }

            errors[block_idx] = total_err;
        }
    }

    errors
}

/// Compute per-block EPF sharpness map using libjxl's two-pass algorithm.
///
/// The algorithm tests sharpness candidates [0, 2, 7] (or [0, 4] at high distance),
/// reconstructs with each, and selects the best per block via greedy + context refinement.
///
/// Returns a Vec<u8> of sharpness values (0-7), one per 8x8 block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_epf_sharpness(
    original_xyb: [&[f32]; 3],
    quant_dc: &[Vec<Vec<i16>>; 3],
    quant_ac: &[Vec<Vec<[i32; 64]>>; 3],
    quant_field: &[u8],
    mask1x1: &[f32],
    params: &DistanceParams,
    cfl_map: &CflMap,
    ac_strategy: &AcStrategyMap,
    enable_gaborish: bool,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> Vec<u8> {
    let nblocks = xsize_blocks * ysize_blocks;
    let padded_width = xsize_blocks * BLOCK_DIM;
    let padded_height = ysize_blocks * BLOCK_DIM;

    // Choose sharpness candidates based on distance
    let candidates: &[u8] = if params.distance > 4.5 {
        &[0, 4]
    } else {
        &[0, 2, 7]
    };

    // For each candidate, reconstruct + gab + EPF, compute per-block error
    let mut error_maps: Vec<Vec<f32>> = Vec::with_capacity(candidates.len());

    for &sharpness_val in candidates {
        // Reconstruct XYB from quantized coefficients
        let mut recon = reconstruct_xyb(
            quant_dc,
            quant_ac,
            params,
            quant_field,
            cfl_map,
            ac_strategy,
            xsize_blocks,
            ysize_blocks,
        );

        // Apply decoder-side gaborish smooth
        if enable_gaborish {
            gab_smooth(&mut recon, padded_width, padded_height);
        }

        // Apply EPF with uniform sharpness
        let uniform_sharpness = vec![sharpness_val; nblocks];
        apply_epf(
            &mut recon,
            quant_field,
            &uniform_sharpness,
            params.scale,
            params.epf_iters,
            xsize_blocks,
            ysize_blocks,
            padded_width,
            padded_height,
        );

        // Compute per-block masked L2 error vs original
        let errors = compute_block_l2_errors(
            original_xyb,
            [&recon[0], &recon[1], &recon[2]],
            mask1x1,
            xsize_blocks,
            ysize_blocks,
        );

        error_maps.push(errors);
    }

    // Map candidate index to sharpness LUT index for context computation
    let candidate_lut: Vec<usize> = candidates
        .iter()
        .map(|&v| match v {
            0 => 0,
            2 => 1,
            4 => 1,
            7 => 2,
            _ => 0,
        })
        .collect();

    // Pass 1: Greedy selection with neighbor preference
    const K_FAVOR_NO_SMOOTHING: f32 = 0.99;
    let mut sharpness_map = vec![4u8; nblocks]; // default 4
    let num_candidates = candidates.len();
    let num_contexts = num_candidates * num_candidates; // top * left contexts
    let mut histo = vec![vec![0u32; num_candidates]; num_contexts];

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let block_idx = by * xsize_blocks + bx;

            // Get top and left neighbor info
            let (top_val, top_err) = if by > 0 {
                let top_idx = (by - 1) * xsize_blocks + bx;
                let top_s = sharpness_map[top_idx];
                let top_ci = candidates.iter().position(|&c| c == top_s).unwrap_or(0);
                (top_ci, error_maps[top_ci][top_idx])
            } else {
                (0, f32::MAX)
            };

            let (left_val, left_err) = if bx > 0 {
                let left_idx = by * xsize_blocks + bx - 1;
                let left_s = sharpness_map[left_idx];
                let left_ci = candidates.iter().position(|&c| c == left_s).unwrap_or(0);
                (left_ci, error_maps[left_ci][left_idx])
            } else {
                (0, f32::MAX)
            };

            // Find best candidate for this block
            let mut best_ci = 0;
            let mut best_err = f32::MAX;
            for ci in 0..num_candidates {
                let mut err = error_maps[ci][block_idx];
                if candidates[ci] == 0 {
                    err *= K_FAVOR_NO_SMOOTHING;
                }
                if err < best_err {
                    best_err = err;
                    best_ci = ci;
                }
            }

            // Neighbor preference: if neighbor is better, use neighbor's sharpness
            let selected_ci = if best_err < top_err.min(left_err) {
                best_ci
            } else if top_err < left_err {
                top_val
            } else {
                left_val
            };

            sharpness_map[block_idx] = candidates[selected_ci];

            // Update histogram
            let ctx = candidate_lut[top_val] * num_candidates + candidate_lut[left_val];
            if ctx < num_contexts {
                histo[ctx][selected_ci] += 1;
            }
        }
    }

    // Pass 2: Context-based re-weighting
    let clamped_d = params.distance.clamp(0.5, 10.0);
    let c3base: f32 = 0.980_172;
    let c3clamp: f32 = 0.859_703_4;
    let c3 = c3clamp.max(c3base.powf(clamped_d));
    let c5: f32 = 0.108_769_04;

    // Compute totals per context
    let mut totals = vec![1.0f32; num_contexts]; // init to 1 to avoid div-by-zero
    for ctx in 0..num_contexts {
        for &count in &histo[ctx][..num_candidates] {
            totals[ctx] += count as f32;
        }
    }

    // Compute multipliers
    let mut muls = vec![vec![1.0f32; num_candidates]; num_contexts];
    for ctx in 0..num_contexts {
        for ci in 0..num_candidates {
            let count = histo[ctx][ci] as f32;
            let mut mul = 1.0 / (1.0 + c5 * (1.0 + count / totals[ctx]).ln() / clamped_d);
            if candidates[ci] == 0 {
                mul *= c3;
            }
            muls[ctx][ci] = mul;
        }
    }

    // Re-scan all blocks with context multipliers
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let block_idx = by * xsize_blocks + bx;

            let top_ci = if by > 0 {
                let top_s = sharpness_map[(by - 1) * xsize_blocks + bx];
                candidates.iter().position(|&c| c == top_s).unwrap_or(0)
            } else {
                0
            };

            let left_ci = if bx > 0 {
                let left_s = sharpness_map[by * xsize_blocks + bx - 1];
                candidates.iter().position(|&c| c == left_s).unwrap_or(0)
            } else {
                0
            };

            let ctx = candidate_lut[top_ci] * num_candidates + candidate_lut[left_ci];

            let mut best_ci = 0;
            let mut best_err = f32::MAX;
            for ci in 0..num_candidates {
                let err = error_maps[ci][block_idx] * muls[ctx.min(num_contexts - 1)][ci];
                if err < best_err {
                    best_err = err;
                    best_ci = ci;
                }
            }

            sharpness_map[block_idx] = candidates[best_ci];
        }
    }

    sharpness_map
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EPF on constant input should produce constant output.
    #[test]
    fn test_epf_constant_passthrough() {
        let w = 16;
        let h = 16;
        let val = 0.5f32;
        let mut planes = [vec![val; w * h], vec![val; w * h], vec![val; w * h]];

        let xsize_blocks = w / BLOCK_DIM;
        let ysize_blocks = h / BLOCK_DIM;
        let quant_field = vec![10u8; xsize_blocks * ysize_blocks];
        let sharpness_map = vec![4u8; xsize_blocks * ysize_blocks];

        apply_epf(
            &mut planes,
            &quant_field,
            &sharpness_map,
            1.0,
            3, // all 3 steps
            xsize_blocks,
            ysize_blocks,
            w,
            h,
        );

        // Constant input -> constant output
        for (c, plane) in planes.iter().enumerate() {
            for (i, &v) in plane.iter().enumerate() {
                let err = (v - val).abs();
                assert!(
                    err < 1e-5,
                    "EPF constant: c={} i={} got {} expected {}",
                    c,
                    i,
                    v,
                    val
                );
            }
        }
    }

    /// EPF with sharpness=0 should have no effect (sigma=0 -> skip).
    #[test]
    fn test_epf_sharpness_zero_noop() {
        let w = 16;
        let h = 16;
        let mut planes: [Vec<f32>; 3] =
            core::array::from_fn(|c| (0..w * h).map(|i| i as f32 * 0.01 + c as f32).collect());

        let original = planes.clone();

        let xsize_blocks = w / BLOCK_DIM;
        let ysize_blocks = h / BLOCK_DIM;
        let quant_field = vec![10u8; xsize_blocks * ysize_blocks];
        let sharpness_map = vec![0u8; xsize_blocks * ysize_blocks]; // sharpness=0

        apply_epf(
            &mut planes,
            &quant_field,
            &sharpness_map,
            1.0,
            2,
            xsize_blocks,
            ysize_blocks,
            w,
            h,
        );

        // sharpness=0 -> sharp_lut[0]=0 -> sigma=0 -> inv_sigma=0 -> no filtering
        for c in 0..3 {
            for i in 0..w * h {
                assert_eq!(
                    planes[c][i], original[c][i],
                    "EPF with sharpness=0 should be noop: c={} i={}",
                    c, i
                );
            }
        }
    }

    /// EPF should smooth high-frequency noise while preserving the mean.
    #[test]
    fn test_epf_smoothing() {
        let w = 16;
        let h = 16;

        // Create a plane with a constant base + random noise
        let base = 0.5f32;
        let mut planes = [vec![base; w * h], vec![base; w * h], vec![base; w * h]];

        // Add alternating noise to Y channel
        for py in 0..h {
            for px in 0..w {
                if (px + py) % 2 == 0 {
                    planes[1][py * w + px] += 0.01;
                } else {
                    planes[1][py * w + px] -= 0.01;
                }
            }
        }

        let original_mean: f32 = planes[1].iter().sum::<f32>() / (w * h) as f32;

        let xsize_blocks = w / BLOCK_DIM;
        let ysize_blocks = h / BLOCK_DIM;
        let quant_field = vec![5u8; xsize_blocks * ysize_blocks];
        let sharpness_map = vec![7u8; xsize_blocks * ysize_blocks]; // max sharpness

        apply_epf(
            &mut planes,
            &quant_field,
            &sharpness_map,
            1.0,
            2,
            xsize_blocks,
            ysize_blocks,
            w,
            h,
        );

        // Mean should be approximately preserved
        let filtered_mean: f32 = planes[1].iter().sum::<f32>() / (w * h) as f32;
        let mean_err = (original_mean - filtered_mean).abs();
        assert!(
            mean_err < 0.01,
            "EPF should preserve mean: orig={}, filtered={}, err={}",
            original_mean,
            filtered_mean,
            mean_err
        );

        // Variance should decrease (smoothing)
        let original_var: f32 = planes[1]
            .iter()
            .map(|&v| (v - filtered_mean).powi(2))
            .sum::<f32>()
            / (w * h) as f32;
        // The original had alternating +-0.01, so variance ~ 0.0001
        // After filtering, variance should be less
        assert!(
            original_var < 0.0001,
            "EPF should reduce variance: var={}",
            original_var
        );
    }

    /// EPF with epf_iters=0 should be a no-op.
    #[test]
    fn test_epf_iters_zero() {
        let w = 16;
        let h = 16;
        let mut planes: [Vec<f32>; 3] =
            core::array::from_fn(|c| (0..w * h).map(|i| i as f32 * 0.01 + c as f32).collect());

        let original = planes.clone();

        let xsize_blocks = w / BLOCK_DIM;
        let ysize_blocks = h / BLOCK_DIM;
        let quant_field = vec![10u8; xsize_blocks * ysize_blocks];
        let sharpness_map = vec![4u8; xsize_blocks * ysize_blocks];

        apply_epf(
            &mut planes,
            &quant_field,
            &sharpness_map,
            1.0,
            0, // no filtering
            xsize_blocks,
            ysize_blocks,
            w,
            h,
        );

        for c in 0..3 {
            for i in 0..w * h {
                assert_eq!(
                    planes[c][i], original[c][i],
                    "EPF iters=0 should be noop: c={} i={}",
                    c, i
                );
            }
        }
    }
}
