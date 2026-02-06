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

use super::common::BLOCK_DIM;

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

/// Compute single-pixel SAD between two positions.
fn sad_1x1(
    planes: &[Vec<f32>; 3],
    cx: isize,
    cy: isize,
    nx: isize,
    ny: isize,
    width: usize,
    height: usize,
) -> f32 {
    let mut sad = 0.0f32;
    for c in 0..3 {
        let cv = get_pixel(&planes[c], cx, cy, width, height);
        let nv = get_pixel(&planes[c], nx, ny, width, height);
        sad += (cv - nv).abs() * EPF_CHANNEL_SCALE[c];
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

/// Apply EPF Step 1: 3x3 cross kernel with 3x3-plus SAD.
fn epf_step1(
    planes: &[Vec<f32>; 3],
    inv_sigma: &[f32],
    xsize_blocks: usize,
    width: usize,
    height: usize,
) -> [Vec<f32>; 3] {
    let base_sm = 1.65;

    // 4 cross neighbors
    const NEIGHBORS: [(isize, isize); 4] = [(0, -1), (-1, 0), (1, 0), (0, 1)];

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

/// Apply EPF Step 2: 3x3 cross kernel with 1x1 SAD.
///
/// This is the lightest filter step, using single-pixel difference for SAD.
fn epf_step2(
    planes: &[Vec<f32>; 3],
    inv_sigma: &[f32],
    xsize_blocks: usize,
    width: usize,
    height: usize,
) -> [Vec<f32>; 3] {
    let base_sm = EPF_PASS2_SIGMA_SCALE * 1.65;

    // 4 cross neighbors
    const NEIGHBORS: [(isize, isize); 4] = [(0, -1), (-1, 0), (1, 0), (0, 1)];

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
                let sad = sad_1x1(planes, cx, cy, nx, ny, width, height);
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

    // Step 1: medium 3x3 cross with multi-point SAD (only at epf_iters >= 2)
    if epf_iters >= 2 {
        let result = epf_step1(planes, &inv_sigma, xsize_blocks, width, height);
        *planes = result;
    }

    // Step 2: light 3x3 cross with single-pixel SAD (always runs when epf_iters >= 1)
    {
        let result = epf_step2(planes, &inv_sigma, xsize_blocks, width, height);
        *planes = result;
    }
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
