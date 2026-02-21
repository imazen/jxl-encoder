// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

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

/// Pad a plane into a pre-allocated output buffer with edge replication.
///
/// Like `jxl_simd::pad_plane` but writes into an existing buffer to avoid allocation.
/// `out` must have length >= `(width + 2*pad) * (height + 2*pad)`.
fn pad_plane_into(plane: &[f32], width: usize, height: usize, pad: usize, out: &mut [f32]) {
    let stride = width + 2 * pad;
    // Copy interior rows
    for y in 0..height {
        let src_off = y * width;
        let dst_off = (y + pad) * stride + pad;
        out[dst_off..dst_off + width].copy_from_slice(&plane[src_off..src_off + width]);
    }
    // Replicate left/right edges for interior rows
    for y in 0..height {
        let row_off = (y + pad) * stride;
        let left_val = out[row_off + pad];
        for p in 0..pad {
            out[row_off + p] = left_val;
        }
        let right_val = out[row_off + pad + width - 1];
        for p in 0..pad {
            out[row_off + pad + width + p] = right_val;
        }
    }
    // Replicate top rows (full padded rows including left/right)
    for p in 0..pad {
        let src_off = pad * stride;
        let dst_off = p * stride;
        out.copy_within(src_off..src_off + stride, dst_off);
    }
    // Replicate bottom rows
    for p in 0..pad {
        let src_off = (pad + height - 1) * stride;
        let dst_off = (pad + height + p) * stride;
        out.copy_within(src_off..src_off + stride, dst_off);
    }
}

/// Compute the SAD (sum of absolute differences) for a 3x3 plus pattern on padded planes.
///
/// Compares the 5-pixel plus pattern centered at (cx, cy) with the same pattern
/// at (nx, ny), using channel importance weights.
/// All coordinates are in padded-buffer space (already offset by pad).
fn sad_3x3_plus_padded(
    planes: &[Vec<f32>; 3],
    cx: usize,
    cy: usize,
    nx: usize,
    ny: usize,
    stride: usize,
) -> f32 {
    // Plus pattern offsets: center, up, left, right, down
    // In padded space, these are safe direct index arithmetic
    let c_offsets = [
        cy * stride + cx,
        (cy - 1) * stride + cx,
        cy * stride + (cx - 1),
        cy * stride + (cx + 1),
        (cy + 1) * stride + cx,
    ];
    let n_offsets = [
        ny * stride + nx,
        (ny - 1) * stride + nx,
        ny * stride + (nx - 1),
        ny * stride + (nx + 1),
        (ny + 1) * stride + nx,
    ];

    let mut sad = 0.0f32;
    for i in 0..5 {
        for c in 0..3 {
            sad += (planes[c][c_offsets[i]] - planes[c][n_offsets[i]]).abs() * EPF_CHANNEL_SCALE[c];
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
///
/// Input planes must be pre-padded with `pad_plane(plane, width, height, 3)`.
/// Output planes are unpadded (width * height).
fn epf_step0(
    padded_planes: &[Vec<f32>; 3],
    inv_sigma: &[f32],
    xsize_blocks: usize,
    width: usize,
    height: usize,
    in_stride: usize,
    pad: usize,
) -> [Vec<f32>; 3] {
    let base_sm = EPF_PASS0_SIGMA_SCALE * 1.65;

    // 12 neighbor offsets for the 5x5 plus pattern (dy, dx)
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
                // No filtering — copy from padded to unpadded output
                let padded_idx = (py + pad) * in_stride + (px + pad);
                for c in 0..3 {
                    output[c][py * width + px] = padded_planes[c][padded_idx];
                }
                continue;
            }

            let sm = base_sm * border_mul(px, py);
            let eff_inv_sigma = is * sm;

            // Coordinates in padded-buffer space
            let cx = px + pad;
            let cy = py + pad;

            let center_idx = cy * in_stride + cx;
            let mut total_weight = 1.0f32;
            let mut sums = [0.0f32; 3];
            for c in 0..3 {
                sums[c] = padded_planes[c][center_idx];
            }

            for &(dy, dx) in &NEIGHBORS {
                let nx = (cx as isize + dx) as usize;
                let ny = (cy as isize + dy) as usize;
                let sad = sad_3x3_plus_padded(padded_planes, cx, cy, nx, ny, in_stride);
                let w = epf_weight(sad, eff_inv_sigma);
                total_weight += w;
                let n_idx = ny * in_stride + nx;
                for c in 0..3 {
                    sums[c] += w * padded_planes[c][n_idx];
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
    // Max reach ±3 pixels (±2 neighbor + ±1 SAD extension)
    if epf_iters >= 3 {
        let pad = 3;
        let in_stride = width + 2 * pad;
        let padded: [Vec<f32>; 3] =
            core::array::from_fn(|c| jxl_simd::pad_plane(&planes[c], width, height, pad));
        let result = epf_step0(
            &padded,
            &inv_sigma,
            xsize_blocks,
            width,
            height,
            in_stride,
            pad,
        );
        *planes = result;
    }

    let n = width * height;

    // Step 1 / EPF1: medium 3x3 cross with multi-point SAD (epf_iters >= 1)
    // Max reach ±2 pixels (±1 neighbor + ±1 SAD extension)
    // libjxl dec_cache.cc:164: if (lf.epf_iters >= 1) AddStage(EPF1)
    // EPF1 always runs when EPF is enabled — it's the primary smoothing step.
    if epf_iters >= 1 {
        let pad = 2;
        let in_stride = width + 2 * pad;
        let padded_x = jxl_simd::pad_plane(&planes[0], width, height, pad);
        let padded_y = jxl_simd::pad_plane(&planes[1], width, height, pad);
        let padded_b = jxl_simd::pad_plane(&planes[2], width, height, pad);
        let mut out_x = vec![0.0f32; n];
        let mut out_y = vec![0.0f32; n];
        let mut out_b = vec![0.0f32; n];
        jxl_simd::epf_step1(
            &padded_x,
            &padded_y,
            &padded_b,
            &mut out_x,
            &mut out_y,
            &mut out_b,
            &inv_sigma,
            xsize_blocks,
            width,
            height,
            in_stride,
            pad,
            1.65, // sigma_scale for step 1
            EPF_BORDER_SAD_MUL,
        );
        planes[0] = out_x;
        planes[1] = out_y;
        planes[2] = out_b;
    }

    // Step 2 / EPF2: light 3x3 cross with single-pixel SAD (epf_iters >= 2)
    // Max reach ±1 pixel (±1 neighbor, no SAD extension)
    // libjxl dec_cache.cc:165: if (lf.epf_iters >= 2) AddStage(EPF2)
    if epf_iters >= 2 {
        let pad = 1;
        let in_stride = width + 2 * pad;
        let padded_x = jxl_simd::pad_plane(&planes[0], width, height, pad);
        let padded_y = jxl_simd::pad_plane(&planes[1], width, height, pad);
        let padded_b = jxl_simd::pad_plane(&planes[2], width, height, pad);
        let mut out_x = vec![0.0f32; n];
        let mut out_y = vec![0.0f32; n];
        let mut out_b = vec![0.0f32; n];
        jxl_simd::epf_step2(
            &padded_x,
            &padded_y,
            &padded_b,
            &mut out_x,
            &mut out_y,
            &mut out_b,
            &inv_sigma,
            xsize_blocks,
            width,
            height,
            in_stride,
            pad,
            EPF_PASS2_SIGMA_SCALE * 1.65, // sigma_scale for step 2
            EPF_BORDER_SAD_MUL,
        );
        planes[0] = out_x;
        planes[1] = out_y;
        planes[2] = out_b;
    }
}

/// Apply EPF with pre-allocated scratch buffers and pre-computed inv_sigma.
///
/// This avoids repeated allocation when called multiple times with different
/// sharpness maps (e.g., EPF sharpness search). The caller provides:
/// - `scratch_x/y/b`: 3 pre-allocated output buffers (length >= width*height each)
/// - `padded_scratch`: pre-allocated buffer for padding (reused across steps)
/// - `inv_sigma`: pre-computed from `compute_inv_sigma_map`
#[allow(clippy::too_many_arguments)]
fn apply_epf_with_scratch(
    planes: &mut [Vec<f32>; 3],
    inv_sigma: &[f32],
    epf_iters: u32,
    xsize_blocks: usize,
    width: usize,
    height: usize,
    scratch_x: &mut Vec<f32>,
    scratch_y: &mut Vec<f32>,
    scratch_b: &mut Vec<f32>,
    padded_scratch: &mut [Vec<f32>; 3],
) {
    if epf_iters == 0 {
        return;
    }

    // Step 0: heavy 5x5 plus (only at epf_iters >= 3)
    // Max reach ±3 pixels
    if epf_iters >= 3 {
        let pad = 3;
        let in_stride = width + 2 * pad;
        let padded: [Vec<f32>; 3] =
            core::array::from_fn(|c| jxl_simd::pad_plane(&planes[c], width, height, pad));
        let result = epf_step0(
            &padded,
            inv_sigma,
            xsize_blocks,
            width,
            height,
            in_stride,
            pad,
        );
        *planes = result;
    }

    // Step 1 / EPF1: medium 3x3 cross with multi-point SAD (epf_iters >= 1)
    // Max reach ±2 pixels
    // libjxl dec_cache.cc:164: if (lf.epf_iters >= 1) AddStage(EPF1)
    if epf_iters >= 1 {
        let pad = 2;
        let in_stride = width + 2 * pad;
        let padded_len = in_stride * (height + 2 * pad);
        // Reuse padded_scratch buffers — resize if needed
        for ps in &mut *padded_scratch {
            ps.resize(padded_len, 0.0);
        }
        pad_plane_into(&planes[0], width, height, pad, &mut padded_scratch[0]);
        pad_plane_into(&planes[1], width, height, pad, &mut padded_scratch[1]);
        pad_plane_into(&planes[2], width, height, pad, &mut padded_scratch[2]);
        // No zeroing needed — EPF kernels write every output pixel
        jxl_simd::epf_step1(
            &padded_scratch[0],
            &padded_scratch[1],
            &padded_scratch[2],
            scratch_x,
            scratch_y,
            scratch_b,
            inv_sigma,
            xsize_blocks,
            width,
            height,
            in_stride,
            pad,
            1.65,
            EPF_BORDER_SAD_MUL,
        );
        core::mem::swap(&mut planes[0], scratch_x);
        core::mem::swap(&mut planes[1], scratch_y);
        core::mem::swap(&mut planes[2], scratch_b);
    }

    // Step 2 / EPF2: light 3x3 cross with single-pixel SAD (epf_iters >= 2)
    // Max reach ±1 pixel
    // libjxl dec_cache.cc:165: if (lf.epf_iters >= 2) AddStage(EPF2)
    if epf_iters >= 2 {
        let pad = 1;
        let in_stride = width + 2 * pad;
        let padded_len = in_stride * (height + 2 * pad);
        for ps in &mut *padded_scratch {
            ps.resize(padded_len, 0.0);
        }
        pad_plane_into(&planes[0], width, height, pad, &mut padded_scratch[0]);
        pad_plane_into(&planes[1], width, height, pad, &mut padded_scratch[1]);
        pad_plane_into(&planes[2], width, height, pad, &mut padded_scratch[2]);
        // No zeroing needed — EPF kernels write every output pixel
        jxl_simd::epf_step2(
            &padded_scratch[0],
            &padded_scratch[1],
            &padded_scratch[2],
            scratch_x,
            scratch_y,
            scratch_b,
            inv_sigma,
            xsize_blocks,
            width,
            height,
            in_stride,
            pad,
            EPF_PASS2_SIGMA_SCALE * 1.65,
            EPF_BORDER_SAD_MUL,
        );
        core::mem::swap(&mut planes[0], scratch_x);
        core::mem::swap(&mut planes[1], scratch_y);
        core::mem::swap(&mut planes[2], scratch_b);
    }
}

/// Compute per-block masked L2 distance between original and reconstructed XYB.
///
/// Channel weights: X=12.34, Y=1.0, B=0.2 (from libjxl ComputeBlockL2Distance).
/// Uses SIMD-accelerated kernel (AVX2 on x86_64).
fn compute_block_l2_errors(
    original: [&[f32]; 3],
    reconstructed: [&[f32]; 3],
    mask1x1: &[f32],
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> Vec<f32> {
    jxl_simd::compute_block_l2_errors(original, reconstructed, mask1x1, xsize_blocks, ysize_blocks)
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

    // Reconstruct once — the dequant→CfL→IDCT→gab result is identical for all
    // sharpness candidates. Only the EPF pass differs.
    let mut base_recon = reconstruct_xyb(
        quant_dc,
        quant_ac,
        params,
        quant_field,
        cfl_map,
        ac_strategy,
        xsize_blocks,
        ysize_blocks,
    );

    if enable_gaborish {
        gab_smooth(&mut base_recon, padded_width, padded_height);
    }

    // Pre-allocate scratch buffers for EPF output (reused across candidates)
    let n = padded_width * padded_height;
    let mut scratch_x = vec![0.0f32; n];
    let mut scratch_y = vec![0.0f32; n];
    let mut scratch_b = vec![0.0f32; n];

    // Pre-allocate padded scratch buffers for padding (reused across steps and candidates)
    // Size for the largest padding needed (pad=3 for step 0)
    let max_pad = if params.epf_iters >= 3 {
        3
    } else if params.epf_iters >= 1 {
        2
    } else {
        1
    };
    let max_padded_stride = padded_width + 2 * max_pad;
    let max_padded_len = max_padded_stride * (padded_height + 2 * max_pad);
    let mut padded_scratch: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0f32; max_padded_len]);

    // Pre-allocate uniform sharpness map (reused across candidates)
    let mut uniform_sharpness = vec![0u8; nblocks];

    // For each candidate, clone the base reconstruction and apply EPF
    let mut error_maps: Vec<Vec<f32>> = Vec::with_capacity(candidates.len());

    for &sharpness_val in candidates {
        let mut recon = base_recon.clone();

        // Compute inv_sigma for this sharpness candidate
        uniform_sharpness.fill(sharpness_val);
        let inv_sigma = compute_inv_sigma_map(
            quant_field,
            &uniform_sharpness,
            params.scale,
            xsize_blocks,
            ysize_blocks,
        );

        // Apply EPF with pre-allocated scratch (avoids repeated mmap/munmap)
        apply_epf_with_scratch(
            &mut recon,
            &inv_sigma,
            params.epf_iters,
            xsize_blocks,
            padded_width,
            padded_height,
            &mut scratch_x,
            &mut scratch_y,
            &mut scratch_b,
            &mut padded_scratch,
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

    // Compute totals per context (integer, matching libjxl's size_t)
    let mut totals = vec![1usize; num_contexts]; // init to 1 matching libjxl
    for ctx in 0..num_contexts {
        for &count in &histo[ctx][..num_candidates] {
            totals[ctx] += count as usize;
        }
    }

    // Compute multipliers
    // NOTE: libjxl uses size_t/size_t (integer division) for ctx_histo[val]/totals[context].
    // For count < total, integer division yields 0, so log1p(0)=0 and mul=1.0.
    // This makes the entropy-based refinement effectively a no-op for most contexts —
    // only the c3 bias for sharpness=0 has real effect. We match this exactly.
    let mut muls = vec![vec![1.0f32; num_candidates]; num_contexts];
    for ctx in 0..num_contexts {
        for ci in 0..num_candidates {
            let count = histo[ctx][ci] as usize;
            // Integer division to match libjxl's size_t / size_t
            let ratio = count / totals[ctx];
            let mut mul = 1.0 / (1.0 + c5 * (1.0 + ratio as f32).ln() / clamped_d);
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
