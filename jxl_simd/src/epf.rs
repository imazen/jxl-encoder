// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! SIMD-accelerated Edge-Preserving Filter (EPF) steps.
//!
//! EPF is a bilateral filter that smooths flat regions while preserving edges.
//! It operates on 3-channel XYB pixel data. Each step processes all pixels,
//! computing a weighted average of neighbors where the weights depend on the
//! sum of absolute differences (SAD) between center and neighbor pixels,
//! scaled by a per-block inverse-sigma parameter.
//!
//! Two steps are provided:
//! - `epf_step2`: Lightest — 4 cross neighbors, 1x1 SAD
//! - `epf_step1`: Medium — 4 cross neighbors, 3x3-plus SAD

/// Channel importance weights for SAD computation (from libjxl epf.h).
const EPF_CHANNEL_SCALE: [f32; 3] = [40.0, 5.0, 3.5];

// ============================================================================
// EPF Step 2: 3x3 cross kernel with 1x1 SAD
// ============================================================================

/// Apply EPF Step 2 to 3-channel XYB planes.
///
/// Uses a 3×3 cross kernel (4 cardinal neighbors) with single-pixel SAD
/// for weight computation. This is the lightest EPF step.
///
/// # Parameters
/// - `in_x/y/b`: Input XYB planes (flat arrays, stride = width)
/// - `out_x/y/b`: Output XYB planes (must be same size as input)
/// - `inv_sigma`: Per-block inverse sigma map (xsize_blocks × ysize_blocks)
/// - `xsize_blocks`: Number of 8×8 blocks horizontally
/// - `width`: Pixel width (= xsize_blocks × 8)
/// - `height`: Pixel height (= ysize_blocks × 8)
/// - `sigma_scale`: Base sigma multiplier for this step
/// - `border_sigma_mul`: Multiplier for block-edge pixels (typically 2/3)
#[allow(clippy::too_many_arguments)]
pub fn epf_step2(
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    out_x: &mut [f32],
    out_y: &mut [f32],
    out_b: &mut [f32],
    inv_sigma: &[f32],
    xsize_blocks: usize,
    width: usize,
    height: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            epf_step2_avx2(
                token,
                in_x,
                in_y,
                in_b,
                out_x,
                out_y,
                out_b,
                inv_sigma,
                xsize_blocks,
                width,
                height,
                sigma_scale,
                border_sigma_mul,
            );
            return;
        }
    }

    epf_step2_scalar(
        in_x,
        in_y,
        in_b,
        out_x,
        out_y,
        out_b,
        inv_sigma,
        xsize_blocks,
        width,
        height,
        sigma_scale,
        border_sigma_mul,
    );
}

#[allow(clippy::too_many_arguments)]
fn epf_step2_scalar(
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    out_x: &mut [f32],
    out_y: &mut [f32],
    out_b: &mut [f32],
    inv_sigma: &[f32],
    xsize_blocks: usize,
    width: usize,
    height: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    let ins = [in_x, in_y, in_b];
    let block_dim = 8;

    for py in 0..height {
        let by = py / block_dim;
        for px in 0..width {
            let bx = px / block_dim;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];
            let pidx = py * width + px;

            if is == 0.0 {
                out_x[pidx] = in_x[pidx];
                out_y[pidx] = in_y[pidx];
                out_b[pidx] = in_b[pidx];
                continue;
            }

            let at_border_x = px % block_dim == 0 || px % block_dim == block_dim - 1;
            let at_border_y = py % block_dim == 0 || py % block_dim == block_dim - 1;
            let bm = if at_border_x || at_border_y {
                border_sigma_mul
            } else {
                1.0
            };
            let eff_is = is * sigma_scale * bm;

            let cx = px as isize;
            let cy = py as isize;

            let mut total_w = 1.0f32;
            let mut sums = [in_x[pidx], in_y[pidx], in_b[pidx]];

            // 4 cross neighbors
            for &(dy, dx) in &[(0isize, -1isize), (-1, 0), (1, 0), (0, 1)] {
                let nx = (cx + dx).clamp(0, width as isize - 1) as usize;
                let ny = (cy + dy).clamp(0, height as isize - 1) as usize;
                let nidx = ny * width + nx;

                let mut sad = 0.0f32;
                for c in 0..3 {
                    sad += (ins[c][pidx] - ins[c][nidx]).abs() * EPF_CHANNEL_SCALE[c];
                }

                let w = (sad * eff_is + 1.0).max(0.0);
                total_w += w;
                sums[0] += w * in_x[nidx];
                sums[1] += w * in_y[nidx];
                sums[2] += w * in_b[nidx];
            }

            let inv_tw = 1.0 / total_w;
            out_x[pidx] = sums[0] * inv_tw;
            out_y[pidx] = sums[1] * inv_tw;
            out_b[pidx] = sums[2] * inv_tw;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn epf_step2_avx2(
    token: archmage::X64V3Token,
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    out_x: &mut [f32],
    out_y: &mut [f32],
    out_b: &mut [f32],
    inv_sigma: &[f32],
    xsize_blocks: usize,
    width: usize,
    height: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    use magetypes::simd::f32x8;

    // Minimum size for SIMD path: need at least 3 blocks wide (1 scalar + 1 SIMD + 1 scalar)
    if xsize_blocks < 3 || height < 2 {
        epf_step2_scalar(
            in_x,
            in_y,
            in_b,
            out_x,
            out_y,
            out_b,
            inv_sigma,
            xsize_blocks,
            width,
            height,
            sigma_scale,
            border_sigma_mul,
        );
        return;
    }

    let ch_w_x = f32x8::splat(token, EPF_CHANNEL_SCALE[0]);
    let ch_w_y = f32x8::splat(token, EPF_CHANNEL_SCALE[1]);
    let ch_w_b = f32x8::splat(token, EPF_CHANNEL_SCALE[2]);
    let one = f32x8::splat(token, 1.0);
    let zero_v = f32x8::zero(token);

    // Border multiplier vectors for block-aligned 8-pixel chunks.
    // Within a block, positions 0 and 7 are at block edges.
    let sm_interior = f32x8::from_array(
        token,
        [
            sigma_scale * border_sigma_mul, // pos 0: block edge
            sigma_scale,
            sigma_scale,
            sigma_scale,
            sigma_scale,
            sigma_scale,
            sigma_scale,
            sigma_scale * border_sigma_mul, // pos 7: block edge
        ],
    );
    let sm_border_row = f32x8::splat(token, sigma_scale * border_sigma_mul);

    let block_dim = 8usize;
    let ins = [in_x, in_y, in_b];

    for py in 0..height {
        let by = py / block_dim;
        let is_border_row = py % block_dim == 0 || py % block_dim == block_dim - 1;
        let sm_vec = if is_border_row {
            sm_border_row
        } else {
            sm_interior
        };

        let r0 = py * width;
        let rt = if py > 0 { (py - 1) * width } else { 0 };
        let rb = if py + 1 < height {
            (py + 1) * width
        } else {
            (height - 1) * width
        };

        // Scalar: first block (bx=0) — left neighbor clamping needed
        {
            let bx = 0;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];
            for lx in 0..block_dim {
                let px = lx;
                let pidx = r0 + px;
                if is == 0.0 {
                    out_x[pidx] = in_x[pidx];
                    out_y[pidx] = in_y[pidx];
                    out_b[pidx] = in_b[pidx];
                    continue;
                }
                let at_border_x = lx == 0 || lx == block_dim - 1;
                let bm = if at_border_x || is_border_row {
                    border_sigma_mul
                } else {
                    1.0
                };
                let eff_is = is * sigma_scale * bm;

                let mut total_w = 1.0f32;
                let mut sums = [in_x[pidx], in_y[pidx], in_b[pidx]];

                for &(dy, dx) in &[(0isize, -1isize), (-1, 0), (1, 0), (0, 1)] {
                    let nx = (px as isize + dx).clamp(0, width as isize - 1) as usize;
                    let ny = (py as isize + dy).clamp(0, height as isize - 1) as usize;
                    let nidx = ny * width + nx;

                    let mut sad = 0.0f32;
                    for c in 0..3 {
                        sad += (ins[c][pidx] - ins[c][nidx]).abs() * EPF_CHANNEL_SCALE[c];
                    }
                    let w = (sad * eff_is + 1.0).max(0.0);
                    total_w += w;
                    sums[0] += w * in_x[nidx];
                    sums[1] += w * in_y[nidx];
                    sums[2] += w * in_b[nidx];
                }

                let inv_tw = 1.0 / total_w;
                out_x[pidx] = sums[0] * inv_tw;
                out_y[pidx] = sums[1] * inv_tw;
                out_b[pidx] = sums[2] * inv_tw;
            }
        }

        // SIMD: interior blocks (bx=1..xsize_blocks-1)
        for bx in 1..xsize_blocks - 1 {
            let x = bx * block_dim;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];

            let base = r0 + x;

            if is == 0.0 {
                // Copy 8 pixels per channel
                out_x[base..base + 8].copy_from_slice(&in_x[base..base + 8]);
                out_y[base..base + 8].copy_from_slice(&in_y[base..base + 8]);
                out_b[base..base + 8].copy_from_slice(&in_b[base..base + 8]);
                continue;
            }

            let is_v = f32x8::splat(token, is);
            let eff_is = is_v * sm_vec;

            // Load center pixels
            let cx = f32x8::from_slice(token, &in_x[base..]);
            let cy = f32x8::from_slice(token, &in_y[base..]);
            let cb = f32x8::from_slice(token, &in_b[base..]);

            let mut sum_x = cx;
            let mut sum_y = cy;
            let mut sum_b = cb;
            let mut total_w = one;

            // Top neighbor (dy=-1, dx=0)
            let top = rt + x;
            let nx = f32x8::from_slice(token, &in_x[top..]);
            let ny = f32x8::from_slice(token, &in_y[top..]);
            let nb = f32x8::from_slice(token, &in_b[top..]);
            let sad =
                (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
            let w = (sad * eff_is + one).max(zero_v);
            total_w += w;
            sum_x = w.mul_add(nx, sum_x);
            sum_y = w.mul_add(ny, sum_y);
            sum_b = w.mul_add(nb, sum_b);

            // Bottom neighbor (dy=+1, dx=0)
            let bot = rb + x;
            let nx = f32x8::from_slice(token, &in_x[bot..]);
            let ny = f32x8::from_slice(token, &in_y[bot..]);
            let nb = f32x8::from_slice(token, &in_b[bot..]);
            let sad =
                (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
            let w = (sad * eff_is + one).max(zero_v);
            total_w += w;
            sum_x = w.mul_add(nx, sum_x);
            sum_y = w.mul_add(ny, sum_y);
            sum_b = w.mul_add(nb, sum_b);

            // Left neighbor (dy=0, dx=-1)
            let left = r0 + x - 1; // safe: bx >= 1, so x >= 8
            let nx = f32x8::from_slice(token, &in_x[left..]);
            let ny = f32x8::from_slice(token, &in_y[left..]);
            let nb = f32x8::from_slice(token, &in_b[left..]);
            let sad =
                (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
            let w = (sad * eff_is + one).max(zero_v);
            total_w += w;
            sum_x = w.mul_add(nx, sum_x);
            sum_y = w.mul_add(ny, sum_y);
            sum_b = w.mul_add(nb, sum_b);

            // Right neighbor (dy=0, dx=+1)
            let right = r0 + x + 1; // safe: bx < xsize_blocks-1, so x+8 < width
            let nx = f32x8::from_slice(token, &in_x[right..]);
            let ny = f32x8::from_slice(token, &in_y[right..]);
            let nb = f32x8::from_slice(token, &in_b[right..]);
            let sad =
                (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
            let w = (sad * eff_is + one).max(zero_v);
            total_w += w;
            sum_x = w.mul_add(nx, sum_x);
            sum_y = w.mul_add(ny, sum_y);
            sum_b = w.mul_add(nb, sum_b);

            // Normalize and store
            let inv_tw = total_w.recip();
            let out_arr_x: &mut [f32; 8] = (&mut out_x[base..base + 8]).try_into().unwrap();
            let out_arr_y: &mut [f32; 8] = (&mut out_y[base..base + 8]).try_into().unwrap();
            let out_arr_b: &mut [f32; 8] = (&mut out_b[base..base + 8]).try_into().unwrap();
            (sum_x * inv_tw).store(out_arr_x);
            (sum_y * inv_tw).store(out_arr_y);
            (sum_b * inv_tw).store(out_arr_b);
        }

        // Scalar: last block (bx=xsize_blocks-1) — right neighbor clamping needed
        {
            let bx = xsize_blocks - 1;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];
            for lx in 0..block_dim {
                let px = bx * block_dim + lx;
                let pidx = r0 + px;
                if is == 0.0 {
                    out_x[pidx] = in_x[pidx];
                    out_y[pidx] = in_y[pidx];
                    out_b[pidx] = in_b[pidx];
                    continue;
                }
                let at_border_x = lx == 0 || lx == block_dim - 1;
                let bm = if at_border_x || is_border_row {
                    border_sigma_mul
                } else {
                    1.0
                };
                let eff_is = is * sigma_scale * bm;

                let mut total_w = 1.0f32;
                let mut sums = [in_x[pidx], in_y[pidx], in_b[pidx]];

                for &(dy, dx) in &[(0isize, -1isize), (-1, 0), (1, 0), (0, 1)] {
                    let nx = (px as isize + dx).clamp(0, width as isize - 1) as usize;
                    let ny = (py as isize + dy).clamp(0, height as isize - 1) as usize;
                    let nidx = ny * width + nx;

                    let mut sad = 0.0f32;
                    for c in 0..3 {
                        sad += (ins[c][pidx] - ins[c][nidx]).abs() * EPF_CHANNEL_SCALE[c];
                    }
                    let w = (sad * eff_is + 1.0).max(0.0);
                    total_w += w;
                    sums[0] += w * in_x[nidx];
                    sums[1] += w * in_y[nidx];
                    sums[2] += w * in_b[nidx];
                }

                let inv_tw = 1.0 / total_w;
                out_x[pidx] = sums[0] * inv_tw;
                out_y[pidx] = sums[1] * inv_tw;
                out_b[pidx] = sums[2] * inv_tw;
            }
        }
    }
}

// ============================================================================
// EPF Step 1: 3x3 cross kernel with 3x3-plus SAD
// ============================================================================

/// Apply EPF Step 1 to 3-channel XYB planes.
///
/// Uses a 3×3 cross kernel (4 cardinal neighbors) with 3×3 plus-pattern SAD.
/// The SAD for each neighbor is the sum over 5 positions in a plus pattern,
/// comparing center vs neighbor at each offset.
///
/// Same parameters as `epf_step2`.
#[allow(clippy::too_many_arguments)]
pub fn epf_step1(
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    out_x: &mut [f32],
    out_y: &mut [f32],
    out_b: &mut [f32],
    inv_sigma: &[f32],
    xsize_blocks: usize,
    width: usize,
    height: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            epf_step1_avx2(
                token,
                in_x,
                in_y,
                in_b,
                out_x,
                out_y,
                out_b,
                inv_sigma,
                xsize_blocks,
                width,
                height,
                sigma_scale,
                border_sigma_mul,
            );
            return;
        }
    }

    epf_step1_scalar(
        in_x,
        in_y,
        in_b,
        out_x,
        out_y,
        out_b,
        inv_sigma,
        xsize_blocks,
        width,
        height,
        sigma_scale,
        border_sigma_mul,
    );
}

/// Scalar helper: compute 3x3-plus SAD between center at (cx,cy) and neighbor at (nx,ny).
#[inline(always)]
fn sad_3x3_plus_scalar(
    planes: [&[f32]; 3],
    cx: usize,
    cy: usize,
    nx: usize,
    ny: usize,
    width: usize,
    height: usize,
) -> f32 {
    const PLUS: [(isize, isize); 5] = [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)];
    let mut sad = 0.0f32;
    for &(dy, dx) in &PLUS {
        let cpx = (cx as isize + dx).clamp(0, width as isize - 1) as usize;
        let cpy = (cy as isize + dy).clamp(0, height as isize - 1) as usize;
        let npx = (nx as isize + dx).clamp(0, width as isize - 1) as usize;
        let npy = (ny as isize + dy).clamp(0, height as isize - 1) as usize;
        for c in 0..3 {
            let cv = planes[c][cpy * width + cpx];
            let nv = planes[c][npy * width + npx];
            sad += (cv - nv).abs() * EPF_CHANNEL_SCALE[c];
        }
    }
    sad
}

#[allow(clippy::too_many_arguments)]
fn epf_step1_scalar(
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    out_x: &mut [f32],
    out_y: &mut [f32],
    out_b: &mut [f32],
    inv_sigma: &[f32],
    xsize_blocks: usize,
    width: usize,
    height: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    let block_dim = 8;

    for py in 0..height {
        let by = py / block_dim;
        for px in 0..width {
            let bx = px / block_dim;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];
            let pidx = py * width + px;

            if is == 0.0 {
                out_x[pidx] = in_x[pidx];
                out_y[pidx] = in_y[pidx];
                out_b[pidx] = in_b[pidx];
                continue;
            }

            let at_border_x = px % block_dim == 0 || px % block_dim == block_dim - 1;
            let at_border_y = py % block_dim == 0 || py % block_dim == block_dim - 1;
            let bm = if at_border_x || at_border_y {
                border_sigma_mul
            } else {
                1.0
            };
            let eff_is = is * sigma_scale * bm;

            let mut total_w = 1.0f32;
            let mut sums = [in_x[pidx], in_y[pidx], in_b[pidx]];

            // 4 cross neighbors with 3x3-plus SAD
            for &(dy, dx) in &[(0isize, -1isize), (-1, 0), (1, 0), (0, 1)] {
                let nx = (px as isize + dx).clamp(0, width as isize - 1) as usize;
                let ny = (py as isize + dy).clamp(0, height as isize - 1) as usize;
                let nidx = ny * width + nx;

                let sad = sad_3x3_plus_scalar([in_x, in_y, in_b], px, py, nx, ny, width, height);

                let w = (sad * eff_is + 1.0).max(0.0);
                total_w += w;
                sums[0] += w * in_x[nidx];
                sums[1] += w * in_y[nidx];
                sums[2] += w * in_b[nidx];
            }

            let inv_tw = 1.0 / total_w;
            out_x[pidx] = sums[0] * inv_tw;
            out_y[pidx] = sums[1] * inv_tw;
            out_b[pidx] = sums[2] * inv_tw;
        }
    }
}

/// Compute 3x3-plus SAD for 8 pixels using SIMD.
///
/// Compares the plus pattern centered on 8 center pixels (at row offsets c_*)
/// with the plus pattern centered on 8 neighbor pixels (at row offsets n_*).
///
/// The plus pattern: (0,0), (-1,0), (0,-1), (1,0), (0,1)
///
/// # Safety invariant (caller must guarantee):
/// - All row offsets + x ± 1 are within bounds of the input slices
/// - All row offsets + x + 8 are within bounds (for SIMD loads)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn sad_3x3_plus_simd(
    token: archmage::X64V3Token,
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    x: usize,
    // Center row offsets for the 5 plus positions
    c_r0: usize,  // center row (y)
    c_rm1: usize, // row y-1
    c_rp1: usize, // row y+1
    // Neighbor row offsets for the 5 plus positions
    n_r0: usize,  // neighbor row (y + ndy)
    n_rm1: usize, // neighbor row y + ndy - 1
    n_rp1: usize, // neighbor row y + ndy + 1
    // Horizontal offset for neighbor
    ndx: isize,
    ch_w_x: magetypes::simd::f32x8,
    ch_w_y: magetypes::simd::f32x8,
    ch_w_b: magetypes::simd::f32x8,
) -> magetypes::simd::f32x8 {
    use magetypes::simd::f32x8;

    let nx = (x as isize + ndx) as usize;

    // Plus pattern: (0,0), (-1,0), (0,-1), (1,0), (0,1)
    // Position (0,0): center row, x vs neighbor row, nx
    let mut sad = {
        let c0x = f32x8::from_slice(token, &in_x[c_r0 + x..]);
        let c0y = f32x8::from_slice(token, &in_y[c_r0 + x..]);
        let c0b = f32x8::from_slice(token, &in_b[c_r0 + x..]);
        let n0x = f32x8::from_slice(token, &in_x[n_r0 + nx..]);
        let n0y = f32x8::from_slice(token, &in_y[n_r0 + nx..]);
        let n0b = f32x8::from_slice(token, &in_b[n_r0 + nx..]);
        (c0x - n0x).abs() * ch_w_x + (c0y - n0y).abs() * ch_w_y + (c0b - n0b).abs() * ch_w_b
    };

    // Position (-1,0): same rows, x-1 vs nx-1
    {
        let c1x = f32x8::from_slice(token, &in_x[c_r0 + x - 1..]);
        let c1y = f32x8::from_slice(token, &in_y[c_r0 + x - 1..]);
        let c1b = f32x8::from_slice(token, &in_b[c_r0 + x - 1..]);
        let n1x = f32x8::from_slice(token, &in_x[n_r0 + nx - 1..]);
        let n1y = f32x8::from_slice(token, &in_y[n_r0 + nx - 1..]);
        let n1b = f32x8::from_slice(token, &in_b[n_r0 + nx - 1..]);
        sad = sad
            + (c1x - n1x).abs() * ch_w_x
            + (c1y - n1y).abs() * ch_w_y
            + (c1b - n1b).abs() * ch_w_b;
    }

    // Position (0,-1): row y-1, x vs row ndy-1, nx
    {
        let c2x = f32x8::from_slice(token, &in_x[c_rm1 + x..]);
        let c2y = f32x8::from_slice(token, &in_y[c_rm1 + x..]);
        let c2b = f32x8::from_slice(token, &in_b[c_rm1 + x..]);
        let n2x = f32x8::from_slice(token, &in_x[n_rm1 + nx..]);
        let n2y = f32x8::from_slice(token, &in_y[n_rm1 + nx..]);
        let n2b = f32x8::from_slice(token, &in_b[n_rm1 + nx..]);
        sad = sad
            + (c2x - n2x).abs() * ch_w_x
            + (c2y - n2y).abs() * ch_w_y
            + (c2b - n2b).abs() * ch_w_b;
    }

    // Position (1,0): same rows, x+1 vs nx+1
    {
        let c3x = f32x8::from_slice(token, &in_x[c_r0 + x + 1..]);
        let c3y = f32x8::from_slice(token, &in_y[c_r0 + x + 1..]);
        let c3b = f32x8::from_slice(token, &in_b[c_r0 + x + 1..]);
        let n3x = f32x8::from_slice(token, &in_x[n_r0 + nx + 1..]);
        let n3y = f32x8::from_slice(token, &in_y[n_r0 + nx + 1..]);
        let n3b = f32x8::from_slice(token, &in_b[n_r0 + nx + 1..]);
        sad = sad
            + (c3x - n3x).abs() * ch_w_x
            + (c3y - n3y).abs() * ch_w_y
            + (c3b - n3b).abs() * ch_w_b;
    }

    // Position (0,1): row y+1, x vs row ndy+1, nx
    {
        let c4x = f32x8::from_slice(token, &in_x[c_rp1 + x..]);
        let c4y = f32x8::from_slice(token, &in_y[c_rp1 + x..]);
        let c4b = f32x8::from_slice(token, &in_b[c_rp1 + x..]);
        let n4x = f32x8::from_slice(token, &in_x[n_rp1 + nx..]);
        let n4y = f32x8::from_slice(token, &in_y[n_rp1 + nx..]);
        let n4b = f32x8::from_slice(token, &in_b[n_rp1 + nx..]);
        sad = sad
            + (c4x - n4x).abs() * ch_w_x
            + (c4y - n4y).abs() * ch_w_y
            + (c4b - n4b).abs() * ch_w_b;
    }

    sad
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn epf_step1_avx2(
    token: archmage::X64V3Token,
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    out_x: &mut [f32],
    out_y: &mut [f32],
    out_b: &mut [f32],
    inv_sigma: &[f32],
    xsize_blocks: usize,
    width: usize,
    height: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    use magetypes::simd::f32x8;

    // For step1 3x3-plus SAD, the neighbor's plus pattern extends ±2 from center
    // in both x and y. We need at least 3 blocks wide and 5 rows tall for safety.
    if xsize_blocks < 3 || height < 4 {
        epf_step1_scalar(
            in_x,
            in_y,
            in_b,
            out_x,
            out_y,
            out_b,
            inv_sigma,
            xsize_blocks,
            width,
            height,
            sigma_scale,
            border_sigma_mul,
        );
        return;
    }

    let ch_w_x = f32x8::splat(token, EPF_CHANNEL_SCALE[0]);
    let ch_w_y = f32x8::splat(token, EPF_CHANNEL_SCALE[1]);
    let ch_w_b = f32x8::splat(token, EPF_CHANNEL_SCALE[2]);
    let one = f32x8::splat(token, 1.0);
    let zero_v = f32x8::zero(token);

    let sm_interior = f32x8::from_array(
        token,
        [
            sigma_scale * border_sigma_mul,
            sigma_scale,
            sigma_scale,
            sigma_scale,
            sigma_scale,
            sigma_scale,
            sigma_scale,
            sigma_scale * border_sigma_mul,
        ],
    );
    let sm_border_row = f32x8::splat(token, sigma_scale * border_sigma_mul);

    let block_dim = 8usize;
    let h_max = height - 1;

    for py in 0..height {
        let by = py / block_dim;
        let is_border_row = py % block_dim == 0 || py % block_dim == block_dim - 1;
        let sm_vec = if is_border_row {
            sm_border_row
        } else {
            sm_interior
        };

        // Precompute clamped row offsets for the plus pattern around center and neighbors.
        // Center's plus needs rows y-1, y, y+1.
        // Neighbor at dy=-1 needs rows y-2, y-1, y (for its plus).
        // Neighbor at dy=+1 needs rows y, y+1, y+2 (for its plus).
        let r_m2 = py.saturating_sub(2) * width;
        let r_m1 = py.saturating_sub(1) * width;
        let r_0 = py * width;
        let r_p1 = (py + 1).min(h_max) * width;
        let r_p2 = (py + 2).min(h_max) * width;

        // Scalar: first block (left edge)
        scalar_step1_block(
            in_x,
            in_y,
            in_b,
            out_x,
            out_y,
            out_b,
            inv_sigma,
            xsize_blocks,
            width,
            height,
            sigma_scale,
            border_sigma_mul,
            is_border_row,
            py,
            by,
            0,
        );

        // SIMD: interior blocks
        for bx in 1..xsize_blocks - 1 {
            let x = bx * block_dim;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];

            let base = r_0 + x;

            if is == 0.0 {
                out_x[base..base + 8].copy_from_slice(&in_x[base..base + 8]);
                out_y[base..base + 8].copy_from_slice(&in_y[base..base + 8]);
                out_b[base..base + 8].copy_from_slice(&in_b[base..base + 8]);
                continue;
            }

            let is_v = f32x8::splat(token, is);
            let eff_is = is_v * sm_vec;

            // Load center pixels (for weighted sum)
            let cx = f32x8::from_slice(token, &in_x[base..]);
            let cy = f32x8::from_slice(token, &in_y[base..]);
            let cb = f32x8::from_slice(token, &in_b[base..]);

            let mut sum_x = cx;
            let mut sum_y = cy;
            let mut sum_b = cb;
            let mut total_w = one;

            // Neighbor: top (dx=0, dy=-1)
            // Center plus rows: r_m1, r_m2(for center's y-1), r_0(for center's y+1)
            //   Wait - center's plus: (y, x), (y, x-1), (y-1, x), (y, x+1), (y+1, x)
            //   So center plus uses rows: r_m1, r_0, r_p1
            // Neighbor at (x, y-1): plus uses (y-1, x), (y-1, x-1), (y-2, x), (y-1, x+1), (y, x)
            //   So neighbor plus uses rows: r_m2, r_m1, r_0
            {
                let sad = sad_3x3_plus_simd(
                    token, in_x, in_y, in_b, x, r_0, r_m1, r_p1, // center rows
                    r_m1, r_m2, r_0, // neighbor rows (shifted up by 1)
                    0,   // ndx = 0
                    ch_w_x, ch_w_y, ch_w_b,
                );
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                let nx = f32x8::from_slice(token, &in_x[r_m1 + x..]);
                let ny = f32x8::from_slice(token, &in_y[r_m1 + x..]);
                let nb = f32x8::from_slice(token, &in_b[r_m1 + x..]);
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);
            }

            // Neighbor: bottom (dx=0, dy=+1)
            // Neighbor at (x, y+1): plus uses (y+1, x), (y+1, x-1), (y, x), (y+1, x+1), (y+2, x)
            //   Neighbor plus rows: r_0, r_p1, r_p2
            {
                let sad = sad_3x3_plus_simd(
                    token, in_x, in_y, in_b, x, r_0, r_m1, r_p1, // center rows
                    r_p1, r_0, r_p2, // neighbor rows (shifted down by 1)
                    0,    // ndx = 0
                    ch_w_x, ch_w_y, ch_w_b,
                );
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                let nx = f32x8::from_slice(token, &in_x[r_p1 + x..]);
                let ny = f32x8::from_slice(token, &in_y[r_p1 + x..]);
                let nb = f32x8::from_slice(token, &in_b[r_p1 + x..]);
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);
            }

            // Neighbor: left (dx=-1, dy=0)
            // Neighbor at (x-1, y): plus uses (y, x-1), (y, x-2), (y-1, x-1), (y, x), (y+1, x-1)
            //   Neighbor plus rows: r_m1, r_0, r_p1 (same as center!)
            //   ndx = -1
            {
                let sad = sad_3x3_plus_simd(
                    token, in_x, in_y, in_b, x, r_0, r_m1, r_p1, // center rows
                    r_0, r_m1, r_p1, // neighbor rows (same, just shifted x)
                    -1,   // ndx = -1
                    ch_w_x, ch_w_y, ch_w_b,
                );
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                let nx = f32x8::from_slice(token, &in_x[r_0 + x - 1..]);
                let ny = f32x8::from_slice(token, &in_y[r_0 + x - 1..]);
                let nb = f32x8::from_slice(token, &in_b[r_0 + x - 1..]);
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);
            }

            // Neighbor: right (dx=+1, dy=0)
            {
                let sad = sad_3x3_plus_simd(
                    token, in_x, in_y, in_b, x, r_0, r_m1, r_p1, r_0, r_m1, r_p1,
                    1, // ndx = +1
                    ch_w_x, ch_w_y, ch_w_b,
                );
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                let nx = f32x8::from_slice(token, &in_x[r_0 + x + 1..]);
                let ny = f32x8::from_slice(token, &in_y[r_0 + x + 1..]);
                let nb = f32x8::from_slice(token, &in_b[r_0 + x + 1..]);
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);
            }

            // Normalize and store
            let inv_tw = total_w.recip();
            let out_arr_x: &mut [f32; 8] = (&mut out_x[base..base + 8]).try_into().unwrap();
            let out_arr_y: &mut [f32; 8] = (&mut out_y[base..base + 8]).try_into().unwrap();
            let out_arr_b: &mut [f32; 8] = (&mut out_b[base..base + 8]).try_into().unwrap();
            (sum_x * inv_tw).store(out_arr_x);
            (sum_y * inv_tw).store(out_arr_y);
            (sum_b * inv_tw).store(out_arr_b);
        }

        // Scalar: last block (right edge)
        scalar_step1_block(
            in_x,
            in_y,
            in_b,
            out_x,
            out_y,
            out_b,
            inv_sigma,
            xsize_blocks,
            width,
            height,
            sigma_scale,
            border_sigma_mul,
            is_border_row,
            py,
            by,
            xsize_blocks - 1,
        );
    }
}

/// Process one 8-pixel block of step1 using scalar code (for edge blocks).
#[allow(clippy::too_many_arguments)]
fn scalar_step1_block(
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    out_x: &mut [f32],
    out_y: &mut [f32],
    out_b: &mut [f32],
    inv_sigma: &[f32],
    xsize_blocks: usize,
    width: usize,
    height: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
    is_border_row: bool,
    py: usize,
    by: usize,
    bx: usize,
) {
    let block_dim = 8;
    let ins = [in_x, in_y, in_b];
    let sigma_idx = by * xsize_blocks + bx;
    let is = inv_sigma[sigma_idx];
    let r0 = py * width;

    for lx in 0..block_dim {
        let px = bx * block_dim + lx;
        let pidx = r0 + px;

        if is == 0.0 {
            out_x[pidx] = in_x[pidx];
            out_y[pidx] = in_y[pidx];
            out_b[pidx] = in_b[pidx];
            continue;
        }

        let at_border_x = lx == 0 || lx == block_dim - 1;
        let bm = if at_border_x || is_border_row {
            border_sigma_mul
        } else {
            1.0
        };
        let eff_is = is * sigma_scale * bm;

        let mut total_w = 1.0f32;
        let mut sums = [in_x[pidx], in_y[pidx], in_b[pidx]];

        for &(dy, dx) in &[(0isize, -1isize), (-1, 0), (1, 0), (0, 1)] {
            let nx = (px as isize + dx).clamp(0, width as isize - 1) as usize;
            let ny = (py as isize + dy).clamp(0, height as isize - 1) as usize;
            let nidx = ny * width + nx;

            let sad = sad_3x3_plus_scalar(ins, px, py, nx, ny, width, height);

            let w = (sad * eff_is + 1.0).max(0.0);
            total_w += w;
            sums[0] += w * in_x[nidx];
            sums[1] += w * in_y[nidx];
            sums[2] += w * in_b[nidx];
        }

        let inv_tw = 1.0 / total_w;
        out_x[pidx] = sums[0] * inv_tw;
        out_y[pidx] = sums[1] * inv_tw;
        out_b[pidx] = sums[2] * inv_tw;
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// EPF step2 on constant input should produce constant output.
    #[test]
    fn test_epf_step2_constant_passthrough() {
        let w = 32;
        let h = 32;
        let val = 0.5f32;
        let in_x = vec![val; w * h];
        let in_y = vec![val; w * h];
        let in_b = vec![val; w * h];
        let mut out_x = vec![0.0f32; w * h];
        let mut out_y = vec![0.0f32; w * h];
        let mut out_b = vec![0.0f32; w * h];

        let xsb = w / 8;
        let ysb = h / 8;
        let inv_sigma = vec![-1.0f32; xsb * ysb]; // nonzero sigma so filtering runs

        epf_step2(
            &in_x,
            &in_y,
            &in_b,
            &mut out_x,
            &mut out_y,
            &mut out_b,
            &inv_sigma,
            xsb,
            w,
            h,
            6.5 * 1.65, // EPF_PASS2_SIGMA_SCALE * 1.65
            2.0 / 3.0,
        );

        for i in 0..w * h {
            assert!(
                (out_x[i] - val).abs() < 1e-5,
                "step2 X: i={} got {} expected {}",
                i,
                out_x[i],
                val,
            );
            assert!(
                (out_y[i] - val).abs() < 1e-5,
                "step2 Y: i={} got {} expected {}",
                i,
                out_y[i],
                val,
            );
            assert!(
                (out_b[i] - val).abs() < 1e-5,
                "step2 B: i={} got {} expected {}",
                i,
                out_b[i],
                val,
            );
        }
    }

    /// EPF step1 on constant input should produce constant output.
    #[test]
    fn test_epf_step1_constant_passthrough() {
        let w = 32;
        let h = 32;
        let val = 0.3f32;
        let in_x = vec![val; w * h];
        let in_y = vec![val; w * h];
        let in_b = vec![val; w * h];
        let mut out_x = vec![0.0f32; w * h];
        let mut out_y = vec![0.0f32; w * h];
        let mut out_b = vec![0.0f32; w * h];

        let xsb = w / 8;
        let ysb = h / 8;
        let inv_sigma = vec![-1.0f32; xsb * ysb];

        epf_step1(
            &in_x,
            &in_y,
            &in_b,
            &mut out_x,
            &mut out_y,
            &mut out_b,
            &inv_sigma,
            xsb,
            w,
            h,
            1.65,
            2.0 / 3.0,
        );

        for i in 0..w * h {
            assert!(
                (out_x[i] - val).abs() < 1e-5,
                "step1 X: i={} got {} expected {}",
                i,
                out_x[i],
                val,
            );
        }
    }

    /// SIMD step2 must match scalar step2 on varied input.
    #[test]
    fn test_epf_step2_simd_matches_scalar() {
        let w = 48; // 6 blocks wide (need ≥3 for SIMD path)
        let h = 32;
        let n = w * h;

        // Generate varied input
        let mut in_x = vec![0.0f32; n];
        let mut in_y = vec![0.0f32; n];
        let mut in_b = vec![0.0f32; n];
        for i in 0..n {
            let x = (i % w) as f32;
            let y = (i / w) as f32;
            in_x[i] = (x * 0.01 + y * 0.007).sin() * 0.5 + 0.5;
            in_y[i] = (x * 0.013 + y * 0.011).cos() * 0.3 + 0.4;
            in_b[i] = (x * 0.009 + y * 0.015).sin() * 0.2 + 0.3;
        }

        let xsb = w / 8;
        let ysb = h / 8;
        // Varied inv_sigma
        let mut inv_sigma = vec![0.0f32; xsb * ysb];
        for i in 0..inv_sigma.len() {
            inv_sigma[i] = if i % 3 == 0 {
                0.0
            } else {
                -0.5 - (i as f32) * 0.1
            };
        }

        let sigma_scale = 6.5 * 1.65;
        let border_mul = 2.0 / 3.0;

        // Scalar reference
        let mut ref_x = vec![0.0f32; n];
        let mut ref_y = vec![0.0f32; n];
        let mut ref_b = vec![0.0f32; n];
        epf_step2_scalar(
            &in_x,
            &in_y,
            &in_b,
            &mut ref_x,
            &mut ref_y,
            &mut ref_b,
            &inv_sigma,
            xsb,
            w,
            h,
            sigma_scale,
            border_mul,
        );

        // SIMD (via dispatch)
        let mut out_x = vec![0.0f32; n];
        let mut out_y = vec![0.0f32; n];
        let mut out_b = vec![0.0f32; n];
        epf_step2(
            &in_x,
            &in_y,
            &in_b,
            &mut out_x,
            &mut out_y,
            &mut out_b,
            &inv_sigma,
            xsb,
            w,
            h,
            sigma_scale,
            border_mul,
        );

        let mut max_err = 0.0f32;
        for i in 0..n {
            let ex = (out_x[i] - ref_x[i]).abs();
            let ey = (out_y[i] - ref_y[i]).abs();
            let eb = (out_b[i] - ref_b[i]).abs();
            let err = ex.max(ey).max(eb);
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 1e-4,
                "step2 mismatch at pixel {}: SIMD=({},{},{}) scalar=({},{},{}) err={}",
                i,
                out_x[i],
                out_y[i],
                out_b[i],
                ref_x[i],
                ref_y[i],
                ref_b[i],
                err,
            );
        }
    }

    /// SIMD step1 must match scalar step1 on varied input.
    #[test]
    fn test_epf_step1_simd_matches_scalar() {
        let w = 48;
        let h = 32;
        let n = w * h;

        let mut in_x = vec![0.0f32; n];
        let mut in_y = vec![0.0f32; n];
        let mut in_b = vec![0.0f32; n];
        for i in 0..n {
            let x = (i % w) as f32;
            let y = (i / w) as f32;
            in_x[i] = (x * 0.01 + y * 0.007).sin() * 0.5 + 0.5;
            in_y[i] = (x * 0.013 + y * 0.011).cos() * 0.3 + 0.4;
            in_b[i] = (x * 0.009 + y * 0.015).sin() * 0.2 + 0.3;
        }

        let xsb = w / 8;
        let ysb = h / 8;
        let mut inv_sigma = vec![0.0f32; xsb * ysb];
        for i in 0..inv_sigma.len() {
            inv_sigma[i] = if i % 3 == 0 {
                0.0
            } else {
                -0.5 - (i as f32) * 0.1
            };
        }

        let sigma_scale = 1.65;
        let border_mul = 2.0 / 3.0;

        let mut ref_x = vec![0.0f32; n];
        let mut ref_y = vec![0.0f32; n];
        let mut ref_b = vec![0.0f32; n];
        epf_step1_scalar(
            &in_x,
            &in_y,
            &in_b,
            &mut ref_x,
            &mut ref_y,
            &mut ref_b,
            &inv_sigma,
            xsb,
            w,
            h,
            sigma_scale,
            border_mul,
        );

        let mut out_x = vec![0.0f32; n];
        let mut out_y = vec![0.0f32; n];
        let mut out_b = vec![0.0f32; n];
        epf_step1(
            &in_x,
            &in_y,
            &in_b,
            &mut out_x,
            &mut out_y,
            &mut out_b,
            &inv_sigma,
            xsb,
            w,
            h,
            sigma_scale,
            border_mul,
        );

        let mut max_err = 0.0f32;
        for i in 0..n {
            let ex = (out_x[i] - ref_x[i]).abs();
            let ey = (out_y[i] - ref_y[i]).abs();
            let eb = (out_b[i] - ref_b[i]).abs();
            let err = ex.max(ey).max(eb);
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 1e-4,
                "step1 mismatch at pixel {}: SIMD=({},{},{}) scalar=({},{},{}) err={}",
                i,
                out_x[i],
                out_y[i],
                out_b[i],
                ref_x[i],
                ref_y[i],
                ref_b[i],
                err,
            );
        }
    }

    /// EPF with inv_sigma=0 should be a no-op (copy input to output).
    #[test]
    fn test_epf_zero_sigma_noop() {
        let w = 32;
        let h = 16;
        let n = w * h;
        let in_x: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let in_y: Vec<f32> = (0..n).map(|i| i as f32 * 0.002 + 1.0).collect();
        let in_b: Vec<f32> = (0..n).map(|i| i as f32 * 0.003 + 2.0).collect();
        let mut out_x = vec![0.0f32; n];
        let mut out_y = vec![0.0f32; n];
        let mut out_b = vec![0.0f32; n];

        let xsb = w / 8;
        let ysb = h / 8;
        let inv_sigma = vec![0.0f32; xsb * ysb]; // all zero

        epf_step2(
            &in_x,
            &in_y,
            &in_b,
            &mut out_x,
            &mut out_y,
            &mut out_b,
            &inv_sigma,
            xsb,
            w,
            h,
            6.5 * 1.65,
            2.0 / 3.0,
        );

        for i in 0..n {
            assert_eq!(out_x[i], in_x[i], "noop X at {}", i);
            assert_eq!(out_y[i], in_y[i], "noop Y at {}", i);
            assert_eq!(out_b[i], in_b[i], "noop B at {}", i);
        }
    }
}
