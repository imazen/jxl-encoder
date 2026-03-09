// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

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

#[cfg(target_arch = "x86_64")]
use crate::load_f32x8;
#[cfg(target_arch = "x86_64")]
use crate::slice_from;

/// Pad a single channel plane with edge replication.
///
/// Returns a new buffer of `(width + 2*pad) x (height + 2*pad)` with
/// stride = `width + 2*pad`. Edge pixels are replicated into the padding region
/// so that kernels never need bounds checks.
pub fn pad_plane(plane: &[f32], width: usize, height: usize, pad: usize) -> alloc::vec::Vec<f32> {
    let stride = width + 2 * pad;
    let mut out = crate::vec_f32_dirty(stride * (height + 2 * pad));

    // Step 1: Copy each image row into the interior of the padded buffer.
    for y in 0..height {
        let src_off = y * width;
        let dst_off = (y + pad) * stride + pad;
        out[dst_off..dst_off + width].copy_from_slice(&plane[src_off..src_off + width]);
    }

    // Step 2: Replicate left and right edges for interior rows.
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

    // Step 3: Replicate top rows (copy entire padded row including L/R padding).
    for p in 0..pad {
        let src_off = pad * stride;
        let dst_off = p * stride;
        out.copy_within(src_off..src_off + stride, dst_off);
    }

    // Step 4: Replicate bottom rows.
    for p in 0..pad {
        let src_off = (pad + height - 1) * stride;
        let dst_off = (pad + height + p) * stride;
        out.copy_within(src_off..src_off + stride, dst_off);
    }

    out
}

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
#[inline]
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
    in_stride: usize,
    pad: usize,
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
                in_stride,
                pad,
                sigma_scale,
                border_sigma_mul,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            epf_step2_neon(
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
                in_stride,
                pad,
                sigma_scale,
                border_sigma_mul,
            );
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            epf_step2_wasm128(
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
                in_stride,
                pad,
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
        in_stride,
        pad,
        sigma_scale,
        border_sigma_mul,
    );
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn epf_step2_scalar(
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
    in_stride: usize,
    pad: usize,
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
            let oidx = py * width + px;
            // Input pixel in padded buffer
            let pidx = (py + pad) * in_stride + (px + pad);

            if is == 0.0 {
                out_x[oidx] = in_x[pidx];
                out_y[oidx] = in_y[pidx];
                out_b[oidx] = in_b[pidx];
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

            // 4 cross neighbors — padding guarantees all offsets are valid
            for &(dy, dx) in &[(0isize, -1isize), (-1, 0), (1, 0), (0, 1)] {
                let nidx = ((py + pad) as isize + dy) as usize * in_stride
                    + ((px + pad) as isize + dx) as usize;

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
            out_x[oidx] = sums[0] * inv_tw;
            out_y[oidx] = sums[1] * inv_tw;
            out_b[oidx] = sums[2] * inv_tw;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn epf_step2_avx2(
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
    in_stride: usize,
    pad: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    use magetypes::simd::f32x8;

    if xsize_blocks < 1 || height < 1 {
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

    for py in 0..height {
        let by = py / block_dim;
        let is_border_row = py % block_dim == 0 || py % block_dim == block_dim - 1;
        let sm_vec = if is_border_row {
            sm_border_row
        } else {
            sm_interior
        };

        // Padded row offsets — padding guarantees all are valid
        let r0 = (py + pad) * in_stride + pad;
        let rt = (py + pad - 1) * in_stride + pad;
        let rb = (py + pad + 1) * in_stride + pad;

        let o0 = py * width;

        // Pre-slice output row
        let orow_x = &mut out_x[o0..o0 + width];
        let orow_y = &mut out_y[o0..o0 + width];
        let orow_b = &mut out_b[o0..o0 + width];

        // All blocks processed uniformly — padding handles edges
        for bx in 0..xsize_blocks {
            let x = bx * block_dim;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];

            if is == 0.0 {
                orow_x[x..x + 8].copy_from_slice(&slice_from(in_x, r0 + x)[..8]);
                orow_y[x..x + 8].copy_from_slice(&slice_from(in_y, r0 + x)[..8]);
                orow_b[x..x + 8].copy_from_slice(&slice_from(in_b, r0 + x)[..8]);
                continue;
            }

            let is_v = f32x8::splat(token, is);
            let eff_is = is_v * sm_vec;

            let cx = load_f32x8(token, in_x, r0 + x);
            let cy = load_f32x8(token, in_y, r0 + x);
            let cb = load_f32x8(token, in_b, r0 + x);

            let mut sum_x = cx;
            let mut sum_y = cy;
            let mut sum_b = cb;
            let mut total_w = one;

            // Top neighbor
            let nx = load_f32x8(token, in_x, rt + x);
            let ny = load_f32x8(token, in_y, rt + x);
            let nb = load_f32x8(token, in_b, rt + x);
            let sad =
                (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
            let w = (sad * eff_is + one).max(zero_v);
            total_w += w;
            sum_x = w.mul_add(nx, sum_x);
            sum_y = w.mul_add(ny, sum_y);
            sum_b = w.mul_add(nb, sum_b);

            // Bottom neighbor
            let nx = load_f32x8(token, in_x, rb + x);
            let ny = load_f32x8(token, in_y, rb + x);
            let nb = load_f32x8(token, in_b, rb + x);
            let sad =
                (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
            let w = (sad * eff_is + one).max(zero_v);
            total_w += w;
            sum_x = w.mul_add(nx, sum_x);
            sum_y = w.mul_add(ny, sum_y);
            sum_b = w.mul_add(nb, sum_b);

            // Left neighbor — padding guarantees x + pad - 1 >= 0
            let nx = load_f32x8(token, in_x, r0 + x - 1);
            let ny = load_f32x8(token, in_y, r0 + x - 1);
            let nb = load_f32x8(token, in_b, r0 + x - 1);
            let sad =
                (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
            let w = (sad * eff_is + one).max(zero_v);
            total_w += w;
            sum_x = w.mul_add(nx, sum_x);
            sum_y = w.mul_add(ny, sum_y);
            sum_b = w.mul_add(nb, sum_b);

            // Right neighbor — padding guarantees x + pad + 8 < in_stride
            let nx = load_f32x8(token, in_x, r0 + x + 1);
            let ny = load_f32x8(token, in_y, r0 + x + 1);
            let nb = load_f32x8(token, in_b, r0 + x + 1);
            let sad =
                (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
            let w = (sad * eff_is + one).max(zero_v);
            total_w += w;
            sum_x = w.mul_add(nx, sum_x);
            sum_y = w.mul_add(ny, sum_y);
            sum_b = w.mul_add(nb, sum_b);

            // Normalize and store
            let inv_tw = total_w.recip();
            let out_arr_x: &mut [f32; 8] = (&mut orow_x[x..x + 8]).try_into().unwrap();
            let out_arr_y: &mut [f32; 8] = (&mut orow_y[x..x + 8]).try_into().unwrap();
            let out_arr_b: &mut [f32; 8] = (&mut orow_b[x..x + 8]).try_into().unwrap();
            (sum_x * inv_tw).store(out_arr_x);
            (sum_y * inv_tw).store(out_arr_y);
            (sum_b * inv_tw).store(out_arr_b);
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
#[inline]
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
    in_stride: usize,
    pad: usize,
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
                in_stride,
                pad,
                sigma_scale,
                border_sigma_mul,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            epf_step1_neon(
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
                in_stride,
                pad,
                sigma_scale,
                border_sigma_mul,
            );
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            epf_step1_wasm128(
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
                in_stride,
                pad,
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
        in_stride,
        pad,
        sigma_scale,
        border_sigma_mul,
    );
}

/// Scalar helper: compute 3x3-plus SAD between center at (cx,cy) and neighbor at (nx,ny).
///
/// All coordinates are in padded-buffer space (already offset by pad).
/// `stride` is the padded buffer stride (width + 2*pad).
#[inline(always)]
fn sad_3x3_plus_scalar(
    planes: [&[f32]; 3],
    cx: usize,
    cy: usize,
    nx: usize,
    ny: usize,
    stride: usize,
) -> f32 {
    const PLUS: [(isize, isize); 5] = [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)];
    let mut sad = 0.0f32;
    for &(dy, dx) in &PLUS {
        let cpx = (cx as isize + dx) as usize;
        let cpy = (cy as isize + dy) as usize;
        let npx = (nx as isize + dx) as usize;
        let npy = (ny as isize + dy) as usize;
        for c in 0..3 {
            let cv = planes[c][cpy * stride + cpx];
            let nv = planes[c][npy * stride + npx];
            sad += (cv - nv).abs() * EPF_CHANNEL_SCALE[c];
        }
    }
    sad
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn epf_step1_scalar(
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
    in_stride: usize,
    pad: usize,
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
            let oidx = py * width + px;
            // Padded-buffer coordinates
            let ipx = px + pad;
            let ipy = py + pad;
            let pidx = ipy * in_stride + ipx;

            if is == 0.0 {
                out_x[oidx] = in_x[pidx];
                out_y[oidx] = in_y[pidx];
                out_b[oidx] = in_b[pidx];
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

            // 4 cross neighbors with 3x3-plus SAD — padding guarantees all offsets valid
            for &(dy, dx) in &[(0isize, -1isize), (-1, 0), (1, 0), (0, 1)] {
                let nix = (ipx as isize + dx) as usize;
                let niy = (ipy as isize + dy) as usize;
                let nidx = niy * in_stride + nix;

                let sad = sad_3x3_plus_scalar([in_x, in_y, in_b], ipx, ipy, nix, niy, in_stride);

                let w = (sad * eff_is + 1.0).max(0.0);
                total_w += w;
                sums[0] += w * in_x[nidx];
                sums[1] += w * in_y[nidx];
                sums[2] += w * in_b[nidx];
            }

            let inv_tw = 1.0 / total_w;
            out_x[oidx] = sums[0] * inv_tw;
            out_y[oidx] = sums[1] * inv_tw;
            out_b[oidx] = sums[2] * inv_tw;
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
#[archmage::rite]
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
    // Compute absolute indices: row offsets already include pad, so adding x or x+ndx
    // directly produces valid padded-buffer positions. Use wrapping arithmetic to avoid
    // overflow when ndx is negative and x is small (the sum c_r0 + x + ndx is always valid
    // because padding guarantees it, but intermediate x+ndx may underflow as usize).
    let cx0 = c_r0 + x;
    let cx_m1 = (c_r0 as isize + x as isize - 1) as usize;
    let cx_p1 = c_r0 + x + 1;
    let nx0 = (n_r0 as isize + x as isize + ndx) as usize;
    let nx_m1 = (n_r0 as isize + x as isize + ndx - 1) as usize;
    let nx_p1 = (n_r0 as isize + x as isize + ndx + 1) as usize;

    // Plus pattern: (0,0), (-1,0), (0,-1), (1,0), (0,1)
    // All offsets are within padded buffer bounds — use load_f32x8 to skip bounds checks.
    // Position (0,0): center row, x vs neighbor row, nx
    let mut sad = {
        let c0x = load_f32x8(token, in_x, cx0);
        let c0y = load_f32x8(token, in_y, cx0);
        let c0b = load_f32x8(token, in_b, cx0);
        let n0x = load_f32x8(token, in_x, nx0);
        let n0y = load_f32x8(token, in_y, nx0);
        let n0b = load_f32x8(token, in_b, nx0);
        (c0x - n0x).abs() * ch_w_x + (c0y - n0y).abs() * ch_w_y + (c0b - n0b).abs() * ch_w_b
    };

    // Position (-1,0): same rows, x-1 vs nx-1
    {
        let c1x = load_f32x8(token, in_x, cx_m1);
        let c1y = load_f32x8(token, in_y, cx_m1);
        let c1b = load_f32x8(token, in_b, cx_m1);
        let n1x = load_f32x8(token, in_x, nx_m1);
        let n1y = load_f32x8(token, in_y, nx_m1);
        let n1b = load_f32x8(token, in_b, nx_m1);
        sad = sad
            + (c1x - n1x).abs() * ch_w_x
            + (c1y - n1y).abs() * ch_w_y
            + (c1b - n1b).abs() * ch_w_b;
    }

    // Position (0,-1): row y-1, x vs row ndy-1, nx
    {
        let c2x = load_f32x8(token, in_x, c_rm1 + x);
        let c2y = load_f32x8(token, in_y, c_rm1 + x);
        let c2b = load_f32x8(token, in_b, c_rm1 + x);
        let nrm1x = (n_rm1 as isize + x as isize + ndx) as usize;
        let n2x = load_f32x8(token, in_x, nrm1x);
        let n2y = load_f32x8(token, in_y, nrm1x);
        let n2b = load_f32x8(token, in_b, nrm1x);
        sad = sad
            + (c2x - n2x).abs() * ch_w_x
            + (c2y - n2y).abs() * ch_w_y
            + (c2b - n2b).abs() * ch_w_b;
    }

    // Position (1,0): same rows, x+1 vs nx+1
    {
        let c3x = load_f32x8(token, in_x, cx_p1);
        let c3y = load_f32x8(token, in_y, cx_p1);
        let c3b = load_f32x8(token, in_b, cx_p1);
        let n3x = load_f32x8(token, in_x, nx_p1);
        let n3y = load_f32x8(token, in_y, nx_p1);
        let n3b = load_f32x8(token, in_b, nx_p1);
        sad = sad
            + (c3x - n3x).abs() * ch_w_x
            + (c3y - n3y).abs() * ch_w_y
            + (c3b - n3b).abs() * ch_w_b;
    }

    // Position (0,1): row y+1, x vs row ndy+1, nx
    {
        let c4x = load_f32x8(token, in_x, c_rp1 + x);
        let c4y = load_f32x8(token, in_y, c_rp1 + x);
        let c4b = load_f32x8(token, in_b, c_rp1 + x);
        let nrp1x = (n_rp1 as isize + x as isize + ndx) as usize;
        let n4x = load_f32x8(token, in_x, nrp1x);
        let n4y = load_f32x8(token, in_y, nrp1x);
        let n4b = load_f32x8(token, in_b, nrp1x);
        sad = sad
            + (c4x - n4x).abs() * ch_w_x
            + (c4y - n4y).abs() * ch_w_y
            + (c4b - n4b).abs() * ch_w_b;
    }

    sad
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn epf_step1_avx2(
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
    in_stride: usize,
    pad: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    use magetypes::simd::f32x8;

    if xsize_blocks < 1 || height < 1 {
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

    for py in 0..height {
        let by = py / block_dim;
        let is_border_row = py % block_dim == 0 || py % block_dim == block_dim - 1;
        let sm_vec = if is_border_row {
            sm_border_row
        } else {
            sm_interior
        };

        // Padded row offsets — padding guarantees ±2 row access is valid
        let r_m2 = (py + pad - 2) * in_stride + pad;
        let r_m1 = (py + pad - 1) * in_stride + pad;
        let r_0 = (py + pad) * in_stride + pad;
        let r_p1 = (py + pad + 1) * in_stride + pad;
        let r_p2 = (py + pad + 2) * in_stride + pad;

        let o0 = py * width;

        // Pre-slice output row
        let orow_x = &mut out_x[o0..o0 + width];
        let orow_y = &mut out_y[o0..o0 + width];
        let orow_b = &mut out_b[o0..o0 + width];

        // All blocks processed uniformly — padding handles edges
        for bx in 0..xsize_blocks {
            let x = bx * block_dim;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];

            if is == 0.0 {
                orow_x[x..x + 8].copy_from_slice(&slice_from(in_x, r_0 + x)[..8]);
                orow_y[x..x + 8].copy_from_slice(&slice_from(in_y, r_0 + x)[..8]);
                orow_b[x..x + 8].copy_from_slice(&slice_from(in_b, r_0 + x)[..8]);
                continue;
            }

            let is_v = f32x8::splat(token, is);
            let eff_is = is_v * sm_vec;

            let cx = load_f32x8(token, in_x, r_0 + x);
            let cy = load_f32x8(token, in_y, r_0 + x);
            let cb = load_f32x8(token, in_b, r_0 + x);

            let mut sum_x = cx;
            let mut sum_y = cy;
            let mut sum_b = cb;
            let mut total_w = one;

            // Neighbor: top (dx=0, dy=-1)
            {
                let sad = sad_3x3_plus_simd(
                    token, in_x, in_y, in_b, x, r_0, r_m1, r_p1, r_m1, r_m2, r_0, 0, ch_w_x,
                    ch_w_y, ch_w_b,
                );
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                let nx = load_f32x8(token, in_x, r_m1 + x);
                let ny = load_f32x8(token, in_y, r_m1 + x);
                let nb = load_f32x8(token, in_b, r_m1 + x);
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);
            }

            // Neighbor: bottom (dx=0, dy=+1)
            {
                let sad = sad_3x3_plus_simd(
                    token, in_x, in_y, in_b, x, r_0, r_m1, r_p1, r_p1, r_0, r_p2, 0, ch_w_x,
                    ch_w_y, ch_w_b,
                );
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                let nx = load_f32x8(token, in_x, r_p1 + x);
                let ny = load_f32x8(token, in_y, r_p1 + x);
                let nb = load_f32x8(token, in_b, r_p1 + x);
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);
            }

            // Neighbor: left (dx=-1, dy=0)
            {
                let sad = sad_3x3_plus_simd(
                    token, in_x, in_y, in_b, x, r_0, r_m1, r_p1, r_0, r_m1, r_p1, -1, ch_w_x,
                    ch_w_y, ch_w_b,
                );
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                let nx = load_f32x8(token, in_x, r_0 + x - 1);
                let ny = load_f32x8(token, in_y, r_0 + x - 1);
                let nb = load_f32x8(token, in_b, r_0 + x - 1);
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);
            }

            // Neighbor: right (dx=+1, dy=0)
            {
                let sad = sad_3x3_plus_simd(
                    token, in_x, in_y, in_b, x, r_0, r_m1, r_p1, r_0, r_m1, r_p1, 1, ch_w_x,
                    ch_w_y, ch_w_b,
                );
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                let nx = load_f32x8(token, in_x, r_0 + x + 1);
                let ny = load_f32x8(token, in_y, r_0 + x + 1);
                let nb = load_f32x8(token, in_b, r_0 + x + 1);
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);
            }

            // Normalize and store
            let inv_tw = total_w.recip();
            let out_arr_x: &mut [f32; 8] = (&mut orow_x[x..x + 8]).try_into().unwrap();
            let out_arr_y: &mut [f32; 8] = (&mut orow_y[x..x + 8]).try_into().unwrap();
            let out_arr_b: &mut [f32; 8] = (&mut orow_b[x..x + 8]).try_into().unwrap();
            (sum_x * inv_tw).store(out_arr_x);
            (sum_y * inv_tw).store(out_arr_y);
            (sum_b * inv_tw).store(out_arr_b);
        }
    }
}

// ============================================================================
// aarch64 NEON implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn epf_step2_neon(
    token: archmage::NeonToken,
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
    in_stride: usize,
    pad: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    use magetypes::simd::f32x4;

    if xsize_blocks < 1 || height < 1 {
        return;
    }

    let ch_w_x = f32x4::splat(token, EPF_CHANNEL_SCALE[0]);
    let ch_w_y = f32x4::splat(token, EPF_CHANNEL_SCALE[1]);
    let ch_w_b = f32x4::splat(token, EPF_CHANNEL_SCALE[2]);
    let one = f32x4::splat(token, 1.0);
    let zero_v = f32x4::zero(token);

    let sm_interior_lo = f32x4::from_array(
        token,
        [
            sigma_scale * border_sigma_mul,
            sigma_scale,
            sigma_scale,
            sigma_scale,
        ],
    );
    let sm_interior_hi = f32x4::from_array(
        token,
        [
            sigma_scale,
            sigma_scale,
            sigma_scale,
            sigma_scale * border_sigma_mul,
        ],
    );
    let sm_border_row = f32x4::splat(token, sigma_scale * border_sigma_mul);

    let block_dim = 8usize;

    for py in 0..height {
        let by = py / block_dim;
        let is_border_row = py % block_dim == 0 || py % block_dim == block_dim - 1;

        // Padded row offsets
        let r0 = (py + pad) * in_stride + pad;
        let rt = (py + pad - 1) * in_stride + pad;
        let rb = (py + pad + 1) * in_stride + pad;
        let o0 = py * width;

        let orow_x = &mut out_x[o0..o0 + width];
        let orow_y = &mut out_y[o0..o0 + width];
        let orow_b = &mut out_b[o0..o0 + width];

        for bx in 0..xsize_blocks {
            let x = bx * block_dim;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];

            if is == 0.0 {
                orow_x[x..x + 8].copy_from_slice(&in_x[r0 + x..r0 + x + 8]);
                orow_y[x..x + 8].copy_from_slice(&in_y[r0 + x..r0 + x + 8]);
                orow_b[x..x + 8].copy_from_slice(&in_b[r0 + x..r0 + x + 8]);
                continue;
            }

            let is_v = f32x4::splat(token, is);

            for half in 0..2usize {
                let hx = x + half * 4;
                let sm_vec = if is_border_row {
                    sm_border_row
                } else if half == 0 {
                    sm_interior_lo
                } else {
                    sm_interior_hi
                };
                let eff_is = is_v * sm_vec;

                let cx = f32x4::from_slice(token, &in_x[r0 + hx..]);
                let cy = f32x4::from_slice(token, &in_y[r0 + hx..]);
                let cb = f32x4::from_slice(token, &in_b[r0 + hx..]);

                let mut sum_x = cx;
                let mut sum_y = cy;
                let mut sum_b = cb;
                let mut total_w = one;

                // Top
                let nx = f32x4::from_slice(token, &in_x[rt + hx..]);
                let ny = f32x4::from_slice(token, &in_y[rt + hx..]);
                let nb = f32x4::from_slice(token, &in_b[rt + hx..]);
                let sad =
                    (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);

                // Bottom
                let nx = f32x4::from_slice(token, &in_x[rb + hx..]);
                let ny = f32x4::from_slice(token, &in_y[rb + hx..]);
                let nb = f32x4::from_slice(token, &in_b[rb + hx..]);
                let sad =
                    (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);

                // Left
                let nx = f32x4::from_slice(token, &in_x[r0 + hx - 1..]);
                let ny = f32x4::from_slice(token, &in_y[r0 + hx - 1..]);
                let nb = f32x4::from_slice(token, &in_b[r0 + hx - 1..]);
                let sad =
                    (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);

                // Right
                let nx = f32x4::from_slice(token, &in_x[r0 + hx + 1..]);
                let ny = f32x4::from_slice(token, &in_y[r0 + hx + 1..]);
                let nb = f32x4::from_slice(token, &in_b[r0 + hx + 1..]);
                let sad =
                    (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);

                let inv_tw = total_w.recip();
                let out_arr_x: &mut [f32; 4] = (&mut orow_x[hx..hx + 4]).try_into().unwrap();
                let out_arr_y: &mut [f32; 4] = (&mut orow_y[hx..hx + 4]).try_into().unwrap();
                let out_arr_b: &mut [f32; 4] = (&mut orow_b[hx..hx + 4]).try_into().unwrap();
                (sum_x * inv_tw).store(out_arr_x);
                (sum_y * inv_tw).store(out_arr_y);
                (sum_b * inv_tw).store(out_arr_b);
            }
        }
    }
}

/// NEON helper: compute 3x3-plus SAD between center at x and neighbor at (x+ndx, ndy rows).
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
#[allow(clippy::too_many_arguments)]
fn sad_3x3_plus_neon(
    token: archmage::NeonToken,
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    x: usize,
    c_r0: usize,
    c_rm1: usize,
    c_rp1: usize,
    n_r0: usize,
    n_rm1: usize,
    n_rp1: usize,
    ndx: isize,
    ch_w_x: magetypes::simd::f32x4,
    ch_w_y: magetypes::simd::f32x4,
    ch_w_b: magetypes::simd::f32x4,
) -> magetypes::simd::f32x4 {
    use magetypes::simd::f32x4;

    // Compute absolute indices: row offsets already include pad, so adding x or x+ndx
    // directly produces valid padded-buffer positions. Use isize arithmetic to avoid
    // overflow when ndx is negative and x is small (the sum c_r0 + x + ndx is always valid
    // because padding guarantees it, but intermediate x+ndx may underflow as usize).
    let cx0 = c_r0 + x;
    let cx_m1 = (c_r0 as isize + x as isize - 1) as usize;
    let cx_p1 = c_r0 + x + 1;
    let nx0 = (n_r0 as isize + x as isize + ndx) as usize;
    let nx_m1 = (n_r0 as isize + x as isize + ndx - 1) as usize;
    let nx_p1 = (n_r0 as isize + x as isize + ndx + 1) as usize;

    // Plus pattern: (0,0), (-1,0), (0,-1), (1,0), (0,1)
    // Position (0,0): center row, x vs neighbor row, nx
    let mut sad = {
        let c0x = f32x4::from_slice(token, &in_x[cx0..]);
        let c0y = f32x4::from_slice(token, &in_y[cx0..]);
        let c0b = f32x4::from_slice(token, &in_b[cx0..]);
        let n0x = f32x4::from_slice(token, &in_x[nx0..]);
        let n0y = f32x4::from_slice(token, &in_y[nx0..]);
        let n0b = f32x4::from_slice(token, &in_b[nx0..]);
        (c0x - n0x).abs() * ch_w_x + (c0y - n0y).abs() * ch_w_y + (c0b - n0b).abs() * ch_w_b
    };

    // Position (-1,0): same rows, x-1 vs nx-1
    {
        let c1x = f32x4::from_slice(token, &in_x[cx_m1..]);
        let c1y = f32x4::from_slice(token, &in_y[cx_m1..]);
        let c1b = f32x4::from_slice(token, &in_b[cx_m1..]);
        let n1x = f32x4::from_slice(token, &in_x[nx_m1..]);
        let n1y = f32x4::from_slice(token, &in_y[nx_m1..]);
        let n1b = f32x4::from_slice(token, &in_b[nx_m1..]);
        sad = sad
            + (c1x - n1x).abs() * ch_w_x
            + (c1y - n1y).abs() * ch_w_y
            + (c1b - n1b).abs() * ch_w_b;
    }

    // Position (0,-1): row y-1, x vs row ndy-1, nx
    {
        let c2x = f32x4::from_slice(token, &in_x[c_rm1 + x..]);
        let c2y = f32x4::from_slice(token, &in_y[c_rm1 + x..]);
        let c2b = f32x4::from_slice(token, &in_b[c_rm1 + x..]);
        let n2x = f32x4::from_slice(token, &in_x[(n_rm1 as isize + x as isize + ndx) as usize..]);
        let n2y = f32x4::from_slice(token, &in_y[(n_rm1 as isize + x as isize + ndx) as usize..]);
        let n2b = f32x4::from_slice(token, &in_b[(n_rm1 as isize + x as isize + ndx) as usize..]);
        sad = sad
            + (c2x - n2x).abs() * ch_w_x
            + (c2y - n2y).abs() * ch_w_y
            + (c2b - n2b).abs() * ch_w_b;
    }

    // Position (1,0): same rows, x+1 vs nx+1
    {
        let c3x = f32x4::from_slice(token, &in_x[cx_p1..]);
        let c3y = f32x4::from_slice(token, &in_y[cx_p1..]);
        let c3b = f32x4::from_slice(token, &in_b[cx_p1..]);
        let n3x = f32x4::from_slice(token, &in_x[nx_p1..]);
        let n3y = f32x4::from_slice(token, &in_y[nx_p1..]);
        let n3b = f32x4::from_slice(token, &in_b[nx_p1..]);
        sad = sad
            + (c3x - n3x).abs() * ch_w_x
            + (c3y - n3y).abs() * ch_w_y
            + (c3b - n3b).abs() * ch_w_b;
    }

    // Position (0,1): row y+1, x vs row ndy+1, nx
    {
        let c4x = f32x4::from_slice(token, &in_x[c_rp1 + x..]);
        let c4y = f32x4::from_slice(token, &in_y[c_rp1 + x..]);
        let c4b = f32x4::from_slice(token, &in_b[c_rp1 + x..]);
        let n4x = f32x4::from_slice(token, &in_x[(n_rp1 as isize + x as isize + ndx) as usize..]);
        let n4y = f32x4::from_slice(token, &in_y[(n_rp1 as isize + x as isize + ndx) as usize..]);
        let n4b = f32x4::from_slice(token, &in_b[(n_rp1 as isize + x as isize + ndx) as usize..]);
        sad = sad
            + (c4x - n4x).abs() * ch_w_x
            + (c4y - n4y).abs() * ch_w_y
            + (c4b - n4b).abs() * ch_w_b;
    }

    sad
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn epf_step1_neon(
    token: archmage::NeonToken,
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
    in_stride: usize,
    pad: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    use magetypes::simd::f32x4;

    if xsize_blocks < 1 || height < 1 {
        return;
    }

    let ch_w_x = f32x4::splat(token, EPF_CHANNEL_SCALE[0]);
    let ch_w_y = f32x4::splat(token, EPF_CHANNEL_SCALE[1]);
    let ch_w_b = f32x4::splat(token, EPF_CHANNEL_SCALE[2]);
    let one = f32x4::splat(token, 1.0);
    let zero_v = f32x4::zero(token);

    let sm_interior_lo = f32x4::from_array(
        token,
        [
            sigma_scale * border_sigma_mul,
            sigma_scale,
            sigma_scale,
            sigma_scale,
        ],
    );
    let sm_interior_hi = f32x4::from_array(
        token,
        [
            sigma_scale,
            sigma_scale,
            sigma_scale,
            sigma_scale * border_sigma_mul,
        ],
    );
    let sm_border_row = f32x4::splat(token, sigma_scale * border_sigma_mul);

    let block_dim = 8usize;

    for py in 0..height {
        let by = py / block_dim;
        let is_border_row = py % block_dim == 0 || py % block_dim == block_dim - 1;

        // Padded row offsets — padding guarantees ±2 row access is valid
        let r_m2 = (py + pad - 2) * in_stride + pad;
        let r_m1 = (py + pad - 1) * in_stride + pad;
        let r_0 = (py + pad) * in_stride + pad;
        let r_p1 = (py + pad + 1) * in_stride + pad;
        let r_p2 = (py + pad + 2) * in_stride + pad;

        let o0 = py * width;

        // Pre-slice output row
        let orow_x = &mut out_x[o0..o0 + width];
        let orow_y = &mut out_y[o0..o0 + width];
        let orow_b = &mut out_b[o0..o0 + width];

        // All blocks processed uniformly — padding handles edges
        for bx in 0..xsize_blocks {
            let x = bx * block_dim;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];

            if is == 0.0 {
                orow_x[x..x + 8].copy_from_slice(&in_x[r_0 + x..r_0 + x + 8]);
                orow_y[x..x + 8].copy_from_slice(&in_y[r_0 + x..r_0 + x + 8]);
                orow_b[x..x + 8].copy_from_slice(&in_b[r_0 + x..r_0 + x + 8]);
                continue;
            }

            let is_v = f32x4::splat(token, is);

            for half in 0..2usize {
                let hx = x + half * 4;
                let sm_vec = if is_border_row {
                    sm_border_row
                } else if half == 0 {
                    sm_interior_lo
                } else {
                    sm_interior_hi
                };
                let eff_is = is_v * sm_vec;

                let cx = f32x4::from_slice(token, &in_x[r_0 + hx..]);
                let cy = f32x4::from_slice(token, &in_y[r_0 + hx..]);
                let cb = f32x4::from_slice(token, &in_b[r_0 + hx..]);

                let mut sum_x = cx;
                let mut sum_y = cy;
                let mut sum_b = cb;
                let mut total_w = one;

                // Neighbor: top (dy=-1)
                {
                    let sad = sad_3x3_plus_neon(
                        token, in_x, in_y, in_b, hx, r_0, r_m1, r_p1, r_m1, r_m2, r_0, 0, ch_w_x,
                        ch_w_y, ch_w_b,
                    );
                    let w = (sad * eff_is + one).max(zero_v);
                    total_w += w;
                    let nx = f32x4::from_slice(token, &in_x[r_m1 + hx..]);
                    let ny = f32x4::from_slice(token, &in_y[r_m1 + hx..]);
                    let nb = f32x4::from_slice(token, &in_b[r_m1 + hx..]);
                    sum_x = w.mul_add(nx, sum_x);
                    sum_y = w.mul_add(ny, sum_y);
                    sum_b = w.mul_add(nb, sum_b);
                }

                // Neighbor: bottom (dy=+1)
                {
                    let sad = sad_3x3_plus_neon(
                        token, in_x, in_y, in_b, hx, r_0, r_m1, r_p1, r_p1, r_0, r_p2, 0, ch_w_x,
                        ch_w_y, ch_w_b,
                    );
                    let w = (sad * eff_is + one).max(zero_v);
                    total_w += w;
                    let nx = f32x4::from_slice(token, &in_x[r_p1 + hx..]);
                    let ny = f32x4::from_slice(token, &in_y[r_p1 + hx..]);
                    let nb = f32x4::from_slice(token, &in_b[r_p1 + hx..]);
                    sum_x = w.mul_add(nx, sum_x);
                    sum_y = w.mul_add(ny, sum_y);
                    sum_b = w.mul_add(nb, sum_b);
                }

                // Neighbor: left (dx=-1)
                {
                    let sad = sad_3x3_plus_neon(
                        token, in_x, in_y, in_b, hx, r_0, r_m1, r_p1, r_0, r_m1, r_p1, -1, ch_w_x,
                        ch_w_y, ch_w_b,
                    );
                    let w = (sad * eff_is + one).max(zero_v);
                    total_w += w;
                    let nx = f32x4::from_slice(token, &in_x[r_0 + hx - 1..]);
                    let ny = f32x4::from_slice(token, &in_y[r_0 + hx - 1..]);
                    let nb = f32x4::from_slice(token, &in_b[r_0 + hx - 1..]);
                    sum_x = w.mul_add(nx, sum_x);
                    sum_y = w.mul_add(ny, sum_y);
                    sum_b = w.mul_add(nb, sum_b);
                }

                // Neighbor: right (dx=+1)
                {
                    let sad = sad_3x3_plus_neon(
                        token, in_x, in_y, in_b, hx, r_0, r_m1, r_p1, r_0, r_m1, r_p1, 1, ch_w_x,
                        ch_w_y, ch_w_b,
                    );
                    let w = (sad * eff_is + one).max(zero_v);
                    total_w += w;
                    let nx = f32x4::from_slice(token, &in_x[r_0 + hx + 1..]);
                    let ny = f32x4::from_slice(token, &in_y[r_0 + hx + 1..]);
                    let nb = f32x4::from_slice(token, &in_b[r_0 + hx + 1..]);
                    sum_x = w.mul_add(nx, sum_x);
                    sum_y = w.mul_add(ny, sum_y);
                    sum_b = w.mul_add(nb, sum_b);
                }

                // Normalize and store
                let inv_tw = total_w.recip();
                let out_arr_x: &mut [f32; 4] = (&mut orow_x[hx..hx + 4]).try_into().unwrap();
                let out_arr_y: &mut [f32; 4] = (&mut orow_y[hx..hx + 4]).try_into().unwrap();
                let out_arr_b: &mut [f32; 4] = (&mut orow_b[hx..hx + 4]).try_into().unwrap();
                (sum_x * inv_tw).store(out_arr_x);
                (sum_y * inv_tw).store(out_arr_y);
                (sum_b * inv_tw).store(out_arr_b);
            }
        }
    }
}

// ============================================================================
// WASM SIMD128 implementations (128-bit / f32x4, mirrors NEON)
// ============================================================================

#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn epf_step2_wasm128(
    token: archmage::Wasm128Token,
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
    in_stride: usize,
    pad: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    use magetypes::simd::f32x4;

    if xsize_blocks < 1 || height < 1 {
        return;
    }

    let ch_w_x = f32x4::splat(token, EPF_CHANNEL_SCALE[0]);
    let ch_w_y = f32x4::splat(token, EPF_CHANNEL_SCALE[1]);
    let ch_w_b = f32x4::splat(token, EPF_CHANNEL_SCALE[2]);
    let one = f32x4::splat(token, 1.0);
    let zero_v = f32x4::zero(token);

    let sm_interior_lo = f32x4::from_array(
        token,
        [
            sigma_scale * border_sigma_mul,
            sigma_scale,
            sigma_scale,
            sigma_scale,
        ],
    );
    let sm_interior_hi = f32x4::from_array(
        token,
        [
            sigma_scale,
            sigma_scale,
            sigma_scale,
            sigma_scale * border_sigma_mul,
        ],
    );
    let sm_border_row = f32x4::splat(token, sigma_scale * border_sigma_mul);

    let block_dim = 8usize;

    for py in 0..height {
        let by = py / block_dim;
        let is_border_row = py % block_dim == 0 || py % block_dim == block_dim - 1;

        // Padded row offsets
        let r0 = (py + pad) * in_stride + pad;
        let rt = (py + pad - 1) * in_stride + pad;
        let rb = (py + pad + 1) * in_stride + pad;
        let o0 = py * width;

        let orow_x = &mut out_x[o0..o0 + width];
        let orow_y = &mut out_y[o0..o0 + width];
        let orow_b = &mut out_b[o0..o0 + width];

        for bx in 0..xsize_blocks {
            let x = bx * block_dim;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];

            if is == 0.0 {
                orow_x[x..x + 8].copy_from_slice(&in_x[r0 + x..r0 + x + 8]);
                orow_y[x..x + 8].copy_from_slice(&in_y[r0 + x..r0 + x + 8]);
                orow_b[x..x + 8].copy_from_slice(&in_b[r0 + x..r0 + x + 8]);
                continue;
            }

            let is_v = f32x4::splat(token, is);

            for half in 0..2usize {
                let hx = x + half * 4;
                let sm_vec = if is_border_row {
                    sm_border_row
                } else if half == 0 {
                    sm_interior_lo
                } else {
                    sm_interior_hi
                };
                let eff_is = is_v * sm_vec;

                let cx = f32x4::from_slice(token, &in_x[r0 + hx..]);
                let cy = f32x4::from_slice(token, &in_y[r0 + hx..]);
                let cb = f32x4::from_slice(token, &in_b[r0 + hx..]);

                let mut sum_x = cx;
                let mut sum_y = cy;
                let mut sum_b = cb;
                let mut total_w = one;

                // Top
                let nx = f32x4::from_slice(token, &in_x[rt + hx..]);
                let ny = f32x4::from_slice(token, &in_y[rt + hx..]);
                let nb = f32x4::from_slice(token, &in_b[rt + hx..]);
                let sad =
                    (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);

                // Bottom
                let nx = f32x4::from_slice(token, &in_x[rb + hx..]);
                let ny = f32x4::from_slice(token, &in_y[rb + hx..]);
                let nb = f32x4::from_slice(token, &in_b[rb + hx..]);
                let sad =
                    (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);

                // Left
                let nx = f32x4::from_slice(token, &in_x[r0 + hx - 1..]);
                let ny = f32x4::from_slice(token, &in_y[r0 + hx - 1..]);
                let nb = f32x4::from_slice(token, &in_b[r0 + hx - 1..]);
                let sad =
                    (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);

                // Right
                let nx = f32x4::from_slice(token, &in_x[r0 + hx + 1..]);
                let ny = f32x4::from_slice(token, &in_y[r0 + hx + 1..]);
                let nb = f32x4::from_slice(token, &in_b[r0 + hx + 1..]);
                let sad =
                    (cx - nx).abs() * ch_w_x + (cy - ny).abs() * ch_w_y + (cb - nb).abs() * ch_w_b;
                let w = (sad * eff_is + one).max(zero_v);
                total_w += w;
                sum_x = w.mul_add(nx, sum_x);
                sum_y = w.mul_add(ny, sum_y);
                sum_b = w.mul_add(nb, sum_b);

                let inv_tw = total_w.recip();
                let out_arr_x: &mut [f32; 4] = (&mut orow_x[hx..hx + 4]).try_into().unwrap();
                let out_arr_y: &mut [f32; 4] = (&mut orow_y[hx..hx + 4]).try_into().unwrap();
                let out_arr_b: &mut [f32; 4] = (&mut orow_b[hx..hx + 4]).try_into().unwrap();
                (sum_x * inv_tw).store(out_arr_x);
                (sum_y * inv_tw).store(out_arr_y);
                (sum_b * inv_tw).store(out_arr_b);
            }
        }
    }
}

/// WASM128 helper: compute 3x3-plus SAD between center at x and neighbor at (x+ndx, ndy rows).
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
#[allow(clippy::too_many_arguments)]
fn sad_3x3_plus_wasm128(
    token: archmage::Wasm128Token,
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    x: usize,
    c_r0: usize,
    c_rm1: usize,
    c_rp1: usize,
    n_r0: usize,
    n_rm1: usize,
    n_rp1: usize,
    ndx: isize,
    ch_w_x: magetypes::simd::f32x4,
    ch_w_y: magetypes::simd::f32x4,
    ch_w_b: magetypes::simd::f32x4,
) -> magetypes::simd::f32x4 {
    use magetypes::simd::f32x4;

    // Compute absolute indices: row offsets already include pad, so adding x or x+ndx
    // directly produces valid padded-buffer positions. Use isize arithmetic to avoid
    // overflow when ndx is negative and x is small (the sum c_r0 + x + ndx is always valid
    // because padding guarantees it, but intermediate x+ndx may underflow as usize).
    let cx0 = c_r0 + x;
    let cx_m1 = (c_r0 as isize + x as isize - 1) as usize;
    let cx_p1 = c_r0 + x + 1;
    let nx0 = (n_r0 as isize + x as isize + ndx) as usize;
    let nx_m1 = (n_r0 as isize + x as isize + ndx - 1) as usize;
    let nx_p1 = (n_r0 as isize + x as isize + ndx + 1) as usize;

    // Plus pattern: (0,0), (-1,0), (0,-1), (1,0), (0,1)
    // Position (0,0): center row, x vs neighbor row, nx
    let mut sad = {
        let c0x = f32x4::from_slice(token, &in_x[cx0..]);
        let c0y = f32x4::from_slice(token, &in_y[cx0..]);
        let c0b = f32x4::from_slice(token, &in_b[cx0..]);
        let n0x = f32x4::from_slice(token, &in_x[nx0..]);
        let n0y = f32x4::from_slice(token, &in_y[nx0..]);
        let n0b = f32x4::from_slice(token, &in_b[nx0..]);
        (c0x - n0x).abs() * ch_w_x + (c0y - n0y).abs() * ch_w_y + (c0b - n0b).abs() * ch_w_b
    };

    // Position (-1,0): same rows, x-1 vs nx-1
    {
        let c1x = f32x4::from_slice(token, &in_x[cx_m1..]);
        let c1y = f32x4::from_slice(token, &in_y[cx_m1..]);
        let c1b = f32x4::from_slice(token, &in_b[cx_m1..]);
        let n1x = f32x4::from_slice(token, &in_x[nx_m1..]);
        let n1y = f32x4::from_slice(token, &in_y[nx_m1..]);
        let n1b = f32x4::from_slice(token, &in_b[nx_m1..]);
        sad = sad
            + (c1x - n1x).abs() * ch_w_x
            + (c1y - n1y).abs() * ch_w_y
            + (c1b - n1b).abs() * ch_w_b;
    }

    // Position (0,-1): row y-1, x vs row ndy-1, nx
    {
        let c2x = f32x4::from_slice(token, &in_x[c_rm1 + x..]);
        let c2y = f32x4::from_slice(token, &in_y[c_rm1 + x..]);
        let c2b = f32x4::from_slice(token, &in_b[c_rm1 + x..]);
        let n2x = f32x4::from_slice(token, &in_x[(n_rm1 as isize + x as isize + ndx) as usize..]);
        let n2y = f32x4::from_slice(token, &in_y[(n_rm1 as isize + x as isize + ndx) as usize..]);
        let n2b = f32x4::from_slice(token, &in_b[(n_rm1 as isize + x as isize + ndx) as usize..]);
        sad = sad
            + (c2x - n2x).abs() * ch_w_x
            + (c2y - n2y).abs() * ch_w_y
            + (c2b - n2b).abs() * ch_w_b;
    }

    // Position (1,0): same rows, x+1 vs nx+1
    {
        let c3x = f32x4::from_slice(token, &in_x[cx_p1..]);
        let c3y = f32x4::from_slice(token, &in_y[cx_p1..]);
        let c3b = f32x4::from_slice(token, &in_b[cx_p1..]);
        let n3x = f32x4::from_slice(token, &in_x[nx_p1..]);
        let n3y = f32x4::from_slice(token, &in_y[nx_p1..]);
        let n3b = f32x4::from_slice(token, &in_b[nx_p1..]);
        sad = sad
            + (c3x - n3x).abs() * ch_w_x
            + (c3y - n3y).abs() * ch_w_y
            + (c3b - n3b).abs() * ch_w_b;
    }

    // Position (0,1): row y+1, x vs row ndy+1, nx
    {
        let c4x = f32x4::from_slice(token, &in_x[c_rp1 + x..]);
        let c4y = f32x4::from_slice(token, &in_y[c_rp1 + x..]);
        let c4b = f32x4::from_slice(token, &in_b[c_rp1 + x..]);
        let n4x = f32x4::from_slice(token, &in_x[(n_rp1 as isize + x as isize + ndx) as usize..]);
        let n4y = f32x4::from_slice(token, &in_y[(n_rp1 as isize + x as isize + ndx) as usize..]);
        let n4b = f32x4::from_slice(token, &in_b[(n_rp1 as isize + x as isize + ndx) as usize..]);
        sad = sad
            + (c4x - n4x).abs() * ch_w_x
            + (c4y - n4y).abs() * ch_w_y
            + (c4b - n4b).abs() * ch_w_b;
    }

    sad
}

#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn epf_step1_wasm128(
    token: archmage::Wasm128Token,
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
    in_stride: usize,
    pad: usize,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    use magetypes::simd::f32x4;

    if xsize_blocks < 1 || height < 1 {
        return;
    }

    let ch_w_x = f32x4::splat(token, EPF_CHANNEL_SCALE[0]);
    let ch_w_y = f32x4::splat(token, EPF_CHANNEL_SCALE[1]);
    let ch_w_b = f32x4::splat(token, EPF_CHANNEL_SCALE[2]);
    let one = f32x4::splat(token, 1.0);
    let zero_v = f32x4::zero(token);

    let sm_interior_lo = f32x4::from_array(
        token,
        [
            sigma_scale * border_sigma_mul,
            sigma_scale,
            sigma_scale,
            sigma_scale,
        ],
    );
    let sm_interior_hi = f32x4::from_array(
        token,
        [
            sigma_scale,
            sigma_scale,
            sigma_scale,
            sigma_scale * border_sigma_mul,
        ],
    );
    let sm_border_row = f32x4::splat(token, sigma_scale * border_sigma_mul);

    let block_dim = 8usize;

    for py in 0..height {
        let by = py / block_dim;
        let is_border_row = py % block_dim == 0 || py % block_dim == block_dim - 1;

        // Padded row offsets — padding guarantees ±2 row access is valid
        let r_m2 = (py + pad - 2) * in_stride + pad;
        let r_m1 = (py + pad - 1) * in_stride + pad;
        let r_0 = (py + pad) * in_stride + pad;
        let r_p1 = (py + pad + 1) * in_stride + pad;
        let r_p2 = (py + pad + 2) * in_stride + pad;

        let o0 = py * width;

        // Pre-slice output row
        let orow_x = &mut out_x[o0..o0 + width];
        let orow_y = &mut out_y[o0..o0 + width];
        let orow_b = &mut out_b[o0..o0 + width];

        // All blocks processed uniformly — padding handles edges
        for bx in 0..xsize_blocks {
            let x = bx * block_dim;
            let sigma_idx = by * xsize_blocks + bx;
            let is = inv_sigma[sigma_idx];

            if is == 0.0 {
                orow_x[x..x + 8].copy_from_slice(&in_x[r_0 + x..r_0 + x + 8]);
                orow_y[x..x + 8].copy_from_slice(&in_y[r_0 + x..r_0 + x + 8]);
                orow_b[x..x + 8].copy_from_slice(&in_b[r_0 + x..r_0 + x + 8]);
                continue;
            }

            let is_v = f32x4::splat(token, is);

            for half in 0..2usize {
                let hx = x + half * 4;
                let sm_vec = if is_border_row {
                    sm_border_row
                } else if half == 0 {
                    sm_interior_lo
                } else {
                    sm_interior_hi
                };
                let eff_is = is_v * sm_vec;

                let cx = f32x4::from_slice(token, &in_x[r_0 + hx..]);
                let cy = f32x4::from_slice(token, &in_y[r_0 + hx..]);
                let cb = f32x4::from_slice(token, &in_b[r_0 + hx..]);

                let mut sum_x = cx;
                let mut sum_y = cy;
                let mut sum_b = cb;
                let mut total_w = one;

                // Neighbor: top (dy=-1)
                {
                    let sad = sad_3x3_plus_wasm128(
                        token, in_x, in_y, in_b, hx, r_0, r_m1, r_p1, r_m1, r_m2, r_0, 0, ch_w_x,
                        ch_w_y, ch_w_b,
                    );
                    let w = (sad * eff_is + one).max(zero_v);
                    total_w += w;
                    let nx = f32x4::from_slice(token, &in_x[r_m1 + hx..]);
                    let ny = f32x4::from_slice(token, &in_y[r_m1 + hx..]);
                    let nb = f32x4::from_slice(token, &in_b[r_m1 + hx..]);
                    sum_x = w.mul_add(nx, sum_x);
                    sum_y = w.mul_add(ny, sum_y);
                    sum_b = w.mul_add(nb, sum_b);
                }

                // Neighbor: bottom (dy=+1)
                {
                    let sad = sad_3x3_plus_wasm128(
                        token, in_x, in_y, in_b, hx, r_0, r_m1, r_p1, r_p1, r_0, r_p2, 0, ch_w_x,
                        ch_w_y, ch_w_b,
                    );
                    let w = (sad * eff_is + one).max(zero_v);
                    total_w += w;
                    let nx = f32x4::from_slice(token, &in_x[r_p1 + hx..]);
                    let ny = f32x4::from_slice(token, &in_y[r_p1 + hx..]);
                    let nb = f32x4::from_slice(token, &in_b[r_p1 + hx..]);
                    sum_x = w.mul_add(nx, sum_x);
                    sum_y = w.mul_add(ny, sum_y);
                    sum_b = w.mul_add(nb, sum_b);
                }

                // Neighbor: left (dx=-1)
                {
                    let sad = sad_3x3_plus_wasm128(
                        token, in_x, in_y, in_b, hx, r_0, r_m1, r_p1, r_0, r_m1, r_p1, -1, ch_w_x,
                        ch_w_y, ch_w_b,
                    );
                    let w = (sad * eff_is + one).max(zero_v);
                    total_w += w;
                    let nx = f32x4::from_slice(token, &in_x[r_0 + hx - 1..]);
                    let ny = f32x4::from_slice(token, &in_y[r_0 + hx - 1..]);
                    let nb = f32x4::from_slice(token, &in_b[r_0 + hx - 1..]);
                    sum_x = w.mul_add(nx, sum_x);
                    sum_y = w.mul_add(ny, sum_y);
                    sum_b = w.mul_add(nb, sum_b);
                }

                // Neighbor: right (dx=+1)
                {
                    let sad = sad_3x3_plus_wasm128(
                        token, in_x, in_y, in_b, hx, r_0, r_m1, r_p1, r_0, r_m1, r_p1, 1, ch_w_x,
                        ch_w_y, ch_w_b,
                    );
                    let w = (sad * eff_is + one).max(zero_v);
                    total_w += w;
                    let nx = f32x4::from_slice(token, &in_x[r_0 + hx + 1..]);
                    let ny = f32x4::from_slice(token, &in_y[r_0 + hx + 1..]);
                    let nb = f32x4::from_slice(token, &in_b[r_0 + hx + 1..]);
                    sum_x = w.mul_add(nx, sum_x);
                    sum_y = w.mul_add(ny, sum_y);
                    sum_b = w.mul_add(nb, sum_b);
                }

                // Normalize and store
                let inv_tw = total_w.recip();
                let out_arr_x: &mut [f32; 4] = (&mut orow_x[hx..hx + 4]).try_into().unwrap();
                let out_arr_y: &mut [f32; 4] = (&mut orow_y[hx..hx + 4]).try_into().unwrap();
                let out_arr_b: &mut [f32; 4] = (&mut orow_b[hx..hx + 4]).try_into().unwrap();
                (sum_x * inv_tw).store(out_arr_x);
                (sum_y * inv_tw).store(out_arr_y);
                (sum_b * inv_tw).store(out_arr_b);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    extern crate std;
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
        let inv_sigma = vec![-1.0f32; xsb * ysb];

        let pad = 1;
        let px = pad_plane(&in_x, w, h, pad);
        let py_p = pad_plane(&in_y, w, h, pad);
        let pb = pad_plane(&in_b, w, h, pad);
        let stride = w + 2 * pad;

        epf_step2(
            &px,
            &py_p,
            &pb,
            &mut out_x,
            &mut out_y,
            &mut out_b,
            &inv_sigma,
            xsb,
            w,
            h,
            stride,
            pad,
            6.5 * 1.65,
            2.0 / 3.0,
        );

        for i in 0..w * h {
            assert!(
                (out_x[i] - val).abs() < 1e-5,
                "step2 X: i={} got {} expected {}",
                i,
                out_x[i],
                val
            );
            assert!(
                (out_y[i] - val).abs() < 1e-5,
                "step2 Y: i={} got {} expected {}",
                i,
                out_y[i],
                val
            );
            assert!(
                (out_b[i] - val).abs() < 1e-5,
                "step2 B: i={} got {} expected {}",
                i,
                out_b[i],
                val
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
        let inv_sigma = vec![-1.0f32; xsb * (h / 8)];

        let pad = 2;
        let px = pad_plane(&in_x, w, h, pad);
        let py_p = pad_plane(&in_y, w, h, pad);
        let pb = pad_plane(&in_b, w, h, pad);
        let stride = w + 2 * pad;

        epf_step1(
            &px,
            &py_p,
            &pb,
            &mut out_x,
            &mut out_y,
            &mut out_b,
            &inv_sigma,
            xsb,
            w,
            h,
            stride,
            pad,
            1.65,
            2.0 / 3.0,
        );

        for (i, &x) in out_x[..w * h].iter().enumerate() {
            assert!(
                (x - val).abs() < 1e-5,
                "step1 X: i={} got {} expected {}",
                i,
                x,
                val
            );
        }
    }

    /// SIMD step2 must match scalar step2 on varied input.
    #[test]
    fn test_epf_step2_simd_matches_scalar() {
        let w = 48;
        let h = 32;
        let n = w * h;

        let mut raw_x = vec![0.0f32; n];
        let mut raw_y = vec![0.0f32; n];
        let mut raw_b = vec![0.0f32; n];
        for i in 0..n {
            let x = (i % w) as f32;
            let y = (i / w) as f32;
            raw_x[i] = (x * 0.01 + y * 0.007).sin() * 0.5 + 0.5;
            raw_y[i] = (x * 0.013 + y * 0.011).cos() * 0.3 + 0.4;
            raw_b[i] = (x * 0.009 + y * 0.015).sin() * 0.2 + 0.3;
        }

        let xsb = w / 8;
        let mut inv_sigma = vec![0.0f32; xsb * (h / 8)];
        for (i, s) in inv_sigma.iter_mut().enumerate() {
            *s = if i % 3 == 0 {
                0.0
            } else {
                -0.5 - (i as f32) * 0.1
            };
        }

        let sigma_scale = 6.5 * 1.65;
        let border_mul = 2.0 / 3.0;
        let pad = 1;
        let in_x = pad_plane(&raw_x, w, h, pad);
        let in_y = pad_plane(&raw_y, w, h, pad);
        let in_b = pad_plane(&raw_b, w, h, pad);
        let stride = w + 2 * pad;

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
            stride,
            pad,
            sigma_scale,
            border_mul,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
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
                    stride,
                    pad,
                    sigma_scale,
                    border_mul,
                );

                for i in 0..n {
                    let ex = (out_x[i] - ref_x[i]).abs();
                    let ey = (out_y[i] - ref_y[i]).abs();
                    let eb = (out_b[i] - ref_b[i]).abs();
                    let err = ex.max(ey).max(eb);
                    assert!(
                        err < 1e-4,
                        "step2 mismatch at pixel {}: SIMD=({},{},{}) scalar=({},{},{}) err={} [{perm}]",
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
            },
        );
        std::eprintln!("{report}");
    }

    /// SIMD step1 must match scalar step1 on varied input.
    #[test]
    fn test_epf_step1_simd_matches_scalar() {
        let w = 48;
        let h = 32;
        let n = w * h;

        let mut raw_x = vec![0.0f32; n];
        let mut raw_y = vec![0.0f32; n];
        let mut raw_b = vec![0.0f32; n];
        for i in 0..n {
            let x = (i % w) as f32;
            let y = (i / w) as f32;
            raw_x[i] = (x * 0.01 + y * 0.007).sin() * 0.5 + 0.5;
            raw_y[i] = (x * 0.013 + y * 0.011).cos() * 0.3 + 0.4;
            raw_b[i] = (x * 0.009 + y * 0.015).sin() * 0.2 + 0.3;
        }

        let xsb = w / 8;
        let mut inv_sigma = vec![0.0f32; xsb * (h / 8)];
        for (i, s) in inv_sigma.iter_mut().enumerate() {
            *s = if i % 3 == 0 {
                0.0
            } else {
                -0.5 - (i as f32) * 0.1
            };
        }

        let sigma_scale = 1.65;
        let border_mul = 2.0 / 3.0;
        let pad = 2;
        let in_x = pad_plane(&raw_x, w, h, pad);
        let in_y = pad_plane(&raw_y, w, h, pad);
        let in_b = pad_plane(&raw_b, w, h, pad);
        let stride = w + 2 * pad;

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
            stride,
            pad,
            sigma_scale,
            border_mul,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
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
                    stride,
                    pad,
                    sigma_scale,
                    border_mul,
                );

                for i in 0..n {
                    let ex = (out_x[i] - ref_x[i]).abs();
                    let ey = (out_y[i] - ref_y[i]).abs();
                    let eb = (out_b[i] - ref_b[i]).abs();
                    let err = ex.max(ey).max(eb);
                    assert!(
                        err < 1e-4,
                        "step1 mismatch at pixel {}: SIMD=({},{},{}) scalar=({},{},{}) err={} [{perm}]",
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
            },
        );
        std::eprintln!("{report}");
    }

    /// EPF with inv_sigma=0 should be a no-op (copy input to output).
    #[test]
    fn test_epf_zero_sigma_noop() {
        let w = 32;
        let h = 16;
        let n = w * h;
        let raw_x: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let raw_y: Vec<f32> = (0..n).map(|i| i as f32 * 0.002 + 1.0).collect();
        let raw_b: Vec<f32> = (0..n).map(|i| i as f32 * 0.003 + 2.0).collect();
        let mut out_x = vec![0.0f32; n];
        let mut out_y = vec![0.0f32; n];
        let mut out_b = vec![0.0f32; n];

        let xsb = w / 8;
        let inv_sigma = vec![0.0f32; xsb * (h / 8)];

        let pad = 1;
        let in_x = pad_plane(&raw_x, w, h, pad);
        let in_y = pad_plane(&raw_y, w, h, pad);
        let in_b = pad_plane(&raw_b, w, h, pad);
        let stride = w + 2 * pad;

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
            stride,
            pad,
            6.5 * 1.65,
            2.0 / 3.0,
        );

        for i in 0..n {
            assert_eq!(out_x[i], raw_x[i], "noop X at {}", i);
            assert_eq!(out_y[i], raw_y[i], "noop Y at {}", i);
            assert_eq!(out_b[i], raw_b[i], "noop B at {}", i);
        }
    }
}
