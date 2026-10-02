// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Separable 5-tap Gaussian convolution (dot detection prefilter).
//!
//! `out[x] = w0*in[x] + w1*(in[x-1]+in[x+1]) + w2*(in[x-2]+in[x+2])`
//! with clamped edge taps, matching `dot_detection::gaussian_separable_5_*`.
//!
//! Every SIMD lane evaluates the identical expression tree as the scalar
//! reference — explicit `mul`/`add`, never `mul_add` (FMA) — so output is
//! bit-identical across tiers.

use archmage::prelude::*;

/// Horizontal pass: `dst[y][x] = w0*src[y][x] + w1*(src[y][x-1]+src[y][x+1])
/// + w2*(src[y][x-2]+src[y][x+2])`, x-clamped taps at the edges.
#[inline]
pub fn gaussian5_horizontal(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst.len(), width * height);
    incant!(gaussian5_h_impl(src, dst, width, height, taps));
}

/// Vertical pass: `dst[y][x] = w0*src[y][x] + w1*(src[y-1][x]+src[y+1][x])
/// + w2*(src[y-2][x]+src[y+2][x])`, y-clamped taps at the edge rows.
#[inline]
pub fn gaussian5_vertical(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst.len(), width * height);
    incant!(gaussian5_v_impl(src, dst, width, height, taps));
}

// ============================================================================
// Scalar reference (also the tiny-image + edge fallback)
// ============================================================================

fn h_scalar_pixel(row: &[f32], x: usize, taps: [f32; 3]) -> f32 {
    let width = row.len();
    let xm2 = x.saturating_sub(2);
    let xm1 = x.saturating_sub(1);
    let xp1 = if x + 1 < width { x + 1 } else { width - 1 };
    let xp2 = if x + 2 < width { x + 2 } else { width - 1 };
    let [w0, w1, w2] = taps;
    w0 * row[x] + w1 * (row[xm1] + row[xp1]) + w2 * (row[xm2] + row[xp2])
}

fn v_scalar_pixel(
    src: &[f32],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    taps: [f32; 3],
) -> f32 {
    let ym2 = y.saturating_sub(2);
    let ym1 = y.saturating_sub(1);
    let yp1 = if y + 1 < height { y + 1 } else { height - 1 };
    let yp2 = if y + 2 < height { y + 2 } else { height - 1 };
    let [w0, w1, w2] = taps;
    w0 * src[y * width + x]
        + w1 * (src[ym1 * width + x] + src[yp1 * width + x])
        + w2 * (src[ym2 * width + x] + src[yp2 * width + x])
}

/// Scalar whole-image horizontal pass.
pub fn gaussian5_h_scalar(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    for y in 0..height {
        let row = &src[y * width..(y + 1) * width];
        let drow = &mut dst[y * width..(y + 1) * width];
        for (x, d) in drow.iter_mut().enumerate() {
            *d = h_scalar_pixel(row, x, taps);
        }
    }
}

/// Scalar whole-image vertical pass.
pub fn gaussian5_v_scalar(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    for y in 0..height {
        for x in 0..width {
            dst[y * width + x] = v_scalar_pixel(src, x, y, width, height, taps);
        }
    }
}

/// incant scalar-tier fallback (horizontal).
pub fn gaussian5_h_impl_scalar(
    _token: archmage::ScalarToken,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    gaussian5_h_scalar(src, dst, width, height, taps)
}

/// incant scalar-tier fallback (vertical).
pub fn gaussian5_v_impl_scalar(
    _token: archmage::ScalarToken,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    gaussian5_v_scalar(src, dst, width, height, taps)
}

// ============================================================================
// magetypes SIMD bodies
// ============================================================================

#[magetypes(define(f32x8), v4, v3, neon, wasm128, -scalar)]
fn gaussian5_h_impl(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    // Interior needs x-2..x+9 in range: one 8-lane chunk starting at x=2
    // requires 2+9 <= width-1 → width >= 12.
    if width < 12 {
        gaussian5_h_scalar(src, dst, width, height, taps);
        return;
    }
    let w0 = f32x8::splat(token, taps[0]);
    let w1 = f32x8::splat(token, taps[1]);
    let w2 = f32x8::splat(token, taps[2]);

    for y in 0..height {
        let row = &src[y * width..(y + 1) * width];
        let drow = &mut dst[y * width..(y + 1) * width];

        // Edge columns x ∈ {0, 1, width-2, width-1} stay scalar (clamped taps).
        drow[0] = h_scalar_pixel(row, 0, taps);
        drow[1] = h_scalar_pixel(row, 1, taps);
        drow[width - 2] = h_scalar_pixel(row, width - 2, taps);
        drow[width - 1] = h_scalar_pixel(row, width - 1, taps);

        // Interior x ∈ [2, width-2). Lanes x..x+7 read x-2..x+9, safe while
        // x + 9 <= width - 1, i.e. x + 10 <= width.
        let mut x = 2usize;
        while x + 10 <= width {
            let c = f32x8::from_slice(token, &row[x..x + 8]);
            let m1 = f32x8::from_slice(token, &row[x - 1..x + 7]);
            let p1 = f32x8::from_slice(token, &row[x + 1..x + 9]);
            let m2 = f32x8::from_slice(token, &row[x - 2..x + 6]);
            let p2 = f32x8::from_slice(token, &row[x + 2..x + 10]);
            // Same expression tree as the scalar reference — no fma.
            let v = w0 * c + w1 * (m1 + p1) + w2 * (m2 + p2);
            v.store((&mut drow[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        for xi in x..width - 2 {
            drow[xi] = h_scalar_pixel(row, xi, taps);
        }
    }
}

#[magetypes(define(f32x8), v4, v3, neon, wasm128, -scalar)]
fn gaussian5_v_impl(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    if height < 5 {
        gaussian5_v_scalar(src, dst, width, height, taps);
        return;
    }
    let w0 = f32x8::splat(token, taps[0]);
    let w1 = f32x8::splat(token, taps[1]);
    let w2 = f32x8::splat(token, taps[2]);

    // Interior rows y ∈ [2, height-2): vectorize the full row over x
    // (contiguous strided loads, no x clamps needed).
    for y in 2..height - 2 {
        let c = &src[y * width..(y + 1) * width];
        let m1 = &src[(y - 1) * width..(y - 1) * width + width];
        let p1 = &src[(y + 1) * width..(y + 1) * width + width];
        let m2 = &src[(y - 2) * width..(y - 2) * width + width];
        let p2 = &src[(y + 2) * width..(y + 2) * width + width];
        let drow = &mut dst[y * width..(y + 1) * width];

        let mut x = 0usize;
        while x + 8 <= width {
            let cv = f32x8::from_slice(token, &c[x..x + 8]);
            let m1v = f32x8::from_slice(token, &m1[x..x + 8]);
            let p1v = f32x8::from_slice(token, &p1[x..x + 8]);
            let m2v = f32x8::from_slice(token, &m2[x..x + 8]);
            let p2v = f32x8::from_slice(token, &p2[x..x + 8]);
            let v = w0 * cv + w1 * (m1v + p1v) + w2 * (m2v + p2v);
            v.store((&mut drow[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        for xi in x..width {
            drow[xi] = v_scalar_pixel(src, xi, y, width, height, taps);
        }
    }

    // Edge rows (y < 2, y >= height-2): scalar with y clamps.
    for y in [0, 1, height - 2, height - 1] {
        for x in 0..width {
            dst[y * width + x] = v_scalar_pixel(src, x, y, width, height, taps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{gen_f32, run_dispatch_parity};
    use alloc::vec;

    const TAPS: [f32; 3] = [0.4, 0.2, 0.1];

    /// Dispatched kernels must be bit-identical to the scalar reference on
    /// every lane-boundary size and token permutation.
    #[test]
    fn gaussian5_scalar_vs_dispatch_sizes() {
        let cases: &[(usize, usize)] = &[
            (1, 1),
            (4, 4),
            (8, 8),
            (9, 9),
            (11, 11),
            (12, 8),
            (16, 16),
            (17, 17),
            (32, 16),
            (33, 17),
            (64, 32),
            (65, 9),
        ];
        for &(w, h) in cases {
            let n = w * h;
            let input = gen_f32(0x5EED_CAFE ^ ((w as u64) << 32) ^ h as u64, n, 5.0);

            let mut ref_h = vec![0.0_f32; n];
            let mut ref_v = vec![0.0_f32; n];
            gaussian5_h_scalar(&input, &mut ref_h, w, h, TAPS);
            gaussian5_v_scalar(&input, &mut ref_v, w, h, TAPS);

            run_dispatch_parity(|_perm| {
                let mut act_h = vec![0.0_f32; n];
                let mut act_v = vec![0.0_f32; n];
                gaussian5_horizontal(&input, &mut act_h, w, h, TAPS);
                gaussian5_vertical(&input, &mut act_v, w, h, TAPS);
                assert_eq!(act_h, ref_h, "horizontal w={w} h={h}");
                assert_eq!(act_v, ref_v, "vertical w={w} h={h}");
            });
        }
    }
}
