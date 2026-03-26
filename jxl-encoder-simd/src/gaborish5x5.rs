// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Gaborish inverse: 5x5 symmetric sharpening kernel with SIMD acceleration.
//!
//! The kernel has 6 weight classes:
//! ```text
//!   D  L  R  L  D
//!   L  d  r  d  L
//!   R  r  c  r  R
//!   L  d  r  d  L
//!   D  L  R  L  D
//! ```
//! where c=center, r=orthogonal(1), d=diagonal(√2), R=orthogonal(2),
//! L=knight's move, D=corner(2√2).

/// Apply the 5x5 gaborish inverse kernel to a single channel.
///
/// `data` is modified in place. `scratch` is used as temporary input copy.
/// Both must be at least `width * height` elements.
///
/// The 6 weights (wc, wr, wd, w_big_r, wl, w_big_d) should already be
/// normalized (sum to 1.0).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn gaborish_5x5_channel(
    data: &mut [f32],
    scratch: &mut [f32],
    width: usize,
    height: usize,
    wc: f32,
    wr: f32,
    wd: f32,
    w_big_r: f32,
    wl: f32,
    w_big_d: f32,
) {
    let n = width * height;
    debug_assert!(data.len() >= n);
    debug_assert!(scratch.len() >= n);

    // Copy input to scratch
    scratch[..n].copy_from_slice(&data[..n]);

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            gaborish_5x5_avx2(
                token, data, scratch, width, height, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            gaborish_5x5_neon(
                token, data, scratch, width, height, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            return;
        }
    }

    gaborish_5x5_scalar(
        data, scratch, width, height, wc, wr, wd, w_big_r, wl, w_big_d,
    );
}

// ============================================================================
// Scalar fallback
// ============================================================================

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn gaborish_5x5_scalar(
    output: &mut [f32],
    input: &[f32],
    width: usize,
    height: usize,
    wc: f32,
    wr: f32,
    wd: f32,
    w_big_r: f32,
    wl: f32,
    w_big_d: f32,
) {
    let px = |x: isize, y: isize| -> f32 {
        let cx = x.clamp(0, (width - 1) as isize) as usize;
        let cy = y.clamp(0, (height - 1) as isize) as usize;
        input[cy * width + cx]
    };

    for y in 0..height {
        let iy = y as isize;
        for x in 0..width {
            let ix = x as isize;

            let mut val = wc * px(ix, iy);

            // r: 4 orthogonal neighbors at distance 1
            val += wr * (px(ix - 1, iy) + px(ix + 1, iy) + px(ix, iy - 1) + px(ix, iy + 1));

            // d: 4 diagonal neighbors at distance sqrt(2)
            val += wd
                * (px(ix - 1, iy - 1)
                    + px(ix + 1, iy - 1)
                    + px(ix - 1, iy + 1)
                    + px(ix + 1, iy + 1));

            // R: 4 orthogonal neighbors at distance 2
            val += w_big_r * (px(ix - 2, iy) + px(ix + 2, iy) + px(ix, iy - 2) + px(ix, iy + 2));

            // L: 8 knight's move neighbors
            val += wl
                * (px(ix - 2, iy - 1)
                    + px(ix - 2, iy + 1)
                    + px(ix + 2, iy - 1)
                    + px(ix + 2, iy + 1)
                    + px(ix - 1, iy - 2)
                    + px(ix + 1, iy - 2)
                    + px(ix - 1, iy + 2)
                    + px(ix + 1, iy + 2));

            // D: 4 corner neighbors at distance 2*sqrt(2)
            val += w_big_d
                * (px(ix - 2, iy - 2)
                    + px(ix + 2, iy - 2)
                    + px(ix - 2, iy + 2)
                    + px(ix + 2, iy + 2));

            output[y * width + x] = val;
        }
    }
}

// ============================================================================
// x86_64 AVX2+FMA implementation
// ============================================================================

/// AVX2+FMA gaborish 5x5: processes 8 pixels per iteration in interior region.
/// Border pixels (within 2 of edge) use scalar fallback.
#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn gaborish_5x5_avx2(
    token: archmage::X64V3Token,
    output: &mut [f32],
    input: &[f32],
    width: usize,
    height: usize,
    wc: f32,
    wr: f32,
    wd: f32,
    w_big_r: f32,
    wl: f32,
    w_big_d: f32,
) {
    use magetypes::simd::f32x8;

    // For images too small for SIMD interior (need x-2..x+8+2), use scalar
    if width < 13 || height < 5 {
        gaborish_5x5_scalar(
            output, input, width, height, wc, wr, wd, w_big_r, wl, w_big_d,
        );
        return;
    }

    let wc_v = f32x8::splat(token, wc);
    let wr_v = f32x8::splat(token, wr);
    let wd_v = f32x8::splat(token, wd);
    let w_big_r_v = f32x8::splat(token, w_big_r);
    let wl_v = f32x8::splat(token, wl);
    let w_big_d_v = f32x8::splat(token, w_big_d);

    // Scalar helper for border pixels
    let px = |x: isize, y: isize| -> f32 {
        let cx = x.clamp(0, (width - 1) as isize) as usize;
        let cy = y.clamp(0, (height - 1) as isize) as usize;
        input[cy * width + cx]
    };

    let scalar_pixel = |ix: isize, iy: isize| -> f32 {
        let mut val = wc * px(ix, iy);
        val += wr * (px(ix - 1, iy) + px(ix + 1, iy) + px(ix, iy - 1) + px(ix, iy + 1));
        val += wd
            * (px(ix - 1, iy - 1) + px(ix + 1, iy - 1) + px(ix - 1, iy + 1) + px(ix + 1, iy + 1));
        val += w_big_r * (px(ix - 2, iy) + px(ix + 2, iy) + px(ix, iy - 2) + px(ix, iy + 2));
        val += wl
            * (px(ix - 2, iy - 1)
                + px(ix - 2, iy + 1)
                + px(ix + 2, iy - 1)
                + px(ix + 2, iy + 1)
                + px(ix - 1, iy - 2)
                + px(ix + 1, iy - 2)
                + px(ix - 1, iy + 2)
                + px(ix + 1, iy + 2));
        val += w_big_d
            * (px(ix - 2, iy - 2) + px(ix + 2, iy - 2) + px(ix - 2, iy + 2) + px(ix + 2, iy + 2));
        val
    };

    for y in 0..height {
        let iy = y as isize;

        // Border rows (y < 2 or y >= height-2): all scalar
        if y < 2 || y >= height - 2 {
            for x in 0..width {
                output[y * width + x] = scalar_pixel(x as isize, iy);
            }
            continue;
        }

        // Interior row: scalar left border (x < 2)
        for x in 0..2 {
            output[y * width + x] = scalar_pixel(x as isize, iy);
        }

        // Pre-slice rows to help compiler eliminate bounds checks.
        // Each row has exactly `width` elements; the loop bound guarantees
        // all f32x8 loads (offset range -2..+9) stay within the row.
        let r_m2 = (y - 2) * width;
        let r_m1 = (y - 1) * width;
        let r_0 = y * width;
        let r_p1 = (y + 1) * width;
        let r_p2 = (y + 2) * width;
        let row_m2 = &input[r_m2..r_m2 + width];
        let row_m1 = &input[r_m1..r_m1 + width];
        let row_0 = &input[r_0..r_0 + width];
        let row_p1 = &input[r_p1..r_p1 + width];
        let row_p2 = &input[r_p2..r_p2 + width];

        // SIMD interior: loads access x-2..x+10, so need x + 10 <= width.
        // For widths that are multiples of 8, width-10 and width-8 produce
        // identical iteration counts (x starts at 2, steps by 8).
        let simd_end = if width >= 12 { width - 10 } else { 2 };
        let mut x = 2;

        while x < simd_end {
            // Center
            let center = crate::load_f32x8(token, row_0, x);

            // r: 4 orthogonal at distance 1
            let left1 = crate::load_f32x8(token, row_0, x - 1);
            let right1 = crate::load_f32x8(token, row_0, x + 1);
            let top1 = crate::load_f32x8(token, row_m1, x);
            let bot1 = crate::load_f32x8(token, row_p1, x);
            let r_sum = left1 + right1 + top1 + bot1;

            // d: 4 diagonal at distance sqrt(2)
            let tl1 = crate::load_f32x8(token, row_m1, x - 1);
            let tr1 = crate::load_f32x8(token, row_m1, x + 1);
            let bl1 = crate::load_f32x8(token, row_p1, x - 1);
            let br1 = crate::load_f32x8(token, row_p1, x + 1);
            let d_sum = tl1 + tr1 + bl1 + br1;

            // R: 4 orthogonal at distance 2
            let left2 = crate::load_f32x8(token, row_0, x - 2);
            let right2 = crate::load_f32x8(token, row_0, x + 2);
            let top2 = crate::load_f32x8(token, row_m2, x);
            let bot2 = crate::load_f32x8(token, row_p2, x);
            let big_r_sum = left2 + right2 + top2 + bot2;

            // L: 8 knight's move neighbors
            let l_a = crate::load_f32x8(token, row_m1, x - 2);
            let l_b = crate::load_f32x8(token, row_p1, x - 2);
            let l_c = crate::load_f32x8(token, row_m1, x + 2);
            let l_d = crate::load_f32x8(token, row_p1, x + 2);
            let l_e = crate::load_f32x8(token, row_m2, x - 1);
            let l_f = crate::load_f32x8(token, row_m2, x + 1);
            let l_g = crate::load_f32x8(token, row_p2, x - 1);
            let l_h = crate::load_f32x8(token, row_p2, x + 1);
            let l_sum = l_a + l_b + l_c + l_d + l_e + l_f + l_g + l_h;

            // D: 4 corner at distance 2*sqrt(2)
            let tl2 = crate::load_f32x8(token, row_m2, x - 2);
            let tr2 = crate::load_f32x8(token, row_m2, x + 2);
            let bl2 = crate::load_f32x8(token, row_p2, x - 2);
            let br2 = crate::load_f32x8(token, row_p2, x + 2);
            let big_d_sum = tl2 + tr2 + bl2 + br2;

            // Combine with FMA chains:
            // result = wc*center + wr*r_sum + wd*d_sum + w_big_r*big_r_sum + wl*l_sum + w_big_d*big_d_sum
            let result = wc_v.mul_add(
                center,
                wr_v.mul_add(
                    r_sum,
                    wd_v.mul_add(
                        d_sum,
                        w_big_r_v.mul_add(big_r_sum, wl_v.mul_add(l_sum, w_big_d_v * big_d_sum)),
                    ),
                ),
            );

            crate::store_f32x8(output, r_0 + x, result);

            x += 8;
        }

        // Scalar right border + remainder
        while x < width {
            output[y * width + x] = scalar_pixel(x as isize, iy);
            x += 1;
        }
    }
}

// ============================================================================
// aarch64 NEON implementation
// ============================================================================

/// NEON gaborish 5x5: processes 4 pixels per iteration in interior region.
#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn gaborish_5x5_neon(
    token: archmage::NeonToken,
    output: &mut [f32],
    input: &[f32],
    width: usize,
    height: usize,
    wc: f32,
    wr: f32,
    wd: f32,
    w_big_r: f32,
    wl: f32,
    w_big_d: f32,
) {
    use magetypes::simd::f32x4;

    if width < 9 || height < 5 {
        gaborish_5x5_scalar(
            output, input, width, height, wc, wr, wd, w_big_r, wl, w_big_d,
        );
        return;
    }

    let wc_v = f32x4::splat(token, wc);
    let wr_v = f32x4::splat(token, wr);
    let wd_v = f32x4::splat(token, wd);
    let w_big_r_v = f32x4::splat(token, w_big_r);
    let wl_v = f32x4::splat(token, wl);
    let w_big_d_v = f32x4::splat(token, w_big_d);

    let px = |x: isize, y: isize| -> f32 {
        let cx = x.clamp(0, (width - 1) as isize) as usize;
        let cy = y.clamp(0, (height - 1) as isize) as usize;
        input[cy * width + cx]
    };

    let scalar_pixel = |ix: isize, iy: isize| -> f32 {
        let mut val = wc * px(ix, iy);
        val += wr * (px(ix - 1, iy) + px(ix + 1, iy) + px(ix, iy - 1) + px(ix, iy + 1));
        val += wd
            * (px(ix - 1, iy - 1) + px(ix + 1, iy - 1) + px(ix - 1, iy + 1) + px(ix + 1, iy + 1));
        val += w_big_r * (px(ix - 2, iy) + px(ix + 2, iy) + px(ix, iy - 2) + px(ix, iy + 2));
        val += wl
            * (px(ix - 2, iy - 1)
                + px(ix - 2, iy + 1)
                + px(ix + 2, iy - 1)
                + px(ix + 2, iy + 1)
                + px(ix - 1, iy - 2)
                + px(ix + 1, iy - 2)
                + px(ix - 1, iy + 2)
                + px(ix + 1, iy + 2));
        val += w_big_d
            * (px(ix - 2, iy - 2) + px(ix + 2, iy - 2) + px(ix - 2, iy + 2) + px(ix + 2, iy + 2));
        val
    };

    for y in 0..height {
        let iy = y as isize;

        if y < 2 || y >= height - 2 {
            for x in 0..width {
                output[y * width + x] = scalar_pixel(x as isize, iy);
            }
            continue;
        }

        for x in 0..2 {
            output[y * width + x] = scalar_pixel(x as isize, iy);
        }

        let r_m2 = (y - 2) * width;
        let r_m1 = (y - 1) * width;
        let r_0 = y * width;
        let r_p1 = (y + 1) * width;
        let r_p2 = (y + 2) * width;

        let simd_end = if width >= 6 { width - 4 } else { 2 };
        let mut x = 2;

        while x < simd_end {
            let center = f32x4::from_slice(token, &input[r_0 + x..]);

            let r_sum = f32x4::from_slice(token, &input[r_0 + x - 1..])
                + f32x4::from_slice(token, &input[r_0 + x + 1..])
                + f32x4::from_slice(token, &input[r_m1 + x..])
                + f32x4::from_slice(token, &input[r_p1 + x..]);

            let d_sum = f32x4::from_slice(token, &input[r_m1 + x - 1..])
                + f32x4::from_slice(token, &input[r_m1 + x + 1..])
                + f32x4::from_slice(token, &input[r_p1 + x - 1..])
                + f32x4::from_slice(token, &input[r_p1 + x + 1..]);

            let big_r_sum = f32x4::from_slice(token, &input[r_0 + x - 2..])
                + f32x4::from_slice(token, &input[r_0 + x + 2..])
                + f32x4::from_slice(token, &input[r_m2 + x..])
                + f32x4::from_slice(token, &input[r_p2 + x..]);

            let l_sum = f32x4::from_slice(token, &input[r_m1 + x - 2..])
                + f32x4::from_slice(token, &input[r_p1 + x - 2..])
                + f32x4::from_slice(token, &input[r_m1 + x + 2..])
                + f32x4::from_slice(token, &input[r_p1 + x + 2..])
                + f32x4::from_slice(token, &input[r_m2 + x - 1..])
                + f32x4::from_slice(token, &input[r_m2 + x + 1..])
                + f32x4::from_slice(token, &input[r_p2 + x - 1..])
                + f32x4::from_slice(token, &input[r_p2 + x + 1..]);

            let big_d_sum = f32x4::from_slice(token, &input[r_m2 + x - 2..])
                + f32x4::from_slice(token, &input[r_m2 + x + 2..])
                + f32x4::from_slice(token, &input[r_p2 + x - 2..])
                + f32x4::from_slice(token, &input[r_p2 + x + 2..]);

            let result = wc_v.mul_add(
                center,
                wr_v.mul_add(
                    r_sum,
                    wd_v.mul_add(
                        d_sum,
                        w_big_r_v.mul_add(big_r_sum, wl_v.mul_add(l_sum, w_big_d_v * big_d_sum)),
                    ),
                ),
            );

            let out_arr: &mut [f32; 4] = (&mut output[r_0 + x..r_0 + x + 4]).try_into().unwrap();
            result.store(out_arr);
            x += 4;
        }

        while x < width {
            output[y * width + x] = scalar_pixel(x as isize, iy);
            x += 1;
        }
    }
}
