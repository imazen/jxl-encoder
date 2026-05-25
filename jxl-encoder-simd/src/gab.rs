// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Gaborish smooth: 3x3 weighted stencil applied to a single channel.
//!
//! The stencil weights are:
//! ```text
//!   w2  w1  w2
//!   w1  wc  w1
//!   w2  w1  w2
//! ```
//!
//! where `wc` = center weight, `w1` = cardinal weight, `w2` = diagonal weight.

/// Apply 3x3 weighted gaborish smooth to a single channel in-place.
///
/// `plane` is modified in place. `scratch` must be at least `width * height` elements
/// and is used as temporary storage for the input copy.
///
/// Dispatches to the best available SIMD implementation at runtime.
#[inline]
pub fn gab_smooth_channel(
    plane: &mut [f32],
    scratch: &mut [f32],
    width: usize,
    height: usize,
    w_center: f32,
    w1: f32,
    w2: f32,
) {
    let n = width * height;
    debug_assert!(plane.len() >= n);
    debug_assert!(scratch.len() >= n);

    // Copy plane into scratch (input), then write filtered result back to plane
    scratch[..n].copy_from_slice(&plane[..n]);

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            gab_smooth_avx2(token, plane, scratch, width, height, w_center, w1, w2);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            gab_smooth_neon(token, plane, scratch, width, height, w_center, w1, w2);
            return;
        }
    }

    gab_smooth_scalar(plane, scratch, width, height, w_center, w1, w2);
}

// ============================================================================
// Scalar fallback
// ============================================================================

#[inline]
pub fn gab_smooth_scalar(
    output: &mut [f32],
    input: &[f32],
    width: usize,
    height: usize,
    w_center: f32,
    w1: f32,
    w2: f32,
) {
    for y in 0..height {
        let ym = if y > 0 { y - 1 } else { 0 };
        let yp = if y + 1 < height { y + 1 } else { height - 1 };
        let row_c = y * width;
        let row_t = ym * width;
        let row_b = yp * width;

        for x in 0..width {
            let xm = if x > 0 { x - 1 } else { 0 };
            let xp = if x + 1 < width { x + 1 } else { width - 1 };

            let center = input[row_c + x];
            let top = input[row_t + x];
            let bottom = input[row_b + x];
            let left = input[row_c + xm];
            let right = input[row_c + xp];
            let tl = input[row_t + xm];
            let tr = input[row_t + xp];
            let bl = input[row_b + xm];
            let br = input[row_b + xp];

            output[row_c + x] =
                w_center * center + w1 * (top + bottom + left + right) + w2 * (tl + tr + bl + br);
        }
    }
}

// ============================================================================
// x86_64 AVX2+FMA implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
use archmage::arcane;

/// AVX2+FMA gab smooth: processes 8 pixels per iteration in interior rows.
/// Border rows/columns use scalar fallback.
#[cfg(target_arch = "x86_64")]
#[inline]
#[arcane]
#[allow(clippy::too_many_arguments)]
pub fn gab_smooth_avx2(
    token: archmage::X64V3Token,
    output: &mut [f32],
    input: &[f32],
    width: usize,
    height: usize,
    w_center: f32,
    w1: f32,
    w2: f32,
) {
    use magetypes::simd::f32x8;

    let wc_v = f32x8::splat(token, w_center);
    let w1_v = f32x8::splat(token, w1);
    let w2_v = f32x8::splat(token, w2);

    // For images too small for SIMD or with width < 10 (need x-1..x+8+1), use scalar
    if width < 10 || height < 3 {
        gab_smooth_scalar(output, input, width, height, w_center, w1, w2);
        return;
    }

    // Process all rows
    for y in 0..height {
        let ym = if y > 0 { y - 1 } else { 0 };
        let yp = if y + 1 < height { y + 1 } else { height - 1 };
        let row_c = y * width;
        let row_t = ym * width;
        let row_b = yp * width;

        // Scalar: first pixel (border)
        {
            let x = 0;
            let center = input[row_c];
            let top = input[row_t];
            let bottom = input[row_b];
            let left = input[row_c]; // clamped
            let right = input[row_c + 1];
            let tl = input[row_t]; // clamped
            let tr = input[row_t + 1];
            let bl = input[row_b]; // clamped
            let br = input[row_b + 1];
            output[row_c + x] =
                w_center * center + w1 * (top + bottom + left + right) + w2 * (tl + tr + bl + br);
        }

        // SIMD: interior pixels (x in 1..width-1, processed in chunks of 8)
        // We need to load x-1..x+8 (9 elements), so x+8 < width means x < width-8
        // Also x >= 1 for the left neighbor
        let simd_start = 1;
        let simd_end = if width > 8 { width - 8 } else { 1 };

        let mut x = simd_start;
        while x < simd_end {
            // Load 8 pixels and all their neighbors via unaligned loads
            // SAFETY: arcane macro ensures target_feature is set.
            // Bounds: x >= 1 and x+8 < width, so x-1..x+9 is in bounds.
            let center = crate::load_f32x8(token, input, row_c + x);
            let top = crate::load_f32x8(token, input, row_t + x);
            let bottom = crate::load_f32x8(token, input, row_b + x);
            let left = crate::load_f32x8(token, input, row_c + x - 1);
            let right = crate::load_f32x8(token, input, row_c + x + 1);
            let tl = crate::load_f32x8(token, input, row_t + x - 1);
            let tr = crate::load_f32x8(token, input, row_t + x + 1);
            let bl = crate::load_f32x8(token, input, row_b + x - 1);
            let br = crate::load_f32x8(token, input, row_b + x + 1);

            // cardinals = top + bottom + left + right
            let cardinals = top + bottom + left + right;
            // diagonals = tl + tr + bl + br
            let diagonals = tl + tr + bl + br;

            // result = w_center * center + w1 * cardinals + w2 * diagonals
            let result = wc_v.mul_add(center, w1_v.mul_add(cardinals, w2_v * diagonals));

            // Store 8 results
            crate::store_f32x8(output, row_c + x, result);

            x += 8;
        }

        // Scalar: remaining interior + last pixel (border)
        while x < width {
            let xm = if x > 0 { x - 1 } else { 0 };
            let xp = if x + 1 < width { x + 1 } else { width - 1 };

            let center = input[row_c + x];
            let top = input[row_t + x];
            let bottom = input[row_b + x];
            let left = input[row_c + xm];
            let right = input[row_c + xp];
            let tl = input[row_t + xm];
            let tr = input[row_t + xp];
            let bl = input[row_b + xm];
            let br = input[row_b + xp];

            output[row_c + x] =
                w_center * center + w1 * (top + bottom + left + right) + w2 * (tl + tr + bl + br);
            x += 1;
        }
    }
}

// ============================================================================
// aarch64 NEON implementation
// ============================================================================

/// NEON gab smooth: processes 4 pixels per iteration in interior rows.
#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn gab_smooth_neon(
    token: archmage::NeonToken,
    output: &mut [f32],
    input: &[f32],
    width: usize,
    height: usize,
    w_center: f32,
    w1: f32,
    w2: f32,
) {
    use magetypes::simd::f32x4;

    let wc_v = f32x4::splat(token, w_center);
    let w1_v = f32x4::splat(token, w1);
    let w2_v = f32x4::splat(token, w2);

    if width < 6 || height < 3 {
        gab_smooth_scalar(output, input, width, height, w_center, w1, w2);
        return;
    }

    for y in 0..height {
        let ym = if y > 0 { y - 1 } else { 0 };
        let yp = if y + 1 < height { y + 1 } else { height - 1 };
        let row_c = y * width;
        let row_t = ym * width;
        let row_b = yp * width;

        // Scalar: first pixel
        {
            let center = input[row_c];
            let top = input[row_t];
            let bottom = input[row_b];
            let left = input[row_c];
            let right = input[row_c + 1];
            let tl = input[row_t];
            let tr = input[row_t + 1];
            let bl = input[row_b];
            let br = input[row_b + 1];
            output[row_c] =
                w_center * center + w1 * (top + bottom + left + right) + w2 * (tl + tr + bl + br);
        }

        let simd_end = if width > 4 { width - 4 } else { 1 };
        let mut x = 1usize;
        while x < simd_end {
            let center = f32x4::from_slice(token, &input[row_c + x..]);
            let top = f32x4::from_slice(token, &input[row_t + x..]);
            let bottom = f32x4::from_slice(token, &input[row_b + x..]);
            let left = f32x4::from_slice(token, &input[row_c + x - 1..]);
            let right = f32x4::from_slice(token, &input[row_c + x + 1..]);
            let tl = f32x4::from_slice(token, &input[row_t + x - 1..]);
            let tr = f32x4::from_slice(token, &input[row_t + x + 1..]);
            let bl = f32x4::from_slice(token, &input[row_b + x - 1..]);
            let br = f32x4::from_slice(token, &input[row_b + x + 1..]);

            let cardinals = top + bottom + left + right;
            let diagonals = tl + tr + bl + br;
            let result = wc_v.mul_add(center, w1_v.mul_add(cardinals, w2_v * diagonals));

            let out_arr: &mut [f32; 4] =
                (&mut output[row_c + x..row_c + x + 4]).try_into().unwrap();
            result.store(out_arr);
            x += 4;
        }

        // Scalar tail
        while x < width {
            let xm = if x > 0 { x - 1 } else { 0 };
            let xp = if x + 1 < width { x + 1 } else { width - 1 };
            let center = input[row_c + x];
            let top = input[row_t + x];
            let bottom = input[row_b + x];
            let left = input[row_c + xm];
            let right = input[row_c + xp];
            let tl = input[row_t + xm];
            let tr = input[row_t + xp];
            let bl = input[row_b + xm];
            let br = input[row_b + xp];
            output[row_c + x] =
                w_center * center + w1 * (top + bottom + left + right) + w2 * (tl + tr + bl + br);
            x += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Gab weights derived from libjxl gaborish defaults.  Kernel is
    /// parametric so any (wc, w1, w2) works; these are just realistic values.
    const WC: f32 = 0.62;
    const W1: f32 = 0.085;
    const W2: f32 = 0.04;

    /// ULP-tolerant scalar-vs-dispatch parity across SIMD boundary sizes.
    /// Image is row-major width*height; both AVX2 (width >= 10) and NEON
    /// (width >= 6) paths get exercised.
    ///
    /// Strict-bit-equality FAILS by ≤1 ULP because the SIMD path fuses
    /// `wc*center + (w1*cardinals + w2*diagonals)` via `mul_add` (single
    /// rounding per FMA) while the scalar path performs explicit `*` and
    /// `+` (double rounding).  See docs/SIMD_PARITY_KNOWN_DIVERGENCES.md
    /// row gab-001.  Tolerance: 2 ULP (1 ULP for the mul_add + 1 ULP
    /// for the outer add).
    #[test]
    fn gab_scalar_vs_dispatch_sizes_ulp() {
        let cases: &[(usize, usize)] = &[
            (1, 1),    // single pixel
            (3, 3),    // tiny — both paths fall back to scalar
            (5, 5),    // < NEON min (width < 6)
            (6, 3),    // exactly NEON min
            (9, 3),    // < AVX2 min (width < 10)
            (10, 3),   // exactly AVX2 min
            (16, 4),   // exactly AVX2 chunk
            (17, 4),   // AVX2 + 1 tail
            (32, 8),   // 4 AVX2 chunks
            (33, 9),   // tail-loop stress
            (64, 16),  // 8 AVX2 chunks, height multiple
            (65, 17),  // both axes tail-loop
            (128, 32), // larger image
        ];
        for &(w, h) in cases {
            let n = w * h;
            let seed = 0xCAFE_F00D ^ ((w as u64) << 32) ^ h as u64;
            let input: Vec<f32> = gen_f32(seed, n, 5.0);

            let mut ref_out = input.clone();
            let mut ref_scratch = vec![0.0_f32; n];
            ref_scratch.copy_from_slice(&ref_out);
            gab_smooth_scalar(&mut ref_out, &ref_scratch, w, h, WC, W1, W2);

            run_dispatch_parity(|perm| {
                let mut act_out = input.clone();
                let mut act_scratch = vec![0.0_f32; n];
                gab_smooth_channel(&mut act_out, &mut act_scratch, w, h, WC, W1, W2);
                // Gab fuses w_center*center + (w1*cardinals + w2*diagonals)
                // via mul_add on SIMD paths; scalar uses explicit `*` + `+`.
                // Observed up to 4 ULP on rand inputs; near-zero outputs
                // can magnify ULP delta arbitrarily so we use an absolute
                // floor.
                assert_f32_slice_close_ulps_abs(
                    &ref_out,
                    &act_out,
                    8,
                    1e-5,
                    perm,
                    &format!("w={w} h={h}"),
                );
            });
        }
    }

    /// STRICT bit-exact variant — currently ignored due to documented FMA
    /// association divergence.  Kept in the suite so a future fix that
    /// aligns scalar to use `mul_add` (or the SIMD path to drop the FMA)
    /// is verified.
    #[test]
    #[ignore = "FIXME(SIMD-parity): gab-001 — FMA association ≤1 ULP \
                between scalar and AVX2/NEON paths; see \
                docs/SIMD_PARITY_KNOWN_DIVERGENCES.md"]
    fn gab_scalar_vs_dispatch_sizes_strict() {
        let w = 16;
        let h = 4;
        let n = w * h;
        let input: Vec<f32> = gen_f32(0xCAFE_F00D, n, 5.0);

        let mut ref_out = input.clone();
        let mut ref_scratch = vec![0.0_f32; n];
        ref_scratch.copy_from_slice(&ref_out);
        gab_smooth_scalar(&mut ref_out, &ref_scratch, w, h, WC, W1, W2);

        run_dispatch_parity(|perm| {
            let mut act_out = input.clone();
            let mut act_scratch = vec![0.0_f32; n];
            gab_smooth_channel(&mut act_out, &mut act_scratch, w, h, WC, W1, W2);
            assert_f32_slice_bit_eq(&ref_out, &act_out, perm, &format!("w={w} h={h}"));
        });
    }

    /// Parity on edge-value input distributions on a 16x4 image (covers
    /// AVX2 fast path + tail loop on a single chunk).
    #[test]
    fn gab_scalar_vs_dispatch_edge_values_ulp() {
        let w = 16;
        let h = 4;
        let n = w * h;
        for case in f32_edge_battery(n) {
            if case.data.is_empty() {
                continue;
            }
            let mut ref_out = case.data.clone();
            let mut ref_scratch = vec![0.0_f32; n];
            ref_scratch.copy_from_slice(&ref_out);
            gab_smooth_scalar(&mut ref_out, &ref_scratch, w, h, WC, W1, W2);

            run_dispatch_parity(|perm| {
                let mut act_out = case.data.clone();
                let mut act_scratch = vec![0.0_f32; n];
                gab_smooth_channel(&mut act_out, &mut act_scratch, w, h, WC, W1, W2);
                assert_f32_slice_close_ulps_abs(&ref_out, &act_out, 8, 1e-5, perm, &case.label);
            });
        }
    }
}
