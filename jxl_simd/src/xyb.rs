// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated XYB ↔ Linear RGB color conversion.
//!
//! Forward (linear RGB → XYB): matrix multiply + cube root + mix.
//! Inverse (XYB → linear RGB): unmix + cube + inverse matrix multiply.
//!
//! The cube root uses Newton-Raphson in f64 with bit-manipulation initial guess,
//! following the proven approach from fast-ssim2/yuvxyb.
//!
//! Data layout: separate channel buffers (SoA), not interleaved.

// --- Constants ---

// Opsin absorbance matrix (libjxl cms/opsin_params.h)
const OPSIN_MATRIX: [[f32; 3]; 3] = [
    [0.30, 0.622, 0.078],
    [0.23, 0.692, 0.078],
    [0.243_422_69, 0.204_767_45, 0.551_809_87],
];

// Inverse opsin absorbance matrix
#[allow(clippy::excessive_precision)]
const INV_OPSIN: [[f32; 3]; 3] = [
    [11.031_566_9, -9.866_943_9, -0.164_623],
    [-3.254_147_4, 4.418_770_4, -0.164_623],
    [-3.658_851_3, 2.712_923, 1.945_928_2],
];

// Bias added before cube root
#[allow(clippy::excessive_precision)]
const OPSIN_BIAS: [f32; 3] = [0.003_793_073_4; 3];

// Precomputed -cbrt(bias) ≈ -0.15595420054
#[allow(clippy::excessive_precision)]
const NEG_CBRT_BIAS: [f32; 3] = [-0.155_954_2; 3];

// --- Forward XYB (linear RGB → XYB) ---

/// Convert separate R, G, B channel buffers to separate X, Y, B channel buffers.
///
/// All buffers must be at least `n` elements. Uses SIMD for the inner loop.
/// The cube root uses Newton-Raphson in f64 for precision.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn linear_rgb_to_xyb_batch(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let n = r
        .len()
        .min(g.len())
        .min(b.len())
        .min(x_out.len())
        .min(y_out.len())
        .min(b_out.len());

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            forward_xyb_avx2(token, r, g, b, x_out, y_out, b_out, n);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            forward_xyb_neon(token, r, g, b, x_out, y_out, b_out, n);
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            forward_xyb_wasm128(token, r, g, b, x_out, y_out, b_out, n);
            return;
        }
    }

    forward_xyb_scalar(r, g, b, x_out, y_out, b_out, n);
}

/// Convert separate X, Y, B channel buffers to planar linear RGB.
///
/// Output is three separate channel slices, each of length `n`.
/// This avoids the interleave overhead when the consumer needs planar data.
#[inline]
pub fn xyb_to_linear_rgb_planar(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    out_r: &mut [f32],
    out_g: &mut [f32],
    out_b: &mut [f32],
    n: usize,
) {
    debug_assert!(xyb_x.len() >= n && xyb_y.len() >= n && xyb_b.len() >= n);
    debug_assert!(out_r.len() >= n && out_g.len() >= n && out_b.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            inverse_xyb_planar_avx2(token, xyb_x, xyb_y, xyb_b, out_r, out_g, out_b, n);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            inverse_xyb_planar_neon(token, xyb_x, xyb_y, xyb_b, out_r, out_g, out_b, n);
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            inverse_xyb_planar_wasm128(token, xyb_x, xyb_y, xyb_b, out_r, out_g, out_b, n);
            return;
        }
    }

    inverse_xyb_planar_scalar(xyb_x, xyb_y, xyb_b, out_r, out_g, out_b, n);
}

/// Convert separate X, Y, B channel buffers to interleaved linear RGB.
///
/// Output is `[R0, G0, B0, R1, G1, B1, ...]` with length `3 * n`.
#[inline]
pub fn xyb_to_linear_rgb_batch(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    linear_rgb: &mut [f32],
    n: usize,
) {
    debug_assert!(xyb_x.len() >= n && xyb_y.len() >= n && xyb_b.len() >= n);
    debug_assert!(linear_rgb.len() >= n * 3);

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            inverse_xyb_avx2(token, xyb_x, xyb_y, xyb_b, linear_rgb, n);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            inverse_xyb_neon(token, xyb_x, xyb_y, xyb_b, linear_rgb, n);
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            inverse_xyb_wasm128(token, xyb_x, xyb_y, xyb_b, linear_rgb, n);
            return;
        }
    }

    inverse_xyb_scalar(xyb_x, xyb_y, xyb_b, linear_rgb, n);
}

// --- Scalar fallbacks ---

/// Newton-Raphson cube root with bit-manipulation initial guess.
/// 2 iterations in f64 gives ~1e-7 relative error.
#[inline]
fn cbrt_fast(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    const B1: u32 = 709_958_130;
    let ui = x.to_bits();
    let sign = ui & 0x8000_0000;
    let hx = ui & 0x7FFF_FFFF;
    let approx = hx / 3 + B1;
    let mut t = f64::from(f32::from_bits(sign | approx));
    let xf64 = f64::from(x);
    // First Newton iteration: t = t * (2x + t³) / (x + 2t³)
    let r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    // Second Newton iteration
    let r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    t as f32
}

#[inline]
pub fn forward_xyb_scalar(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    n: usize,
) {
    for i in 0..n {
        // Matrix multiply + bias (chained FMA for single-rounding parity with SIMD path)
        let mixed0 = OPSIN_MATRIX[0][0].mul_add(
            r[i],
            OPSIN_MATRIX[0][1].mul_add(g[i], OPSIN_MATRIX[0][2].mul_add(b[i], OPSIN_BIAS[0])),
        );
        let mixed1 = OPSIN_MATRIX[1][0].mul_add(
            r[i],
            OPSIN_MATRIX[1][1].mul_add(g[i], OPSIN_MATRIX[1][2].mul_add(b[i], OPSIN_BIAS[1])),
        );
        let mixed2 = OPSIN_MATRIX[2][0].mul_add(
            r[i],
            OPSIN_MATRIX[2][1].mul_add(g[i], OPSIN_MATRIX[2][2].mul_add(b[i], OPSIN_BIAS[2])),
        );

        // Clamp + cube root + bias offset
        let l = cbrt_fast(mixed0.max(0.0)) + NEG_CBRT_BIAS[0];
        let m = cbrt_fast(mixed1.max(0.0)) + NEG_CBRT_BIAS[1];
        let s = cbrt_fast(mixed2.max(0.0)) + NEG_CBRT_BIAS[2];

        // Mix into XYB
        x_out[i] = 0.5 * (l - m);
        y_out[i] = 0.5 * (l + m);
        b_out[i] = s;
    }
}

#[inline]
pub fn inverse_xyb_planar_scalar(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    out_r: &mut [f32],
    out_g: &mut [f32],
    out_b: &mut [f32],
    n: usize,
) {
    for i in 0..n {
        let x = xyb_x[i];
        let y = xyb_y[i];
        let b = xyb_b[i];

        let gamma_r = y + x - NEG_CBRT_BIAS[0];
        let gamma_g = y - x - NEG_CBRT_BIAS[1];
        let gamma_b = b - NEG_CBRT_BIAS[2];

        let mixed_r = gamma_r * gamma_r * gamma_r - OPSIN_BIAS[0];
        let mixed_g = gamma_g * gamma_g * gamma_g - OPSIN_BIAS[1];
        let mixed_b = gamma_b * gamma_b * gamma_b - OPSIN_BIAS[2];

        out_r[i] = INV_OPSIN[0][0].mul_add(
            mixed_r,
            INV_OPSIN[0][1].mul_add(mixed_g, INV_OPSIN[0][2] * mixed_b),
        );
        out_g[i] = INV_OPSIN[1][0].mul_add(
            mixed_r,
            INV_OPSIN[1][1].mul_add(mixed_g, INV_OPSIN[1][2] * mixed_b),
        );
        out_b[i] = INV_OPSIN[2][0].mul_add(
            mixed_r,
            INV_OPSIN[2][1].mul_add(mixed_g, INV_OPSIN[2][2] * mixed_b),
        );
    }
}

#[inline]
pub fn inverse_xyb_scalar(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    linear_rgb: &mut [f32],
    n: usize,
) {
    for i in 0..n {
        let x = xyb_x[i];
        let y = xyb_y[i];
        let b = xyb_b[i];

        // Unmix XYB to gamma-domain LMS + add cbrt(bias)
        let gamma_r = y + x - NEG_CBRT_BIAS[0];
        let gamma_g = y - x - NEG_CBRT_BIAS[1];
        let gamma_b = b - NEG_CBRT_BIAS[2];

        // Cube and subtract bias to get mixed (opsin LMS)
        let mixed_r = gamma_r * gamma_r * gamma_r - OPSIN_BIAS[0];
        let mixed_g = gamma_g * gamma_g * gamma_g - OPSIN_BIAS[1];
        let mixed_b = gamma_b * gamma_b * gamma_b - OPSIN_BIAS[2];

        // Inverse opsin matrix → linear RGB (chained FMA for SIMD parity)
        let r = INV_OPSIN[0][0].mul_add(
            mixed_r,
            INV_OPSIN[0][1].mul_add(mixed_g, INV_OPSIN[0][2] * mixed_b),
        );
        let g = INV_OPSIN[1][0].mul_add(
            mixed_r,
            INV_OPSIN[1][1].mul_add(mixed_g, INV_OPSIN[1][2] * mixed_b),
        );
        let b_lin = INV_OPSIN[2][0].mul_add(
            mixed_r,
            INV_OPSIN[2][1].mul_add(mixed_g, INV_OPSIN[2][2] * mixed_b),
        );

        linear_rgb[i * 3] = r;
        linear_rgb[i * 3 + 1] = g;
        linear_rgb[i * 3 + 2] = b_lin;
    }
}

// --- AVX2 implementations ---

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn forward_xyb_avx2(
    token: archmage::X64V3Token,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    n: usize,
) {
    use magetypes::simd::f32x8;

    let m00 = f32x8::splat(token, OPSIN_MATRIX[0][0]);
    let m01 = f32x8::splat(token, OPSIN_MATRIX[0][1]);
    let m02 = f32x8::splat(token, OPSIN_MATRIX[0][2]);
    let m10 = f32x8::splat(token, OPSIN_MATRIX[1][0]);
    let m11 = f32x8::splat(token, OPSIN_MATRIX[1][1]);
    let m12 = f32x8::splat(token, OPSIN_MATRIX[1][2]);
    let m20 = f32x8::splat(token, OPSIN_MATRIX[2][0]);
    let m21 = f32x8::splat(token, OPSIN_MATRIX[2][1]);
    let m22 = f32x8::splat(token, OPSIN_MATRIX[2][2]);
    let bias0 = f32x8::splat(token, OPSIN_BIAS[0]);
    let bias1 = f32x8::splat(token, OPSIN_BIAS[1]);
    let bias2 = f32x8::splat(token, OPSIN_BIAS[2]);
    let neg_cbrt0 = f32x8::splat(token, NEG_CBRT_BIAS[0]);
    let neg_cbrt1 = f32x8::splat(token, NEG_CBRT_BIAS[1]);
    let neg_cbrt2 = f32x8::splat(token, NEG_CBRT_BIAS[2]);
    let half = f32x8::splat(token, 0.5);
    let zero = f32x8::splat(token, 0.0);

    let chunks = n / 8;
    let simd_n = chunks * 8;
    let r_s = &r[..simd_n];
    let g_s = &g[..simd_n];
    let b_s = &b[..simd_n];

    for chunk in 0..chunks {
        let base = chunk * 8;
        let rv = crate::load_f32x8(token, r_s, base);
        let gv = crate::load_f32x8(token, g_s, base);
        let bv = crate::load_f32x8(token, b_s, base);

        // Matrix multiply + bias (FMA chains)
        let mixed0 = m00.mul_add(rv, m01.mul_add(gv, m02.mul_add(bv, bias0)));
        let mixed1 = m10.mul_add(rv, m11.mul_add(gv, m12.mul_add(bv, bias1)));
        let mixed2 = m20.mul_add(rv, m21.mul_add(gv, m22.mul_add(bv, bias2)));

        // Clamp negative to zero
        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        // Cube root: extract to scalar, Newton-Raphson, reload
        // This is the proven pattern from fast-ssim2 — precision-critical
        let mut m0_arr = [0.0f32; 8];
        let mut m1_arr = [0.0f32; 8];
        let mut m2_arr = [0.0f32; 8];
        mixed0.store(m0_arr.as_mut_slice().try_into().unwrap());
        mixed1.store(m1_arr.as_mut_slice().try_into().unwrap());
        mixed2.store(m2_arr.as_mut_slice().try_into().unwrap());
        for j in 0..8 {
            m0_arr[j] = cbrt_fast(m0_arr[j]);
            m1_arr[j] = cbrt_fast(m1_arr[j]);
            m2_arr[j] = cbrt_fast(m2_arr[j]);
        }
        let l = f32x8::from_slice(token, &m0_arr) + neg_cbrt0;
        let m = f32x8::from_slice(token, &m1_arr) + neg_cbrt1;
        let s = f32x8::from_slice(token, &m2_arr) + neg_cbrt2;

        // XYB mixing
        let xv = half * (l - m);
        let yv = half * (l + m);

        crate::store_f32x8(x_out, base, xv);
        crate::store_f32x8(y_out, base, yv);
        crate::store_f32x8(b_out, base, s);
    }

    // Scalar remainder
    let start = simd_n;
    forward_xyb_scalar(
        &r[start..],
        &g[start..],
        &b[start..],
        &mut x_out[start..],
        &mut y_out[start..],
        &mut b_out[start..],
        n - start,
    );
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
pub fn inverse_xyb_avx2(
    token: archmage::X64V3Token,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    linear_rgb: &mut [f32],
    n: usize,
) {
    use magetypes::simd::f32x8;

    let neg_cbrt0 = f32x8::splat(token, -NEG_CBRT_BIAS[0]); // positive: cbrt(bias)
    let neg_cbrt1 = f32x8::splat(token, -NEG_CBRT_BIAS[1]);
    let neg_cbrt2 = f32x8::splat(token, -NEG_CBRT_BIAS[2]);
    let neg_bias0 = f32x8::splat(token, -OPSIN_BIAS[0]);
    let neg_bias1 = f32x8::splat(token, -OPSIN_BIAS[1]);
    let neg_bias2 = f32x8::splat(token, -OPSIN_BIAS[2]);
    let inv00 = f32x8::splat(token, INV_OPSIN[0][0]);
    let inv01 = f32x8::splat(token, INV_OPSIN[0][1]);
    let inv02 = f32x8::splat(token, INV_OPSIN[0][2]);
    let inv10 = f32x8::splat(token, INV_OPSIN[1][0]);
    let inv11 = f32x8::splat(token, INV_OPSIN[1][1]);
    let inv12 = f32x8::splat(token, INV_OPSIN[1][2]);
    let inv20 = f32x8::splat(token, INV_OPSIN[2][0]);
    let inv21 = f32x8::splat(token, INV_OPSIN[2][1]);
    let inv22 = f32x8::splat(token, INV_OPSIN[2][2]);

    let chunks = n / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let x = crate::load_f32x8(token, xyb_x, base);
        let y = crate::load_f32x8(token, xyb_y, base);
        let b = crate::load_f32x8(token, xyb_b, base);

        // Unmix to gamma-domain LMS + add cbrt(bias)
        let gamma_r = y + x + neg_cbrt0;
        let gamma_g = y - x + neg_cbrt1;
        let gamma_b = b + neg_cbrt2;

        // Cube and subtract bias (gamma^3 + neg_bias)
        let mixed_r = gamma_r * gamma_r * gamma_r + neg_bias0;
        let mixed_g = gamma_g * gamma_g * gamma_g + neg_bias1;
        let mixed_b = gamma_b * gamma_b * gamma_b + neg_bias2;

        // Inverse opsin matrix (FMA chains)
        let rv = inv00.mul_add(mixed_r, inv01.mul_add(mixed_g, inv02 * mixed_b));
        let gv = inv10.mul_add(mixed_r, inv11.mul_add(mixed_g, inv12 * mixed_b));
        let bv = inv20.mul_add(mixed_r, inv21.mul_add(mixed_g, inv22 * mixed_b));

        // Store interleaved: [R0,G0,B0, R1,G1,B1, ...]
        // No efficient SIMD scatter for AoS, so extract and interleave
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        rv.store(r_arr.as_mut_slice().try_into().unwrap());
        gv.store(g_arr.as_mut_slice().try_into().unwrap());
        bv.store(b_arr.as_mut_slice().try_into().unwrap());
        let out = &mut linear_rgb[base * 3..];
        for i in 0..8 {
            out[i * 3] = r_arr[i];
            out[i * 3 + 1] = g_arr[i];
            out[i * 3 + 2] = b_arr[i];
        }
    }

    // Scalar remainder
    let start = chunks * 8;
    inverse_xyb_scalar(
        &xyb_x[start..],
        &xyb_y[start..],
        &xyb_b[start..],
        &mut linear_rgb[start * 3..],
        n - start,
    );
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn inverse_xyb_planar_avx2(
    token: archmage::X64V3Token,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    out_r: &mut [f32],
    out_g: &mut [f32],
    out_b: &mut [f32],
    n: usize,
) {
    use magetypes::simd::f32x8;

    let neg_cbrt0 = f32x8::splat(token, -NEG_CBRT_BIAS[0]);
    let neg_cbrt1 = f32x8::splat(token, -NEG_CBRT_BIAS[1]);
    let neg_cbrt2 = f32x8::splat(token, -NEG_CBRT_BIAS[2]);
    let neg_bias0 = f32x8::splat(token, -OPSIN_BIAS[0]);
    let neg_bias1 = f32x8::splat(token, -OPSIN_BIAS[1]);
    let neg_bias2 = f32x8::splat(token, -OPSIN_BIAS[2]);
    let inv00 = f32x8::splat(token, INV_OPSIN[0][0]);
    let inv01 = f32x8::splat(token, INV_OPSIN[0][1]);
    let inv02 = f32x8::splat(token, INV_OPSIN[0][2]);
    let inv10 = f32x8::splat(token, INV_OPSIN[1][0]);
    let inv11 = f32x8::splat(token, INV_OPSIN[1][1]);
    let inv12 = f32x8::splat(token, INV_OPSIN[1][2]);
    let inv20 = f32x8::splat(token, INV_OPSIN[2][0]);
    let inv21 = f32x8::splat(token, INV_OPSIN[2][1]);
    let inv22 = f32x8::splat(token, INV_OPSIN[2][2]);

    let chunks = n / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let x = crate::load_f32x8(token, xyb_x, base);
        let y = crate::load_f32x8(token, xyb_y, base);
        let b = crate::load_f32x8(token, xyb_b, base);

        // Unmix to gamma-domain LMS + add cbrt(bias)
        let gamma_r = y + x + neg_cbrt0;
        let gamma_g = y - x + neg_cbrt1;
        let gamma_b = b + neg_cbrt2;

        // Cube and subtract bias
        let mixed_r = gamma_r * gamma_r * gamma_r + neg_bias0;
        let mixed_g = gamma_g * gamma_g * gamma_g + neg_bias1;
        let mixed_b = gamma_b * gamma_b * gamma_b + neg_bias2;

        // Inverse opsin matrix (FMA chains)
        let rv = inv00.mul_add(mixed_r, inv01.mul_add(mixed_g, inv02 * mixed_b));
        let gv = inv10.mul_add(mixed_r, inv11.mul_add(mixed_g, inv12 * mixed_b));
        let bv = inv20.mul_add(mixed_r, inv21.mul_add(mixed_g, inv22 * mixed_b));

        // Store planar — direct SIMD store, no scalar interleave needed
        crate::store_f32x8(out_r, base, rv);
        crate::store_f32x8(out_g, base, gv);
        crate::store_f32x8(out_b, base, bv);
    }

    // Scalar remainder
    let start = chunks * 8;
    inverse_xyb_planar_scalar(
        &xyb_x[start..],
        &xyb_y[start..],
        &xyb_b[start..],
        &mut out_r[start..],
        &mut out_g[start..],
        &mut out_b[start..],
        n - start,
    );
}

// --- aarch64 NEON implementations ---

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn forward_xyb_neon(
    token: archmage::NeonToken,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    n: usize,
) {
    use magetypes::simd::f32x4;

    let m00 = f32x4::splat(token, OPSIN_MATRIX[0][0]);
    let m01 = f32x4::splat(token, OPSIN_MATRIX[0][1]);
    let m02 = f32x4::splat(token, OPSIN_MATRIX[0][2]);
    let m10 = f32x4::splat(token, OPSIN_MATRIX[1][0]);
    let m11 = f32x4::splat(token, OPSIN_MATRIX[1][1]);
    let m12 = f32x4::splat(token, OPSIN_MATRIX[1][2]);
    let m20 = f32x4::splat(token, OPSIN_MATRIX[2][0]);
    let m21 = f32x4::splat(token, OPSIN_MATRIX[2][1]);
    let m22 = f32x4::splat(token, OPSIN_MATRIX[2][2]);
    let bias0 = f32x4::splat(token, OPSIN_BIAS[0]);
    let bias1 = f32x4::splat(token, OPSIN_BIAS[1]);
    let bias2 = f32x4::splat(token, OPSIN_BIAS[2]);
    let neg_cbrt0 = f32x4::splat(token, NEG_CBRT_BIAS[0]);
    let neg_cbrt1 = f32x4::splat(token, NEG_CBRT_BIAS[1]);
    let neg_cbrt2 = f32x4::splat(token, NEG_CBRT_BIAS[2]);
    let half = f32x4::splat(token, 0.5);
    let zero = f32x4::zero(token);

    let chunks = n / 4;
    let simd_n = chunks * 4;
    let r_s = &r[..simd_n];
    let g_s = &g[..simd_n];
    let b_s = &b[..simd_n];

    for chunk in 0..chunks {
        let base = chunk * 4;
        let rv = f32x4::from_slice(token, &r_s[base..]);
        let gv = f32x4::from_slice(token, &g_s[base..]);
        let bv = f32x4::from_slice(token, &b_s[base..]);

        let mixed0 = m00.mul_add(rv, m01.mul_add(gv, m02.mul_add(bv, bias0)));
        let mixed1 = m10.mul_add(rv, m11.mul_add(gv, m12.mul_add(bv, bias1)));
        let mixed2 = m20.mul_add(rv, m21.mul_add(gv, m22.mul_add(bv, bias2)));

        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        // Scalar cbrt (same approach as AVX2 — precision-critical)
        let mut m0_arr = [0.0f32; 4];
        let mut m1_arr = [0.0f32; 4];
        let mut m2_arr = [0.0f32; 4];
        mixed0.store(m0_arr.as_mut_slice().try_into().unwrap());
        mixed1.store(m1_arr.as_mut_slice().try_into().unwrap());
        mixed2.store(m2_arr.as_mut_slice().try_into().unwrap());
        for j in 0..4 {
            m0_arr[j] = cbrt_fast(m0_arr[j]);
            m1_arr[j] = cbrt_fast(m1_arr[j]);
            m2_arr[j] = cbrt_fast(m2_arr[j]);
        }
        let l = f32x4::from_slice(token, &m0_arr) + neg_cbrt0;
        let m = f32x4::from_slice(token, &m1_arr) + neg_cbrt1;
        let s = f32x4::from_slice(token, &m2_arr) + neg_cbrt2;

        let xv = half * (l - m);
        let yv = half * (l + m);

        xv.store((&mut x_out[base..base + 4]).try_into().unwrap());
        yv.store((&mut y_out[base..base + 4]).try_into().unwrap());
        s.store((&mut b_out[base..base + 4]).try_into().unwrap());
    }

    let start = simd_n;
    forward_xyb_scalar(
        &r[start..],
        &g[start..],
        &b[start..],
        &mut x_out[start..],
        &mut y_out[start..],
        &mut b_out[start..],
        n - start,
    );
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
pub fn inverse_xyb_neon(
    token: archmage::NeonToken,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    linear_rgb: &mut [f32],
    n: usize,
) {
    use magetypes::simd::f32x4;

    let neg_cbrt0 = f32x4::splat(token, -NEG_CBRT_BIAS[0]);
    let neg_cbrt1 = f32x4::splat(token, -NEG_CBRT_BIAS[1]);
    let neg_cbrt2 = f32x4::splat(token, -NEG_CBRT_BIAS[2]);
    let neg_bias0 = f32x4::splat(token, -OPSIN_BIAS[0]);
    let neg_bias1 = f32x4::splat(token, -OPSIN_BIAS[1]);
    let neg_bias2 = f32x4::splat(token, -OPSIN_BIAS[2]);
    let inv00 = f32x4::splat(token, INV_OPSIN[0][0]);
    let inv01 = f32x4::splat(token, INV_OPSIN[0][1]);
    let inv02 = f32x4::splat(token, INV_OPSIN[0][2]);
    let inv10 = f32x4::splat(token, INV_OPSIN[1][0]);
    let inv11 = f32x4::splat(token, INV_OPSIN[1][1]);
    let inv12 = f32x4::splat(token, INV_OPSIN[1][2]);
    let inv20 = f32x4::splat(token, INV_OPSIN[2][0]);
    let inv21 = f32x4::splat(token, INV_OPSIN[2][1]);
    let inv22 = f32x4::splat(token, INV_OPSIN[2][2]);

    let chunks = n / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        let x = f32x4::from_slice(token, &xyb_x[base..]);
        let y = f32x4::from_slice(token, &xyb_y[base..]);
        let b = f32x4::from_slice(token, &xyb_b[base..]);

        let gamma_r = y + x + neg_cbrt0;
        let gamma_g = y - x + neg_cbrt1;
        let gamma_b = b + neg_cbrt2;

        let mixed_r = gamma_r * gamma_r * gamma_r + neg_bias0;
        let mixed_g = gamma_g * gamma_g * gamma_g + neg_bias1;
        let mixed_b = gamma_b * gamma_b * gamma_b + neg_bias2;

        let rv = inv00.mul_add(mixed_r, inv01.mul_add(mixed_g, inv02 * mixed_b));
        let gv = inv10.mul_add(mixed_r, inv11.mul_add(mixed_g, inv12 * mixed_b));
        let bv = inv20.mul_add(mixed_r, inv21.mul_add(mixed_g, inv22 * mixed_b));

        let mut r_arr = [0.0f32; 4];
        let mut g_arr = [0.0f32; 4];
        let mut b_arr = [0.0f32; 4];
        rv.store(r_arr.as_mut_slice().try_into().unwrap());
        gv.store(g_arr.as_mut_slice().try_into().unwrap());
        bv.store(b_arr.as_mut_slice().try_into().unwrap());
        let out = &mut linear_rgb[base * 3..];
        for i in 0..4 {
            out[i * 3] = r_arr[i];
            out[i * 3 + 1] = g_arr[i];
            out[i * 3 + 2] = b_arr[i];
        }
    }

    let start = chunks * 4;
    inverse_xyb_scalar(
        &xyb_x[start..],
        &xyb_y[start..],
        &xyb_b[start..],
        &mut linear_rgb[start * 3..],
        n - start,
    );
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn inverse_xyb_planar_neon(
    token: archmage::NeonToken,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    out_r: &mut [f32],
    out_g: &mut [f32],
    out_b: &mut [f32],
    n: usize,
) {
    use magetypes::simd::f32x4;

    let neg_cbrt0 = f32x4::splat(token, -NEG_CBRT_BIAS[0]);
    let neg_cbrt1 = f32x4::splat(token, -NEG_CBRT_BIAS[1]);
    let neg_cbrt2 = f32x4::splat(token, -NEG_CBRT_BIAS[2]);
    let neg_bias0 = f32x4::splat(token, -OPSIN_BIAS[0]);
    let neg_bias1 = f32x4::splat(token, -OPSIN_BIAS[1]);
    let neg_bias2 = f32x4::splat(token, -OPSIN_BIAS[2]);
    let inv00 = f32x4::splat(token, INV_OPSIN[0][0]);
    let inv01 = f32x4::splat(token, INV_OPSIN[0][1]);
    let inv02 = f32x4::splat(token, INV_OPSIN[0][2]);
    let inv10 = f32x4::splat(token, INV_OPSIN[1][0]);
    let inv11 = f32x4::splat(token, INV_OPSIN[1][1]);
    let inv12 = f32x4::splat(token, INV_OPSIN[1][2]);
    let inv20 = f32x4::splat(token, INV_OPSIN[2][0]);
    let inv21 = f32x4::splat(token, INV_OPSIN[2][1]);
    let inv22 = f32x4::splat(token, INV_OPSIN[2][2]);

    let chunks = n / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        let x = f32x4::from_slice(token, &xyb_x[base..]);
        let y = f32x4::from_slice(token, &xyb_y[base..]);
        let b = f32x4::from_slice(token, &xyb_b[base..]);

        let gamma_r = y + x + neg_cbrt0;
        let gamma_g = y - x + neg_cbrt1;
        let gamma_b = b + neg_cbrt2;

        let mixed_r = gamma_r * gamma_r * gamma_r + neg_bias0;
        let mixed_g = gamma_g * gamma_g * gamma_g + neg_bias1;
        let mixed_b = gamma_b * gamma_b * gamma_b + neg_bias2;

        let rv = inv00.mul_add(mixed_r, inv01.mul_add(mixed_g, inv02 * mixed_b));
        let gv = inv10.mul_add(mixed_r, inv11.mul_add(mixed_g, inv12 * mixed_b));
        let bv = inv20.mul_add(mixed_r, inv21.mul_add(mixed_g, inv22 * mixed_b));

        rv.store((&mut out_r[base..base + 4]).try_into().unwrap());
        gv.store((&mut out_g[base..base + 4]).try_into().unwrap());
        bv.store((&mut out_b[base..base + 4]).try_into().unwrap());
    }

    let start = chunks * 4;
    inverse_xyb_planar_scalar(
        &xyb_x[start..],
        &xyb_y[start..],
        &xyb_b[start..],
        &mut out_r[start..],
        &mut out_g[start..],
        &mut out_b[start..],
        n - start,
    );
}

// --- wasm32 SIMD128 implementations ---

#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn forward_xyb_wasm128(
    token: archmage::Wasm128Token,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    n: usize,
) {
    use magetypes::simd::f32x4;

    let m00 = f32x4::splat(token, OPSIN_MATRIX[0][0]);
    let m01 = f32x4::splat(token, OPSIN_MATRIX[0][1]);
    let m02 = f32x4::splat(token, OPSIN_MATRIX[0][2]);
    let m10 = f32x4::splat(token, OPSIN_MATRIX[1][0]);
    let m11 = f32x4::splat(token, OPSIN_MATRIX[1][1]);
    let m12 = f32x4::splat(token, OPSIN_MATRIX[1][2]);
    let m20 = f32x4::splat(token, OPSIN_MATRIX[2][0]);
    let m21 = f32x4::splat(token, OPSIN_MATRIX[2][1]);
    let m22 = f32x4::splat(token, OPSIN_MATRIX[2][2]);
    let bias0 = f32x4::splat(token, OPSIN_BIAS[0]);
    let bias1 = f32x4::splat(token, OPSIN_BIAS[1]);
    let bias2 = f32x4::splat(token, OPSIN_BIAS[2]);
    let neg_cbrt0 = f32x4::splat(token, NEG_CBRT_BIAS[0]);
    let neg_cbrt1 = f32x4::splat(token, NEG_CBRT_BIAS[1]);
    let neg_cbrt2 = f32x4::splat(token, NEG_CBRT_BIAS[2]);
    let half = f32x4::splat(token, 0.5);
    let zero = f32x4::zero(token);

    let chunks = n / 4;
    let simd_n = chunks * 4;
    let r_s = &r[..simd_n];
    let g_s = &g[..simd_n];
    let b_s = &b[..simd_n];

    for chunk in 0..chunks {
        let base = chunk * 4;
        let rv = f32x4::from_slice(token, &r_s[base..]);
        let gv = f32x4::from_slice(token, &g_s[base..]);
        let bv = f32x4::from_slice(token, &b_s[base..]);

        let mixed0 = m00.mul_add(rv, m01.mul_add(gv, m02.mul_add(bv, bias0)));
        let mixed1 = m10.mul_add(rv, m11.mul_add(gv, m12.mul_add(bv, bias1)));
        let mixed2 = m20.mul_add(rv, m21.mul_add(gv, m22.mul_add(bv, bias2)));

        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        // Scalar cbrt (same approach as AVX2 — precision-critical)
        let mut m0_arr = [0.0f32; 4];
        let mut m1_arr = [0.0f32; 4];
        let mut m2_arr = [0.0f32; 4];
        mixed0.store(m0_arr.as_mut_slice().try_into().unwrap());
        mixed1.store(m1_arr.as_mut_slice().try_into().unwrap());
        mixed2.store(m2_arr.as_mut_slice().try_into().unwrap());
        for j in 0..4 {
            m0_arr[j] = cbrt_fast(m0_arr[j]);
            m1_arr[j] = cbrt_fast(m1_arr[j]);
            m2_arr[j] = cbrt_fast(m2_arr[j]);
        }
        let l = f32x4::from_slice(token, &m0_arr) + neg_cbrt0;
        let m = f32x4::from_slice(token, &m1_arr) + neg_cbrt1;
        let s = f32x4::from_slice(token, &m2_arr) + neg_cbrt2;

        let xv = half * (l - m);
        let yv = half * (l + m);

        xv.store((&mut x_out[base..base + 4]).try_into().unwrap());
        yv.store((&mut y_out[base..base + 4]).try_into().unwrap());
        s.store((&mut b_out[base..base + 4]).try_into().unwrap());
    }

    let start = simd_n;
    forward_xyb_scalar(
        &r[start..],
        &g[start..],
        &b[start..],
        &mut x_out[start..],
        &mut y_out[start..],
        &mut b_out[start..],
        n - start,
    );
}

#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
pub fn inverse_xyb_wasm128(
    token: archmage::Wasm128Token,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    linear_rgb: &mut [f32],
    n: usize,
) {
    use magetypes::simd::f32x4;

    let neg_cbrt0 = f32x4::splat(token, -NEG_CBRT_BIAS[0]);
    let neg_cbrt1 = f32x4::splat(token, -NEG_CBRT_BIAS[1]);
    let neg_cbrt2 = f32x4::splat(token, -NEG_CBRT_BIAS[2]);
    let neg_bias0 = f32x4::splat(token, -OPSIN_BIAS[0]);
    let neg_bias1 = f32x4::splat(token, -OPSIN_BIAS[1]);
    let neg_bias2 = f32x4::splat(token, -OPSIN_BIAS[2]);
    let inv00 = f32x4::splat(token, INV_OPSIN[0][0]);
    let inv01 = f32x4::splat(token, INV_OPSIN[0][1]);
    let inv02 = f32x4::splat(token, INV_OPSIN[0][2]);
    let inv10 = f32x4::splat(token, INV_OPSIN[1][0]);
    let inv11 = f32x4::splat(token, INV_OPSIN[1][1]);
    let inv12 = f32x4::splat(token, INV_OPSIN[1][2]);
    let inv20 = f32x4::splat(token, INV_OPSIN[2][0]);
    let inv21 = f32x4::splat(token, INV_OPSIN[2][1]);
    let inv22 = f32x4::splat(token, INV_OPSIN[2][2]);

    let chunks = n / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        let x = f32x4::from_slice(token, &xyb_x[base..]);
        let y = f32x4::from_slice(token, &xyb_y[base..]);
        let b = f32x4::from_slice(token, &xyb_b[base..]);

        let gamma_r = y + x + neg_cbrt0;
        let gamma_g = y - x + neg_cbrt1;
        let gamma_b = b + neg_cbrt2;

        let mixed_r = gamma_r * gamma_r * gamma_r + neg_bias0;
        let mixed_g = gamma_g * gamma_g * gamma_g + neg_bias1;
        let mixed_b = gamma_b * gamma_b * gamma_b + neg_bias2;

        let rv = inv00.mul_add(mixed_r, inv01.mul_add(mixed_g, inv02 * mixed_b));
        let gv = inv10.mul_add(mixed_r, inv11.mul_add(mixed_g, inv12 * mixed_b));
        let bv = inv20.mul_add(mixed_r, inv21.mul_add(mixed_g, inv22 * mixed_b));

        let mut r_arr = [0.0f32; 4];
        let mut g_arr = [0.0f32; 4];
        let mut b_arr = [0.0f32; 4];
        rv.store(r_arr.as_mut_slice().try_into().unwrap());
        gv.store(g_arr.as_mut_slice().try_into().unwrap());
        bv.store(b_arr.as_mut_slice().try_into().unwrap());
        let out = &mut linear_rgb[base * 3..];
        for i in 0..4 {
            out[i * 3] = r_arr[i];
            out[i * 3 + 1] = g_arr[i];
            out[i * 3 + 2] = b_arr[i];
        }
    }

    let start = chunks * 4;
    inverse_xyb_scalar(
        &xyb_x[start..],
        &xyb_y[start..],
        &xyb_b[start..],
        &mut linear_rgb[start * 3..],
        n - start,
    );
}

#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn inverse_xyb_planar_wasm128(
    token: archmage::Wasm128Token,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    out_r: &mut [f32],
    out_g: &mut [f32],
    out_b: &mut [f32],
    n: usize,
) {
    use magetypes::simd::f32x4;

    let neg_cbrt0 = f32x4::splat(token, -NEG_CBRT_BIAS[0]);
    let neg_cbrt1 = f32x4::splat(token, -NEG_CBRT_BIAS[1]);
    let neg_cbrt2 = f32x4::splat(token, -NEG_CBRT_BIAS[2]);
    let neg_bias0 = f32x4::splat(token, -OPSIN_BIAS[0]);
    let neg_bias1 = f32x4::splat(token, -OPSIN_BIAS[1]);
    let neg_bias2 = f32x4::splat(token, -OPSIN_BIAS[2]);
    let inv00 = f32x4::splat(token, INV_OPSIN[0][0]);
    let inv01 = f32x4::splat(token, INV_OPSIN[0][1]);
    let inv02 = f32x4::splat(token, INV_OPSIN[0][2]);
    let inv10 = f32x4::splat(token, INV_OPSIN[1][0]);
    let inv11 = f32x4::splat(token, INV_OPSIN[1][1]);
    let inv12 = f32x4::splat(token, INV_OPSIN[1][2]);
    let inv20 = f32x4::splat(token, INV_OPSIN[2][0]);
    let inv21 = f32x4::splat(token, INV_OPSIN[2][1]);
    let inv22 = f32x4::splat(token, INV_OPSIN[2][2]);

    let chunks = n / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        let x = f32x4::from_slice(token, &xyb_x[base..]);
        let y = f32x4::from_slice(token, &xyb_y[base..]);
        let b = f32x4::from_slice(token, &xyb_b[base..]);

        let gamma_r = y + x + neg_cbrt0;
        let gamma_g = y - x + neg_cbrt1;
        let gamma_b = b + neg_cbrt2;

        let mixed_r = gamma_r * gamma_r * gamma_r + neg_bias0;
        let mixed_g = gamma_g * gamma_g * gamma_g + neg_bias1;
        let mixed_b = gamma_b * gamma_b * gamma_b + neg_bias2;

        let rv = inv00.mul_add(mixed_r, inv01.mul_add(mixed_g, inv02 * mixed_b));
        let gv = inv10.mul_add(mixed_r, inv11.mul_add(mixed_g, inv12 * mixed_b));
        let bv = inv20.mul_add(mixed_r, inv21.mul_add(mixed_g, inv22 * mixed_b));

        rv.store((&mut out_r[base..base + 4]).try_into().unwrap());
        gv.store((&mut out_g[base..base + 4]).try_into().unwrap());
        bv.store((&mut out_b[base..base + 4]).try_into().unwrap());
    }

    let start = chunks * 4;
    inverse_xyb_planar_scalar(
        &xyb_x[start..],
        &xyb_y[start..],
        &xyb_b[start..],
        &mut out_r[start..],
        &mut out_g[start..],
        &mut out_b[start..],
        n - start,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    extern crate std;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Sweep test: compare SIMD forward XYB against reference std cbrt.
    #[test]
    fn test_forward_xyb_sweep() {
        let n = 256;
        let mut r = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];

        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            r[i] = t;
            g[i] = 1.0 - t;
            b[i] = (t * 2.0).min(1.0);
        }

        // Reference: use std cbrt
        let mut x_ref = vec![0.0f32; n];
        let mut y_ref = vec![0.0f32; n];
        let mut b_ref = vec![0.0f32; n];
        for i in 0..n {
            let mixed0 = OPSIN_MATRIX[0][0] * r[i]
                + OPSIN_MATRIX[0][1] * g[i]
                + OPSIN_MATRIX[0][2] * b[i]
                + OPSIN_BIAS[0];
            let mixed1 = OPSIN_MATRIX[1][0] * r[i]
                + OPSIN_MATRIX[1][1] * g[i]
                + OPSIN_MATRIX[1][2] * b[i]
                + OPSIN_BIAS[1];
            let mixed2 = OPSIN_MATRIX[2][0] * r[i]
                + OPSIN_MATRIX[2][1] * g[i]
                + OPSIN_MATRIX[2][2] * b[i]
                + OPSIN_BIAS[2];
            let l = mixed0.max(0.0).cbrt() + NEG_CBRT_BIAS[0];
            let m = mixed1.max(0.0).cbrt() + NEG_CBRT_BIAS[1];
            let s = mixed2.max(0.0).cbrt() + NEG_CBRT_BIAS[2];
            x_ref[i] = 0.5 * (l - m);
            y_ref[i] = 0.5 * (l + m);
            b_ref[i] = s;
        }

        // Dispatch — test all token permutations
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut x_out = vec![0.0f32; n];
                let mut y_out = vec![0.0f32; n];
                let mut b_out = vec![0.0f32; n];
                linear_rgb_to_xyb_batch(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);

                for i in 0..n {
                    let ex = (x_out[i] - x_ref[i]).abs();
                    let ey = (y_out[i] - y_ref[i]).abs();
                    let eb = (b_out[i] - b_ref[i]).abs();
                    assert!(
                        ex < 1e-5 && ey < 1e-5 && eb < 1e-5,
                        "Pixel {}: SIMD=({},{},{}), ref=({},{},{}), err=({},{},{}) [{perm}]",
                        i,
                        x_out[i],
                        y_out[i],
                        b_out[i],
                        x_ref[i],
                        y_ref[i],
                        b_ref[i],
                        ex,
                        ey,
                        eb
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }

    /// Sweep test: compare SIMD inverse XYB against reference scalar.
    #[test]
    fn test_inverse_xyb_sweep() {
        let n = 256;
        let mut xyb_x = vec![0.0f32; n];
        let mut xyb_y = vec![0.0f32; n];
        let mut xyb_b = vec![0.0f32; n];

        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            xyb_x[i] = (t - 0.5) * 0.8;
            xyb_y[i] = t * 1.1;
            xyb_b[i] = t * 0.9 - 0.1;
        }

        // Reference scalar
        let mut ref_rgb = vec![0.0f32; n * 3];
        inverse_xyb_scalar(&xyb_x, &xyb_y, &xyb_b, &mut ref_rgb, n);

        // Dispatch — test all token permutations
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_rgb = vec![0.0f32; n * 3];
                xyb_to_linear_rgb_batch(&xyb_x, &xyb_y, &xyb_b, &mut simd_rgb, n);

                for i in 0..n * 3 {
                    let err = (simd_rgb[i] - ref_rgb[i]).abs();
                    assert!(
                        err < 1e-5,
                        "Component {}: SIMD={}, ref={}, err={:.2e} [{perm}]",
                        i,
                        simd_rgb[i],
                        ref_rgb[i],
                        err
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }

    /// Roundtrip test: RGB → XYB → RGB should be approximately identity.
    #[test]
    fn test_xyb_roundtrip_sweep() {
        let n = 256;
        let mut r = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];

        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            r[i] = t * 0.8 + 0.01;
            g[i] = (1.0 - t) * 0.9 + 0.01;
            b[i] = (t * 1.5).min(0.95) + 0.01;
        }

        // Forward: RGB → XYB
        let mut x = vec![0.0f32; n];
        let mut y = vec![0.0f32; n];
        let mut bv = vec![0.0f32; n];
        linear_rgb_to_xyb_batch(&r, &g, &b, &mut x, &mut y, &mut bv);

        // Inverse: XYB → RGB
        let mut rgb_out = vec![0.0f32; n * 3];
        xyb_to_linear_rgb_batch(&x, &y, &bv, &mut rgb_out, n);

        let mut max_err = 0.0f32;
        for i in 0..n {
            let er = (rgb_out[i * 3] - r[i]).abs();
            let eg = (rgb_out[i * 3 + 1] - g[i]).abs();
            let eb = (rgb_out[i * 3 + 2] - b[i]).abs();
            max_err = max_err.max(er).max(eg).max(eb);
        }
        assert!(
            max_err < 1e-4,
            "Roundtrip max error {:.2e} exceeds 1e-4",
            max_err
        );
    }

    /// Edge cases: black, white, primary colors, near-zero values.
    #[test]
    fn test_forward_xyb_edge_cases() {
        let test_cases: &[(f32, f32, f32)] = &[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.001, 0.001, 0.001),
            (0.999, 0.999, 0.999),
            (0.5, 0.5, 0.5),
        ];

        let n = test_cases.len();
        let r: Vec<f32> = test_cases.iter().map(|c| c.0).collect();
        let g: Vec<f32> = test_cases.iter().map(|c| c.1).collect();
        let b: Vec<f32> = test_cases.iter().map(|c| c.2).collect();

        // Reference using std cbrt
        let mut x_ref = vec![0.0f32; n];
        let mut y_ref = vec![0.0f32; n];
        let mut b_ref = vec![0.0f32; n];
        for i in 0..n {
            let mixed0 = OPSIN_MATRIX[0][0] * r[i]
                + OPSIN_MATRIX[0][1] * g[i]
                + OPSIN_MATRIX[0][2] * b[i]
                + OPSIN_BIAS[0];
            let mixed1 = OPSIN_MATRIX[1][0] * r[i]
                + OPSIN_MATRIX[1][1] * g[i]
                + OPSIN_MATRIX[1][2] * b[i]
                + OPSIN_BIAS[1];
            let mixed2 = OPSIN_MATRIX[2][0] * r[i]
                + OPSIN_MATRIX[2][1] * g[i]
                + OPSIN_MATRIX[2][2] * b[i]
                + OPSIN_BIAS[2];
            let l = mixed0.max(0.0).cbrt() + NEG_CBRT_BIAS[0];
            let m = mixed1.max(0.0).cbrt() + NEG_CBRT_BIAS[1];
            let s = mixed2.max(0.0).cbrt() + NEG_CBRT_BIAS[2];
            x_ref[i] = 0.5 * (l - m);
            y_ref[i] = 0.5 * (l + m);
            b_ref[i] = s;
        }

        // Dispatch — test all token permutations
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut x_out = vec![0.0f32; n];
                let mut y_out = vec![0.0f32; n];
                let mut b_out = vec![0.0f32; n];
                linear_rgb_to_xyb_batch(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);

                for i in 0..n {
                    let ex = (x_out[i] - x_ref[i]).abs();
                    let ey = (y_out[i] - y_ref[i]).abs();
                    let eb = (b_out[i] - b_ref[i]).abs();
                    assert!(
                        ex < 1e-5 && ey < 1e-5 && eb < 1e-5,
                        "Edge case {:?}: SIMD=({},{},{}), ref=({},{},{}), err=({:.2e},{:.2e},{:.2e}) [{perm}]",
                        test_cases[i],
                        x_out[i],
                        y_out[i],
                        b_out[i],
                        x_ref[i],
                        y_ref[i],
                        b_ref[i],
                        ex,
                        ey,
                        eb
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }

    /// Test that planar inverse matches interleaved inverse.
    #[test]
    fn test_inverse_xyb_planar_matches_interleaved() {
        let n = 256;
        let mut r = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];

        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            r[i] = t;
            g[i] = 1.0 - t;
            b[i] = (t * 2.0).min(1.0);
        }

        // Use scalar forward to get deterministic XYB input
        let mut x = vec![0.0f32; n];
        let mut y = vec![0.0f32; n];
        let mut bv = vec![0.0f32; n];
        forward_xyb_scalar(&r, &g, &b, &mut x, &mut y, &mut bv, n);

        // Scalar reference for interleaved inverse
        let mut ref_rgb = vec![0.0f32; n * 3];
        inverse_xyb_scalar(&x, &y, &bv, &mut ref_rgb, n);

        // Scalar reference for planar inverse
        let mut ref_r = vec![0.0f32; n];
        let mut ref_g = vec![0.0f32; n];
        let mut ref_b = vec![0.0f32; n];
        inverse_xyb_planar_scalar(&x, &y, &bv, &mut ref_r, &mut ref_g, &mut ref_b, n);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                // Interleaved inverse
                let mut interleaved = vec![0.0f32; n * 3];
                xyb_to_linear_rgb_batch(&x, &y, &bv, &mut interleaved, n);

                // Planar inverse
                let mut pr = vec![0.0f32; n];
                let mut pg = vec![0.0f32; n];
                let mut pb = vec![0.0f32; n];
                xyb_to_linear_rgb_planar(&x, &y, &bv, &mut pr, &mut pg, &mut pb, n);

                for i in 0..n {
                    let ir = interleaved[i * 3];
                    let ig = interleaved[i * 3 + 1];
                    let ib = interleaved[i * 3 + 2];
                    // Both dispatch paths must match scalar
                    assert!(
                        (ir - ref_rgb[i * 3]).abs() < 1e-5,
                        "Interleaved R mismatch at {i}: got {ir}, ref {} [{perm}]",
                        ref_rgb[i * 3]
                    );
                    assert!(
                        (pr[i] - ref_r[i]).abs() < 1e-5,
                        "Planar R mismatch at {i}: got {}, ref {} [{perm}]",
                        pr[i],
                        ref_r[i]
                    );
                    // And they must match each other
                    assert!(
                        (pr[i] - ir).abs() < 1e-6
                            && (pg[i] - ig).abs() < 1e-6
                            && (pb[i] - ib).abs() < 1e-6,
                        "Planar/interleaved mismatch at {i}: planar=({},{},{}) interleaved=({ir},{ig},{ib}) [{perm}]",
                        pr[i],
                        pg[i],
                        pb[i]
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }
}
