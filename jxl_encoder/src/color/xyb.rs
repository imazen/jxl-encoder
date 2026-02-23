// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! XYB color space transform for JPEG XL encoding.
//!
//! XYB is a perceptually-motivated color space used by JPEG XL for lossy encoding.
//! It provides better rate-distortion characteristics than traditional color spaces.
//!
//! The forward transform is:
//! 1. sRGB → Linear RGB (gamma expansion)
//! 2. Linear RGB → Opsin LMS (3x3 matrix)
//! 3. Add bias and apply cube root
//! 4. Mix into XYB: X = 0.5*(L-M), Y = 0.5*(L+M), B = S

// Opsin absorbance matrix from libjxl cms/opsin_params.h
// These are "frozen" and must not change.
// The constants below are exact spec values; allow excessive precision for parity.
#[allow(clippy::excessive_precision)]
const K_M00: f32 = 0.30;
#[allow(clippy::excessive_precision)]
const K_M01: f32 = 1.0 - 0.078 - 0.30; // 0.622
#[allow(clippy::excessive_precision)]
const K_M02: f32 = 0.078;

#[allow(clippy::excessive_precision)]
const K_M10: f32 = 0.23;
#[allow(clippy::excessive_precision)]
const K_M11: f32 = 1.0 - 0.078 - 0.23; // 0.692
#[allow(clippy::excessive_precision)]
const K_M12: f32 = 0.078;

#[allow(clippy::excessive_precision)]
const K_M20: f32 = 0.24342268924547819;
#[allow(clippy::excessive_precision)]
const K_M21: f32 = 0.20476744424496821;
#[allow(clippy::excessive_precision)]
const K_M22: f32 = 1.0 - K_M20 - K_M21; // 0.55180986651

/// Opsin absorbance matrix (linear RGB to LMS-like)
pub const OPSIN_ABSORBANCE_MATRIX: [[f32; 3]; 3] = [
    [K_M00, K_M01, K_M02],
    [K_M10, K_M11, K_M12],
    [K_M20, K_M21, K_M22],
];

/// Opsin absorbance bias - added before cube root
#[allow(clippy::excessive_precision)]
pub const OPSIN_ABSORBANCE_BIAS: [f32; 3] = [
    0.0037930732552754493,
    0.0037930732552754493,
    0.0037930732552754493,
];

/// Cube root of bias - subtracted AFTER cube root (libjxl's CubeRootAndAdd pattern)
/// This is the negative bias that gets added after taking the cube root.
#[allow(clippy::excessive_precision)]
pub const NEG_OPSIN_ABSORBANCE_BIAS_CBRT: [f32; 3] = [
    -0.15595420054, // -cbrt(OPSIN_ABSORBANCE_BIAS[0])
    -0.15595420054,
    -0.15595420054,
];

/// Convert a single sRGB value (0-255 range) to linear (0-1 range).
///
/// Uses the standard sRGB transfer function:
/// - Linear region: x / 12.92 for x <= 0.04045
/// - Gamma region: ((x + 0.055) / 1.055)^2.4 for x > 0.04045
#[inline]
pub fn srgb_to_linear_value(srgb: f32) -> f32 {
    let normalized = srgb / 255.0;
    if normalized <= 0.04045 {
        normalized / 12.92
    } else {
        jxl_simd::fast_powf((normalized + 0.055) / 1.055, 2.4)
    }
}

/// Convert sRGB u8 values to linear f32 in-place.
///
/// Input: sRGB values in 0-255 range
/// Output: Linear values in 0-1 range
pub fn srgb_to_linear(pixels: &mut [f32]) {
    for p in pixels.iter_mut() {
        *p = srgb_to_linear_value(*p);
    }
}

/// Convert linear RGB to XYB color space.
///
/// Input: Linear RGB values (0-1 range for SDR), intensity_target (nits, default 255)
/// Output: XYB values
///
/// The XYB transform:
/// 1. Scale by intensity_target/255.0 (matches libjxl ComputePremulAbsorb)
/// 2. Apply opsin absorbance matrix to get mixed0, mixed1, mixed2
/// 3. ZeroIfNegative clamp, add bias, apply cube root
/// 4. Combine: X = 0.5*(L-M), Y = 0.5*(L+M), B = S
///
/// For SDR (intensity_target=255), the scaling factor is 1.0 (no-op).
pub fn linear_rgb_to_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    linear_rgb_to_xyb_scaled(r, g, b, 255.0)
}

/// Convert linear RGB to XYB with explicit intensity_target scaling.
///
/// Matches libjxl's `ComputePremulAbsorb` (enc_xyb.cc:214-228): each linear RGB
/// component is multiplied by `intensity_target / 255.0` before the matrix multiply.
pub fn linear_rgb_to_xyb_scaled(r: f32, g: f32, b: f32, intensity_target: f32) -> (f32, f32, f32) {
    let scale = intensity_target / 255.0;
    let r = r * scale;
    let g = g * scale;
    let b = b * scale;

    // Apply opsin absorbance matrix
    // ZeroIfNegative clamp before bias (matches libjxl enc_xyb.cc:91-95).
    // For in-gamut sRGB this is a no-op (bias ensures positivity), but
    // wide-gamut inputs (P3, Rec2020) can produce negative mixed values.
    let mixed0 = (OPSIN_ABSORBANCE_MATRIX[0][0] * r
        + OPSIN_ABSORBANCE_MATRIX[0][1] * g
        + OPSIN_ABSORBANCE_MATRIX[0][2] * b)
        .max(0.0);
    let mixed1 = (OPSIN_ABSORBANCE_MATRIX[1][0] * r
        + OPSIN_ABSORBANCE_MATRIX[1][1] * g
        + OPSIN_ABSORBANCE_MATRIX[1][2] * b)
        .max(0.0);
    let mixed2 = (OPSIN_ABSORBANCE_MATRIX[2][0] * r
        + OPSIN_ABSORBANCE_MATRIX[2][1] * g
        + OPSIN_ABSORBANCE_MATRIX[2][2] * b)
        .max(0.0);

    // Add bias, apply cube root, then subtract cbrt(bias)
    // This matches libjxl's CubeRootAndAdd pattern: cbrt(x + bias) - cbrt(bias)
    let l = (mixed0 + OPSIN_ABSORBANCE_BIAS[0]).cbrt() + NEG_OPSIN_ABSORBANCE_BIAS_CBRT[0];
    let m = (mixed1 + OPSIN_ABSORBANCE_BIAS[1]).cbrt() + NEG_OPSIN_ABSORBANCE_BIAS_CBRT[1];
    let s = (mixed2 + OPSIN_ABSORBANCE_BIAS[2]).cbrt() + NEG_OPSIN_ABSORBANCE_BIAS_CBRT[2];

    // Mix into XYB
    let x = 0.5 * (l - m);
    let y = 0.5 * (l + m);
    let b_out = s;

    (x, y, b_out)
}

/// Convert sRGB (u8 values as f32) directly to XYB.
///
/// This is a convenience function combining sRGB→linear and linear→XYB.
pub fn srgb_to_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let r_linear = srgb_to_linear_value(r);
    let g_linear = srgb_to_linear_value(g);
    let b_linear = srgb_to_linear_value(b);
    linear_rgb_to_xyb(r_linear, g_linear, b_linear)
}

/// Convert an entire image from sRGB to XYB.
///
/// Input: Separate R, G, B channel buffers with u8 values stored as f32
/// Output: X, Y, B channel buffers
pub fn srgb_image_to_xyb(
    r_in: &[f32],
    g_in: &[f32],
    b_in: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    assert_eq!(r_in.len(), g_in.len());
    assert_eq!(r_in.len(), b_in.len());
    assert_eq!(r_in.len(), x_out.len());
    assert_eq!(r_in.len(), y_out.len());
    assert_eq!(r_in.len(), b_out.len());

    for i in 0..r_in.len() {
        let (x, y, b) = srgb_to_xyb(r_in[i], g_in[i], b_in[i]);
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = b;
    }
}

/// Convert linear RGB image to XYB.
///
/// Input: Linear RGB values (0-1 range) in separate channel buffers
/// Output: XYB values in separate channel buffers
pub fn linear_image_to_xyb(
    r_in: &[f32],
    g_in: &[f32],
    b_in: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    assert_eq!(r_in.len(), g_in.len());
    assert_eq!(r_in.len(), b_in.len());
    assert_eq!(r_in.len(), x_out.len());
    assert_eq!(r_in.len(), y_out.len());
    assert_eq!(r_in.len(), b_out.len());

    for i in 0..r_in.len() {
        let (x, y, b) = linear_rgb_to_xyb(r_in[i], g_in[i], b_in[i]);
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_to_linear() {
        // Black
        assert!((srgb_to_linear_value(0.0) - 0.0).abs() < 1e-6);

        // White (fast_powf has ~3e-5 relative error)
        assert!((srgb_to_linear_value(255.0) - 1.0).abs() < 1e-4);

        // Mid-gray (sRGB 128 ≈ linear 0.2158)
        let mid = srgb_to_linear_value(128.0);
        assert!((mid - 0.2158).abs() < 0.01);

        // Linear region boundary (sRGB ~10.31 ≈ linear 0.04045/12.92)
        let boundary = srgb_to_linear_value(10.31);
        assert!(boundary < 0.004);
    }

    #[test]
    fn test_opsin_matrix_row_sums() {
        // Each row of the opsin matrix should sum to 1.0 (for neutral colors)
        for row in &OPSIN_ABSORBANCE_MATRIX {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Row sum: {}", sum);
        }
    }

    #[test]
    fn test_black_to_xyb() {
        let (x, y, _b) = srgb_to_xyb(0.0, 0.0, 0.0);
        // Black: L = cbrt(0 + bias) - cbrt(bias) = 0
        // This matches libjxl's behavior where black maps to (0, 0, 0) in XYB
        assert!(y.abs() < 1e-6, "Y for black should be ~0: {}", y);
        assert!(x.abs() < 1e-6, "X for black should be ~0: {}", x);
    }

    #[test]
    fn test_white_to_xyb() {
        let (x, y, b) = srgb_to_xyb(255.0, 255.0, 255.0);
        // White should have X ≈ 0 (L ≈ M for neutral colors)
        assert!(x.abs() < 1e-6, "X for white should be ~0: {}", x);
        // Y should be positive and larger than B
        assert!(y > 0.0, "Y for white should be positive");
        assert!(b > 0.0, "B for white should be positive");
    }

    #[test]
    fn test_red_to_xyb() {
        let (x, y, _b) = srgb_to_xyb(255.0, 0.0, 0.0);
        // Red: L > M, so X > 0
        assert!(x > 0.0, "X for red should be positive: {}", x);
        assert!(y > 0.0, "Y for red should be positive: {}", y);
    }

    #[test]
    fn test_green_to_xyb() {
        let (x, y, _b) = srgb_to_xyb(0.0, 255.0, 0.0);
        // Green: M > L (green has larger weight in M), so X < 0
        assert!(x < 0.0, "X for green should be negative: {}", x);
        assert!(y > 0.0, "Y for green should be positive: {}", y);
    }

    #[test]
    fn test_blue_to_xyb() {
        let (_x, _y, b) = srgb_to_xyb(0.0, 0.0, 255.0);
        // Blue affects S channel most (B output)
        assert!(b > 0.0, "B for blue should be positive: {}", b);
    }

    #[test]
    fn test_image_conversion() {
        let r = vec![255.0, 0.0, 0.0];
        let g = vec![0.0, 255.0, 0.0];
        let b = vec![0.0, 0.0, 255.0];

        let mut x_out = vec![0.0; 3];
        let mut y_out = vec![0.0; 3];
        let mut b_out = vec![0.0; 3];

        srgb_image_to_xyb(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);

        // Red pixel
        assert!(x_out[0] > 0.0);
        // Green pixel
        assert!(x_out[1] < 0.0);
        // Blue pixel
        assert!(b_out[2] > b_out[0]);
    }
}
