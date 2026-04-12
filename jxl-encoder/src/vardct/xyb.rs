// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! XYB color space conversion with padding.
//!
//! Converts linear RGB to XYB color space and pads to block boundaries.
//! Top SIMD optimization target — the inner loop is a pure per-pixel transform.
//!
//! When the input uses non-sRGB primaries (Display P3, BT.2020), the linear RGB
//! values are first transformed to linear sRGB via a 3x3 matrix. The XYB opsin
//! absorbance matrix is defined for sRGB/BT.709 primaries — feeding P3 or BT.2020
//! linear RGB directly would produce wrong colors.

use super::encoder::VarDctEncoder;
use crate::headers::color_encoding::Primaries;

/// 3x3 matrix to convert linear Display P3 RGB to linear sRGB RGB.
/// Derived from P3→XYZ→sRGB chromatic adaptation (both D65 white point).
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
pub(crate) const P3_TO_SRGB: [[f32; 3]; 3] = [
    [ 1.2249401763, -0.2249401763,  0.0000000000],
    [-0.0420569547,  1.0420569547,  0.0000000000],
    [-0.0196375546, -0.0786360456,  1.0982736001],
];

/// 3x3 matrix to convert linear BT.2020 RGB to linear sRGB RGB.
/// Derived from BT.2020→XYZ→sRGB chromatic adaptation (both D65 white point).
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
pub(crate) const BT2020_TO_SRGB: [[f32; 3]; 3] = [
    [ 1.6604910021, -0.5876411388, -0.0728498633],
    [-0.1245504745,  1.1328998971, -0.0083494226],
    [-0.0181507634, -0.1005788980,  1.1187296614],
];

/// Compute a 3x3 matrix to convert from custom primaries (D65 white point) to sRGB.
///
/// Uses the standard xy-chromaticity → XYZ → sRGB pipeline.
/// Panics if any primary has y=0 or if the matrix is singular.
pub(crate) fn compute_primaries_to_srgb(
    r: (f64, f64),
    g: (f64, f64),
    b: (f64, f64),
) -> [[f32; 3]; 3] {
    // D65 white point
    let (wx, wy) = (0.3127, 0.3290);

    // xy → XYZ: X=x/y, Y=1, Z=(1-x-y)/y
    let xy_to_xyz = |x: f64, y: f64| -> [f64; 3] { [x / y, 1.0, (1.0 - x - y) / y] };

    let [xr, yr, zr] = xy_to_xyz(r.0, r.1);
    let [xg, yg, zg] = xy_to_xyz(g.0, g.1);
    let [xb, yb, zb] = xy_to_xyz(b.0, b.1);
    let [xw, yw, zw] = xy_to_xyz(wx, wy);

    // Solve M * S = W for S (scaling factors)
    // M = [[Xr,Xg,Xb],[Yr,Yg,Yb],[Zr,Zg,Zb]]
    // Using Cramer's rule for 3x3
    let det = xr * (yg * zb - yb * zg) - xg * (yr * zb - yb * zr) + xb * (yr * zg - yg * zr);
    assert!(det.abs() > 1e-10, "singular primaries matrix");

    let inv_det = 1.0 / det;
    let sr =
        ((yg * zb - yb * zg) * xw + (xb * zg - xg * zb) * yw + (xg * yb - xb * yg) * zw) * inv_det;
    let sg =
        ((yb * zr - yr * zb) * xw + (xr * zb - xb * zr) * yw + (xb * yr - xr * yb) * zw) * inv_det;
    let sb =
        ((yr * zg - yg * zr) * xw + (xg * zr - xr * zg) * yw + (xr * yg - xg * yr) * zw) * inv_det;

    // primaries_to_xyz[i][j] = M[i][j] * S[j]
    let p2x = [
        [xr * sr, xg * sg, xb * sb],
        [yr * sr, yg * sg, yb * sb],
        [zr * sr, zg * sg, zb * sb],
    ];

    // sRGB to XYZ (hardcoded for D65, BT.709 primaries)
    #[allow(clippy::excessive_precision)]
    let srgb_to_xyz = [
        [0.4123907993, 0.3575843394, 0.1804807884],
        [0.2126390059, 0.7151686788, 0.0721923154],
        [0.0193308187, 0.1191947798, 0.9505321522],
    ];

    // Invert srgb_to_xyz to get xyz_to_srgb
    let inv3 = |m: [[f64; 3]; 3]| -> [[f64; 3]; 3] {
        let d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let id = 1.0 / d;
        [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id,
            ],
        ]
    };

    let xyz_to_srgb = inv3(srgb_to_xyz);

    // Result = xyz_to_srgb @ primaries_to_xyz
    let mut result = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0f64;
            for k in 0..3 {
                sum += xyz_to_srgb[i][k] * p2x[k][j];
            }
            result[i][j] = sum as f32;
        }
    }
    result
}

/// Compute the primaries-to-sRGB matrix for a given color encoding, if needed.
/// Returns None for sRGB (no transform needed).
pub(crate) fn primaries_to_srgb_matrix(
    ce: &crate::headers::color_encoding::ColorEncoding,
) -> Option<[[f32; 3]; 3]> {
    match ce.primaries {
        Primaries::P3 => Some(P3_TO_SRGB),
        Primaries::Bt2100 => Some(BT2020_TO_SRGB),
        Primaries::Custom => {
            let cp = ce
                .custom_primaries
                .as_ref()
                .expect("custom_primaries must be set when primaries is Custom");
            Some(compute_primaries_to_srgb(
                (cp.red.x, cp.red.y),
                (cp.green.x, cp.green.y),
                (cp.blue.x, cp.blue.y),
            ))
        }
        Primaries::Srgb => None,
    }
}

/// Apply a 3x3 matrix to RGB row buffers in-place.
///
/// Uses chunks of 8 for autovectorization — LLVM emits SIMD for the inner
/// multiply-accumulate on the fixed-size slices without any bounds checks.
pub(crate) fn apply_matrix_3x3(r: &mut [f32], g: &mut [f32], b: &mut [f32], m: &[[f32; 3]; 3]) {
    let m00 = m[0][0];
    let m01 = m[0][1];
    let m02 = m[0][2];
    let m10 = m[1][0];
    let m11 = m[1][1];
    let m12 = m[1][2];
    let m20 = m[2][0];
    let m21 = m[2][1];
    let m22 = m[2][2];

    let len = r.len();
    let chunks = len / 8;
    let remainder = chunks * 8;

    for chunk in 0..chunks {
        let base = chunk * 8;
        let rs: &mut [f32; 8] = (&mut r[base..base + 8]).try_into().unwrap();
        let gs: &mut [f32; 8] = (&mut g[base..base + 8]).try_into().unwrap();
        let bs: &mut [f32; 8] = (&mut b[base..base + 8]).try_into().unwrap();
        for j in 0..8 {
            let ri = rs[j];
            let gi = gs[j];
            let bi = bs[j];
            rs[j] = m00 * ri + m01 * gi + m02 * bi;
            gs[j] = m10 * ri + m11 * gi + m12 * bi;
            bs[j] = m20 * ri + m21 * gi + m22 * bi;
        }
    }
    for i in remainder..len {
        let ri = r[i];
        let gi = g[i];
        let bi = b[i];
        r[i] = m00 * ri + m01 * gi + m02 * bi;
        g[i] = m10 * ri + m11 * gi + m12 * bi;
        b[i] = m20 * ri + m21 * gi + m22 * bi;
    }
}

impl VarDctEncoder {
    /// Convert linear RGB to XYB color space with padding to block boundaries.
    ///
    /// Returns (xyb_x, xyb_y, xyb_b) arrays padded to `padded_width × padded_height`
    /// using edge replication (last pixel value extended to the boundary).
    /// This allows SIMD code to process full blocks without bounds checking.
    pub(crate) fn convert_to_xyb_padded(
        &self,
        width: usize,
        height: usize,
        padded_width: usize,
        padded_height: usize,
        linear_rgb: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Determine if we need a primaries-to-sRGB conversion.
        // The XYB opsin matrix is defined for sRGB/BT.709 primaries.
        let primaries_matrix = self
            .color_encoding
            .as_ref()
            .and_then(primaries_to_srgb_matrix);

        let padded_n = padded_width * padded_height;
        let mut xyb_x = vec![0.0f32; padded_n];
        let mut xyb_y = vec![0.0f32; padded_n];
        let mut xyb_b = vec![0.0f32; padded_n];

        // Scratch buffers for deinterleaving one row of RGB input
        let mut row_r = vec![0.0f32; width];
        let mut row_g = vec![0.0f32; width];
        let mut row_b = vec![0.0f32; width];

        // Convert the actual image pixels row by row
        for y in 0..height {
            let src_row = y * width;

            // Deinterleave RGB row
            for x in 0..width {
                let si = (src_row + x) * 3;
                row_r[x] = linear_rgb[si];
                row_g[x] = linear_rgb[si + 1];
                row_b[x] = linear_rgb[si + 2];
            }

            // Transform non-sRGB primaries to sRGB before XYB conversion
            if let Some(ref m) = primaries_matrix {
                apply_matrix_3x3(&mut row_r, &mut row_g, &mut row_b, m);
            }

            // Convert row via SIMD (or scalar fallback)
            let dst_row = y * padded_width;
            jxl_simd::linear_rgb_to_xyb_batch(
                &row_r,
                &row_g,
                &row_b,
                &mut xyb_x[dst_row..dst_row + width],
                &mut xyb_y[dst_row..dst_row + width],
                &mut xyb_b[dst_row..dst_row + width],
            );

            #[cfg(feature = "debug-dc")]
            if y == 0 {
                eprintln!(
                    "XYB[0,0]: linear_rgb=({:.6},{:.6},{:.6}) -> XYB=({:.6},{:.6},{:.6})",
                    row_r[0], row_g[0], row_b[0], xyb_x[0], xyb_y[0], xyb_b[0]
                );
            }

            // Pad right edge with last pixel value (edge replication)
            if padded_width > width {
                let last_x_idx = dst_row + width - 1;
                let last_x = xyb_x[last_x_idx];
                let last_y = xyb_y[last_x_idx];
                let last_b = xyb_b[last_x_idx];
                for x in width..padded_width {
                    let dst_idx = dst_row + x;
                    xyb_x[dst_idx] = last_x;
                    xyb_y[dst_idx] = last_y;
                    xyb_b[dst_idx] = last_b;
                }
            }
        }

        // Pad bottom rows by copying the last row
        if padded_height > height {
            let last_row_start = (height - 1) * padded_width;
            for y in height..padded_height {
                let dst_row_start = y * padded_width;
                xyb_x.copy_within(last_row_start..last_row_start + padded_width, dst_row_start);
                xyb_y.copy_within(last_row_start..last_row_start + padded_width, dst_row_start);
                xyb_b.copy_within(last_row_start..last_row_start + padded_width, dst_row_start);
            }
        }

        (xyb_x, xyb_y, xyb_b)
    }
}
