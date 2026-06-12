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
pub(crate) const P3_TO_SRGB: [[f32; 3]; 3] = [
    [ 1.2249401763, -0.2249401763,  0.0000000000],
    [-0.0420569547,  1.0420569547,  0.0000000000],
    [-0.0196375546, -0.0786360456,  1.0982736001],
];

/// 3x3 matrix to convert linear BT.2020 RGB to linear sRGB RGB.
/// Derived from BT.2020→XYZ→sRGB chromatic adaptation (both D65 white point).
#[rustfmt::skip]
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
///
/// Returns `None` for sRGB (no transform needed). For `Primaries::Custom`
/// without a `custom_primaries` field, returns `None` to skip the matrix
/// application — this is a defensive fallback only; validation happens
/// upstream via [`validate_color_encoding`] which surfaces the
/// inconsistency as a regular [`crate::error::Error::InvalidInput`].
pub(crate) fn primaries_to_srgb_matrix(
    ce: &crate::headers::color_encoding::ColorEncoding,
) -> Option<[[f32; 3]; 3]> {
    match ce.primaries {
        Primaries::P3 => Some(P3_TO_SRGB),
        Primaries::Bt2100 => Some(BT2020_TO_SRGB),
        Primaries::Custom => {
            let cp = ce.custom_primaries.as_ref()?;
            Some(compute_primaries_to_srgb(
                (cp.red.x, cp.red.y),
                (cp.green.x, cp.green.y),
                (cp.blue.x, cp.blue.y),
            ))
        }
        Primaries::Srgb => None,
    }
}

/// Validate that a `ColorEncoding` is internally consistent.
///
/// Use this at every public encode boundary that accepts a `ColorEncoding`
/// to surface bad configurations as `Err(InvalidInput)` rather than letting
/// downstream code silently fall back or (pre-this-change) panic.
pub(crate) fn validate_color_encoding(
    ce: &crate::headers::color_encoding::ColorEncoding,
) -> crate::error::Result<()> {
    if matches!(ce.primaries, Primaries::Custom) && ce.custom_primaries.is_none() {
        return Err(crate::error::Error::InvalidInput(
            "ColorEncoding has Primaries::Custom but custom_primaries is None".into(),
        ));
    }
    Ok(())
}

/// Apply a 3x3 matrix to RGB row buffers in-place.
///
/// Uses chunks of 8 for autovectorization — LLVM emits SIMD for the inner
/// multiply-accumulate on the fixed-size slices without any bounds checks.
pub(crate) fn apply_matrix_3x3(r: &mut [f32], g: &mut [f32], b: &mut [f32], m: &[[f32; 3]; 3]) {
    // Callers MUST pass three slices of identical length. assert_eq!
    // surfaces a bug at the entry boundary with a clear message,
    // before the inner loop's slice indexing would panic with a less
    // useful "index out of range" message.
    assert_eq!(r.len(), g.len(), "apply_matrix_3x3: r/g length mismatch");
    assert_eq!(r.len(), b.len(), "apply_matrix_3x3: r/b length mismatch");
    let len = r.len();
    let m00 = m[0][0];
    let m01 = m[0][1];
    let m02 = m[0][2];
    let m10 = m[1][0];
    let m11 = m[1][1];
    let m12 = m[1][2];
    let m20 = m[2][0];
    let m21 = m[2][1];
    let m22 = m[2][2];

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
    ) -> crate::error::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        // Determine if we need a primaries-to-sRGB conversion.
        // The XYB opsin matrix is defined for sRGB/BT.709 primaries.
        let primaries_matrix = self
            .color_encoding
            .as_ref()
            .and_then(primaries_to_srgb_matrix);

        let padded_n = padded_width.checked_mul(padded_height).ok_or(
            crate::error::Error::DimensionOverflow {
                width: padded_width,
                height: padded_height,
                channels: 1,
            },
        )?;
        // Output planes are fully overwritten: rows 0..height by the per-row XYB
        // conversion + right-edge pad, rows height..padded_height by the bottom
        // pad loop below. Safe to use vec_f32_dirty.
        //
        // Reserve the three planes against the per-encode budget before
        // touching the heap. These buffers live for the full encode, so
        // we use the "permanent" helper that doesn't return a guard —
        // the budget itself is dropped at end-of-encode and recovers all
        // accounting wholesale. The `None` branch is a single cold
        // predicted-not-taken atomic load; allocations on the unbounded
        // path pay no overhead.
        let budget = self.budget.as_ref();
        let mut xyb_x = crate::budget::try_alloc_vec_f32_dirty_permanent(budget, padded_n)?;
        let mut xyb_y = crate::budget::try_alloc_vec_f32_dirty_permanent(budget, padded_n)?;
        let mut xyb_b = crate::budget::try_alloc_vec_f32_dirty_permanent(budget, padded_n)?;

        // libjxl ComputePremulAbsorb parity (issue #73): HDR transfer
        // functions resolve to intensity_target 10,000 (PQ) / 1,000
        // (HLG); SDR stays 255 → mul == 1.0 exactly (identity).
        let intensity_mul = self.intensity_target / 255.0;
        convert_rows_to_xyb(
            width,
            height,
            padded_width,
            linear_rgb,
            primaries_matrix.as_ref(),
            intensity_mul,
            &mut xyb_x,
            &mut xyb_y,
            &mut xyb_b,
        );

        // Pad bottom rows by copying the last row. Each channel's pad loop
        // is independent; run the three channels in parallel via rayon::scope
        // when the `parallel` feature is enabled.
        if padded_height > height {
            let last_row_start = (height - 1) * padded_width;
            pad_bottom_three_channels(
                &mut xyb_x,
                &mut xyb_y,
                &mut xyb_b,
                last_row_start,
                padded_width,
                height,
                padded_height,
            );
        }

        Ok((xyb_x, xyb_y, xyb_b))
    }
}

/// Pad bottom rows of 3 XYB planes in parallel. Each plane's pad loop only
/// reads its own last source row (snapshotted) and writes to disjoint rows,
/// so the three channels are independent.
#[allow(clippy::too_many_arguments)]
fn pad_bottom_three_channels(
    xyb_x: &mut [f32],
    xyb_y: &mut [f32],
    xyb_b: &mut [f32],
    last_row_start: usize,
    padded_width: usize,
    height: usize,
    padded_height: usize,
) {
    fn pad_one(
        plane: &mut [f32],
        last_row_start: usize,
        padded_width: usize,
        height: usize,
        padded_height: usize,
    ) {
        let mut last = jxl_simd::vec_f32_dirty(padded_width);
        last.copy_from_slice(&plane[last_row_start..last_row_start + padded_width]);
        for y in height..padded_height {
            let dst = y * padded_width;
            plane[dst..dst + padded_width].copy_from_slice(&last);
        }
    }

    #[cfg(feature = "parallel")]
    {
        let (((), ()), ()) = rayon::join(
            || {
                rayon::join(
                    || pad_one(xyb_x, last_row_start, padded_width, height, padded_height),
                    || pad_one(xyb_y, last_row_start, padded_width, height, padded_height),
                )
            },
            || pad_one(xyb_b, last_row_start, padded_width, height, padded_height),
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        pad_one(xyb_x, last_row_start, padded_width, height, padded_height);
        pad_one(xyb_y, last_row_start, padded_width, height, padded_height);
        pad_one(xyb_b, last_row_start, padded_width, height, padded_height);
    }
}

/// Strip height (in rows) for parallel XYB conversion. Chosen large enough to
/// amortize per-task scheduling overhead; small enough that typical images
/// produce multiple strips.
const XYB_STRIP_ROWS: usize = 16;

/// Convert rows 0..height of linear_rgb into XYB planes (with right-edge pad).
///
/// Output planes must be sized padded_width * padded_height (or at least
/// `height * padded_width`). Rows `height..padded_height` are NOT touched here —
/// the caller is responsible for bottom padding.
///
/// Parallelized in strips of `XYB_STRIP_ROWS` rows when the `parallel` feature
/// is enabled; each strip owns its own row scratch and writes to disjoint
/// output plane ranges. Bit-exact equivalent to the serial fallback: every
/// floating-point operation happens in the same order (inside `linear_rgb_to_xyb_batch`
/// for each pixel, and the per-row right-edge replication).
#[allow(clippy::too_many_arguments)]
fn convert_rows_to_xyb(
    width: usize,
    height: usize,
    padded_width: usize,
    linear_rgb: &[f32],
    primaries_matrix: Option<&[[f32; 3]; 3]>,
    intensity_mul: f32,
    xyb_x: &mut [f32],
    xyb_y: &mut [f32],
    xyb_b: &mut [f32],
) {
    let strip_len = XYB_STRIP_ROWS * padded_width;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        xyb_x[..height * padded_width]
            .par_chunks_mut(strip_len)
            .zip(xyb_y[..height * padded_width].par_chunks_mut(strip_len))
            .zip(xyb_b[..height * padded_width].par_chunks_mut(strip_len))
            .enumerate()
            .for_each(|(strip_idx, ((strip_x, strip_y), strip_b))| {
                let y_start = strip_idx * XYB_STRIP_ROWS;
                let strip_rows = strip_x.len() / padded_width;
                convert_strip(
                    width,
                    padded_width,
                    y_start,
                    strip_rows,
                    linear_rgb,
                    primaries_matrix,
                    intensity_mul,
                    strip_x,
                    strip_y,
                    strip_b,
                );
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let full_len = height * padded_width;
        let mut y_start = 0;
        let mut offset = 0;
        while offset < full_len {
            let this_len = strip_len.min(full_len - offset);
            let strip_rows = this_len / padded_width;
            convert_strip(
                width,
                padded_width,
                y_start,
                strip_rows,
                linear_rgb,
                primaries_matrix,
                intensity_mul,
                &mut xyb_x[offset..offset + this_len],
                &mut xyb_y[offset..offset + this_len],
                &mut xyb_b[offset..offset + this_len],
            );
            y_start += strip_rows;
            offset += this_len;
        }
    }
}

/// Convert a single strip of rows. Self-contained: owns its per-row scratch.
///
/// Writes every element of `strip_x`, `strip_y`, `strip_b` (which correspond to
/// the full padded_width of each row in the strip). Safe to call with
/// dirty-initialized output slices.
#[allow(clippy::too_many_arguments)]
fn convert_strip(
    width: usize,
    padded_width: usize,
    y_start: usize,
    strip_rows: usize,
    linear_rgb: &[f32],
    primaries_matrix: Option<&[[f32; 3]; 3]>,
    intensity_mul: f32,
    strip_x: &mut [f32],
    strip_y: &mut [f32],
    strip_b: &mut [f32],
) {
    // Per-strip scratch for deinterleaving one row of RGB input. Allocated
    // fresh per strip; for a sequential run this is one allocation per strip
    // (at most ~padded_height/16), negligible vs the row allocations it replaces.
    let mut row_r = jxl_simd::vec_f32_dirty(width);
    let mut row_g = jxl_simd::vec_f32_dirty(width);
    let mut row_b = jxl_simd::vec_f32_dirty(width);

    for local_y in 0..strip_rows {
        let y = y_start + local_y;
        let src_row = y * width;

        // Deinterleave RGB row. `intensity_mul` is libjxl's
        // ComputePremulAbsorb scale (`enc_xyb.cc:217`,
        // `intensity_target / 255`): the opsin matrix is linear in RGB,
        // so pre-scaling the samples is exactly equivalent to
        // premultiplying the matrix. For SDR it is exactly 1.0 and the
        // multiply is an IEEE identity — byte-identical output. For PQ
        // (intensity_target 10,000) this is the issue-#73 fix: without
        // it, PQ-linear data (1.0 = 10,000 nits) entered XYB on the
        // 255-nit SDR scale and the image was destroyed.
        for x in 0..width {
            let si = (src_row + x) * 3;
            row_r[x] = linear_rgb[si] * intensity_mul;
            row_g[x] = linear_rgb[si + 1] * intensity_mul;
            row_b[x] = linear_rgb[si + 2] * intensity_mul;
        }

        // Transform non-sRGB primaries to sRGB before XYB conversion
        if let Some(m) = primaries_matrix {
            apply_matrix_3x3(&mut row_r, &mut row_g, &mut row_b, m);
        }

        // Convert row via SIMD (or scalar fallback). Output goes to the strip's
        // row slice, offset by local_y within the strip.
        let dst_row = local_y * padded_width;
        jxl_simd::linear_rgb_to_xyb_batch(
            &row_r,
            &row_g,
            &row_b,
            &mut strip_x[dst_row..dst_row + width],
            &mut strip_y[dst_row..dst_row + width],
            &mut strip_b[dst_row..dst_row + width],
        );

        #[cfg(feature = "debug-dc")]
        if y == 0 {
            eprintln!(
                "XYB[0,0]: linear_rgb=({:.6},{:.6},{:.6}) -> XYB=({:.6},{:.6},{:.6})",
                row_r[0], row_g[0], row_b[0], strip_x[0], strip_y[0], strip_b[0]
            );
        }

        // Pad right edge with last pixel value (edge replication)
        if padded_width > width {
            let last_x_idx = dst_row + width - 1;
            let last_x = strip_x[last_x_idx];
            let last_y = strip_y[last_x_idx];
            let last_b = strip_b[last_x_idx];
            for x in width..padded_width {
                let dst_idx = dst_row + x;
                strip_x[dst_idx] = last_x;
                strip_y[dst_idx] = last_y;
                strip_b[dst_idx] = last_b;
            }
        }
    }
}
