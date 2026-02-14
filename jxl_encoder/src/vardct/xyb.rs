// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! XYB color space conversion with padding.
//!
//! Converts linear RGB to XYB color space and pads to block boundaries.
//! Top SIMD optimization target — the inner loop is a pure per-pixel transform.

use super::encoder::VarDctEncoder;

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
