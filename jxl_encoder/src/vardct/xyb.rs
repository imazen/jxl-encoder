// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! XYB color space conversion with padding.
//!
//! Converts linear RGB to XYB color space and pads to block boundaries.
//! Top SIMD optimization target — the inner loop is a pure per-pixel transform.

use super::encoder::VarDctEncoder;
use crate::color::xyb::linear_rgb_to_xyb;

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

        // Convert the actual image pixels
        for y in 0..height {
            for x in 0..width {
                let src_idx = y * width + x;
                let dst_idx = y * padded_width + x;
                let r = linear_rgb[src_idx * 3];
                let g = linear_rgb[src_idx * 3 + 1];
                let b = linear_rgb[src_idx * 3 + 2];
                let (xv, yv, bv) = linear_rgb_to_xyb(r, g, b);
                #[cfg(feature = "debug-dc")]
                if x == 0 && y == 0 {
                    eprintln!(
                        "XYB[0,0]: linear_rgb=({:.6},{:.6},{:.6}) -> XYB=({:.6},{:.6},{:.6})",
                        r, g, b, xv, yv, bv
                    );
                }
                xyb_x[dst_idx] = xv;
                xyb_y[dst_idx] = yv;
                xyb_b[dst_idx] = bv;
            }

            // Pad right edge with last pixel value (edge replication)
            if padded_width > width {
                let last_x_idx = y * padded_width + (width - 1);
                let last_x = xyb_x[last_x_idx];
                let last_y = xyb_y[last_x_idx];
                let last_b = xyb_b[last_x_idx];
                for x in width..padded_width {
                    let dst_idx = y * padded_width + x;
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
                for x in 0..padded_width {
                    xyb_x[dst_row_start + x] = xyb_x[last_row_start + x];
                    xyb_y[dst_row_start + x] = xyb_y[last_row_start + x];
                    xyb_b[dst_row_start + x] = xyb_b[last_row_start + x];
                }
            }
        }

        (xyb_x, xyb_y, xyb_b)
    }
}
