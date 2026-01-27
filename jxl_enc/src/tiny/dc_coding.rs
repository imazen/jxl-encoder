// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! DC coefficient coding with gradient predictor.
//!
//! DC coefficients are coded using a ClampedGradient predictor that uses
//! the left, top, and top-left neighbors to predict each value. The residual
//! (actual - prediction) is then entropy coded with context based on the
//! gradient property.

use super::common::pack_signed;
use super::entropy_code::{EntropyCode, write_token};
use super::token::Token;
use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Compute the clamped gradient prediction from neighbors.
///
/// Given the north (top), west (left), and northwest (topleft) neighbors,
/// computes a prediction that is:
/// - The gradient (n + w - l) if it falls between min(n,w) and max(n,w)
/// - Clamped to the range [min(n,w), max(n,w)] otherwise
///
/// This predictor is good for smooth gradients while handling edges well.
#[inline]
pub fn clamped_gradient(n: i32, w: i32, l: i32) -> i32 {
    let m = n.min(w);
    let big_m = n.max(w);
    // Compute gradient with overflow protection
    let grad = (n as i64 + w as i64 - l as i64) as i32;
    // Clamp to [m, M]
    let grad_clamp_m = if l < m { big_m } else { grad };
    if l > big_m { m } else { grad_clamp_m }
}

/// Context lookup table for DC coding based on gradient property.
///
/// The gradient property is computed as 512 + top + left - topleft, clamped to [0, 1023].
/// This table maps gradient properties to one of 45 DC contexts (values 11-44).
#[rustfmt::skip]
pub static GRADIENT_CONTEXT_LUT: [u8; 1024] = [
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 39, 39, 39, 39, 39, 39,
    39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39,
    39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39,
    39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 38,
    38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
    38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
    38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
    38, 38, 38, 38, 38, 38, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
    36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 35, 35, 35, 35, 35, 35,
    35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32, 32, 32,
    32, 32, 32, 32, 31, 31, 31, 31, 30, 30, 30, 30, 29, 29, 29, 28, 27, 27, 26,
    42, 41, 41, 25, 25, 24, 24, 23, 23, 23, 23, 22, 22, 22, 22, 21, 21, 21, 21,
    21, 21, 21, 21, 20, 20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
];

/// Constants for gradient range computation.
const GRAD_RANGE_MIN: i64 = 0;
const GRAD_RANGE_MID: i64 = 512;
const GRAD_RANGE_MAX: i64 = 1023;

/// Number of DC contexts.
pub const NUM_DC_CONTEXTS: usize = 45;

/// Encode DC coefficients using gradient predictor and entropy coding.
///
/// DC coefficients are organized as [channel][y][x] where channel order is:
/// Y (1), X (0), B (2) for encoding.
///
/// # Arguments
/// * `quant_dc` - Quantized DC coefficients for each channel, shape [3][height][width]
/// * `dc_code` - DC entropy code to use for token writing
/// * `writer` - BitWriter to write encoded data
pub fn write_dc_tokens(
    quant_dc: &[Vec<Vec<i16>>; 3],
    dc_code: &EntropyCode,
    writer: &mut BitWriter,
) -> Result<()> {
    if quant_dc[0].is_empty() || quant_dc[0][0].is_empty() {
        return Ok(());
    }

    let height = quant_dc[0].len();
    let width = quant_dc[0][0].len();

    // Encode in channel order: Y (1), X (0), B (2)
    for &c in &[1, 0, 2] {
        let channel = &quant_dc[c];
        for y in 0..height {
            for x in 0..width {
                // Get neighbor values with edge handling
                let left = if x > 0 {
                    channel[y][x - 1] as i32
                } else if y > 0 {
                    channel[y - 1][x] as i32
                } else {
                    0
                };

                let top = if y > 0 {
                    channel[y - 1][x] as i32
                } else {
                    left
                };

                let topleft = if x > 0 && y > 0 {
                    channel[y - 1][x - 1] as i32
                } else {
                    left
                };

                // Compute prediction and residual
                let guess = clamped_gradient(top, left, topleft);
                let actual = channel[y][x] as i32;
                let residual = actual - guess;

                // Compute gradient property for context lookup
                let grad_prop = (GRAD_RANGE_MID + top as i64 + left as i64 - topleft as i64)
                    .clamp(GRAD_RANGE_MIN, GRAD_RANGE_MAX) as usize;
                let ctx_id = GRADIENT_CONTEXT_LUT[grad_prop] as u32;

                // Create and write token
                let token = Token::new(ctx_id, pack_signed(residual));
                write_token(&token, dc_code, writer)?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamped_gradient_simple() {
        // When all neighbors are equal, prediction = gradient = the value
        assert_eq!(clamped_gradient(10, 10, 10), 10);

        // Gradient prediction: n + w - l = 10 + 20 - 15 = 15
        // Which is in range [10, 20], so use it
        assert_eq!(clamped_gradient(10, 20, 15), 15);

        // Gradient prediction: n + w - l = 10 + 20 - 5 = 25
        // 25 > max(10, 20) = 20, and l=5 < min(10,20)=10, so return M=20
        assert_eq!(clamped_gradient(10, 20, 5), 20);

        // Gradient prediction: n + w - l = 10 + 20 - 25 = 5
        // 5 < min(10, 20) = 10, and l=25 > max(10,20)=20, so return m=10
        assert_eq!(clamped_gradient(10, 20, 25), 10);
    }

    #[test]
    fn test_clamped_gradient_edges() {
        // Test with zeros (common at image edges)
        assert_eq!(clamped_gradient(0, 0, 0), 0);
        assert_eq!(clamped_gradient(100, 0, 0), 100);
        assert_eq!(clamped_gradient(0, 100, 0), 100);
    }

    #[test]
    fn test_gradient_context_lut_bounds() {
        // Verify all LUT values are valid context IDs (11-44)
        for &ctx in &GRADIENT_CONTEXT_LUT {
            assert!(
                ctx >= 11 && ctx <= 44,
                "Context {} out of expected range [11, 44]",
                ctx
            );
        }
    }

    #[test]
    fn test_gradient_context_lut_size() {
        assert_eq!(GRADIENT_CONTEXT_LUT.len(), 1024);
    }

    #[test]
    fn test_write_dc_tokens_empty() {
        let quant_dc: [Vec<Vec<i16>>; 3] = [vec![], vec![], vec![]];
        let dc_code = super::super::static_codes::get_dc_entropy_code();
        let mut writer = BitWriter::new();
        assert!(write_dc_tokens(&quant_dc, &dc_code, &mut writer).is_ok());
        assert_eq!(writer.bits_written(), 0);
    }

    #[test]
    fn test_write_dc_tokens_simple() {
        // Create a simple 2x2 DC image with all zeros
        let quant_dc: [Vec<Vec<i16>>; 3] = [
            vec![vec![0, 0], vec![0, 0]],
            vec![vec![0, 0], vec![0, 0]],
            vec![vec![0, 0], vec![0, 0]],
        ];
        let dc_code = super::super::static_codes::get_dc_entropy_code();
        let mut writer = BitWriter::new();
        assert!(write_dc_tokens(&quant_dc, &dc_code, &mut writer).is_ok());
        // Should have written some bits (12 tokens total: 3 channels * 2 * 2)
        assert!(writer.bits_written() > 0);
    }
}
