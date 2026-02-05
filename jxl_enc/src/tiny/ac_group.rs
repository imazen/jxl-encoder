// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! AC group encoding - quantization and tokenization of AC coefficients.
//!
//! This module handles encoding AC coefficients within a group:
//! - Counting non-zero coefficients
//! - Predicting non-zeros from neighbors
//! - Tokenizing coefficients in zig-zag scan order
//!
//! Ported from libjxl-tiny enc_group.cc.

use std::sync::LazyLock;

use super::ac_context::{NON_ZERO_BUCKETS, ZERO_DENSITY_CONTEXT_COUNT, zero_density_context};
use super::common::{DCT_BLOCK_SIZE, pack_signed};
use super::entropy_code::{EntropyCode, write_token};
use super::token::Token;
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::error::Result;

/// Compute non-zero context from predicted non-zeros, block context, and num_ctxs.
/// Same formula as BlockCtxMap::non_zero_context but standalone.
#[inline]
fn nz_context(non_zeros: usize, block_ctx: usize, num_ctxs: usize) -> usize {
    let nz_bucket = if non_zeros < 8 {
        non_zeros
    } else if non_zeros >= 64 {
        36
    } else {
        4 + non_zeros / 2
    };
    nz_bucket * num_ctxs + block_ctx
}

/// Compute zero density contexts offset from block context and num_ctxs.
/// Same formula as BlockCtxMap::zero_density_contexts_offset but standalone.
#[inline]
fn zd_offset(block_ctx: usize, num_ctxs: usize) -> usize {
    num_ctxs * NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT * block_ctx
}

/// Default zig-zag coefficient order for DCT8 (8x8).
/// Excludes DC at position 0, so indices 1-63 map to AC coefficients.
#[rustfmt::skip]
pub static COEFF_ORDER_8X8: [u32; 64] = [
    0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,
    5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14,
   21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30,
   37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54,
   47, 55, 62, 63,
];

/// Default zig-zag coefficient order for DCT8x16 / DCT16x8 (128 coefficients).
/// This follows the pattern used in libjxl-tiny.
#[rustfmt::skip]
pub static COEFF_ORDER_8X16: [u32; 128] = [
    0,   1,  16,   2,   3,  17,  32,  18,   4,   5,  19,
   33,  48,  34,  20,   6,   7,  21,  35,  49,  64,  50,  36,  22,   8,   9,
   23,  37,  51,  65,  80,  66,  52,  38,  24,  10,  11,  25,  39,  53,  67,
   81,  96,  82,  68,  54,  40,  26,  12,  13,  27,  41,  55,  69,  83,  97,
  112,  98,  84,  70,  56,  42,  28,  14,  15,  29,  43,  57,  71,  85,  99,
  113, 114, 100,  86,  72,  58,  44,  30,  31,  45,  59,  73,  87, 101, 115,
  116, 102,  88,  74,  60,  46,  47,  61,  75,  89, 103, 117, 118, 104,  90,
   76,  62,  63,  77,  91, 105, 119, 120, 106,  92,  78,  79,  93, 107, 121,
  122, 108,  94,  95, 109, 123, 124, 110, 111, 125, 126, 127,
];

/// Default zig-zag coefficient order for DCT16x16 (256 coefficients).
/// Natural scan order for the 16x16 coefficient grid.
#[rustfmt::skip]
pub static COEFF_ORDER_16X16: [u32; 256] = [
      0,   1,  16,  17,  32,   2,   3,  18,  33,  48,  64,  49,  34,  19,   4,   5,
     20,  35,  50,  65,  80,  96,  81,  66,  51,  36,  21,   6,   7,  22,  37,  52,
     67,  82,  97, 112, 128, 113,  98,  83,  68,  53,  38,  23,   8,   9,  24,  39,
     54,  69,  84,  99, 114, 129, 144, 160, 145, 130, 115, 100,  85,  70,  55,  40,
     25,  10,  11,  26,  41,  56,  71,  86, 101, 116, 131, 146, 161, 176, 192, 177,
    162, 147, 132, 117, 102,  87,  72,  57,  42,  27,  12,  13,  28,  43,  58,  73,
     88, 103, 118, 133, 148, 163, 178, 193, 208, 224, 209, 194, 179, 164, 149, 134,
    119, 104,  89,  74,  59,  44,  29,  14,  15,  30,  45,  60,  75,  90, 105, 120,
    135, 150, 165, 180, 195, 210, 225, 240, 241, 226, 211, 196, 181, 166, 151, 136,
    121, 106,  91,  76,  61,  46,  31,  47,  62,  77,  92, 107, 122, 137, 152, 167,
    182, 197, 212, 227, 242, 243, 228, 213, 198, 183, 168, 153, 138, 123, 108,  93,
     78,  63,  79,  94, 109, 124, 139, 154, 169, 184, 199, 214, 229, 244, 245, 230,
    215, 200, 185, 170, 155, 140, 125, 110,  95, 111, 126, 141, 156, 171, 186, 201,
    216, 231, 246, 247, 232, 217, 202, 187, 172, 157, 142, 127, 143, 158, 173, 188,
    203, 218, 233, 248, 249, 234, 219, 204, 189, 174, 159, 175, 190, 205, 220, 235,
    250, 251, 236, 221, 206, 191, 207, 222, 237, 252, 253, 238, 223, 239, 254, 255,
];

/// Default zig-zag coefficient order for DCT32x32 (1024 coefficients).
/// Generated at runtime via CoefficientLayout scan: LLF positions first,
/// then remaining AC positions in zig-zag order.
static COEFF_ORDER_32X32: LazyLock<Vec<u32>> =
    LazyLock::new(|| coefficient_layout_order(32, 32, 4, 4));

/// Default zig-zag coefficient order for DCT32x16 (512 coefficients).
/// DCT32x16: 32 rows × 16 cols, LLF region is 4×2 (4 rows × 2 cols).
static COEFF_ORDER_32X16: LazyLock<Vec<u32>> =
    LazyLock::new(|| coefficient_layout_order(32, 16, 4, 2));

/// Default zig-zag coefficient order for DCT16x32 (512 coefficients).
/// DCT16x32: 16 rows × 32 cols, LLF region is 2×4 (2 rows × 4 cols).
static COEFF_ORDER_16X32: LazyLock<Vec<u32>> =
    LazyLock::new(|| coefficient_layout_order(16, 32, 2, 4));

/// Generate a coefficient order with LLF positions first, then AC in zig-zag.
///
/// For transforms larger than 8x8, the first `llf_x * llf_y` positions must be
/// the LLF (Lowest Low Frequency) coefficients that will be filled from DC.
/// The remaining positions are AC coefficients scanned in zig-zag order.
///
/// The LLF positions in our layout (stride = `cols`) are at:
///   `lx * cols + ly` for lx in 0..llf_x, ly in 0..llf_y
///
/// This matches how `tokenize_ac_coefficients` skips the first `covered_blocks`
/// entries (treating them as LLF) and encodes the rest as AC.
fn coefficient_layout_order(rows: usize, cols: usize, llf_x: usize, llf_y: usize) -> Vec<u32> {
    let size = rows * cols;
    let mut order = Vec::with_capacity(size);

    // Build set of LLF positions for quick lookup
    let mut is_llf = vec![false; size];
    for lx in 0..llf_x {
        for ly in 0..llf_y {
            let pos = lx * cols + ly;
            is_llf[pos] = true;
        }
    }

    // Phase 1: LLF positions in row-major order within the LLF region
    for ly in 0..llf_y {
        for lx in 0..llf_x {
            order.push((lx * cols + ly) as u32);
        }
    }
    debug_assert_eq!(order.len(), llf_x * llf_y);

    // Phase 2: Remaining (non-LLF) positions in zig-zag diagonal scan
    for d in 0..(rows + cols - 1) {
        if d % 2 == 0 {
            // Even diagonal: go from bottom-left to top-right
            let r_start = d.min(rows - 1);
            let r_end = if d >= cols { d - cols + 1 } else { 0 };
            let mut r = r_start;
            loop {
                let c = d - r;
                let pos = r * cols + c;
                if !is_llf[pos] {
                    order.push(pos as u32);
                }
                if r == r_end {
                    break;
                }
                r -= 1;
            }
        } else {
            // Odd diagonal: go from top-right to bottom-left
            let r_start = if d >= cols { d - cols + 1 } else { 0 };
            let r_end = d.min(rows - 1);
            for r in r_start..=r_end {
                let c = d - r;
                let pos = r * cols + c;
                if !is_llf[pos] {
                    order.push(pos as u32);
                }
            }
        }
    }
    debug_assert_eq!(order.len(), size);
    order
}

/// Get coefficient order based on AC strategy (bitstream code).
///
/// Bitstream strategy codes:
/// - 0 = DCT8 (8x8, 64 coeffs)
/// - 4 = DCT16X16 (16x16, 256 coeffs)
/// - 5 = DCT32X32 (32x32, 1024 coeffs)
/// - 6 = DCT8X16 (8x16, 128 coeffs)
/// - 7 = DCT16X8 (16x8, 128 coeffs)
/// - 12 = DCT4X8 (8x8 with 4x8 sub-blocks, 64 coeffs)
/// - 13 = DCT8X4 (8x8 with 8x4 sub-blocks, 64 coeffs)
/// - 3 = DCT4X4 (8x8 with 4x4 sub-blocks, 64 coeffs)
/// - 14-17 = AFV0-AFV3 (8x8 hybrid transform, 64 coeffs)
pub fn get_coeff_order(strategy_code: u8) -> &'static [u32] {
    match strategy_code {
        0 => &COEFF_ORDER_8X8,                 // DCT8
        3 | 12 | 13 => &COEFF_ORDER_8X8,       // DCT4X4, DCT4X8, DCT8X4 (64 coeffs like DCT8)
        14..=17 => &COEFF_ORDER_8X8,            // AFV0-AFV3 (64 coeffs like DCT8)
        4 => &COEFF_ORDER_16X16,               // DCT16X16
        5 => &COEFF_ORDER_32X32,               // DCT32X32
        6 | 7 => &COEFF_ORDER_8X16,            // DCT8X16, DCT16X8
        10 => &COEFF_ORDER_32X16,              // DCT32X16
        11 => &COEFF_ORDER_16X32,              // DCT16X32
        _ => &COEFF_ORDER_8X8,                 // Default to 8x8 for unknown strategies
    }
}

/// Count non-zero coefficients in an 8x8 block, excluding DC.
///
/// Returns the count and also stores it in the nzeros_pos slot.
#[inline]
pub fn num_nonzero_8x8_except_dc(block: &[i32; DCT_BLOCK_SIZE], nzeros_pos: &mut u8) -> u8 {
    let mut nzeros = 0u8;
    // Skip DC at position 0, count non-zeros in positions 1-63
    for &coef in &block[1..] {
        if coef != 0 {
            nzeros += 1;
        }
    }
    *nzeros_pos = nzeros;
    nzeros
}

/// Count non-zero coefficients excluding LLF (Low-Low Frequency / DC region).
///
/// For larger transforms (DCT16x8, DCT8x16), the LLF region is larger than 1 coefficient.
/// The LLF region is cx*cy coefficients at the top-left.
///
/// Also stores the shifted nzeros (nzeros / covered_blocks) in each 8x8 block position.
pub fn num_nonzero_except_llf(
    cx: usize,
    cy: usize,
    block: &[i32],
    nzeros_stride: usize,
    nzeros_pos: &mut [u8],
    covered_blocks_x: usize,
    covered_blocks_y: usize,
) -> u8 {
    let block_dim = 8;
    let covered_blocks = cx * cy;
    let log2_covered_blocks = covered_blocks.trailing_zeros() as usize;

    let mut nzeros = 0i32;
    let total_coeffs = cx * cy * DCT_BLOCK_SIZE;

    // Count all non-zeros first
    for y in 0..(cy * block_dim) {
        let row_start = y * cx * block_dim;
        for x in 0..(cx * block_dim) {
            let coef = block[row_start + x];
            if coef != 0 {
                nzeros += 1;
            }
        }
    }

    // Subtract LLF region (top-left cx*cy coefficients that form DC for the larger transform)
    for y in 0..cy {
        for x in 0..cx {
            if block[y * cx * block_dim + x] != 0 {
                nzeros -= 1;
            }
        }
    }

    // Clamp to valid range
    let nzeros = nzeros.max(0) as u8;

    // Compute shifted nzeros for per-8x8-block storage
    let shifted_nzeros = ((nzeros as usize + covered_blocks - 1) >> log2_covered_blocks) as u8;

    // Fill in all covered 8x8 block positions
    for y in 0..covered_blocks_y {
        for x in 0..covered_blocks_x {
            nzeros_pos[x + y * nzeros_stride] = shifted_nzeros;
        }
    }

    // Return actual nzeros (not the total_coeffs, but only AC)
    // For consistency, clamp to the number of AC coefficients
    let max_ac = (total_coeffs - covered_blocks) as u8;
    nzeros.min(max_ac)
}

/// Predict number of non-zeros from top and left neighbors.
///
/// If at left edge, use top value (or default if no top).
/// If at top edge, use left value.
/// Otherwise, average top and left with rounding.
#[inline]
pub fn predict_from_top_and_left(
    row_top: Option<&[u8]>,
    row: &[u8],
    x: usize,
    default_val: i32,
) -> i32 {
    if x == 0 {
        match row_top {
            Some(top) => top[x] as i32,
            None => default_val,
        }
    } else {
        match row_top {
            Some(top) => (top[x] as i32 + row[x - 1] as i32 + 1) / 2,
            None => row[x - 1] as i32,
        }
    }
}

use super::ac_strategy::{COVERED_X, COVERED_Y, STRATEGY_CODE_LUT};

/// Get block size info for AC strategy.
/// Returns (cx, cy, covered_blocks, log2_covered_blocks, strategy_code).
///
/// Uses RAW strategy codes (0-11) as input, returns bitstream strategy code.
pub fn ac_strategy_info(raw_strategy: u8) -> (usize, usize, usize, usize, u8) {
    let cx = COVERED_X[raw_strategy as usize];
    let cy = COVERED_Y[raw_strategy as usize];
    let covered_blocks = cx * cy;
    let log2_covered_blocks = covered_blocks.trailing_zeros() as usize;
    let strategy_code = STRATEGY_CODE_LUT[raw_strategy as usize];

    (cx, cy, covered_blocks, log2_covered_blocks, strategy_code)
}

/// Tokenize AC coefficients for a single block/transform.
///
/// This is the core tokenization loop that:
/// 1. Writes the number of non-zeros as first token
/// 2. Iterates through coefficients in zig-zag order
/// 3. Writes each coefficient with appropriate context
///
/// # Arguments
/// * `quantized` - Quantized coefficients in natural order (not zig-zag)
/// * `raw_strategy` - AC strategy code
/// * `nzeros` - Number of non-zero AC coefficients (already computed)
/// * `predicted_nzeros` - Predicted nzeros from neighbors
/// * `block_ctx` - Block context ID (from BlockCtxMap)
/// * `num_ctxs` - Total number of block contexts (from BlockCtxMap)
/// * `ac_code` - Entropy code for AC coefficients
/// * `writer` - Bitstream writer
#[allow(clippy::too_many_arguments)]
pub fn tokenize_ac_coefficients(
    quantized: &[i32],
    raw_strategy: u8,
    nzeros: u8,
    predicted_nzeros: i32,
    block_ctx: usize,
    num_ctxs: usize,
    ac_code: &EntropyCode,
    writer: &mut BitWriter,
    custom_order: Option<&[u32]>,
) -> Result<()> {
    let (cx, cy, covered_blocks, log2_covered_blocks, strategy_code) =
        ac_strategy_info(raw_strategy);
    let size = cx * cy * DCT_BLOCK_SIZE;

    // Compute contexts from block_ctx and num_ctxs
    let nzero_ctx = nz_context(predicted_nzeros as usize, block_ctx, num_ctxs);
    let histo_offset = zd_offset(block_ctx, num_ctxs);

    // Write number of non-zeros as first token
    let nz_token = Token::new(nzero_ctx as u32, nzeros as u32);
    #[cfg(feature = "debug-tokens")]
    let bits_before = writer.bits_written();
    write_token(&nz_token, ac_code, None, writer)?;
    #[cfg(feature = "debug-tokens")]
    {
        let bits_after = writer.bits_written();
        let prefix_idx = ac_code.context_map[nzero_ctx] as usize;
        let pc = &ac_code.prefix_codes[prefix_idx];
        debug_log!(
            "  AC nzeros token: ctx={}, value={}, prefix_idx={}, depth={}, bits={:#x}, wrote {} bits",
            nzero_ctx,
            nzeros,
            prefix_idx,
            pc.depths[0],
            pc.bits[0],
            bits_after - bits_before
        );
    }

    // Get coefficient order (custom or default)
    // NOTE: get_coeff_order takes bitstream strategy_code, not raw_strategy
    let order = custom_order.unwrap_or_else(|| get_coeff_order(strategy_code));

    // Track remaining non-zeros and previous coefficient status
    let mut nzeros_left = nzeros as usize;
    let mut prev = if nzeros_left > size / 16 { 0 } else { 1 };

    // Skip LLF coefficients, start at covered_blocks
    for k in covered_blocks..size.min(order.len()) {
        if nzeros_left == 0 {
            break;
        }

        let coef = quantized[order[k] as usize];

        // Compute context for this coefficient
        let ctx = histo_offset
            + zero_density_context(nzeros_left, k, covered_blocks, log2_covered_blocks, prev);

        // Encode coefficient
        let u_coef = pack_signed(coef);
        let token = Token::new(ctx as u32, u_coef);
        write_token(&token, ac_code, None, writer)?;

        // Update tracking
        if coef != 0 {
            prev = 1;
            nzeros_left -= 1;
        } else {
            prev = 0;
        }
    }

    debug_assert_eq!(nzeros_left, 0, "Not all non-zeros were encoded");
    Ok(())
}

/// Collect AC coefficient tokens for a single block/transform (without writing).
///
/// Same logic as `tokenize_ac_coefficients()` but returns a `Vec<Token>`.
pub fn collect_ac_coefficients(
    quantized: &[i32],
    raw_strategy: u8,
    nzeros: u8,
    predicted_nzeros: i32,
    block_ctx: usize,
    num_ctxs: usize,
    custom_order: Option<&[u32]>,
) -> Vec<Token> {
    let (cx, cy, covered_blocks, log2_covered_blocks, strategy_code) =
        ac_strategy_info(raw_strategy);
    let size = cx * cy * DCT_BLOCK_SIZE;

    let nzero_ctx = nz_context(predicted_nzeros as usize, block_ctx, num_ctxs);
    let histo_offset = zd_offset(block_ctx, num_ctxs);

    // Capacity: 1 nzeros token + up to nzeros coefficient tokens
    let mut tokens = Vec::with_capacity(1 + nzeros as usize);

    // Write number of non-zeros as first token
    tokens.push(Token::new(nzero_ctx as u32, nzeros as u32));

    // Get coefficient order (custom or default)
    // NOTE: get_coeff_order takes bitstream strategy_code, not raw_strategy
    let order = custom_order.unwrap_or_else(|| get_coeff_order(strategy_code));

    let mut nzeros_left = nzeros as usize;
    let mut prev = if nzeros_left > size / 16 { 0 } else { 1 };

    for k in covered_blocks..size.min(order.len()) {
        if nzeros_left == 0 {
            break;
        }

        let coef = quantized[order[k] as usize];
        let ctx = histo_offset
            + zero_density_context(nzeros_left, k, covered_blocks, log2_covered_blocks, prev);
        let u_coef = pack_signed(coef);
        tokens.push(Token::new(ctx as u32, u_coef));

        if coef != 0 {
            prev = 1;
            nzeros_left -= 1;
        } else {
            prev = 0;
        }
    }

    debug_assert_eq!(nzeros_left, 0, "Not all non-zeros were collected");
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coeff_order_8x8() {
        // Verify zig-zag order properties
        assert_eq!(COEFF_ORDER_8X8[0], 0); // DC first
        assert_eq!(COEFF_ORDER_8X8[1], 1); // Then (0,1)
        assert_eq!(COEFF_ORDER_8X8[2], 8); // Then (1,0)
        assert_eq!(COEFF_ORDER_8X8[63], 63); // Last is (7,7)

        // Verify all 64 positions are present
        let mut seen = [false; 64];
        for &idx in &COEFF_ORDER_8X8 {
            seen[idx as usize] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "Not all positions in zig-zag order"
        );
    }

    #[test]
    fn test_coeff_order_8x16() {
        // Verify first element is DC
        assert_eq!(COEFF_ORDER_8X16[0], 0);

        // Verify all 128 positions are present
        let mut seen = [false; 128];
        for &idx in &COEFF_ORDER_8X16 {
            seen[idx as usize] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "Not all positions in 8x16 zig-zag order"
        );
    }

    #[test]
    fn test_num_nonzero_8x8_except_dc() {
        let mut block = [0i32; 64];
        let mut nzeros_pos = 0u8;

        // All zeros
        let nz = num_nonzero_8x8_except_dc(&block, &mut nzeros_pos);
        assert_eq!(nz, 0);
        assert_eq!(nzeros_pos, 0);

        // DC only (should not count)
        block[0] = 100;
        let nz = num_nonzero_8x8_except_dc(&block, &mut nzeros_pos);
        assert_eq!(nz, 0);
        assert_eq!(nzeros_pos, 0);

        // Some AC coefficients
        block[1] = 5;
        block[8] = -3;
        block[63] = 1;
        let nz = num_nonzero_8x8_except_dc(&block, &mut nzeros_pos);
        assert_eq!(nz, 3);
        assert_eq!(nzeros_pos, 3);
    }

    #[test]
    fn test_predict_from_top_and_left() {
        // No top, at x=0: use default
        assert_eq!(predict_from_top_and_left(None, &[0, 1, 2], 0, 32), 32);

        // No top, at x>0: use left
        let row = [5, 10, 15];
        assert_eq!(predict_from_top_and_left(None, &row, 1, 32), 5);
        assert_eq!(predict_from_top_and_left(None, &row, 2, 32), 10);

        // Has top, at x=0: use top
        let top = [20, 30, 40];
        assert_eq!(predict_from_top_and_left(Some(&top), &row, 0, 32), 20);

        // Has top, at x>0: average with rounding
        // (30 + 5 + 1) / 2 = 18
        assert_eq!(predict_from_top_and_left(Some(&top), &row, 1, 32), 18);
        // (40 + 10 + 1) / 2 = 25
        assert_eq!(predict_from_top_and_left(Some(&top), &row, 2, 32), 25);
    }

    #[test]
    fn test_ac_strategy_info() {
        use crate::tiny::ac_strategy::{
            RAW_STRATEGY_DCT4X8, RAW_STRATEGY_DCT8, RAW_STRATEGY_DCT8X4, RAW_STRATEGY_DCT8X16,
            RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16, RAW_STRATEGY_DCT32X32,
        };

        // DCT8: raw=0, bitstream=0, 1x1 blocks
        let (cx, cy, cb, log2cb, code) = ac_strategy_info(RAW_STRATEGY_DCT8);
        assert_eq!((cx, cy, cb, log2cb, code), (1, 1, 1, 0, 0));

        // DCT16X8: raw=1, bitstream=6, 1x2 blocks
        let (cx, cy, cb, log2cb, code) = ac_strategy_info(RAW_STRATEGY_DCT16X8);
        assert_eq!((cx, cy, cb, log2cb, code), (1, 2, 2, 1, 6));

        // DCT8X16: raw=2, bitstream=7, 2x1 blocks
        let (cx, cy, cb, log2cb, code) = ac_strategy_info(RAW_STRATEGY_DCT8X16);
        assert_eq!((cx, cy, cb, log2cb, code), (2, 1, 2, 1, 7));

        // DCT16X16: raw=3, bitstream=4, 2x2 blocks
        let (cx, cy, cb, log2cb, code) = ac_strategy_info(RAW_STRATEGY_DCT16X16);
        assert_eq!((cx, cy, cb, log2cb, code), (2, 2, 4, 2, 4));

        // DCT32X32: raw=4, bitstream=5, 4x4 blocks
        let (cx, cy, cb, log2cb, code) = ac_strategy_info(RAW_STRATEGY_DCT32X32);
        assert_eq!((cx, cy, cb, log2cb, code), (4, 4, 16, 4, 5));

        // DCT4X8: raw=5, bitstream=12, 1x1 blocks (64 coeffs)
        let (cx, cy, cb, log2cb, code) = ac_strategy_info(RAW_STRATEGY_DCT4X8);
        assert_eq!((cx, cy, cb, log2cb, code), (1, 1, 1, 0, 12));

        // DCT8X4: raw=6, bitstream=13, 1x1 blocks (64 coeffs)
        let (cx, cy, cb, log2cb, code) = ac_strategy_info(RAW_STRATEGY_DCT8X4);
        assert_eq!((cx, cy, cb, log2cb, code), (1, 1, 1, 0, 13));
    }

    #[test]
    fn test_coeff_order_32x32_llf_first() {
        let order = &*COEFF_ORDER_32X32;
        assert_eq!(order.len(), 1024);

        // Build set of LLF positions (4x4 in 32x32 grid, stride 32)
        let mut llf_positions = std::collections::HashSet::new();
        for lx in 0..4 {
            for ly in 0..4 {
                llf_positions.insert((lx * 32 + ly) as u32);
            }
        }
        assert_eq!(llf_positions.len(), 16);

        // First 16 entries must all be LLF positions
        for (k, &pos) in order.iter().enumerate().take(16) {
            assert!(
                llf_positions.contains(&pos),
                "order[{}] = {} is not an LLF position",
                k,
                pos
            );
        }

        // Entries 16..1024 must NOT be LLF positions
        for (k, &pos) in order.iter().enumerate().skip(16) {
            assert!(
                !llf_positions.contains(&pos),
                "order[{}] = {} is an LLF position but should be AC",
                k,
                pos
            );
        }

        // All 1024 positions must be present exactly once
        let mut seen = [false; 1024];
        for &pos in order.iter() {
            assert!(!seen[pos as usize], "duplicate position {}", pos);
            seen[pos as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "not all positions covered");
    }

    #[test]
    fn test_coeff_order_16x16_llf_first() {
        // Verify existing 16x16 order also has LLF first (regression test)
        let mut llf_positions = std::collections::HashSet::new();
        for lx in 0..2 {
            for ly in 0..2 {
                llf_positions.insert((lx * 16 + ly) as u32);
            }
        }

        for (k, &pos) in COEFF_ORDER_16X16.iter().enumerate().take(4) {
            assert!(
                llf_positions.contains(&pos),
                "16x16 order[{}] = {} is not LLF",
                k,
                pos
            );
        }
    }
}
