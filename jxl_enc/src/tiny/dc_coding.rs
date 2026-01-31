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
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
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

    // Write entire image (single DC group case)
    write_dc_tokens_region(quant_dc, 0, 0, width, height, dc_code, writer)
}

/// Encode DC coefficients for a specific region using gradient predictor.
///
/// For multi-group encoding, each DC group only writes its portion of DC tokens.
/// The region is specified in block coordinates.
///
/// # Arguments
/// * `quant_dc` - Quantized DC coefficients for each channel, shape [3][full_height][full_width]
/// * `start_bx` - Starting block x coordinate (inclusive)
/// * `start_by` - Starting block y coordinate (inclusive)
/// * `end_bx` - Ending block x coordinate (exclusive)
/// * `end_by` - Ending block y coordinate (exclusive)
/// * `dc_code` - DC entropy code to use for token writing
/// * `writer` - BitWriter to write encoded data
pub fn write_dc_tokens_region(
    quant_dc: &[Vec<Vec<i16>>; 3],
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
    dc_code: &EntropyCode,
    writer: &mut BitWriter,
) -> Result<()> {
    let region_width = end_bx - start_bx;
    let region_height = end_by - start_by;

    if region_width == 0 || region_height == 0 {
        return Ok(());
    }

    #[cfg(feature = "debug-tokens")]
    {
        debug_log!(
            "write_dc_tokens_region: blocks ({},{}) to ({},{}) = {}x{}",
            start_bx,
            start_by,
            end_bx,
            end_by,
            region_width,
            region_height
        );
    }

    // Counter for limiting debug output
    #[cfg(feature = "debug-tokens")]
    let mut dc_debug_count = 0usize;
    #[cfg(feature = "debug-tokens")]
    const DC_DEBUG_LIMIT: usize = 16;

    // Encode in channel order: Y (1), X (0), B (2)
    for &c in &[1, 0, 2] {
        let channel = &quant_dc[c];
        for y in start_by..end_by {
            for x in start_bx..end_bx {
                // Get neighbor values with edge handling
                // Note: we use actual coordinates, not region-local coordinates,
                // because we have access to the full DC array and neighbors may be
                // outside this DC group's region
                let left = if x > start_bx {
                    channel[y][x - 1] as i32
                } else if y > start_by {
                    channel[y - 1][x] as i32
                } else {
                    0
                };

                let top = if y > start_by {
                    channel[y - 1][x] as i32
                } else {
                    left
                };

                let topleft = if x > start_bx && y > start_by {
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
                #[cfg(feature = "debug-tokens")]
                {
                    let before = writer.bits_written();
                    if dc_debug_count < DC_DEBUG_LIMIT {
                        debug_log!(
                            "  DC[c={},y={},x={}]: actual={}, guess={}, residual={}, ctx={}, token_val={}",
                            c,
                            y,
                            x,
                            actual,
                            guess,
                            residual,
                            ctx_id,
                            pack_signed(residual)
                        );
                    }
                    write_token(&token, dc_code, writer)?;
                    let after = writer.bits_written();
                    if dc_debug_count < DC_DEBUG_LIMIT {
                        debug_log!("    -> wrote {} bits", after - before);
                    }
                    dc_debug_count += 1;
                    if dc_debug_count == DC_DEBUG_LIMIT {
                        let total_tokens = region_width * region_height * 3;
                        debug_log!("  ... ({} more DC tokens)", total_tokens - DC_DEBUG_LIMIT);
                    }
                }
                #[cfg(not(feature = "debug-tokens"))]
                write_token(&token, dc_code, writer)?;
            }
        }
    }

    Ok(())
}

/// Collect DC tokens for a specific region (without writing).
///
/// Same logic as `write_dc_tokens_region()` but returns a `Vec<Token>` instead
/// of writing to a bitstream. Used by the two-pass encoding mode.
pub fn collect_dc_tokens_region(
    quant_dc: &[Vec<Vec<i16>>; 3],
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
) -> Vec<Token> {
    let region_width = end_bx - start_bx;
    let region_height = end_by - start_by;

    if region_width == 0 || region_height == 0 {
        return Vec::new();
    }

    let mut tokens = Vec::with_capacity(region_width * region_height * 3);

    for &c in &[1, 0, 2] {
        let channel = &quant_dc[c];
        for y in start_by..end_by {
            for x in start_bx..end_bx {
                let left = if x > start_bx {
                    channel[y][x - 1] as i32
                } else if y > start_by {
                    channel[y - 1][x] as i32
                } else {
                    0
                };
                let top = if y > start_by {
                    channel[y - 1][x] as i32
                } else {
                    left
                };
                let topleft = if x > start_bx && y > start_by {
                    channel[y - 1][x - 1] as i32
                } else {
                    left
                };
                let guess = clamped_gradient(top, left, topleft);
                let actual = channel[y][x] as i32;
                let residual = actual - guess;
                let grad_prop = (GRAD_RANGE_MID + top as i64 + left as i64 - topleft as i64)
                    .clamp(GRAD_RANGE_MIN, GRAD_RANGE_MAX) as usize;
                let ctx_id = GRADIENT_CONTEXT_LUT[grad_prop] as u32;
                tokens.push(Token::new(ctx_id, pack_signed(residual)));
            }
        }
    }

    tokens
}

/// Collect AC metadata tokens for a specific region (without writing).
///
/// Same logic as `write_ac_metadata_tokens_region()` but returns a `Vec<Token>`.
pub fn collect_ac_metadata_tokens_region(
    region_xsize_blocks: usize,
    region_ysize_blocks: usize,
    quant_field: &[u8],
    full_xsize_blocks: usize,
    start_bx: usize,
    start_by: usize,
) -> Vec<Token> {
    let xsize_pixels = region_xsize_blocks * BLOCK_DIM;
    let ysize_pixels = region_ysize_blocks * BLOCK_DIM;
    let cfl_xsize = div_ceil(xsize_pixels, COLOR_TILE_DIM);
    let cfl_ysize = div_ceil(ysize_pixels, COLOR_TILE_DIM);

    let nblocks = region_xsize_blocks * region_ysize_blocks;
    // CFL (2 * cfl tiles) + ACS (nblocks) + QF (nblocks) + EPF (nblocks)
    let capacity = 2 * cfl_xsize * cfl_ysize + 3 * nblocks;
    let mut tokens = Vec::with_capacity(capacity);

    // YtoX and YtoB tokens (CFL = 0, so all residuals are 0)
    for c in 0..2 {
        let ctx_id = (2 - c) as u32;
        for _y in 0..cfl_ysize {
            for _x in 0..cfl_xsize {
                tokens.push(Token::new(ctx_id, pack_signed(0)));
            }
        }
    }

    // AC strategy tokens (all DCT8 = 0)
    let mut left_acs = 0i32;
    for _y in 0..region_ysize_blocks {
        for _x in 0..region_xsize_blocks {
            let cur = 0i32;
            let ctx_id = if left_acs > 11 {
                7
            } else if left_acs > 5 {
                8
            } else if left_acs > 3 {
                9
            } else {
                10
            };
            tokens.push(Token::new(ctx_id, pack_signed(cur)));
            left_acs = cur;
        }
    }

    // Quant field tokens
    let mut left_qf = 0i32;
    for y in 0..region_ysize_blocks {
        for x in 0..region_xsize_blocks {
            let abs_by = start_by + y;
            let abs_bx = start_bx + x;
            let block_idx = abs_by * full_xsize_blocks + abs_bx;
            let cur = (quant_field[block_idx] as i32) - 1;
            let residual = cur - left_qf;
            let ctx_id = if left_qf > 11 {
                3
            } else if left_qf > 5 {
                4
            } else if left_qf > 3 {
                5
            } else {
                6
            };
            tokens.push(Token::new(ctx_id, pack_signed(residual)));
            left_qf = cur;
        }
    }

    // EPF tokens
    for _ in 0..nblocks {
        tokens.push(Token::new(0, pack_signed(4)));
    }

    tokens
}

/// Color tile dimension (64 pixels) for CFL maps.
const COLOR_TILE_DIM: usize = 64;

/// Block dimension (8 pixels).
const BLOCK_DIM: usize = 8;

/// Ceiling division helper.
#[inline]
const fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Write AC metadata tokens (YtoX, YtoB, AC strategy, quant field, EPF) using gradient predictor.
///
/// AC metadata is encoded in the DC group section using the DC entropy code.
///
/// Context assignments:
/// - YtoX: context 2
/// - YtoB: context 1
/// - AC strategy: contexts 10, 9, 8, 7 based on left value
/// - Quant field: contexts 6, 5, 4, 3 based on left value
/// - EPF: context 0
///
/// # Arguments
/// * `xsize_blocks` - Number of 8x8 blocks in x direction (for the region)
/// * `ysize_blocks` - Number of 8x8 blocks in y direction (for the region)
/// * `quant_field` - Per-block raw quantization values (1-255), indexed as `[by * full_xsize_blocks + bx]`
/// * `full_xsize_blocks` - Full image width in blocks (for quant_field indexing)
/// * `dc_code` - DC entropy code to use for token writing
/// * `writer` - BitWriter to write encoded data
pub fn write_ac_metadata_tokens(
    xsize_blocks: usize,
    ysize_blocks: usize,
    quant_field: &[u8],
    full_xsize_blocks: usize,
    dc_code: &EntropyCode,
    writer: &mut BitWriter,
) -> Result<()> {
    // For single DC group, the region is the entire image (start at block 0,0)
    write_ac_metadata_tokens_region(
        xsize_blocks,
        ysize_blocks,
        quant_field,
        full_xsize_blocks,
        0,
        0,
        dc_code,
        writer,
    )
}

/// Write AC metadata tokens for a specific region.
///
/// For multi-group encoding, each DC group writes metadata only for its blocks.
/// The region dimensions are in blocks (not pixels).
///
/// # Arguments
/// * `region_xsize_blocks` - Number of blocks in x direction for this region
/// * `region_ysize_blocks` - Number of blocks in y direction for this region
/// * `quant_field` - Per-block raw quantization values (1-255), indexed as `[by * full_xsize_blocks + bx]`
/// * `full_xsize_blocks` - Full image width in blocks (for quant_field indexing)
/// * `start_bx` - Starting block x coordinate of this region
/// * `start_by` - Starting block y coordinate of this region
/// * `dc_code` - DC entropy code
/// * `writer` - BitWriter
pub fn write_ac_metadata_tokens_region(
    region_xsize_blocks: usize,
    region_ysize_blocks: usize,
    quant_field: &[u8],
    full_xsize_blocks: usize,
    start_bx: usize,
    start_by: usize,
    dc_code: &EntropyCode,
    writer: &mut BitWriter,
) -> Result<()> {
    #[cfg(feature = "debug-tokens")]
    let start_bits = writer.bits_written();
    // CFL maps use 64-pixel tiles, not 8-pixel blocks
    let xsize_pixels = region_xsize_blocks * BLOCK_DIM;
    let ysize_pixels = region_ysize_blocks * BLOCK_DIM;
    let cfl_xsize = div_ceil(xsize_pixels, COLOR_TILE_DIM);
    let cfl_ysize = div_ceil(ysize_pixels, COLOR_TILE_DIM);

    #[cfg(feature = "debug-tokens")]
    let after_start = writer.bits_written();

    // YtoX and YtoB tokens
    // For simple encoder, all CFL values are 0, so all residuals are 0
    for c in 0..2 {
        // YtoX uses context 2, YtoB uses context 1
        let ctx_id = (2 - c) as u32;
        for y in 0..cfl_ysize {
            for x in 0..cfl_xsize {
                // Neighbors for gradient prediction
                let left = if x > 0 {
                    0i64
                } else if y > 0 {
                    0i64
                } else {
                    0i64
                };
                let top = if y > 0 { 0i64 } else { left };
                let topleft = if x > 0 && y > 0 { 0i64 } else { left };
                let guess = clamped_gradient(top as i32, left as i32, topleft as i32);
                let actual = 0i32; // All CFL values are 0
                let residual = actual - guess;
                let token = Token::new(ctx_id, pack_signed(residual));
                write_token(&token, dc_code, writer)?;
            }
        }
    }

    #[cfg(feature = "debug-tokens")]
    let after_cfl = writer.bits_written();

    // AC strategy tokens
    // All DCT8 (code 0), so all residuals are 0
    let mut left_acs = 0i32;
    for y in 0..region_ysize_blocks {
        for x in 0..region_xsize_blocks {
            // For DCT8, every block is a first block
            let cur = 0i32; // DCT8 strategy code
            // Context based on left value
            let ctx_id = if left_acs > 11 {
                7
            } else if left_acs > 5 {
                8
            } else if left_acs > 3 {
                9
            } else {
                10
            };
            let token = Token::new(ctx_id, pack_signed(cur));
            write_token(&token, dc_code, writer)?;
            left_acs = cur;
        }
    }

    #[cfg(feature = "debug-tokens")]
    let after_acs = writer.bits_written();

    // Quant field tokens - per-block values from adaptive quantization
    // cur = quant_field[by][bx] - 1 (offset by 1 in the encoding)
    // Initial left is ac_strategy[0][0].StrategyCode() = 0 for DCT8
    let mut left_qf = 0i32;
    for y in 0..region_ysize_blocks {
        for x in 0..region_xsize_blocks {
            // Look up per-block quant value from the full image quant field
            let abs_by = start_by + y;
            let abs_bx = start_bx + x;
            let block_idx = abs_by * full_xsize_blocks + abs_bx;
            let cur = (quant_field[block_idx] as i32) - 1;
            let residual = cur - left_qf;
            // Context based on left value
            let ctx_id = if left_qf > 11 {
                3
            } else if left_qf > 5 {
                4
            } else if left_qf > 3 {
                5
            } else {
                6
            };
            let token = Token::new(ctx_id, pack_signed(residual));
            write_token(&token, dc_code, writer)?;
            left_qf = cur;
        }
    }

    #[cfg(feature = "debug-tokens")]
    let after_qf = writer.bits_written();

    // EPF (Edge-Preserving Filter) tokens
    // Write one EPF token per block with value PackSigned(4) = 8
    // Context 0 is used for EPF tokens
    let nblocks = region_xsize_blocks * region_ysize_blocks;
    for _ in 0..nblocks {
        let token = Token::new(0, pack_signed(4)); // EPF default value 4
        write_token(&token, dc_code, writer)?;
    }

    #[cfg(feature = "debug-tokens")]
    {
        let after_epf = writer.bits_written();
        debug_log!("  ac_metadata breakdown:");
        debug_log!(
            "    cfl (YtoX+YtoB): {} bits ({} tokens)",
            after_cfl - after_start,
            cfl_xsize * cfl_ysize * 2
        );
        debug_log!(
            "    ac_strategy: {} bits ({} tokens)",
            after_acs - after_cfl,
            region_xsize_blocks * region_ysize_blocks
        );
        debug_log!(
            "    quant_field: {} bits ({} tokens)",
            after_qf - after_acs,
            region_xsize_blocks * region_ysize_blocks
        );
        debug_log!(
            "    epf: {} bits ({} tokens)",
            after_epf - after_qf,
            nblocks
        );
        debug_log!("    total: {} bits", after_epf - start_bits);
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
