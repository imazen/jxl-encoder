// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! DC coefficient coding with gradient predictor.
//!
//! DC coefficients are coded using a ClampedGradient predictor that uses
//! the left, top, and top-left neighbors to predict each value. The residual
//! (actual - prediction) is then entropy coded with context based on the
//! gradient property.

use super::ac_strategy::AcStrategyMap;
use super::chroma_from_luma::CflMap;
use super::common::pack_signed;
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::entropy_coding::encode::{EntropyCode, write_token};
use crate::entropy_coding::token::Token;
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

/// Number of DC contexts (gradient predictor path, used by JPEG recompression).
#[allow(dead_code)]
pub const NUM_DC_CONTEXTS: usize = 45;

/// Number of DC contexts on the JPEG-transcode tree (lever A, 2026-05-28).
///
/// Mirrors libjxl's `kJpegTranscodeACMeta` + gradient-fixed-DC pair:
/// 1 AC-metadata context (single `Leaf(Zero)`) + 34 DC gradient contexts.
#[cfg(feature = "jpeg-reencoding")]
pub const NUM_DC_CONTEXTS_JPEG_TRANSCODE: usize = 35;

/// Number of AC metadata contexts (contexts 0-10).
/// These are used for: EPF (0), YtoB (1), YtoX (2), quant field (3-6), AC strategy (7-10).
/// Used by dc_tree_learn.rs for tree structure.
#[allow(dead_code)]
pub const NUM_AC_METADATA_CONTEXTS: usize = 11;

/// First context ID for DC values (after AC metadata contexts).
/// When using the default GRADIENT_CONTEXT_LUT, DC values use contexts 11-44.
/// Used by dc_tree_learn.rs for context offset calculation.
#[allow(dead_code)]
pub const DC_CONTEXT_OFFSET: usize = NUM_AC_METADATA_CONTEXTS;

/// Single AC-metadata context ID on the JPEG-transcode tree (lever A).
///
/// All AC-metadata tokens (CFL, ACS, QF, EPF) collapse to this one context
/// when the JPEG-transcode tree is in use (see [`write_jpeg_transcode_context_tree`]).
/// Predictor is `Zero` (raw values, no gradient/left residuals).
///
/// [`write_jpeg_transcode_context_tree`]: super::context_tree::write_jpeg_transcode_context_tree
#[cfg(feature = "jpeg-reencoding")]
pub const JPEG_TRANSCODE_AC_META_CONTEXT: u32 = 0;

/// First DC context ID on the JPEG-transcode tree (lever A).
///
/// DC values use contexts `JPEG_TRANSCODE_DC_CONTEXT_OFFSET .. +34` =
/// `1..=34`, because the AC-metadata subtree shrinks from 11 leaves to 1
/// and BFS shifts the DC leaf IDs down by 10. Computed as
/// `GRADIENT_CONTEXT_LUT[grad] - 10` per the bit-identical mapping verified
/// at generation time.
pub const JPEG_TRANSCODE_DC_CONTEXT_OFFSET: u32 = 1;

/// Encode DC coefficients using gradient predictor and entropy coding.
///
/// DC coefficients are organized as [channel][y][x] where channel order is:
/// Y (1), X (0), B (2) for encoding.
///
/// # Arguments
/// * `quant_dc` - Quantized DC coefficients for each channel, shape [3][height][width]
/// * `dc_code` - DC entropy code to use for token writing
/// * `writer` - BitWriter to write encoded data
#[allow(dead_code)]
pub fn write_dc_tokens(
    quant_dc: &[Vec<Vec<i32>>; 3],
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
    quant_dc: &[Vec<Vec<i32>>; 3],
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
                    channel[y][x - 1]
                } else if y > start_by {
                    channel[y - 1][x]
                } else {
                    0
                };

                let top = if y > start_by {
                    channel[y - 1][x]
                } else {
                    left
                };

                let topleft = if x > start_bx && y > start_by {
                    channel[y - 1][x - 1]
                } else {
                    left
                };

                // Compute prediction and residual
                let guess = clamped_gradient(top, left, topleft);
                let actual = channel[y][x];
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
                    write_token(&token, dc_code, None, writer)?;
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
                write_token(&token, dc_code, None, writer)?;
            }
        }
    }

    Ok(())
}

/// Collect DC tokens using Weighted Predictor and kWPFixedDC tree.
///
/// This replaces `collect_dc_tokens_region()` when the kWPFixedDC tree is used.
/// Each DC channel gets its own WeightedPredictorState (reset per channel).
///
/// # Arguments
/// * `quant_dc` - Quantized DC coefficients [channel][y][x]
/// * `wp_tree` - kWPFixedDC tree for context assignment
/// * `start_bx` / `start_by` - Starting block coordinates (inclusive)
/// * `end_bx` / `end_by` - Ending block coordinates (exclusive)
pub fn collect_dc_tokens_wp(
    quant_dc: &[Vec<Vec<i32>>; 3],
    wp_tree: &super::dc_tree_learn::DcTree,
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
) -> Vec<Token> {
    collect_dc_tokens_wp_with_budget(quant_dc, wp_tree, start_bx, start_by, end_bx, end_by, None)
        .expect("budget-less collect_dc_tokens_wp must not return AllocationLimit")
}

/// `collect_dc_tokens_wp` with explicit allocation budget.
///
/// The WP state's per-channel scratch is `(region_width + 2) * 2` errors
/// plus four sub-predictor errors of the same length — region-driven,
/// charged against the cap. `budget = None` is zero-overhead.
pub(crate) fn collect_dc_tokens_wp_with_budget(
    quant_dc: &[Vec<Vec<i32>>; 3],
    wp_tree: &super::dc_tree_learn::DcTree,
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<Vec<Token>> {
    use crate::modular::predictor::{Neighbors, WeightedPredictorState};

    let region_width = end_bx - start_bx;
    let region_height = end_by - start_by;

    if region_width == 0 || region_height == 0 {
        return Ok(Vec::new());
    }

    // Honor the budget's runtime fallible-alloc policy for this dimension-driven
    // DC-token buffer (`Limits::fallible_alloc`); byte-identical when infallible.
    let mut tokens = crate::budget::vec_with_capacity_fallible(
        budget.is_some_and(|b| b.is_fallible()),
        region_width * region_height * 3,
    )?;

    // Encode in channel order: Y (1), X (0), B (2)
    // Each channel gets a FRESH WP state (matches libjxl per-channel processing)
    for &c in &[1, 0, 2] {
        let channel = &quant_dc[c];
        let mut wp_state = WeightedPredictorState::with_defaults_and_budget(region_width, budget)?;

        for y in start_by..end_by {
            for x in start_bx..end_bx {
                let actual = channel[y][x];

                // Gather neighbors matching modular edge handling
                let w = if x > start_bx {
                    channel[y][x - 1]
                } else if y > start_by {
                    channel[y - 1][x]
                } else {
                    0
                };

                let n = if y > start_by { channel[y - 1][x] } else { w };

                let nw = if x > start_bx && y > start_by {
                    channel[y - 1][x - 1]
                } else {
                    w
                };

                let ne = if x + 1 < end_bx && y > start_by {
                    channel[y - 1][x + 1]
                } else {
                    n
                };

                let ww = if x > start_bx + 1 {
                    channel[y][x - 2]
                } else {
                    w
                };

                let nn = if y > start_by + 1 {
                    channel[y - 2][x]
                } else {
                    n
                };

                let nee = if x + 2 < end_bx && y > start_by {
                    channel[y - 1][x + 2]
                } else {
                    ne
                };

                let neighbors = Neighbors {
                    n,
                    w,
                    nw,
                    ne,
                    nn,
                    ww,
                    nee,
                };

                // Use region-local coordinates for WP state
                let local_x = x - start_bx;
                let local_y = y - start_by;

                // Get WP prediction and max_error property
                let (prediction, wp_max_error) =
                    wp_state.predict_and_property(local_x, local_y, region_width, &neighbors);

                let residual = actual - prediction as i32;

                // Get context from kWPFixedDC tree using wp_max_error
                let ctx_id = super::dc_tree_learn::get_wp_dc_context(wp_tree, wp_max_error);

                tokens.push(Token::new(ctx_id, pack_signed(residual)));

                // Update WP error state with actual value
                wp_state.update_errors(actual, local_x, local_y, region_width);
            }
        }
    }

    Ok(tokens)
}

/// Collect DC tokens using the Weighted Predictor + kWPFixedDC tree, with
/// per-channel chroma-subsampling support (EX-J31, JPEG transcode path).
///
/// Mirrors [`collect_dc_tokens_wp`] (same WP prediction + kWPFixedDC context
/// assignment) but applies `channel_shifts` like [`collect_dc_tokens_region`]
/// so subsampled chroma DC planes (4:2:0 / 4:2:2 / 4:4:0) iterate at their
/// native resolution. Each channel gets a fresh `WeightedPredictorState`
/// sized to that channel's region width — matching how the decoder runs WP
/// per modular channel at its declared dimension.
///
/// libjxl encodes the JPEG-transcode DC stream with `kWPFixedDC`
/// (`Predictor::Weighted`, tree split on `wp_max_error` / property 15) at
/// `speed_tier >= kSquirrel` (cjxl effort 7); this is the per-channel,
/// subsampling-aware analog we wire into the JPEG path.
#[cfg(feature = "jpeg-reencoding")]
#[allow(clippy::too_many_arguments)]
pub fn collect_dc_tokens_wp_region_jpeg(
    quant_dc: &[Vec<Vec<i32>>; 3],
    wp_tree: &super::dc_tree_learn::DcTree,
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
    channel_shifts: &[(usize, usize); 3],
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<Vec<Token>> {
    use crate::modular::predictor::{Neighbors, WeightedPredictorState};

    let region_width = end_bx.saturating_sub(start_bx);
    let region_height = end_by.saturating_sub(start_by);
    if region_width == 0 || region_height == 0 {
        return Ok(Vec::new());
    }

    // Honor the budget's runtime fallible-alloc policy for this dimension-driven
    // DC-token buffer (`Limits::fallible_alloc`); byte-identical when infallible.
    let mut tokens = crate::budget::vec_with_capacity_fallible(
        budget.is_some_and(|b| b.is_fallible()),
        region_width * region_height * 3,
    )?;

    // Channel order Y(1), X(0), B(2) — matches collect_dc_tokens_region and
    // collect_dc_tokens_wp.
    for &c in &[1usize, 0, 2] {
        let (hs, vs) = channel_shifts[c];
        let channel = &quant_dc[c];
        let ch_start_bx = start_bx >> hs;
        let ch_start_by = start_by >> vs;
        let ch_end_bx = (end_bx >> hs).min(channel.first().map_or(0, |r| r.len()));
        let ch_end_by = (end_by >> vs).min(channel.len());
        if ch_end_bx <= ch_start_bx || ch_end_by <= ch_start_by {
            continue;
        }
        let ch_region_width = ch_end_bx - ch_start_bx;

        // budget-less: this path is only used by the JPEG transcode encoder
        // which encodes trusted, already-decoded coefficient data.
        let mut wp_state = WeightedPredictorState::with_defaults(ch_region_width);

        for y in ch_start_by..ch_end_by {
            for x in ch_start_bx..ch_end_bx {
                let actual = channel[y][x] as i32;

                let w = if x > ch_start_bx {
                    channel[y][x - 1] as i32
                } else if y > ch_start_by {
                    channel[y - 1][x] as i32
                } else {
                    0
                };
                let n = if y > ch_start_by {
                    channel[y - 1][x] as i32
                } else {
                    w
                };
                let nw = if x > ch_start_bx && y > ch_start_by {
                    channel[y - 1][x - 1] as i32
                } else {
                    w
                };
                let ne = if x + 1 < ch_end_bx && y > ch_start_by {
                    channel[y - 1][x + 1] as i32
                } else {
                    n
                };
                let ww = if x > ch_start_bx + 1 {
                    channel[y][x - 2] as i32
                } else {
                    w
                };
                let nn = if y > ch_start_by + 1 {
                    channel[y - 2][x] as i32
                } else {
                    n
                };
                let nee = if x + 2 < ch_end_bx && y > ch_start_by {
                    channel[y - 1][x + 2] as i32
                } else {
                    ne
                };

                let neighbors = Neighbors {
                    n,
                    w,
                    nw,
                    ne,
                    nn,
                    ww,
                    nee,
                };

                let local_x = x - ch_start_bx;
                let local_y = y - ch_start_by;

                let (prediction, wp_max_error) =
                    wp_state.predict_and_property(local_x, local_y, ch_region_width, &neighbors);
                let residual = actual - prediction as i32;
                let ctx_id = super::dc_tree_learn::get_wp_dc_context(wp_tree, wp_max_error);
                tokens.push(Token::new(ctx_id, pack_signed(residual)));
                wp_state.update_errors(actual, local_x, local_y, ch_region_width);
            }
        }
    }

    Ok(tokens)
}

/// Collect DC tokens for a specific region (gradient predictor path).
///
/// Same logic as `write_dc_tokens_region()` but returns a `Vec<Token>` instead
/// of writing to a bitstream. Used by JPEG recompression encoder.
///
/// The `channel_shifts` parameter specifies per-channel (h_shift, v_shift) for
/// chroma subsampling. For 4:4:4, pass `&[(0,0); 3]`. For 4:2:0, chroma
/// channels have shift (1,1), causing the DC iteration to use halved bounds.
#[allow(dead_code)]
pub fn collect_dc_tokens_region(
    quant_dc: &[Vec<Vec<i32>>; 3],
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
    channel_shifts: &[(usize, usize); 3],
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<Vec<Token>> {
    let region_width = end_bx - start_bx;
    let region_height = end_by - start_by;

    if region_width == 0 || region_height == 0 {
        return Ok(Vec::new());
    }

    // Honor the budget's runtime fallible-alloc policy for this dimension-driven
    // DC-token buffer (`Limits::fallible_alloc`); byte-identical when infallible.
    let mut tokens = crate::budget::vec_with_capacity_fallible(
        budget.is_some_and(|b| b.is_fallible()),
        region_width * region_height * 3,
    )?;

    for &c in &[1, 0, 2] {
        let (hs, vs) = channel_shifts[c];
        let ch_start_bx = start_bx >> hs;
        let ch_start_by = start_by >> vs;
        let ch_end_bx = (end_bx >> hs).min(quant_dc[c].first().map_or(0, |r| r.len()));
        let ch_end_by = (end_by >> vs).min(quant_dc[c].len());

        let channel = &quant_dc[c];
        for y in ch_start_by..ch_end_by {
            for x in ch_start_bx..ch_end_bx {
                let left = if x > ch_start_bx {
                    channel[y][x - 1]
                } else if y > ch_start_by {
                    channel[y - 1][x]
                } else {
                    0
                };
                let top = if y > ch_start_by {
                    channel[y - 1][x]
                } else {
                    left
                };
                let topleft = if x > ch_start_bx && y > ch_start_by {
                    channel[y - 1][x - 1]
                } else {
                    left
                };
                let guess = clamped_gradient(top, left, topleft);
                let actual = channel[y][x];
                let residual = actual - guess;
                let grad_prop = (GRAD_RANGE_MID + top as i64 + left as i64 - topleft as i64)
                    .clamp(GRAD_RANGE_MIN, GRAD_RANGE_MAX) as usize;
                let ctx_id = GRADIENT_CONTEXT_LUT[grad_prop] as u32;
                tokens.push(Token::new(ctx_id, pack_signed(residual)));
            }
        }
    }

    Ok(tokens)
}

/// Collect DC tokens for a region using the JPEG-transcode context-tree's
/// shifted context IDs (lever A, 2026-05-28).
///
/// Identical to [`collect_dc_tokens_region`] except every emitted context is
/// shifted down by 10 to match the BFS leaf order of
/// [`super::context_tree::JPEG_TRANSCODE_CONTEXT_TREE_TOKENS`] (DC leaves
/// occupy `JPEG_TRANSCODE_DC_CONTEXT_OFFSET..=34` instead of the old
/// `11..=44`).
///
/// The decoder reads exactly the same residuals — only the byte cost of
/// the context-map and per-context histogram serialisation moves (smaller
/// context space, ~one fewer cluster).
#[allow(dead_code)]
pub fn collect_dc_tokens_region_jpeg_transcode(
    quant_dc: &[Vec<Vec<i32>>; 3],
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
    channel_shifts: &[(usize, usize); 3],
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<Vec<Token>> {
    let region_width = end_bx - start_bx;
    let region_height = end_by - start_by;

    if region_width == 0 || region_height == 0 {
        return Ok(Vec::new());
    }

    // Honor the budget's runtime fallible-alloc policy for this dimension-driven
    // DC-token buffer (`Limits::fallible_alloc`); byte-identical when infallible.
    let mut tokens = crate::budget::vec_with_capacity_fallible(
        budget.is_some_and(|b| b.is_fallible()),
        region_width * region_height * 3,
    )?;

    for &c in &[1, 0, 2] {
        let (hs, vs) = channel_shifts[c];
        let ch_start_bx = start_bx >> hs;
        let ch_start_by = start_by >> vs;
        let ch_end_bx = (end_bx >> hs).min(quant_dc[c].first().map_or(0, |r| r.len()));
        let ch_end_by = (end_by >> vs).min(quant_dc[c].len());

        let channel = &quant_dc[c];
        for y in ch_start_by..ch_end_by {
            for x in ch_start_bx..ch_end_bx {
                let left = if x > ch_start_bx {
                    channel[y][x - 1]
                } else if y > ch_start_by {
                    channel[y - 1][x]
                } else {
                    0
                };
                let top = if y > ch_start_by {
                    channel[y - 1][x]
                } else {
                    left
                };
                let topleft = if x > ch_start_bx && y > ch_start_by {
                    channel[y - 1][x - 1]
                } else {
                    left
                };
                let guess = clamped_gradient(top, left, topleft);
                let actual = channel[y][x];
                let residual = actual - guess;
                let grad_prop = (GRAD_RANGE_MID + top as i64 + left as i64 - topleft as i64)
                    .clamp(GRAD_RANGE_MIN, GRAD_RANGE_MAX) as usize;
                // Shift down by 10: old GRADIENT_CONTEXT_LUT values [11..=44]
                // become new DC contexts [1..=34] on the JPEG-transcode tree.
                let ctx_id = (GRADIENT_CONTEXT_LUT[grad_prop] as u32)
                    .saturating_sub(NUM_AC_METADATA_CONTEXTS as u32)
                    + JPEG_TRANSCODE_DC_CONTEXT_OFFSET;
                tokens.push(Token::new(ctx_id, pack_signed(residual)));
            }
        }
    }

    Ok(tokens)
}

/// Collect AC metadata tokens for a specific region (without writing).
///
/// Same logic as `write_ac_metadata_tokens_region()` but returns a `Vec<Token>`.
#[allow(clippy::too_many_arguments)]
pub fn collect_ac_metadata_tokens_region(
    region_xsize_blocks: usize,
    region_ysize_blocks: usize,
    quant_field: &[u8],
    full_xsize_blocks: usize,
    start_bx: usize,
    start_by: usize,
    cfl_map: &CflMap,
    ac_strategy: &AcStrategyMap,
    sharpness_map: Option<&[u8]>,
) -> Vec<Token> {
    let xsize_pixels = region_xsize_blocks * BLOCK_DIM;
    let ysize_pixels = region_ysize_blocks * BLOCK_DIM;
    let cfl_xsize = xsize_pixels.div_ceil(COLOR_TILE_DIM);
    let cfl_ysize = ysize_pixels.div_ceil(COLOR_TILE_DIM);

    let nblocks = region_xsize_blocks * region_ysize_blocks;
    // CFL (2 * cfl tiles) + ACS (nblocks) + QF (nblocks) + EPF (nblocks)
    let capacity = 2 * cfl_xsize * cfl_ysize + 3 * nblocks;
    let mut tokens = Vec::with_capacity(capacity);

    // Compute the global tile offset for this region
    let global_tile_x0 = start_bx / TILES_IN_BLOCKS;
    let global_tile_y0 = start_by / TILES_IN_BLOCKS;

    // YtoX and YtoB tokens with gradient prediction
    for c in 0..2 {
        let ctx_id = (2 - c) as u32;
        for y in 0..cfl_ysize {
            for x in 0..cfl_xsize {
                let global_tx = global_tile_x0 + x;
                let global_ty = global_tile_y0 + y;
                let actual = if c == 0 {
                    cfl_map.ytox_at(global_tx, global_ty) as i32
                } else {
                    cfl_map.ytob_at(global_tx, global_ty) as i32
                };
                // Gradient prediction from neighbors in the CfL map
                let left = if x > 0 {
                    if c == 0 {
                        cfl_map.ytox_at(global_tx - 1, global_ty) as i64
                    } else {
                        cfl_map.ytob_at(global_tx - 1, global_ty) as i64
                    }
                } else if y > 0 {
                    if c == 0 {
                        cfl_map.ytox_at(global_tx, global_ty - 1) as i64
                    } else {
                        cfl_map.ytob_at(global_tx, global_ty - 1) as i64
                    }
                } else {
                    0i64
                };
                let top = if y > 0 {
                    if c == 0 {
                        cfl_map.ytox_at(global_tx, global_ty - 1) as i64
                    } else {
                        cfl_map.ytob_at(global_tx, global_ty - 1) as i64
                    }
                } else {
                    left
                };
                let topleft = if x > 0 && y > 0 {
                    if c == 0 {
                        cfl_map.ytox_at(global_tx - 1, global_ty - 1) as i64
                    } else {
                        cfl_map.ytob_at(global_tx - 1, global_ty - 1) as i64
                    }
                } else {
                    left
                };
                let guess = clamped_gradient(top as i32, left as i32, topleft as i32);
                let residual = actual - guess;
                tokens.push(Token::new(ctx_id, pack_signed(residual)));
            }
        }
    }

    // AC strategy tokens — first blocks only
    let mut left_acs = 0i32;
    for y in 0..region_ysize_blocks {
        for x in 0..region_xsize_blocks {
            let abs_bx = start_bx + x;
            let abs_by = start_by + y;
            if !ac_strategy.is_first(abs_bx, abs_by) {
                continue;
            }
            let cur = ac_strategy.strategy_code(abs_bx, abs_by) as i32;
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

    // Quant field tokens — first blocks only
    let initial_acs_code = ac_strategy.strategy_code(start_bx, start_by) as i32;
    let mut left_qf = initial_acs_code;
    for y in 0..region_ysize_blocks {
        for x in 0..region_xsize_blocks {
            let abs_by = start_by + y;
            let abs_bx = start_bx + x;
            if !ac_strategy.is_first(abs_bx, abs_by) {
                continue;
            }
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

    // EPF tokens - per-block sharpness values
    for by_local in 0..region_ysize_blocks {
        for bx_local in 0..region_xsize_blocks {
            let abs_by = start_by + by_local;
            let abs_bx = start_bx + bx_local;
            let sharpness = if let Some(sm) = sharpness_map {
                sm[abs_by * full_xsize_blocks + abs_bx] as i32
            } else {
                4 // default EPF sharpness
            };
            tokens.push(Token::new(0, pack_signed(sharpness)));
        }
    }

    tokens
}

/// Color tile dimension (64 pixels) for CFL maps.
const COLOR_TILE_DIM: usize = 64;

/// Block dimension (8 pixels).
const BLOCK_DIM: usize = 8;

/// CFL tile dimension in blocks (64 / 8 = 8 blocks per tile).
const TILES_IN_BLOCKS: usize = COLOR_TILE_DIM / BLOCK_DIM;

/// Collect AC-metadata tokens for the JPEG-transcode tree (lever A,
/// 2026-05-28).
///
/// Mirrors libjxl's `kJpegTranscodeACMeta` semantics: every emitted token
/// uses `context = JPEG_TRANSCODE_AC_META_CONTEXT` (= 0) and a `Zero`
/// predictor, so the value field carries the *raw* AC-metadata datum
/// (CFL multiplier, ACS code, QF-1, EPF sharpness) packed via
/// `pack_signed`. This sheds the per-token gradient/left/zero predictor
/// machinery the multi-context [`collect_ac_metadata_tokens_region`]
/// applies. Token emission order — channels 0 (YtoX), 1 (YtoB), 2 (ACS),
/// 2 (QF), 3 (EPF) — matches the libjxl AC-metadata stream layout so the
/// decoder can route tokens via the property-1 (channel) / property-2 (y)
/// splits embedded in the tree.
///
/// Pair with [`super::context_tree::write_jpeg_transcode_context_tree`].
#[cfg(feature = "jpeg-reencoding")]
#[allow(clippy::too_many_arguments)]
pub fn collect_ac_metadata_tokens_region_jpeg_transcode(
    region_xsize_blocks: usize,
    region_ysize_blocks: usize,
    quant_field: &[u8],
    full_xsize_blocks: usize,
    start_bx: usize,
    start_by: usize,
    cfl_map: &CflMap,
    ac_strategy: &AcStrategyMap,
    sharpness_map: Option<&[u8]>,
) -> Vec<Token> {
    let xsize_pixels = region_xsize_blocks * BLOCK_DIM;
    let ysize_pixels = region_ysize_blocks * BLOCK_DIM;
    let cfl_xsize = xsize_pixels.div_ceil(COLOR_TILE_DIM);
    let cfl_ysize = ysize_pixels.div_ceil(COLOR_TILE_DIM);

    let nblocks = region_xsize_blocks * region_ysize_blocks;
    let capacity = 2 * cfl_xsize * cfl_ysize + 3 * nblocks;
    let mut tokens = Vec::with_capacity(capacity);
    let ctx = JPEG_TRANSCODE_AC_META_CONTEXT;

    let global_tile_x0 = start_bx / TILES_IN_BLOCKS;
    let global_tile_y0 = start_by / TILES_IN_BLOCKS;

    // CfL (YtoX = channel 0, YtoB = channel 1): emit raw multipliers.
    for c in 0..2 {
        for y in 0..cfl_ysize {
            for x in 0..cfl_xsize {
                let global_tx = global_tile_x0 + x;
                let global_ty = global_tile_y0 + y;
                let actual = if c == 0 {
                    cfl_map.ytox_at(global_tx, global_ty) as i32
                } else {
                    cfl_map.ytob_at(global_tx, global_ty) as i32
                };
                tokens.push(Token::new(ctx, pack_signed(actual)));
            }
        }
    }

    // AC strategy tokens — first blocks only, raw values.
    for y in 0..region_ysize_blocks {
        for x in 0..region_xsize_blocks {
            let abs_bx = start_bx + x;
            let abs_by = start_by + y;
            if !ac_strategy.is_first(abs_bx, abs_by) {
                continue;
            }
            let cur = ac_strategy.strategy_code(abs_bx, abs_by) as i32;
            tokens.push(Token::new(ctx, pack_signed(cur)));
        }
    }

    // Quant field tokens — first blocks only, raw `qf - 1` values
    // (decoder adds 1 back). On the JPEG transcode path `qf == 1`
    // everywhere → `qf - 1 == 0` → these compress to a single
    // peak symbol in context 0's histogram.
    for y in 0..region_ysize_blocks {
        for x in 0..region_xsize_blocks {
            let abs_by = start_by + y;
            let abs_bx = start_bx + x;
            if !ac_strategy.is_first(abs_bx, abs_by) {
                continue;
            }
            let block_idx = abs_by * full_xsize_blocks + abs_bx;
            let cur = (quant_field[block_idx] as i32) - 1;
            tokens.push(Token::new(ctx, pack_signed(cur)));
        }
    }

    // EPF tokens — per-block sharpness, raw values.
    //
    // EX-J25 (2026-05-28): libjxl `enc_frame.cc:790` explicitly sets
    // `FillImage(static_cast<uint8_t>(0), &shared.epf_sharpness)` for
    // JPEG transcode. Our default was 4. Measurement on 50-file paired
    // A/B: 0 bytes difference (both constants encode equally well by
    // ANS). Kept sharpness=0 for libjxl parity even though it's a
    // no-op on the byte count. Frame header `epf_iters=0` makes the
    // value functionally irrelevant at decode time.
    let default_sharpness = 0i32;
    for by_local in 0..region_ysize_blocks {
        for bx_local in 0..region_xsize_blocks {
            let abs_by = start_by + by_local;
            let abs_bx = start_bx + bx_local;
            let sharpness = if let Some(sm) = sharpness_map {
                sm[abs_by * full_xsize_blocks + abs_bx] as i32
            } else {
                default_sharpness
            };
            tokens.push(Token::new(ctx, pack_signed(sharpness)));
        }
    }

    tokens
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
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn write_ac_metadata_tokens(
    xsize_blocks: usize,
    ysize_blocks: usize,
    quant_field: &[u8],
    full_xsize_blocks: usize,
    cfl_map: &CflMap,
    ac_strategy: &AcStrategyMap,
    sharpness_map: Option<&[u8]>,
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
        cfl_map,
        ac_strategy,
        sharpness_map,
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
#[allow(clippy::too_many_arguments)]
pub fn write_ac_metadata_tokens_region(
    region_xsize_blocks: usize,
    region_ysize_blocks: usize,
    quant_field: &[u8],
    full_xsize_blocks: usize,
    start_bx: usize,
    start_by: usize,
    cfl_map: &CflMap,
    ac_strategy: &AcStrategyMap,
    sharpness_map: Option<&[u8]>,
    dc_code: &EntropyCode,
    writer: &mut BitWriter,
) -> Result<()> {
    #[cfg(feature = "debug-tokens")]
    let start_bits = writer.bits_written();
    // CFL maps use 64-pixel tiles, not 8-pixel blocks
    let xsize_pixels = region_xsize_blocks * BLOCK_DIM;
    let ysize_pixels = region_ysize_blocks * BLOCK_DIM;
    let cfl_xsize = xsize_pixels.div_ceil(COLOR_TILE_DIM);
    let cfl_ysize = ysize_pixels.div_ceil(COLOR_TILE_DIM);

    #[cfg(feature = "debug-tokens")]
    let after_start = writer.bits_written();

    // Compute the global tile offset for this region
    let global_tile_x0 = start_bx / TILES_IN_BLOCKS;
    let global_tile_y0 = start_by / TILES_IN_BLOCKS;

    // YtoX and YtoB tokens with gradient prediction from actual CfL map values
    for c in 0..2 {
        // YtoX uses context 2, YtoB uses context 1
        let ctx_id = (2 - c) as u32;
        for y in 0..cfl_ysize {
            for x in 0..cfl_xsize {
                let global_tx = global_tile_x0 + x;
                let global_ty = global_tile_y0 + y;
                let actual = if c == 0 {
                    cfl_map.ytox_at(global_tx, global_ty) as i32
                } else {
                    cfl_map.ytob_at(global_tx, global_ty) as i32
                };
                // Gradient prediction from neighbors in the CfL map
                let left = if x > 0 {
                    if c == 0 {
                        cfl_map.ytox_at(global_tx - 1, global_ty) as i64
                    } else {
                        cfl_map.ytob_at(global_tx - 1, global_ty) as i64
                    }
                } else if y > 0 {
                    if c == 0 {
                        cfl_map.ytox_at(global_tx, global_ty - 1) as i64
                    } else {
                        cfl_map.ytob_at(global_tx, global_ty - 1) as i64
                    }
                } else {
                    0i64
                };
                let top = if y > 0 {
                    if c == 0 {
                        cfl_map.ytox_at(global_tx, global_ty - 1) as i64
                    } else {
                        cfl_map.ytob_at(global_tx, global_ty - 1) as i64
                    }
                } else {
                    left
                };
                let topleft = if x > 0 && y > 0 {
                    if c == 0 {
                        cfl_map.ytox_at(global_tx - 1, global_ty - 1) as i64
                    } else {
                        cfl_map.ytob_at(global_tx - 1, global_ty - 1) as i64
                    }
                } else {
                    left
                };
                let guess = clamped_gradient(top as i32, left as i32, topleft as i32);
                let residual = actual - guess;
                let token = Token::new(ctx_id, pack_signed(residual));
                write_token(&token, dc_code, None, writer)?;
            }
        }
    }

    #[cfg(feature = "debug-tokens")]
    let after_cfl = writer.bits_written();

    // AC strategy tokens — write strategy code for each first block only
    // C++ does: if (!acs.IsFirstBlock()) continue;
    let mut left_acs = 0i32;
    for y in 0..region_ysize_blocks {
        for x in 0..region_xsize_blocks {
            let abs_bx = start_bx + x;
            let abs_by = start_by + y;
            if !ac_strategy.is_first(abs_bx, abs_by) {
                continue;
            }
            let cur = ac_strategy.strategy_code(abs_bx, abs_by) as i32;
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
            write_token(&token, dc_code, None, writer)?;
            left_acs = cur;
        }
    }

    #[cfg(feature = "debug-tokens")]
    let after_acs = writer.bits_written();

    // Quant field tokens — write for first blocks only, skip non-first
    // The initial left_qf = strategy_code of block (0,0) in the region
    let initial_acs_code = ac_strategy.strategy_code(start_bx, start_by) as i32;
    let mut left_qf = initial_acs_code;
    for y in 0..region_ysize_blocks {
        for x in 0..region_xsize_blocks {
            let abs_by = start_by + y;
            let abs_bx = start_bx + x;
            if !ac_strategy.is_first(abs_bx, abs_by) {
                continue;
            }
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
            let token = Token::new(ctx_id, pack_signed(residual));
            write_token(&token, dc_code, None, writer)?;
            left_qf = cur;
        }
    }

    #[cfg(feature = "debug-tokens")]
    let after_qf = writer.bits_written();

    // EPF (Edge-Preserving Filter) tokens - per-block sharpness values
    for by_local in 0..region_ysize_blocks {
        for bx_local in 0..region_xsize_blocks {
            let abs_by = start_by + by_local;
            let abs_bx = start_bx + bx_local;
            let sharpness = if let Some(sm) = sharpness_map {
                sm[abs_by * full_xsize_blocks + abs_bx] as i32
            } else {
                4 // default EPF sharpness
            };
            let token = Token::new(0, pack_signed(sharpness));
            write_token(&token, dc_code, None, writer)?;
        }
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
            region_xsize_blocks * region_ysize_blocks
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
                (11..=44).contains(&ctx),
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
        let quant_dc: [Vec<Vec<i32>>; 3] = [vec![], vec![], vec![]];
        let dc_code = super::super::static_codes::get_dc_entropy_code();
        let mut writer = BitWriter::new();
        assert!(write_dc_tokens(&quant_dc, &dc_code, &mut writer).is_ok());
        assert_eq!(writer.bits_written(), 0);
    }

    #[test]
    fn test_write_dc_tokens_simple() {
        // Create a simple 2x2 DC image with all zeros
        let quant_dc: [Vec<Vec<i32>>; 3] = [
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
