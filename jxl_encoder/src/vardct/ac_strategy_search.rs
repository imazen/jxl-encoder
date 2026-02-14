// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Hierarchical AC strategy search for multi-block transforms.
//!
//! Evaluates DCT16x16, DCT32x32, DCT64x64, and their rectangular variants
//! against the base DCT8 cost to find the optimal transform for each region.

use super::ac_strategy::*;

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(super) fn find_best_16x16_transform(
    xyb: [&[f32]; 3],
    stride: usize,
    bx0: usize,
    by0: usize,
    cx: usize,
    cy: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    ac_strategy: &mut AcStrategyMap,
    scratch: &mut EntropyEstScratch,
) {
    // In pixel-domain mode (mask1x1.is_some()), entropy_mul is applied internally
    // by estimate_entropy_full using fixed values per transform. External multipliers
    // are 1.0. In coefficient-domain mode, use libjxl-tiny distance-dependent multipliers.
    let use_pixel_domain = mask1x1.is_some();

    // Distance-dependent multipliers (from libjxl-tiny) - only used in coefficient-domain mode
    let (mul8x8, mul16x8, mul16x16, mul4x8, mul4x4) = if use_pixel_domain {
        // In pixel-domain mode, entropy_mul is handled internally. No external multiplier.
        (1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32)
    } else {
        let k8x8mul1: f32 = -0.55 * 0.75;
        let k8x8mul2: f32 = 1.073_575_8 * 0.75;
        let k8x8base: f32 = 1.4;
        let m8x8 = k8x8mul2 + k8x8mul1 / (distance + k8x8base);

        let k8x16mul1: f32 = -0.55;
        let k8x16mul2: f32 = 0.901_958_8;
        let k8x16base: f32 = 1.6;
        let m16x8 = k8x16mul2 + k8x16mul1 / (distance + k8x16base);

        let k16x16mul1: f32 = -0.65;
        let k16x16mul2: f32 = 0.88;
        let k16x16base: f32 = 1.8;
        let m16x16 = k16x16mul2 + k16x16mul1 / (distance + k16x16base);

        let k4x8mul1: f32 = -0.50 * 0.75;
        let k4x8mul2: f32 = 0.88;
        let k4x8base: f32 = 1.3;
        let m4x8 = k4x8mul2 + k4x8mul1 / (distance + k4x8base);

        let k4x4mul1: f32 = -0.45 * 0.75;
        let k4x4mul2: f32 = 0.85;
        let k4x4base: f32 = 1.2;
        let m4x4 = k4x4mul2 + k4x4mul1 / (distance + k4x4base);

        (m8x8, m16x8, m16x16, m4x8, m4x4)
    };

    // Base cost added for DCT8 transforms (from libjxl-tiny)
    // In pixel-domain mode, this is 0 since costs are already calibrated
    let base_cost_8x8 = if use_pixel_domain { 0.0 } else { 3.0 * mul8x8 };

    // Entropy_mul adjustments from libjxl enc_ac_strategy.cc:585-600.
    // These are applied INSIDE EstimateEntropy to the entropy portion only,
    // NOT as post-hoc cost multipliers (which would incorrectly scale loss too).

    // kFavor2X2AtHighQuality: bonus for IDENTITY/DCT2X2 at distance < 5.0.
    // Matches libjxl enc_ac_strategy.cc:585-590: -0.4 * ((5-d)/5)^2
    // AdjustQuantBlockAC is now implemented, which prevents over-selection.
    let favor_weight = if distance < 5.0 {
        ((5.0 - distance) / 5.0_f32).powi(2)
    } else {
        0.0
    };
    let favor_2x2_adjust = -0.4 * favor_weight; // matches libjxl

    // kAvoidEntropyOfTransforms: penalty for non-DCT/non-2x2/non-IDENTITY at distance > 4.0
    let avoid_transforms_adjust = if distance > 4.0 {
        let mul = if distance < 12.0 {
            (12.0 - 4.0) / (distance - 4.0)
        } else {
            1.0
        };
        0.5 * mul // positive = increases entropy_mul = higher cost
    } else {
        0.0
    };

    let abs_bx = bx0 + cx;
    let abs_by = by0 + cy;

    // Evaluate four 8×8 blocks with DCT8, DCT4X8, DCT8X4, DCT4X4, IDENTITY, DCT2X2
    // Track entropy and best strategy for each block
    let mut entropy = [[0.0f32; 2]; 2];
    let mut best_single_strategy = [[RAW_STRATEGY_DCT8; 2]; 2];
    for (dy, (entropy_row, strat_row)) in entropy
        .iter_mut()
        .zip(best_single_strategy.iter_mut())
        .enumerate()
    {
        for (dx, (entropy_val, best_strat)) in
            entropy_row.iter_mut().zip(strat_row.iter_mut()).enumerate()
        {
            let block_x = abs_bx + dx;
            let block_y = abs_by + dy;

            // Helper macro: evaluate a single-block strategy with entropy_mul adjustment
            macro_rules! eval {
                ($strategy:expr, $adjust:expr) => {
                    estimate_entropy_with_mask(
                        $strategy,
                        xyb,
                        stride,
                        block_x,
                        block_y,
                        distance,
                        quant_field,
                        xsize_blocks,
                        masking,
                        ytox,
                        ytob,
                        mask1x1,
                        mask1x1_stride,
                        $adjust,
                        scratch,
                    )
                };
            }

            // DCT8 (no adjustment)
            let e8 = eval!(RAW_STRATEGY_DCT8, 0.0);
            let cost8 = base_cost_8x8 + mul8x8 * e8;

            // DCT4X8 (kAvoidEntropy penalty at high distance)
            let e4x8 = eval!(RAW_STRATEGY_DCT4X8, avoid_transforms_adjust);
            let base_cost_4x8 = if use_pixel_domain { 0.0 } else { 3.0 * mul4x8 };
            let cost4x8 = base_cost_4x8 + mul4x8 * e4x8;

            // DCT8X4
            let e8x4 = eval!(RAW_STRATEGY_DCT8X4, avoid_transforms_adjust);
            let cost8x4 = base_cost_4x8 + mul4x8 * e8x4;

            // DCT4X4
            let e4x4 = eval!(RAW_STRATEGY_DCT4X4, avoid_transforms_adjust);
            let base_cost_4x4 = if use_pixel_domain { 0.0 } else { 3.0 * mul4x4 };
            let cost4x4 = base_cost_4x4 + mul4x4 * e4x4;

            // IDENTITY (kFavor2X2 bonus at low distance)
            let e_identity = eval!(RAW_STRATEGY_IDENTITY, favor_2x2_adjust);
            let base_cost_identity = if use_pixel_domain { 0.0 } else { 3.0 * mul8x8 };
            let cost_identity = base_cost_identity + mul8x8 * e_identity;

            // DCT2X2 (kFavor2X2 bonus at low distance)
            let e_dct2 = eval!(RAW_STRATEGY_DCT2X2, favor_2x2_adjust);
            let base_cost_dct2 = if use_pixel_domain { 0.0 } else { 3.0 * mul8x8 };
            let cost_dct2 = base_cost_dct2 + mul8x8 * e_dct2;

            // Pick the best single-block strategy
            *entropy_val = cost8;
            *best_strat = RAW_STRATEGY_DCT8;

            if cost4x8 < *entropy_val {
                *entropy_val = cost4x8;
                *best_strat = RAW_STRATEGY_DCT4X8;
            }
            if cost8x4 < *entropy_val {
                *entropy_val = cost8x4;
                *best_strat = RAW_STRATEGY_DCT8X4;
            }
            if cost4x4 < *entropy_val {
                *entropy_val = cost4x4;
                *best_strat = RAW_STRATEGY_DCT4X4;
            }
            if cost_identity < *entropy_val {
                *entropy_val = cost_identity;
                *best_strat = RAW_STRATEGY_IDENTITY;
            }
            if cost_dct2 < *entropy_val {
                *entropy_val = cost_dct2;
                *best_strat = RAW_STRATEGY_DCT2X2;
            }

            // AFV0-3 corner DCT
            // AFV auto-selection disabled in pixel-domain mode: the inverse AFV
            // transform produces systematically underestimated pixel-domain error,
            // causing AFV to be selected too aggressively (35% AFV vs libjxl's <5%).
            // This caused a massive quality regression (SSIM2 84→57 on frymire).
            // Re-enable once the AFV pixel-domain cost model is calibrated.
            if !use_pixel_domain {
                let base_cost_afv = 3.0 * mul8x8;
                let e_afv0 = eval!(RAW_STRATEGY_AFV0, avoid_transforms_adjust);
                let e_afv1 = eval!(RAW_STRATEGY_AFV1, avoid_transforms_adjust);
                let e_afv2 = eval!(RAW_STRATEGY_AFV2, avoid_transforms_adjust);
                let e_afv3 = eval!(RAW_STRATEGY_AFV3, avoid_transforms_adjust);
                let cost_afv0 = base_cost_afv + mul8x8 * e_afv0;
                let cost_afv1 = base_cost_afv + mul8x8 * e_afv1;
                let cost_afv2 = base_cost_afv + mul8x8 * e_afv2;
                let cost_afv3 = base_cost_afv + mul8x8 * e_afv3;

                if cost_afv0 < *entropy_val {
                    *entropy_val = cost_afv0;
                    *best_strat = RAW_STRATEGY_AFV0;
                }
                if cost_afv1 < *entropy_val {
                    *entropy_val = cost_afv1;
                    *best_strat = RAW_STRATEGY_AFV1;
                }
                if cost_afv2 < *entropy_val {
                    *entropy_val = cost_afv2;
                    *best_strat = RAW_STRATEGY_AFV2;
                }
                if cost_afv3 < *entropy_val {
                    *entropy_val = cost_afv3;
                    *best_strat = RAW_STRATEGY_AFV3;
                }
            }
        }
    }

    // Evaluate two DCT16X8 options (left column, right column)
    let entropy_16x8_left = mul16x8
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT16X8,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );
    let entropy_16x8_right = mul16x8
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT16X8,
            xyb,
            stride,
            abs_bx + 1,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );

    // Evaluate two DCT8X16 options (top row, bottom row)
    let entropy_8x16_top = mul16x8
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT8X16,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );
    let entropy_8x16_bottom = mul16x8
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT8X16,
            xyb,
            stride,
            abs_bx,
            abs_by + 1,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );

    // Evaluate DCT16x16 (one transform covering the entire 2x2 region)
    let entropy_16x16 = mul16x16
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT16X16,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );

    // Compare all options: four single-block, 16x8 split, 8x16 split, or one 16x16
    let cost_all_single = entropy[0][0] + entropy[0][1] + entropy[1][0] + entropy[1][1];
    let cost16x8 = (entropy_16x8_left).min(entropy[0][0] + entropy[1][0])
        + (entropy_16x8_right).min(entropy[0][1] + entropy[1][1]);
    let cost8x16 = (entropy_8x16_top).min(entropy[0][0] + entropy[0][1])
        + (entropy_8x16_bottom).min(entropy[1][0] + entropy[1][1]);
    let cost16x16 = entropy_16x16;

    // Find best non-single-block cost (minimum of 16x8, 8x16, 16x16)
    let best_rect = cost16x8.min(cost8x16);
    let best_large = best_rect.min(cost16x16);

    // Only use a non-single-block strategy if it beats four single-block transforms
    if best_large >= cost_all_single {
        // Keep all four as their best single-block strategy (DCT8, DCT4X8, or DCT8X4)
        for dy in 0..2 {
            for dx in 0..2 {
                let strat = best_single_strategy[dy][dx];
                if strat != RAW_STRATEGY_DCT8 {
                    ac_strategy.set(abs_bx + dx, abs_by + dy, strat);
                }
            }
        }
        return;
    }

    if cost16x16 <= best_rect {
        // DCT16x16 is the overall best
        ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT16X16);
    } else if cost16x8 < cost8x16 {
        // Try 16x8 for each column
        if entropy_16x8_left < entropy[0][0] + entropy[1][0] {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT16X8);
        } else {
            // Use best single-block for both blocks in left column
            for dy in 0..2 {
                let strat = best_single_strategy[dy][0];
                if strat != RAW_STRATEGY_DCT8 {
                    ac_strategy.set(abs_bx, abs_by + dy, strat);
                }
            }
        }
        if entropy_16x8_right < entropy[0][1] + entropy[1][1] {
            ac_strategy.set(abs_bx + 1, abs_by, RAW_STRATEGY_DCT16X8);
        } else {
            // Use best single-block for both blocks in right column
            for dy in 0..2 {
                let strat = best_single_strategy[dy][1];
                if strat != RAW_STRATEGY_DCT8 {
                    ac_strategy.set(abs_bx + 1, abs_by + dy, strat);
                }
            }
        }
    } else {
        // Try 8x16 for each row
        if entropy_8x16_top < entropy[0][0] + entropy[0][1] {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT8X16);
        } else {
            // Use best single-block for both blocks in top row
            for dx in 0..2 {
                let strat = best_single_strategy[0][dx];
                if strat != RAW_STRATEGY_DCT8 {
                    ac_strategy.set(abs_bx + dx, abs_by, strat);
                }
            }
        }
        if entropy_8x16_bottom < entropy[1][0] + entropy[1][1] {
            ac_strategy.set(abs_bx, abs_by + 1, RAW_STRATEGY_DCT8X16);
        } else {
            // Use best single-block for both blocks in bottom row
            for dx in 0..2 {
                let strat = best_single_strategy[1][dx];
                if strat != RAW_STRATEGY_DCT8 {
                    ac_strategy.set(abs_bx + dx, abs_by + 1, strat);
                }
            }
        }
    }
}

// ─── 32×32 transform selection ──────────────────────────────────────────────

/// Find the best transform for a 32×32 block region (4×4 group of 8×8 blocks).
///
/// Evaluates one DCT32x32 against four `find_best_16x16_transform` results.
/// Returns true if DCT32x32 was selected.
///
/// Only call when `bx + 3 < xsize_blocks && by + 3 < ysize_blocks`.
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    unreachable_code
)]
pub(super) fn find_best_32x32_transform(
    xyb: [&[f32]; 3],
    stride: usize,
    bx0: usize,
    by0: usize,
    cx: usize,
    cy: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    ac_strategy: &mut AcStrategyMap,
    scratch: &mut EntropyEstScratch,
) -> bool {
    // Large transforms (32x32, 32x16, 16x32) average large pixel blocks, which
    // works well for smooth content but produces blur on high-contrast edges.
    // The cost model correctly avoids them for high-contrast blocks.
    // Enable at d >= 2.0 where compression benefit outweighs edge blur risk.
    if distance < 2.0 {
        // At low distances, evaluate 16x16 and smaller transforms only
        for qy in (0..4).step_by(2) {
            for qx in (0..4).step_by(2) {
                find_best_16x16_transform(
                    xyb,
                    stride,
                    bx0,
                    by0,
                    cx + qx,
                    cy + qy,
                    distance,
                    quant_field,
                    xsize_blocks,
                    masking,
                    ytox,
                    ytob,
                    mask1x1,
                    mask1x1_stride,
                    ac_strategy,
                    scratch,
                );
            }
        }
        return false;
    }

    // At higher distances (d >= 2.0), evaluate DCT32x32, DCT32x16, DCT16x32 as options
    let k32x32mul1: f32 = -0.75;
    let k32x32mul2: f32 = 1.2; // Very conservative
    let k32x32base: f32 = 2.0;
    let mul32x32 = k32x32mul2 + k32x32mul1 / (distance + k32x32base);

    // DCT32x16/DCT16x32 use similar multipliers to DCT32x32
    let k32x16mul1: f32 = -0.70;
    let k32x16mul2: f32 = 1.1;
    let k32x16base: f32 = 2.0;
    let mul32x16 = k32x16mul2 + k32x16mul1 / (distance + k32x16base);

    let abs_bx = bx0 + cx;
    let abs_by = by0 + cy;

    // Evaluate DCT32x32 cost
    let entropy_32x32 = mul32x32
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT32X32,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );

    // Evaluate DCT32x16 costs (two transforms: at (0,0) and (0,2))
    // DCT32x16 covers 4 rows × 2 cols of 8x8 blocks
    let entropy_32x16_0 = mul32x16
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT32X16,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );
    let entropy_32x16_1 = mul32x16
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT32X16,
            xyb,
            stride,
            abs_bx + 2,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );
    let entropy_32x16_total = entropy_32x16_0 + entropy_32x16_1;

    // Evaluate DCT16x32 costs (two transforms: at (0,0) and (2,0))
    // DCT16x32 covers 2 rows × 4 cols of 8x8 blocks
    let entropy_16x32_0 = mul32x16
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT16X32,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );
    let entropy_16x32_1 = mul32x16
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT16X32,
            xyb,
            stride,
            abs_bx,
            abs_by + 2,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );
    let entropy_16x32_total = entropy_16x32_0 + entropy_16x32_1;

    // Run four 16x16 evaluations (each covers 2×2 blocks)
    for qy in (0..4).step_by(2) {
        for qx in (0..4).step_by(2) {
            find_best_16x16_transform(
                xyb,
                stride,
                bx0,
                by0,
                cx + qx,
                cy + qy,
                distance,
                quant_field,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                ac_strategy,
                scratch,
            );
        }
    }

    // Compute the combined cost of the four 16x16 sub-evaluations.
    // We need to re-estimate using whatever strategies were selected.
    let mut cost_sub = 0.0f32;
    for iy in 0..4 {
        for ix in 0..4 {
            if !ac_strategy.is_first(abs_bx + ix, abs_by + iy) {
                continue;
            }
            let sub_raw = ac_strategy.raw_strategy(abs_bx + ix, abs_by + iy);
            // Distance-dependent multipliers (must match find_best_16x16_transform)
            let k8x8mul1: f32 = -0.55 * 0.75;
            let k8x8mul2: f32 = 1.073_575_8 * 0.75;
            let k8x8base: f32 = 1.4;
            let mul8x8 = k8x8mul2 + k8x8mul1 / (distance + k8x8base);
            let k8x16mul1: f32 = -0.55;
            let k8x16mul2: f32 = 0.901_958_8;
            let k8x16base: f32 = 1.6;
            let mul16x8 = k8x16mul2 + k8x16mul1 / (distance + k8x16base);
            let k16x16mul1: f32 = -0.65;
            let k16x16mul2: f32 = 0.88;
            let k16x16base: f32 = 1.8;
            let mul16x16 = k16x16mul2 + k16x16mul1 / (distance + k16x16base);

            let mul = match sub_raw {
                RAW_STRATEGY_DCT8 => mul8x8,
                RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => mul16x8,
                RAW_STRATEGY_DCT16X16 => mul16x16,
                _ => mul8x8,
            };
            let base = if sub_raw == RAW_STRATEGY_DCT8 {
                3.0 * mul8x8
            } else {
                0.0
            };

            let e = estimate_entropy_with_mask(
                sub_raw,
                xyb,
                stride,
                abs_bx + ix,
                abs_by + iy,
                distance,
                quant_field,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                0.0,
                scratch,
            );
            cost_sub += base + mul * e;
        }
    }

    // Find the best option among: DCT32x32, DCT32x16 pair, DCT16x32 pair, 16x16 sub-evaluations
    let mut best_cost = cost_sub;
    let mut best_choice = 0u8; // 0 = keep sub, 1 = DCT32x32, 2 = DCT32x16, 3 = DCT16x32

    if entropy_32x32 < best_cost {
        best_cost = entropy_32x32;
        best_choice = 1;
    }
    // DCT32x16/DCT16x32 now enabled (fixed pixel extraction bug Feb 4, 2026)
    if entropy_32x16_total < best_cost {
        best_cost = entropy_32x16_total;
        best_choice = 2;
    }
    if entropy_16x32_total < best_cost {
        // best_cost = entropy_16x32_total; // Not needed, just using best_choice
        best_choice = 3;
    }

    match best_choice {
        1 => {
            // DCT32x32 wins
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT32X32);
            true
        }
        2 => {
            // Two DCT32x16 transforms win
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT32X16);
            ac_strategy.set(abs_bx + 2, abs_by, RAW_STRATEGY_DCT32X16);
            true
        }
        3 => {
            // Two DCT16x32 transforms win
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT16X32);
            ac_strategy.set(abs_bx, abs_by + 2, RAW_STRATEGY_DCT16X32);
            true
        }
        _ => {
            // Keep the 16x16 sub-evaluation results (already in ac_strategy)
            false
        }
    }
}

// ─── 64×64 transform selection ──────────────────────────────────────────────

/// Find the best transform for a 64×64 pixel region (8×8 group of 8×8 blocks).
///
/// Evaluates DCT64x64, two DCT64x32, two DCT32x64, and four find_best_32x32_transform.
/// Only evaluated at d >= 3.0 (conservative — DCT64 averages 64x64 blocks).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(super) fn find_best_64x64_transform(
    xyb: [&[f32]; 3],
    stride: usize,
    bx0: usize,
    by0: usize,
    cx: usize,
    cy: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    ac_strategy: &mut AcStrategyMap,
    scratch: &mut EntropyEstScratch,
) {
    // DCT64 transforms only at d >= 3.0
    if distance < 3.0 {
        // At lower distances, fall through to 32x32 evaluation
        for qy in (0..8).step_by(4) {
            for qx in (0..8).step_by(4) {
                find_best_32x32_transform(
                    xyb,
                    stride,
                    bx0,
                    by0,
                    cx + qx,
                    cy + qy,
                    distance,
                    quant_field,
                    xsize_blocks,
                    masking,
                    ytox,
                    ytob,
                    mask1x1,
                    mask1x1_stride,
                    ac_strategy,
                    scratch,
                );
            }
        }
        return;
    }

    // Conservative multipliers for DCT64 transforms
    let k64x64mul1: f32 = -0.80;
    let k64x64mul2: f32 = 1.3;
    let k64x64base: f32 = 2.5;
    let mul64x64 = k64x64mul2 + k64x64mul1 / (distance + k64x64base);

    let k64x32mul1: f32 = -0.75;
    let k64x32mul2: f32 = 1.2;
    let k64x32base: f32 = 2.5;
    let mul64x32 = k64x32mul2 + k64x32mul1 / (distance + k64x32base);

    let abs_bx = bx0 + cx;
    let abs_by = by0 + cy;

    // Evaluate DCT64x64 cost
    let entropy_64x64 = mul64x64
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT64X64,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );

    // Evaluate DCT64x32 costs (two transforms stacked vertically)
    // DCT64x32 covers 8 rows × 4 cols of 8×8 blocks
    // Split: left half (bx, by) and right half (bx+4, by)
    let entropy_64x32_0 = mul64x32
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT64X32,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );
    let entropy_64x32_1 = mul64x32
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT64X32,
            xyb,
            stride,
            abs_bx + 4,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );
    let entropy_64x32_total = entropy_64x32_0 + entropy_64x32_1;

    // Evaluate DCT32x64 costs (two transforms side by side)
    // DCT32x64 covers 4 rows × 8 cols of 8×8 blocks
    // Split: top half (bx, by) and bottom half (bx, by+4)
    let entropy_32x64_0 = mul64x32
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT32X64,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );
    let entropy_32x64_1 = mul64x32
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT32X64,
            xyb,
            stride,
            abs_bx,
            abs_by + 4,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
            scratch,
        );
    let entropy_32x64_total = entropy_32x64_0 + entropy_32x64_1;

    // Run four 32x32 evaluations (each covers 4×4 blocks)
    for qy in (0..8).step_by(4) {
        for qx in (0..8).step_by(4) {
            find_best_32x32_transform(
                xyb,
                stride,
                bx0,
                by0,
                cx + qx,
                cy + qy,
                distance,
                quant_field,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                ac_strategy,
                scratch,
            );
        }
    }

    // Compute the combined cost of the four 32x32 sub-evaluations
    let mut cost_sub = 0.0f32;
    for iy in 0..8 {
        for ix in 0..8 {
            if !ac_strategy.is_first(abs_bx + ix, abs_by + iy) {
                continue;
            }
            let sub_raw = ac_strategy.raw_strategy(abs_bx + ix, abs_by + iy);
            // Distance-dependent multipliers (must match find_best_32x32/16x16_transform)
            let k8x8mul1: f32 = -0.55 * 0.75;
            let k8x8mul2: f32 = 1.073_575_8 * 0.75;
            let k8x8base: f32 = 1.4;
            let mul8x8 = k8x8mul2 + k8x8mul1 / (distance + k8x8base);
            let k8x16mul1: f32 = -0.55;
            let k8x16mul2: f32 = 0.901_958_8;
            let k8x16base: f32 = 1.6;
            let mul16x8 = k8x16mul2 + k8x16mul1 / (distance + k8x16base);
            let k16x16mul1: f32 = -0.65;
            let k16x16mul2: f32 = 0.88;
            let k16x16base: f32 = 1.8;
            let mul16x16 = k16x16mul2 + k16x16mul1 / (distance + k16x16base);
            let k32x32mul1: f32 = -0.75;
            let k32x32mul2: f32 = 1.2;
            let k32x32base: f32 = 2.0;
            let mul32x32 = k32x32mul2 + k32x32mul1 / (distance + k32x32base);
            let k32x16mul1: f32 = -0.70;
            let k32x16mul2: f32 = 1.1;
            let k32x16base: f32 = 2.0;
            let mul32x16 = k32x16mul2 + k32x16mul1 / (distance + k32x16base);

            let mul = match sub_raw {
                RAW_STRATEGY_DCT8 => mul8x8,
                RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => mul16x8,
                RAW_STRATEGY_DCT16X16 => mul16x16,
                RAW_STRATEGY_DCT32X32 => mul32x32,
                RAW_STRATEGY_DCT32X16 | RAW_STRATEGY_DCT16X32 => mul32x16,
                _ => mul8x8,
            };
            let base = if sub_raw == RAW_STRATEGY_DCT8 {
                3.0 * mul8x8
            } else {
                0.0
            };

            let e = estimate_entropy_with_mask(
                sub_raw,
                xyb,
                stride,
                abs_bx + ix,
                abs_by + iy,
                distance,
                quant_field,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                0.0,
                scratch,
            );
            cost_sub += base + mul * e;
        }
    }

    // Find the best option
    let mut best_cost = cost_sub;
    let mut best_choice = 0u8; // 0=keep sub, 1=DCT64x64, 2=DCT64x32, 3=DCT32x64

    if entropy_64x64 < best_cost {
        best_cost = entropy_64x64;
        best_choice = 1;
    }
    if entropy_64x32_total < best_cost {
        best_cost = entropy_64x32_total;
        best_choice = 2;
    }
    if entropy_32x64_total < best_cost {
        let _ = best_cost;
        best_choice = 3;
    }

    match best_choice {
        1 => {
            // DCT64x64 wins
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT64X64);
        }
        2 => {
            // Two DCT64x32 transforms win
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT64X32);
            ac_strategy.set(abs_bx + 4, abs_by, RAW_STRATEGY_DCT64X32);
        }
        3 => {
            // Two DCT32x64 transforms win
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT32X64);
            ac_strategy.set(abs_bx, abs_by + 4, RAW_STRATEGY_DCT32X64);
        }
        _ => {
            // Keep the 32x32 sub-evaluation results (already in ac_strategy)
        }
    }
}
