// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

// archmage::arcane macro expansion doesn't propagate function-level allows.
#![allow(clippy::too_many_arguments)]

//! Hierarchical AC strategy search for multi-block transforms.
//!
//! Evaluates DCT16x16, DCT32x32, DCT64x64, and their rectangular variants
//! against the base DCT8 cost to find the optimal transform for each region.

use super::ac_strategy::*;
use crate::debug_rect;
use crate::effort::EffortProfile;

/// Store the cost of a transform covering `cx × cy` blocks at offset `(ox, oy)`
/// within the 64-element entropy_estimate cache.
/// Matches libjxl's `SetEntropyForTransform`: total cost at top-left, 0 elsewhere.
#[inline]
fn set_entropy_for_transform(
    entropy_estimate: &mut [f32; 64],
    ox: usize,
    oy: usize,
    cx: usize,
    cy: usize,
    entropy: f32,
) {
    for iy in 0..cy {
        for ix in 0..cx {
            entropy_estimate[(oy + iy) * 8 + (ox + ix)] = 0.0;
        }
    }
    entropy_estimate[oy * 8 + ox] = entropy;
}

/// libjxl `kFavor2X2AtHighQuality` weighting factor: returns `((5-d)/5)^2` at `d < 5`, else `0`.
/// See `enc_ac_strategy.cc::FindBest8x8Transform` (line 585-590).
#[inline]
pub(super) fn favor_2x2_weight(distance: f32) -> f32 {
    if distance < 5.0 {
        ((5.0 - distance) / 5.0_f32).powi(2)
    } else {
        0.0
    }
}

/// libjxl `kAvoidEntropyOfTransforms` multiplier: `(12-4)/(d-4)` at `4 < d < 12`,
/// `1.0` at `d >= 12`, and `0.0` (the penalty is OFF) at `d <= 4`.
///
/// Returned value is multiplied by `k_avoid_transforms_base` (libjxl: 0.5) and ADDED
/// to the entropy_mul of every non-DCT8 / non-DCT2X2 / non-IDENTITY 8×8-class
/// strategy (DCT4X4, DCT4X8, DCT8X4, AFV0-3).
///
/// See `enc_ac_strategy.cc::FindBest8x8Transform` (line 592-601).
///
/// Returning `0.0` below `d=4` (rather than the raw formula value) preserves the
/// libjxl semantics that the penalty is gated by `butteraugli_target > 4.0`.
#[inline]
pub(super) fn avoid_entropy_of_transforms_mul(distance: f32) -> f32 {
    if distance <= 4.0 {
        0.0
    } else if distance < 12.0 {
        (12.0 - 4.0) / (distance - 4.0)
    } else {
        1.0
    }
}

/// Dispatch wrapper: selects SIMD or scalar path for the entire search function.
/// When SIMD is available, the #[arcane] wrapper runs under #[target_feature], enabling
/// LLVM to inline estimate_entropy_with_mask → estimate_entropy_full_impl and optimize
/// the entire search (10-30 estimate_entropy calls) under a single target_feature scope.
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
    favor_single_mul: f32,
    cache_offset: Option<(usize, usize)>,
    profile: &EffortProfile,
) {
    #[cfg(target_arch = "x86_64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::X64V3Token::summon() {
            find_best_16x16_transform_avx2(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                favor_single_mul,
                cache_offset,
                profile,
            );
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::NeonToken::summon() {
            find_best_16x16_transform_neon(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                favor_single_mul,
                cache_offset,
                profile,
            );
            return;
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::Wasm128Token::summon() {
            find_best_16x16_transform_wasm128(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                favor_single_mul,
                cache_offset,
                profile,
            );
            return;
        }
    }
    find_best_16x16_transform_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        favor_single_mul,
        cache_offset,
        profile,
    );
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn find_best_16x16_transform_avx2(
    _token: jxl_simd::X64V3Token,
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
    favor_single_mul: f32,
    cache_offset: Option<(usize, usize)>,
    profile: &EffortProfile,
) {
    find_best_16x16_transform_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        favor_single_mul,
        cache_offset,
        profile,
    );
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn find_best_16x16_transform_neon(
    _token: jxl_simd::NeonToken,
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
    favor_single_mul: f32,
    cache_offset: Option<(usize, usize)>,
    profile: &EffortProfile,
) {
    find_best_16x16_transform_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        favor_single_mul,
        cache_offset,
        profile,
    );
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn find_best_16x16_transform_wasm128(
    _token: jxl_simd::Wasm128Token,
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
    favor_single_mul: f32,
    cache_offset: Option<(usize, usize)>,
    profile: &EffortProfile,
) {
    find_best_16x16_transform_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        favor_single_mul,
        cache_offset,
        profile,
    );
}

#[inline(always)]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn find_best_16x16_transform_impl(
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
    favor_single_mul: f32,
    cache_offset: Option<(usize, usize)>,
    profile: &EffortProfile,
) {
    // In pixel-domain mode (mask1x1.is_some()), entropy_mul is applied internally
    // by estimate_entropy_full using fixed values per transform. External multipliers
    // are 1.0. In coefficient-domain mode, use libjxl-tiny distance-dependent multipliers.
    let use_pixel_domain = mask1x1.is_some();

    // Distance-dependent multipliers from EffortProfile (libjxl enc_ac_strategy.cc:790-818).
    //
    // In pixel-domain mode: libjxl normalizes 8×8 entropy_mul (÷0.8) then multiplies the
    // TOTAL 8×8 cost (including loss) by mul8x8. Larger transforms use raw entropy_mul
    // internally with no external multiplier. So: mul8x8 for all 8×8-class, 1.0 for rest.
    //
    // IMPORTANT: pixel-domain mul8x8 uses libjxl's RAW constants (-0.4, 1.0, 1.4), NOT
    // the EffortProfile k8x8 which has a 0.75 factor baked in for coefficient-domain mode.
    //
    // In coefficient-domain mode: each size class has its own distance-dependent multiplier.
    let compute_mul = |k: (f32, f32, f32)| k.1 + k.0 / (distance + k.2);
    let (mul8x8, mul16x8, mul16x16, mul4x8, mul4x4) = if use_pixel_domain {
        // libjxl enc_ac_strategy.cc:863-866: k8x8mul1=-0.4, k8x8mul2=1.0, k8x8base=1.4
        let mul8x8 = 1.0 + (-0.4) / (distance + 1.4);
        // libjxl applies mul8x8 to ALL 8×8-class transforms (DCT8/4x8/8x4/4x4/AFV/ID/2x2)
        // Larger transforms: entropy_mul is applied internally, no external mul.
        (mul8x8, 1.0_f32, 1.0_f32, mul8x8, mul8x8)
    } else {
        (
            compute_mul(profile.k8x8),
            compute_mul(profile.k16x8),
            compute_mul(profile.k16x16),
            compute_mul(profile.k4x8),
            compute_mul(profile.k4x4),
        )
    };

    // Base cost added for DCT8 transforms (from libjxl-tiny)
    // In pixel-domain mode, this is 0 since costs are already calibrated
    let base_cost_8x8 = if use_pixel_domain { 0.0 } else { 3.0 * mul8x8 };

    // Entropy_mul adjustments from libjxl enc_ac_strategy.cc:585-600.
    // These are applied INSIDE EstimateEntropy to the entropy portion only,
    // NOT as post-hoc cost multipliers (which would incorrectly scale loss too).

    // kFavor2X2AtHighQuality: bonus for IDENTITY/DCT2X2 at distance < 5.0.
    // Matches libjxl enc_ac_strategy.cc:585-590.
    let favor_2x2_adjust = profile.k_favor_2x2 * favor_2x2_weight(distance);

    // kAvoidEntropyOfTransforms: penalty for non-DCT/non-2x2/non-IDENTITY at distance > 4.0.
    // Matches libjxl enc_ac_strategy.cc:592-601.
    let avoid_transforms_adjust =
        profile.k_avoid_transforms_base * avoid_entropy_of_transforms_mul(distance);

    let abs_bx = bx0 + cx;
    let abs_by = by0 + cy;
    // Pre-compute scaled constants once (was recomputed per estimate_entropy call)
    let scaled_constants = if use_pixel_domain {
        compute_scaled_constants(
            distance,
            (
                profile.k_info_loss_mul_base,
                profile.k_zeros_mul_base,
                profile.k_cost_delta_base,
            ),
        )
    } else {
        COEFF_DOMAIN_CONSTANTS
    };

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
                        scaled_constants,
                        &profile.entropy_mul_table,
                        scratch,
                    )
                };
            }

            // DCT8 (no adjustment)
            let e8 = eval!(RAW_STRATEGY_DCT8, 0.0);
            let cost8 = base_cost_8x8 + mul8x8 * e8;

            // Pick the best single-block strategy
            *entropy_val = cost8;
            *best_strat = RAW_STRATEGY_DCT8;

            // DCT4X8, DCT8X4, DCT4X4, AFV: gated by try_dct4x8_afv (effort >= 6 in libjxl)
            if profile.try_dct4x8_afv {
                let e4x8 = eval!(RAW_STRATEGY_DCT4X8, avoid_transforms_adjust);
                let base_cost_4x8 = if use_pixel_domain { 0.0 } else { 3.0 * mul4x8 };
                let cost4x8 = base_cost_4x8 + mul4x8 * e4x8;

                let e8x4 = eval!(RAW_STRATEGY_DCT8X4, avoid_transforms_adjust);
                let cost8x4 = base_cost_4x8 + mul4x8 * e8x4;

                let e4x4 = eval!(RAW_STRATEGY_DCT4X4, avoid_transforms_adjust);
                let base_cost_4x4 = if use_pixel_domain { 0.0 } else { 3.0 * mul4x4 };
                let cost4x4 = base_cost_4x4 + mul4x4 * e4x4;

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

                // AFV0-3: evaluate ALL 4 corner variants for every block.
                // Each AFV variant applies the corner DCT to a different quadrant
                // of the 8x8 block, so the optimal variant depends on pixel content,
                // not the block's position in the 2x2 group. libjxl evaluates all 4
                // per block in FindBest8x8Transform (enc_ac_strategy.cc:557-574).
                let base_cost_afv = if use_pixel_domain { 0.0 } else { 3.0 * mul8x8 };
                for afv_idx in 0..4u8 {
                    let afv_strategy = RAW_STRATEGY_AFV0 + afv_idx;
                    let e_afv = eval!(afv_strategy, avoid_transforms_adjust);
                    let cost_afv = base_cost_afv + mul8x8 * e_afv;
                    if cost_afv < *entropy_val {
                        *entropy_val = cost_afv;
                        *best_strat = afv_strategy;
                    }
                }
            }

            // IDENTITY (kFavor2X2 bonus at low distance)
            let e_identity = eval!(RAW_STRATEGY_IDENTITY, favor_2x2_adjust);
            let base_cost_identity = if use_pixel_domain { 0.0 } else { 3.0 * mul8x8 };
            let cost_identity = base_cost_identity + mul8x8 * e_identity;

            // DCT2X2 (kFavor2X2 bonus at low distance)
            let e_dct2 = eval!(RAW_STRATEGY_DCT2X2, favor_2x2_adjust);
            let base_cost_dct2 = if use_pixel_domain { 0.0 } else { 3.0 * mul8x8 };
            let cost_dct2 = base_cost_dct2 + mul8x8 * e_dct2;

            if cost_identity < *entropy_val {
                *entropy_val = cost_identity;
                *best_strat = RAW_STRATEGY_IDENTITY;
            }
            if cost_dct2 < *entropy_val {
                *entropy_val = cost_dct2;
                *best_strat = RAW_STRATEGY_DCT2X2;
            }
        }
    }

    // If max strategy size is 8 (try_dct16 = false), skip multi-block evaluation
    // and just assign the best 8x8-class strategy per block.
    if !profile.try_dct16 {
        for dy in 0..2 {
            for dx in 0..2 {
                let strat = best_single_strategy[dy][dx];
                if strat != RAW_STRATEGY_DCT8 {
                    ac_strategy.set(abs_bx + dx, abs_by + dy, strat);
                }
            }
        }
        if let Some((ox, oy)) = cache_offset {
            for dy in 0..2 {
                for dx in 0..2 {
                    scratch.entropy_estimate[(oy + dy) * 8 + (ox + dx)] = entropy[dy][dx];
                }
            }
        }
        return;
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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
            scratch,
        );

    // Apply favor_single_mul to single-block costs (matches libjxl's mul8x8).
    // This makes multi-block transforms harder to select at low distances,
    // preventing marginal multi-block wins that hurt butteraugli.
    if favor_single_mul != 1.0 {
        for row in &mut entropy {
            for val in row {
                *val *= favor_single_mul;
            }
        }
    }

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

    // Gather input context for the decision
    let qi = abs_by * xsize_blocks + abs_bx;
    let qf_avg = if qi + 1 + xsize_blocks < quant_field.len() {
        (quant_field[qi]
            + quant_field[qi + 1]
            + quant_field[qi + xsize_blocks]
            + quant_field[qi + 1 + xsize_blocks])
            / 4.0
    } else {
        quant_field.get(qi).copied().unwrap_or(0.0)
    };
    let mask_avg = if qi + 1 + xsize_blocks < masking.len() {
        (masking[qi]
            + masking[qi + 1]
            + masking[qi + xsize_blocks]
            + masking[qi + 1 + xsize_blocks])
            / 4.0
    } else {
        masking.get(qi).copied().unwrap_or(0.0)
    };
    debug_rect!(
        "acs/16x16",
        abs_bx * 8,
        abs_by * 8,
        16,
        16,
        "winner={} singles={:.0} 16x16={:.0} 16x8={:.0} 8x16={:.0} | qf={:.1} mask={:.2} mul8={:.2} mul16={:.2}",
        if best_large >= cost_all_single {
            "singles"
        } else if cost16x16 <= best_rect {
            "16x16"
        } else if cost16x8 < cost8x16 {
            "16x8"
        } else {
            "8x16"
        },
        cost_all_single,
        cost16x16,
        cost16x8,
        cost8x16,
        qf_avg,
        mask_avg,
        mul8x8,
        mul16x16
    );

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
        if let Some((ox, oy)) = cache_offset {
            for dy in 0..2 {
                for dx in 0..2 {
                    scratch.entropy_estimate[(oy + dy) * 8 + (ox + dx)] = entropy[dy][dx];
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

    // Store winning costs in entropy_estimate cache for parent 32×32/64×64 to read.
    if let Some((ox, oy)) = cache_offset {
        if cost16x16 <= best_rect {
            set_entropy_for_transform(&mut scratch.entropy_estimate, ox, oy, 2, 2, entropy_16x16);
        } else if cost16x8 < cost8x16 {
            // Left column
            if entropy_16x8_left < entropy[0][0] + entropy[1][0] {
                set_entropy_for_transform(
                    &mut scratch.entropy_estimate,
                    ox,
                    oy,
                    1,
                    2,
                    entropy_16x8_left,
                );
            } else {
                scratch.entropy_estimate[oy * 8 + ox] = entropy[0][0];
                scratch.entropy_estimate[(oy + 1) * 8 + ox] = entropy[1][0];
            }
            // Right column
            if entropy_16x8_right < entropy[0][1] + entropy[1][1] {
                set_entropy_for_transform(
                    &mut scratch.entropy_estimate,
                    ox + 1,
                    oy,
                    1,
                    2,
                    entropy_16x8_right,
                );
            } else {
                scratch.entropy_estimate[oy * 8 + (ox + 1)] = entropy[0][1];
                scratch.entropy_estimate[(oy + 1) * 8 + (ox + 1)] = entropy[1][1];
            }
        } else {
            // Top row
            if entropy_8x16_top < entropy[0][0] + entropy[0][1] {
                set_entropy_for_transform(
                    &mut scratch.entropy_estimate,
                    ox,
                    oy,
                    2,
                    1,
                    entropy_8x16_top,
                );
            } else {
                scratch.entropy_estimate[oy * 8 + ox] = entropy[0][0];
                scratch.entropy_estimate[oy * 8 + (ox + 1)] = entropy[0][1];
            }
            // Bottom row
            if entropy_8x16_bottom < entropy[1][0] + entropy[1][1] {
                set_entropy_for_transform(
                    &mut scratch.entropy_estimate,
                    ox,
                    oy + 1,
                    2,
                    1,
                    entropy_8x16_bottom,
                );
            } else {
                scratch.entropy_estimate[(oy + 1) * 8 + ox] = entropy[1][0];
                scratch.entropy_estimate[(oy + 1) * 8 + (ox + 1)] = entropy[1][1];
            }
        }
    }
}

// ─── Lightweight non-aligned merge check ────────────────────────────────────

/// Lightweight merge check for non-aligned 16×16 positions.
///
/// Instead of re-evaluating all single-block strategies (~45 `estimate_entropy`
/// calls per position), reads cached single-block costs from
/// `scratch.entropy_estimate` and only evaluates merge candidates (DCT16×16,
/// DCT16×8, DCT8×16) — 5 calls total. Matches libjxl's TryMergeAcs approach.
///
/// Returns true if a multi-block transform was accepted.
///
/// PRECONDITIONS:
/// - `scratch.entropy_estimate` must be populated by the aligned pass
///   (i.e., this must be a full tile where `find_best_64x64_transform` ran)
/// - `ac_strategy.can_evaluate_region(abs_bx, abs_by, 2)` must be true
///   (all 4 blocks are single-block transforms)
#[allow(clippy::too_many_arguments)]
pub(super) fn try_merge_16x16(
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
    profile: &EffortProfile,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::X64V3Token::summon() {
            return try_merge_16x16_avx2(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                profile,
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::NeonToken::summon() {
            return try_merge_16x16_neon(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                profile,
            );
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::Wasm128Token::summon() {
            return try_merge_16x16_wasm128(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                profile,
            );
        }
    }
    try_merge_16x16_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        profile,
    )
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn try_merge_16x16_avx2(
    _token: jxl_simd::X64V3Token,
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
    profile: &EffortProfile,
) -> bool {
    try_merge_16x16_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        profile,
    )
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn try_merge_16x16_neon(
    _token: jxl_simd::NeonToken,
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
    profile: &EffortProfile,
) -> bool {
    try_merge_16x16_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        profile,
    )
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn try_merge_16x16_wasm128(
    _token: jxl_simd::Wasm128Token,
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
    profile: &EffortProfile,
) -> bool {
    try_merge_16x16_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        profile,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn try_merge_16x16_impl(
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
    profile: &EffortProfile,
) -> bool {
    let use_pixel_domain = mask1x1.is_some();
    // Pre-compute scaled constants once (was recomputed per estimate_entropy call)
    let scaled_constants = if use_pixel_domain {
        compute_scaled_constants(
            distance,
            (
                profile.k_info_loss_mul_base,
                profile.k_zeros_mul_base,
                profile.k_cost_delta_base,
            ),
        )
    } else {
        COEFF_DOMAIN_CONSTANTS
    };

    let (mul16x8, mul16x16) = if use_pixel_domain {
        (1.0_f32, 1.0_f32)
    } else {
        let compute_mul = |k: (f32, f32, f32)| k.1 + k.0 / (distance + k.2);
        (compute_mul(profile.k16x8), compute_mul(profile.k16x16))
    };

    let abs_bx = bx0 + cx;
    let abs_by = by0 + cy;

    // Read cached single-block costs from the aligned pass.
    // (cx, cy) are tile-relative coords; entropy_estimate is indexed as [iy * 8 + ix].
    let cached = [
        [
            scratch.entropy_estimate[cy * 8 + cx],
            scratch.entropy_estimate[cy * 8 + (cx + 1)],
        ],
        [
            scratch.entropy_estimate[(cy + 1) * 8 + cx],
            scratch.entropy_estimate[(cy + 1) * 8 + (cx + 1)],
        ],
    ];
    let cost_all_single = cached[0][0] + cached[0][1] + cached[1][0] + cached[1][1];

    // Evaluate merge candidates only (5 calls instead of ~45)
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
            scaled_constants,
            &profile.entropy_mul_table,
            scratch,
        );

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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
            scratch,
        );

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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
            scratch,
        );

    // Cost comparison (same logic as find_best_16x16_transform)
    let cost_single_left = cached[0][0] + cached[1][0];
    let cost_single_right = cached[0][1] + cached[1][1];
    let cost_single_top = cached[0][0] + cached[0][1];
    let cost_single_bottom = cached[1][0] + cached[1][1];

    let cost16x8 =
        entropy_16x8_left.min(cost_single_left) + entropy_16x8_right.min(cost_single_right);
    let cost8x16 =
        entropy_8x16_top.min(cost_single_top) + entropy_8x16_bottom.min(cost_single_bottom);
    let cost16x16 = entropy_16x16;

    let best_rect = cost16x8.min(cost8x16);
    let best_large = best_rect.min(cost16x16);

    if best_large >= cost_all_single {
        return false; // No merge improvement
    }

    // A merge won — reset 2×2 region to DCT8 first to avoid orphaned non-first
    // blocks from any previous multi-block transform in this region.
    //
    // NOTE (F-D Sub-G, 2026-05-18): root cause pinned by source-diff vs libjxl
    // `FindBestFirstLevelDivisionForSquare` (enc_ac_strategy.cc:703-825). libjxl
    // does NOT reset because its inner-split crossing checks (`allow_JXK`/
    // `allow_KXJ` via `MultiBlockTransformCrossesVerticalBoundary` at
    // `bx+cx+blocks_half`) refuse to evaluate JXK/KXJ when an existing
    // multi-block transform spans both halves — so partial-JXK wins never
    // orphan an existing JXJ's non-first markers. Our `can_evaluate_region`
    // is LOOSER: it accepts regions containing fully-contained existing
    // multi-blocks, so dropping the reset corrupts those markers on a
    // partial-JXK win → "varblocks overlap" decode failures (reproduced at
    // d=0.25/0.5/1.0 on real photos in the prior session). Two ways to close
    // the divergence without the reset:
    //   (A) tighten `can_evaluate_region` to require all-single-block (loses
    //       merge attempts but simplest);
    //   (B) port libjxl's per-half inner-crossing checks into the merge
    //       comparison (most faithful, larger change touching both impls and
    //       the three SIMD variants each).
    // Divergence is real (we lose best 8×8-class strategies on non-winning
    // halves) but the fix needs hash-lock regen and decoder re-verification
    // across the full sweep — out of scope for this audit chunk.
    // Audit memory: f_d_sub_chunk_g_try_merge_2026-05-18.md.
    for dy in 0..2usize {
        for dx in 0..2usize {
            ac_strategy.set(abs_bx + dx, abs_by + dy, RAW_STRATEGY_DCT8);
        }
    }

    // Apply the winning transform and update entropy cache.
    if cost16x16 <= best_rect {
        ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT16X16);
        set_entropy_for_transform(&mut scratch.entropy_estimate, cx, cy, 2, 2, entropy_16x16);
    } else if cost16x8 < cost8x16 {
        // Try 16x8 for each column independently
        if entropy_16x8_left < cost_single_left {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT16X8);
            set_entropy_for_transform(
                &mut scratch.entropy_estimate,
                cx,
                cy,
                1,
                2,
                entropy_16x8_left,
            );
        }
        if entropy_16x8_right < cost_single_right {
            ac_strategy.set(abs_bx + 1, abs_by, RAW_STRATEGY_DCT16X8);
            set_entropy_for_transform(
                &mut scratch.entropy_estimate,
                cx + 1,
                cy,
                1,
                2,
                entropy_16x8_right,
            );
        }
    } else {
        // Try 8x16 for each row independently
        if entropy_8x16_top < cost_single_top {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT8X16);
            set_entropy_for_transform(
                &mut scratch.entropy_estimate,
                cx,
                cy,
                2,
                1,
                entropy_8x16_top,
            );
        }
        if entropy_8x16_bottom < cost_single_bottom {
            ac_strategy.set(abs_bx, abs_by + 1, RAW_STRATEGY_DCT8X16);
            set_entropy_for_transform(
                &mut scratch.entropy_estimate,
                cx,
                cy + 1,
                2,
                1,
                entropy_8x16_bottom,
            );
        }
    }

    true
}

/// Lightweight merge check for non-aligned 32×32 positions.
///
/// Like `try_merge_16x16` but for 4×4 block regions. Reads cached sub-costs
/// from `scratch.entropy_estimate` and only evaluates 5 large candidates
/// (DCT32×32, 2×DCT32×16, 2×DCT16×32). Replaces the full
/// `find_best_32x32_transform` which internally runs 4×find_best_16x16 +
/// re-evaluates all sub-block costs (~60+ estimate_entropy calls → 5 calls).
///
/// Returns true if a multi-block transform was accepted.
///
/// PRECONDITIONS:
/// - `scratch.entropy_estimate` must be populated by the aligned pass
/// - `ac_strategy.can_evaluate_region(abs_bx, abs_by, 4)` must be true
#[allow(clippy::too_many_arguments)]
pub(super) fn try_merge_32x32(
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
    profile: &EffortProfile,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::X64V3Token::summon() {
            return try_merge_32x32_avx2(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                profile,
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::NeonToken::summon() {
            return try_merge_32x32_neon(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                profile,
            );
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::Wasm128Token::summon() {
            return try_merge_32x32_wasm128(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                profile,
            );
        }
    }
    try_merge_32x32_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        profile,
    )
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn try_merge_32x32_avx2(
    _token: jxl_simd::X64V3Token,
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
    profile: &EffortProfile,
) -> bool {
    try_merge_32x32_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        profile,
    )
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn try_merge_32x32_neon(
    _token: jxl_simd::NeonToken,
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
    profile: &EffortProfile,
) -> bool {
    try_merge_32x32_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        profile,
    )
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn try_merge_32x32_wasm128(
    _token: jxl_simd::Wasm128Token,
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
    profile: &EffortProfile,
) -> bool {
    try_merge_32x32_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        profile,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn try_merge_32x32_impl(
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
    profile: &EffortProfile,
) -> bool {
    let use_pixel_domain = mask1x1.is_some();
    // Pre-compute scaled constants once (was recomputed per estimate_entropy call)
    let scaled_constants = if use_pixel_domain {
        compute_scaled_constants(
            distance,
            (
                profile.k_info_loss_mul_base,
                profile.k_zeros_mul_base,
                profile.k_cost_delta_base,
            ),
        )
    } else {
        COEFF_DOMAIN_CONSTANTS
    };

    let (mul32x32, mul32x16) = if use_pixel_domain {
        (1.0_f32, 1.0_f32)
    } else {
        let k32x32mul1: f32 = -0.75;
        let k32x32mul2: f32 = 1.2;
        let k32x32base: f32 = 2.0;
        let m32 = k32x32mul2 + k32x32mul1 / (distance + k32x32base);

        let k32x16mul1: f32 = -0.70;
        let k32x16mul2: f32 = 1.1;
        let k32x16base: f32 = 2.0;
        let m16 = k32x16mul2 + k32x16mul1 / (distance + k32x16base);
        (m32, m16)
    };

    let abs_bx = bx0 + cx;
    let abs_by = by0 + cy;

    // Read cached sub-costs from the aligned + non-aligned 16×16 passes.
    // Sum into per-quadrant costs (2×2 array of quadrants, each covering 2×2 blocks).
    let mut quadrant_cost = [[0.0f32; 2]; 2];
    for iy in 0..4 {
        for ix in 0..4 {
            quadrant_cost[iy / 2][ix / 2] += scratch.entropy_estimate[(cy + iy) * 8 + (cx + ix)];
        }
    }
    let sub_left = quadrant_cost[0][0] + quadrant_cost[1][0];
    let sub_right = quadrant_cost[0][1] + quadrant_cost[1][1];
    let sub_top = quadrant_cost[0][0] + quadrant_cost[0][1];
    let sub_bottom = quadrant_cost[1][0] + quadrant_cost[1][1];
    let cost_sub = sub_left + sub_right;

    // Evaluate 5 large transform candidates.
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
            scaled_constants,
            &profile.entropy_mul_table,
            scratch,
        );
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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
            scratch,
        );
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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
            scratch,
        );

    // Three-way comparison matching libjxl FindBestFirstLevelDivisionForSquare.
    let cost_jxn = entropy_32x16_0.min(sub_left) + entropy_32x16_1.min(sub_right);
    let cost_nxj = entropy_16x32_0.min(sub_top) + entropy_16x32_1.min(sub_bottom);

    // Check if any large transform beats the sub-costs.
    let best_large = entropy_32x32.min(cost_jxn).min(cost_nxj);
    if best_large >= cost_sub {
        return false;
    }

    // A merge won — reset 4×4 region to DCT8 first to avoid orphaned non-first blocks.
    // (See note in try_merge_16x16_impl: root cause is our `can_evaluate_region`
    //  accepting fully-contained existing multi-block transforms, vs libjxl's
    //  per-half inner-crossing checks that reject those cases. Dropping the
    //  reset triggers "varblocks overlap" decode failures on real photos.
    //  Audit memory: f_d_sub_chunk_g_try_merge_2026-05-18.md.)
    for dy in 0..4usize {
        for dx in 0..4usize {
            ac_strategy.set(abs_bx + dx, abs_by + dy, RAW_STRATEGY_DCT8);
        }
    }

    if entropy_32x32 < cost_jxn && entropy_32x32 < cost_nxj {
        // DCT32x32 wins
        ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT32X32);
        set_entropy_for_transform(&mut scratch.entropy_estimate, cx, cy, 4, 4, entropy_32x32);
    } else if cost_jxn < cost_nxj {
        // Vertical split (DCT32X16) — try each half independently
        if entropy_32x16_0 < sub_left {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT32X16);
            set_entropy_for_transform(&mut scratch.entropy_estimate, cx, cy, 2, 4, entropy_32x16_0);
        }
        if entropy_32x16_1 < sub_right {
            ac_strategy.set(abs_bx + 2, abs_by, RAW_STRATEGY_DCT32X16);
            set_entropy_for_transform(
                &mut scratch.entropy_estimate,
                cx + 2,
                cy,
                2,
                4,
                entropy_32x16_1,
            );
        }
    } else {
        // Horizontal split (DCT16X32) — try each half independently
        if entropy_16x32_0 < sub_top {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT16X32);
            set_entropy_for_transform(&mut scratch.entropy_estimate, cx, cy, 4, 2, entropy_16x32_0);
        }
        if entropy_16x32_1 < sub_bottom {
            ac_strategy.set(abs_bx, abs_by + 2, RAW_STRATEGY_DCT16X32);
            set_entropy_for_transform(
                &mut scratch.entropy_estimate,
                cx,
                cy + 2,
                4,
                2,
                entropy_16x32_1,
            );
        }
    }

    true
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
    cache_offset: Option<(usize, usize)>,
    profile: &EffortProfile,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::X64V3Token::summon() {
            return find_best_32x32_transform_avx2(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                cache_offset,
                profile,
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::NeonToken::summon() {
            return find_best_32x32_transform_neon(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                cache_offset,
                profile,
            );
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::Wasm128Token::summon() {
            return find_best_32x32_transform_wasm128(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                cache_offset,
                profile,
            );
        }
    }
    find_best_32x32_transform_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        cache_offset,
        profile,
    )
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    unreachable_code
)]
fn find_best_32x32_transform_avx2(
    _token: jxl_simd::X64V3Token,
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
    cache_offset: Option<(usize, usize)>,
    profile: &EffortProfile,
) -> bool {
    find_best_32x32_transform_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        cache_offset,
        profile,
    )
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    unreachable_code
)]
fn find_best_32x32_transform_neon(
    _token: jxl_simd::NeonToken,
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
    cache_offset: Option<(usize, usize)>,
    profile: &EffortProfile,
) -> bool {
    find_best_32x32_transform_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        cache_offset,
        profile,
    )
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    unreachable_code
)]
fn find_best_32x32_transform_wasm128(
    _token: jxl_simd::Wasm128Token,
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
    cache_offset: Option<(usize, usize)>,
    profile: &EffortProfile,
) -> bool {
    find_best_32x32_transform_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        cache_offset,
        profile,
    )
}

#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    unreachable_code
)]
fn find_best_32x32_transform_impl(
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
    cache_offset: Option<(usize, usize)>,
    profile: &EffortProfile,
) -> bool {
    // libjxl evaluates all strategies at all distances — no distance gates.
    // The cost model (EstimateEntropy with pixel-domain loss) naturally avoids
    // large transforms when they're not beneficial.
    // In pixel-domain mode, entropy_mul is applied internally by estimate_entropy_with_mask
    // using libjxl's static constants (1.48 for DCT32x32, 1.49 for DCT32x16).
    // In coefficient-domain mode, use distance-dependent multipliers.
    let use_pixel_domain = mask1x1.is_some();
    // Pre-compute scaled constants once (was recomputed per estimate_entropy call)
    let scaled_constants = if use_pixel_domain {
        compute_scaled_constants(
            distance,
            (
                profile.k_info_loss_mul_base,
                profile.k_zeros_mul_base,
                profile.k_cost_delta_base,
            ),
        )
    } else {
        COEFF_DOMAIN_CONSTANTS
    };
    let (mul32x32, mul32x16) = if use_pixel_domain {
        (1.0_f32, 1.0_f32)
    } else {
        let k32x32mul1: f32 = -0.75;
        let k32x32mul2: f32 = 1.2;
        let k32x32base: f32 = 2.0;
        let m32 = k32x32mul2 + k32x32mul1 / (distance + k32x32base);

        let k32x16mul1: f32 = -0.70;
        let k32x16mul2: f32 = 1.1;
        let k32x16base: f32 = 2.0;
        let m16 = k32x16mul2 + k32x16mul1 / (distance + k32x16base);
        (m32, m16)
    };

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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
            scratch,
        );
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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
            scratch,
        );
    // Run four 16x16 evaluations (each covers 2×2 blocks).
    // When caching, pass sub-offsets so find_best_16x16_transform stores costs
    // in scratch.entropy_estimate for us to read back (avoiding re-evaluation).
    for qy in (0..4).step_by(2) {
        for qx in (0..4).step_by(2) {
            let sub_cache = cache_offset.map(|(ox, oy)| (ox + qx, oy + qy));
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
                1.0, // aligned pass: no single-block favoritism
                sub_cache,
                profile,
            );
        }
    }

    // Compute per-quadrant sub-costs.
    // When cache_offset is active, read directly from entropy_estimate (populated
    // by find_best_16x16_transform above). Otherwise re-evaluate — needed for
    // non-aligned passes and coefficient-domain mode where cached costs may differ.
    let mut quadrant_cost = [[0.0f32; 2]; 2];
    if let Some((ox, oy)) = cache_offset {
        // Read cached costs: each position holds either a single-block cost or 0
        // (with multi-block cost at the top-left of the covered region).
        for iy in 0..4 {
            for ix in 0..4 {
                quadrant_cost[iy / 2][ix / 2] +=
                    scratch.entropy_estimate[(oy + iy) * 8 + (ox + ix)];
            }
        }
    } else {
        // Re-evaluate sub-costs from ac_strategy map (original path).
        // In pixel-domain mode: 8×8-class costs get mul8x8 (libjxl applies it post-hoc),
        // larger transforms get 1.0 (entropy_mul applied internally).
        // In coefficient-domain mode: distance-dependent multipliers per size class.
        let (sub_mul8x8, sub_mul16x8, sub_mul16x16) = if use_pixel_domain {
            let mul8x8 = 1.0 + (-0.4) / (distance + 1.4);
            (mul8x8, 1.0_f32, 1.0_f32)
        } else {
            let k8x8mul1: f32 = -0.55 * 0.75;
            let k8x8mul2: f32 = 1.073_575_8 * 0.75;
            let k8x8base: f32 = 1.4;
            let m8 = k8x8mul2 + k8x8mul1 / (distance + k8x8base);
            let k8x16mul1: f32 = -0.55;
            let k8x16mul2: f32 = 0.901_958_8;
            let k8x16base: f32 = 1.6;
            let m16x8 = k8x16mul2 + k8x16mul1 / (distance + k8x16base);
            let k16x16mul1: f32 = -0.65;
            let k16x16mul2: f32 = 0.88;
            let k16x16base: f32 = 1.8;
            let m16x16 = k16x16mul2 + k16x16mul1 / (distance + k16x16base);
            (m8, m16x8, m16x16)
        };

        // CRITICAL: Re-evaluation must apply the same entropy_mul adjustments
        // (kFavor2X2, kAvoidEntropyOfTransforms) that were used during the 8x8
        // selection phase. In libjxl these adjustments are baked into the stored
        // entropy_estimate[] values. Without them, DCT2x2/IDENTITY sub-costs are
        // inflated, making merge candidates relatively cheaper and causing
        // over-merging on borderline blocks (e.g. 1025469 d=2.0 hotspot).
        let favor_2x2_adjust = profile.k_favor_2x2 * favor_2x2_weight(distance);
        let avoid_transforms_adjust =
            profile.k_avoid_transforms_base * avoid_entropy_of_transforms_mul(distance);

        for iy in 0..4 {
            for ix in 0..4 {
                if !ac_strategy.is_first(abs_bx + ix, abs_by + iy) {
                    continue;
                }
                let sub_raw = ac_strategy.raw_strategy(abs_bx + ix, abs_by + iy);

                let mul = match sub_raw {
                    RAW_STRATEGY_DCT8 => sub_mul8x8,
                    RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => sub_mul16x8,
                    RAW_STRATEGY_DCT16X16 => sub_mul16x16,
                    _ => sub_mul8x8,
                };
                let base = if !use_pixel_domain && sub_raw == RAW_STRATEGY_DCT8 {
                    3.0 * sub_mul8x8
                } else {
                    0.0
                };

                // Apply the same entropy_mul adjustments as FindBest8x8Transform.
                let adjust = match sub_raw {
                    RAW_STRATEGY_DCT2X2 | RAW_STRATEGY_IDENTITY => favor_2x2_adjust,
                    RAW_STRATEGY_DCT4X8 | RAW_STRATEGY_DCT8X4 | RAW_STRATEGY_DCT4X4
                    | RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2
                    | RAW_STRATEGY_AFV3 => avoid_transforms_adjust,
                    _ => 0.0,
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
                    adjust,
                    scaled_constants,
                    &profile.entropy_mul_table,
                    scratch,
                );
                let cost = base + mul * e;
                quadrant_cost[iy / 2][ix / 2] += cost;
            }
        }
    }

    // Per-half sub-costs matching libjxl's entropy[2][2] sums.
    // Vertical split (DCT32X16 left/right): left = columns 0-1, right = columns 2-3
    let sub_left = quadrant_cost[0][0] + quadrant_cost[1][0];
    let sub_right = quadrant_cost[0][1] + quadrant_cost[1][1];
    // Horizontal split (DCT16X32 top/bottom): top = rows 0-1, bottom = rows 2-3
    let sub_top = quadrant_cost[0][0] + quadrant_cost[0][1];
    let sub_bottom = quadrant_cost[1][0] + quadrant_cost[1][1];
    let cost_sub = sub_left + sub_right;

    // Per-half minimums: libjxl computes costJxN/costNxJ using the better of
    // (rect half, sub half) for each half independently.
    // This means a partially-beneficial rect (one half wins, other doesn't) still
    // contributes its winning half to the total, making the square harder to beat.
    let cost_jxn = entropy_32x16_0.min(sub_left) + entropy_32x16_1.min(sub_right);
    let cost_nxj = entropy_16x32_0.min(sub_top) + entropy_16x32_1.min(sub_bottom);

    debug_rect!(
        "acs/32x32",
        abs_bx * 8,
        abs_by * 8,
        32,
        32,
        "sub={:.0} 32x32={:.0} costJxN={:.0}(32x16: {:.0}+{:.0}) costNxJ={:.0}(16x32: {:.0}+{:.0}) | mul32={:.2} mul16={:.2}",
        cost_sub,
        entropy_32x32,
        cost_jxn,
        entropy_32x16_0,
        entropy_32x16_1,
        cost_nxj,
        entropy_16x32_0,
        entropy_16x32_1,
        mul32x32,
        mul32x16
    );

    // Three-way comparison matching libjxl FindBestFirstLevelDivisionForSquare.
    // The square must beat BOTH rect orientations (which account for partial merges).
    if entropy_32x32 < cost_jxn && entropy_32x32 < cost_nxj {
        // DCT32x32 wins over both rect orientations
        ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT32X32);
        if let Some((ox, oy)) = cache_offset {
            set_entropy_for_transform(&mut scratch.entropy_estimate, ox, oy, 4, 4, entropy_32x32);
        }
        true
    } else if cost_jxn < cost_nxj {
        // Vertical split (DCT32X16) orientation is better — try each half independently.
        let mut any_merged = false;
        if entropy_32x16_0 < sub_left {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT32X16);
            if let Some((ox, oy)) = cache_offset {
                // DCT32X16 covers 2 cols × 4 rows
                set_entropy_for_transform(
                    &mut scratch.entropy_estimate,
                    ox,
                    oy,
                    2,
                    4,
                    entropy_32x16_0,
                );
            }
            any_merged = true;
        }
        if entropy_32x16_1 < sub_right {
            ac_strategy.set(abs_bx + 2, abs_by, RAW_STRATEGY_DCT32X16);
            if let Some((ox, oy)) = cache_offset {
                set_entropy_for_transform(
                    &mut scratch.entropy_estimate,
                    ox + 2,
                    oy,
                    2,
                    4,
                    entropy_32x16_1,
                );
            }
            any_merged = true;
        }
        any_merged
    } else {
        // Horizontal split (DCT16X32) orientation is better — try each half independently.
        let mut any_merged = false;
        if entropy_16x32_0 < sub_top {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT16X32);
            if let Some((ox, oy)) = cache_offset {
                // DCT16X32 covers 4 cols × 2 rows
                set_entropy_for_transform(
                    &mut scratch.entropy_estimate,
                    ox,
                    oy,
                    4,
                    2,
                    entropy_16x32_0,
                );
            }
            any_merged = true;
        }
        if entropy_16x32_1 < sub_bottom {
            ac_strategy.set(abs_bx, abs_by + 2, RAW_STRATEGY_DCT16X32);
            if let Some((ox, oy)) = cache_offset {
                set_entropy_for_transform(
                    &mut scratch.entropy_estimate,
                    ox,
                    oy + 2,
                    4,
                    2,
                    entropy_16x32_1,
                );
            }
            any_merged = true;
        }
        any_merged
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
    profile: &EffortProfile,
) {
    #[cfg(target_arch = "x86_64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::X64V3Token::summon() {
            find_best_64x64_transform_avx2(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                profile,
            );
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::NeonToken::summon() {
            find_best_64x64_transform_neon(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                profile,
            );
            return;
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::Wasm128Token::summon() {
            find_best_64x64_transform_wasm128(
                token,
                xyb,
                stride,
                bx0,
                by0,
                cx,
                cy,
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
                profile,
            );
            return;
        }
    }
    find_best_64x64_transform_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        profile,
    );
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn find_best_64x64_transform_avx2(
    _token: jxl_simd::X64V3Token,
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
    profile: &EffortProfile,
) {
    find_best_64x64_transform_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        profile,
    );
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn find_best_64x64_transform_neon(
    _token: jxl_simd::NeonToken,
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
    profile: &EffortProfile,
) {
    find_best_64x64_transform_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        profile,
    );
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn find_best_64x64_transform_wasm128(
    _token: jxl_simd::Wasm128Token,
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
    profile: &EffortProfile,
) {
    find_best_64x64_transform_impl(
        xyb,
        stride,
        bx0,
        by0,
        cx,
        cy,
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
        profile,
    );
}

#[inline(always)]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn find_best_64x64_transform_impl(
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
    profile: &EffortProfile,
) {
    // libjxl evaluates all strategies at all distances — no distance gates.
    // The cost model (EstimateEntropy with pixel-domain loss) naturally avoids
    // large transforms when they're not beneficial.

    // In pixel-domain mode, entropy_mul is applied internally by estimate_entropy_with_mask
    // using libjxl's static constants (2.25 for DCT64x64, 2.25 for DCT64x32).
    // In coefficient-domain mode, use distance-dependent multipliers.
    let use_pixel_domain = mask1x1.is_some();
    // Pre-compute scaled constants once (was recomputed per estimate_entropy call)
    let scaled_constants = if use_pixel_domain {
        compute_scaled_constants(
            distance,
            (
                profile.k_info_loss_mul_base,
                profile.k_zeros_mul_base,
                profile.k_cost_delta_base,
            ),
        )
    } else {
        COEFF_DOMAIN_CONSTANTS
    };
    let (mul64x64, mul64x32) = if use_pixel_domain {
        (1.0_f32, 1.0_f32)
    } else {
        let k64x64mul1: f32 = -0.80;
        let k64x64mul2: f32 = 1.3;
        let k64x64base: f32 = 2.5;
        let m64 = k64x64mul2 + k64x64mul1 / (distance + k64x64base);

        let k64x32mul1: f32 = -0.75;
        let k64x32mul2: f32 = 1.2;
        let k64x32base: f32 = 2.5;
        let m32 = k64x32mul2 + k64x32mul1 / (distance + k64x32base);
        (m64, m32)
    };

    let abs_bx = bx0 + cx;
    let abs_by = by0 + cy;

    // Zero the entropy_estimate cache for this 64×64 region.
    // Will be populated by find_best_16x16_transform via find_best_32x32_transform
    // when use_pixel_domain is true.
    scratch.entropy_estimate = [0.0; 64];

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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
            scratch,
        );
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
            scaled_constants,
            &profile.entropy_mul_table,
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
            scaled_constants,
            &profile.entropy_mul_table,
            scratch,
        );
    // Run four 32x32 evaluations (each covers 4×4 blocks).
    // In pixel-domain mode, pass cache offsets so sub-costs are stored in
    // entropy_estimate and we can sum them below without re-evaluation.
    let use_cache = use_pixel_domain;
    for qy in (0..8).step_by(4) {
        for qx in (0..8).step_by(4) {
            let sub_cache = if use_cache { Some((qx, qy)) } else { None };
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
                sub_cache,
                profile,
            );
        }
    }

    // Compute per-quadrant sub-costs.
    // When the cache was populated (pixel-domain mode), read directly from
    // entropy_estimate. Otherwise re-evaluate from the ac_strategy map.
    let mut quadrant_cost = [[0.0f32; 2]; 2];
    if use_cache {
        // Read cached costs: sum all 64 positions, accumulating into quadrants.
        for iy in 0..8 {
            for ix in 0..8 {
                quadrant_cost[iy / 4][ix / 4] += scratch.entropy_estimate[iy * 8 + ix];
            }
        }
    } else {
        // Re-evaluate sub-costs from ac_strategy map (coefficient-domain fallback).
        // Apply the same entropy_mul adjustments libjxl bakes into entropy_estimate[]
        // during FindBest8x8Transform (enc_ac_strategy.cc:585-601).
        let favor_2x2_adjust = profile.k_favor_2x2 * favor_2x2_weight(distance);
        let avoid_transforms_adjust =
            profile.k_avoid_transforms_base * avoid_entropy_of_transforms_mul(distance);

        for iy in 0..8 {
            for ix in 0..8 {
                if !ac_strategy.is_first(abs_bx + ix, abs_by + iy) {
                    continue;
                }
                let sub_raw = ac_strategy.raw_strategy(abs_bx + ix, abs_by + iy);

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

                let adjust = match sub_raw {
                    RAW_STRATEGY_DCT2X2 | RAW_STRATEGY_IDENTITY => favor_2x2_adjust,
                    RAW_STRATEGY_DCT4X8 | RAW_STRATEGY_DCT8X4 | RAW_STRATEGY_DCT4X4
                    | RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2
                    | RAW_STRATEGY_AFV3 => avoid_transforms_adjust,
                    _ => 0.0,
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
                    adjust,
                    scaled_constants,
                    &profile.entropy_mul_table,
                    scratch,
                );
                let cost = base + mul * e;
                quadrant_cost[iy / 4][ix / 4] += cost;
            }
        }
    }

    // Per-half sub-costs.
    // Vertical split (DCT64X32 left/right): left = columns 0-3, right = columns 4-7
    let sub_left = quadrant_cost[0][0] + quadrant_cost[1][0];
    let sub_right = quadrant_cost[0][1] + quadrant_cost[1][1];
    // Horizontal split (DCT32X64 top/bottom): top = rows 0-3, bottom = rows 4-7
    let sub_top = quadrant_cost[0][0] + quadrant_cost[0][1];
    let sub_bottom = quadrant_cost[1][0] + quadrant_cost[1][1];
    let cost_sub = sub_left + sub_right;

    // Per-half minimums matching libjxl FindBestFirstLevelDivisionForSquare.
    let cost_jxn = entropy_64x32_0.min(sub_left) + entropy_64x32_1.min(sub_right);
    let cost_nxj = entropy_32x64_0.min(sub_top) + entropy_32x64_1.min(sub_bottom);

    debug_rect!(
        "acs/64x64",
        abs_bx * 8,
        abs_by * 8,
        64,
        64,
        "sub={:.0} 64x64={:.0} costJxN={:.0}(64x32: {:.0}+{:.0}) costNxJ={:.0}(32x64: {:.0}+{:.0}) | mul64={:.2} mul32={:.2}",
        cost_sub,
        entropy_64x64,
        cost_jxn,
        entropy_64x32_0,
        entropy_64x32_1,
        cost_nxj,
        entropy_32x64_0,
        entropy_32x64_1,
        mul64x64,
        mul64x32
    );

    // Three-way comparison matching libjxl FindBestFirstLevelDivisionForSquare.
    if entropy_64x64 < cost_jxn && entropy_64x64 < cost_nxj {
        ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT64X64);
    } else if cost_jxn < cost_nxj {
        // Vertical split (DCT64X32) — try each half independently.
        if entropy_64x32_0 < sub_left {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT64X32);
        }
        if entropy_64x32_1 < sub_right {
            ac_strategy.set(abs_bx + 4, abs_by, RAW_STRATEGY_DCT64X32);
        }
    } else {
        // Horizontal split (DCT32X64) — try each half independently.
        if entropy_32x64_0 < sub_top {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT32X64);
        }
        if entropy_32x64_1 < sub_bottom {
            ac_strategy.set(abs_bx, abs_by + 4, RAW_STRATEGY_DCT32X64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lock the `kAvoidEntropyOfTransforms` formula against libjxl
    /// `enc_ac_strategy.cc::FindBest8x8Transform` (line 592-601):
    ///
    /// ```cpp
    /// if ((tx.type != DCT && tx.type != DCT2X2 && tx.type != IDENTITY) &&
    ///     butteraugli_target > 4.0) {
    ///   static const float kAvoidEntropyOfTransforms = 0.5;
    ///   float mul = 1.0;
    ///   if (butteraugli_target < 12.0) {
    ///     mul *= (12.0 - 4.0) / (butteraugli_target - 4.0);
    ///   }
    ///   entropy_mul += kAvoidEntropyOfTransforms * mul;
    /// }
    /// ```
    #[test]
    fn test_avoid_entropy_of_transforms_mul_libjxl_parity() {
        // d <= 4.0: penalty is OFF (returns 0)
        assert_eq!(avoid_entropy_of_transforms_mul(0.5), 0.0);
        assert_eq!(avoid_entropy_of_transforms_mul(1.0), 0.0);
        assert_eq!(avoid_entropy_of_transforms_mul(2.0), 0.0);
        assert_eq!(avoid_entropy_of_transforms_mul(4.0), 0.0);

        // 4 < d < 12: penalty multiplier = 8 / (d - 4)
        // Just above 4.0: very large multiplier (4.1 → 80x).
        let v = avoid_entropy_of_transforms_mul(4.1);
        assert!((v - 80.0).abs() < 1e-4, "d=4.1 expected ~80, got {v}");

        // d = 5.0 → 8 / 1 = 8.0
        let v = avoid_entropy_of_transforms_mul(5.0);
        assert!((v - 8.0).abs() < 1e-5, "d=5 expected 8.0, got {v}");

        // d = 6.0 → 8 / 2 = 4.0
        let v = avoid_entropy_of_transforms_mul(6.0);
        assert!((v - 4.0).abs() < 1e-5, "d=6 expected 4.0, got {v}");

        // d = 8.0 → 8 / 4 = 2.0
        let v = avoid_entropy_of_transforms_mul(8.0);
        assert!((v - 2.0).abs() < 1e-5, "d=8 expected 2.0, got {v}");

        // d = 12.0: boundary — formula switches to mul = 1.0
        let v = avoid_entropy_of_transforms_mul(12.0);
        assert!((v - 1.0).abs() < 1e-5, "d=12 expected 1.0, got {v}");

        // d > 12: mul = 1.0
        assert_eq!(avoid_entropy_of_transforms_mul(15.0), 1.0);
        assert_eq!(avoid_entropy_of_transforms_mul(100.0), 1.0);
    }

    /// Lock the `kFavor2X2AtHighQuality` weight against libjxl
    /// `enc_ac_strategy.cc::FindBest8x8Transform` (line 585-590):
    ///
    /// ```cpp
    /// if ((tx.type == DCT2X2 || tx.type == IDENTITY) &&
    ///     butteraugli_target < 5.0) {
    ///   float weight = pow((5.0f - butteraugli_target) / 5.0f, 2.0f);
    ///   entropy_mul -= kFavor2X2AtHighQuality * weight;
    /// }
    /// ```
    #[test]
    fn test_favor_2x2_weight_libjxl_parity() {
        // d = 0: weight = (5/5)^2 = 1.0
        assert!((favor_2x2_weight(0.0) - 1.0).abs() < 1e-6);
        // d = 1: weight = (4/5)^2 = 0.64
        assert!((favor_2x2_weight(1.0) - 0.64).abs() < 1e-6);
        // d = 2.5: weight = (2.5/5)^2 = 0.25
        assert!((favor_2x2_weight(2.5) - 0.25).abs() < 1e-6);
        // d = 5: cutoff — weight = 0
        assert_eq!(favor_2x2_weight(5.0), 0.0);
        // d > 5: weight = 0
        assert_eq!(favor_2x2_weight(7.0), 0.0);
    }

    /// Full libjxl `entropy_mul` adjustment with both penalties combined.
    /// Applied to DCT4X4 (a non-DCT/non-2x2/non-IDENTITY transform) at d=6.0.
    ///
    /// In libjxl entry: `entropy_mul_DCT4X4 = 1.08 / 0.8 = 1.35`.
    /// With kAvoidEntropyOfTransforms at d=6: `+= 0.5 * 4.0 = +2.0`.
    /// Final mul: 1.35 + 2.0 = 3.35.
    #[test]
    fn test_avoid_entropy_dct4x4_at_d6() {
        let base = 1.08_f32 / 0.8;
        let adjust = 0.5 * avoid_entropy_of_transforms_mul(6.0);
        let final_mul = base + adjust;
        assert!((final_mul - 3.35).abs() < 1e-4, "got {final_mul}");
    }
}
