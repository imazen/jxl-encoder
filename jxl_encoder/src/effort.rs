// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Centralized effort-derived encoder decisions.
//!
//! Every effort-gated decision in the encoder reads from an [`EffortProfile`]
//! instead of checking `if effort >= N` inline. Construct once from
//! `(effort, mode)`, then pass to all subsystems.

use crate::api::EncoderMode;
use crate::entropy_coding::lz77::Lz77Method;

/// Per-strategy raw entropy multipliers for the AC strategy cost model.
///
/// These control the relative preference for each transform type in AC strategy
/// selection. Higher values penalize a strategy (making it less likely to be chosen);
/// lower values favor it. The 8x8-class values are normalized by DCT8's value before
/// use, so DCT8 always evaluates at 1.0. Larger transforms use raw values directly.
///
/// Default values match libjxl `enc_ac_strategy.cc:584` (`kTransforms8x8[i].entropy_mul`).
/// Experimental values from libjxl PR #4506 (Jon Sneyers, VarDCT cost tuning).
#[derive(Clone, Debug)]
pub struct EntropyMulTable {
    /// DCT8 base value. All 8x8-class transforms are normalized by this.
    /// Reference: 0.8 (libjxl `enc_ac_strategy.cc:357`, `kTransforms8x8[0].entropy_mul`).
    pub dct8: f32,

    /// DCT4x4 (four 4x4 sub-blocks per 8x8 block).
    /// Reference: 1.08. Experimental: 0.88 (PR #4506, ~19% reduction).
    /// Lowering favors DCT4x4 for textured/detailed regions (screenshots, text).
    pub dct4x4: f32,

    /// DCT4x8 / DCT8x4 (half-block transforms for edges/detail).
    /// Reference: 0.859316 (libjxl `enc_ac_strategy.cc`).
    pub dct4x8: f32,

    /// Identity (pixel copy, no transform).
    /// Reference: 1.0428. Experimental: 0.88 (PR #4506, ~16% reduction).
    /// Lowering favors identity blocks for flat/noisy regions.
    pub identity: f32,

    /// DCT2x2 (2x2 Hadamard-like transform).
    /// Reference: 0.95 (libjxl `enc_ac_strategy.cc`).
    pub dct2x2: f32,

    /// AFV (Adaptive Frequency Variable, corner DCT).
    /// Reference: 0.818. Experimental: 0.75 (PR #4506, ~8% reduction).
    /// Lowering favors AFV for edge blocks with mixed content.
    pub afv: f32,

    /// DCT16x8 / DCT8x16 (larger transforms use raw values, not normalized by DCT8).
    /// Reference: 1.21 (libjxl `enc_ac_strategy.cc`).
    pub dct16x8: f32,

    /// DCT16x16.
    /// Reference: 1.34 (libjxl `enc_ac_strategy.cc`).
    pub dct16x16: f32,

    /// DCT16x32 / DCT32x16.
    /// Reference: 1.49 (libjxl `enc_ac_strategy.cc`).
    pub dct16x32: f32,

    /// DCT32x32.
    /// Reference: 1.48 (libjxl `enc_ac_strategy.cc`).
    pub dct32x32: f32,

    /// DCT64x32 / DCT32x64.
    /// Reference: 2.25 (libjxl `enc_ac_strategy.cc`).
    pub dct64x32: f32,

    /// DCT64x64.
    /// Reference: 2.25 (libjxl `enc_ac_strategy.cc`).
    pub dct64x64: f32,
}

impl EntropyMulTable {
    /// Default values matching libjxl `enc_ac_strategy.cc:584`.
    pub fn reference() -> Self {
        Self {
            dct8: 0.8,
            dct4x4: 1.08,
            dct4x8: 0.859_316_37,
            identity: 1.0428,
            dct2x2: 0.95,
            afv: 0.817_794_9,
            dct16x8: 1.21,
            dct16x16: 1.34,
            dct16x32: 1.49,
            dct32x32: 1.48,
            dct64x32: 2.25,
            dct64x64: 2.25,
        }
    }

    /// Experimental values from libjxl PR #4506 (Jon Sneyers, VarDCT cost tuning).
    ///
    /// Changes vs reference:
    /// - dct4x4: 1.08 → 0.88 (~19% reduction) — favor detail-preserving 4x4 sub-blocks
    /// - identity: 1.0428 → 0.88 (~16% reduction) — favor pixel-copy for flat regions
    /// - afv: 0.818 → 0.75 (~8% reduction) — favor corner DCT for edge blocks
    pub fn experimental() -> Self {
        Self {
            dct4x4: 0.88,
            identity: 0.88,
            afv: 0.75,
            ..Self::reference()
        }
    }
}

/// All effort-derived encoder decisions, centralized.
///
/// Replaces scattered `if effort >= N` checks throughout the codebase.
/// Construct once from (effort, mode, encoding path), pass to all subsystems.
#[derive(Clone, Debug)]
pub struct EffortProfile {
    /// The raw effort level (1–10).
    pub effort: u8,

    // ─── Feature flags ───────────────────────────────────────────────────
    /// Use ANS entropy coding instead of Huffman.
    pub use_ans: bool,
    /// Use two-pass mode with optimized entropy codes.
    pub optimize_codes: bool,
    /// Use custom coefficient ordering (AC scan order from statistics).
    pub custom_orders: bool,
    /// Enable gaborish inverse pre-filter.
    pub gaborish: bool,
    /// Enable pixel-domain loss in AC strategy selection.
    pub pixel_domain_loss: bool,
    /// Enable error diffusion in AC quantization.
    pub error_diffusion: bool,
    /// Enable patches/dictionary detection.
    pub patches: bool,
    /// Enable content-adaptive MA tree learning (modular path).
    pub tree_learning: bool,
    /// Enable LZ77 backward references in entropy coding.
    pub lz77: bool,
    /// LZ77 method when lz77 is enabled.
    pub lz77_method: Lz77Method,
    /// Number of butteraugli quantization loop iterations.
    pub butteraugli_iters: u32,

    // ─── AC strategy search ──────────────────────────────────────────────
    /// Enable adaptive AC strategy selection (multi-block transforms).
    pub ac_strategy_enabled: bool,
    /// Try DCT16x16/DCT16x8/DCT8x16 transforms (multi-block 16x16 merges).
    pub try_dct16: bool,
    /// Try DCT32x32/DCT32x16/DCT16x32 transforms.
    pub try_dct32: bool,
    /// Try DCT64x64/DCT64x32/DCT32x64 transforms.
    pub try_dct64: bool,
    /// Try DCT4x8/DCT8x4/DCT4x4/AFV transforms (effort >= 6 in libjxl).
    pub try_dct4x8_afv: bool,
    /// Enable non-aligned evaluation pass (odd-aligned 16x16 regions).
    pub non_aligned_eval: bool,
    /// Step size for fine-grained AC strategy search on 32x32+ blocks.
    /// 1 = every position (effort 9+), 2 = every other (default).
    pub fine_grained_step: u8,

    // ─── VarDCT pipeline options ──────────────────────────────────────────
    /// Apply pixel-level chromacity adjustments (effort >= 7 in libjxl).
    pub chromacity_adjustment: bool,
    /// Use pair-merge clustering for VarDCT entropy codes (effort >= 9 in libjxl).
    /// When false, uses fast k-means-only clustering.
    pub enhanced_clustering_vardct: bool,
    /// Compute per-block dynamic EPF sharpness (effort >= 6 in libjxl).
    pub epf_dynamic_sharpness: bool,
    /// Recompute CfL map after initial quantization for better estimates (effort >= 7 in libjxl).
    pub cfl_two_pass: bool,
    /// Use Newton's method (perceptual cost model) for CfL fitting (effort >= 7 in libjxl).
    /// When false, uses fast least-squares fitting (quadratic cost, single-pass).
    pub cfl_newton: bool,
    /// Newton finite-difference epsilon for CfL fitting.
    /// Controls second-derivative accuracy. Default 1.0 (libjxl uses 100.0, which oscillates).
    pub cfl_newton_eps: f32,
    /// Maximum Newton iterations for CfL fitting. Default 10 (libjxl uses 20).
    pub cfl_newton_max_iters: usize,

    // ─── Quantization ────────────────────────────────────────────────────
    /// Use adaptive (content-dependent) quant field via InitialQuantField.
    /// When false (effort < 5), uses flat quant field = 0.79/distance.
    /// Matches libjxl enc_heuristics.cc:1097-1128.
    pub use_adaptive_quant: bool,
    /// Enable per-block AdjustQuantBlockAC (effort >= 5 in libjxl).
    pub adjust_quant_ac: bool,
    /// Numerator for the effort-fixed q parameter used in global_scale computation.
    /// libjxl: 0.39 at effort >= 5, 0.79 at effort < 5.
    /// global_scale = 65536 * (initial_q_numerator / distance) / 5.0
    pub initial_q_numerator: f32,
    /// Fixed thresholds for Y channel when adjust_quant_ac is false.
    /// From libjxl enc_group.cc:358.
    pub fixed_thresholds_y: [f32; 4],
    /// Initial thresholds when adjust_quant_ac is true.
    /// From libjxl enc_group.cc:390.
    pub adjust_thresholds: [f32; 4],

    // ─── Cost model constants ────────────────────────────────────────────
    /// kFavor2X2AtHighQuality weight (-0.4 in libjxl).
    /// Applied as `-0.4 * ((5-d)/5)^2` to IDENTITY/DCT2X2 entropy.
    pub k_favor_2x2: f32,
    /// kAvoidEntropyOfTransforms base penalty (0.5 in libjxl).
    pub k_avoid_transforms_base: f32,
    /// Base multiplier for info loss estimation (1.2 in libjxl).
    pub k_info_loss_mul_base: f32,
    /// Base multiplier for zero coefficient cost (9.309 in libjxl).
    pub k_zeros_mul_base: f32,
    /// Base delta for cost model (10.833 in libjxl).
    pub k_cost_delta_base: f32,
    /// Quantization constant (0.765 in libjxl).
    pub k_ac_quant: f32,

    // ─── Coefficient-domain multiplier constants ─────────────────────────
    /// DCT8x8 coefficient-domain multiplier (mul1, mul2, base).
    pub k8x8: (f32, f32, f32),
    /// DCT16x8/8x16 coefficient-domain multiplier.
    pub k16x8: (f32, f32, f32),
    /// DCT16x16 coefficient-domain multiplier.
    pub k16x16: (f32, f32, f32),
    /// DCT4x8/8x4 coefficient-domain multiplier.
    pub k4x8: (f32, f32, f32),
    /// DCT4x4 coefficient-domain multiplier.
    pub k4x4: (f32, f32, f32),

    // ─── Entropy multiplier table ──────────────────────────────────────────
    /// Per-strategy entropy multipliers for AC strategy cost model.
    /// Controls relative preference for each transform type.
    pub entropy_mul_table: EntropyMulTable,

    // ─── Patch encoding ────────────────────────────────────────────────────
    /// Use tree learning for patch reference frame encoding.
    /// When true AND ref frame is large enough (>= 128×128), enables adaptive
    /// prediction in the modular encoder for patch ref frames.
    /// Reference: false (libjxl uses simple Gradient predictor).
    /// Experimental: true at effort >= 7 (PR #4533 style improvement).
    pub patch_ref_tree_learning: bool,

    // ─── RCT selection ───────────────────────────────────────────────────
    /// Number of RCT variants to try (0 = no selection, use YCoCg).
    pub nb_rcts_to_try: u8,

    // ─── WP parameter search ───────────────────────────────────────────────
    /// Number of weighted predictor parameter sets to try (0 = default only).
    /// libjxl: 2 at effort 8 (kKitten), 5 at effort 9+ (kTortoise).
    pub wp_num_param_sets: u8,

    // ─── Tree learning parameters ────────────────────────────────────────
    /// Number of MA tree properties to evaluate.
    pub tree_num_properties: u8,
    /// Maximum quantization buckets per property.
    pub tree_max_buckets: u16,
    /// Base threshold for tree splitting (75 + 14 * speed_tier in libjxl).
    pub tree_threshold_base: f32,
    /// Fixed sample cap for tree learning (0 = use fraction instead).
    pub tree_max_samples_fixed: u32,
    /// Fraction of total pixels to sample (0.0 = use fixed cap).
    pub tree_sample_fraction: f32,
}

impl EffortProfile {
    /// Create an effort profile for lossy (VarDCT) encoding.
    pub fn lossy(effort: u8, mode: EncoderMode) -> Self {
        let effort = effort.clamp(1, 10);
        match mode {
            EncoderMode::Reference => Self::lossy_reference(effort),
            EncoderMode::Experimental => Self::lossy_experimental(effort),
        }
    }

    /// Create an effort profile for lossless (modular) encoding.
    pub fn lossless(effort: u8, mode: EncoderMode) -> Self {
        let effort = effort.clamp(1, 10);
        match mode {
            EncoderMode::Reference => Self::lossless_reference(effort),
            EncoderMode::Experimental => Self::lossless_experimental(effort),
        }
    }

    fn lossy_reference(effort: u8) -> Self {
        let speed_tier = 10u8.saturating_sub(effort);

        Self {
            effort,

            // ── Feature flags ──
            use_ans: effort >= 4,
            optimize_codes: effort >= 4,
            custom_orders: effort >= 4,
            gaborish: effort >= 5,
            pixel_domain_loss: effort >= 5,
            error_diffusion: false, // libjxl accepts param but never uses it
            patches: effort >= 7,
            tree_learning: effort >= 7,
            lz77: effort >= 7,
            lz77_method: match effort {
                0..=7 => Lz77Method::Rle,
                8 => Lz77Method::Greedy,
                _ => Lz77Method::Optimal,
            },
            butteraugli_iters: match effort {
                // libjxl runs FindBestQuantization unconditionally for lossy
                // encoding. Gated at speed_tier <= kKitten (effort >= 8) in libjxl
                // (enc_adaptive_quantization.cc:1282). kDefaultButteraugliIters=2,
                // kMaxButteraugliIters=4 for kTortoise (effort 9+).
                0..=7 => 0,
                8 => 2,
                _ => 4,
            },

            // ── AC strategy search ──
            ac_strategy_enabled: effort >= 5,
            try_dct16: effort >= 5,
            try_dct32: effort >= 5,
            try_dct64: effort >= 7,
            try_dct4x8_afv: effort >= 6,
            non_aligned_eval: effort >= 6,
            fine_grained_step: if effort >= 9 { 1 } else { 2 },

            // ── VarDCT pipeline ──
            chromacity_adjustment: effort >= 7,
            enhanced_clustering_vardct: effort >= 9,
            epf_dynamic_sharpness: effort >= 6,
            cfl_two_pass: effort >= 7,
            cfl_newton: effort >= 7,
            cfl_newton_eps: jxl_simd::NEWTON_EPS_DEFAULT,
            cfl_newton_max_iters: jxl_simd::NEWTON_MAX_ITERS_DEFAULT,

            // ── Quantization ──
            use_adaptive_quant: effort >= 5,
            adjust_quant_ac: effort >= 5,
            initial_q_numerator: if effort >= 5 { 0.39 } else { 0.79 },
            fixed_thresholds_y: [0.56, 0.62, 0.62, 0.62],
            adjust_thresholds: [0.58, 0.64, 0.64, 0.64],

            // ── Cost model constants (from libjxl) ──
            k_favor_2x2: -0.4,
            k_avoid_transforms_base: 0.5,
            k_info_loss_mul_base: 1.2,
            k_zeros_mul_base: 9.308_906,
            k_cost_delta_base: 10.833_273,
            k_ac_quant: 0.765,

            // ── Coefficient-domain multipliers ──
            // Note: k8x8 mul1 has 0.75 factor applied (libjxl enc_ac_strategy.cc:790)
            k8x8: (-0.55 * 0.75, 1.073_575_8 * 0.75, 1.4),
            k16x8: (-0.55, 0.901_958_8, 1.6),
            k16x16: (-0.65, 0.88, 1.8),
            k4x8: (-0.50 * 0.75, 0.88, 1.3),
            k4x4: (-0.45 * 0.75, 0.85, 1.2),

            // ── Entropy multiplier table ──
            entropy_mul_table: EntropyMulTable::reference(),

            // ── Patch encoding ──
            patch_ref_tree_learning: false,

            // ── RCT selection ──
            nb_rcts_to_try: match effort {
                0..=4 => 0,
                5 => 4,
                6 => 5,
                7 => 7,
                8 => 9,
                _ => 19,
            },

            // ── WP parameter search ──
            wp_num_param_sets: match effort {
                0..=7 => 0,
                8 => 2,
                _ => 5,
            },

            // ── Tree learning ──
            tree_num_properties: Self::tree_num_properties_for(effort),
            tree_max_buckets: Self::tree_max_buckets_for(effort),
            tree_threshold_base: 75.0 + 14.0 * speed_tier as f32,
            tree_max_samples_fixed: if effort <= 4 { 65_000 } else { 0 },
            // Effort-scaled nb_repeats matching libjxl PR #4236
            tree_sample_fraction: Self::tree_sample_fraction_for(effort),
        }
    }

    fn lossless_reference(effort: u8) -> Self {
        let speed_tier = 10u8.saturating_sub(effort);

        Self {
            effort,

            // ── Feature flags ──
            use_ans: effort >= 4,
            optimize_codes: effort >= 2,
            custom_orders: effort >= 3,
            gaborish: false,          // N/A for lossless
            pixel_domain_loss: false, // N/A for lossless
            error_diffusion: false,   // N/A for lossless
            patches: effort >= 5,
            tree_learning: effort >= 7,
            lz77: effort >= 7,
            lz77_method: match effort {
                0..=7 => Lz77Method::Rle,
                8 => Lz77Method::Greedy,
                _ => Lz77Method::Optimal,
            },
            butteraugli_iters: 0, // N/A for lossless

            // ── AC strategy (N/A for lossless) ──
            ac_strategy_enabled: false,
            try_dct16: false,
            try_dct32: false,
            try_dct64: false,
            try_dct4x8_afv: false,
            non_aligned_eval: false,
            fine_grained_step: 2,

            // ── VarDCT pipeline (N/A for lossless) ──
            chromacity_adjustment: false,
            enhanced_clustering_vardct: false,
            epf_dynamic_sharpness: false,
            cfl_two_pass: false,
            cfl_newton: false,
            cfl_newton_eps: jxl_simd::NEWTON_EPS_DEFAULT,
            cfl_newton_max_iters: jxl_simd::NEWTON_MAX_ITERS_DEFAULT,

            // ── Quantization (N/A for lossless) ──
            use_adaptive_quant: false,
            adjust_quant_ac: false,
            initial_q_numerator: 0.39,
            fixed_thresholds_y: [0.56, 0.62, 0.62, 0.62],
            adjust_thresholds: [0.58, 0.64, 0.64, 0.64],

            // ── Cost model constants (used for tree learning cost estimates) ──
            k_favor_2x2: -0.4,
            k_avoid_transforms_base: 0.5,
            k_info_loss_mul_base: 1.2,
            k_zeros_mul_base: 9.308_906,
            k_cost_delta_base: 10.833_273,
            k_ac_quant: 0.765,

            // ── Coefficient-domain multipliers (N/A for lossless) ──
            k8x8: (-0.55 * 0.75, 1.073_575_8 * 0.75, 1.4),
            k16x8: (-0.55, 0.901_958_8, 1.6),
            k16x16: (-0.65, 0.88, 1.8),
            k4x8: (-0.50 * 0.75, 0.88, 1.3),
            k4x4: (-0.45 * 0.75, 0.85, 1.2),

            // ── Entropy multiplier table (N/A for lossless, but struct requires it) ──
            entropy_mul_table: EntropyMulTable::reference(),

            // ── Patch encoding ──
            patch_ref_tree_learning: false,

            // ── RCT selection ──
            nb_rcts_to_try: match effort {
                0..=4 => 0,
                5 => 4,
                6 => 5,
                7 => 7,
                8 => 9,
                _ => 19,
            },

            // ── WP parameter search ──
            wp_num_param_sets: match effort {
                0..=7 => 0,
                8 => 2,
                _ => 5,
            },

            // ── Tree learning ──
            tree_num_properties: Self::tree_num_properties_for(effort),
            tree_max_buckets: Self::tree_max_buckets_for(effort),
            tree_threshold_base: 75.0 + 14.0 * speed_tier as f32,
            tree_max_samples_fixed: if effort <= 4 { 65_000 } else { 0 },
            // Effort-scaled nb_repeats matching libjxl PR #4236
            tree_sample_fraction: Self::tree_sample_fraction_for(effort),
        }
    }

    /// Experimental lossy profile with tuning from libjxl PRs and our own improvements.
    ///
    /// Divergences from reference (documented per-field):
    /// - `k_info_loss_mul_base`: 1.2 → 1.3 (PR #4506, +8% pixel-domain loss weight)
    /// - `entropy_mul_table`: PR #4506 values (favor DCT4x4, Identity, AFV)
    /// - `enhanced_clustering_vardct`: enabled at effort >= 7 (was e9+)
    /// - `patch_ref_tree_learning`: true at effort >= 7 (tree learning for patch ref frames)
    fn lossy_experimental(effort: u8) -> Self {
        let mut p = Self::lossy_reference(effort);

        // PR #4506 (Jon Sneyers): +8% weight on pixel-domain loss improves visual quality
        // on detailed content. The info_loss_mul scales the IDCT-domain error term in
        // EstimateEntropy, making the cost model more sensitive to visible artifacts.
        // Reference: 1.2 (libjxl enc_ac_strategy.cc). Experimental: 1.3.
        p.k_info_loss_mul_base = 1.3;

        // PR #4506 entropy multiplier rebalancing: favor small/detail-preserving transforms.
        p.entropy_mul_table = EntropyMulTable::experimental();

        // Pair-merge histogram clustering helps VarDCT at effort 7+ (not just e9+).
        // The ANS header cost savings from merging similar distributions outweigh the
        // slight data cost increase from sharing code tables across contexts.
        if effort >= 7 {
            p.enhanced_clustering_vardct = true;
        }

        // Tree learning for patch reference frames: adapts prediction to packed glyphs
        // instead of using fixed Gradient predictor. Significant on large ref frames
        // (screenshots with many unique patterns). Gated at effort >= 7.
        if effort >= 7 {
            p.patch_ref_tree_learning = true;
        }

        p
    }

    fn lossless_experimental(effort: u8) -> Self {
        Self::lossless_reference(effort)
    }

    fn tree_num_properties_for(effort: u8) -> u8 {
        match effort {
            0..=4 => 3,
            5 => 4,
            6 => 5,
            7 => 7,
            8 => 10,
            // 16 = all properties including group_id.
            // Non-squeeze array has 15 elements, so .min(15) caps correctly.
            // Squeeze array has 16 elements (group_id always included).
            _ => 16,
        }
    }

    /// Effort-scaled pixel sampling fraction for tree learning (libjxl PR #4236).
    fn tree_sample_fraction_for(effort: u8) -> f32 {
        match effort {
            0..=4 => 0.15,
            5 => 0.25,
            6 => 0.35,
            7 => 0.5,
            8 => 0.55,
            _ => 0.65,
        }
    }

    fn tree_max_buckets_for(effort: u8) -> u16 {
        // Matches libjxl enc_modular.cc:556-590 max_property_values by speed_tier.
        match effort {
            0..=4 => 32, // <=Cheetah
            5 => 48,     // Hare
            6 => 64,     // Wombat
            7 => 96,     // Squirrel
            8 => 128,    // Kitten
            _ => 256,    // Tortoise
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lossy_reference_e7() {
        let p = EffortProfile::lossy(7, EncoderMode::Reference);
        assert_eq!(p.effort, 7);
        assert!(p.use_ans);
        assert!(p.optimize_codes);
        assert!(p.custom_orders);
        assert!(p.gaborish);
        assert!(p.pixel_domain_loss);
        assert!(!p.error_diffusion);
        assert!(p.patches);
        assert!(p.lz77);
        assert_eq!(p.lz77_method, Lz77Method::Rle);
        assert_eq!(p.butteraugli_iters, 0); // libjxl gates at speed_tier <= kKitten (e8+)
        assert!(p.ac_strategy_enabled);
        assert!(p.try_dct32);
        assert!(p.try_dct64);
        assert!(p.try_dct4x8_afv); // e6+
        assert!(p.non_aligned_eval);
        assert_eq!(p.fine_grained_step, 2);
        assert!(p.chromacity_adjustment); // e7+
        assert!(!p.enhanced_clustering_vardct); // e9+
        assert!(p.epf_dynamic_sharpness); // e6+
        assert!(p.cfl_two_pass); // e7+
        assert!(p.cfl_newton); // e7+ with pass 2
        assert!(p.use_adaptive_quant);
        assert!(p.adjust_quant_ac);
        assert_eq!(p.initial_q_numerator, 0.39);
        assert_eq!(p.k_favor_2x2, -0.4);
        assert_eq!(p.k_ac_quant, 0.765);
        assert_eq!(p.nb_rcts_to_try, 7);
        assert_eq!(p.wp_num_param_sets, 0); // e8+
        assert_eq!(p.tree_num_properties, 7);
        assert_eq!(p.tree_max_buckets, 96);
    }

    #[test]
    fn test_lossy_reference_e5() {
        let p = EffortProfile::lossy(5, EncoderMode::Reference);
        assert_eq!(p.effort, 5);
        assert!(p.use_ans);
        assert!(p.gaborish);
        assert!(p.pixel_domain_loss);
        assert!(!p.error_diffusion); // e7+
        assert!(!p.patches); // e7+
        assert!(!p.lz77); // e7+
        assert!(p.ac_strategy_enabled);
        assert!(p.try_dct32);
        assert!(!p.try_dct64); // e7+
        assert!(!p.try_dct4x8_afv); // e6+
        assert!(!p.non_aligned_eval); // e6+
        assert!(!p.chromacity_adjustment); // e7+
        assert!(!p.enhanced_clustering_vardct); // e9+
        assert!(!p.epf_dynamic_sharpness); // e6+
        assert!(!p.cfl_two_pass); // e7+
        assert!(!p.cfl_newton); // e7+
        assert!(p.use_adaptive_quant);
        assert!(p.adjust_quant_ac);
        assert_eq!(p.initial_q_numerator, 0.39);
        assert_eq!(p.butteraugli_iters, 0); // libjxl gates at speed_tier <= kKitten (e8+)
        assert_eq!(p.nb_rcts_to_try, 4);
        assert_eq!(p.wp_num_param_sets, 0); // e8+
    }

    #[test]
    fn test_lossy_reference_e9() {
        let p = EffortProfile::lossy(9, EncoderMode::Reference);
        assert_eq!(p.lz77_method, Lz77Method::Optimal);
        assert_eq!(p.butteraugli_iters, 4);
        assert_eq!(p.fine_grained_step, 1);
        assert!(p.enhanced_clustering_vardct); // e9+
        assert_eq!(p.nb_rcts_to_try, 19);
        assert_eq!(p.wp_num_param_sets, 5); // e9+
        assert_eq!(p.tree_num_properties, 16);
        assert_eq!(p.tree_max_buckets, 256);
    }

    #[test]
    fn test_lossy_reference_e8() {
        let p = EffortProfile::lossy(8, EncoderMode::Reference);
        assert_eq!(p.lz77_method, Lz77Method::Greedy);
        assert_eq!(p.butteraugli_iters, 2);
        assert_eq!(p.fine_grained_step, 2);
        assert!(!p.enhanced_clustering_vardct); // e9+
        assert_eq!(p.wp_num_param_sets, 2); // e8
    }

    #[test]
    fn test_lossy_reference_e3() {
        let p = EffortProfile::lossy(3, EncoderMode::Reference);
        assert!(!p.use_ans);
        assert!(!p.optimize_codes);
        assert!(!p.gaborish);
        assert!(!p.ac_strategy_enabled);
        assert!(!p.use_adaptive_quant);
        assert!(!p.adjust_quant_ac);
        assert_eq!(p.initial_q_numerator, 0.79);
    }

    #[test]
    fn test_lossless_reference_e7() {
        let p = EffortProfile::lossless(7, EncoderMode::Reference);
        assert!(p.use_ans);
        assert!(p.tree_learning);
        assert!(p.lz77);
        assert_eq!(p.lz77_method, Lz77Method::Rle);
        assert!(p.patches);
        assert!(!p.gaborish); // N/A
        assert!(!p.pixel_domain_loss); // N/A
        assert!(!p.ac_strategy_enabled); // N/A
    }

    #[test]
    fn test_lossless_reference_e4() {
        let p = EffortProfile::lossless(4, EncoderMode::Reference);
        assert!(p.use_ans);
        assert!(!p.tree_learning); // e7+
        assert!(!p.lz77); // e7+
        assert!(!p.patches); // e5+
    }

    #[test]
    fn test_effort_clamp() {
        let p = EffortProfile::lossy(0, EncoderMode::Reference);
        assert_eq!(p.effort, 1);
        let p = EffortProfile::lossy(99, EncoderMode::Reference);
        assert_eq!(p.effort, 10);
    }

    #[test]
    fn test_experimental_diverges_from_reference() {
        // Experimental should share effort/feature-flag structure with reference
        for effort in 1..=10 {
            let r = EffortProfile::lossy(effort, EncoderMode::Reference);
            let e = EffortProfile::lossy(effort, EncoderMode::Experimental);
            assert_eq!(r.effort, e.effort);
            assert_eq!(r.use_ans, e.use_ans);
            assert_eq!(r.k_favor_2x2, e.k_favor_2x2);
            assert_eq!(r.butteraugli_iters, e.butteraugli_iters);
            assert_eq!(r.nb_rcts_to_try, e.nb_rcts_to_try);
        }

        // Verify specific divergences at effort 7
        let r = EffortProfile::lossy(7, EncoderMode::Reference);
        let e = EffortProfile::lossy(7, EncoderMode::Experimental);

        // k_info_loss_mul_base: 1.2 → 1.3 (PR #4506)
        assert_eq!(r.k_info_loss_mul_base, 1.2);
        assert_eq!(e.k_info_loss_mul_base, 1.3);

        // entropy_mul_table: PR #4506 rebalancing
        assert_eq!(r.entropy_mul_table.dct4x4, 1.08);
        assert_eq!(e.entropy_mul_table.dct4x4, 0.88);
        assert_eq!(r.entropy_mul_table.identity, 1.0428);
        assert_eq!(e.entropy_mul_table.identity, 0.88);
        assert_eq!(r.entropy_mul_table.afv, 0.817_794_9);
        assert_eq!(e.entropy_mul_table.afv, 0.75);
        // Unchanged values should match
        assert_eq!(r.entropy_mul_table.dct8, e.entropy_mul_table.dct8);
        assert_eq!(r.entropy_mul_table.dct16x8, e.entropy_mul_table.dct16x8);
        assert_eq!(r.entropy_mul_table.dct32x32, e.entropy_mul_table.dct32x32);

        // enhanced_clustering_vardct: e9+ → e7+ in experimental
        assert!(!r.enhanced_clustering_vardct); // reference e7: off
        assert!(e.enhanced_clustering_vardct); // experimental e7: on

        // patch_ref_tree_learning: false → true at e7+
        assert!(!r.patch_ref_tree_learning);
        assert!(e.patch_ref_tree_learning);

        // At effort 5, experimental should NOT enable the e7+ features
        let e5 = EffortProfile::lossy(5, EncoderMode::Experimental);
        assert!(!e5.enhanced_clustering_vardct);
        assert!(!e5.patch_ref_tree_learning);
        // But should still have the entropy_mul and info_loss_mul changes
        assert_eq!(e5.k_info_loss_mul_base, 1.3);
        assert_eq!(e5.entropy_mul_table.dct4x4, 0.88);
    }

    #[test]
    fn test_entropy_mul_table_reference_values() {
        // Verify all reference values match libjxl enc_ac_strategy.cc:584
        let t = EntropyMulTable::reference();
        assert_eq!(t.dct8, 0.8);
        assert_eq!(t.dct4x4, 1.08);
        assert_eq!(t.dct4x8, 0.859_316_37);
        assert_eq!(t.identity, 1.0428);
        assert_eq!(t.dct2x2, 0.95);
        assert_eq!(t.afv, 0.817_794_9);
        assert_eq!(t.dct16x8, 1.21);
        assert_eq!(t.dct16x16, 1.34);
        assert_eq!(t.dct16x32, 1.49);
        assert_eq!(t.dct32x32, 1.48);
        assert_eq!(t.dct64x32, 2.25);
        assert_eq!(t.dct64x64, 2.25);
    }

    #[test]
    fn test_entropy_mul_table_experimental_values() {
        // Verify PR #4506 changes and that unchanged values are preserved
        let t = EntropyMulTable::experimental();
        let r = EntropyMulTable::reference();

        // Changed values (PR #4506)
        assert_eq!(t.dct4x4, 0.88); // was 1.08
        assert_eq!(t.identity, 0.88); // was 1.0428
        assert_eq!(t.afv, 0.75); // was 0.818

        // Unchanged values
        assert_eq!(t.dct8, r.dct8);
        assert_eq!(t.dct4x8, r.dct4x8);
        assert_eq!(t.dct2x2, r.dct2x2);
        assert_eq!(t.dct16x8, r.dct16x8);
        assert_eq!(t.dct16x16, r.dct16x16);
        assert_eq!(t.dct16x32, r.dct16x32);
        assert_eq!(t.dct32x32, r.dct32x32);
        assert_eq!(t.dct64x32, r.dct64x32);
        assert_eq!(t.dct64x64, r.dct64x64);
    }

    #[test]
    fn test_lossless_experimental_matches_reference() {
        // Lossless experimental is currently identical to reference
        for effort in 1..=10 {
            let r = EffortProfile::lossless(effort, EncoderMode::Reference);
            let e = EffortProfile::lossless(effort, EncoderMode::Experimental);
            assert_eq!(r.effort, e.effort);
            assert_eq!(r.use_ans, e.use_ans);
            assert_eq!(r.tree_learning, e.tree_learning);
            assert_eq!(r.lz77, e.lz77);
        }
    }

    #[test]
    fn test_tree_threshold_base_formula() {
        // speed_tier = 10 - effort
        // threshold = 75 + 14 * speed_tier
        let p = EffortProfile::lossy(7, EncoderMode::Reference);
        assert_eq!(p.tree_threshold_base, 75.0 + 14.0 * 3.0); // speed_tier=3
        let p = EffortProfile::lossy(9, EncoderMode::Reference);
        assert_eq!(p.tree_threshold_base, 75.0 + 14.0 * 1.0); // speed_tier=1
        let p = EffortProfile::lossy(5, EncoderMode::Reference);
        assert_eq!(p.tree_threshold_base, 75.0 + 14.0 * 5.0); // speed_tier=5
    }
}
