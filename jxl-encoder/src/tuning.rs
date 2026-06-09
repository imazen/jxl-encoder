// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! # W44-211 — Canonical access path to every VarDCT tunable constant.
//!
//! This module is the single point of access for every picker-/sweep-tunable
//! numeric constant in the VarDCT encoder. The values themselves still live
//! in their original sites (`vardct/encoder.rs`, `vardct/butteraugli_loop.rs`,
//! `vardct/coeff_order.rs`, etc.) so per-callsite documentation, asserts,
//! and tests stay together with the code that consumes them. This module
//! re-exports each tunable under a stable path so:
//!
//! 1. Future sweep runners can read every tunable from one import path
//!    (`use jxl_encoder::tuning::{discriminator_thresholds, buttloop, ...}`).
//! 2. The [`docs/TUNING_RELATIONS.md`](../../docs/TUNING_RELATIONS.md)
//!    inventory can reference canonical paths.
//! 3. The new opt-in `tuning-override` feature (see [`runtime`]) deserialises
//!    a runtime override struct whose field names mirror the const paths here.
//!
//! ## Production-binary safety
//!
//! Without `--features tuning-override`, this module is purely re-exports
//! — production source still reads every const through its original
//! identifier (e.g. `vardct::encoder::CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD`
//! resolves to the same `pub(crate) const` it always did). Production
//! binaries built from this commit are byte-identical to pre-W44-211
//! (hash-locks 36/36 pass).
//!
//! ## Section structure
//!
//! Mirrors [`memory/w44_210_a_const_inventory`] and the table-of-contents
//! of [`docs/TUNING_RELATIONS.md`](../../docs/TUNING_RELATIONS.md):
//!
//! - [`discriminator_thresholds`] — per-image content discriminator
//!   thresholds (mask/m3/edge_density/fcbr/distance windows)
//! - [`entropy_mul_tables`] — `EntropyMulTable` variant constructors
//!   (re-exported from [`crate::effort`])
//! - [`buttloop`] — buttloop QF seed, EPF sharpness seed, adaptive_quant
//!   QF pre-scale, kPow / max-increase deviation, terminal-class exclude
//! - [`coeff_orders`] — W44-82 / W44-201 / W44-205 cost-gate and per-bucket
//!   skip predicates (gate booleans live in [`crate::gate_registry`])
//! - [`epf`] — per-block sharpness search constants
//! - [`patches`] — patches detection + cost-benefit guards
//! - [`splines`] — spline auto-detection thresholds
//! - [`noise`] — sensor physics constants
//! - [`cfl`] — chroma-from-luma Newton tuning
//! - [`quant_weights`] — parametric DCT quant-weight bands
//! - [`ac_strategy`] — cost-model exponents and channel offsets
//! - [`gates`] — top-level effort/pixel/distance gate constants
//!
//! ## DO NOT
//!
//! - DO NOT change the values here unless you simultaneously update the
//!   corresponding `pub(crate) const` definition in the source-of-truth
//!   file. These are re-exports; the source site owns the value.
//! - DO NOT add new `pub const`s to this module that shadow originals.
//!   Use `pub use crate::vardct::<file>::CONST_NAME;` instead.
//! - DO NOT touch `quant_weights` or `ac_strategy::K_BIAS` / `K_POW_*`
//!   without decoder agreement — these are libjxl-spec values.
//! - DO NOT plumb the [`runtime`] override struct through production
//!   code paths. The override layer is for the sweep runner ONLY; the
//!   default `--features tuning-override` disabled path keeps the const
//!   values bitwise-identical to baseline.

// W44-211: every re-export below is intentional for sweep-runner /
// future picker access. Suppress unused-import warnings for the whole
// tuning module — production code paths still read each const through
// its source-of-truth path, so the re-exports look unused unless a
// sweep-runner binary consumes them.

// ─── Section: per-image content discriminator thresholds ────────────────
// W44-210-A `vardct/encoder.rs` section. Owners: W22-1 / W37-2 / W41-2 /
// W44-29..W44-176 stack. Source-of-truth: `crate::vardct::encoder`.

/// W44-210-A row 1: discriminator thresholds (mask/m3/edge_density/fcbr/distance).
#[allow(unused_imports)]
pub mod discriminator_thresholds {
    // mask1x1 thresholds
    pub(crate) use crate::vardct::encoder::CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD;
    pub(crate) use crate::vardct::encoder::HIGH_D_PHOTO_MIN_DISTANCE;
    pub(crate) use crate::vardct::encoder::HIGH_D_PHOTO_SMOOTH_THRESHOLD;
    pub(crate) use crate::vardct::encoder::HIGH_D_PHOTO_W44_91_MASK_UPPER;
    pub(crate) use crate::vardct::encoder::HIGH_D_PHOTO_W44_91_MAX_DISTANCE;

    // W44-65 DCT-suppress
    pub(crate) use crate::vardct::encoder::W44_65_DCT_SUPPRESS_MEDIAN_THRESHOLD;

    // W44-91 zenanalyze-proxy auto-dispatch (variant Z')
    pub(crate) use crate::vardct::encoder::W44_91_FCBR_MAX;
    pub(crate) use crate::vardct::encoder::W44_91_M3_COLOURFULNESS_MIN;

    // W44-96 narrower sub-gate (variant Z inside W44-29 mask<50)
    pub(crate) use crate::vardct::encoder::W44_96_EDGE_DENSITY_MIN;
    pub(crate) use crate::vardct::encoder::W44_96_FCBR_MAX;
    pub(crate) use crate::vardct::encoder::W44_96_VARIANT_Z_MIN_DISTANCE;

    // W44-98 high/low colour splitter (m3 boundary)
    pub(crate) use crate::vardct::encoder::W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN;

    // W44-124 DCT32 keep gate
    pub(crate) use crate::vardct::encoder::W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE;
    pub(crate) use crate::vardct::encoder::W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE;
    pub(crate) use crate::vardct::encoder::W44_124_DCT32_KEEP_EDGE_DENSITY_MAX;
    pub(crate) use crate::vardct::encoder::W44_124_DCT32_KEEP_M3_MIN;

    // W44-150 / W44-151 / W44-152 / W44-166 / W44-168 / W44-169 photo admission
    pub(crate) use crate::vardct::encoder::W44_150_PHOTO_W44_117_MASK_P25_MIN;
    pub(crate) use crate::vardct::encoder::W44_150_PHOTO_W44_117_MIN_DISTANCE;
    pub(crate) use crate::vardct::encoder::W44_151_HIGH_MASK_P25_MIN;
    pub(crate) use crate::vardct::encoder::W44_152_W44_151_MAX_DISTANCE;
    pub(crate) use crate::vardct::encoder::W44_152_W44_151_MIN_DISTANCE;
    pub(crate) use crate::vardct::encoder::W44_156_VARIANT_Z_D_HIGH_THRESHOLD;
    pub(crate) use crate::vardct::encoder::W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN;
    pub(crate) use crate::vardct::encoder::W44_168_SCREENSHOT_MEDIAN_MIN;
    pub(crate) use crate::vardct::encoder::W44_168_SMOOTH_MASK_P25_MIN;
    pub(crate) use crate::vardct::encoder::W44_168_TEXTURED_EDGE_DENSITY_MIN;
    pub(crate) use crate::vardct::encoder::W44_168_TEXTURED_ITERS_AT_E7;
    pub(crate) use crate::vardct::encoder::W44_169_NARROW_MAX_DISTANCE;
    pub(crate) use crate::vardct::encoder::W44_169_NARROW_MIN_DISTANCE;

    // Top-level dispatch thresholds
    pub(crate) use crate::vardct::encoder::PATCHES_DISPATCH_BLOCK_MASK_THRESHOLD;
    pub(crate) use crate::vardct::encoder::PIXEL_LOSS_DISPATCH_MEDIAN_THRESHOLD;
    pub(crate) use crate::vardct::encoder::SINGLE_PASS_ENTROPY_MAX_DISTANCE;
    pub(crate) use crate::vardct::encoder::SINGLE_PASS_ENTROPY_MAX_EFFORT;
    pub(crate) use crate::vardct::encoder::SINGLE_PASS_ENTROPY_SMOOTH_PHOTO_MAX_MEDIAN;

    /// W44-210-D / W44-211 — shared `mask1x1_p25 >= 85.0` threshold value.
    /// The 4-site duplicate (`W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN`,
    /// `W44_150_PHOTO_W44_117_MASK_P25_MIN`, `W44_151_HIGH_MASK_P25_MIN`,
    /// `W44_168_SMOOTH_MASK_P25_MIN`) is left in place because each site
    /// has independent owner / commit metadata in
    /// `docs/LIBJXL_DIVERGENCES.md`. Use this alias when expressing the
    /// SEMANTIC threshold instead of binding to a specific W44 owner.
    pub const SMART_ZENJXL_PHOTO_MASK_P25_MIN: f32 = 85.0;

    /// W44-210-D / W44-211 — shared `mask1x1_median >= 95.0` threshold value.
    /// The 4-site duplicate (`CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD`,
    /// `buttloop::SCREENSHOT_MEDIAN_THRESHOLD`,
    /// `W44_168_SCREENSHOT_MEDIAN_MIN`, `splines::SCREENSHOT_MEDIAN_MASK_THRESHOLD`)
    /// is left in place because each site has independent owner metadata.
    /// Use this alias when expressing the SEMANTIC screenshot-class
    /// threshold instead of binding to a specific W22 / W44 owner.
    pub const SCREENSHOT_MEDIAN_THRESHOLD: f32 = 95.0;

    // Compile-time assertion that the shared aliases agree with the
    // canonical sites. If a single site diverges, the assertion fires.
    const _: () = assert!(SMART_ZENJXL_PHOTO_MASK_P25_MIN == W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN);
    const _: () = assert!(SMART_ZENJXL_PHOTO_MASK_P25_MIN == W44_150_PHOTO_W44_117_MASK_P25_MIN);
    const _: () = assert!(SMART_ZENJXL_PHOTO_MASK_P25_MIN == W44_151_HIGH_MASK_P25_MIN);
    const _: () = assert!(SMART_ZENJXL_PHOTO_MASK_P25_MIN == W44_168_SMOOTH_MASK_P25_MIN);
    const _: () = assert!(SCREENSHOT_MEDIAN_THRESHOLD == CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD);
    const _: () = assert!(SCREENSHOT_MEDIAN_THRESHOLD == W44_168_SCREENSHOT_MEDIAN_MIN);
    const _: () =
        assert!(SCREENSHOT_MEDIAN_THRESHOLD == super::buttloop::SCREENSHOT_MEDIAN_THRESHOLD);
    const _: () =
        assert!(SCREENSHOT_MEDIAN_THRESHOLD == super::splines::SCREENSHOT_MEDIAN_MASK_THRESHOLD);
}

/// W44-210-A row 2: entropy-mul table variants (per-strategy cost-model
/// multipliers, picker-tunable per content class).
#[allow(unused_imports)]
pub mod entropy_mul_tables {
    pub(crate) use crate::effort::EntropyMulTable;
}

/// W44-210-A row 3: butteraugli loop and adaptive-quant qf seed.
#[allow(unused_imports)]
pub mod buttloop {
    pub(crate) use crate::vardct::perceptual_tuning::ADAPTIVE_QUANT_QF_SEED_SCALE_MAX_EFFORT;
    pub(crate) use crate::vardct::perceptual_tuning::BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX;
    pub(crate) use crate::vardct::perceptual_tuning::BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE;
    pub(crate) use crate::vardct::perceptual_tuning::BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE;
    pub(crate) use crate::vardct::perceptual_tuning::DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6;
    pub(crate) use crate::vardct::perceptual_tuning::DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7;
    pub(crate) use crate::vardct::perceptual_tuning::DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE;
    pub(crate) use crate::vardct::perceptual_tuning::DEFAULT_CUR_POW_HIGH;
    pub(crate) use crate::vardct::perceptual_tuning::DEFAULT_CUR_POW_LOW;
    pub(crate) use crate::vardct::perceptual_tuning::DEFAULT_DISTANCE_SPLIT;
    pub(crate) use crate::vardct::perceptual_tuning::DEFAULT_MAX_INCREASE_HIGH;
    pub(crate) use crate::vardct::perceptual_tuning::DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT;
    pub(crate) use crate::vardct::perceptual_tuning::DEFAULT_MAX_INCREASE_LOW;
    pub(crate) use crate::vardct::perceptual_tuning::LIBJXL_INIT_MUL;
    pub(crate) use crate::vardct::perceptual_tuning::SCREENSHOT_MEDIAN_THRESHOLD;
    pub(crate) use crate::vardct::perceptual_tuning::W44_120_EPF_SEED_MIN_DISTANCE;
    pub(crate) use crate::vardct::perceptual_tuning::W44_140_EPF_SEED_FADE_MAX;
    pub(crate) use crate::vardct::perceptual_tuning::W44_142_EPF_SEED_SUPPRESS_EDGE_DENSITY_MAX;
    pub(crate) use crate::vardct::perceptual_tuning::W44_142_EPF_SEED_SUPPRESS_M3_MIN;
    pub(crate) use crate::vardct::perceptual_tuning::W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE;
    pub(crate) use crate::vardct::perceptual_tuning::W44_145_PER_BLOCK_MASK_HIGH;
    pub(crate) use crate::vardct::perceptual_tuning::W44_145_PER_BLOCK_MASK_LOW;
    pub(crate) use crate::vardct::perceptual_tuning::W44_176_TERMINAL_CLASS_FCBR_MIN;
    pub(crate) use crate::vardct::perceptual_tuning::W44_176_TERMINAL_CLASS_LUMA_VAR_MAX;
    pub(crate) use crate::vardct::perceptual_tuning::W44_176_TERMINAL_CLASS_LUMA_VAR_MIN;
}

/// W44-210-A row 4: coefficient-order cost-gate + per-bucket skip
/// constants. Boolean gates (`coeff_orders_disable_*_buckets`) live in
/// [`crate::gate_registry`].
#[allow(unused_imports)]
pub mod coeff_orders {
    pub(crate) use crate::vardct::coeff_order::NUM_ORDER_BUCKETS;
    pub(crate) use crate::vardct::coeff_order::NUM_PERMUTATION_CONTEXTS;
    pub(crate) use crate::vardct::coeff_order::STRATEGY_TO_BUCKET;
}

/// W44-210-A row 5: EPF sharpness search constants.
#[allow(unused_imports)]
pub mod epf {
    pub(crate) use crate::vardct::epf::EPF_AUTO_SMOOTH_MASK_THRESHOLD;
    pub(crate) use crate::vardct::epf::EPF_BORDER_SAD_MUL;
    pub(crate) use crate::vardct::epf::EPF_CHANNEL_SCALE;
    pub(crate) use crate::vardct::epf::EPF_DEFAULT_SHARPNESS;
    pub(crate) use crate::vardct::epf::EPF_PASS0_SIGMA_SCALE;
    pub(crate) use crate::vardct::epf::EPF_PASS2_SIGMA_SCALE;
    pub(crate) use crate::vardct::epf::EPF_QUANT_MUL;
    pub(crate) use crate::vardct::epf::EPF_SHARP_LUT;
    pub(crate) use crate::vardct::epf::K_INV_SIGMA_NUM;
}

/// W44-210-A row 6: patches detection and cost-benefit guards.
///
/// The imazen-tuned cost-benefit constants
/// (`SAVINGS_BYTES_PER_PIXEL_LOSSLESS` + the fn-local
/// `SAVINGS_BYTES_PER_PIXEL`, `SAFETY_MULTIPLIER`, `SAFETY_DIVISOR`)
/// prevent regressions on non-screenshot content.
#[allow(unused_imports)]
pub mod patches {
    pub(crate) use crate::vardct::patches::BIN_PACKING_SLACKNESS;
    pub(crate) use crate::vardct::patches::CHANNEL_DEQUANT_RGB;
    pub(crate) use crate::vardct::patches::CHANNEL_DEQUANT_XYB;
    pub(crate) use crate::vardct::patches::CHANNEL_WEIGHTS_RGB;
    pub(crate) use crate::vardct::patches::CHANNEL_WEIGHTS_XYB;
    pub(crate) use crate::vardct::patches::DISTANCE_LIMIT;
    pub(crate) use crate::vardct::patches::HAS_SIMILAR_RADIUS;
    pub(crate) use crate::vardct::patches::HAS_SIMILAR_THRESHOLD;
    pub(crate) use crate::vardct::patches::MAX_PATCH_SIZE;
    pub(crate) use crate::vardct::patches::MIN_MAX_PATCH_SIZE;
    pub(crate) use crate::vardct::patches::MIN_PATCH_OCCURRENCES;
    pub(crate) use crate::vardct::patches::MIN_PEAK;
    pub(crate) use crate::vardct::patches::PATCH_SIDE;
    pub(crate) use crate::vardct::patches::SAVINGS_BYTES_PER_PIXEL_LOSSLESS;
    pub(crate) use crate::vardct::patches::SCREENSHOT_FLAT_NEIGHBOR_RATIO;
    pub(crate) use crate::vardct::patches::SIMILAR_THRESHOLD;
    pub(crate) use crate::vardct::patches::VERY_SIMILAR_THRESHOLD;
}

/// W44-210-A row 7: splines auto-detection. The whole submodule is
/// already a pub re-export.
#[allow(unused_imports)]
pub mod splines {
    pub(crate) use crate::vardct::splines::SCREENSHOT_MEDIAN_MASK_THRESHOLD;
    pub(crate) use crate::vardct::splines::detect_params::COST_BENEFIT_MARGIN;
    pub(crate) use crate::vardct::splines::detect_params::INIT_SIGMA;
    pub(crate) use crate::vardct::splines::detect_params::MAX_POLYLINE_LEN;
    pub(crate) use crate::vardct::splines::detect_params::MAX_SPLINES;
    pub(crate) use crate::vardct::splines::detect_params::MIN_BBOX_SPAN_OF_IMAGE_LONG_DIM;
    pub(crate) use crate::vardct::splines::detect_params::MIN_EIG_RATIO;
    pub(crate) use crate::vardct::splines::detect_params::MIN_GRAD_MAG;
    pub(crate) use crate::vardct::splines::detect_params::MIN_POLYLINE_LEN;
    pub(crate) use crate::vardct::splines::detect_params::SIGMA_MAX;
    pub(crate) use crate::vardct::splines::detect_params::SIGMA_MIN;
    pub(crate) use crate::vardct::splines::detect_params::TARGET_CONTROL_POINTS;
}

/// W44-210-A row 8: gaborish sharpening + adaptive params.
#[allow(unused_imports)]
pub mod gaborish {
    pub(crate) use crate::vardct::gaborish::ADAPTIVE_RADIUS;
    pub(crate) use crate::vardct::gaborish::ADAPTIVE_TILE;
    pub(crate) use crate::vardct::gaborish::K_GABORISH;
}

/// W44-210-A row 9: noise synthesis + sensor physics constants.
#[allow(unused_imports)]
pub mod noise {
    pub(crate) use crate::vardct::noise::EFFECTIVE_QUANTUM_EFFICIENCY;
    pub(crate) use crate::vardct::noise::INPUT_REFERRED_READ_NOISE;
    pub(crate) use crate::vardct::noise::NOISE_LUT_MAX;
    pub(crate) use crate::vardct::noise::NOISE_PRECISION;
    pub(crate) use crate::vardct::noise::NUM_NOISE_POINTS;
    pub(crate) use crate::vardct::noise::OPSIN_ABSORBANCE_BIAS_Y;
    pub(crate) use crate::vardct::noise::PHOTO_RESPONSE_NON_UNIFORMITY;
    pub(crate) use crate::vardct::noise::PHOTONS_PER_LX_S_PER_UM2;
    pub(crate) use crate::vardct::noise::SENSOR_AREA_UM2;
}

/// W44-210-A row 10: chroma-from-luma Newton method tuning (lives in
/// the `jxl-encoder-simd` companion crate). The default-path Newton
/// params diverge from libjxl (W44-183 / W44-184) and are gated by
/// `gate_registry::cfl_newton_libjxl_parity`.
#[allow(unused_imports)]
pub mod cfl {
    // JPEG-CfL constants are feature-gated on `jpeg-reencoding`.
    #[cfg(feature = "jpeg-reencoding")]
    pub(crate) use crate::vardct::chroma_from_luma::CFL_FIXED_POINT_PRECISION;
    #[cfg(feature = "jpeg-reencoding")]
    pub(crate) use crate::vardct::chroma_from_luma::DEFAULT_COLOR_FACTOR;
    #[cfg(feature = "jpeg-reencoding")]
    pub(crate) use crate::vardct::chroma_from_luma::JPEG_CFL_ZERO_BIAS_DEFAULT;
    pub(crate) use crate::vardct::chroma_from_luma::K_DISTANCE_MULTIPLIER_AC;
    // K_INV_COLOR_FACTOR appears in both encoder and simd-cfl; re-export
    // the encoder side (the simd side is the same arithmetic).
    pub(crate) use crate::vardct::chroma_from_luma::K_INV_COLOR_FACTOR;
    // Newton-method tuning. `EPS_DEFAULT` / `MAX_ITERS_DEFAULT` is the
    // ZENJXL path; `EPS_LIBJXL` / `MAX_ITERS_LIBJXL` is the
    // bit-exact-libjxl gated path. The simd-side `cfl` module is
    // private; the parent crate re-exports the four NEWTON_* constants
    // at its root.
    pub(crate) use jxl_simd::{
        NEWTON_EPS_DEFAULT, NEWTON_EPS_LIBJXL, NEWTON_MAX_ITERS_DEFAULT, NEWTON_MAX_ITERS_LIBJXL,
    };
    // The remaining 5 inner Newton constants (NEWTON_CLAMP, NEWTON_COEFF,
    // NEWTON_THRES, NEWTON_STABILIZER, NEWTON_CONVERGENCE) live inside the
    // private `jxl_simd::cfl` module and are only consumed within that
    // module's implementation. They're not exposed at the simd crate
    // root; the sweep runner reads them only by editing the source file
    // directly (they're stable bit-exact libjxl values; future picker
    // could re-export at the simd crate root if needed).
}

/// W44-210-A row 11: parametric DCT quant-weight bands. ALL values are
/// libjxl-spec / decoder-mandated; touching them requires decoder
/// agreement. Re-exported here for sweep-runner READ access only.
#[allow(unused_imports)]
pub mod quant_weights {
    pub(crate) use crate::vardct::quant::AFV_FREQS;
    pub(crate) use crate::vardct::quant::AFV_WEIGHTS;
    pub(crate) use crate::vardct::quant::DC_QUANT;
    pub(crate) use crate::vardct::quant::DCT2_WEIGHTS;
    pub(crate) use crate::vardct::quant::DCT4_BAND_PARAMS;
    pub(crate) use crate::vardct::quant::DCT4_LLF_PARAMS;
    pub(crate) use crate::vardct::quant::DCT4X8_BAND_PARAMS;
    pub(crate) use crate::vardct::quant::DCT8_PARAMS;
    pub(crate) use crate::vardct::quant::DCT16X8_PARAMS;
    pub(crate) use crate::vardct::quant::DCT16X16_PARAMS;
    pub(crate) use crate::vardct::quant::DCT16X32_BAND_PARAMS;
    pub(crate) use crate::vardct::quant::DCT32X32_BAND_PARAMS;
    pub(crate) use crate::vardct::quant::DCT32X64_BAND_PARAMS;
    pub(crate) use crate::vardct::quant::DCT64X64_BAND_PARAMS;
    pub(crate) use crate::vardct::quant::IDENTITY_WEIGHTS;
    pub(crate) use crate::vardct::quant::INV_DC_QUANT;
    pub(crate) use crate::vardct::quant::NUM_VALID_STRATEGIES;
}

/// W44-210-A row 12: AC-strategy cost-model exponents + channel offsets.
/// `K_BIAS`, `K_POW_*` are libjxl-spec distance scaling exponents — DO
/// NOT touch as picker targets. The picker tunes the BASE values via
/// [`crate::effort::EffortProfile`], not these exponents.
#[allow(unused_imports)]
pub mod ac_strategy {
    pub(crate) use crate::vardct::ac_strategy::CHANNEL_MUL;
    pub(crate) use crate::vardct::ac_strategy::K_BIAS;
    pub(crate) use crate::vardct::ac_strategy::K_POW_COST_DELTA;
    pub(crate) use crate::vardct::ac_strategy::K_POW_INFO_LOSS;
    pub(crate) use crate::vardct::ac_strategy::K_POW_ZEROS_MUL;
    pub(crate) use crate::vardct::ac_strategy::MASK_CHANNEL_OFFSET;
}

/// W44-210-A row 13: DC tree learning effort gates.
#[allow(unused_imports)]
pub mod dc_tree {
    pub(crate) use crate::vardct::bitstream::DC_TREE_VARIABLE_PREDICTOR_FULL_MIN_EFFORT;
    pub(crate) use crate::vardct::bitstream::DC_TREE_VARIABLE_TRIAL_MIN_EFFORT;
}

/// W44-210-A row 14: top-level effort / pixel-count / distance gates.
#[allow(unused_imports)]
pub mod gates {
    pub(crate) use crate::effort::CONTENT_CLASS_MIN_PIXELS;
    pub(crate) use crate::effort::LARGE_E9_TREE_MAX_BUCKETS;
    pub(crate) use crate::effort::LARGE_IMAGE_PIXEL_THRESHOLD;
    pub(crate) use crate::effort::LOSSY_LOW_DISTANCE_THRESHOLD;
    pub(crate) use crate::effort::LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD;
    pub(crate) use crate::effort::SMALL_IMAGE_PIXEL_THRESHOLD;
}

/// W44-210-A row 15: modular alpha extra-channel squeeze quantizer
/// constants (responsive=1 path on modular alpha).
#[allow(unused_imports)]
pub mod squeeze {
    pub(crate) use crate::vardct::encoder::SQUEEZE_LUMA_FACTOR_CONST;
    pub(crate) use crate::vardct::encoder::SQUEEZE_LUMA_QTABLE;
    pub(crate) use crate::vardct::encoder::SQUEEZE_LUMA_QTABLE_LEN;
    pub(crate) use crate::vardct::encoder::SQUEEZE_QUALITY_FACTOR_CONST;
}

// W44-217: RuntimeTuning parameter-coupling skeleton.
//
// Module-level docs are inside `pub mod coupling { //! ... }` below.
#[allow(dead_code)]
pub mod coupling {
    //! W44-217: empirical coupling structure between the 6 W44-213-wired
    //! [`super::runtime::RuntimeTuning`] fields, derived from numerical
    //! analysis of the W44-216 Stage B sweep corpus
    //! (`s3://zentrain/zenjxl-tuning/2026-05-22/w44-216-stage-b/merged.parquet`,
    //! 4,938 cells × 13 parameter blobs × 27 images × 5 efforts × 7
    //! distances).
    //!
    //! **Each function in this module is a coupling expansion**
    //! that lets a small set of high-level Tier-2 knobs (W44-221+)
    //! drive the full 6-parameter [`super::runtime::RuntimeTuning`]
    //! vector while respecting the empirical interactions.
    //!
    //! **W44-218 status** (shipped 2026-05-22): 7 of 7 ridge fns
    //! implemented as closed-form curves through the production
    //! defaults.
    //!
    //! **W44-222 status** (shipped 2026-05-22): [`Tier2Knobs`] gains a 5th
    //! field `buttloop_aq_balance ∈ [-1, +1]` (default 0). The direction is
    //! derived from the orthogonal-complement SVD of W44-221 Phase 2b PC
    //! residuals (captures 76.5 % of weighted-residual variance not spanned
    //! by the 4 W44-218 mechanism-ridges). Mechanism: rebalance screenshot
    //! quant budget along `(p3, p5)` ↓ + `p6` ↑ (matches W44-217 §6 PC2
    //! "buttloop-vs-AQ-balance" narrative). Validated by re-running the
    //! W44-221 Phase 4b Pareto coverage check with a 5-knob 7^5 grid: max
    //! coverage gap on `screen/very_high` **7.86 % → 1.15 %** (-6.71pp);
    //! all 5 strata now PASS the 2pp-max gate; mean coverage gap stays
    //! <0.1 % everywhere. Defaults round-trip byte-exact (`k5 = 0` →
    //! every K5_DIR contribution is 0).
    //!
    //! **W44-221 status** (shipped 2026-05-22): [`Tier2Knobs`] + the
    //! [`Tier2Knobs::expand_to_runtime_tuning`] expander compose the 4
    //! W44-218 ridges into the full 6-param [`super::runtime::RuntimeTuning`]
    //! vector via additive deviation-from-default composition. Defaults
    //! round-trip byte-exact. The W44-221 Phase 2b gradient-SVD found a
    //! natural rank of 4-5 in the joint response surface (4 components
    //! explain 88.3 % of variance; 5 components explain 96.1 %). The 4
    //! W44-218 ridges span 68.5 % of joint gradient variance; the
    //! remaining 31.5 % lives in directions the mechanism-derived ridges
    //! do not span — closed by W44-222 5th knob.
    //!
    //! **W44-220 status** (2026-05-22): per-pair response R² REFIT
    //! attempt on the W44-216+W44-219 combined corpus (267 blobs,
    //! 21× density) HONEST-STOPPED below the 0.5 acceptance gate.
    //! 0 of 7 pairs clear the gate with linear+cross-term forms;
    //! 0 of 7 pairs clear with GBR-pair-only; 3 of 14 (pair, outcome)
    //! cells clear with GBR-all-6-params — all on `log_bytes_resid`
    //! at `class=screen/dist_band=very_high` (exactly R²=0.5009,
    //! shared across p3_p6/p4_p5/p4_p6 because they fit the same
    //! 6-param surface on the same data). The structural ceiling
    //! on the highest-signal stratum is `ssim2 R² ≈ 0.41`,
    //! `log_bytes R² ≈ 0.44` — below the gate even for an
    //! upper-bound non-parametric 6-param GBR. **The algebraic
    //! forms are wrong, not the corpus** — re-derivation queued
    //! as W44-221 (six-knob expansion / per-class formula families
    //! / RD-theoretic derivation / Tier-3 zenanalyze conditioning).
    //!
    //! **Calibration source**: ridge bounds + saturation strengths
    //! come from the W44-216 LHS empirical ranges (13 param blobs,
    //! 27 images, 4938 cells). Per-pair response R² fits were
    //! ATTEMPTED in W44-218 Phase 3 + W44-220 on the densified
    //! corpus — neither attempt cleared the 0.5 acceptance gate.
    //! The ridges are therefore calibrated to (a) round-trip
    //! defaults byte-exact and (b) cover the empirical envelope.
    //! See `docs/PARAM_INTERACTIONS.md` "W44-220 status" and
    //! `benchmarks/sweeps/w44-219-densify/analysis/w44_220/README.md`
    //! for the gate-failure measurement.
    //!
    //! **Single source of truth** for the interaction structure:
    //! [`docs/PARAM_INTERACTIONS.md`](../../../docs/PARAM_INTERACTIONS.md).
    //!
    //! ## DO NOT
    //!
    //! - DO NOT implement these without first reading
    //!   `PARAM_INTERACTIONS.md` — the bodies are scope for W44-218+,
    //!   NOT W44-217.
    //! - DO NOT wire these into production. The whole Tier-2 knob layer
    //!   lands as `EncoderStrategy::Zenjxl`-only via W44-225.
    //! - DO NOT remove the `unimplemented!()` body without filling in a
    //!   measured formula derived from a hypothesised mechanism + corpus
    //!   validation (per the math/stats-grounded constraint in
    //!   `memory/zenjxl_mode_design_goal_2026-05-22`).
    //! - DO NOT cite "FMA precision" for any deviation between a coupling
    //!   prediction and a measured outcome (binding user directive
    //!   2026-05-19).
    //!
    //! ## Empirical findings (zenjxl subset, 2,475 rows)
    //!
    //! ANOVA decomposition (R² 0.71-0.83 across 5 outcomes) shows:
    //! - **The 6 parameters explain 17-42 % of total variance** on
    //!   `cvvdp`, `ssim2`, `encoded_bytes`, `encode_ms` per outcome.
    //! - **Pairwise interaction terms dominate the main effects** —
    //!   single-param main effects are 0.3-5.4 % of variance, but the
    //!   pairwise terms reach 10-22 % each (top: `z_p1 × z_p2` at
    //!   20-22 %; `z_p3 × z_p6` at 9-11 %; `z_p4 × z_p5` at 8-10 %).
    //! - **Marginal PDPs are ADDITIVE** when integrated over the full
    //!   corpus, but **conditional PDPs on `class=screen` + high
    //!   distance** show strong SUPPRESSIVE / SYNERGISTIC shapes.
    //!
    //! Therefore: the couplings below are **conditional couplings** —
    //! the cross-term coefficient depends on which content-class /
    //! effort / distance stratum the cell belongs to. The Tier-2 knob
    //! layer (W44-221+) will model this via a small per-stratum mixing
    //! formula whose coefficients come from
    //! [`docs/PARAM_INTERACTIONS.md`](../../docs/PARAM_INTERACTIONS.md).
    //!
    //! ## Strata
    //!
    //! Content class is the dominant stratifier (W44-91/96/166 dispatch
    //! pattern): `screen` (mask_median > ~5000, fcbr > ~0.5) vs `photo`.
    //! Within each class, distance bands `{low: d<1, mid: 1≤d<2,
    //! high: 2≤d<3.5, very_high: d≥3.5}` cluster the cells. The W44-216
    //! corpus covers all 4 distance bands × both content classes ≈ 35-100
    //! cells per stratum.

    /// Discovered coupling shape on a `(param_i, param_j)` pair.
    ///
    /// The classification names are the standard interaction types from
    /// experimental design / ANOVA decomposition. Per the W44-217
    /// analysis:
    /// - **ADDITIVE**: `y(i, j) ≈ a + f(i) + g(j)`. Residual variance
    ///   after subtracting the marginal sums is < 5 % of total. The
    ///   parameters can be tuned INDEPENDENTLY at the Tier-2 layer.
    /// - **MULTIPLICATIVE**: `y(i, j) ≈ a × f(i) × g(j)` (i.e.
    ///   `log(y)` is additive). When both `f` and `g` are mostly
    ///   positive and the outcome scales over multiple orders of
    ///   magnitude (e.g. `encoded_bytes`).
    /// - **GATED**: one param's effect kicks in only when the other
    ///   crosses a threshold. Detection: `|∂y/∂i| at low j` >>
    ///   `|∂y/∂i| at high j` (ratio > 3×). Models the W44-29 /
    ///   W44-96 / W44-150 family — a discriminator gate that opens a
    ///   secondary cost-model lift.
    /// - **SUPPRESSIVE**: cross-term coefficient is negative; the
    ///   joint effect is LESS than the sum of individual effects. The
    ///   classic "two interventions that both target the same
    ///   substrate → diminishing return" pattern. Example: `p5_aq_qf`
    ///   and `p6_aq_qf` both scale screen-class adaptive quant; lifting
    ///   both gives less than the sum of individual lifts.
    /// - **SYNERGISTIC**: cross-term coefficient is positive; the
    ///   joint effect is MORE than the sum. Rarer; appears when two
    ///   params unblock each other (e.g. `p4_butt_min_dist` opens the
    ///   buttloop dispatch, then `p6_aq_qf` modulates it).
    /// - **WEAKLY_COUPLED**: small significant cross term; report value
    ///   but treat as additive at Tier-2.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CouplingClass {
        Additive,
        Multiplicative,
        Gated,
        Suppressive,
        Synergistic,
        WeaklyCoupled,
    }

    /// Per-stratum coupling summary returned by the Tier-2 prediction
    /// fns below. Values come from the W44-217 corpus analysis (see
    /// `docs/PARAM_INTERACTIONS.md`).
    #[derive(Debug, Clone, Copy)]
    pub struct CouplingSummary {
        pub class: CouplingClass,
        /// Normalised cross-term coefficient (β / σ_y on the
        /// `y ~ p_i + p_j + p_i:p_j` regression). Positive =
        /// synergistic, negative = suppressive.
        pub cross_normalized: f32,
        /// Gating ratio: `max(|∂y/∂i| at low j, high j) / min(...)`.
        /// >3 indicates GATED structure.
        pub gating_ratio: f32,
        /// Sample size used to estimate the cross term (W44-216 cells
        /// matching the stratum).
        pub n_samples: u32,
    }

    // ─────────────────────────────────────────────────────────────
    // W44-218 SHIPPED defaults (production-default values, used by
    // every coupling fn to keep the round-trip at knob defaults
    // byte-identical to the corresponding source-of-truth consts).
    // These mirror `RuntimeTuning::default()` field-by-field.
    // ─────────────────────────────────────────────────────────────

    /// W44-218 default of `p1_smart_zenjxl_photo_mask_p25_min`.
    pub const DEFAULT_P1: f32 = 85.0;
    /// W44-218 default of `p2_screenshot_median_threshold`.
    pub const DEFAULT_P2: f32 = 95.0;
    /// W44-218 default of `p3_buttloop_default_screenshot_qf_seed_scale`.
    pub const DEFAULT_P3: f32 = 4.0;
    /// W44-218 default of `p4_buttloop_qf_seed_scale_min_distance`.
    pub const DEFAULT_P4: f32 = 3.5;
    /// W44-218 default of `p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6`.
    pub const DEFAULT_P5: f32 = 2.0;
    /// W44-218 default of `p6_adaptive_quant_screenshot_qf_seed_scale_e7`.
    pub const DEFAULT_P6: f32 = 3.0;

    // ─────────────────────────────────────────────────────────────
    // W44-218 ridge calibration constants (derived from the W44-216
    // LHS empirical ranges + the W44-217 coupling-class diagnoses).
    // ─────────────────────────────────────────────────────────────

    /// p1 upper bound used by [`p1_p2_smoothness_dispatch_ridge`].
    /// Source: W44-216 LHS max `p1 ≈ 192.86`. The lower bound is the
    /// mirror about the default (`2 * 85 - 192.86 ≈ -22.86`); the ridge
    /// is clamped to ≥ 0.0 to keep p1 physically meaningful.
    pub const P1_RIDGE_MAX: f32 = 192.86;
    /// p2 upper bound used by [`p1_p2_smoothness_dispatch_ridge`].
    /// Source: W44-216 LHS max `p2 ≈ 108.15`.
    pub const P2_RIDGE_MAX: f32 = 108.15;
    /// Soft-saturation strength for [`p5_p6_effort_conditional_lift`].
    /// `k_eff = k` for `k ≤ 1.0`; `k_eff = 1 + (k - 1) * 0.8` above.
    /// W44-217 finding: SSIM2 saturates at ~6× combined lift (W44-217
    /// PDP `pdp_p5_aq_qf_e56_x_p6_aq_qf_e7_classscreen_ssim2.png` —
    /// L-shape).
    pub const P5_P6_SATURATION_STRENGTH: f32 = 0.8;
    /// Soft-saturation strength for [`p3_p6_screenshot_qac_lift`].
    /// Stronger than P5_P6 (0.7 < 0.8) because (p3, p6) is the FULL
    /// multiplicative lift on the qac field at e7+ where the
    /// buttloop seed AND the adaptive_quant pre-scale BOTH fire.
    /// Saturation cap kicks in past `a = 1.0`.
    pub const P3_P6_SATURATION_STRENGTH: f32 = 0.7;

    // ─────────────────────────────────────────────────────────────
    // W44-222 SHIPPED: 5th knob `buttloop_aq_balance` direction +
    // scale. Captures the dominant uncovered direction in the
    // orthogonal complement of the W44-218 4-ridge span.
    // ─────────────────────────────────────────────────────────────

    /// W44-222 5th-knob direction vector in 6-param space, derived from
    /// the orthogonal-complement SVD of the W44-221 Phase 2b PC residuals.
    /// Captures **76.5 %** of weighted-residual variance not spanned by
    /// the 4 W44-218 ridges (the dominant uncovered direction).
    ///
    /// Mechanism: rebalance screenshot quant budget — `(p3, p5)` down +
    /// `p6` up = shift from "buttloop seed scale + e5/e6 AQ scale" toward
    /// "e7 AQ scale". Matches the W44-217 §6 PC2 "buttloop-vs-AQ-balance"
    /// narrative.
    ///
    /// Direction (unit vector in 6-param space):
    /// `[p1: -0.148, p2: +0.259, p3: -0.650, p4: 0.000, p5: -0.504, p6: +0.485]`
    ///
    /// Source: `benchmarks/sweeps/w44-219-densify/analysis/w44_222/phase_a_5knob_coverage.log`
    /// (orthogonal-complement SVD; first uncovered direction).
    pub const K5_DIR: [f32; 6] = [-0.1479, 0.2589, -0.6501, 0.0, -0.5035, 0.4848];

    /// W44-222 5th-knob magnitude scale. Chosen so `|k5| = 1` produces
    /// a meaningful but physically bounded deviation: at `k5 = +1`,
    /// p3 → 2.37, p5 → 0.74, p6 → 4.21; at `k5 = -1`, p3 → 5.62,
    /// p5 → 3.26, p6 → 1.79. No param crosses its physical floor when
    /// `|k5| ≤ 1`, and the range spans the W44-216 LHS empirical
    /// envelope on the buttloop-vs-AQ axis.
    pub const K5_SCALE: f32 = 2.5;

    /// W44-218 shared helper: clamp a value to `[lo, hi]` for f32.
    /// Used by ridge fns below to enforce physical-meaning bounds.
    #[inline]
    fn clamp_f32(v: f32, lo: f32, hi: f32) -> f32 {
        if v < lo {
            lo
        } else if v > hi {
            hi
        } else {
            v
        }
    }

    /// W44-218 SHIPPED: p1 (`smart_zenjxl_photo_mask_p25_min`)
    /// × p2 (`screenshot_median_threshold`) ridge through default.
    ///
    /// **Empirical strength (W44-217)**: variance term 19.9 % of
    /// `log(encoded_bytes)`, 18.1 % of `ssim2` — the **strongest pair
    /// in the corpus**. Both params control content-class discriminators
    /// (W44-29 / W44-91 / W44-150 / W44-166 / W44-168). Marginal coupling
    /// is ADDITIVE; conditional on `class=screen`, the cross term is
    /// modest (<0.04). The dominance comes from the routing effect:
    /// images that fall on the boundary between photo and screen
    /// dispatch see large discrete jumps as either threshold moves.
    ///
    /// **Hypothesised mechanism**: SHARED-DISCRIMINATOR. p1 gates W44-166
    /// variant Z admission (`mask_p25 >= p1`); p2 gates the screenshot
    /// family (`mask_median >= p2`). They sweep the photo↔screen routing
    /// boundary jointly. The W44-216 LHS sampler covaried both thresholds
    /// roughly along a positive-slope ridge; the per-image dispatch
    /// decision shifts as either threshold crosses an image's
    /// (mask_p25, mask_median) point.
    ///
    /// **Tier-2 ridge** (W44-218): `smoothness_bias s ∈ [0, 1]`
    /// reparameterises `(p1, p2)` along the linear ridge:
    /// - `p1(s) = DEFAULT_P1 + (P1_RIDGE_MAX - DEFAULT_P1) * (1 - 2s)`
    /// - `p2(s) = DEFAULT_P2 + (P2_RIDGE_MAX - DEFAULT_P2) * (1 - 2s)`
    ///
    /// At `s = 0.5` → `(DEFAULT_P1, DEFAULT_P2) = (85, 95)` (default).
    /// At `s = 0.0` → `(192.86, 108.15)` — loosest (lowest smoothness
    /// bias, admit fewer images to screen path).
    /// At `s = 1.0` → `(~0, ~81.85)` — tightest (most smoothness bias,
    /// admit more images to screen path).
    /// p1 is clamped to ≥ 0.0 (cannot have a negative mask threshold).
    ///
    /// **Validation status (W44-218)**: ridge round-trips byte-exact at
    /// `s = 0.5`. Per-pair response R² (ssim2 ~ f(p1, p2) over the
    /// W44-216 corpus) is below the 0.5 acceptance gate because only
    /// 13 LHS blobs available — the ridge geometry is calibrated from
    /// empirical max bounds, not from a response fit. W44-219 denser
    /// sweep (50+ blobs) will let a per-image response surface be fit
    /// inside this ridge.
    pub fn p1_p2_smoothness_dispatch_ridge(smoothness_bias: f32) -> (f32, f32) {
        let s = smoothness_bias;
        let p1_unclamped = DEFAULT_P1 + (P1_RIDGE_MAX - DEFAULT_P1) * (1.0 - 2.0 * s);
        let p2_unclamped = DEFAULT_P2 + (P2_RIDGE_MAX - DEFAULT_P2) * (1.0 - 2.0 * s);
        // Physical-meaning clamps: mask thresholds must be ≥ 0.0.
        // p2 upper bound at s=0 is P2_RIDGE_MAX; lower bound at s=1
        // is `2 * DEFAULT_P2 - P2_RIDGE_MAX ≈ 81.85` which is well
        // above 0, so the clamp only fires on p1 in extreme cases.
        let p1_lo = (2.0 * DEFAULT_P1 - P1_RIDGE_MAX).max(0.0);
        let p2_lo = (2.0 * DEFAULT_P2 - P2_RIDGE_MAX).max(0.0);
        (
            clamp_f32(p1_unclamped, p1_lo, P1_RIDGE_MAX),
            clamp_f32(p2_unclamped, p2_lo, P2_RIDGE_MAX),
        )
    }

    /// W44-218 SHIPPED: p3 (`buttloop_default_screenshot_qf_seed_scale`)
    /// × p6 (`adaptive_quant_screenshot_qf_seed_scale_e7`) joint-lift
    /// ray with soft saturation cap.
    ///
    /// **Empirical strength (W44-217)**: variance term 9.6 % of
    /// `log(encoded_bytes)`, 8.5 % of `ssim2`. Classification on
    /// `class=screen/dist_band=very_high`: SUPPRESSIVE (cross_norm = −0.148,
    /// p < 0.01, n = 206). Both params multiply the screen-class qac seed
    /// (one fires the buttloop seed at d ≥ p4, the other fires the
    /// adaptive_quant pre-scale at e7). Stacking both lifts gives
    /// diminishing returns because the qac field saturates.
    ///
    /// **Hypothesised mechanism**: SUPPRESSIVE / SATURATION. Both
    /// interventions multiply into the same per-block `qac` field on
    /// screenshot-class blocks. Past a certain joint lift (~6× combined),
    /// the field saturates against the quant matrix dynamic range and
    /// additional scale produces zero quality change but extra entropy
    /// from the now-coarser quantization at every block.
    ///
    /// **Tier-2 ridge** (W44-218): single
    /// `screenshot_quant_aggressiveness a ∈ [0.0, 2.0]` knob:
    /// - `a_eff = a` for `a ≤ 1.0`
    /// - `a_eff = 1 + (a - 1) * P3_P6_SATURATION_STRENGTH` for `a > 1.0`
    /// - `p3(a) = DEFAULT_P3 * a_eff`
    /// - `p6(a) = DEFAULT_P6 * a_eff`
    ///
    /// At `a = 1.0` → `(4.0, 3.0)` (default). At `a = 0.0` → `(0, 0)`
    /// (zenjxl screen lifts disabled — but the original `RuntimeTuning`
    /// fields are physical seed scales, so callers should clamp `a ≥ 0`).
    /// At `a = 2.0` → `(~6.80, ~5.10)` — past the W44-217 saturation
    /// cap, included in the knob range so callers can experiment.
    ///
    /// **Validation status (W44-218)**: ridge round-trips byte-exact at
    /// `a = 1.0`. Per-pair response R² (ssim2 ~ f(p3, p6) over the W44-216
    /// `class=screen/dist_band=very_high` stratum) is below the 0.5
    /// acceptance gate (best model: ~0.08). The saturation strength
    /// (`0.7`) is calibrated from the empirical p3/p6 distribution in
    /// the W44-216 LHS top-blobs (mean (p3, p6) of the top-3
    /// best-ssim2 blobs ≈ (5.4, 4.0) = `1.35 × default`, consistent
    /// with `a_eff ≈ 1.25` at `a = 1.5`).
    ///
    /// **Co-coordination note**: this ridge ALSO modifies `p6`. The
    /// W44-222 expander composes by averaging the `p6` values returned
    /// by this fn and [`p5_p6_effort_conditional_lift`] (or by exposing
    /// a separate `e7_lift_balance` knob; current default is averaging).
    pub fn p3_p6_screenshot_qac_lift(screenshot_quant_aggressiveness: f32) -> (f32, f32) {
        let a = screenshot_quant_aggressiveness;
        let a_eff = if a <= 1.0 {
            a
        } else {
            1.0 + (a - 1.0) * P3_P6_SATURATION_STRENGTH
        };
        (DEFAULT_P3 * a_eff, DEFAULT_P6 * a_eff)
    }

    /// **PROPOSED** (W44-218): p4 (`buttloop_qf_seed_scale_min_distance`)
    /// × p5 (`adaptive_quant_screenshot_qf_seed_scale_e5_e6`) coupling
    /// on ssim2.
    ///
    /// **Empirical strength (W44-217)**: variance term 9.0 % of
    /// `log(encoded_bytes)`, 8.1 % of `ssim2`. p4 is the distance
    /// threshold that opens the buttloop screen lift; p5 is the e5/e6
    /// adaptive_quant scale (a separate cost-model layer). The pair
    /// shows SYNERGISTIC behaviour at `class=screen/dist_band=very_high`
    /// (cross_norm = 0.21 on bytes, ssim2 0.05) — when buttloop is
    /// open AND adaptive_quant is lifted, the two lifts compose because
    /// they target different parts of the encoder pipeline (rate-control
    /// loop vs static qac field).
    ///
    /// **Hypothesised mechanism**: GATED (by p4) → multiplicative
    /// inside the gate (by p5). Lowering p4 opens the buttloop screen
    /// dispatch at more cells; once open, p5 modulates the in-loop qac
    /// scaling.
    ///
    /// **Tier-2 use**: NOT separable into a single knob — these are
    /// genuinely orthogonal. Tier-2 may expose both `buttloop_screen_d_gate`
    /// and `adaptive_quant_aggressiveness` directly, with a per-image
    /// soft-OR rule that prefers buttloop when distance > p4 and
    /// adaptive_quant otherwise.
    pub fn p4_p5_buttloop_vs_adaptive_quant_dispatch(
        buttloop_screen_d_gate: f32,
        adaptive_quant_aggressiveness: f32,
    ) -> (f32, f32) {
        // W44-218 SHIPPED: composition of two orthogonal knobs:
        //   p4 ← buttloop_screen_d_gate (direct expose, clamped to [1.5, 5.0])
        //   p5 ← screen_quant_lift ridge (diagonal w/ soft cap, see
        //         `p5_p6_effort_conditional_lift`)
        //
        // The W44-217 GATED-by-p4 surface is implicitly preserved: at
        // low `buttloop_screen_d_gate`, p4 is small → buttloop fires at
        // more cells → p5's inside-the-gate scaling has more effect.
        // The "soft-OR" rule from the original PROPOSED docstring is
        // a Tier-3 (multi-knob composition) concern, not a Tier-2 fn
        // contract — at Tier-2 the user exposes both knobs.
        //
        // Default at (3.5, 1.0) → (3.5, 2.0) byte-exact.
        let p4 = clamp_f32(buttloop_screen_d_gate, 1.5, 5.5);
        let (p5, _p6) = p5_p6_effort_conditional_lift(adaptive_quant_aggressiveness);
        (p4, p5)
    }

    /// **PROPOSED** (W44-218): p5 (`adaptive_quant_screenshot_qf_seed_scale_e5_e6`)
    /// × p6 (`adaptive_quant_screenshot_qf_seed_scale_e7`) saturation
    /// curve.
    ///
    /// **Empirical strength (W44-217)**: variance term 8.4 % of
    /// `log(encoded_bytes)`, 7.3 % of `ssim2`. Same family — both scale
    /// the screen-class adaptive_quant qac seed at different effort
    /// ranges (p5 for e5/e6, p6 for e7). At `class=screen/effort=8`
    /// the joint surface shows STRONG SATURATION (cross_norm = −0.177
    /// on ssim2): the two scales compose multiplicatively but with a
    /// soft cap because the qac field has a finite dynamic range.
    ///
    /// **Hypothesised mechanism**: MULTIPLICATIVE with SOFT-SATURATION.
    /// Effective screen-class qac lift at any effort = `base × p5^χ × p6^(1-χ)`
    /// where `χ` depends on the effort the current call sees. At e5/e6,
    /// `χ ≈ 1` (only p5 fires); at e7, `χ ≈ 0` (only p6); at e8/e9 the
    /// buttloop seed (p3) takes over. The saturation comes from the
    /// dynamic-range cap.
    ///
    /// **Tier-2 use**: ONE knob "screen_quant_lift" sweeps a diagonal
    /// `(p5, p6) = (k × 2.0, k × 3.0)` where `k ∈ [0.5, 2.0]`. The
    /// libjxl-parity default lives at `k = 1.0`.
    pub fn p5_p6_effort_conditional_lift(screen_quant_lift: f32) -> (f32, f32) {
        // W44-218 SHIPPED: diagonal ridge `(k * 2.0, k * 3.0)` with
        // soft-saturation cap above `k = 1.0`.
        //
        // Mechanism: p5 and p6 BOTH lift the screen-class
        // adaptive_quant qac seed at different effort ranges (p5 at
        // e5/e6, p6 at e7). They compose multiplicatively but the
        // qac field has a finite dynamic range → soft cap past 1.0.
        //
        // Formula:
        //   k_eff = k                              for k ≤ 1.0
        //   k_eff = 1 + (k - 1) * P5_P6_SATURATION_STRENGTH   for k > 1.0
        //   p5(k) = DEFAULT_P5 * k_eff
        //   p6(k) = DEFAULT_P6 * k_eff
        //
        // Defaults: k=1.0 → (2.0, 3.0) byte-exact.
        // At k=0.5 → (1.0, 1.5). At k=2.0 → (3.6, 5.4).
        let k = screen_quant_lift;
        let k_eff = if k <= 1.0 {
            k
        } else {
            1.0 + (k - 1.0) * P5_P6_SATURATION_STRENGTH
        };
        (DEFAULT_P5 * k_eff, DEFAULT_P6 * k_eff)
    }

    /// **PROPOSED** (W44-218): p4 (`buttloop_qf_seed_scale_min_distance`)
    /// × p6 (`adaptive_quant_screenshot_qf_seed_scale_e7`) ssim2 coupling.
    ///
    /// **Empirical strength (W44-217)**: variance term 6.5 % of
    /// `log(encoded_bytes)`, 5.6 % of `ssim2`. The strongest per-stratum
    /// SYNERGISTIC coupling: `class=screen/dist_band=very_high`
    /// cross_norm = 0.256 (p < 0.01, n = 206). The PDP on
    /// `pdp_p4_butt_min_dist_x_p6_aq_qf_e7_classscreen_ssim2.png` shows
    /// a strong GATED-by-p4 shape: at low p4 (buttloop opens early)
    /// AND high p6 (e7 quant scale lifted), ssim2 jumps to peak yellow.
    /// At low p4 + low p6, ssim2 drops to dark purple (worst).
    ///
    /// **Hypothesised mechanism**: GATED-by-p4 → multiplicative inside.
    /// Same pattern as p4_p5 above — p4 opens a gate, p6 modulates
    /// inside. Cumulative cross-coverage (p4 × p5) and (p4 × p6) gives
    /// the buttloop family three orthogonal levers.
    ///
    /// **Tier-2 use**: see [`p4_p5_buttloop_vs_adaptive_quant_dispatch`].
    /// The pair shares the buttloop_screen_d_gate knob with p4_p5.
    pub fn p4_p6_e7_buttloop_synergy(
        screen_quant_lift: f32,
        buttloop_screen_d_gate: f32,
    ) -> (f32, f32) {
        // W44-218 SHIPPED: composition. Shares the same orthogonal
        // knobs as `p4_p5_buttloop_vs_adaptive_quant_dispatch`:
        //   p4 ← buttloop_screen_d_gate (direct)
        //   p6 ← screen_quant_lift ridge (diagonal w/ soft cap)
        //
        // The W44-217 SYNERGISTIC surface at
        // `class=screen/dist_band=very_high` (cross_norm = +0.256,
        // strongest signed coupling) is implicit in the joint
        // structure: low p4 + high p6 → both lifts fire, ssim2 climbs.
        // The Tier-2 user controls both knobs separately.
        //
        // Default at (1.0, 3.5) → (3.5, 3.0) byte-exact.
        let p4 = clamp_f32(buttloop_screen_d_gate, 1.5, 5.5);
        let (_p5, p6) = p5_p6_effort_conditional_lift(screen_quant_lift);
        (p4, p6)
    }

    /// **PROPOSED** (W44-218): p1 (`smart_zenjxl_photo_mask_p25_min`)
    /// × p3 (`buttloop_default_screenshot_qf_seed_scale`) coupling.
    ///
    /// **Empirical strength (W44-217)**: variance term 9.1 % of
    /// `log(encoded_bytes)`. p1 controls W44-166 photo admission to
    /// variant Z; p3 controls the screen buttloop seed. The pair
    /// shows ADDITIVE marginal but moderate per-stratum cross term
    /// (~0.05). The interaction is via the dispatch decision: when
    /// p1 ADMITS a photo to variant Z, p3's screen-class lift is
    /// inactive (different code path). When p1 keeps the photo in
    /// the photo bucket, p3 has no effect (photos don't fire the
    /// screenshot dispatch).
    ///
    /// **Hypothesised mechanism**: STRUCTURALLY MUTUALLY EXCLUSIVE
    /// (XOR-like). The cross term in the corpus is a measurement
    /// artifact of the LHS having no images that fire BOTH paths
    /// simultaneously. Per-image, exactly one of `(p1's variant Z
    /// admit, p3's buttloop lift)` fires.
    ///
    /// **Tier-2 use**: do not couple at Tier-2; these are dispatch-
    /// independent. The "smoothness_bias" knob (from p1_p2) and the
    /// "screenshot_quant_aggressiveness" knob (from p3_p6) cover both.
    pub fn p1_p3_mutually_exclusive_dispatch(smoothness_bias: f32, screen_aggr: f32) -> (f32, f32) {
        // W44-218 SHIPPED: composition. p1 and p3 are STRUCTURALLY
        // MUTUALLY EXCLUSIVE per W44-217 — they fire on disjoint
        // image sets (p1 = photo→variant-Z admit, p3 = screen
        // buttloop seed). Tier-2 exposes them as two independent
        // ridges:
        //   p1 ← smoothness_bias ridge (the p1 component of
        //         p1_p2_smoothness_dispatch_ridge)
        //   p3 ← screenshot_quant_aggressiveness ridge (the p3
        //         component of p3_p6_screenshot_qac_lift)
        //
        // Mutual exclusion is preserved at the encoder dispatch
        // layer (W44-166 vs W44-176/29 are different code paths) —
        // this fn just builds the (p1, p3) values, the encoder picks
        // which one applies per-image.
        //
        // Default at (0.5, 1.0) → (85, 4.0) byte-exact.
        let (p1, _p2) = p1_p2_smoothness_dispatch_ridge(smoothness_bias);
        let (p3, _p6) = p3_p6_screenshot_qac_lift(screen_aggr);
        (p1, p3)
    }

    /// **PROPOSED** (W44-218): p3 × p4 buttloop chained-lift coupling
    /// (photo, high distance).
    ///
    /// **Empirical strength (W44-217)**: at `class=photo/dist_band=very_high`
    /// cross_norm = +0.151 (SYNERGISTIC) on `log(encoded_bytes)`.
    /// Both params are gates/scales for the buttloop screen seed —
    /// when high-distance photos fall onto the W44-176 terminal-class
    /// path, the seed scale (p3) AND the distance gate (p4) compose.
    ///
    /// **Hypothesised mechanism**: PHOTO-CONDITIONAL gate widening.
    /// At high d on terminal-class photos, lowering p4 opens the
    /// dispatch earlier; combined with a larger p3 it pushes more
    /// aggressive seed scaling on a content class that wasn't the
    /// original target.
    ///
    /// **Tier-2 use**: optional. May not need its own knob; falls
    /// out from the (p3, p4) defaults when combined with a
    /// content-class-conditional buttloop_screen_d_gate.
    pub fn p3_p4_photo_high_d_gate(buttloop_screen_d_gate: f32, screen_aggr: f32) -> (f32, f32) {
        // W44-218 SHIPPED: composition. p3 lifts the screen buttloop
        // seed (via `screenshot_quant_aggressiveness`); p4 sets the
        // buttloop distance gate (direct).
        //
        // The W44-217 photo/very_high SYNERGISTIC term (+0.151 on
        // log_bytes, n=521) reflects the W44-176 terminal-class path
        // where photos with high `luma_var + fcbr` fall onto the
        // screen-class lift chain. Tier-2 exposes both knobs; the
        // encoder picks per-image which path applies.
        //
        // Default at (3.5, 1.0) → (4.0, 3.5) byte-exact.
        let (p3, _p6) = p3_p6_screenshot_qac_lift(screen_aggr);
        let p4 = clamp_f32(buttloop_screen_d_gate, 1.5, 5.5);
        (p3, p4)
    }

    /// W44-221 Tier-2 high-level interpretable knob set.
    ///
    /// Each knob is a normalised direction in 6-param
    /// [`super::runtime::RuntimeTuning`] space. The
    /// [`Tier2Knobs::expand_to_runtime_tuning`] expander composes them
    /// additively into the full 6-vector consumed by the production
    /// encoder.
    ///
    /// **Design provenance**: the 4 knobs are exactly the W44-218
    /// per-pair ridge knobs, validated by W44-221 Phase 2b
    /// gradient-SVD on the W44-216+W44-219 combined corpus
    /// (13,991 rows, 267 LHS blobs). Rank-4 explains **88.3 %** of joint
    /// `(ssim2, log_bytes)` response variance to param perturbations;
    /// rank-5 explains 96.1 %. The 4 W44-218 ridges span **68.5 %** of
    /// gradient variance (not orthogonal to the data-driven PCs, but
    /// each ridge is mechanism-derived per W44-217 §6 narratives).
    ///
    /// **Knob meanings** (from W44-217 + W44-218 mechanism analysis):
    /// - `smoothness_bias ∈ [0, 1]` (default 0.5): jointly moves p1
    ///   (`smart_zenjxl_photo_mask_p25_min`) and p2 (`screenshot_median_threshold`)
    ///   along the W44-216 LHS ridge. Lower = admit more images to the
    ///   screen-class dispatch path (W44-29/91/166).
    /// - `screenshot_quant_aggressiveness ∈ [0, 2]` (default 1.0):
    ///   multiplicatively scales p3 (buttloop screen seed) AND p6
    ///   (e7 adaptive_quant screen scale) along a soft-saturating ridge
    ///   (W44-217 SUPPRESSIVE pattern).
    /// - `screen_quant_lift ∈ [0.5, 2.0]` (default 1.0): scales p5
    ///   (e5/e6 adaptive_quant scale) along a soft-saturating ridge.
    ///   When this knob and `screenshot_quant_aggressiveness` BOTH
    ///   move p6 away from default, the contributions sum (additive
    ///   composition).
    /// - `buttloop_screen_d_gate ∈ [1.5, 5.5]` (default 3.5): direct
    ///   p4 (buttloop dispatch distance threshold). Lower = open
    ///   buttloop screen dispatch at more cells.
    /// - `buttloop_aq_balance ∈ [-1, +1]` (default 0.0) **[W44-222]**:
    ///   rebalances screenshot quant budget along the dominant
    ///   uncovered direction (PC2 "buttloop-vs-AQ-balance"). Positive =
    ///   shift weight from `(p3, p5)` (buttloop seed + e5/e6 AQ) toward
    ///   `p6` (e7 AQ); negative = opposite. Direction
    ///   [`K5_DIR`] from orthogonal-complement SVD; scale
    ///   [`K5_SCALE`] = 2.5 (so `|k5| = 1` produces a meaningful but
    ///   bounded deviation that stays inside the W44-216 LHS envelope).
    ///
    /// **Default round-trip contract**: with all 4 knobs at their
    /// documented defaults, `expand_to_runtime_tuning()` MUST return
    /// `RuntimeTuning::default()` byte-for-byte. This preserves the
    /// hash-lock contract (36/36 lossy + 13/13 lossless fixtures
    /// byte-identical when the expander is wired into production).
    ///
    /// **Bounds**: knob values outside the documented ranges are clamped
    /// to the documented range; the underlying ridges have their own
    /// saturation caps.
    ///
    /// Gated on `feature = "tuning-override"` because the return type
    /// `super::runtime::RuntimeTuning` only exists under that feature.
    #[cfg(feature = "tuning-override")]
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Tier2Knobs {
        pub smoothness_bias: f32,
        pub screenshot_quant_aggressiveness: f32,
        pub screen_quant_lift: f32,
        pub buttloop_screen_d_gate: f32,
        /// W44-222: rebalances screenshot quant budget along the dominant
        /// uncovered direction (PC2 "buttloop-vs-AQ-balance"). Range
        /// `[-1, +1]`, default `0.0` (no rebalance — defaults round-trip
        /// byte-exact). Positive = shift weight from `(p3, p5)` toward
        /// `p6`; negative = opposite.
        pub buttloop_aq_balance: f32,
    }

    #[cfg(feature = "tuning-override")]
    impl Default for Tier2Knobs {
        /// Defaults defined so [`Tier2Knobs::expand_to_runtime_tuning`]
        /// round-trips to [`super::runtime::RuntimeTuning::default()`]
        /// byte-for-byte.
        fn default() -> Self {
            Self {
                smoothness_bias: 0.5,
                screenshot_quant_aggressiveness: 1.0,
                screen_quant_lift: 1.0,
                buttloop_screen_d_gate: DEFAULT_P4,
                buttloop_aq_balance: 0.0,
            }
        }
    }

    #[cfg(feature = "tuning-override")]
    impl Tier2Knobs {
        /// Default smoothness_bias (defaults round-trip).
        pub const DEFAULT_SMOOTHNESS_BIAS: f32 = 0.5;
        /// Default screenshot_quant_aggressiveness (defaults round-trip).
        pub const DEFAULT_SCREENSHOT_QUANT_AGGRESSIVENESS: f32 = 1.0;
        /// Default screen_quant_lift (defaults round-trip).
        pub const DEFAULT_SCREEN_QUANT_LIFT: f32 = 1.0;
        /// Default buttloop_screen_d_gate (defaults round-trip == DEFAULT_P4).
        pub const DEFAULT_BUTTLOOP_SCREEN_D_GATE: f32 = DEFAULT_P4;
        /// Default buttloop_aq_balance (defaults round-trip; W44-222).
        pub const DEFAULT_BUTTLOOP_AQ_BALANCE: f32 = 0.0;

        /// Knob range bounds. Values outside the range are clamped.
        pub const SMOOTHNESS_BIAS_MIN: f32 = 0.0;
        pub const SMOOTHNESS_BIAS_MAX: f32 = 1.0;
        pub const SCREENSHOT_QUANT_AGGRESSIVENESS_MIN: f32 = 0.0;
        pub const SCREENSHOT_QUANT_AGGRESSIVENESS_MAX: f32 = 2.0;
        pub const SCREEN_QUANT_LIFT_MIN: f32 = 0.5;
        pub const SCREEN_QUANT_LIFT_MAX: f32 = 2.0;
        pub const BUTTLOOP_SCREEN_D_GATE_MIN: f32 = 1.5;
        pub const BUTTLOOP_SCREEN_D_GATE_MAX: f32 = 5.5;
        /// W44-222 5th-knob range bounds.
        pub const BUTTLOOP_AQ_BALANCE_MIN: f32 = -1.0;
        pub const BUTTLOOP_AQ_BALANCE_MAX: f32 = 1.0;

        /// Expand the 4 high-level knobs into a full
        /// [`super::runtime::RuntimeTuning`] vector via additive composition
        /// of the W44-218 per-pair ridges.
        ///
        /// **Composition rule** (additive deviations from defaults):
        /// for each of the 6 params, we sum the deviations contributed
        /// by each ridge that touches it, then add the result to the
        /// param's default:
        /// ```text
        /// p_i(knobs) = DEFAULT_P_i + Σ_{knob_j touching p_i} (ridge_j(knob_j)_p_i - DEFAULT_P_i)
        /// ```
        ///
        /// The `Σ` over ridges is additive (not multiplicative) so each
        /// knob can be reasoned about in isolation. At all-defaults, every
        /// ridge returns its default → every deviation is 0 → expansion
        /// equals defaults exactly.
        ///
        /// **Per-param contributors**:
        /// - p1 ← `smoothness_bias` (via [`p1_p2_smoothness_dispatch_ridge`])
        ///   + `buttloop_aq_balance` (W44-222 K5_DIR contribution)
        /// - p2 ← `smoothness_bias` (via same ridge)
        ///   + `buttloop_aq_balance` (W44-222 K5_DIR contribution)
        /// - p3 ← `screenshot_quant_aggressiveness` (via [`p3_p6_screenshot_qac_lift`])
        ///   + `buttloop_aq_balance` (W44-222 K5_DIR contribution)
        /// - p4 ← `buttloop_screen_d_gate` (direct, with clamp [1.5, 5.5])
        ///   (W44-222 K5_DIR component on p4 is 0.0 by construction)
        /// - p5 ← `screen_quant_lift` (via [`p5_p6_effort_conditional_lift`])
        ///   + `buttloop_aq_balance` (W44-222 K5_DIR contribution)
        /// - p6 ← `screenshot_quant_aggressiveness` + `screen_quant_lift`
        ///   (BOTH contribute via additive deviation; matches the W44-217
        ///   p3_p6 + p5_p6 SUPPRESSIVE pattern that both ridges touch p6)
        ///   + `buttloop_aq_balance` (W44-222 K5_DIR contribution)
        ///
        /// **W44-222 5th-knob composition**: `buttloop_aq_balance` contributes
        /// an additional `K5_SCALE * k5 * K5_DIR[i]` deviation to each
        /// param `i`. At `k5 = 0` (default), every contribution is 0 →
        /// the byte-exact round-trip is preserved.
        pub fn expand_to_runtime_tuning(self) -> super::runtime::RuntimeTuning {
            // Clamp knob values to documented ranges.
            let smoothness_bias = clamp_f32(
                self.smoothness_bias,
                Self::SMOOTHNESS_BIAS_MIN,
                Self::SMOOTHNESS_BIAS_MAX,
            );
            let screenshot_quant_aggressiveness = clamp_f32(
                self.screenshot_quant_aggressiveness,
                Self::SCREENSHOT_QUANT_AGGRESSIVENESS_MIN,
                Self::SCREENSHOT_QUANT_AGGRESSIVENESS_MAX,
            );
            let screen_quant_lift = clamp_f32(
                self.screen_quant_lift,
                Self::SCREEN_QUANT_LIFT_MIN,
                Self::SCREEN_QUANT_LIFT_MAX,
            );
            let buttloop_screen_d_gate = clamp_f32(
                self.buttloop_screen_d_gate,
                Self::BUTTLOOP_SCREEN_D_GATE_MIN,
                Self::BUTTLOOP_SCREEN_D_GATE_MAX,
            );
            let buttloop_aq_balance = clamp_f32(
                self.buttloop_aq_balance,
                Self::BUTTLOOP_AQ_BALANCE_MIN,
                Self::BUTTLOOP_AQ_BALANCE_MAX,
            );

            // Compute each ridge's per-param contribution.
            let (p1_smooth, p2_smooth) = p1_p2_smoothness_dispatch_ridge(smoothness_bias);
            let (p3_aggr, p6_aggr) = p3_p6_screenshot_qac_lift(screenshot_quant_aggressiveness);
            let (p5_lift, p6_lift) = p5_p6_effort_conditional_lift(screen_quant_lift);

            // W44-222 K5 contribution: scaled direction in 6-param space.
            // At buttloop_aq_balance = 0 (default), every k5_delta term
            // evaluates to 0 → adds 0 to every param → defaults round-trip
            // byte-exact.
            let k5 = buttloop_aq_balance;
            let k5_scale = K5_SCALE * k5;
            let k5_delta_p1 = k5_scale * K5_DIR[0];
            let k5_delta_p2 = k5_scale * K5_DIR[1];
            let k5_delta_p3 = k5_scale * K5_DIR[2];
            // K5_DIR[3] is exactly 0.0 by construction (W44-218 ridge
            // span fully covers the p4 axis via buttloop_screen_d_gate);
            // do not perturb p4.
            let k5_delta_p5 = k5_scale * K5_DIR[4];
            let k5_delta_p6 = k5_scale * K5_DIR[5];

            // Additive composition: deviation from default per ridge,
            // plus the W44-222 K5_DIR contribution.
            // At all-defaults each ridge returns the default for the
            // params it touches → every (val - default) term is 0 →
            // expanded param == default. Same is true for K5 at k5 = 0.
            let p1 = DEFAULT_P1 + (p1_smooth - DEFAULT_P1) + k5_delta_p1;
            let p2 = DEFAULT_P2 + (p2_smooth - DEFAULT_P2) + k5_delta_p2;
            let p3 = DEFAULT_P3 + (p3_aggr - DEFAULT_P3) + k5_delta_p3;
            let p4 = buttloop_screen_d_gate;
            let p5 = DEFAULT_P5 + (p5_lift - DEFAULT_P5) + k5_delta_p5;
            let p6 = DEFAULT_P6 + (p6_aggr - DEFAULT_P6) + (p6_lift - DEFAULT_P6) + k5_delta_p6;

            // p1, p2 must stay non-negative (encoder semantics: a mask
            // threshold cannot be negative; matches the
            // p1_p2_smoothness_dispatch_ridge clamp).
            let p1 = p1.max(0.0);
            let p2 = p2.max(0.0);
            // p3..p6 are physical quant seed scales — must stay > 0.
            let p3 = p3.max(0.0);
            let p5 = p5.max(0.0);
            let p6 = p6.max(0.0);

            super::runtime::RuntimeTuning {
                smart_zenjxl_photo_mask_p25_min: p1,
                screenshot_median_threshold: p2,
                buttloop_default_screenshot_qf_seed_scale: p3,
                buttloop_qf_seed_scale_min_distance: p4,
                adaptive_quant_screenshot_qf_seed_scale_e5_e6: p5,
                adaptive_quant_screenshot_qf_seed_scale_e7: p6,
            }
        }
    }

    /// W44-221 / W44-222 thin functional wrapper around
    /// [`Tier2Knobs::expand_to_runtime_tuning`].
    ///
    /// Retained as a free fn (with the W44-217-original signature) for
    /// backwards-compat with the original W44-222 task spec. New
    /// callers should prefer the struct API
    /// ([`Tier2Knobs::expand_to_runtime_tuning`]) which supports adding
    /// future knobs without an API break.
    ///
    /// Gated on `feature = "tuning-override"` because the return type
    /// `super::runtime::RuntimeTuning` only exists under that feature.
    #[cfg(feature = "tuning-override")]
    pub fn expand_knobs_to_runtime(
        smoothness_bias: f32,
        screen_quant_lift: f32,
        buttloop_screen_d_gate: f32,
    ) -> super::runtime::RuntimeTuning {
        // The original W44-217 signature only carried 3 knobs (it omitted
        // `screenshot_quant_aggressiveness`). For backwards compatibility
        // we treat the missing knob as its default value (1.0) which is
        // the no-op for the p3_p6_screenshot_qac_lift ridge.
        Tier2Knobs {
            smoothness_bias,
            screenshot_quant_aggressiveness: 1.0,
            screen_quant_lift,
            buttloop_screen_d_gate,
            // W44-222: 5th knob defaults to 0.0 (no rebalance — preserves
            // the 3-knob backwards-compat contract).
            buttloop_aq_balance: 0.0,
        }
        .expand_to_runtime_tuning()
    }

    // ─── W44-228b / W44-PHASE4-S2-refit: per-stratum optimal Tier2Knobs lookup (OPT-IN API) ───
    //
    // **W44-PHASE4-S2-refit (2026-05-24)**: lookup rebaked from the
    // post-W44-RECON-DEEP corpus (W44-PHASE4-S1 sweep, 22,770 zenjxl rows).
    // The post-RECON-DEEP encoder (A10 HDR dispatch, A11 XYB→linear FMA +
    // INV_OPSIN, B1/B4 GPU butteraugli, B5 default-ON GPU when compiled,
    // B7 buttloop buffer reuse) shifted optima on ALL 8/8 strata vs the
    // W44-228a baseline (L2 distances 1.27 - 2.55 per stratum).
    //
    // Source-of-truth TSV (current):
    // `benchmarks/sweeps/w44-phase4-s1-recon-deep-revalidate/analysis/per_stratum_optima/per_stratum_optima.tsv`.
    //
    // Original W44-228a TSV (predecessor):
    // `benchmarks/sweeps/w44-219-densify/analysis/w44_228a/per_stratum_optima.tsv`.
    //
    // SHIPPED in W44-228b as OPT-IN: callers must explicitly chain
    // `.with_knobs(Tier2Knobs::auto_for_distance(class, distance))`. The
    // production default (no `.with_knobs(...)`) is byte-identical to
    // pre-W44-228b — every hash-lock fixture passes byte-for-byte.
    // Default-on flip is gated on W44-228c encode-decode validation
    // against the W44-105 SHIP cells (terminal / imac_g3 / codec_wiki
    // e8+ d=4-6 SSIM2 wins). W44-228c1 RULED-OUT default-flip because
    // `screenshot_quant_aggressiveness=0` on screen/very_high still
    // disables the W44-105 lift catastrophically on those cells.
    //
    // **W44-PHASE4-S2 status of W44-105 protection**: screen/very_high
    // still has `screenshot_quant_aggressiveness = 0` in the refit
    // (matches W44-228a) — the SHIP-cell catastrophe regime is
    // unchanged in that stratum. screen/high, screen/mid, screen/low
    // newly have `screenshot_quant_aggressiveness = 0.333` (was 0.0 in
    // W44-228a) — these moves are AWAY from the catastrophe regime,
    // not toward it. No new W44-228c1 violation.
    //
    // **W44-PHASE4-S2-refit-c1 HONEST-STOP (2026-05-24)**: c1 attempted
    // to floor screen/very_high k2 ≥ 0.333 to close the W44-228c1
    // violation S2-validate (`e9054b53`) caught on the opt-in path.
    // Per-knob ablation proved k2 is NOT the catastrophe driver — k1=0
    // (smoothness_bias=0 → p1 saturates at RIDGE_MAX) is. Source
    // reverted; the W44-105 SHIP-cell catastrophe on the opt-in path
    // for screen/very_high remains unfixed; W44-228c1 RULED-OUT default-
    // flip stays in force. See the in-source comment on the
    // ScreenVeryHigh tuple + CLAUDE.md §"W44-PHASE4-S2-refit-c1
    // HONEST-STOP" for the ablation table + follow-on candidates.

    /// W44-228b: distance-band stratification matching the W44-217 / W44-228a
    /// corpus binning convention.
    ///
    /// **Convention** (from `tuning.rs` line ~504 W44-217 + the W44-228a
    /// derivation script): `low: d < 1.0`, `mid: 1.0 ≤ d < 2.0`,
    /// `high: 2.0 ≤ d < 3.5`, `very_high: d ≥ 3.5`. This is the binning
    /// the W44-228a optima TSV was computed against; do not change the
    /// boundaries without re-deriving the lookup table.
    ///
    /// Note: an earlier W44-228b task draft documented an INCORRECT
    /// convention (`{very_high: d<1.5, high: [1.5, 3.0), mid: [3.0, 5.0),
    /// low: d>=5.0}`). The W44-217 / W44-228a convention is the one bound
    /// to the data; this enum and [`Tier2Knobs::default_for_stratum`] are
    /// canonical against the TSV.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ContentStratum {
        ScreenVeryHigh,
        ScreenHigh,
        ScreenMid,
        ScreenLow,
        PhotoVeryHigh,
        PhotoHigh,
        PhotoMid,
        PhotoLow,
    }

    impl ContentStratum {
        /// W44-228b: derive the stratum from a content-class + distance pair.
        ///
        /// Returns `None` for `ImageContentClass::{Unknown, Other}` —
        /// only `Photo` and `Screenshot` map to a stratum (matches the
        /// W44-228a corpus, which only stratified those two classes).
        ///
        /// Distance binning follows the W44-217 / W44-228a convention
        /// (see the [`ContentStratum`] doc). NaN / negative `distance`
        /// inputs are treated as `0.0` → `Low`.
        pub fn from_distance_band(
            class: crate::effort::ImageContentClass,
            distance: f32,
        ) -> Option<Self> {
            use crate::effort::ImageContentClass;
            // Normalise NaN / negative distances to 0 so the comparison
            // chain below always lands in a finite band.
            let d = if distance.is_finite() && distance >= 0.0 {
                distance
            } else {
                0.0
            };
            match class {
                ImageContentClass::Screenshot => Some(if d < 1.0 {
                    Self::ScreenLow
                } else if d < 2.0 {
                    Self::ScreenMid
                } else if d < 3.5 {
                    Self::ScreenHigh
                } else {
                    Self::ScreenVeryHigh
                }),
                ImageContentClass::Photo => Some(if d < 1.0 {
                    Self::PhotoLow
                } else if d < 2.0 {
                    Self::PhotoMid
                } else if d < 3.5 {
                    Self::PhotoHigh
                } else {
                    Self::PhotoVeryHigh
                }),
                // Unknown / Other → no per-stratum mapping.
                _ => None,
            }
        }
    }

    #[cfg(feature = "tuning-override")]
    impl Tier2Knobs {
        /// W44-228b: return the optimal Tier2Knobs for a given
        /// `(content_class × dist_band)` stratum, derived from the W44-228a
        /// 7^5 grid search on the W44-219 densified corpus.
        ///
        /// **Source data**: `benchmarks/sweeps/w44-219-densify/analysis/w44_228a/per_stratum_optima.tsv`
        /// (8 rows, one per stratum). Each row was the knob tuple that
        /// minimised the Pareto coverage gap on that stratum (max_gap_pct
        /// across all candidate knob points). The W44-228a derivation
        /// memo (`memory/w44_228a_per_stratum_optima_2026-05-22.md`)
        /// documents the methodology + 5 caveats.
        ///
        /// **W44-105 caveat**: the W44-228a optimisation corpus did NOT
        /// include the W44-105 SHIP cells (terminal / imac_g3 / codec_wiki
        /// e8+ d=4-6, where W44-105 closed SSIM2 wins via the buttloop
        /// screen seed lift `p3_buttloop_default_screenshot_qf_seed_scale =
        /// 4.0`). Every screen-stratum optimum here sets
        /// `screenshot_quant_aggressiveness = 0.0`, which DISABLES the
        /// W44-105 lift. Callers using this method on screen-class content
        /// at d=4-6 should validate encode-decode roundtrip themselves
        /// against representative cells before deploying. The default-on
        /// flip in production is gated on W44-228c paired encode-decode
        /// bench against the W44-105 SHIP cells.
        ///
        /// **Hash-lock invariant**: this method does NOT change the
        /// production default behaviour. Callers must explicitly opt in
        /// via `.with_knobs(Tier2Knobs::auto_for_distance(...))` (or
        /// `.with_knobs(Tier2Knobs::default_for_stratum(...))`). The
        /// no-opt-in path stays byte-identical to pre-W44-228b.
        pub fn default_for_stratum(stratum: ContentStratum) -> Self {
            // Source: per_stratum_optima.tsv columns (k1..k5).
            //
            // **W44-PHASE4-S2-refit (2026-05-24)**: refit from the
            // post-W44-RECON-DEEP corpus (`benchmarks/sweeps/w44-phase4-s1-recon-deep-revalidate/analysis/per_stratum_optima/per_stratum_optima.tsv`,
            // 22,770 zenjxl rows). ALL 8/8 strata refit; previous
            // W44-228a (commit aed37da8, W44-219-densify corpus) tuples
            // listed in each per-stratum comment as `OLD W44-228a:`.
            //
            // DO NOT EDIT these tuples without re-deriving the table.
            #[allow(clippy::approx_constant)]
            let (k1, k2, k3, k4, k5) = match stratum {
                // ── screen strata ──
                // screen/very_high (n=2112, delta_pp=+37.428, L2=1.841)
                // OLD W44-228a: (0.0, 0.0, 0.5, 1.5, 0.0)
                // NEW W44-PHASE4-S2: (0.0, 0.0, 0.5, 2.167, -0.333)
                // Δ: k4 1.500 → 2.167 (+0.667), k5 0.000 → -0.333 (-0.333)
                //
                // **W44-PHASE4-S2-refit-c1 (2026-05-24) — HONEST-STOP**:
                // attempted to floor k2 at 0.333 (and sweep up to 1.5) to
                // preserve W44-105 SHIP cells. Per-k2 sweep on terminal
                // e8 d=4 found bytes BYTE-IDENTICAL (SHA256 33c32773…)
                // and SSIM2=81.98 at EVERY k2 in {0, 0.333, 0.5, 0.75,
                // 1.0, 1.25, 1.5} — even when expand_to_runtime_tuning
                // produced p3=5.94 (well above DEFAULT_P3=4.0). Single-
                // knob ablation against true-defaults proved: k2 alone
                // is NOT the catastrophe driver; isolated k4=2.167 +
                // defaults is byte-identical to baseline; isolated
                // k3=0.5 + defaults is byte-identical; **isolated k1=0
                // + defaults REPRODUCES the catastrophe** (28185 B /
                // SSIM2 81.98 — same SHA as full S2-refit tuple).
                //
                // Mechanism: `k1 = smoothness_bias = 0` drives
                // `p1 = smart_zenjxl_photo_mask_p25_min` to its RIDGE
                // MAX (~145), which disrupts the W44-91/96/166 photo
                // admit gates inside the screen-class W44-105 chain. The
                // c1 floor on k2 cannot close this because k2 only
                // touches p3/p6 (buttloop seed scale + e7 aq lift), not
                // p1 (the admit-gate threshold).
                //
                // Source REVERTED to the S2-refit raw optimum. Follow-on
                // chunks should bisect a k1 floor (target: k1 ≥ 0.25
                // brings p1 within ±10 of DEFAULT_P1=85, which is the
                // band the W44-105 chain was tuned against) and/or per-
                // stratum opt-in gates that route screen/very_high SHIP
                // cells around the S2-refit lookup. The W44-228c1
                // RULED-OUT default-flip constraint stays in force on
                // the opt-in path until a follow-on closes the cliff.
                // Reference: `/home/lilith/work/zen/jxl-encoder/CLAUDE.md`
                // §"W44-PHASE4-S2-refit-c1 HONEST-STOP" for the full
                // per-cell measurement table and DO-NOT list.
                //
                // **W44-PHASE4-S2-refit-c2 (2026-05-24)**: 8-stratum × 5-knob
                // ablation audit (`benchmarks/w44_phase4_s2_refit_c2_audit_2026-05-24.tsv`)
                // measured the W44-228c1 catastrophe on the W44-105 SHIP cell
                // (terminal e8 d=4): full S2-refit → SSIM2 81.98 (baseline
                // 86.91, Δ -4.93). Per-knob ablation found only k1 effects
                // SSIM2 on this cell (k2/k3/k4/k5 single-default reverts are
                // byte-identical to full s2refit). 2-knob ablation found
                // (k1=0.5, k2=1.0) restores SSIM2 to 87.04 (Δ +0.13 vs
                // baseline). Mechanism: k1=0 drives p1 to ridge max (~145),
                // disrupting W44-91/96/166 photo admit gates the W44-105
                // chain depends on; k2=0 zeros p3 (buttloop seed scale).
                // BOTH must be at defaults for SHIP-cell SSIM2 preservation.
                // c2 fix floors k1=0.5 (default) + k2=1.0 (default) on
                // screen/very_high. k3/k4/k5 retained at S2-refit values
                // (byte-identical no-op on terminal e8 d=4 per ablation;
                // may still affect other screen/very_high cells per sweep
                // data the S2-refit was derived from).
                ContentStratum::ScreenVeryHigh => (
                    0.5,
                    1.0,
                    0.5,
                    2.1666666666666665_f32,
                    -0.3333333333333334_f32,
                ),
                // screen/high (n=2411, delta_pp=+13.739, L2=1.780)
                // OLD W44-228a: (0.0, 0.0, 0.5, 3.5, -0.333)
                // NEW W44-PHASE4-S2: (0.0, 0.333, 0.5, 2.167, -0.667)
                // Δ: k2 0.000 → 0.333 (+0.333), k4 3.500 → 2.167 (-1.333),
                //    k5 -0.333 → -0.667 (-0.333)
                //
                // **W44-PHASE4-S2-refit-c2 (2026-05-24)**: 8-stratum ablation
                // audit found graph e8 d=3 cliffs to SSIM2 82.31 (baseline
                // 87.38, Δ -5.07) at full S2-refit. Single-k1-default ablation
                // recovers SSIM2 to 85.44 (Δ -1.94, still over ±0.5 budget).
                // 2-knob (k1=0.5, k2=1.0) revert recovers SSIM2 to 87.28
                // (Δ -0.10 ✓). Same mechanism as screen/very_high — k1=0
                // disrupts photo admit gates, k2=0.333 zeros p3 too aggressively.
                // c2 fix: floor k1=0.5 + k2=1.0 (defaults); keep k3/k4/k5.
                ContentStratum::ScreenHigh => (
                    0.5,
                    1.0,
                    0.5,
                    2.1666666666666665_f32,
                    -0.6666666666666667_f32,
                ),
                // screen/mid (n=1282, delta_pp=+3.592, L2=1.500)
                // OLD W44-228a: (0.0, 0.0, 0.5, 3.5, 0.0)
                // NEW W44-PHASE4-S2: (0.167, 0.333, 0.5, 4.167, -1.0)
                // Δ: k1 0.000 → 0.167 (+0.167), k2 0.000 → 0.333 (+0.333),
                //    k4 3.500 → 4.167 (+0.667), k5 0.000 → -1.000 (-1.000)
                ContentStratum::ScreenMid => (
                    0.16666666666666666_f32,
                    0.3333333333333333_f32,
                    0.5,
                    4.166666666666666_f32,
                    -1.0,
                ),
                // screen/low (n=1197, delta_pp=+4.729, L2=1.929)
                // OLD W44-228a: (1.0, 0.0, 0.5, 2.167, 0.0)
                // NEW W44-PHASE4-S2: (0.0, 0.333, 0.5, 2.167, -1.0)
                // Δ: k1 1.000 → 0.000 (-1.000), k2 0.000 → 0.333 (+0.333),
                //    k5 0.000 → -1.000 (-1.000)
                ContentStratum::ScreenLow => (
                    0.0,
                    0.3333333333333333_f32,
                    0.5,
                    2.1666666666666665_f32,
                    -1.0,
                ),
                // ── photo strata ──
                // photo/very_high (n=4972, delta_pp=+3.387, L2=1.434)
                // OLD W44-228a: (0.333, 0.0, 0.5, 4.833, -0.667)
                // NEW W44-PHASE4-S2: (0.0, 0.0, 0.5, 2.833, 0.333)
                // Δ: k1 0.333 → 0.000 (-0.333), k4 4.833 → 2.833 (-2.000),
                //    k5 -0.667 → 0.333 (+1.000)
                ContentStratum::PhotoVeryHigh => (
                    0.0,
                    0.0,
                    0.5,
                    2.833333333333333_f32,
                    0.33333333333333326_f32,
                ),
                // photo/high (n=4954, delta_pp=+2.877, L2=1.269)
                // OLD W44-228a: (0.167, 0.0, 1.25, 4.833, -0.667)
                // NEW W44-PHASE4-S2: (0.0, 0.0, 0.5, 3.5, -0.333)
                // Δ: k1 0.167 → 0.000 (-0.167), k3 1.250 → 0.500 (-0.750),
                //    k4 4.833 → 3.500 (-1.333), k5 -0.667 → -0.333 (+0.333)
                ContentStratum::PhotoHigh => (0.0, 0.0, 0.5, 3.5, -0.3333333333333334_f32),
                // photo/mid (n=2476, delta_pp=+0.531, L2=2.550)
                // OLD W44-228a: (1.0, 0.0, 2.0, 2.833, 0.333)
                // NEW W44-PHASE4-S2: (0.0, 0.0, 0.5, 1.5, -1.0)
                // Δ: k1 1.000 → 0.000 (-1.000), k3 2.000 → 0.500 (-1.500),
                //    k4 2.833 → 1.500 (-1.333), k5 0.333 → -1.000 (-1.333)
                ContentStratum::PhotoMid => (0.0, 0.0, 0.5, 1.5, -1.0),
                // photo/low (n=2475, delta_pp=+0.489, L2=2.550)
                // OLD W44-228a: (0.833, 0.0, 0.5, 2.167, 0.667)
                // NEW W44-PHASE4-S2: (0.0, 0.0, 0.5, 1.5, -1.0)
                // Δ: k1 0.833 → 0.000 (-0.833), k4 2.167 → 1.500 (-0.667),
                //    k5 0.667 → -1.000 (-1.667)
                ContentStratum::PhotoLow => (0.0, 0.0, 0.5, 1.5, -1.0),
            };
            Tier2Knobs {
                smoothness_bias: k1,
                screenshot_quant_aggressiveness: k2,
                screen_quant_lift: k3,
                buttloop_screen_d_gate: k4,
                buttloop_aq_balance: k5,
            }
        }

        /// W44-228b convenience: combine [`ContentStratum::from_distance_band`]
        /// + [`Tier2Knobs::default_for_stratum`] in one call.
        ///
        /// For `ImageContentClass::{Unknown, Other}` (where no stratum is
        /// defined), returns `Tier2Knobs::default()` — the universal W44-222
        /// default — so callers can route every encode through this method
        /// without a per-class branch. The default preserves byte-identical
        /// production behaviour (defaults round-trip → no `RuntimeTuning`
        /// install fires).
        ///
        /// **Opt-in semantics**: production default is unchanged unless
        /// the caller explicitly chains `.with_knobs(...)`. See
        /// [`Tier2Knobs::default_for_stratum`] for the W44-105 SHIP-cell
        /// caveat that applies to screen strata.
        pub fn auto_for_distance(class: crate::effort::ImageContentClass, distance: f32) -> Self {
            match ContentStratum::from_distance_band(class, distance) {
                Some(stratum) => Self::default_for_stratum(stratum),
                None => Self::default(),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        //! W44-218 SHIPPED tests: each coupling fn has a measured ridge
        //! implementation. Tests assert:
        //! 1. Defaults round-trip byte-exact (k = k_default → production
        //!    values).
        //! 2. Knob range covers the W44-216 LHS empirical range.
        //! 3. Saturation cap engages where claimed.
        //! 4. Composition fns delegate to the underlying ridges
        //!    correctly.
        //!
        //! The W44-217 `unimplemented!()` marker tests were converted
        //! to round-trip + range assertions when the implementations
        //! shipped.
        //!
        //! W44-221 SHIPPED tests: [`Tier2Knobs`] +
        //! [`Tier2Knobs::expand_to_runtime_tuning`] now have round-trip,
        //! range, additive-composition, single-knob-locality, and
        //! physical-floor assertions (all `tuning-override`-gated).

        use super::*;

        // Equality tolerance: ridge fns are pure-arithmetic on f32 so
        // they're bit-exact at literal-valued knobs (no FMA-precision
        // wiggle). Use a tight tolerance for non-default points.
        const EPS: f32 = 1e-5;

        // ─── default round-trip (the hash-lock contract) ───

        #[test]
        fn p1_p2_smoothness_dispatch_ridge_default_roundtrip() {
            let (p1, p2) = p1_p2_smoothness_dispatch_ridge(0.5);
            assert_eq!(p1, DEFAULT_P1, "p1 default round-trip");
            assert_eq!(p2, DEFAULT_P2, "p2 default round-trip");
        }

        #[test]
        fn p3_p6_screenshot_qac_lift_default_roundtrip() {
            let (p3, p6) = p3_p6_screenshot_qac_lift(1.0);
            assert_eq!(p3, DEFAULT_P3, "p3 default round-trip");
            assert_eq!(p6, DEFAULT_P6, "p6 default round-trip");
        }

        #[test]
        fn p5_p6_effort_conditional_lift_default_roundtrip() {
            let (p5, p6) = p5_p6_effort_conditional_lift(1.0);
            assert_eq!(p5, DEFAULT_P5, "p5 default round-trip");
            assert_eq!(p6, DEFAULT_P6, "p6 default round-trip");
        }

        #[test]
        fn p4_p5_buttloop_vs_adaptive_quant_dispatch_default_roundtrip() {
            let (p4, p5) = p4_p5_buttloop_vs_adaptive_quant_dispatch(3.5, 1.0);
            assert_eq!(p4, DEFAULT_P4, "p4 default round-trip");
            assert_eq!(p5, DEFAULT_P5, "p5 default round-trip");
        }

        #[test]
        fn p4_p6_e7_buttloop_synergy_default_roundtrip() {
            let (p4, p6) = p4_p6_e7_buttloop_synergy(1.0, 3.5);
            assert_eq!(p4, DEFAULT_P4, "p4 default round-trip");
            assert_eq!(p6, DEFAULT_P6, "p6 default round-trip");
        }

        #[test]
        fn p1_p3_mutually_exclusive_dispatch_default_roundtrip() {
            let (p1, p3) = p1_p3_mutually_exclusive_dispatch(0.5, 1.0);
            assert_eq!(p1, DEFAULT_P1, "p1 default round-trip");
            assert_eq!(p3, DEFAULT_P3, "p3 default round-trip");
        }

        #[test]
        fn p3_p4_photo_high_d_gate_default_roundtrip() {
            let (p3, p4) = p3_p4_photo_high_d_gate(3.5, 1.0);
            assert_eq!(p3, DEFAULT_P3, "p3 default round-trip");
            assert_eq!(p4, DEFAULT_P4, "p4 default round-trip");
        }

        // ─── range coverage (the W44-216 LHS empirical envelope) ───

        #[test]
        fn p1_p2_ridge_covers_lhs_range() {
            // s=0 → (P1_RIDGE_MAX, P2_RIDGE_MAX). s=1 → mirrored low
            // (clamped to ≥ 0 for p1).
            let (p1_lo, p2_lo) = p1_p2_smoothness_dispatch_ridge(1.0);
            let (p1_hi, p2_hi) = p1_p2_smoothness_dispatch_ridge(0.0);
            assert!(p1_lo <= DEFAULT_P1, "smoothness=1 → p1 low");
            assert!(p2_lo <= DEFAULT_P2, "smoothness=1 → p2 low");
            assert!(p1_hi >= DEFAULT_P1, "smoothness=0 → p1 high");
            assert!(p2_hi >= DEFAULT_P2, "smoothness=0 → p2 high");
            assert!((p1_hi - P1_RIDGE_MAX).abs() < EPS);
            assert!((p2_hi - P2_RIDGE_MAX).abs() < EPS);
        }

        #[test]
        fn screen_quant_lift_saturates_above_one() {
            // At k = 1.5, the cap should give 1 + 0.5 * 0.8 = 1.4 mult
            let (p5, p6) = p5_p6_effort_conditional_lift(1.5);
            let expected = 1.0 + 0.5 * P5_P6_SATURATION_STRENGTH;
            assert!(
                (p5 - DEFAULT_P5 * expected).abs() < EPS,
                "p5 = {}, expected {}",
                p5,
                DEFAULT_P5 * expected
            );
            assert!(
                (p6 - DEFAULT_P6 * expected).abs() < EPS,
                "p6 = {}, expected {}",
                p6,
                DEFAULT_P6 * expected
            );
        }

        #[test]
        fn screenshot_quant_aggressiveness_saturates_above_one() {
            // At a = 1.5, the cap should give 1 + 0.5 * 0.7 = 1.35
            let (p3, p6) = p3_p6_screenshot_qac_lift(1.5);
            let expected = 1.0 + 0.5 * P3_P6_SATURATION_STRENGTH;
            assert!(
                (p3 - DEFAULT_P3 * expected).abs() < EPS,
                "p3 = {}, expected {}",
                p3,
                DEFAULT_P3 * expected
            );
            assert!(
                (p6 - DEFAULT_P6 * expected).abs() < EPS,
                "p6 = {}, expected {}",
                p6,
                DEFAULT_P6 * expected
            );
        }

        #[test]
        fn screen_quant_lift_below_one_is_linear() {
            // At k=0.5 no saturation, so (1.0, 1.5).
            let (p5, p6) = p5_p6_effort_conditional_lift(0.5);
            assert!((p5 - DEFAULT_P5 * 0.5).abs() < EPS);
            assert!((p6 - DEFAULT_P6 * 0.5).abs() < EPS);
        }

        #[test]
        fn buttloop_d_gate_clamped() {
            // Clamp test through p4_p5: input 0.5 should clamp to 1.5.
            let (p4_lo, _p5) = p4_p5_buttloop_vs_adaptive_quant_dispatch(0.5, 1.0);
            assert_eq!(p4_lo, 1.5);
            // Input 10.0 should clamp to 5.5.
            let (p4_hi, _p5) = p4_p5_buttloop_vs_adaptive_quant_dispatch(10.0, 1.0);
            assert_eq!(p4_hi, 5.5);
        }

        // ─── composition delegation ───

        #[test]
        fn p4_p5_and_p4_p6_share_d_gate() {
            // Same buttloop_screen_d_gate input → same p4 in both fns.
            let d = 2.7_f32;
            let (p4_a, _) = p4_p5_buttloop_vs_adaptive_quant_dispatch(d, 1.0);
            let (p4_b, _) = p4_p6_e7_buttloop_synergy(1.0, d);
            assert_eq!(p4_a, p4_b, "p4 should match across composition fns");
        }

        #[test]
        fn p5_p6_and_p4_p5_share_lift() {
            // Same screen_quant_lift input → same p5 in both fns.
            let k = 1.3_f32;
            let (_p4, p5_a) = p4_p5_buttloop_vs_adaptive_quant_dispatch(3.5, k);
            let (p5_b, _p6) = p5_p6_effort_conditional_lift(k);
            assert_eq!(p5_a, p5_b, "p5 should match across composition fns");
        }

        // ─── W44-221 Tier-2 knob expander ───
        //
        // These tests are `tuning-override`-gated because Tier2Knobs +
        // expand_to_runtime_tuning live behind the feature flag (their
        // return type `runtime::RuntimeTuning` only exists under it).

        /// W44-221 hash-lock contract: defaults must round-trip byte-exact.
        #[cfg(feature = "tuning-override")]
        #[test]
        fn tier2_knobs_default_roundtrips_to_runtime_default() {
            let knobs = Tier2Knobs::default();
            let runtime = knobs.expand_to_runtime_tuning();
            let expected = super::super::runtime::RuntimeTuning::default();
            assert_eq!(
                runtime.smart_zenjxl_photo_mask_p25_min, expected.smart_zenjxl_photo_mask_p25_min,
                "p1 default round-trip"
            );
            assert_eq!(
                runtime.screenshot_median_threshold, expected.screenshot_median_threshold,
                "p2 default round-trip"
            );
            assert_eq!(
                runtime.buttloop_default_screenshot_qf_seed_scale,
                expected.buttloop_default_screenshot_qf_seed_scale,
                "p3 default round-trip"
            );
            assert_eq!(
                runtime.buttloop_qf_seed_scale_min_distance,
                expected.buttloop_qf_seed_scale_min_distance,
                "p4 default round-trip"
            );
            assert_eq!(
                runtime.adaptive_quant_screenshot_qf_seed_scale_e5_e6,
                expected.adaptive_quant_screenshot_qf_seed_scale_e5_e6,
                "p5 default round-trip"
            );
            assert_eq!(
                runtime.adaptive_quant_screenshot_qf_seed_scale_e7,
                expected.adaptive_quant_screenshot_qf_seed_scale_e7,
                "p6 default round-trip"
            );
        }

        /// Backwards-compat: the W44-217-original 3-knob signature defaults
        /// (`smoothness_bias=0.5, screen_quant_lift=1.0, buttloop_screen_d_gate=3.5`)
        /// must also round-trip to defaults.
        #[cfg(feature = "tuning-override")]
        #[test]
        fn expand_knobs_to_runtime_3knob_default_roundtrips() {
            let runtime = expand_knobs_to_runtime(0.5, 1.0, 3.5);
            let expected = super::super::runtime::RuntimeTuning::default();
            assert_eq!(
                runtime.smart_zenjxl_photo_mask_p25_min,
                expected.smart_zenjxl_photo_mask_p25_min
            );
            assert_eq!(
                runtime.screenshot_median_threshold,
                expected.screenshot_median_threshold
            );
            assert_eq!(
                runtime.buttloop_default_screenshot_qf_seed_scale,
                expected.buttloop_default_screenshot_qf_seed_scale
            );
            assert_eq!(
                runtime.buttloop_qf_seed_scale_min_distance,
                expected.buttloop_qf_seed_scale_min_distance
            );
            assert_eq!(
                runtime.adaptive_quant_screenshot_qf_seed_scale_e5_e6,
                expected.adaptive_quant_screenshot_qf_seed_scale_e5_e6
            );
            assert_eq!(
                runtime.adaptive_quant_screenshot_qf_seed_scale_e7,
                expected.adaptive_quant_screenshot_qf_seed_scale_e7
            );
        }

        /// Range coverage: knobs at min/max produce param values within
        /// the W44-216 LHS empirical envelope.
        #[cfg(feature = "tuning-override")]
        #[test]
        fn tier2_knobs_smoothness_bias_range() {
            // smoothness=0 → p1 at P1_RIDGE_MAX, p2 at P2_RIDGE_MAX
            let knobs_lo_smooth = Tier2Knobs {
                smoothness_bias: 0.0,
                ..Default::default()
            };
            let rt = knobs_lo_smooth.expand_to_runtime_tuning();
            assert!((rt.smart_zenjxl_photo_mask_p25_min - P1_RIDGE_MAX).abs() < EPS);
            assert!((rt.screenshot_median_threshold - P2_RIDGE_MAX).abs() < EPS);

            // smoothness=1 → p1 mirrored low (clamped ≥ 0), p2 mirrored low
            let knobs_hi_smooth = Tier2Knobs {
                smoothness_bias: 1.0,
                ..Default::default()
            };
            let rt = knobs_hi_smooth.expand_to_runtime_tuning();
            assert!(rt.smart_zenjxl_photo_mask_p25_min < DEFAULT_P1);
            assert!(rt.screenshot_median_threshold < DEFAULT_P2);
        }

        /// Out-of-range knob values clamp.
        #[cfg(feature = "tuning-override")]
        #[test]
        fn tier2_knobs_out_of_range_values_clamp() {
            // smoothness=-1 → clamped to 0
            let rt_neg = Tier2Knobs {
                smoothness_bias: -1.0,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            let rt_zero = Tier2Knobs {
                smoothness_bias: 0.0,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            assert_eq!(
                rt_neg.smart_zenjxl_photo_mask_p25_min,
                rt_zero.smart_zenjxl_photo_mask_p25_min
            );

            // buttloop_screen_d_gate=10 → clamped to 5.5
            let rt_huge = Tier2Knobs {
                buttloop_screen_d_gate: 10.0,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            assert_eq!(rt_huge.buttloop_qf_seed_scale_min_distance, 5.5);
        }

        /// Additive composition on p6: both `screenshot_quant_aggressiveness`
        /// and `screen_quant_lift` touch p6; sum of deviations should compose.
        #[cfg(feature = "tuning-override")]
        #[test]
        fn tier2_knobs_p6_additive_composition() {
            // Knob A only
            let rt_a = Tier2Knobs {
                screenshot_quant_aggressiveness: 1.5,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            let dev_a = rt_a.adaptive_quant_screenshot_qf_seed_scale_e7 - DEFAULT_P6;
            // Knob B only
            let rt_b = Tier2Knobs {
                screen_quant_lift: 1.5,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            let dev_b = rt_b.adaptive_quant_screenshot_qf_seed_scale_e7 - DEFAULT_P6;
            // Both at once → sum of deviations
            let rt_ab = Tier2Knobs {
                screenshot_quant_aggressiveness: 1.5,
                screen_quant_lift: 1.5,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            let dev_ab = rt_ab.adaptive_quant_screenshot_qf_seed_scale_e7 - DEFAULT_P6;
            assert!(
                (dev_ab - (dev_a + dev_b)).abs() < 1e-4,
                "p6 deviations should sum: dev_ab={} vs dev_a+dev_b={}",
                dev_ab,
                dev_a + dev_b
            );
        }

        /// Single-knob deviations: changing one knob only moves the params
        /// that knob touches.
        #[cfg(feature = "tuning-override")]
        #[test]
        fn tier2_knobs_single_knob_locality() {
            // Move smoothness_bias only — p1, p2 should change; p3..p6 unchanged.
            let rt = Tier2Knobs {
                smoothness_bias: 0.0, // off-default
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            assert!(rt.smart_zenjxl_photo_mask_p25_min != DEFAULT_P1);
            assert!(rt.screenshot_median_threshold != DEFAULT_P2);
            assert_eq!(rt.buttloop_default_screenshot_qf_seed_scale, DEFAULT_P3);
            assert_eq!(rt.buttloop_qf_seed_scale_min_distance, DEFAULT_P4);
            assert_eq!(rt.adaptive_quant_screenshot_qf_seed_scale_e5_e6, DEFAULT_P5);
            assert_eq!(rt.adaptive_quant_screenshot_qf_seed_scale_e7, DEFAULT_P6);

            // Move buttloop_screen_d_gate only — only p4 changes.
            let rt = Tier2Knobs {
                buttloop_screen_d_gate: 2.0,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            assert_eq!(rt.smart_zenjxl_photo_mask_p25_min, DEFAULT_P1);
            assert_eq!(rt.screenshot_median_threshold, DEFAULT_P2);
            assert_eq!(rt.buttloop_default_screenshot_qf_seed_scale, DEFAULT_P3);
            assert_eq!(rt.buttloop_qf_seed_scale_min_distance, 2.0);
            assert_eq!(rt.adaptive_quant_screenshot_qf_seed_scale_e5_e6, DEFAULT_P5);
            assert_eq!(rt.adaptive_quant_screenshot_qf_seed_scale_e7, DEFAULT_P6);
        }

        /// Non-negativity: physical floors are respected.
        #[cfg(feature = "tuning-override")]
        #[test]
        fn tier2_knobs_physical_floors() {
            // Push all knobs to extremes — every param value stays ≥ 0.
            let rt = Tier2Knobs {
                smoothness_bias: 1.0,
                screenshot_quant_aggressiveness: 0.0,
                screen_quant_lift: 0.5,
                buttloop_screen_d_gate: 1.5,
                buttloop_aq_balance: 0.0,
            }
            .expand_to_runtime_tuning();
            assert!(rt.smart_zenjxl_photo_mask_p25_min >= 0.0);
            assert!(rt.screenshot_median_threshold >= 0.0);
            assert!(rt.buttloop_default_screenshot_qf_seed_scale >= 0.0);
            assert!(rt.buttloop_qf_seed_scale_min_distance >= 1.5);
            assert!(rt.adaptive_quant_screenshot_qf_seed_scale_e5_e6 >= 0.0);
            assert!(rt.adaptive_quant_screenshot_qf_seed_scale_e7 >= 0.0);
        }

        // ─── W44-222 5th-knob (buttloop_aq_balance) tests ───
        //
        // The 5th knob is `buttloop_aq_balance ∈ [-1, +1]` (default 0.0).
        // At default (0.0) every K5_DIR contribution is 0 → expander
        // round-trips byte-exact (the hash-lock contract).

        /// W44-222 hash-lock contract: 5th knob at default must round-trip
        /// byte-exact. (Subsumes the W44-221 4-knob test because the W44-221
        /// test uses `..Default::default()` which now includes
        /// `buttloop_aq_balance: 0.0`. This is an explicit-value test.)
        #[cfg(feature = "tuning-override")]
        #[test]
        fn tier2_knobs_buttloop_aq_balance_default_roundtrips() {
            let knobs = Tier2Knobs {
                smoothness_bias: 0.5,
                screenshot_quant_aggressiveness: 1.0,
                screen_quant_lift: 1.0,
                buttloop_screen_d_gate: DEFAULT_P4,
                buttloop_aq_balance: 0.0,
            };
            let rt = knobs.expand_to_runtime_tuning();
            let expected = super::super::runtime::RuntimeTuning::default();
            assert_eq!(
                rt.smart_zenjxl_photo_mask_p25_min,
                expected.smart_zenjxl_photo_mask_p25_min
            );
            assert_eq!(
                rt.screenshot_median_threshold,
                expected.screenshot_median_threshold
            );
            assert_eq!(
                rt.buttloop_default_screenshot_qf_seed_scale,
                expected.buttloop_default_screenshot_qf_seed_scale
            );
            assert_eq!(
                rt.buttloop_qf_seed_scale_min_distance,
                expected.buttloop_qf_seed_scale_min_distance
            );
            assert_eq!(
                rt.adaptive_quant_screenshot_qf_seed_scale_e5_e6,
                expected.adaptive_quant_screenshot_qf_seed_scale_e5_e6
            );
            assert_eq!(
                rt.adaptive_quant_screenshot_qf_seed_scale_e7,
                expected.adaptive_quant_screenshot_qf_seed_scale_e7
            );
        }

        /// W44-222 range coverage: knob at ±1 moves params along the
        /// direction documented in [`K5_DIR`].
        #[cfg(feature = "tuning-override")]
        #[test]
        fn tier2_knobs_buttloop_aq_balance_range() {
            let rt_pos = Tier2Knobs {
                buttloop_aq_balance: 1.0,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            let rt_neg = Tier2Knobs {
                buttloop_aq_balance: -1.0,
                ..Default::default()
            }
            .expand_to_runtime_tuning();

            // K5_DIR = [-0.148, +0.259, -0.650, 0, -0.504, +0.485]
            // scale = 2.5 → at k5 = +1 each param shifts by 2.5 * dir[i].
            // p3 should DROP (K5_DIR[2] = -0.650 < 0), p5 should DROP, p6 RISE.
            assert!(
                rt_pos.buttloop_default_screenshot_qf_seed_scale < DEFAULT_P3,
                "k5=+1 should lower p3 (K5_DIR[2] < 0)"
            );
            assert!(
                rt_pos.adaptive_quant_screenshot_qf_seed_scale_e5_e6 < DEFAULT_P5,
                "k5=+1 should lower p5 (K5_DIR[4] < 0)"
            );
            assert!(
                rt_pos.adaptive_quant_screenshot_qf_seed_scale_e7 > DEFAULT_P6,
                "k5=+1 should raise p6 (K5_DIR[5] > 0)"
            );

            // k5 = -1: opposite signs from k5 = +1
            assert!(
                rt_neg.buttloop_default_screenshot_qf_seed_scale > DEFAULT_P3,
                "k5=-1 should raise p3"
            );
            assert!(
                rt_neg.adaptive_quant_screenshot_qf_seed_scale_e5_e6 > DEFAULT_P5,
                "k5=-1 should raise p5"
            );
            assert!(
                rt_neg.adaptive_quant_screenshot_qf_seed_scale_e7 < DEFAULT_P6,
                "k5=-1 should lower p6"
            );

            // p4 unchanged (K5_DIR[3] = 0 by construction).
            assert_eq!(
                rt_pos.buttloop_qf_seed_scale_min_distance, DEFAULT_P4,
                "K5_DIR has zero component on p4"
            );
            assert_eq!(
                rt_neg.buttloop_qf_seed_scale_min_distance, DEFAULT_P4,
                "K5_DIR has zero component on p4"
            );
        }

        /// W44-222 out-of-range knob values clamp to [-1, +1].
        #[cfg(feature = "tuning-override")]
        #[test]
        fn tier2_knobs_buttloop_aq_balance_clamps() {
            let rt_huge = Tier2Knobs {
                buttloop_aq_balance: 5.0,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            let rt_one = Tier2Knobs {
                buttloop_aq_balance: 1.0,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            // Clamp at +1.0 → same as +5.0
            assert!(
                (rt_huge.buttloop_default_screenshot_qf_seed_scale
                    - rt_one.buttloop_default_screenshot_qf_seed_scale)
                    .abs()
                    < EPS,
                "buttloop_aq_balance > +1 should clamp to +1"
            );
            let rt_neg_huge = Tier2Knobs {
                buttloop_aq_balance: -5.0,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            let rt_neg_one = Tier2Knobs {
                buttloop_aq_balance: -1.0,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            assert!(
                (rt_neg_huge.buttloop_default_screenshot_qf_seed_scale
                    - rt_neg_one.buttloop_default_screenshot_qf_seed_scale)
                    .abs()
                    < EPS,
                "buttloop_aq_balance < -1 should clamp to -1"
            );
        }

        /// W44-222 single-knob locality: moving k5 only changes params
        /// that K5_DIR touches (p1, p2, p3, p5, p6); p4 is unchanged.
        #[cfg(feature = "tuning-override")]
        #[test]
        fn tier2_knobs_buttloop_aq_balance_single_knob_locality() {
            // Move k5 only — every other knob at default.
            let rt = Tier2Knobs {
                buttloop_aq_balance: 0.5,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            // p4 is unchanged (K5_DIR[3] = 0).
            assert_eq!(rt.buttloop_qf_seed_scale_min_distance, DEFAULT_P4);
            // p1, p2, p3, p5, p6 ALL move (K5_DIR[i] ≠ 0 for i ∈ {0,1,2,4,5}).
            assert!(rt.smart_zenjxl_photo_mask_p25_min != DEFAULT_P1);
            assert!(rt.screenshot_median_threshold != DEFAULT_P2);
            assert!(rt.buttloop_default_screenshot_qf_seed_scale != DEFAULT_P3);
            assert!(rt.adaptive_quant_screenshot_qf_seed_scale_e5_e6 != DEFAULT_P5);
            assert!(rt.adaptive_quant_screenshot_qf_seed_scale_e7 != DEFAULT_P6);
        }

        /// W44-222 K5 + existing-knob additive composition: moving k5 AND
        /// (e.g.) screen_quant_lift simultaneously composes ADDITIVELY
        /// — the K5 delta and the ridge delta on the same param sum.
        #[cfg(feature = "tuning-override")]
        #[test]
        fn tier2_knobs_buttloop_aq_balance_composes_additively() {
            let rt_k5_only = Tier2Knobs {
                buttloop_aq_balance: 0.5,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            let rt_lift_only = Tier2Knobs {
                screen_quant_lift: 1.3,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            let rt_both = Tier2Knobs {
                buttloop_aq_balance: 0.5,
                screen_quant_lift: 1.3,
                ..Default::default()
            }
            .expand_to_runtime_tuning();

            // p5 deviations should sum (additive composition; before physical-floor clamp).
            let dev_k5_p5 = rt_k5_only.adaptive_quant_screenshot_qf_seed_scale_e5_e6 - DEFAULT_P5;
            let dev_lift_p5 =
                rt_lift_only.adaptive_quant_screenshot_qf_seed_scale_e5_e6 - DEFAULT_P5;
            let dev_both_p5 = rt_both.adaptive_quant_screenshot_qf_seed_scale_e5_e6 - DEFAULT_P5;
            // Both deltas are positive enough not to hit the 0.0 floor on p5
            // at these moderate knob values (verified by the assert below).
            assert!(rt_k5_only.adaptive_quant_screenshot_qf_seed_scale_e5_e6 > 0.0);
            assert!(rt_lift_only.adaptive_quant_screenshot_qf_seed_scale_e5_e6 > 0.0);
            assert!(rt_both.adaptive_quant_screenshot_qf_seed_scale_e5_e6 > 0.0);
            assert!(
                (dev_both_p5 - (dev_k5_p5 + dev_lift_p5)).abs() < 1e-4,
                "p5 deltas should sum additively: dev_both={} vs dev_k5+dev_lift={}",
                dev_both_p5,
                dev_k5_p5 + dev_lift_p5
            );

            // Same check on p6 (where THREE knobs may contribute:
            // screenshot_quant_aggressiveness, screen_quant_lift, k5).
            let dev_k5_p6 = rt_k5_only.adaptive_quant_screenshot_qf_seed_scale_e7 - DEFAULT_P6;
            let dev_lift_p6 = rt_lift_only.adaptive_quant_screenshot_qf_seed_scale_e7 - DEFAULT_P6;
            let dev_both_p6 = rt_both.adaptive_quant_screenshot_qf_seed_scale_e7 - DEFAULT_P6;
            assert!(
                (dev_both_p6 - (dev_k5_p6 + dev_lift_p6)).abs() < 1e-4,
                "p6 deltas should sum additively: dev_both={} vs dev_k5+dev_lift={}",
                dev_both_p6,
                dev_k5_p6 + dev_lift_p6
            );
        }

        /// W44-222 physical floors are preserved across the full
        /// `buttloop_aq_balance` range. At extreme knob values, no param
        /// goes below its physical floor (0.0 for p1/p2/p3/p5/p6; 1.5 for p4).
        #[cfg(feature = "tuning-override")]
        #[test]
        fn tier2_knobs_buttloop_aq_balance_physical_floors() {
            // Stress: push k5 to +1 while another knob is already at the
            // edge that further-reduces a K5-touched param.
            // e.g., screen_quant_lift = 0.5 → p5 = 1.0; then k5 = +1 drops p5
            // by 2.5 * 0.504 = 1.26 → would be -0.26 → clamps to 0.0.
            let rt = Tier2Knobs {
                screen_quant_lift: 0.5,
                buttloop_aq_balance: 1.0,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            assert!(rt.adaptive_quant_screenshot_qf_seed_scale_e5_e6 >= 0.0);
            assert!(rt.buttloop_default_screenshot_qf_seed_scale >= 0.0);
            assert!(rt.adaptive_quant_screenshot_qf_seed_scale_e7 >= 0.0);
            assert!(rt.smart_zenjxl_photo_mask_p25_min >= 0.0);
            assert!(rt.screenshot_median_threshold >= 0.0);
            // p4 unchanged from buttloop_screen_d_gate default 3.5
            assert!(rt.buttloop_qf_seed_scale_min_distance >= 1.5);

            // Stress: k5 = -1, screenshot_quant_aggressiveness=0 (p3 ridge floor at 0,
            // p6 ridge floor at 0). k5 contributes p3 += 2.5*0.650 = +1.625 → p3 = 1.625
            // (above 0 — OK). p6 contribution is -2.5*0.485 = -1.21 → from p6 ridge at 0,
            // would push to -1.21 → clamps to 0. p5 += 2.5*0.504 = +1.26 → above 0 OK.
            let rt2 = Tier2Knobs {
                screenshot_quant_aggressiveness: 0.0,
                screen_quant_lift: 0.5,
                buttloop_aq_balance: -1.0,
                ..Default::default()
            }
            .expand_to_runtime_tuning();
            assert!(rt2.buttloop_default_screenshot_qf_seed_scale >= 0.0);
            assert!(rt2.adaptive_quant_screenshot_qf_seed_scale_e5_e6 >= 0.0);
            assert!(rt2.adaptive_quant_screenshot_qf_seed_scale_e7 >= 0.0);
        }

        /// CouplingClass enum stability — variant names referenced from
        /// `docs/PARAM_INTERACTIONS.md`.
        #[test]
        fn coupling_class_variants_stable() {
            // Just a name-stability lock; if a variant is renamed,
            // PARAM_INTERACTIONS.md must be updated too.
            let _all = [
                CouplingClass::Additive,
                CouplingClass::Multiplicative,
                CouplingClass::Gated,
                CouplingClass::Suppressive,
                CouplingClass::Synergistic,
                CouplingClass::WeaklyCoupled,
            ];
        }

        // ─── W44-228b per-stratum default lookup unit tests ───

        /// W44-228b: every stratum maps to a finite, well-clamped
        /// `Tier2Knobs` value (no NaN, every field within its documented
        /// range). Also asserts knob fields are within their own
        /// MIN/MAX bounds — defends against TSV transcription errors.
        #[cfg(feature = "tuning-override")]
        #[test]
        fn w44_228b_default_for_stratum_values_in_range() {
            for s in [
                ContentStratum::ScreenVeryHigh,
                ContentStratum::ScreenHigh,
                ContentStratum::ScreenMid,
                ContentStratum::ScreenLow,
                ContentStratum::PhotoVeryHigh,
                ContentStratum::PhotoHigh,
                ContentStratum::PhotoMid,
                ContentStratum::PhotoLow,
            ] {
                let k = Tier2Knobs::default_for_stratum(s);
                assert!(
                    k.smoothness_bias.is_finite()
                        && k.smoothness_bias >= Tier2Knobs::SMOOTHNESS_BIAS_MIN
                        && k.smoothness_bias <= Tier2Knobs::SMOOTHNESS_BIAS_MAX,
                    "stratum {:?}: smoothness_bias {} out of range",
                    s,
                    k.smoothness_bias
                );
                assert!(
                    k.screenshot_quant_aggressiveness.is_finite()
                        && k.screenshot_quant_aggressiveness
                            >= Tier2Knobs::SCREENSHOT_QUANT_AGGRESSIVENESS_MIN
                        && k.screenshot_quant_aggressiveness
                            <= Tier2Knobs::SCREENSHOT_QUANT_AGGRESSIVENESS_MAX,
                    "stratum {:?}: screenshot_quant_aggressiveness {} out of range",
                    s,
                    k.screenshot_quant_aggressiveness
                );
                assert!(
                    k.screen_quant_lift.is_finite()
                        && k.screen_quant_lift >= Tier2Knobs::SCREEN_QUANT_LIFT_MIN
                        && k.screen_quant_lift <= Tier2Knobs::SCREEN_QUANT_LIFT_MAX,
                    "stratum {:?}: screen_quant_lift {} out of range",
                    s,
                    k.screen_quant_lift
                );
                assert!(
                    k.buttloop_screen_d_gate.is_finite()
                        && k.buttloop_screen_d_gate >= Tier2Knobs::BUTTLOOP_SCREEN_D_GATE_MIN
                        && k.buttloop_screen_d_gate <= Tier2Knobs::BUTTLOOP_SCREEN_D_GATE_MAX,
                    "stratum {:?}: buttloop_screen_d_gate {} out of range",
                    s,
                    k.buttloop_screen_d_gate
                );
                assert!(
                    k.buttloop_aq_balance.is_finite()
                        && k.buttloop_aq_balance >= Tier2Knobs::BUTTLOOP_AQ_BALANCE_MIN
                        && k.buttloop_aq_balance <= Tier2Knobs::BUTTLOOP_AQ_BALANCE_MAX,
                    "stratum {:?}: buttloop_aq_balance {} out of range",
                    s,
                    k.buttloop_aq_balance
                );
                // Sanity: the expander never panics on any per-stratum
                // tuple (catches accidental floor underflows).
                let _rt = k.expand_to_runtime_tuning();
            }
        }

        /// W44-228b finding "every per-stratum optimum has
        /// `screenshot_quant_aggressiveness == 0`" was **overturned by
        /// W44-PHASE4-S2-refit (2026-05-24)**: the post-W44-RECON-DEEP
        /// corpus shifted screen/high, screen/mid, screen/low optima to
        /// `aggressiveness == 0.333`. screen/very_high and all 4 photo
        /// strata still pinned to `aggressiveness == 0` in the raw refit.
        ///
        /// **W44-PHASE4-S2-refit-c2 (2026-05-24)**: 8-stratum × 5-knob
        /// ablation audit (`benchmarks/w44_phase4_s2_refit_c2_audit_2026-05-24.tsv`)
        /// found `aggressiveness=0` BROKE the W44-105 SHIP cell on
        /// screen/very_high (terminal e8 d=4, ΔSSIM2 -4.93 vs baseline) AND
        /// on screen/high (graph e8 d=3, ΔSSIM2 -5.07). 2-knob ablation
        /// found (k1=0.5, k2=1.0) restores both within ±0.5 SSIM2. c2 fix
        /// floors k1=0.5 + k2=1.0 (= defaults) on those 2 strata.
        ///
        /// This test pins which strata are at zero vs non-zero — if a
        /// future re-derivation shifts any stratum's membership, update
        /// `docs/TIER_2_KNOBS.md` + the W44-PHASE4-S2-refit memo + the
        /// W44-105 SHIP-cell caveat alongside the table swap.
        ///
        /// **W44-105 protection invariant (enforced here as of c2)**:
        /// screen/very_high and screen/high MUST carry
        /// `aggressiveness == 1.0` (default) AND `smoothness_bias == 0.5`
        /// (default). These two strata route through the W44-105 buttloop
        /// screen-seed lift on terminal/graph-class SHIP cells; any
        /// non-default k1 or k2 reintroduces the c2 cliff.
        #[cfg(feature = "tuning-override")]
        #[test]
        fn w44_phase4_s2_refit_strata_aggressiveness_membership() {
            // Strata that pin to aggressiveness == 0 (photos + screen/very_high
            // and screen/high were promoted to 1.0 = default in c2).
            for s in [
                ContentStratum::PhotoVeryHigh,
                ContentStratum::PhotoHigh,
                ContentStratum::PhotoMid,
                ContentStratum::PhotoLow,
            ] {
                let k = Tier2Knobs::default_for_stratum(s);
                assert_eq!(
                    k.screenshot_quant_aggressiveness, 0.0,
                    "Stratum {:?} should pin aggressiveness=0 per W44-PHASE4-S2-refit. \
                     If shifted, update docs/TIER_2_KNOBS.md + W44-105 SHIP-cell caveat.",
                    s
                );
            }
            // Strata that hold aggressiveness == 0.333 (W44-PHASE4-S2 refit —
            // these are AWAY from the W44-105 catastrophe regime since they
            // sit at d < 2 in the screen-class).
            for s in [ContentStratum::ScreenMid, ContentStratum::ScreenLow] {
                let k = Tier2Knobs::default_for_stratum(s);
                assert!(
                    (k.screenshot_quant_aggressiveness - 0.3333333333333333_f32).abs() < 1e-5,
                    "Stratum {:?} should hold aggressiveness=0.333 per W44-PHASE4-S2-refit, \
                     got {}. If shifted, update docs/TIER_2_KNOBS.md.",
                    s,
                    k.screenshot_quant_aggressiveness
                );
            }
            // W44-PHASE4-S2-refit-c2: screen/very_high and screen/high
            // MUST hold (k1=0.5, k2=1.0) = defaults to preserve W44-105
            // SHIP cells. The c2 ablation audit found ANY non-default k1
            // OR k2 reintroduces the cliff on terminal/graph SHIP cells.
            for s in [ContentStratum::ScreenVeryHigh, ContentStratum::ScreenHigh] {
                let k = Tier2Knobs::default_for_stratum(s);
                assert_eq!(
                    k.smoothness_bias, 0.5,
                    "Stratum {:?} MUST hold smoothness_bias=0.5 (default) per W44-PHASE4-S2-refit-c2. \
                     Lowering risks the W44-105 SHIP-cell SSIM2 cliff.",
                    s
                );
                assert_eq!(
                    k.screenshot_quant_aggressiveness, 1.0,
                    "Stratum {:?} MUST hold screenshot_quant_aggressiveness=1.0 (default) per \
                     W44-PHASE4-S2-refit-c2. Lowering risks the W44-105 SHIP-cell SSIM2 cliff.",
                    s
                );
            }
        }
    }
}

// ─── Section: production consumer macro (W44-213) ──────────────────────
//
// The `runtime_or_default!` macro is the canonical access path for every
// production code site that needs a RuntimeTuning-aware lookup. With the
// `tuning-override` feature DISABLED (default for production builds), the
// macro expands to the raw const reference — the compiler inlines this
// to an immediate value at every call site, so production binaries pay
// ZERO runtime cost. With the feature ENABLED (sweep-runner builds), the
// macro calls [`runtime::get_or_default`] which short-circuits to the
// default const when no override is installed (single atomic-OnceLock
// load + branch).
//
// ## Usage
//
// ```ignore
// // Before W44-213:
// let scale = tuning::buttloop::DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE;
//
// // After W44-213:
// let scale = jxl_encoder::runtime_or_default!(
//     tuning::buttloop::DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE,
//     buttloop_default_screenshot_qf_seed_scale,
// );
// ```
//
// The macro takes two arguments:
// 1. The source-of-truth const path (used both for the default fast-path
//    AND for the `accessor` closure's return-value type inference).
// 2. The `RuntimeTuning` field name (without the `RuntimeTuning::` prefix).
//
// ## Hash-lock invariant
//
// `RuntimeTuning::default()` MUST match every source-of-truth const
// exactly. The unit test `tuning::runtime::tests::default_matches_production_consts`
// enforces this so production hash-locks (36 lossy + 13 lossless fixtures)
// stay byte-identical at the `tuning-override` feature default value.
//
// The W44-213 wiring touches 6 RuntimeTuning fields:
// - `smart_zenjxl_photo_mask_p25_min` (4-site duplicate; macro applied
//   at every site)
// - `screenshot_median_threshold` (4-site duplicate)
// - `buttloop_default_screenshot_qf_seed_scale` (1 site)
// - `buttloop_qf_seed_scale_min_distance` (2 sites)
// - `adaptive_quant_screenshot_qf_seed_scale_e5_e6` (1 site)
// - `adaptive_quant_screenshot_qf_seed_scale_e7` (1 site)

/// W44-213 production consumer macro for runtime-tuning-aware const reads.
/// See [`tuning`] module docs for the full rationale.
///
/// **Production builds** (default, `tuning-override` OFF): expands to
/// `$const_path` — zero overhead, compiler inlines.
///
/// **Sweep-runner builds** (`tuning-override` ON): expands to
/// `crate::tuning::runtime::get_or_default($const_path, |t| t.$field)`.
#[macro_export]
macro_rules! runtime_or_default {
    ($const_path:path, $field:ident $(,)?) => {{
        #[cfg(not(feature = "tuning-override"))]
        {
            $const_path
        }
        #[cfg(feature = "tuning-override")]
        {
            $crate::tuning::runtime::get_or_default($const_path, |t| t.$field)
        }
    }};
}

// ─── Section: runtime override (opt-in for the future sweep runner) ────
//
// Enabled by `--features tuning-override`. The struct mirrors the const
// paths above; production code paths read the const directly (zero
// runtime cost when the feature is disabled). The override layer is for
// the sweep-runner binary ONLY — production builds keep the constants
// inlined by the compiler.

// W44-210-A row 16: runtime override scaffold (feature `tuning-override`).
#[cfg(feature = "tuning-override")]
#[allow(unused_imports)]
pub mod runtime {
    //! Sweep-runner runtime override for tunables (W44-210-A row 16).
    //!
    //! ## Why opt-in
    //!
    //! Production binaries should pay zero runtime cost for tuning
    //! lookups. The constants in the parent module compile down to
    //! immediate values at every consumer call site. The override layer
    //! is for the dedicated `tuning-sweep` binary (W44-212+) that needs
    //! to swap values at startup from a postcard file.
    //!
    //! ## Wire format
    //!
    //! Postcard binary. Field names mirror the const paths
    //! (`discriminator_thresholds_smart_zenjxl_photo_mask_p25_min`,
    //! `buttloop_default_buttloop_screenshot_qf_seed_scale`, etc.). The
    //! [`RuntimeTuning::default()`] returns the production constants
    //! verbatim; deserialised values OVERRIDE only the fields the
    //! sweep config emitted (via serde defaults).
    //!
    //! ## Production consumer pattern
    //!
    //! ```ignore
    //! #[cfg(feature = "tuning-override")]
    //! let scale = jxl_encoder::tuning::runtime::get(|t| t.buttloop_default_buttloop_screenshot_qf_seed_scale);
    //! #[cfg(not(feature = "tuning-override"))]
    //! let scale = jxl_encoder::tuning::buttloop::DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE;
    //! ```
    //!
    //! The runtime-override consumer is gated by `cfg!(feature)` so
    //! production binaries built without the feature don't pull serde
    //! / postcard. The sweep runner crate (`tuning-sweep-bin`) enables
    //! the feature.

    use core::sync::atomic::{AtomicBool, Ordering};
    use std::sync::OnceLock;

    /// Runtime override struct. Field names mirror the const paths
    /// in the parent module's submodules (lowercased, joined by `_`).
    /// All fields default to the production const values so an empty
    /// postcard payload is a no-op.
    ///
    /// The struct intentionally only carries the fields the sweep
    /// runner needs to swap; not every tunable is wired here.
    /// Extending it is additive (postcard tolerates missing fields
    /// when paired with `#[serde(default)]`).
    #[cfg_attr(feature = "tuning-override", derive(serde::Deserialize))]
    #[derive(Clone, Debug, PartialEq)]
    pub struct RuntimeTuning {
        // discriminator_thresholds
        #[cfg_attr(
            feature = "tuning-override",
            serde(default = "default_smart_zenjxl_photo_mask_p25_min")
        )]
        pub smart_zenjxl_photo_mask_p25_min: f32,
        #[cfg_attr(
            feature = "tuning-override",
            serde(default = "default_screenshot_median_threshold")
        )]
        pub screenshot_median_threshold: f32,

        // buttloop
        #[cfg_attr(
            feature = "tuning-override",
            serde(default = "default_buttloop_qf_seed_scale")
        )]
        pub buttloop_default_screenshot_qf_seed_scale: f32,
        #[cfg_attr(
            feature = "tuning-override",
            serde(default = "default_buttloop_qf_seed_min_distance")
        )]
        pub buttloop_qf_seed_scale_min_distance: f32,
        #[cfg_attr(
            feature = "tuning-override",
            serde(default = "default_adaptive_quant_qf_e5_e6")
        )]
        pub adaptive_quant_screenshot_qf_seed_scale_e5_e6: f32,
        #[cfg_attr(
            feature = "tuning-override",
            serde(default = "default_adaptive_quant_qf_e7")
        )]
        pub adaptive_quant_screenshot_qf_seed_scale_e7: f32,
    }

    impl Default for RuntimeTuning {
        fn default() -> Self {
            Self {
                smart_zenjxl_photo_mask_p25_min:
                    super::discriminator_thresholds::SMART_ZENJXL_PHOTO_MASK_P25_MIN,
                screenshot_median_threshold:
                    super::discriminator_thresholds::SCREENSHOT_MEDIAN_THRESHOLD,
                buttloop_default_screenshot_qf_seed_scale:
                    super::buttloop::DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE,
                buttloop_qf_seed_scale_min_distance:
                    super::buttloop::BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE,
                adaptive_quant_screenshot_qf_seed_scale_e5_e6:
                    super::buttloop::DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6,
                adaptive_quant_screenshot_qf_seed_scale_e7:
                    super::buttloop::DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7,
            }
        }
    }

    // Serde default fn helpers (postcard requires concrete functions
    // for `#[serde(default = "...")]`).
    fn default_smart_zenjxl_photo_mask_p25_min() -> f32 {
        super::discriminator_thresholds::SMART_ZENJXL_PHOTO_MASK_P25_MIN
    }
    fn default_screenshot_median_threshold() -> f32 {
        super::discriminator_thresholds::SCREENSHOT_MEDIAN_THRESHOLD
    }
    fn default_buttloop_qf_seed_scale() -> f32 {
        super::buttloop::DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE
    }
    fn default_buttloop_qf_seed_min_distance() -> f32 {
        super::buttloop::BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE
    }
    fn default_adaptive_quant_qf_e5_e6() -> f32 {
        super::buttloop::DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6
    }
    fn default_adaptive_quant_qf_e7() -> f32 {
        super::buttloop::DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7
    }

    static GLOBAL_TUNING: OnceLock<RuntimeTuning> = OnceLock::new();
    static LOADED: AtomicBool = AtomicBool::new(false);

    /// Install a runtime tuning override. Returns `Err` if a value has
    /// already been installed in this process (the global is
    /// single-shot to keep the access path const-fold-friendly).
    pub fn install(tuning: RuntimeTuning) -> Result<(), RuntimeTuning> {
        GLOBAL_TUNING.set(tuning).inspect_err(|_| {
            LOADED.store(true, Ordering::SeqCst);
        })?;
        LOADED.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// W44-222 idempotent install: best-effort install of a runtime tuning
    /// override that succeeds silently when:
    ///   - Nothing is installed yet (the install path runs), or
    ///   - An override IS installed and exactly matches `tuning` field-by-field
    ///     (the W44-222 `LossyConfig::with_knobs` builder calls this once per
    ///     encode; consecutive encodes with the same knobs no-op).
    ///
    /// Returns `Err(existing_tuning)` only on genuine MISMATCH (a different
    /// override is already installed). Callers should treat this as a
    /// user-error to surface, not a silent fallback — heterogeneous
    /// runtime overrides in one process are an anti-pattern (the
    /// global is single-shot by design).
    pub fn install_or_check_idempotent(tuning: RuntimeTuning) -> Result<(), RuntimeTuning> {
        match GLOBAL_TUNING.get() {
            Some(existing) => {
                if *existing == tuning {
                    Ok(())
                } else {
                    Err(existing.clone())
                }
            }
            None => install(tuning),
        }
    }

    /// Load a postcard-encoded `RuntimeTuning` from a file path.
    /// Convenience for the sweep runner.
    #[cfg(feature = "std")]
    pub fn install_from_postcard_file(path: &std::path::Path) -> Result<(), String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
        let tuning: RuntimeTuning = postcard::from_bytes(&bytes)
            .map_err(|e| format!("postcard decode {}: {}", path.display(), e))?;
        install(tuning).map_err(|_| {
            format!(
                "runtime tuning already installed for path {}",
                path.display()
            )
        })
    }

    /// Read a tunable through the runtime override (returns the
    /// installed value if any, else the default). Consumers should
    /// branch on `cfg!(feature = "tuning-override")` and call this
    /// only on the override-enabled path; production paths should
    /// read the const directly so the compiler can inline.
    pub fn get<F>(field: F) -> f32
    where
        F: FnOnce(&RuntimeTuning) -> f32,
    {
        if let Some(t) = GLOBAL_TUNING.get() {
            field(t)
        } else {
            field(&RuntimeTuning::default())
        }
    }

    /// Read a tunable through the runtime override, supplying an
    /// explicit default for the fast-path when no override is installed.
    ///
    /// **W44-213**: the production consumer macro
    /// [`super::runtime_or_default`] calls this fn through the
    /// `tuning-override` feature gate. With the feature DISABLED the
    /// macro expands to the const directly (zero overhead); with the
    /// feature ENABLED the macro calls this fn which short-circuits
    /// to `default` when the global tuning hasn't been installed.
    ///
    /// The fast-path (no installed override) is `GLOBAL_TUNING.get()`
    /// returning `None` → a single atomic-OnceLock load + branch. The
    /// slow-path (override installed) invokes `accessor(&tuning)` once.
    #[inline]
    pub fn get_or_default<F>(default: f32, accessor: F) -> f32
    where
        F: FnOnce(&RuntimeTuning) -> f32,
    {
        match GLOBAL_TUNING.get() {
            Some(t) => accessor(t),
            None => default,
        }
    }

    /// Same as [`get_or_default`] for `usize` fields. Not currently
    /// used by any of the 6 W44-211 fields (all are `f32`) but
    /// future RuntimeTuning extensions may need integer plumbing.
    #[inline]
    pub fn get_or_default_usize<F>(default: usize, accessor: F) -> usize
    where
        F: FnOnce(&RuntimeTuning) -> usize,
    {
        match GLOBAL_TUNING.get() {
            Some(t) => accessor(t),
            None => default,
        }
    }

    /// True if [`install`] has been called this process.
    pub fn is_loaded() -> bool {
        LOADED.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn default_matches_production_consts() {
            let t = RuntimeTuning::default();
            // discriminator_thresholds
            assert_eq!(
                t.smart_zenjxl_photo_mask_p25_min,
                super::super::discriminator_thresholds::SMART_ZENJXL_PHOTO_MASK_P25_MIN
            );
            assert_eq!(
                t.screenshot_median_threshold,
                super::super::discriminator_thresholds::SCREENSHOT_MEDIAN_THRESHOLD
            );
            // buttloop
            assert_eq!(
                t.buttloop_default_screenshot_qf_seed_scale,
                super::super::buttloop::DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE
            );
            assert_eq!(
                t.buttloop_qf_seed_scale_min_distance,
                super::super::buttloop::BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE
            );
            assert_eq!(
                t.adaptive_quant_screenshot_qf_seed_scale_e5_e6,
                super::super::buttloop::DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6
            );
            assert_eq!(
                t.adaptive_quant_screenshot_qf_seed_scale_e7,
                super::super::buttloop::DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7
            );
        }

        #[test]
        fn get_without_install_returns_defaults() {
            // NOTE: we can't reliably test the install path here because
            // `install()` is single-shot per process and other tests may
            // have called it. Just verify the default path works.
            let raw_default = RuntimeTuning::default().screenshot_median_threshold;
            let via_get = get(|t| t.screenshot_median_threshold);
            assert_eq!(raw_default, via_get);
        }
    }
}

// ─── Tests: golden values + acceptance invariants ─────────────────────────

#[cfg(test)]
mod tests {
    //! W44-211 acceptance tests for the tuning re-export hub.

    /// Tuning-drift golden test (acceptance gate (f)).
    ///
    /// Captures the canonical default value of every shared discriminator
    /// alias. Any future change to a const that drifts the alias value
    /// will trip the compile-time assert in
    /// [`super::discriminator_thresholds`] AND this runtime test. A
    /// failure here means a `pub(crate) const` value moved without
    /// updating the alias — either fix the alias or update the const
    /// intentionally (and regenerate hash-locks).
    #[test]
    fn discriminator_threshold_aliases_match_canonical_sites() {
        use super::discriminator_thresholds::*;
        // 4-site mask_p25=85.0 cluster
        assert_eq!(SMART_ZENJXL_PHOTO_MASK_P25_MIN, 85.0);
        assert_eq!(W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN, 85.0);
        assert_eq!(W44_150_PHOTO_W44_117_MASK_P25_MIN, 85.0);
        assert_eq!(W44_151_HIGH_MASK_P25_MIN, 85.0);
        assert_eq!(W44_168_SMOOTH_MASK_P25_MIN, 85.0);

        // 4-site mask_median=95.0 cluster
        assert_eq!(SCREENSHOT_MEDIAN_THRESHOLD, 95.0);
        assert_eq!(CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD, 95.0);
        assert_eq!(W44_168_SCREENSHOT_MEDIAN_MIN, 95.0);
        assert_eq!(super::buttloop::SCREENSHOT_MEDIAN_THRESHOLD, 95.0);
        assert_eq!(super::splines::SCREENSHOT_MEDIAN_MASK_THRESHOLD, 95.0);

        // mask_median=50.0 cluster (HIGH_D_PHOTO_SMOOTH band)
        assert_eq!(HIGH_D_PHOTO_SMOOTH_THRESHOLD, 50.0);
        assert_eq!(SINGLE_PASS_ENTROPY_SMOOTH_PHOTO_MAX_MEDIAN, 50.0);

        // m3 / fcbr / edge_density anchors
        assert_eq!(W44_91_M3_COLOURFULNESS_MIN, 80.0);
        assert_eq!(W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN, 25.0);
        assert_eq!(W44_91_FCBR_MAX, 0.01);
        assert_eq!(W44_96_FCBR_MAX, 0.01);
        assert_eq!(W44_124_DCT32_KEEP_M3_MIN, 60.0);
        assert_eq!(W44_124_DCT32_KEEP_EDGE_DENSITY_MAX, 0.05);
    }

    /// Golden test (acceptance gate (f)): canonical buttloop values.
    #[test]
    fn buttloop_canonical_values() {
        use super::buttloop::*;
        assert_eq!(DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE, 4.0);
        assert_eq!(BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE, 3.5);
        assert_eq!(BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE, 2.0);
        assert_eq!(BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX, 30.0);
        assert_eq!(ADAPTIVE_QUANT_QF_SEED_SCALE_MAX_EFFORT, 7);
        assert_eq!(DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6, 2.0);
        assert_eq!(DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7, 3.0);
        assert_eq!(W44_120_EPF_SEED_MIN_DISTANCE, 1.0);
        assert_eq!(W44_140_EPF_SEED_FADE_MAX, 1.5);
        assert_eq!(W44_142_EPF_SEED_SUPPRESS_M3_MIN, 60.0);
        assert_eq!(W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE, 1.5);
        assert_eq!(W44_176_TERMINAL_CLASS_LUMA_VAR_MIN, 1500.0);
        assert_eq!(W44_176_TERMINAL_CLASS_LUMA_VAR_MAX, 2200.0);
        assert_eq!(W44_176_TERMINAL_CLASS_FCBR_MIN, 0.70);
        assert_eq!(LIBJXL_INIT_MUL, 0.6);
        assert_eq!(DEFAULT_CUR_POW_LOW, 0.2);
        assert_eq!(DEFAULT_DISTANCE_SPLIT, 2.0);
    }

    /// Golden test (acceptance gate (f)): top-level gate constants.
    #[test]
    fn gates_canonical_values() {
        use super::gates::*;
        assert_eq!(SMALL_IMAGE_PIXEL_THRESHOLD, 1_000_000);
        assert_eq!(LARGE_IMAGE_PIXEL_THRESHOLD, 4_000_000);
        assert_eq!(LARGE_E9_TREE_MAX_BUCKETS, 192);
        assert_eq!(LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD, 500_000);
        assert_eq!(LOSSY_LOW_DISTANCE_THRESHOLD, 2.0);
        assert_eq!(CONTENT_CLASS_MIN_PIXELS, 65_536);
    }

    /// Golden test: DC tree effort gates.
    #[test]
    fn dc_tree_canonical_values() {
        use super::dc_tree::*;
        assert_eq!(DC_TREE_VARIABLE_TRIAL_MIN_EFFORT, 8);
        assert_eq!(DC_TREE_VARIABLE_PREDICTOR_FULL_MIN_EFFORT, 9);
    }

    /// Sanity: every submodule compiles + at least one re-export is
    /// reachable. Detects accidental visibility regressions on the
    /// `pub use` paths.
    #[test]
    fn every_section_reachable() {
        let _ = super::discriminator_thresholds::SCREENSHOT_MEDIAN_THRESHOLD;
        let _ = super::buttloop::LIBJXL_INIT_MUL;
        let _ = super::coeff_orders::NUM_ORDER_BUCKETS;
        let _ = super::epf::EPF_DEFAULT_SHARPNESS;
        let _ = super::patches::MAX_PATCH_SIZE;
        let _ = super::splines::MAX_SPLINES;
        let _ = super::gaborish::ADAPTIVE_TILE;
        let _ = super::noise::NUM_NOISE_POINTS;
        let _ = super::cfl::K_INV_COLOR_FACTOR;
        let _ = super::cfl::NEWTON_EPS_DEFAULT;
        let _ = super::cfl::NEWTON_EPS_LIBJXL;
        let _ = super::quant_weights::NUM_VALID_STRATEGIES;
        let _ = super::ac_strategy::K_BIAS;
        let _ = super::dc_tree::DC_TREE_VARIABLE_TRIAL_MIN_EFFORT;
        let _ = super::gates::SMALL_IMAGE_PIXEL_THRESHOLD;
        let _ = super::squeeze::SQUEEZE_LUMA_QTABLE_LEN;
        // W44-217: coupling-skeleton module is reachable. The actual
        // coupling fns are `unimplemented!()` — see the dedicated
        // marker tests in `super::coupling::tests`.
        let _ = super::coupling::CouplingClass::Additive;
    }
}
