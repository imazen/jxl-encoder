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
    pub(crate) use crate::vardct::butteraugli_loop::ADAPTIVE_QUANT_QF_SEED_SCALE_MAX_EFFORT;
    pub(crate) use crate::vardct::butteraugli_loop::BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX;
    pub(crate) use crate::vardct::butteraugli_loop::BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE;
    pub(crate) use crate::vardct::butteraugli_loop::BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE;
    pub(crate) use crate::vardct::butteraugli_loop::DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6;
    pub(crate) use crate::vardct::butteraugli_loop::DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7;
    pub(crate) use crate::vardct::butteraugli_loop::DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE;
    pub(crate) use crate::vardct::butteraugli_loop::DEFAULT_CUR_POW_HIGH;
    pub(crate) use crate::vardct::butteraugli_loop::DEFAULT_CUR_POW_LOW;
    pub(crate) use crate::vardct::butteraugli_loop::DEFAULT_DISTANCE_SPLIT;
    pub(crate) use crate::vardct::butteraugli_loop::DEFAULT_MAX_INCREASE_HIGH;
    pub(crate) use crate::vardct::butteraugli_loop::DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT;
    pub(crate) use crate::vardct::butteraugli_loop::DEFAULT_MAX_INCREASE_LOW;
    pub(crate) use crate::vardct::butteraugli_loop::LIBJXL_INIT_MUL;
    pub(crate) use crate::vardct::butteraugli_loop::SCREENSHOT_MEDIAN_THRESHOLD;
    pub(crate) use crate::vardct::butteraugli_loop::W44_120_EPF_SEED_MIN_DISTANCE;
    pub(crate) use crate::vardct::butteraugli_loop::W44_140_EPF_SEED_FADE_MAX;
    pub(crate) use crate::vardct::butteraugli_loop::W44_142_EPF_SEED_SUPPRESS_EDGE_DENSITY_MAX;
    pub(crate) use crate::vardct::butteraugli_loop::W44_142_EPF_SEED_SUPPRESS_M3_MIN;
    pub(crate) use crate::vardct::butteraugli_loop::W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE;
    pub(crate) use crate::vardct::butteraugli_loop::W44_145_PER_BLOCK_MASK_HIGH;
    pub(crate) use crate::vardct::butteraugli_loop::W44_145_PER_BLOCK_MASK_LOW;
    pub(crate) use crate::vardct::butteraugli_loop::W44_176_TERMINAL_CLASS_FCBR_MIN;
    pub(crate) use crate::vardct::butteraugli_loop::W44_176_TERMINAL_CLASS_LUMA_VAR_MAX;
    pub(crate) use crate::vardct::butteraugli_loop::W44_176_TERMINAL_CLASS_LUMA_VAR_MIN;
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
    //! **Each function in this module is a PROPOSED coupling expansion**
    //! that would let a small set of high-level Tier-2 knobs (W44-221+)
    //! drive the full 6-parameter [`super::runtime::RuntimeTuning`]
    //! vector while respecting the empirical interactions. **None are
    //! implemented yet** — every body is `unimplemented!()`; the
    //! signatures + doc-comments capture the hypothesis, and tests
    //! assert the marker so future agents (W44-218+) know to fill them
    //! in.
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

    /// **PROPOSED** (W44-218): p1 (`smart_zenjxl_photo_mask_p25_min`)
    /// × p2 (`screenshot_median_threshold`) coupling on encoded_bytes.
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
    /// **Hypothesised mechanism**: GATED. p1 gates W44-166 variant Z
    /// admission (`mask_p25 >= p1`); p2 gates the screenshot family
    /// (`mask_median >= p2`). When both gates open simultaneously, the
    /// content-class lift table swaps; otherwise the base table is used.
    /// The coupling is the discrete dispatch decision, not a smooth
    /// interaction.
    ///
    /// **Tier-2 use**: a single "smoothness_bias" knob ∈ [0, 1]
    /// reparameterises `(p1, p2)` along a 1-D ridge that preserves the
    /// joint dispatch decision while sweeping the photo↔screen boundary.
    pub fn p1_p2_smoothness_dispatch_ridge(_smoothness_bias: f32) -> (f32, f32) {
        unimplemented!(
            "W44-218 scope: derive the 1-D ridge through (p1, p2) space \
             that preserves the W44-166/W44-168 dispatch decision for \
             each image class. Calibrate against \
             benchmarks/sweeps/w44-216-stage-b/analysis/stratum_pdp/."
        )
    }

    /// **PROPOSED** (W44-218): p3 (`buttloop_default_screenshot_qf_seed_scale`)
    /// × p6 (`adaptive_quant_screenshot_qf_seed_scale_e7`) coupling on
    /// ssim2.
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
    /// **Tier-2 use**: a single "screenshot_quant_aggressiveness" knob
    /// reparameterises `(p3, p6)` along a single ray that respects the
    /// saturation bound, capped at ~6× joint lift.
    pub fn p3_p6_screenshot_qac_lift(_screenshot_quant_aggressiveness: f32) -> (f32, f32) {
        unimplemented!(
            "W44-218 scope: fit the saturation curve from \
             benchmarks/sweeps/w44-216-stage-b/analysis/stratum_pdp/\
             pdp_p3_butt_qf_scale_x_p6_aq_qf_e7_classscreen_ssim2.png \
             and derive the joint-lift ray that stays below saturation."
        )
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
        _buttloop_screen_d_gate: f32,
        _adaptive_quant_aggressiveness: f32,
    ) -> (f32, f32) {
        unimplemented!(
            "W44-218 scope: derive the soft-OR rule between buttloop \
             dispatch (driven by p4) and adaptive_quant pre-scale \
             (driven by p5) at screen-class cells. See \
             benchmarks/sweeps/w44-216-stage-b/analysis/stratum_pdp/\
             pdp_p4_butt_min_dist_x_p6_aq_qf_e7_classscreen_ssim2.png \
             (GATED-by-p4 surface) for the structure."
        )
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
    pub fn p5_p6_effort_conditional_lift(_screen_quant_lift: f32) -> (f32, f32) {
        unimplemented!(
            "W44-218 scope: fit the diagonal ridge (k × 2.0, k × 3.0) \
             k ∈ [0.5, 2.0] and the saturation cap. See \
             benchmarks/sweeps/w44-216-stage-b/analysis/stratum_pdp/\
             pdp_p5_aq_qf_e56_x_p6_aq_qf_e7_classscreen_ssim2.png for \
             the L-shaped saturation surface."
        )
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
        _screen_quant_lift: f32,
        _buttloop_screen_d_gate: f32,
    ) -> (f32, f32) {
        unimplemented!(
            "W44-218 scope: the (p4, p6) coupling shares the \
             buttloop_screen_d_gate knob with p4_p5_buttloop_*. \
             See benchmarks/sweeps/w44-216-stage-b/analysis/stratum_pdp/\
             pdp_p4_butt_min_dist_x_p6_aq_qf_e7_classscreen_ssim2.png \
             for the GATED-by-p4 surface."
        )
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
    pub fn p1_p3_mutually_exclusive_dispatch(
        _smoothness_bias: f32,
        _screen_aggr: f32,
    ) -> (f32, f32) {
        unimplemented!(
            "W44-218 scope: confirm mutual-exclusion via per-image \
             logging in the W44-219 sweep. Until then, expose as two \
             independent Tier-2 knobs."
        )
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
    pub fn p3_p4_photo_high_d_gate(_buttloop_screen_d_gate: f32, _screen_aggr: f32) -> (f32, f32) {
        unimplemented!(
            "W44-218 scope: optional 3-arity refinement; not blocking. \
             Consider after Tier-2 prototype lands."
        )
    }

    /// **PROPOSED** (W44-222): rebuild the full
    /// [`super::runtime::RuntimeTuning`] from the Tier-2 knob set.
    ///
    /// Composes the individual coupling fns above into the 6-vector
    /// the production encoder consumes. Tier-2 knobs (W44-221):
    /// - `smoothness_bias` ∈ [0, 1] → (p1, p2)
    /// - `screen_quant_lift` ∈ [0.5, 2.0] → (p5, p6) + interacts
    ///   with p3 via [`p3_p6_screenshot_qac_lift`]
    /// - `buttloop_screen_d_gate` ∈ [1.5, 5.0] → p4 + interacts
    ///   with p3/p5/p6 via the gated couplings above
    ///
    /// The default values of all 3 knobs MUST produce the production
    /// defaults `(85, 95, 4, 3.5, 2, 3)` byte-for-byte (regression
    /// test contract for W44-222).
    ///
    /// Gated on `feature = "tuning-override"` because the return type
    /// `super::runtime::RuntimeTuning` only exists under that feature.
    #[cfg(feature = "tuning-override")]
    pub fn expand_knobs_to_runtime(
        _smoothness_bias: f32,
        _screen_quant_lift: f32,
        _buttloop_screen_d_gate: f32,
    ) -> super::runtime::RuntimeTuning {
        unimplemented!(
            "W44-222 scope (NOT W44-217 or W44-218): compose the per-\
             pair coupling fns above into a full RuntimeTuning expansion. \
             Default (smoothness_bias=0.5, screen_quant_lift=1.0, \
             buttloop_screen_d_gate=3.5) MUST round-trip to \
             RuntimeTuning::default() byte-for-byte."
        )
    }

    #[cfg(test)]
    mod tests {
        //! W44-217 marker tests: every coupling fn must remain
        //! `unimplemented!()` until a W44-218+ chunk fills it in.
        //!
        //! Each test calls the fn inside `std::panic::catch_unwind`
        //! and asserts the panic message starts with "W44-218 scope"
        //! (or W44-222 for the expander). When a future chunk
        //! implements a coupling, the corresponding test will FAIL
        //! — that's the signal the test should be updated to verify
        //! the new behaviour.
        //!
        //! Do NOT remove these tests. They're the lock-in that
        //! prevents accidental partial implementations.

        use super::*;

        fn expect_unimplemented<F>(f: F, expected_prefix: &str)
        where
            F: FnOnce() + std::panic::UnwindSafe,
        {
            let result = std::panic::catch_unwind(f);
            assert!(result.is_err(), "expected unimplemented!() panic");
            let panic_payload = result.unwrap_err();
            let msg = if let Some(s) = panic_payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                String::new()
            };
            assert!(
                msg.contains(expected_prefix),
                "expected panic message containing {:?}, got {:?}",
                expected_prefix,
                msg
            );
        }

        #[test]
        fn p1_p2_smoothness_dispatch_ridge_unimplemented() {
            expect_unimplemented(
                || {
                    let _ = p1_p2_smoothness_dispatch_ridge(0.5);
                },
                "W44-218 scope",
            );
        }

        #[test]
        fn p3_p6_screenshot_qac_lift_unimplemented() {
            expect_unimplemented(
                || {
                    let _ = p3_p6_screenshot_qac_lift(1.0);
                },
                "W44-218 scope",
            );
        }

        #[test]
        fn p4_p5_buttloop_vs_adaptive_quant_dispatch_unimplemented() {
            expect_unimplemented(
                || {
                    let _ = p4_p5_buttloop_vs_adaptive_quant_dispatch(3.5, 1.0);
                },
                "W44-218 scope",
            );
        }

        #[test]
        fn p5_p6_effort_conditional_lift_unimplemented() {
            expect_unimplemented(
                || {
                    let _ = p5_p6_effort_conditional_lift(1.0);
                },
                "W44-218 scope",
            );
        }

        #[test]
        fn p4_p6_e7_buttloop_synergy_unimplemented() {
            expect_unimplemented(
                || {
                    let _ = p4_p6_e7_buttloop_synergy(1.0, 3.5);
                },
                "W44-218 scope",
            );
        }

        #[test]
        fn p1_p3_mutually_exclusive_dispatch_unimplemented() {
            expect_unimplemented(
                || {
                    let _ = p1_p3_mutually_exclusive_dispatch(0.5, 1.0);
                },
                "W44-218 scope",
            );
        }

        #[test]
        fn p3_p4_photo_high_d_gate_unimplemented() {
            expect_unimplemented(
                || {
                    let _ = p3_p4_photo_high_d_gate(3.5, 1.0);
                },
                "W44-218 scope",
            );
        }

        // The expander test is `tuning-override`-gated because the
        // fn returns `runtime::RuntimeTuning`, which is only in scope
        // under that feature.
        #[cfg(feature = "tuning-override")]
        #[test]
        fn expand_knobs_to_runtime_unimplemented() {
            expect_unimplemented(
                || {
                    let _ = expand_knobs_to_runtime(0.5, 1.0, 3.5);
                },
                "W44-222 scope",
            );
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
    #[derive(Clone, Debug)]
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
