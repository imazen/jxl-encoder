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
    }
}
