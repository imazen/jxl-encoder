// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JPEG XL encoder in pure Rust.
//!
//! This crate provides a complete JPEG XL encoder implementation, supporting
//! both lossless (modular) and lossy (VarDCT) encoding modes.

#![forbid(unsafe_code)]

extern crate alloc;

pub mod api;
pub mod bit_writer;
pub(crate) mod budget;
pub mod color;
pub mod container;
pub mod debug_rect;
// `effort` carries internal effort-derived knobs. Kept `pub` for
// backwards-compatibility with 0.3.0 (which re-exported `EffortProfile`
// at the crate root). The actual sweep / picker escape-hatch entry point
// (`LosslessConfig::with_effort_profile_override` / its lossy twin) is
// gated behind the `__expert` feature.
pub mod effort;
pub mod entropy_coding;
pub mod error;
pub(crate) mod f16;
pub mod headers;
pub(crate) mod icc;
pub mod image;
#[cfg(feature = "jpeg-reencoding")]
pub mod jpeg;
pub mod modular;
pub(crate) mod parallel;
pub mod trace;
pub mod validation;
#[cfg(test)]
mod validation_tests;
pub mod vardct;

#[cfg(feature = "convenience")]
pub mod convenience;

// Re-export new API as primary
pub use api::{
    AnimationFrame, AnimationParams, At, EncodeError, EncodeMode, EncodeRequest, EncodeResult,
    EncodeStats, EncoderMode, ImageMetadata, Limits, LosslessConfig, LosslessEncoder, LossyConfig,
    LossyEncoder, Lz77Method, NonFiniteAction, PixelLayout, ProgressiveMode, Quality, ResultAtExt,
    Stop, Unstoppable, at, calibrated_jxl_quality, quality_to_distance,
};
// `EffortProfile` was re-exported at the crate root in 0.3.0; it is now an
// **internal** type that drives the encoder's effort-derived decisions.
// The public picker / sweep escape hatch is the segmented
// `LossyInternalParams` / `LosslessInternalParams` pair, applied via
// `LossyConfig::with_internal_params` / `LosslessConfig::with_internal_params`
// (gated behind `__expert`). `EntropyMulTable` remains reachable because
// `LossyInternalParams::entropy_mul_table` carries it. The `EffortProfile`
// re-export is `#[doc(hidden)]` to discourage new use; existing callers
// that still reference it keep working.
#[doc(hidden)]
pub use effort::EffortProfile;
pub use effort::EntropyMulTable;
#[cfg(feature = "__expert")]
pub use effort::{LosslessInternalParams, LossyInternalParams};
pub use headers::color_encoding::{
    CIExy, ColorEncoding, ColorSpace, CustomPrimaries, Primaries, RenderingIntent,
    TransferFunction, WhitePoint,
};
pub use modular::rct::RctType;
pub use validation::ValidationError;
pub use vardct::splines::{Spline, SplinePoint};

#[cfg(feature = "convenience")]
pub use convenience::{
    encode_bgra8, encode_bgra8_lossless, encode_gray8, encode_gray8_lossless, encode_rgb8,
    encode_rgb8_lossless, encode_rgba8, encode_rgba8_lossless,
};

/// Group dimension in pixels (256x256 groups).
pub const GROUP_DIM: usize = 256;

/// DCT block dimension (8x8 blocks).
pub const BLOCK_DIM: usize = 8;

/// Size of a single DCT block (64 coefficients).
pub const BLOCK_SIZE: usize = BLOCK_DIM * BLOCK_DIM;

/// JXL signature bytes.
pub const JXL_SIGNATURE: [u8; 2] = [0xFF, 0x0A];

/// Test path helpers for integration tests and examples.
///
/// Provides configurable paths via environment variables for corpus directories,
/// tool binaries, and output directories. Not part of the public API.
#[doc(hidden)]
pub mod test_helpers;

/// Re-exports of private vardct internals for downstream parity testing
/// (notably from `jxl-encoder-gpu`'s `forks::*` G5.1 parity tests).
///
/// **Not part of the stable API.** Items here may move or change at any time.
/// Gated behind the `__internals` cargo feature; off by default.
///
/// Naming convention: `pub use crate::path::to::Symbol;` — pure re-exports,
/// no wrapping logic. Wrapper functions (e.g., for `pub(crate)` impl methods
/// that can't be `pub use`d directly) live in `crate::vardct::__internals_wrappers`.
#[cfg(feature = "__internals")]
#[doc(hidden)]
pub mod __internals {
    // Pub re-exports of already-pub items in private/pub(crate) modules.
    pub use crate::vardct::chroma_from_luma::{ytob_ratio, ytox_ratio};
    pub use crate::vardct::quant::INV_DC_QUANT;
    // Wrappers around pub(crate) helpers and pub(crate) impl methods that
    // can't be `pub use`d directly.
    pub use crate::vardct::ac_strategy::compute_scaled_constants_free;
    pub use crate::vardct::epf::epf_step0_strip_free;
    pub use crate::vardct::quantize::adjust_quant_block_ac_free;
}

/// Pre-quantized AC entry point — accepts an already-prepared
/// [`vardct::precomputed::EncoderPrecomputed`] (built via
/// `EncoderPrecomputed::from_parts(...)`) plus a `quant_field`, runs
/// the bitstream emit path. Skips the encoder's own XYB conversion,
/// gaborish, CfL, masking, AC strategy search, and butteraugli
/// refinement — all of which the caller is expected to have computed
/// on their own pipeline (e.g. jxl-encoder-gpu's GPU pipeline).
///
/// This module is `#[doc(hidden)]` and the API is unstable; use only
/// from downstream crates that pin a specific jxl-encoder version.
#[cfg(feature = "__pre_quantized")]
#[doc(hidden)]
pub mod __pre_quantized {
    pub use crate::vardct::ac_strategy::AcStrategyMap;
    /// Compute the float quant field + masking from XYB planes.
    /// Returns `(quant_field_float, masking)`. Pair with
    /// `quantize_quant_field` to get the `u8` field for
    /// `encode_from_precomputed`. Mirrors what
    /// `EncoderPrecomputed::compute` does internally on the
    /// adaptive_quant path.
    pub use crate::vardct::adaptive_quant::compute_quant_field_float_free;
    /// Convert a per-block float quant field (matches the GPU
    /// pipeline's `aq_field` shape) to the per-block `u8` quant field
    /// the bitstream encode path expects. Multiplies each entry by
    /// `inv_scale` (from `DistanceParams::compute_for_profile`) and
    /// clamps to `[1, 255]`.
    pub use crate::vardct::adaptive_quant::quantize_quant_field;
    pub use crate::vardct::chroma_from_luma::CflMap;
    /// Compute a per-tile chroma-from-luma (`CflMap`) from XYB planes.
    /// `stride = padded_width`, `buf_height = padded_height`. Tiles are
    /// 64×64 pixels = 8×8 blocks. Use `use_newton=true` for effort >=7
    /// quality (Newton-Raphson fit; matches libjxl
    /// `enc_chroma_from_luma.cc` at `speed_tier <= kSquirrel`).
    pub use crate::vardct::chroma_from_luma::compute_cfl_map;
    /// CfL pass 2 refinement. Recomputes the CfL map using the
    /// per-block actual AC strategy and the per-block quantization
    /// factor (versus pass 1's forced-DCT8, q=1 fit). Mutates
    /// `cfl_map` in place. Required for parity with libjxl
    /// `enc_chroma_from_luma.cc` at `speed_tier <= kSquirrel` (i.e.
    /// effort >= 7) and the CPU encoder's `cfl_two_pass` profile gate.
    ///
    /// Inputs:
    /// - `cfl_map`: pass-1 result (e.g. from `compute_cfl_map`).
    /// - `xyb_x/y/b`: same XYB planes pass 1 used (gaborish-applied if
    ///   the encoder enabled it).
    /// - `stride`: padded width matching the XYB planes.
    /// - `xsize_blocks` / `ysize_blocks`: image block dims (CPU grid).
    /// - `ac_strategy`: final per-block strategy assignments.
    /// - `quant_field`: final per-block `u8` quant field (post
    ///   `quantize_quant_field` + `adjust_quant_field_with_distance`).
    /// - `quant_scale`: `DistanceParams::scale` matching the
    ///   `quant_field` / `inv_scale` pair (i.e. `1.0 / inv_scale`).
    /// - `use_newton` / `newton_eps` / `newton_max_iters`: same knobs
    ///   `compute_cfl_map` takes; pull from the same `EffortProfile`
    ///   (`cfl_newton`, `cfl_newton_eps`, `cfl_newton_max_iters`).
    ///
    /// Call after AC strategy + quant field are finalized and
    /// before `encode_from_precomputed`. Replace
    /// `EncoderPrecomputed::cfl_map` with the refined map (or call
    /// in place on the same `CflMap` instance).
    pub use crate::vardct::chroma_from_luma::refine_cfl_map;
    pub use crate::vardct::common::DCT_BLOCK_SIZE;
    pub use crate::vardct::encoder::VarDctEncoder;
    /// libjxl `INV_DC_QUANT[c]` — per-channel DC scale factor.
    /// Used by GPU producers building `quantize_dc_dct8` `inv_factor`:
    ///   `inv_factor_c = INV_DC_QUANT[c] * params.scale_dc`
    pub use crate::vardct::quant::INV_DC_QUANT;
    /// Pull the per-coefficient DCT8 quant weights for a channel
    /// (X = 0, Y = 1, B = 2). Caller passes the returned slice as
    /// the broadcast weights template for the GPU DCT8 quantize
    /// kernels.
    pub fn quant_weights_dct8(channel: usize) -> &'static [f32] {
        crate::vardct::quant::quant_weights(0, channel)
    }
    /// Diagnostic-only re-export: the per-channel CPU transform
    /// output `transform_and_quantize` produces. Used by the
    /// `__pre_quantized` parity test to isolate "is the entry point
    /// correct?" from "is the GPU producer correct?". Feed an
    /// instance of this directly into `encode_from_pre_quantized_ac`
    /// and compare its bitstream output to `encode_from_precomputed`.
    pub use crate::vardct::transform::TransformOutput;

    /// Default dead-zone thresholds for DCT8 (covered_x=covered_y=1)
    /// per channel. Mirrors `VarDctEncoder::default_thresholds`.
    pub fn default_thresholds_dct8(channel: usize) -> [f32; 4] {
        let mut t = if channel == 1 {
            [0.56_f32, 0.62, 0.62, 0.62]
        } else {
            [0.58_f32, 0.62, 0.62, 0.62]
        };
        // For DCT8, covered_x*covered_y == 1 < 4, so the X/B
        // multi-block reduction at enc_group.cc:66-72 doesn't fire.
        // Match VarDctEncoder::default_thresholds exactly.
        let _ = &mut t;
        t
    }
    /// `DistanceParams` carries the per-distance scaling constants
    /// (notably `inv_scale`) needed to convert a float quant field to
    /// `u8`. Construct via `DistanceParams::compute_for_profile(distance,
    /// &EffortProfile)`.
    pub use crate::vardct::frame::DistanceParams;
    /// Multi-block transform quant-field adjustment. Mirrors what
    /// `encode_from_precomputed` runs internally on its `quant_field`
    /// argument (and what `encode_image_lossy` runs pre-buttloop in
    /// the CPU encoder, `vardct/encoder.rs:1146`). Downstream callers
    /// who need the *adjusted* u8 quant field for an out-of-band step
    /// (notably `refine_cfl_map`) can run it manually on a copy
    /// without doing it twice — `encode_from_precomputed` is
    /// idempotent on already-adjusted fields under most conditions
    /// but the safe pattern is to hand it the raw `quantize_quant_field`
    /// output and only adjust a separate copy for `refine_cfl_map`.
    pub use crate::vardct::ac_strategy::adjust_quant_field_with_distance;
    pub use crate::vardct::noise::NoiseParams;
    pub use crate::vardct::precomputed::EncoderPrecomputed;
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "api_tests.rs"]
mod api_tests;

#[cfg(all(test, feature = "__expert"))]
#[path = "effort_expert_tests.rs"]
mod effort_expert_tests;
