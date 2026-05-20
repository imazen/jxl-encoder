// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! VarDCT (lossy) encoder for JPEG XL.
//!
//! Variable-DCT encoding transforms image blocks using DCT of various sizes,
//! quantizes coefficients with perceptual weighting, and entropy codes the result.
//!
//! Supports 19 of 27 DCT strategies (all that libjxl evaluates through effort 9),
//! Huffman or ANS entropy coding, custom coefficient ordering, LZ77 backward
//! references, adaptive quantization, chroma-from-luma, gaborish inverse,
//! noise synthesis, and butteraugli-guided rate control.

pub(crate) mod ac_context;
pub(crate) mod ac_group;
pub(crate) mod ac_strategy;
mod ac_strategy_search;
pub(crate) mod adaptive_quant;
mod afv;
mod bitstream;
mod block_extract;
#[cfg(feature = "butteraugli-loop")]
pub(crate) mod butteraugli_loop;
pub(crate) mod chroma_from_luma;
/// Chroma subsampling helpers — RGB → YCbCr conversion and
/// Sharp YUV 4:2:0 chroma downsample via the zenyuv crate.
/// Issue #47 chunk 3: foundational helpers for the eventual end-to-end
/// Sub420 / Sub422 / Sub440 lossy pipeline (chunk 4). See the module
/// docs for the public API.
#[cfg(feature = "chroma-subsampling")]
pub mod chroma_subsampling;
pub(crate) mod cluster;
pub(crate) mod coeff_order;
pub(crate) mod common;
pub(crate) mod context_tree;
pub(crate) mod dc_coding;
mod dc_tree_learn;
pub mod dct;
pub(crate) mod debug_log;
// libjxl enc_detect_dots.cc port (refs #19). Wired into encoder.rs at
// effort >= 7, distance >= 3.0; dots get promoted to a fresh
// PatchesData via from_dots() and travel through the regular patch
// subtract + decode pipeline. `ConnectedComponent::pixels` is
// retained for future tuning even though only the bounds field is
// consumed today.
#[allow(dead_code)]
pub(crate) mod dot_detection;
pub(crate) mod encoder;
pub(crate) mod entropy_code;
pub(crate) mod epf;
pub(crate) mod extras;
pub(crate) mod frame;
pub(crate) mod gaborish;
/// HDR-aware perceptual loss dispatch for the butteraugli quantization loop
/// (EX-J11 chunk 1). See module docs for the chunk-1/chunk-2 split — chunk 1
/// ships only the [`hdr_metrics::HdrLoss`] enum + validation; chunk 2 lands
/// the HDR-VDP-2 maths.
///
/// Gated behind `feature = "butteraugli-loop"` because the loss dispatch
/// is only meaningful inside the butteraugli quantization loop.
#[cfg(feature = "butteraugli-loop")]
pub mod hdr_metrics;
/// VDP2-lite: a calibrated subset of HDR-VDP-2 sufficient for in-loop
/// quality steering on HDR (PQ / HLG / BT.2100) content. Chunk-2 deliverable
/// for EX-J11 (see [`hdr_metrics`] for the chunk-1/2 split).
///
/// Gated behind `feature = "butteraugli-loop"` because it's only consumed
/// inside the butteraugli quantization loop.
#[cfg(feature = "butteraugli-loop")]
pub(crate) mod hdr_vdp2_lite;
pub(crate) mod lf_frame;
pub(crate) mod noise;
pub(crate) mod patches;
#[cfg(any(feature = "rate-control", feature = "__pre_quantized"))]
pub(crate) mod precomputed;
#[cfg(feature = "rate-control")]
pub mod rate_control;
/// Region-source abstraction for XYB planes consumed by
/// [`transform::VarDctEncoder::transform_and_quantize_with_source`]
/// (streaming refactor chunk 8b, jxl-encoder#11). See module docs for
/// scope; whole-image impl is the only one shipping in chunk 8b.
pub(crate) mod region_source;
pub(crate) mod resampling;
pub(crate) mod simplify_invisible;
pub(crate) mod splines;
#[cfg(feature = "ssim2-loop")]
mod ssim2_loop;
#[cfg(feature = "rate-control")]
mod tile_distmap;
#[cfg(feature = "zensim-loop")]
mod zensim_loop;

#[cfg(feature = "investigate-adjust-quant-block-ac")]
pub mod aqba_diag;
pub(crate) mod quant;
pub(crate) mod quantize;
pub(crate) mod reconstruct;
mod static_codes;
pub mod transform;
pub(crate) mod w44_76_dump;
pub(crate) mod xyb;

pub use encoder::{VarDctEncoder, VarDctOutput};
#[cfg(any(feature = "rate-control", feature = "__pre_quantized"))]
pub use precomputed::EncoderPrecomputed;
#[cfg(feature = "rate-control")]
pub use rate_control::RateControlConfig;

/// **Investigation-only** (W44-9 Sub-chunk B): toggle the process-wide
/// override that forces DCT8 entropy estimation through the explicit
/// non-fused fallback (separate DCT then separate entropy_estimate_coeffs)
/// instead of the fused AVX2 kernel.
///
/// Default is `false` (uses fused path). Flipping has no effect on
/// production callers — this is purely for A/B harnesses investigating
/// whether FMA op-ordering in the fused kernel gives DCT8 a borderline
/// cost advantage that explains F-D wedge AC overspend.
///
/// Gated behind the `__expert` cargo feature.
#[cfg(feature = "__expert")]
pub fn set_force_unfused_dct8_entropy(v: bool) {
    ac_strategy::set_force_unfused_dct8_entropy(v);
}

/// **Investigation-only** (W44-9 Sub-chunk B): reset per-branch hit
/// counters used to verify the override is taking effect.
#[cfg(feature = "__expert")]
pub fn reset_dct8_branch_counters() {
    ac_strategy::reset_dct8_branch_counters();
}

/// **Investigation-only** (W44-9 Sub-chunk B): read per-branch hit
/// counters. Returns `(fused_hits, unfused_hits)` since the last reset.
#[cfg(feature = "__expert")]
pub fn dct8_branch_counters() -> (u64, u64) {
    ac_strategy::dct8_branch_counters()
}

/// Debug hook for capturing the butteraugli loop's internal reconstruction at
/// the final iteration. **Not part of the stable API** — for the drift-investigation
/// Layer-1 test only. Gated by `feature = "__internal_recon_hook"`.
///
/// See [`crate::vardct::butteraugli_loop`] module docs for the rationale and
/// memory/quality_drift_investigation_2026-05-15.md for the bug context.
#[cfg(feature = "__internal_recon_hook")]
#[doc(hidden)]
pub mod __recon_hook {
    pub use super::butteraugli_loop::recon_hook::*;
}

/// W44-114 AFV IDCT parity test hook — re-exports AFV transform entry
/// points for the `tests/afv_idct_parity.rs` impulse-response test.
///
/// **Not part of the stable API.** Used only to verify bit-parity of
/// `inverse_afv_transform` and `afv_transform_from_pixels` against a
/// hand-ported libjxl `AFVIDCT4x4` reference (chunk W44-114).
#[doc(hidden)]
pub mod __afv {
    pub use super::afv::{afv_transform_from_pixels, inverse_afv_transform};
}

/// Sweep-only atomic overrides for the distance-aware butteraugli-loop
/// tuning scaffolding (W38-2 #3.1; infrastructure ported from GPU
/// commit `d75bf7c`).
///
/// **Not part of the stable API.** These statics let an A/B harness
/// hot-swap `cur_pow` / `max_increase` / split-distance values per
/// regime without rebuilding. Production code never sets these — they
/// default to the values documented in
/// [`crate::vardct::butteraugli_loop`] (both regimes: `cur_pow=0.2`,
/// `max_increase=100.0` ≈ libjxl's "no cap"). The GPU's tuned LOW
/// values (`cur_pow=0.5`, `max_increase=1.3`) regressed CPU
/// RD-pareto in A/B and are NOT the CPU production default.
///
/// `i32::MIN` on any of `CUR_POW_X1000_*` / `MAX_INCREASE_X1000_*`
/// means "use the production default". `DISTANCE_SPLIT_X1000`
/// initialises to `2000` (= 2.0) and is read every time.
///
/// See `benchmarks/buttloop_distance_split_port_*.{tsv,meta}` for
/// reference sweep output and `examples/buttloop_distance_split_ab.rs`
/// for the canonical harness.
///
/// Gated behind `feature = "butteraugli-loop"` because every re-exported
/// item lives in [`butteraugli_loop`]. Without this gate, `wasm32-wasip1`
/// (and any other `--no-default-features --features "std"` build) fails
/// to compile the `pub use` below.
#[cfg(feature = "butteraugli-loop")]
#[doc(hidden)]
pub mod __buttloop_overrides {
    pub use super::butteraugli_loop::{
        CUR_POW_X1000_HIGH, CUR_POW_X1000_LOW, DEFAULT_CUR_POW_HIGH, DEFAULT_CUR_POW_LOW,
        DEFAULT_DISTANCE_SPLIT, DEFAULT_MAX_INCREASE_HIGH, DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT,
        DEFAULT_MAX_INCREASE_LOW, DISTANCE_SPLIT_X1000, MAX_INCREASE_X1000_HIGH,
        MAX_INCREASE_X1000_HIGH_SCREENSHOT, MAX_INCREASE_X1000_LOW, SCREENSHOT_MEDIAN_THRESHOLD,
    };
}

#[cfg(test)]
mod tests;
