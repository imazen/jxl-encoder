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
// W44-211: `pub(crate)` so `crate::tuning::dc_tree` can re-export
// `DC_TREE_VARIABLE_TRIAL_MIN_EFFORT` / `_PREDICTOR_FULL_MIN_EFFORT`.
pub(crate) mod bitstream;
mod block_extract;
// The quantization-refinement loop (the "buttloop") was renamed from
// `butteraugli_loop` → `perceptual_loop` in cvvdp-fork Phase 4
// (2026-05-24 — see `docs/RFC_CVVDP_FORK.md` §2.1 and
// `docs/RFC_CVVDP_PHASE4_BRIEF.md` §1). The historical function name
// `run_buttloop` is preserved (load-bearing in W44-* commit messages and
// docs); only the file/module name changes. A backward-compat alias
// `butteraugli_loop` re-exports the new module so existing `use
// crate::vardct::butteraugli_loop::...` import sites across the crate
// keep working without a 30+ file touch.
#[cfg(feature = "butteraugli-loop")]
pub(crate) mod perceptual_loop;
// Always-compiled adaptive-quant qf-seed pre-scale tuning surface extracted
// from `perceptual_loop` (which is gated behind `butteraugli-loop`). The core
// encoder + the `tuning::buttloop` re-export reference these symbols
// unconditionally, so this module must compile in encode-only builds.
pub(crate) mod perceptual_tuning;
/// Backward-compat alias for the pre-Phase-4 module name. New code SHOULD
/// import from `crate::vardct::perceptual_loop`; this alias exists so
/// existing call-sites compile unchanged. cvvdp-fork Phase 4 (2026-05-24).
#[cfg(feature = "butteraugli-loop")]
pub(crate) use perceptual_loop as butteraugli_loop;
// Pluggable perceptual-metric backend (renamed from `butteraugli_backend`
// in cvvdp-fork Phase 2, 2026-05-24 — see docs/RFC_CVVDP_FORK.md §2.1).
// Hosts `PerceptualBackend` trait + CPU/GPU butteraugli impls + (cvvdp-fork
// Phase 3, 2026-05-24) routes to the cvvdp impls in `cvvdp_backend`
// when the caller opts in via `LossyConfig::with_cvvdp_loop`.
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
#[cfg(feature = "butteraugli-loop")]
pub(crate) mod perceptual_backend;
// cvvdp-fork Phase 3 (2026-05-24): cvvdp-based `PerceptualBackend`
// implementations. Gated on the `cvvdp-loop` cargo feature (which
// itself implies `butteraugli-loop` — the trait surface lives in
// `perceptual_backend`). Hosts `GpuCvvdpBackend` (wraps
// `cvvdp_gpu::CvvdpOpaque` via Agent B's `*_from_linear_planes_*` API,
// zenmetrics master `8b658b4`) plus a stub `CpuCvvdpBackend` reserved
// for Phase 5 `cvvdp-cpu` integration. See `docs/RFC_CVVDP_FORK.md` §2.1
// and `docs/RFC_CVVDP_PHASE3_BRIEF.md` for the deliverable shape.
#[cfg(feature = "cvvdp-loop")]
pub(crate) mod cvvdp_backend;
// cvvdp-fork Phase 4 (2026-05-24): per-distance JOD calibration table.
// Read by `perceptual_loop::run_buttloop` when the active backend is
// cvvdp to scale `target_distance` (butteraugli units) into a cvvdp-
// direction `target_score` via the seed table at `cvvdp_targets.rs`.
// See `docs/RFC_CVVDP_PHASE4_BRIEF.md` Step 3.
#[cfg(feature = "cvvdp-loop")]
pub(crate) mod cvvdp_targets;
// Phase 1 of RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md` (2026-05-26):
// inverse-direction `target_score → effective_distance` calibration
// table for the butteraugli arm of the multi-metric loop. Closes the
// implicit-identity gap left by multi-metric Phase 0 commit `23da77b1`
// (the `with_perceptual_target_score(Some(_))` setter was a no-op for
// all three metrics because the dispatch site in `perceptual_loop.rs`
// + `perceptual_backend.rs` ignored the field). Default
// `perceptual_target_score = None` keeps the identity arm; hash-locks
// 36/36 stay byte-identical on the default path.
#[cfg(feature = "butteraugli-loop")]
pub(crate) mod butteraugli_targets;
// zensim-fork Phase 3 (2026-05-25): zensim backend impl for the
// perceptual quantization loop (RFC `docs/RFC_ZENSIM_FORK_PLAN.md` §5).
// Gated on the zensim cargo features. Hosts `CpuZensimBackend` (feature
// `zensim-loop`, wraps `zensim::Zensim` + linear-planar diffmap) and
// `GpuZensimBackend` (feature `zensim-loop-gpu`, wraps
// `zensim_gpu::ZensimOpaque` via Phase 1 commit `1175b49` on zenmetrics
// master). Phase 4 (2026-05-25) added the per-distance target table at
// `zensim_targets.rs`; per-block reducer constants reuse butter
// defaults (Phase 8-zensim follow-on may refit if Pareto shows < 85%).
// See `docs/RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md` for the trait +
// API surface and `docs/RFC_ZENSIM_FORK_PLAN.md` §6 for the Phase 4
// brief.
//
// zensim-fork Phase 4 (2026-05-25): per-distance zensim calibration
// table. Read by `perceptual_loop::run_buttloop` when the active
// backend is zensim to scale `target_distance` (butteraugli units)
// into a zensim butter-direction `target_score` via the seed table at
// `zensim_targets.rs`. See `docs/RFC_ZENSIM_FORK_PLAN.md` §6 Step 3.
pub(crate) mod dc_coding;
pub(crate) mod dc_tree_learn;
pub mod dct;
pub(crate) mod debug_log;
#[cfg(any(feature = "zensim-loop", feature = "zensim-loop-gpu"))]
pub(crate) mod zensim_backend;
#[cfg(any(feature = "zensim-loop", feature = "zensim-loop-gpu"))]
pub(crate) mod zensim_targets;
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
pub(crate) mod quantize_wp;
pub(crate) mod reconstruct;
mod static_codes;
pub mod transform;
pub(crate) mod w44_181_dump;
pub(crate) mod w44_182_dump;
pub(crate) mod w44_76_dump;
pub(crate) mod w44_audit_8_p4_dump;
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
    // The atomic sweep-override statics live in the `butteraugli-loop`-gated
    // `perceptual_loop` (only meaningful when the loop runs).
    pub use super::butteraugli_loop::{
        CUR_POW_X1000_HIGH, CUR_POW_X1000_LOW, DISTANCE_SPLIT_X1000, MAX_INCREASE_X1000_HIGH,
        MAX_INCREASE_X1000_HIGH_SCREENSHOT, MAX_INCREASE_X1000_LOW,
    };
    // The production-default constants moved to the always-compiled
    // `perceptual_tuning`; re-export them from there so this `pub use`
    // surfaces their true `pub` visibility (the `perceptual_loop` glob
    // re-export is `pub(crate)`, which would cap them at crate-private).
    pub use super::perceptual_tuning::{
        DEFAULT_CUR_POW_HIGH, DEFAULT_CUR_POW_LOW, DEFAULT_DISTANCE_SPLIT,
        DEFAULT_MAX_INCREASE_HIGH, DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT, DEFAULT_MAX_INCREASE_LOW,
        SCREENSHOT_MEDIAN_THRESHOLD,
    };
}

/// W44-PHASE3-B5b: process-global counters for the GPU butteraugli
/// divergence detector. Bench harnesses
/// (`examples/w44_phase3_b5b_divergence_detector_ab.rs`) call
/// `reset()` between cells and `snapshot()` to read per-cell observations.
///
/// **Not part of the stable API.** Counters exist only to instrument the
/// W44-PHASE3-B5b validation chunk; production encoder never reads
/// them. Gated on `gpu-butteraugli` because the counters live alongside
/// the GPU backend that updates them.
#[cfg(feature = "gpu-butteraugli")]
#[doc(hidden)]
pub mod __b5b_counters {
    pub use super::perceptual_backend::b5b_counters::{Snapshot, reset, snapshot};
}

#[cfg(test)]
mod tests;
