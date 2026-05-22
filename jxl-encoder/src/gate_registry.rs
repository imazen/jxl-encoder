// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! # W44-193 — Production gate registry via `strategy_def!` macro
//!
//! Big-bang migration (per W44-190 RFC + W44-192 prototype + user
//! 2026-05-22 signoff on the "single PR" approach) of the 24
//! hand-written gates that previously lived as:
//!
//! - `api::EncoderImprovementsCustom` struct + `impl Default`
//! - `api::ResolvedImprovements` struct + `impl Default`
//! - `impl ResolvedImprovements` (libjxl / lean_faster / zenjxl /
//!   aggressive / from_custom)
//! - `api::apply_env_var_fallbacks`
//!
//! The [`jxl_encoder_macros::strategy_def!`] invocation below generates
//! all of the above (and per-gate divergence-table metadata consts the
//! W44-194 build-script will harvest). The hand-maintained
//! `EncoderStrategy` enum (with the `Custom(Box<...>)` variant) stays
//! in `api.rs` because it carries a load-bearing `resolve(&self,
//! overrides: &StrategyOverrides)` method whose `overrides` parameter
//! the macro does not yet support.
//!
//! ## Why a separate module
//!
//! The macro emits a fixed `<Name>EncoderImprovements` /
//! `<Name>ResolvedImprovements` / `<Name>EncoderStrategy` name shape.
//! Production has shipped `EncoderImprovementsCustom` and
//! `ResolvedImprovements` (no prefix on the second) for the entire
//! W44-127..W44-192 arc. To preserve public-API names byte-identically,
//! we run the macro here with `name = Custom` and re-export the
//! generated structs as type aliases under their existing names from
//! `api.rs`. The macro's `CustomEncoderStrategy` is unused (we keep
//! `api::EncoderStrategy` as the canonical enum); only the structs +
//! constructors + env-fallback fn migrate.
//!
//! ## Macro-limitation supplements
//!
//! Two production hand-written behaviours don't fit the W44-192 macro
//! syntax:
//!
//! 1. **Dual env-var feeding one gate** —
//!    [`crate::api::EpfSharpnessSeed`] is fed by both
//!    `JXL_W44_117_DISABLE` (→ `LegacyUniform4`) and
//!    `JXL_W44_120_EPF_SEED_MIN_DISTANCE` (→ `AutoW44_117 {
//!    min_distance }`); the disable env wins. The macro slot below
//!    carries the disable hook (precedence-wise correct because it
//!    runs first); the [`apply_w44_120_min_distance_env_fallback`]
//!    function in this module handles the second env-var as a
//!    post-step. Both run only when the resolved field is at its
//!    `Default::default()` value (mirrors macro semantics).
//!
//! 2. **`..Default::default()` short-hand inside `lean_faster()`** —
//!    The original hand-written constructor closed with
//!    `..Default::default()` for the perf-dispatch / effort-gate /
//!    block-ctx-map fields. The macro requires every strategy lists
//!    every gate explicitly (compile-time check). Each LeanFaster
//!    gate is listed inline below; the values shown match the
//!    pre-W44-193 `..Default::default()` resolution byte-for-byte (the
//!    fields are all at their type-level defaults under Zenjxl, which
//!    LeanFaster inherits via the Zenjxl-matching values).
//!
//! ## Divergence-table metadata
//!
//! Each gate carries `divergence_section` + `divergence_row_ref` metadata
//! that the macro emits as `pub const __CUSTOM_DIVERGENCE_<GATE>: &str`
//! constants (visible via `cargo doc --json` or a syn-based source
//! walker). The W44-194 build-script will harvest these to auto-generate
//! [`docs/LIBJXL_DIVERGENCES.md`](../../docs/LIBJXL_DIVERGENCES.md);
//! until then the hand-maintained table stays canonical and the
//! `divergence_row_ref` strings should be kept in sync with the table
//! row headings.
//!
//! ## Provenance
//!
//! - W44-190: RFC for the refactor (chose Solution B + the macro
//!   approach with proc-macro syntax, opt-in env hooks, big-bang
//!   migration).
//! - W44-192: shipped the macro + 3-gate prototype validation.
//! - W44-193 (THIS COMMIT): big-bang production migration of the 24
//!   gates from `api.rs`.

#![allow(dead_code)]

use crate::api::{
    AdaptiveQuantQfSeedPolicy, ButtloopQfSeedPolicy, Dct32SearchPolicy, Dct64SearchPolicy,
    EffortGate, EpfDispatch, EpfSharpnessSeed, HighDPhotoEntropyMulPolicy, PatchesDispatch,
    PixelLossDispatch, ScreenshotEntropyMulPolicy, SinglePassEntropyDispatch,
    SmoothPhotoDct64Policy,
};

// ──────────────────────────────────────────────────────────────────────
// Env-var parsers (per W44-192 README §5; signature
// `fn(&str) -> Option<T>`).
// ──────────────────────────────────────────────────────────────────────

/// `JXL_W44_184_FORCE_LIBJXL_NEWTON=1` → `Some(true)`. Anything else
/// (incl. unset) returns `None`, which keeps the resolved field at its
/// post-strategy / post-overrides value.
fn parse_bool_one(s: &str) -> Option<bool> {
    if s == "1" { Some(true) } else { None }
}

/// `JXL_W44_117_DISABLE=1` → `Some(EpfSharpnessSeed::LegacyUniform4)`
/// (force-off the W44-117 EPF sharpness seed compute regardless of
/// distance gate). Anything else returns `None`.
///
/// W44-120 `JXL_W44_120_EPF_SEED_MIN_DISTANCE` is fed through a
/// separate supplemental fallback ([`apply_w44_120_min_distance_env_fallback`])
/// because the W44-192 macro only supports one env hook per gate.
/// Disable wins (the macro hook fires first; once the field is no
/// longer at `Default::default()`, the W44-120 supplement
/// short-circuits — same precedence as the pre-W44-193 hand-written
/// `apply_env_var_fallbacks`).
fn parse_epf_sharpness_disable(s: &str) -> Option<EpfSharpnessSeed> {
    if s == "1" {
        Some(EpfSharpnessSeed::LegacyUniform4)
    } else {
        None
    }
}

/// `JXL_BUTTLOOP_INITIAL_QF_SCALE=<f32>` →
/// `Some(ButtloopQfSeedPolicy::AutoScale(v))` (custom scale at the
/// auto gate). Anything else returns `None`.
fn parse_buttloop_qf_scale(s: &str) -> Option<ButtloopQfSeedPolicy> {
    s.parse::<f32>().ok().map(ButtloopQfSeedPolicy::AutoScale)
}

/// `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=<f32>` →
/// `Some(AdaptiveQuantQfSeedPolicy::AutoScaleCustom { e5_e6: v, e7: v })`
/// (the historical env var was a single scalar; the per-effort split is
/// kept internal to the default). Anything else returns `None`.
fn parse_adaptive_quant_qf_scale(s: &str) -> Option<AdaptiveQuantQfSeedPolicy> {
    s.parse::<f32>()
        .ok()
        .map(|v| AdaptiveQuantQfSeedPolicy::AutoScaleCustom { e5_e6: v, e7: v })
}

// ──────────────────────────────────────────────────────────────────────
// `strategy_def!` invocation — generates `CustomEncoderImprovements`,
// `CustomResolvedImprovements`, `CustomEncoderStrategy`, four named
// constructors per strategy, `from_custom`, `Default` impls, and
// `apply_custom_env_var_fallbacks`. The matching W44-194 divergence-
// table consts are also emitted.
// ──────────────────────────────────────────────────────────────────────

jxl_encoder_macros::strategy_def! {
    name = Custom;
    default_strategy = Zenjxl;

    enums {}

    strategies {
        /// Strict libjxl-parity bundle. Mirrors pre-W44-193
        /// `ResolvedImprovements::libjxl()` byte-for-byte. Includes
        /// Section A effort-gate flips AND the Section D KNOWN-BUG
        /// `BlockCtxMap` 15-cluster re-enable — see
        /// [`crate::api::EncoderStrategy::Libjxl`] doc-comment.
        Libjxl {
            screenshot_entropy_mul = ScreenshotEntropyMulPolicy::Disabled,
            high_d_photo_entropy_mul = HighDPhotoEntropyMulPolicy::Disabled,
            dct64_search_policy = Dct64SearchPolicy::ForceAllow,
            dct32_search_policy = Dct32SearchPolicy::FollowDct64Suppression,
            smooth_photo_dct64_admission = SmoothPhotoDct64Policy::ForceSkip,
            buttloop_qf_seed = ButtloopQfSeedPolicy::Off,
            adaptive_quant_qf_seed = AdaptiveQuantQfSeedPolicy::Off,
            buttloop_epf_sharpness_seed = EpfSharpnessSeed::LegacyUniform4,
            // Perf dispatches: leave at Default (Auto). Libjxl is
            // byte-identical on `Auto` for libjxl-shaped inputs; the
            // dispatch enums are perf-only supersets of libjxl behaviour.
            epf_dispatch = EpfDispatch::AlwaysSelect,
            pixel_loss_dispatch = PixelLossDispatch::AlwaysOn,
            single_pass_entropy_dispatch = SinglePassEntropyDispatch::AlwaysTwoPass,
            patches_dispatch = PatchesDispatch::Auto,
            // Section A: flip to libjxl gates
            cfl_two_pass_min_effort = EffortGate::Libjxl,
            try_dct64_min_effort = EffortGate::Libjxl,
            epf_dynamic_sharpness_min_effort = EffortGate::Libjxl,
            // Section D KNOWN-BUG: deliberately re-enable to match libjxl
            block_ctx_map_15_cluster = true,
            // Smart-Zenjxl gates: strict parity — every per-image
            // discriminator disabled. Callers can still opt in via
            // `LossyConfig::with_content_class(Some(class))` etc.
            content_class_auto_classify = false,
            photo_epf_seed_admit = false,
            photo_variant_z_admit = false,
            find_best_32_per_m3_lift = false,
            adaptive_buttloop_iters = false,
            adaptive_buttloop_iters_narrow = false,
            terminal_class_exclude = false,
            // Section C CfL Newton: flip to libjxl bit-exact params.
            // Safe here because every other divergence is also flipped
            // (no W44-29..W44-172 calibration to throw off).
            cfl_newton_libjxl_parity = true,
        },

        /// LeanFaster — drops the heavy per-image content gates
        /// (W22-1 screenshot lift, W44-65/68/123 DCT64/DCT32 admission,
        /// W44-105/107/108 buttloop chain, W44-109 adaptive-quant chain,
        /// W44-117/118/120 EPF chain, W44-34/35 smooth-photo DCT64,
        /// W44-164..176 Smart-Zenjxl chain). Keeps the cheap
        /// photo-class entropy-mul lowering and our effort-gate
        /// values. The "..Default::default()" tail in the pre-W44-193
        /// hand-written ctor is expanded inline below.
        LeanFaster {
            screenshot_entropy_mul = ScreenshotEntropyMulPolicy::Disabled,
            high_d_photo_entropy_mul = HighDPhotoEntropyMulPolicy::Auto,
            dct64_search_policy = Dct64SearchPolicy::ForceAllow,
            dct32_search_policy = Dct32SearchPolicy::FollowDct64Suppression,
            smooth_photo_dct64_admission = SmoothPhotoDct64Policy::ForceSkip,
            buttloop_qf_seed = ButtloopQfSeedPolicy::Off,
            adaptive_quant_qf_seed = AdaptiveQuantQfSeedPolicy::Off,
            buttloop_epf_sharpness_seed = EpfSharpnessSeed::LegacyUniform4,
            // Perf dispatches: Default (matches the pre-W44-193
            // `..Default::default()` tail).
            epf_dispatch = EpfDispatch::AlwaysSelect,
            pixel_loss_dispatch = PixelLossDispatch::AlwaysOn,
            single_pass_entropy_dispatch = SinglePassEntropyDispatch::AlwaysTwoPass,
            patches_dispatch = PatchesDispatch::Auto,
            // Section A effort gates: Default (Ours). LeanFaster does
            // not flip the Section A gates — it only drops per-image
            // content gates.
            cfl_two_pass_min_effort = EffortGate::Ours,
            try_dct64_min_effort = EffortGate::Ours,
            epf_dynamic_sharpness_min_effort = EffortGate::Ours,
            // Section D: NOT re-enabled on LeanFaster (only on Libjxl).
            block_ctx_map_15_cluster = false,
            // Smart-Zenjxl: drop every per-image gate.
            content_class_auto_classify = false,
            photo_epf_seed_admit = false,
            photo_variant_z_admit = false,
            find_best_32_per_m3_lift = false,
            adaptive_buttloop_iters = false,
            adaptive_buttloop_iters_narrow = false,
            terminal_class_exclude = false,
            // W44-184: NOT a per-image gate; LeanFaster keeps Zenjxl's
            // cost-model calibration which is incompatible with the
            // libjxl-parity Newton (W44-183 measured 25/27 regressions).
            cfl_newton_libjxl_parity = false,
        },

        /// Zenjxl — production-shipping bundle. Every field matches
        /// the pre-W44-193 `EncoderImprovementsCustom::default()` /
        /// `ResolvedImprovements::default()` byte-for-byte. The macro
        /// re-emits these as `Self { ... }` constructors and the
        /// generated `Default::default()` delegates to `zenjxl()`.
        Zenjxl {
            // W44-130 Chunk D: Disabled (not the field default `Auto`)
            // — preserves pre-Chunk-D opt-in behaviour. `Auto` here
            // would fire the mask1x1 discriminator on every
            // screenshot-like input.
            screenshot_entropy_mul = ScreenshotEntropyMulPolicy::Disabled,
            high_d_photo_entropy_mul = HighDPhotoEntropyMulPolicy::Auto,
            dct64_search_policy = Dct64SearchPolicy::Auto,
            dct32_search_policy = Dct32SearchPolicy::FollowDct64Suppression,
            smooth_photo_dct64_admission = SmoothPhotoDct64Policy::Auto,
            buttloop_qf_seed = ButtloopQfSeedPolicy::AutoScale4,
            adaptive_quant_qf_seed = AdaptiveQuantQfSeedPolicy::AutoScalePerEffort,
            // `EpfSharpnessSeed::default()` = `AutoW44_117 { min_distance: 1.0 }`.
            buttloop_epf_sharpness_seed = EpfSharpnessSeed::AutoW44_117 { min_distance: 1.0 },
            epf_dispatch = EpfDispatch::AlwaysSelect,
            pixel_loss_dispatch = PixelLossDispatch::AlwaysOn,
            single_pass_entropy_dispatch = SinglePassEntropyDispatch::AlwaysTwoPass,
            patches_dispatch = PatchesDispatch::Auto,
            cfl_two_pass_min_effort = EffortGate::Ours,
            try_dct64_min_effort = EffortGate::Ours,
            epf_dynamic_sharpness_min_effort = EffortGate::Ours,
            block_ctx_map_15_cluster = false,
            content_class_auto_classify = true,
            photo_epf_seed_admit = true,
            photo_variant_z_admit = true,
            find_best_32_per_m3_lift = true,
            adaptive_buttloop_iters = true,
            adaptive_buttloop_iters_narrow = true,
            terminal_class_exclude = true,
            cfl_newton_libjxl_parity = false,
        },

        /// Aggressive — currently equivalent to `Zenjxl` after
        /// W44-124's auto-discriminator obsoleted the previous
        /// "Aggressive flips W44-123 globally" behaviour. Forward-
        /// compatible slot for the next opt-in chunk with a too-narrow
        /// auto-discriminator for the Zenjxl bundle.
        Aggressive {
            screenshot_entropy_mul = ScreenshotEntropyMulPolicy::Disabled,
            high_d_photo_entropy_mul = HighDPhotoEntropyMulPolicy::Auto,
            dct64_search_policy = Dct64SearchPolicy::Auto,
            dct32_search_policy = Dct32SearchPolicy::FollowDct64Suppression,
            smooth_photo_dct64_admission = SmoothPhotoDct64Policy::Auto,
            buttloop_qf_seed = ButtloopQfSeedPolicy::AutoScale4,
            adaptive_quant_qf_seed = AdaptiveQuantQfSeedPolicy::AutoScalePerEffort,
            buttloop_epf_sharpness_seed = EpfSharpnessSeed::AutoW44_117 { min_distance: 1.0 },
            epf_dispatch = EpfDispatch::AlwaysSelect,
            pixel_loss_dispatch = PixelLossDispatch::AlwaysOn,
            single_pass_entropy_dispatch = SinglePassEntropyDispatch::AlwaysTwoPass,
            patches_dispatch = PatchesDispatch::Auto,
            cfl_two_pass_min_effort = EffortGate::Ours,
            try_dct64_min_effort = EffortGate::Ours,
            epf_dynamic_sharpness_min_effort = EffortGate::Ours,
            block_ctx_map_15_cluster = false,
            content_class_auto_classify = true,
            photo_epf_seed_admit = true,
            photo_variant_z_admit = true,
            find_best_32_per_m3_lift = true,
            adaptive_buttloop_iters = true,
            adaptive_buttloop_iters_narrow = true,
            terminal_class_exclude = true,
            cfl_newton_libjxl_parity = false,
        },
    }

    gates {
        // ── Section B (content-aware gates) ─────────────────────────
        /// W22-1 screenshot entropy_mul lift table. Section B.
        screenshot_entropy_mul: ScreenshotEntropyMulPolicy {
            divergence_section = "B",
            divergence_row_ref = "W22-1 screenshot entropy_mul lift",
        },

        /// W44-29 + nested sub-gates (W44-91 / W44-96 / W44-98 /
        /// W44-99 / W44-100). Section B.
        high_d_photo_entropy_mul: HighDPhotoEntropyMulPolicy {
            divergence_section = "B",
            divergence_row_ref = "W44-29 high-d photo entropy_mul lowering",
        },

        /// W44-65 / W44-68 DCT64-class suppression on screenshot content.
        /// Section B.
        dct64_search_policy: Dct64SearchPolicy {
            divergence_section = "B",
            divergence_row_ref = "W44-65/68 DCT64 screenshot suppression",
        },

        /// W44-123 / W44-124 DCT32-class search retention. Section B.
        dct32_search_policy: Dct32SearchPolicy {
            divergence_section = "B",
            divergence_row_ref = "W44-123/124 DCT32 retention on m3>=60 / edge_density<0.05",
        },

        /// W44-34 / W44-35 smooth-photo DCT64 admission inside the
        /// `pixels < 500_000 AND distance < 2.0` smart-dispatch gate.
        /// Section B.
        smooth_photo_dct64_admission: SmoothPhotoDct64Policy {
            divergence_section = "B",
            divergence_row_ref = "W44-34/35 smooth-photo DCT64 smart-dispatch admit",
        },

        /// W44-105 / W44-107 / W44-108 buttloop qf seed scale. Promoted
        /// from env var `JXL_BUTTLOOP_INITIAL_QF_SCALE`. Section B.
        buttloop_qf_seed: ButtloopQfSeedPolicy {
            env_hook = "JXL_BUTTLOOP_INITIAL_QF_SCALE" => parse_buttloop_qf_scale,
            divergence_section = "B",
            divergence_row_ref = "W44-105/107/108 buttloop qf seed scale (effort >= 8)",
        },

        /// W44-109 adaptive_quant qf pre-scale at effort ∈ [5, 7].
        /// Promoted from env var `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE`.
        /// Section B.
        adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy {
            env_hook = "JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE" => parse_adaptive_quant_qf_scale,
            divergence_section = "B",
            divergence_row_ref = "W44-109 adaptive_quant qf seed (effort 5..=7)",
        },

        /// W44-117 / W44-118 / W44-120 EPF sharpness seed for buttloop.
        /// Promoted from env var `JXL_W44_117_DISABLE` (→ `LegacyUniform4`);
        /// `JXL_W44_120_EPF_SEED_MIN_DISTANCE` is fed by
        /// [`apply_w44_120_min_distance_env_fallback`] in this module
        /// because the W44-192 macro supports only one env hook per
        /// gate. Section B.
        buttloop_epf_sharpness_seed: EpfSharpnessSeed {
            env_hook = "JXL_W44_117_DISABLE" => parse_epf_sharpness_disable,
            divergence_section = "B",
            divergence_row_ref = "W44-117/118/120 EPF seed (buttloop sharpness)",
        },

        // ── Perf dispatches (absorbed into Custom per W44-130) ──────
        /// W37-2 EPF per-block sharpness search dispatch. Section E.
        epf_dispatch: EpfDispatch {
            divergence_section = "E",
            divergence_row_ref = "W37-2 EPF dispatch (perf-only)",
        },

        /// W38-2 / W44-90 pixel-domain loss dispatch. Section E.
        pixel_loss_dispatch: PixelLossDispatch {
            divergence_section = "E",
            divergence_row_ref = "W38-2/W44-90 pixel_loss dispatch (perf-only)",
        },

        /// W44-87 single-pass entropy dispatch at e=5 on smooth photos.
        /// Section E.
        single_pass_entropy_dispatch: SinglePassEntropyDispatch {
            divergence_section = "E",
            divergence_row_ref = "W44-87 single-pass entropy dispatch (e=5 smooth photos)",
        },

        /// W37-1 / W41-2 patches scan dispatch. Section E.
        patches_dispatch: PatchesDispatch {
            divergence_section = "E",
            divergence_row_ref = "W37-1/W41-2 patches dispatch (perf-only)",
        },

        // ── Section A effort-gate divergences (Libjxl-only flips) ────
        /// `cfl_two_pass` effort gate. Section A.
        cfl_two_pass_min_effort: EffortGate {
            divergence_section = "A",
            divergence_row_ref = "cfl_two_pass effort gate (ours >=7, libjxl >=5)",
        },

        /// `try_dct64` effort gate. Section A.
        try_dct64_min_effort: EffortGate {
            divergence_section = "A",
            divergence_row_ref = "try_dct64 effort gate (ours >=7, libjxl none)",
        },

        /// `epf_dynamic_sharpness` effort gate. Section A.
        epf_dynamic_sharpness_min_effort: EffortGate {
            divergence_section = "A",
            divergence_row_ref = "epf_dynamic_sharpness effort gate (ours >=6, libjxl none)",
        },

        // ── Section D KNOWN-BUG re-enables (Libjxl-only) ─────────────
        /// 15-cluster default for `BlockCtxMap`. Issue #59 KNOWN-BUG.
        /// Section D + Section F.
        block_ctx_map_15_cluster: bool {
            divergence_section = "D",
            divergence_row_ref = "BlockCtxMap 15-cluster default (issue #59 KNOWN-BUG)",
        },

        // ── Smart-Zenjxl content-class dispatch ──────────────────────
        /// W44-164: auto-classify `ImageContentClass` via zenanalyze
        /// proxies on 8-bit sRGB layouts at the encode entry.
        /// Section B.
        content_class_auto_classify: bool {
            divergence_section = "B",
            divergence_row_ref = "W44-164 auto_classify_content_class_from_layout",
        },

        /// W44-165: admit high-mask photos (mask_p25 >= 85 AND d >= 4)
        /// to the W44-117 EPF sharpness seed in addition to the
        /// `is_screenshot` gate. Section B.
        photo_epf_seed_admit: bool {
            divergence_section = "B",
            divergence_row_ref = "W44-165 photo EPF seed admit (mask_p25 >= 85)",
        },

        /// W44-166: admit high-mask photos to the W44-96 variant Z
        /// dispatch via `mask_p25 >= 85`. Section B.
        photo_variant_z_admit: bool {
            divergence_section = "B",
            divergence_row_ref = "W44-166 photo variant Z admit (mask_p25 >= 85)",
        },

        /// W44-167: per-m3 sub-discriminator `dct16x32` lift on INNER
        /// variant Z tables. Section B.
        find_best_32_per_m3_lift: bool {
            divergence_section = "B",
            divergence_row_ref = "W44-167 find_best_32 per-m3 dct16x32 lift",
        },

        /// W44-168: adaptive `butteraugli_iters` per content
        /// (smooth-skip + textured-extend modes). Section B.
        adaptive_buttloop_iters: bool {
            divergence_section = "B",
            divergence_row_ref = "W44-168 adaptive butteraugli_iters (JXL_W44_168_MODE)",
        },

        /// W44-169: distance-narrowed SmoothSkip iter-reduction at
        /// d ∈ [4.0, 5.0] (production SHIPPED). Section B.
        adaptive_buttloop_iters_narrow: bool {
            divergence_section = "B",
            divergence_row_ref = "W44-169 distance-narrowed buttloop iter reduction",
        },

        /// W44-176: exclude terminal-class screenshots from W44-108
        /// sub-gate at e ∈ {5, 6, 7}. Section B.
        terminal_class_exclude: bool {
            divergence_section = "B",
            divergence_row_ref = "W44-176 terminal-class exclude from W44-108",
        },

        // ── Section C CfL Newton parity ──────────────────────────────
        /// W44-184 (Pass-2) + W44-195 (Pass-1): bit-exact libjxl CfL Newton
        /// parameters (eps=100, max_iters=20, start x=0, no LS fallback)
        /// applied at BOTH Pass-1 and Pass-2 CfL dispatch sites.
        ///
        /// - **Pass-1** (`encoder::compute_cfl_map` call site): when this
        ///   field is `true`, Pass-1 dispatches to Newton at e>=7 (matching
        ///   libjxl `enc_heuristics.cc:1170-1174` with `fast=false`). When
        ///   `false` (Zenjxl default), Pass-1 stays on LS to preserve the
        ///   W44-29..W44-172 downstream cost-model calibration. Wired by
        ///   W44-195 (closes W44-189 D1 audit finding).
        /// - **Pass-2** (`chroma_from_luma::refine_cfl_map`): when this
        ///   field is `true`, the SIMD Newton kernel
        ///   ([`jxl_simd::cfl_find_best_multiplier_newton`]) overrides
        ///   eps/max_iters/start-x/fallback with libjxl-bit-exact values
        ///   (matching libjxl `enc_chroma_from_luma.cc:152-167`). Wired
        ///   by W44-184 (`b8517c09`).
        ///
        /// Promoted from env var `JXL_W44_184_FORCE_LIBJXL_NEWTON`. Section C.
        cfl_newton_libjxl_parity: bool {
            env_hook = "JXL_W44_184_FORCE_LIBJXL_NEWTON" => parse_bool_one,
            divergence_section = "C",
            divergence_row_ref = "W44-184/W44-195 CfL Newton libjxl parity (Pass-1 dispatch + Pass-2 internals, eps=100, max_iters=20)",
        },
    }
}

// ──────────────────────────────────────────────────────────────────────
// Macro-limitation supplement: dual env var on `buttloop_epf_sharpness_seed`.
// ──────────────────────────────────────────────────────────────────────

/// Hand-written supplement for `JXL_W44_120_EPF_SEED_MIN_DISTANCE`
/// (the W44-192 macro env-hook slot was taken by `JXL_W44_117_DISABLE`;
/// see [`parse_epf_sharpness_disable`] for the precedence rationale).
///
/// Applied AFTER [`apply_custom_env_var_fallbacks`] in
/// [`resolve_with_overrides`]. Fires only when the resolved field is
/// still at `EpfSharpnessSeed::default()`. Because the W44-117 disable
/// hook runs first and sets the field to `LegacyUniform4` (non-default),
/// the W44-120 min-distance supplement is auto-short-circuited when
/// both env vars are set — exact precedence parity with the pre-W44-193
/// hand-written `apply_env_var_fallbacks`.
///
/// Mirrors the existing `#[cfg(feature = "std")]` guard pattern from
/// the macro's emitted env-fallback fn (no-op when env vars are
/// unreadable in `no_std`).
#[cfg_attr(not(feature = "std"), allow(unused_variables))]
pub(crate) fn apply_w44_120_min_distance_env_fallback(r: &mut CustomResolvedImprovements) {
    #[cfg(feature = "std")]
    {
        if r.buttloop_epf_sharpness_seed == EpfSharpnessSeed::default()
            && let Ok(s) = std::env::var("JXL_W44_120_EPF_SEED_MIN_DISTANCE")
            && let Ok(d) = s.parse::<f32>()
        {
            r.buttloop_epf_sharpness_seed = EpfSharpnessSeed::AutoW44_117 { min_distance: d };
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Type aliases bridging the macro-generated names to the public-API
// names that `api.rs` and its consumers expect.
//
// Pre-W44-193, `api.rs` defined `EncoderImprovementsCustom` and
// `ResolvedImprovements` as concrete structs. Post-W44-193, the macro
// emits `CustomEncoderImprovements` and `CustomResolvedImprovements`
// (the `name = Custom` prefix is mandatory in the macro syntax).
// Type aliases preserve every existing call site byte-identically
// (struct literal syntax, `..Default::default()`, all builder traits).
// ──────────────────────────────────────────────────────────────────────

/// Fine-grained per-divergence picks. Alias for the macro-generated
/// [`CustomEncoderImprovements`] (see this module's `strategy_def!`
/// invocation). See [`crate::api::EncoderImprovementsCustom`] for the
/// public re-export and full field-by-field docs.
pub(crate) type EncoderImprovementsCustom = CustomEncoderImprovements;

/// Fully-resolved per-divergence flags consumed by the internal
/// encoder. Alias for the macro-generated
/// [`CustomResolvedImprovements`]. `pub(crate)` — not part of the
/// public API.
pub(crate) type ResolvedImprovements = CustomResolvedImprovements;

// ──────────────────────────────────────────────────────────────────────
// Public resolve helpers consumed by `api::EncoderStrategy::resolve`.
//
// The macro-generated `CustomEncoderStrategy` enum is unused by
// production (we keep `api::EncoderStrategy` because it carries the
// `Custom(Box<...>)` variant + the `resolve(&self, overrides:
// &StrategyOverrides)` signature the macro doesn't model). Production
// resolves by calling these constructors directly + the
// `StrategyOverrides::apply_to` + the env-fallback layer
// (macro-generated + W44-120 supplement). See
// [`crate::api::EncoderStrategy::resolve`].
// ──────────────────────────────────────────────────────────────────────

/// Apply both env-fallback layers — the macro-generated one for the
/// single-env gates, then the W44-120 supplement for the dual-env
/// `buttloop_epf_sharpness_seed` gate. Mirrors the pre-W44-193
/// `apply_env_var_fallbacks` ordering.
pub(crate) fn apply_env_var_fallbacks(r: &mut ResolvedImprovements) {
    apply_custom_env_var_fallbacks(r);
    apply_w44_120_min_distance_env_fallback(r);
}

// ──────────────────────────────────────────────────────────────────────
// W44-194: divergence-metadata harvesting hook for the CI drift test.
//
// The `strategy_def!` macro emits one `pub const __CUSTOM_DIVERGENCE_<GATE>:
// &str` constant per gate that carries a `divergence_section` clause. The
// string format is `"section=X ; row_ref=Y"`. The
// [`crate::__internals::gate_registry::all_divergence_metadata`] re-export
// (cfg-gated on `__internals`) gives the W44-194 inline drift test a
// canonical list. We hand-curate the slice here rather than syn-walking
// the macro at compile time — the maintenance cost is "add one line when
// you add a gate" which is the SAME cost as adding the gate row to
// `docs/LIBJXL_DIVERGENCES.md` (which the drift test catches).
//
// Per-row schema: `(gate_name, section_letter, row_ref_substring,
// raw_const_value)`. The drift test only consults `(gate_name, section,
// row_ref)`; `raw_const_value` is exposed so future tooling (e.g. a
// build-script that auto-generates docs) can read the format-string
// authoritatively without re-parsing.
pub(crate) struct DivergenceEntry {
    pub gate_name: &'static str,
    pub section: &'static str,
    pub row_ref: &'static str,
    pub raw: &'static str,
}

/// All gates that carry divergence metadata (= every gate in the
/// `strategy_def!` invocation above with a `divergence_section` clause).
/// Order matches the macro declaration order.
pub(crate) const ALL_DIVERGENCE_ENTRIES: &[DivergenceEntry] = &[
    // Section B — content-aware gates
    DivergenceEntry {
        gate_name: "screenshot_entropy_mul",
        section: "B",
        row_ref: "W22-1 screenshot entropy_mul lift",
        raw: __CUSTOM_DIVERGENCE_SCREENSHOT_ENTROPY_MUL,
    },
    DivergenceEntry {
        gate_name: "high_d_photo_entropy_mul",
        section: "B",
        row_ref: "W44-29 high-d photo entropy_mul lowering",
        raw: __CUSTOM_DIVERGENCE_HIGH_D_PHOTO_ENTROPY_MUL,
    },
    DivergenceEntry {
        gate_name: "dct64_search_policy",
        section: "B",
        row_ref: "W44-65/68 DCT64 screenshot suppression",
        raw: __CUSTOM_DIVERGENCE_DCT64_SEARCH_POLICY,
    },
    DivergenceEntry {
        gate_name: "dct32_search_policy",
        section: "B",
        row_ref: "W44-123/124 DCT32 retention on m3>=60 / edge_density<0.05",
        raw: __CUSTOM_DIVERGENCE_DCT32_SEARCH_POLICY,
    },
    DivergenceEntry {
        gate_name: "smooth_photo_dct64_admission",
        section: "B",
        row_ref: "W44-34/35 smooth-photo DCT64 smart-dispatch admit",
        raw: __CUSTOM_DIVERGENCE_SMOOTH_PHOTO_DCT64_ADMISSION,
    },
    DivergenceEntry {
        gate_name: "buttloop_qf_seed",
        section: "B",
        row_ref: "W44-105/107/108 buttloop qf seed scale (effort >= 8)",
        raw: __CUSTOM_DIVERGENCE_BUTTLOOP_QF_SEED,
    },
    DivergenceEntry {
        gate_name: "adaptive_quant_qf_seed",
        section: "B",
        row_ref: "W44-109 adaptive_quant qf seed (effort 5..=7)",
        raw: __CUSTOM_DIVERGENCE_ADAPTIVE_QUANT_QF_SEED,
    },
    DivergenceEntry {
        gate_name: "buttloop_epf_sharpness_seed",
        section: "B",
        row_ref: "W44-117/118/120 EPF seed (buttloop sharpness)",
        raw: __CUSTOM_DIVERGENCE_BUTTLOOP_EPF_SHARPNESS_SEED,
    },
    // Section E — perf dispatches
    DivergenceEntry {
        gate_name: "epf_dispatch",
        section: "E",
        row_ref: "W37-2 EPF dispatch (perf-only)",
        raw: __CUSTOM_DIVERGENCE_EPF_DISPATCH,
    },
    DivergenceEntry {
        gate_name: "pixel_loss_dispatch",
        section: "E",
        row_ref: "W38-2/W44-90 pixel_loss dispatch (perf-only)",
        raw: __CUSTOM_DIVERGENCE_PIXEL_LOSS_DISPATCH,
    },
    DivergenceEntry {
        gate_name: "single_pass_entropy_dispatch",
        section: "E",
        row_ref: "W44-87 single-pass entropy dispatch (e=5 smooth photos)",
        raw: __CUSTOM_DIVERGENCE_SINGLE_PASS_ENTROPY_DISPATCH,
    },
    DivergenceEntry {
        gate_name: "patches_dispatch",
        section: "E",
        row_ref: "W37-1/W41-2 patches dispatch (perf-only)",
        raw: __CUSTOM_DIVERGENCE_PATCHES_DISPATCH,
    },
    // Section A — effort gates (Libjxl-only flips)
    DivergenceEntry {
        gate_name: "cfl_two_pass_min_effort",
        section: "A",
        row_ref: "cfl_two_pass effort gate (ours >=7, libjxl >=5)",
        raw: __CUSTOM_DIVERGENCE_CFL_TWO_PASS_MIN_EFFORT,
    },
    DivergenceEntry {
        gate_name: "try_dct64_min_effort",
        section: "A",
        row_ref: "try_dct64 effort gate (ours >=7, libjxl none)",
        raw: __CUSTOM_DIVERGENCE_TRY_DCT64_MIN_EFFORT,
    },
    DivergenceEntry {
        gate_name: "epf_dynamic_sharpness_min_effort",
        section: "A",
        row_ref: "epf_dynamic_sharpness effort gate (ours >=6, libjxl none)",
        raw: __CUSTOM_DIVERGENCE_EPF_DYNAMIC_SHARPNESS_MIN_EFFORT,
    },
    // Section D — KNOWN-BUG re-enables (Libjxl-only)
    DivergenceEntry {
        gate_name: "block_ctx_map_15_cluster",
        section: "D",
        row_ref: "BlockCtxMap 15-cluster default (issue #59 KNOWN-BUG)",
        raw: __CUSTOM_DIVERGENCE_BLOCK_CTX_MAP_15_CLUSTER,
    },
    // Section B — Smart-Zenjxl content-class dispatch
    DivergenceEntry {
        gate_name: "content_class_auto_classify",
        section: "B",
        row_ref: "W44-164 auto_classify_content_class_from_layout",
        raw: __CUSTOM_DIVERGENCE_CONTENT_CLASS_AUTO_CLASSIFY,
    },
    DivergenceEntry {
        gate_name: "photo_epf_seed_admit",
        section: "B",
        row_ref: "W44-165 photo EPF seed admit (mask_p25 >= 85)",
        raw: __CUSTOM_DIVERGENCE_PHOTO_EPF_SEED_ADMIT,
    },
    DivergenceEntry {
        gate_name: "photo_variant_z_admit",
        section: "B",
        row_ref: "W44-166 photo variant Z admit (mask_p25 >= 85)",
        raw: __CUSTOM_DIVERGENCE_PHOTO_VARIANT_Z_ADMIT,
    },
    DivergenceEntry {
        gate_name: "find_best_32_per_m3_lift",
        section: "B",
        row_ref: "W44-167 find_best_32 per-m3 dct16x32 lift",
        raw: __CUSTOM_DIVERGENCE_FIND_BEST_32_PER_M3_LIFT,
    },
    DivergenceEntry {
        gate_name: "adaptive_buttloop_iters",
        section: "B",
        row_ref: "W44-168 adaptive butteraugli_iters (JXL_W44_168_MODE)",
        raw: __CUSTOM_DIVERGENCE_ADAPTIVE_BUTTLOOP_ITERS,
    },
    DivergenceEntry {
        gate_name: "adaptive_buttloop_iters_narrow",
        section: "B",
        row_ref: "W44-169 distance-narrowed buttloop iter reduction",
        raw: __CUSTOM_DIVERGENCE_ADAPTIVE_BUTTLOOP_ITERS_NARROW,
    },
    DivergenceEntry {
        gate_name: "terminal_class_exclude",
        section: "B",
        row_ref: "W44-176 terminal-class exclude from W44-108",
        raw: __CUSTOM_DIVERGENCE_TERMINAL_CLASS_EXCLUDE,
    },
    // Section C — CfL Newton parity
    DivergenceEntry {
        gate_name: "cfl_newton_libjxl_parity",
        section: "C",
        row_ref: "W44-184/W44-195 CfL Newton libjxl parity (Pass-1 dispatch + Pass-2 internals, eps=100, max_iters=20)",
        raw: __CUSTOM_DIVERGENCE_CFL_NEWTON_LIBJXL_PARITY,
    },
];

/// Internal helper: count of declared divergence entries. Used by the
/// W44-194 inline test as a count-vs-macro-gate-count cross-check (catches
/// the case where someone adds a gate to the macro without adding a row
/// here).
#[allow(dead_code)]
pub(crate) const ALL_DIVERGENCE_ENTRIES_LEN: usize = ALL_DIVERGENCE_ENTRIES.len();

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: `CustomEncoderImprovements::default()` matches the
    /// pre-W44-193 hand-written `EncoderImprovementsCustom::default()`
    /// field-by-field. The macro derives Default from the
    /// `default_strategy = Zenjxl` declaration; this test pins the
    /// resulting values against the pre-W44-193 Zenjxl ctor.
    #[test]
    fn default_matches_zenjxl_byte_for_byte() {
        let d = CustomEncoderImprovements::default();
        // Section B
        assert_eq!(
            d.screenshot_entropy_mul,
            ScreenshotEntropyMulPolicy::Disabled
        );
        assert_eq!(d.high_d_photo_entropy_mul, HighDPhotoEntropyMulPolicy::Auto);
        assert_eq!(d.dct64_search_policy, Dct64SearchPolicy::Auto);
        assert_eq!(
            d.dct32_search_policy,
            Dct32SearchPolicy::FollowDct64Suppression
        );
        assert_eq!(d.smooth_photo_dct64_admission, SmoothPhotoDct64Policy::Auto);
        assert_eq!(d.buttloop_qf_seed, ButtloopQfSeedPolicy::AutoScale4);
        assert_eq!(
            d.adaptive_quant_qf_seed,
            AdaptiveQuantQfSeedPolicy::AutoScalePerEffort
        );
        assert_eq!(
            d.buttloop_epf_sharpness_seed,
            EpfSharpnessSeed::AutoW44_117 { min_distance: 1.0 }
        );
        // Perf dispatches (each type's `Default` value matches the
        // pre-W44-193 hand-written `Default` impl on `ResolvedImprovements`
        // which delegated to the type-level `Default::default()` of each
        // enum).
        assert_eq!(d.epf_dispatch, EpfDispatch::AlwaysSelect);
        assert_eq!(d.pixel_loss_dispatch, PixelLossDispatch::AlwaysOn);
        assert_eq!(
            d.single_pass_entropy_dispatch,
            SinglePassEntropyDispatch::AlwaysTwoPass
        );
        assert_eq!(d.patches_dispatch, PatchesDispatch::Auto);
        // Section A
        assert_eq!(d.cfl_two_pass_min_effort, EffortGate::Ours);
        assert_eq!(d.try_dct64_min_effort, EffortGate::Ours);
        assert_eq!(d.epf_dynamic_sharpness_min_effort, EffortGate::Ours);
        // Section D
        assert!(!d.block_ctx_map_15_cluster);
        // Smart-Zenjxl
        assert!(d.content_class_auto_classify);
        assert!(d.photo_epf_seed_admit);
        assert!(d.photo_variant_z_admit);
        assert!(d.find_best_32_per_m3_lift);
        assert!(d.adaptive_buttloop_iters);
        assert!(d.adaptive_buttloop_iters_narrow);
        assert!(d.terminal_class_exclude);
        // Section C
        assert!(!d.cfl_newton_libjxl_parity);
    }

    /// Sanity: `CustomResolvedImprovements::default()` matches
    /// `CustomResolvedImprovements::zenjxl()` (the macro's `Default`
    /// delegates to the `default_strategy` constructor).
    #[test]
    fn resolved_default_equals_zenjxl_ctor() {
        assert_eq!(
            CustomResolvedImprovements::default(),
            CustomResolvedImprovements::zenjxl()
        );
    }

    /// Libjxl constructor diverges from Zenjxl on every Section A / B
    /// / D / Smart-Zenjxl gate. Spot-check the divergence is wired.
    #[test]
    fn libjxl_diverges_from_zenjxl_on_key_gates() {
        let l = CustomResolvedImprovements::libjxl();
        let z = CustomResolvedImprovements::zenjxl();
        assert_ne!(l.high_d_photo_entropy_mul, z.high_d_photo_entropy_mul);
        assert_ne!(l.dct64_search_policy, z.dct64_search_policy);
        assert_ne!(l.buttloop_qf_seed, z.buttloop_qf_seed);
        assert_ne!(l.adaptive_quant_qf_seed, z.adaptive_quant_qf_seed);
        assert_ne!(l.buttloop_epf_sharpness_seed, z.buttloop_epf_sharpness_seed);
        assert_ne!(l.cfl_two_pass_min_effort, z.cfl_two_pass_min_effort);
        assert_ne!(l.try_dct64_min_effort, z.try_dct64_min_effort);
        assert_ne!(
            l.epf_dynamic_sharpness_min_effort,
            z.epf_dynamic_sharpness_min_effort
        );
        assert_ne!(l.block_ctx_map_15_cluster, z.block_ctx_map_15_cluster);
        assert_ne!(l.content_class_auto_classify, z.content_class_auto_classify);
        assert_ne!(l.cfl_newton_libjxl_parity, z.cfl_newton_libjxl_parity);
    }

    /// Aggressive == Zenjxl (forward-compat slot per W44-124).
    #[test]
    fn aggressive_equals_zenjxl() {
        assert_eq!(
            CustomResolvedImprovements::aggressive(),
            CustomResolvedImprovements::zenjxl()
        );
    }

    /// `from_custom` copies every field. The macro emits a
    /// field-by-field clone constructor.
    #[test]
    fn from_custom_field_for_field_copy() {
        let mut custom = CustomEncoderImprovements::default();
        custom.cfl_newton_libjxl_parity = true;
        custom.block_ctx_map_15_cluster = true;
        let r = CustomResolvedImprovements::from_custom(&custom);
        assert!(r.cfl_newton_libjxl_parity);
        assert!(r.block_ctx_map_15_cluster);
        // Zenjxl-default fields propagate
        assert!(r.content_class_auto_classify);
    }

    /// Divergence metadata consts are emitted and harvestable. The
    /// W44-194 build-script will walk these.
    #[test]
    fn divergence_metadata_consts_exposed() {
        assert!(__CUSTOM_DIVERGENCE_CFL_NEWTON_LIBJXL_PARITY.contains("section=C"));
        assert!(__CUSTOM_DIVERGENCE_CFL_NEWTON_LIBJXL_PARITY.contains("W44-184"));
        assert!(__CUSTOM_DIVERGENCE_BLOCK_CTX_MAP_15_CLUSTER.contains("section=D"));
        assert!(__CUSTOM_DIVERGENCE_CONTENT_CLASS_AUTO_CLASSIFY.contains("section=B"));
        assert!(__CUSTOM_DIVERGENCE_EPF_DISPATCH.contains("section=E"));
        assert!(__CUSTOM_DIVERGENCE_CFL_TWO_PASS_MIN_EFFORT.contains("section=A"));
    }
}
