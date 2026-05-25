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

/// **W44-AUDIT-5 Phase 2**: extended env parser that accepts BOTH `0`
/// and `1` so the new Mode C default-flip can be A/B-disabled via
/// `JXL_W44_AUDIT_5_FORCE_LS_WARM_START=0`. `1` forces ON, `0` forces
/// OFF, anything else (incl. unset) returns `None` (keeps the resolved
/// field at its post-strategy value).
///
/// Needed because the Mode C default flipped to `true` on
/// Zenjxl/Aggressive/LeanFaster — paired bench harnesses need a way to
/// disable it without rebuilding. The original `parse_bool_one` only
/// accepts `1` (force-on) which couldn't reach OFF when the default is
/// already ON.
fn parse_bool_zero_or_one(s: &str) -> Option<bool> {
    match s {
        "1" => Some(true),
        "0" => Some(false),
        _ => None,
    }
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
            // W44-AUDIT-6 Phase 1: strict libjxl parity → high-colour
            // class exclude is OFF on Libjxl strategy (the W44-109 lift
            // itself is already off via `adaptive_quant_qf_seed = Off`,
            // making this a redundancy guard).
            high_colour_class_exclude = false,
            // Section C CfL Newton: flip to libjxl bit-exact params.
            // Safe here because every other divergence is also flipped
            // (no W44-29..W44-172 calibration to throw off).
            cfl_newton_libjxl_parity = true,
            // W44-AUDIT-5 Phase 2 (Mode C): MUST stay `false` on Libjxl —
            // `cfl_newton_libjxl_parity = true` (above) takes priority
            // inside the SIMD kernel. Strict cjxl byte-parity is required
            // here; the mutual exclusion is enforced structurally.
            cfl_newton_libjxl_math_with_ls_warm_start = false,
            // W44-AUDIT-5 Phase 3: moot on Libjxl — `cfl_newton_libjxl_parity
            // = true` (above) forces `x=0` start for every tile, so this
            // per-image route is structurally redundant. Kept `false` to
            // preserve the byte-lock invariant (which asserts no
            // per-image dispatch on Libjxl).
            cfl_pass1_screenshot_x0_start = false,
            // W44-197 Candidate B: enable LS-only Pass-2 at e=5/6 to
            // match libjxl `fast=true` dispatch. Pairs with the
            // `cfl_two_pass_min_effort = EffortGate::Libjxl` widening
            // above so Libjxl strategy ships TRUE Pass-2 parity (LS at
            // e=5/6, Newton at e>=7) — closes W44-189 D12 audit finding.
            cfl_pass2_ls_at_low_effort = true,
            // W44-201: keep libjxl-parity coeff_orders behaviour — admit
            // ALL buckets to the custom-order cost-benefit gate (the
            // W44-201 Zenjxl-only tightening is OFF on Libjxl strategy).
            coeff_orders_disable_large_buckets = false,
            // W44-205: same — Libjxl preserves the libjxl `is_nondefault`
            // admission of ALL buckets, including medium 2 + 4.
            coeff_orders_disable_medium_buckets = false,
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
            // W44-AUDIT-6 Phase 1: LeanFaster drops every per-image
            // content gate (matches the W44-176 pattern); the W44-109
            // lift itself is already off via
            // `adaptive_quant_qf_seed = Off` so this is a redundancy
            // guard like terminal_class_exclude above.
            high_colour_class_exclude = false,
            // W44-184: NOT a per-image gate; LeanFaster keeps Zenjxl's
            // cost-model calibration which is incompatible with the
            // libjxl-parity Newton (W44-183 measured 25/27 regressions).
            cfl_newton_libjxl_parity = false,
            // W44-AUDIT-5 Phase 2 (Mode C): OPT-IN ONLY. LeanFaster
            // mirrors Zenjxl per the standing pattern. See Zenjxl
            // preset for the HONEST-STOP narrative.
            cfl_newton_libjxl_math_with_ls_warm_start = false,
            // W44-AUDIT-5 Phase 3: LeanFaster drops every per-image
            // content gate (matches the W44-176 / W44-AUDIT-6 pattern).
            cfl_pass1_screenshot_x0_start = false,
            // W44-197: same cost-model-calibration concern — LeanFaster
            // keeps the e=5/6 no-Pass-2 baseline.
            cfl_pass2_ls_at_low_effort = false,
            // W44-201: LeanFaster keeps the Zenjxl bucket-disable fix
            // (Variant C: skip buckets 3 + 6 in custom-order cost-benefit).
            // The fix is not per-image and not calibration-sensitive; it
            // is a strict improvement on the photo cluster with zero
            // measured regressions across 53 cells.
            coeff_orders_disable_large_buckets = true,
            // W44-205: same — LeanFaster keeps the extension to medium
            // buckets 2 + 4 (DCT16x16 + DCT16x8/8x16). Per-bucket
            // extension is strictly additive to the W44-201 fix and
            // not per-image / calibration-sensitive.
            coeff_orders_disable_medium_buckets = true,
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
            // W44-AUDIT-6 Phase 1 (2026-05-24): Zenjxl default ON —
            // excludes codec_wiki-class high-colour mixed-content
            // screenshots from the W44-109 lift (per AUDIT-4 measurement
            // of the +44% bytes wedge at e7 d=4). Composes with the
            // W44-176 terminal exclude via OR.
            high_colour_class_exclude = true,
            cfl_newton_libjxl_parity = false,
            // W44-AUDIT-5 Phase 2 (Mode C): OPT-IN ONLY on Zenjxl. The
            // Phase 2 3-mode bisect (`benchmarks/w44_audit_5_phase2_mode_bisect_2026-05-24.tsv`,
            // codec_wiki e7 d=4 + 1418519 + 1531677) measured Mode C =
            // byte-identical to Mode A (pre-Phase-2 Zenjxl LS-only) on
            // all 3 cells, because the encoder's i8 CfL multipliers
            // round identically when both Newton paths (libjxl-math vs
            // LS-only refinement) start from `ls_x` warm-start on these
            // inputs. The codec_wiki SSIM2 deficit (-5.51 Mode B vs cjxl,
            // -4.31 Mode A/C vs cjxl) is NOT closed by Mode C on
            // Zenjxl — the deficit lives on Mode B (Libjxl strategy)
            // because of the bit-exact libjxl Newton (x=0 start, no LS
            // fallback) which picks DIFFERENT multipliers on screenshots.
            // Libjxl strategy MUST keep bit-exact parity per the
            // byte-lock invariant, so Mode C ships as opt-in only.
            // Callers who want to A/B Mode C can flip via env
            // `JXL_W44_AUDIT_5_FORCE_LS_WARM_START=1` or by setting the
            // field on `EncoderImprovementsCustom`. Phase 3 needs a
            // different mechanism (e.g. lift the screenshot-class
            // discriminator before CfL Pass-1) to close the Libjxl-side
            // gap.
            cfl_newton_libjxl_math_with_ls_warm_start = false,
            // W44-AUDIT-5 Phase 3 (2026-05-24): Zenjxl default OFF
            // initially while the bisect + 36-cell regression validation
            // run. If the bisect passes (codec_wiki SSIM2 recovery
            // ≥ Mode B − 0.3, bytes within +5% Mode A, photos
            // byte-identical OR within ±1.0 SSIM2), the default flips
            // to `true` in a follow-on commit. Until then, callers can
            // opt in by setting the field directly OR via a future env
            // hook. See the Phase 3 ship commit for the measured numbers.
            cfl_pass1_screenshot_x0_start = false,
            // W44-197: Zenjxl preserves cost-model calibration; no LS-only
            // Pass-2 at e=5/6.
            cfl_pass2_ls_at_low_effort = false,
            // W44-201: skip buckets 3 (DCT32x32) + 6 (DCT32x16/16x32) in
            // the custom-order cost-benefit gate. The W44-82-RULED-OUT
            // cost model overestimates AC savings for these large buckets
            // when the per-position zero count distribution spans 3+
            // distinct quantized bins (Variant C, 53-cell paired bench).
            coeff_orders_disable_large_buckets = true,
            // W44-205: extend the W44-201 cost-benefit skip to medium
            // buckets 2 (DCT16x16) + 4 (DCT16x8/DCT8x16). Phase-1 probe
            // measured -0.97% on 27 cells with ZERO PROTECT regressions.
            coeff_orders_disable_medium_buckets = true,
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
            // W44-AUDIT-6 Phase 1: Aggressive mirrors Zenjxl per the
            // standing pattern (Aggressive is a forward-compatible slot
            // for the next opt-in chunk with a too-narrow auto-
            // discriminator for the Zenjxl bundle).
            high_colour_class_exclude = true,
            cfl_newton_libjxl_parity = false,
            // W44-AUDIT-5 Phase 2 (Mode C): OPT-IN ONLY. Aggressive
            // mirrors Zenjxl per the standing pattern; bench measured
            // byte-identical to Mode A on the codec_wiki + 2 photo cells
            // (see Zenjxl preset for the full HONEST-STOP narrative).
            cfl_newton_libjxl_math_with_ls_warm_start = false,
            // W44-AUDIT-5 Phase 3: Aggressive mirrors Zenjxl. Same
            // bisect/validation gate; the default-flip lands in the
            // same follow-on commit.
            cfl_pass1_screenshot_x0_start = false,
            // W44-197: Aggressive mirrors Zenjxl on Section C calibration
            // concerns.
            cfl_pass2_ls_at_low_effort = false,
            // W44-201: same as Zenjxl — skip buckets 3 + 6 in
            // custom-order cost-benefit. Strict improvement on photos.
            coeff_orders_disable_large_buckets = true,
            // W44-205: same as Zenjxl — extend to medium buckets 2 + 4.
            coeff_orders_disable_medium_buckets = true,
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

        /// W44-AUDIT-6 Phase 1 (2026-05-24): exclude high-colour
        /// mixed-content screenshots (`m3_colourfulness >= 80.0`) from
        /// the W44-109 adaptive-quant qf seed lift. Companion of
        /// `terminal_class_exclude`; the two compose via OR inside the
        /// W44-109 gate. Section B.
        high_colour_class_exclude: bool {
            divergence_section = "B",
            divergence_row_ref = "W44-AUDIT-6 high-colour-class exclude from W44-109",
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

        /// **W44-AUDIT-5 Phase 2 (Mode C)**: hybrid CfL Newton — libjxl
        /// math (eps=100, max_iters=20) starting from the LS warm-start
        /// (`ls_x`) with the existing LS fallback on non-convergence.
        /// Engages Pass-1 AND Pass-2 Newton when set, same dispatch
        /// shape as `cfl_newton_libjxl_parity`.
        ///
        /// Mutually-exclusive with `cfl_newton_libjxl_parity` inside the
        /// SIMD kernel — `libjxl_parity = true` takes priority and
        /// overrides this flag. The two are intentionally NOT a single
        /// enum because their independent strategy-by-strategy defaults
        /// are easier to reason about as booleans (per W44-193 macro
        /// philosophy).
        ///
        /// Designed to close the W44-AUDIT-5 Phase 1 codec_wiki SSIM2
        /// deficit (-5.51 vs cjxl on `e7 d=4`) without sacrificing the
        /// W44-29..W44-172 photo cost-model wins. The LS warm-start
        /// preserves the calibrated baseline; the libjxl Newton math
        /// recovers the chroma-multiplier accuracy on high-detail
        /// screenshot content.
        ///
        /// **Strategy defaults**:
        /// - Libjxl: `false` (uses `libjxl_parity = true`, which takes
        ///   priority — strict bit-exact path required by the byte-lock
        ///   invariant).
        /// - Zenjxl / Aggressive / LeanFaster: `false` (opt-in only —
        ///   the Phase 2 3-mode bisect measured Mode C byte-identical to
        ///   Mode A on the codec_wiki SSIM2-wedge cell + 2 photos, so
        ///   the default-flip was reverted; see Zenjxl preset HONEST-STOP
        ///   comment for the full narrative).
        ///
        /// Promoted from env var `JXL_W44_AUDIT_5_FORCE_LS_WARM_START`
        /// for opt-in A/B debugging without rebuild (accepts `0` AND
        /// `1` per W44-AUDIT-5 Phase 2's extended parser
        /// `parse_bool_zero_or_one`). Section C.
        cfl_newton_libjxl_math_with_ls_warm_start: bool {
            env_hook = "JXL_W44_AUDIT_5_FORCE_LS_WARM_START" => parse_bool_zero_or_one,
            divergence_section = "C",
            divergence_row_ref = "W44-184/W44-195 Mode C — CfL Newton libjxl-math (eps=100, iters=20) + LS warm-start (W44-AUDIT-5 Phase 2)",
        },

        /// **W44-AUDIT-5 Phase 3**: per-image content-class CfL warm-start
        /// route. When `true` AND the per-image
        /// [`crate::vardct::encoder::ZenanalyzeProxies`] satisfy
        /// `m3_colourfulness >= W44_AUDIT_6_HIGH_COLOUR_M3_MIN` (= 80.0),
        /// the encoder routes CfL Pass-1 (and Pass-2 if it fires) through
        /// the libjxl-bit-exact `x=0` start path for that single image.
        /// Photos and non-sRGB-u8 layouts stay on the LS warm-start /
        /// LS-only path (the W44-29..W44-172 cost-model calibration is
        /// preserved).
        ///
        /// **Why this exists**: the W44-AUDIT-5 Phase 1 + Phase 2 chain
        /// established that the codec_wiki-class SSIM2 deficit (-5.51 vs
        /// cjxl on `e7 d=4`) is caused by the CfL warm-start choice
        /// (`x=0` start vs `ls_x` warm-start), not the Newton math. Mode
        /// C (libjxl-math + ls_x warm-start, Phase 2) was byte-identical
        /// to Mode A (LS-only, the Zenjxl baseline) on screenshots:
        /// both refinement paths land at the same `i8` multiplier when
        /// started from `ls_x`. The deficit lives on the START position.
        ///
        /// Phase 3 routes by content class: the W44-AUDIT-6 Phase 1
        /// discriminator (`m3 >= 80`) admits only mixed-content
        /// screenshots (codec_wiki etc.) to the `x=0` start path.
        ///
        /// **Strategy defaults**:
        /// - Libjxl: `false` — moot (`libjxl_parity = true` already
        ///   forces `x=0` for every tile).
        /// - LeanFaster: `false` — drops per-image content gates per
        ///   the standing pattern (W44-176 / W44-AUDIT-6).
        /// - Zenjxl / Aggressive: `true` if Phase 3 bisect + regression
        ///   validation pass; otherwise `false` (opt-in only).
        ///
        /// Promoted from env var `JXL_W44_AUDIT_5_P3_DISABLE` (negative
        /// hook — `=1` forces OFF, mirrors W44-176 / W44-AUDIT-6 style).
        /// Section C.
        cfl_pass1_screenshot_x0_start: bool {
            divergence_section = "C",
            divergence_row_ref = "W44-184/W44-195 Phase 3 — per-image M3>=80 CfL `x=0` start route (Pass-1 + Pass-2 dispatch, W44-AUDIT-5 Phase 3)",
        },

        /// **W44-197 Candidate B**: enable CfL Pass-2 with LS-only solver
        /// (libjxl `fast=true`) at effort ∈ {5, 6} on top of the existing
        /// `cfl_two_pass: effort >= 7` Newton path.
        ///
        /// libjxl `enc_heuristics.cc:1190-1194` runs Pass-2 at
        /// `speed_tier <= kHare` (effort >= 5) with
        /// `fast = (speed_tier >= kWombat)` (true at e=5/6 → LS, false at
        /// e>=7 → Newton). We currently gate Pass-2 entirely at effort >= 7
        /// (W44-102 RULED OUT widening with FULL Newton because the
        /// downstream cost model was calibrated against no-Pass-2 at e=5/6
        /// and Newton-widened Pass-2 introduced 2 SSIM2 regressions).
        ///
        /// W44-197 ships the DIFFERENT mechanism (LS-only at e=5/6) gated
        /// to `EncoderStrategy::Libjxl` only. The full Newton widening
        /// remains opt-in via `EffortGate::Libjxl` on
        /// `cfl_two_pass_min_effort`; the LS-only widening adds a SEPARATE
        /// axis. Both gates can fire simultaneously under `Libjxl`
        /// (cfl_two_pass=true at e>=5 AND cfl_pass2_ls_at_low_effort=true
        /// at e=5/6 → Newton at e>=7, but at e=5/6 we use LS regardless of
        /// Newton because the speed_tier dispatch in libjxl picks LS there).
        ///
        /// Default (Zenjxl/Aggressive/LeanFaster): `false` — preserves
        /// W44-29..W44-172 calibration; Pass-2 stays off at e=5/6.
        /// Libjxl strategy: `true` — fires LS-only Pass-2 at e=5/6 to
        /// match libjxl `fast=true` dispatch bit-for-bit.
        ///
        /// Section C (CfL parity, like W44-184/W44-195).
        ///
        /// See `docs/LIBJXL_DIVERGENCES.md` Section A (the `cfl_two_pass`
        /// effort-gate row already documents the Newton widening RULED
        /// OUT by W44-102; this gate is the orthogonal LS-only widening
        /// recommended by W44-189 D12 as a candidate the W44-102
        /// measurement did NOT cover).
        cfl_pass2_ls_at_low_effort: bool {
            divergence_section = "C",
            divergence_row_ref = "W44-197 CfL Pass-2 LS-only at e=5/6 (libjxl fast=true)",
        },

        // ── Section D Zenjxl tightening of W44-82 cost-benefit gate ──
        /// **W44-201**: skip buckets 3 (DCT32x32) and 6 (DCT32x16/DCT16x32)
        /// when admitting custom coefficient orders via the W44-82
        /// cost-benefit gate in `compute_custom_orders`
        /// (`vardct/coeff_order.rs:441-566`).
        ///
        /// W44-200 measurement on `3637739.png` e7 d=4 traced the
        /// Pareto-loser (+6.24% bytes / -2.37 SSIM2 vs cjxl) to a 488 B
        /// `coeff_orders` overspend in HfGlobal vs cjxl's <125 B. The
        /// dominant offender was DCT32x32 Y emitting 308 nonzero Lehmer
        /// codes vs cjxl's ~5. W44-201 per-block dump confirmed
        /// coefficient VALUES are bit-identical on shared blocks
        /// (98.4% strategy agreement with cjxl); the divergence is
        /// purely in the per-position zero count distribution falling
        /// into 3 distinct quantized bins (vs cjxl's 2 effective bins)
        /// causing the sort to produce a 308-Lehmer permutation.
        ///
        /// The W44-82-RULED-OUT cost-benefit model
        /// (`total_savings_bits = (nzeros_custom - nzeros_natural) *
        /// max_count`) assumes 1 bit saved per extra trailing zero
        /// per block; the empirical AC encoding cost per trailing zero
        /// is closer to 0.3-0.5 bits, leading the gate to admit
        /// permutations for buckets 3 and 6 that cost more bits in
        /// Lehmer encoding than they save in AC tokenization.
        ///
        /// Variant C bench (53 cells: 9 CID22 photos × 4 d + 5
        /// gb82-sc screenshots × 3 d + 2 W44-82 spot cells) shows
        /// **-0.65% total bytes, ZERO regressions** (worst +0.06%
        /// noise on `windows95_d1.0`). Effort sweep (e4..=e8) confirms
        /// gate fires at e5..=e8 on photos, never on the 4 screenshot
        /// cells (W44-82 cost-benefit was already not admitting
        /// bucket 3/6 there).
        ///
        /// Default (Zenjxl/Aggressive/LeanFaster): `true` — apply the
        /// fix. Libjxl strategy: `false` — preserves libjxl-parity
        /// (libjxl admits all buckets without per-bucket exclusion;
        /// the divergence is documented in Section D).
        ///
        /// Pixel-identical decoding verified A/C across 6 cells × 3
        /// distances via jxl-oxide (scan order only affects encoded
        /// Lehmer bytes, not coefficient values).
        ///
        /// Promoted from env var `JXL_W44_201_DISABLE_BUCKETS=3,6`
        /// (which can disable any comma-separated bucket set; the
        /// gate field hard-codes the W44-201 chosen set {3, 6}).
        coeff_orders_disable_large_buckets: bool {
            divergence_section = "D",
            divergence_row_ref = "W44-201 coeff_orders skip buckets 3+6 (Zenjxl tightening of W44-82 cost-benefit gate)",
        },

        /// **W44-205**: extension of W44-201's coeff_orders bucket-skip
        /// to the MEDIUM-sized buckets 2 (DCT16x16) and 4
        /// (DCT16x8/DCT8x16). Same mechanism, same root cause, same
        /// `compute_custom_orders_with_options` call site
        /// (`vardct/coeff_order.rs:603-623`).
        ///
        /// W44-204 audit C1: ranked as the #1 EV chunk follow-on to
        /// W44-201 because the W44-82 cost-model
        /// `total_savings_bits = (nzeros_custom - nzeros_natural) *
        /// max_count` is per-bucket and the same 1-bit-per-extra-zero
        /// overshoot that hurt buckets 3+6 also applies to medium
        /// buckets when per-position zero counts span 3+ quantized
        /// bins.
        ///
        /// W44-205 Phase 1 probe
        /// (`benchmarks/w44_205_bucket_probe_2026-05-22.tsv`, 27 cells:
        /// 16 LOSER_DOMINANT photos × 4 d at e=7 + 5 PROTECT cells +
        /// 6 SCRN cells) under env-hook `JXL_W44_201_DISABLE_BUCKETS=2,4`
        /// (on top of production buckets 3+6 disable): **-0.97% total
        /// bytes vs W44-201 baseline, ZERO PROTECT regressions, worst
        /// regression +0.14% noise on `SCRN_terminal_d4.0` (single
        /// cell)**. Per-bucket isolation (variant C bucket 2 only:
        /// -0.24%; variant D bucket 4 only: -0.73%) confirms most of
        /// the win comes from bucket 4 with bucket 2 contributing
        /// independently.
        ///
        /// Default (Zenjxl/Aggressive/LeanFaster): `true` — apply the
        /// extension. Libjxl strategy: `false` — preserves libjxl
        /// behaviour (libjxl admits ALL buckets via single
        /// `is_nondefault` check).
        ///
        /// Env hook `JXL_W44_205_FORCE_LEGACY_MEDIUM_BUCKETS=1` restores
        /// the pre-W44-205 behaviour (admit buckets 2 + 4) for
        /// diagnostic A/B benching. The pre-existing
        /// `JXL_W44_201_DISABLE_BUCKETS` env hook is preserved for
        /// arbitrary bucket-set tests.
        coeff_orders_disable_medium_buckets: bool {
            divergence_section = "D",
            divergence_row_ref = "W44-205 coeff_orders skip buckets 2+4 (Zenjxl extension of W44-201)",
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
    DivergenceEntry {
        gate_name: "high_colour_class_exclude",
        section: "B",
        row_ref: "W44-AUDIT-6 high-colour-class exclude from W44-109",
        raw: __CUSTOM_DIVERGENCE_HIGH_COLOUR_CLASS_EXCLUDE,
    },
    // Section C — CfL Newton parity
    DivergenceEntry {
        gate_name: "cfl_newton_libjxl_parity",
        section: "C",
        row_ref: "W44-184/W44-195 CfL Newton libjxl parity (Pass-1 dispatch + Pass-2 internals, eps=100, max_iters=20)",
        raw: __CUSTOM_DIVERGENCE_CFL_NEWTON_LIBJXL_PARITY,
    },
    // Section C — W44-AUDIT-5 Phase 2 Mode C hybrid CfL Newton
    DivergenceEntry {
        gate_name: "cfl_newton_libjxl_math_with_ls_warm_start",
        section: "C",
        row_ref: "W44-184/W44-195 Mode C — CfL Newton libjxl-math (eps=100, iters=20) + LS warm-start (W44-AUDIT-5 Phase 2)",
        raw: __CUSTOM_DIVERGENCE_CFL_NEWTON_LIBJXL_MATH_WITH_LS_WARM_START,
    },
    // Section C — W44-AUDIT-5 Phase 3 per-image M3>=80 CfL x=0 route
    DivergenceEntry {
        gate_name: "cfl_pass1_screenshot_x0_start",
        section: "C",
        row_ref: "W44-184/W44-195 Phase 3 — per-image M3>=80 CfL `x=0` start route (Pass-1 + Pass-2 dispatch, W44-AUDIT-5 Phase 3)",
        raw: __CUSTOM_DIVERGENCE_CFL_PASS1_SCREENSHOT_X0_START,
    },
    // Section C — W44-197 CfL Pass-2 LS-only at e=5/6
    DivergenceEntry {
        gate_name: "cfl_pass2_ls_at_low_effort",
        section: "C",
        row_ref: "W44-197 CfL Pass-2 LS-only at e=5/6 (libjxl fast=true)",
        raw: __CUSTOM_DIVERGENCE_CFL_PASS2_LS_AT_LOW_EFFORT,
    },
    // Section D — W44-201 Zenjxl tightening of W44-82 cost-benefit gate
    DivergenceEntry {
        gate_name: "coeff_orders_disable_large_buckets",
        section: "D",
        row_ref: "W44-201 coeff_orders skip buckets 3+6 (Zenjxl tightening of W44-82 cost-benefit gate)",
        raw: __CUSTOM_DIVERGENCE_COEFF_ORDERS_DISABLE_LARGE_BUCKETS,
    },
    // Section D — W44-205 extension of W44-201 to medium buckets 2 + 4
    DivergenceEntry {
        gate_name: "coeff_orders_disable_medium_buckets",
        section: "D",
        row_ref: "W44-205 coeff_orders skip buckets 2+4 (Zenjxl extension of W44-201)",
        raw: __CUSTOM_DIVERGENCE_COEFF_ORDERS_DISABLE_MEDIUM_BUCKETS,
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
        // W44-AUDIT-6 Phase 1: Zenjxl default = ON.
        assert!(d.high_colour_class_exclude);
        // Section C
        assert!(!d.cfl_newton_libjxl_parity);
        // W44-AUDIT-5 Phase 2: opt-in only on Zenjxl (HONEST-STOP — Mode C
        // measured byte-identical to Mode A on bisect cells).
        assert!(!d.cfl_newton_libjxl_math_with_ls_warm_start);
        assert!(!d.cfl_pass2_ls_at_low_effort);
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
        // W44-197: Libjxl flips Pass-2 LS-only at e=5/6 on; Zenjxl off.
        assert_ne!(l.cfl_pass2_ls_at_low_effort, z.cfl_pass2_ls_at_low_effort);
        // W44-AUDIT-5 Phase 2 (Mode C): both Libjxl and Zenjxl default
        // to `false` after the HONEST-STOP (Libjxl because parity takes
        // priority; Zenjxl because Mode C measured byte-identical to
        // Mode A on the bisect cells). So this is the ONE Section C
        // field where Libjxl and Zenjxl agree on default value.
        assert_eq!(
            l.cfl_newton_libjxl_math_with_ls_warm_start,
            z.cfl_newton_libjxl_math_with_ls_warm_start
        );
        assert!(!l.cfl_newton_libjxl_math_with_ls_warm_start);
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
        // W44-AUDIT-5 Phase 2 (Mode C)
        assert!(
            __CUSTOM_DIVERGENCE_CFL_NEWTON_LIBJXL_MATH_WITH_LS_WARM_START.contains("section=C")
        );
        assert!(
            __CUSTOM_DIVERGENCE_CFL_NEWTON_LIBJXL_MATH_WITH_LS_WARM_START.contains("W44-AUDIT-5")
        );
    }
}
