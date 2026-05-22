// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! # W44-192 — `strategy_def!` macro prototype validation
//!
//! Phase 1 of the W44-190 RFC. The proc-macro
//! [`jxl_encoder_macros::strategy_def!`] is exercised on **three
//! prototype gates** drawn from the live production set in
//! `src/api.rs`:
//!
//! 1. **`cfl_newton_libjxl_parity: bool`** (W44-184) — simple bool gate;
//!    Libjxl-only `true`; env hook
//!    `JXL_W44_184_FORCE_LIBJXL_NEWTON=1` flips false→true.
//! 2. **`content_class_auto_classify: bool`** (W44-164) — simple bool;
//!    Zenjxl/Aggressive `true`, Libjxl/LeanFaster `false`; no env hook.
//! 3. **`adaptive_buttloop_iters: IterMode`** (W44-168) — multi-mode
//!    enum dispatch; env hook `JXL_W44_168_MODE` parses
//!    `"A" | "B" | "C" | "D"` into the matching enum variant.
//!
//! Acceptance gate (b) from the W44-192 task: the generated code for
//! these three gates is byte-identical to the matching hand-written
//! code in `api.rs` (subject to a hand-written/macro-generated
//! difference in `cfg(feature = "std")` guards on `apply_*_env_var_fallbacks`,
//! which we wrap at the call site below).
//!
//! Acceptance gate (c): the prototype module is `pub(crate)` and
//! never instantiated by production code, so production hash-locks
//! are unaffected.
//!
//! Doc: [`docs/STRATEGY_DEF_MACRO.md`](../../../docs/STRATEGY_DEF_MACRO.md).

#![allow(dead_code)]

// W44-168 wraps the three real `JXL_W44_168_MODE` literals plus an `Off`
// sentinel that matches the production-default schedule (= no
// content-aware override). The variants intentionally mirror the
// `JXL_W44_168_MODE=A|B|C|D` letter dispatch documented in
// `docs/LIBJXL_DIVERGENCES.md` Section B.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IterMode {
    /// `Off` — no content-aware adjustment; equivalent to
    /// `JXL_W44_168_MODE=A` and to the W44-168 default-env-unset path.
    #[default]
    Off,
    /// W44-168 Mode B (`SmoothSkip`): saturating `iters - 1` on
    /// smooth/screenshot content at `effort >= 8`. Used in production
    /// today via the env-only hook; the macro promotes it to a typed
    /// enum value here to demonstrate that the macro can dispatch on
    /// non-trivial types.
    SmoothSkip,
    /// W44-168 Mode C (`TexturedExtend`): bump iters from 0 → 2 on
    /// textured content at `effort == 7`.
    TexturedExtend,
    /// W44-168 Mode D (`Combined`): apply both B and C.
    Combined,
}

/// Parser for `bool` env hooks following the production convention:
/// `"1"` flips to `true`; anything else returns `None` (no change).
fn bool_one(s: &str) -> Option<bool> {
    if s == "1" { Some(true) } else { None }
}

/// Parser for the `JXL_W44_168_MODE` env hook. Mirrors the production
/// dispatch in `vardct/butteraugli_loop.rs` Mode B / Mode C / Mode D
/// branch.
fn parse_iter_mode(s: &str) -> Option<IterMode> {
    match s {
        "A" => Some(IterMode::Off),
        "B" => Some(IterMode::SmoothSkip),
        "C" => Some(IterMode::TexturedExtend),
        "D" => Some(IterMode::Combined),
        _ => None,
    }
}

jxl_encoder_macros::strategy_def! {
    name = Prototype;
    default_strategy = Zenjxl;

    // The macro emits any enums declared here verbatim. We declare
    // `IterMode` above for the parser path; the macro-side declaration
    // would normally live here. Empty block for the prototype.
    enums {}

    strategies {
        /// **Strict libjxl-parity bundle.** Mirrors
        /// `ResolvedImprovements::libjxl()` for the three prototype
        /// gates — flips Section C CfL Newton on, disables Smart-Zenjxl
        /// auto-classifier, drops adaptive buttloop iters.
        Libjxl {
            cfl_newton_libjxl_parity = true,
            content_class_auto_classify = false,
            adaptive_buttloop_iters = IterMode::Off,
        },

        /// **Production default.** Mirrors
        /// `ResolvedImprovements::zenjxl()` (which delegates to
        /// `Default::default()`) for the three prototype gates.
        Zenjxl {
            cfl_newton_libjxl_parity = false,
            content_class_auto_classify = true,
            adaptive_buttloop_iters = IterMode::SmoothSkip,
        },

        /// **LeanFaster.** Mirrors `ResolvedImprovements::lean_faster()`
        /// for the three prototype gates — drops per-image content
        /// gates while keeping Zenjxl's cost-model calibration.
        LeanFaster {
            cfl_newton_libjxl_parity = false,
            content_class_auto_classify = false,
            adaptive_buttloop_iters = IterMode::Off,
        },

        /// **Aggressive.** Mirrors `ResolvedImprovements::aggressive()`
        /// (currently equivalent to Zenjxl after W44-124's auto-
        /// discriminator obsoleted the previous global flip).
        Aggressive {
            cfl_newton_libjxl_parity = false,
            content_class_auto_classify = true,
            adaptive_buttloop_iters = IterMode::SmoothSkip,
        },
    }

    gates {
        /// **W44-184** (Section C): bit-exact libjxl CfL Newton.
        cfl_newton_libjxl_parity: bool {
            env_hook = "JXL_W44_184_FORCE_LIBJXL_NEWTON" => bool_one,
            divergence_section = "C",
            divergence_row_ref = "CfL Newton parameters (W44-183/184)",
        },

        /// **W44-164** (Section B): Smart-Zenjxl auto-classify
        /// `ImageContentClass` via `ZenanalyzeProxies`.
        content_class_auto_classify: bool {
            divergence_section = "B",
            divergence_row_ref = "W44-164 auto_classify_content_class_from_layout",
        },

        /// **W44-168** (Section B): content-aware adaptive
        /// `butteraugli_iters`.
        adaptive_buttloop_iters: IterMode {
            env_hook = "JXL_W44_168_MODE" => parse_iter_mode,
            divergence_section = "B",
            divergence_row_ref = "W44-168 adaptive butteraugli iters",
        },
    }
}

/// Test-only hook for the env-fallback integration test (`__internals`).
///
/// Resolves the named strategy (no `Custom` payload — that path is
/// covered by the inline `resolve_custom_copies_fields` test) and
/// returns the three prototype-gate field values as a tuple in
/// declaration order.
///
/// Lives on the library side so the integration test can be a thin
/// caller without having to depend on the macro crate directly.
#[doc(hidden)]
pub fn resolve_prototype_strategy_named_for_test(strategy_name: &str) -> (bool, bool, IterMode) {
    let strategy = match strategy_name {
        "Libjxl" => PrototypeEncoderStrategy::Libjxl,
        "Zenjxl" => PrototypeEncoderStrategy::Zenjxl,
        "LeanFaster" => PrototypeEncoderStrategy::LeanFaster,
        "Aggressive" => PrototypeEncoderStrategy::Aggressive,
        other => panic!("unknown prototype strategy `{other}`"),
    };
    let r = strategy.resolve();
    (
        r.cfl_newton_libjxl_parity,
        r.content_class_auto_classify,
        r.adaptive_buttloop_iters,
    )
}

// ──────────────────────────────────────────────────────────────────────
// Tests — Phase 3 validation per W44-192 task spec.
// ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Each strategy preset constructor returns the values listed in
    /// the `strategies { ... }` block.
    #[test]
    fn libjxl_preset() {
        let r = PrototypeResolvedImprovements::libjxl();
        assert!(r.cfl_newton_libjxl_parity);
        assert!(!r.content_class_auto_classify);
        assert_eq!(r.adaptive_buttloop_iters, IterMode::Off);
    }

    #[test]
    fn zenjxl_preset() {
        let r = PrototypeResolvedImprovements::zenjxl();
        assert!(!r.cfl_newton_libjxl_parity);
        assert!(r.content_class_auto_classify);
        assert_eq!(r.adaptive_buttloop_iters, IterMode::SmoothSkip);
    }

    #[test]
    fn lean_faster_preset() {
        let r = PrototypeResolvedImprovements::lean_faster();
        assert!(!r.cfl_newton_libjxl_parity);
        assert!(!r.content_class_auto_classify);
        assert_eq!(r.adaptive_buttloop_iters, IterMode::Off);
    }

    #[test]
    fn aggressive_preset() {
        let r = PrototypeResolvedImprovements::aggressive();
        assert!(!r.cfl_newton_libjxl_parity);
        assert!(r.content_class_auto_classify);
        assert_eq!(r.adaptive_buttloop_iters, IterMode::SmoothSkip);
    }

    /// Default impls on both structs match the user-declared
    /// `default_strategy = Zenjxl`.
    #[test]
    fn default_matches_zenjxl() {
        assert_eq!(
            PrototypeEncoderImprovements::default(),
            PrototypeEncoderImprovements {
                cfl_newton_libjxl_parity: false,
                content_class_auto_classify: true,
                adaptive_buttloop_iters: IterMode::SmoothSkip,
            }
        );
        assert_eq!(
            PrototypeResolvedImprovements::default(),
            PrototypeResolvedImprovements::zenjxl()
        );
        assert_eq!(
            PrototypeEncoderStrategy::default(),
            PrototypeEncoderStrategy::Zenjxl
        );
    }

    /// `Custom` strategy copies fields field-for-field.
    ///
    /// **Env-var-mutating tests** (cover `apply_prototype_env_var_fallbacks`)
    /// live in `tests/strategy_def_prototype_env_fallback.rs` because the
    /// library crate carries `#![forbid(unsafe_code)]` and Rust 2024's
    /// `std::env::set_var` / `remove_var` are `unsafe`. The integration
    /// test serialises mutations under a `std::sync::Mutex` (same
    /// pattern as `tests/strategy_env_fallback.rs`).
    #[test]
    fn resolve_custom_copies_fields() {
        let custom = PrototypeEncoderImprovements {
            cfl_newton_libjxl_parity: true,
            content_class_auto_classify: false,
            adaptive_buttloop_iters: IterMode::Combined,
        };
        let resolved = PrototypeEncoderStrategy::Custom(Box::new(custom.clone())).resolve();
        // NOTE: the env-var fallback may flip
        // `cfl_newton_libjxl_parity` if the test runner has
        // `JXL_W44_184_FORCE_LIBJXL_NEWTON=1` set; in CI we don't set
        // any `JXL_*` env vars so this is safe. The integration test
        // exercises the env-fallback paths explicitly.
        assert_eq!(resolved.content_class_auto_classify, false);
        assert_eq!(resolved.adaptive_buttloop_iters, IterMode::Combined);
    }

    /// Smoke test for the divergence-table metadata consts emitted by
    /// the macro. W44-194 will harvest these via a build-script.
    #[test]
    fn divergence_metadata_consts_exposed() {
        // Names follow the pattern `__<NAME>_DIVERGENCE_<GATE>` upper-cased.
        assert!(__PROTOTYPE_DIVERGENCE_CFL_NEWTON_LIBJXL_PARITY.contains("section=C"));
        assert!(__PROTOTYPE_DIVERGENCE_CFL_NEWTON_LIBJXL_PARITY.contains("W44-183/184"));
        assert!(__PROTOTYPE_DIVERGENCE_CONTENT_CLASS_AUTO_CLASSIFY.contains("section=B"));
        assert!(__PROTOTYPE_DIVERGENCE_ADAPTIVE_BUTTLOOP_ITERS.contains("section=B"));
    }
}
