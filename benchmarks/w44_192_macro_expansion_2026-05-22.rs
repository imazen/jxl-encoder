pub(crate) mod strategy_def_prototype {
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
    #[automatically_derived]
    #[doc(hidden)]
    unsafe impl ::core::clone::TrivialClone for IterMode {}
    #[automatically_derived]
    impl ::core::clone::Clone for IterMode {
        #[inline]
        fn clone(&self) -> IterMode {
            *self
        }
    }
    #[automatically_derived]
    impl ::core::marker::Copy for IterMode {}
    #[automatically_derived]
    impl ::core::fmt::Debug for IterMode {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::write_str(
                f,
                match self {
                    IterMode::Off => "Off",
                    IterMode::SmoothSkip => "SmoothSkip",
                    IterMode::TexturedExtend => "TexturedExtend",
                    IterMode::Combined => "Combined",
                },
            )
        }
    }
    #[automatically_derived]
    impl ::core::default::Default for IterMode {
        #[inline]
        fn default() -> IterMode {
            Self::Off
        }
    }
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for IterMode {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for IterMode {
        #[inline]
        fn eq(&self, other: &IterMode) -> bool {
            let __self_discr = ::core::intrinsics::discriminant_value(self);
            let __arg1_discr = ::core::intrinsics::discriminant_value(other);
            __self_discr == __arg1_discr
        }
    }
    #[automatically_derived]
    impl ::core::cmp::Eq for IterMode {
        #[doc(hidden)]
        #[coverage(off)]
        fn assert_fields_are_eq(&self) {}
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
    pub struct PrototypeEncoderImprovements {
        /// **W44-184** (Section C): bit-exact libjxl CfL Newton.
        pub cfl_newton_libjxl_parity: bool,
        /// **W44-164** (Section B): Smart-Zenjxl auto-classify
        /// `ImageContentClass` via `ZenanalyzeProxies`.
        pub content_class_auto_classify: bool,
        /// **W44-168** (Section B): content-aware adaptive
        /// `butteraugli_iters`.
        pub adaptive_buttloop_iters: IterMode,
    }
    #[automatically_derived]
    impl ::core::clone::Clone for PrototypeEncoderImprovements {
        #[inline]
        fn clone(&self) -> PrototypeEncoderImprovements {
            PrototypeEncoderImprovements {
                cfl_newton_libjxl_parity: ::core::clone::Clone::clone(
                    &self.cfl_newton_libjxl_parity,
                ),
                content_class_auto_classify: ::core::clone::Clone::clone(
                    &self.content_class_auto_classify,
                ),
                adaptive_buttloop_iters: ::core::clone::Clone::clone(
                    &self.adaptive_buttloop_iters,
                ),
            }
        }
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for PrototypeEncoderImprovements {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field3_finish(
                f,
                "PrototypeEncoderImprovements",
                "cfl_newton_libjxl_parity",
                &self.cfl_newton_libjxl_parity,
                "content_class_auto_classify",
                &self.content_class_auto_classify,
                "adaptive_buttloop_iters",
                &&self.adaptive_buttloop_iters,
            )
        }
    }
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for PrototypeEncoderImprovements {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for PrototypeEncoderImprovements {
        #[inline]
        fn eq(&self, other: &PrototypeEncoderImprovements) -> bool {
            self.cfl_newton_libjxl_parity == other.cfl_newton_libjxl_parity
                && self.content_class_auto_classify == other.content_class_auto_classify
                && self.adaptive_buttloop_iters == other.adaptive_buttloop_iters
        }
    }
    impl ::core::default::Default for PrototypeEncoderImprovements {
        fn default() -> Self {
            Self {
                cfl_newton_libjxl_parity: false,
                content_class_auto_classify: true,
                adaptive_buttloop_iters: IterMode::SmoothSkip,
            }
        }
    }
    #[allow(dead_code)]
    pub(crate) struct PrototypeResolvedImprovements {
        pub(crate) cfl_newton_libjxl_parity: bool,
        pub(crate) content_class_auto_classify: bool,
        pub(crate) adaptive_buttloop_iters: IterMode,
    }
    #[automatically_derived]
    #[allow(dead_code)]
    impl ::core::clone::Clone for PrototypeResolvedImprovements {
        #[inline]
        fn clone(&self) -> PrototypeResolvedImprovements {
            PrototypeResolvedImprovements {
                cfl_newton_libjxl_parity: ::core::clone::Clone::clone(
                    &self.cfl_newton_libjxl_parity,
                ),
                content_class_auto_classify: ::core::clone::Clone::clone(
                    &self.content_class_auto_classify,
                ),
                adaptive_buttloop_iters: ::core::clone::Clone::clone(
                    &self.adaptive_buttloop_iters,
                ),
            }
        }
    }
    #[automatically_derived]
    #[allow(dead_code)]
    impl ::core::fmt::Debug for PrototypeResolvedImprovements {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field3_finish(
                f,
                "PrototypeResolvedImprovements",
                "cfl_newton_libjxl_parity",
                &self.cfl_newton_libjxl_parity,
                "content_class_auto_classify",
                &self.content_class_auto_classify,
                "adaptive_buttloop_iters",
                &&self.adaptive_buttloop_iters,
            )
        }
    }
    #[automatically_derived]
    #[allow(dead_code)]
    impl ::core::marker::StructuralPartialEq for PrototypeResolvedImprovements {}
    #[automatically_derived]
    #[allow(dead_code)]
    impl ::core::cmp::PartialEq for PrototypeResolvedImprovements {
        #[inline]
        fn eq(&self, other: &PrototypeResolvedImprovements) -> bool {
            self.cfl_newton_libjxl_parity == other.cfl_newton_libjxl_parity
                && self.content_class_auto_classify == other.content_class_auto_classify
                && self.adaptive_buttloop_iters == other.adaptive_buttloop_iters
        }
    }
    impl ::core::default::Default for PrototypeResolvedImprovements {
        fn default() -> Self {
            PrototypeResolvedImprovements::zenjxl()
        }
    }
    impl PrototypeResolvedImprovements {
        /**Generated by `strategy_def!` for strategy variant `Libjxl`.

Set per-gate values are listed inline.*/
        pub fn libjxl() -> Self {
            Self {
                cfl_newton_libjxl_parity: true,
                content_class_auto_classify: false,
                adaptive_buttloop_iters: IterMode::Off,
            }
        }
        /**Generated by `strategy_def!` for strategy variant `Zenjxl`.

Set per-gate values are listed inline.*/
        pub fn zenjxl() -> Self {
            Self {
                cfl_newton_libjxl_parity: false,
                content_class_auto_classify: true,
                adaptive_buttloop_iters: IterMode::SmoothSkip,
            }
        }
        /**Generated by `strategy_def!` for strategy variant `LeanFaster`.

Set per-gate values are listed inline.*/
        pub fn lean_faster() -> Self {
            Self {
                cfl_newton_libjxl_parity: false,
                content_class_auto_classify: false,
                adaptive_buttloop_iters: IterMode::Off,
            }
        }
        /**Generated by `strategy_def!` for strategy variant `Aggressive`.

Set per-gate values are listed inline.*/
        pub fn aggressive() -> Self {
            Self {
                cfl_newton_libjxl_parity: false,
                content_class_auto_classify: true,
                adaptive_buttloop_iters: IterMode::SmoothSkip,
            }
        }
        /// Copy every field from the Custom struct.
        pub fn from_custom(c: &PrototypeEncoderImprovements) -> Self {
            Self {
                cfl_newton_libjxl_parity: c.cfl_newton_libjxl_parity.clone(),
                content_class_auto_classify: c.content_class_auto_classify.clone(),
                adaptive_buttloop_iters: c.adaptive_buttloop_iters.clone(),
            }
        }
    }
    pub enum PrototypeEncoderStrategy {
        /// **Strict libjxl-parity bundle.** Mirrors
        /// `ResolvedImprovements::libjxl()` for the three prototype
        /// gates — flips Section C CfL Newton on, disables Smart-Zenjxl
        /// auto-classifier, drops adaptive buttloop iters.
        Libjxl,
        /// **Production default.** Mirrors
        /// `ResolvedImprovements::zenjxl()` (which delegates to
        /// `Default::default()`) for the three prototype gates.
        #[default]
        Zenjxl,
        /// **LeanFaster.** Mirrors `ResolvedImprovements::lean_faster()`
        /// for the three prototype gates — drops per-image content
        /// gates while keeping Zenjxl's cost-model calibration.
        LeanFaster,
        /// **Aggressive.** Mirrors `ResolvedImprovements::aggressive()`
        /// (currently equivalent to Zenjxl after W44-124's auto-
        /// discriminator obsoleted the previous global flip).
        Aggressive,
        /// Caller-defined per-field overrides. Always reachable.
        Custom(::std::boxed::Box<PrototypeEncoderImprovements>),
    }
    #[automatically_derived]
    impl ::core::clone::Clone for PrototypeEncoderStrategy {
        #[inline]
        fn clone(&self) -> PrototypeEncoderStrategy {
            match self {
                PrototypeEncoderStrategy::Libjxl => PrototypeEncoderStrategy::Libjxl,
                PrototypeEncoderStrategy::Zenjxl => PrototypeEncoderStrategy::Zenjxl,
                PrototypeEncoderStrategy::LeanFaster => {
                    PrototypeEncoderStrategy::LeanFaster
                }
                PrototypeEncoderStrategy::Aggressive => {
                    PrototypeEncoderStrategy::Aggressive
                }
                PrototypeEncoderStrategy::Custom(__self_0) => {
                    PrototypeEncoderStrategy::Custom(
                        ::core::clone::Clone::clone(__self_0),
                    )
                }
            }
        }
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for PrototypeEncoderStrategy {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match self {
                PrototypeEncoderStrategy::Libjxl => {
                    ::core::fmt::Formatter::write_str(f, "Libjxl")
                }
                PrototypeEncoderStrategy::Zenjxl => {
                    ::core::fmt::Formatter::write_str(f, "Zenjxl")
                }
                PrototypeEncoderStrategy::LeanFaster => {
                    ::core::fmt::Formatter::write_str(f, "LeanFaster")
                }
                PrototypeEncoderStrategy::Aggressive => {
                    ::core::fmt::Formatter::write_str(f, "Aggressive")
                }
                PrototypeEncoderStrategy::Custom(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Custom",
                        &__self_0,
                    )
                }
            }
        }
    }
    #[automatically_derived]
    impl ::core::default::Default for PrototypeEncoderStrategy {
        #[inline]
        fn default() -> PrototypeEncoderStrategy {
            Self::Zenjxl
        }
    }
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for PrototypeEncoderStrategy {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for PrototypeEncoderStrategy {
        #[inline]
        fn eq(&self, other: &PrototypeEncoderStrategy) -> bool {
            let __self_discr = ::core::intrinsics::discriminant_value(self);
            let __arg1_discr = ::core::intrinsics::discriminant_value(other);
            __self_discr == __arg1_discr
                && match (self, other) {
                    (
                        PrototypeEncoderStrategy::Custom(__self_0),
                        PrototypeEncoderStrategy::Custom(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    _ => true,
                }
        }
    }
    impl PrototypeEncoderStrategy {
        /// Resolve the strategy enum to a `ResolvedImprovements`
        /// struct. Env-var fallbacks are applied at the end (only
        /// for gates whose resolved value equals its type-level
        /// `Default::default()`).
        pub fn resolve(&self) -> PrototypeResolvedImprovements {
            let mut __resolved = match self {
                Self::Libjxl => PrototypeResolvedImprovements::libjxl(),
                Self::Zenjxl => PrototypeResolvedImprovements::zenjxl(),
                Self::LeanFaster => PrototypeResolvedImprovements::lean_faster(),
                Self::Aggressive => PrototypeResolvedImprovements::aggressive(),
                Self::Custom(c) => PrototypeResolvedImprovements::from_custom(c.as_ref()),
            };
            apply_prototype_env_var_fallbacks(&mut __resolved);
            __resolved
        }
    }
    /// Per-gate env-var fallback applied by `EncoderStrategy::resolve`.
    ///
    /// Std-only (the macro currently emits an unconditional
    /// `std::env::var` call — wrap in `#[cfg(feature = "std")]` at
    /// the call site if a `no_std` build is supported).
    #[allow(dead_code)]
    fn apply_prototype_env_var_fallbacks(r: &mut PrototypeResolvedImprovements) {
        if r.cfl_newton_libjxl_parity == <bool as ::core::default::Default>::default()
            && let Ok(__s) = ::std::env::var("JXL_W44_184_FORCE_LIBJXL_NEWTON")
            && let Some(__v) = bool_one(__s.as_str())
        {
            r.cfl_newton_libjxl_parity = __v;
        }
        if r.adaptive_buttloop_iters == <IterMode as ::core::default::Default>::default()
            && let Ok(__s) = ::std::env::var("JXL_W44_168_MODE")
            && let Some(__v) = parse_iter_mode(__s.as_str())
        {
            r.adaptive_buttloop_iters = __v;
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, dead_code)]
    pub const __PROTOTYPE_DIVERGENCE_CFL_NEWTON_LIBJXL_PARITY: &str = "section=C ; row_ref=CfL Newton parameters (W44-183/184)";
    #[doc(hidden)]
    #[allow(non_upper_case_globals, dead_code)]
    pub const __PROTOTYPE_DIVERGENCE_CONTENT_CLASS_AUTO_CLASSIFY: &str = "section=B ; row_ref=W44-164 auto_classify_content_class_from_layout";
    #[doc(hidden)]
    #[allow(non_upper_case_globals, dead_code)]
    pub const __PROTOTYPE_DIVERGENCE_ADAPTIVE_BUTTLOOP_ITERS: &str = "section=B ; row_ref=W44-168 adaptive butteraugli iters";
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
    pub fn resolve_prototype_strategy_named_for_test(
        strategy_name: &str,
    ) -> (bool, bool, IterMode) {
        let strategy = match strategy_name {
            "Libjxl" => PrototypeEncoderStrategy::Libjxl,
            "Zenjxl" => PrototypeEncoderStrategy::Zenjxl,
            "LeanFaster" => PrototypeEncoderStrategy::LeanFaster,
            "Aggressive" => PrototypeEncoderStrategy::Aggressive,
            other => {
                ::core::panicking::panic_fmt(
                    format_args!("unknown prototype strategy `{0}`", other),
                );
            }
        };
        let r = strategy.resolve();
        (
            r.cfl_newton_libjxl_parity,
            r.content_class_auto_classify,
            r.adaptive_buttloop_iters,
        )
    }
}
