//! # jxl-encoder-macros
//!
//! Proc-macros backing jxl-encoder's declarative gate registry (W44-192).
//!
//! The flagship macro [`strategy_def!`] generates the boilerplate for one
//! self-contained _strategy bundle_: a pair of structs
//! (`<Name>EncoderImprovements` and `<Name>ResolvedImprovements`), one
//! `<Name>EncoderStrategy` enum, four named constructors per strategy
//! variant, a `resolve()` method that composes user overrides with
//! optional env-var fallbacks, and an `EncoderStrategy::default()` impl
//! pointing at the user-declared default strategy.
//!
//! See [`docs/STRATEGY_DEF_MACRO.md`](../docs/STRATEGY_DEF_MACRO.md) and
//! the W44-190 RFC at
//! [`docs/RFC-strategy-refactor-2026-05-22.md`](../docs/RFC-strategy-refactor-2026-05-22.md)
//! for the design context and migration plan.
//!
//! ## Quick example
//!
//! ```ignore
//! use jxl_encoder_macros::strategy_def;
//!
//! strategy_def! {
//!     name = Prototype;
//!     default_strategy = Zenjxl;
//!
//!     enums {
//!         pub enum IterMode { Off, Auto, SmoothSkip }
//!     }
//!
//!     strategies {
//!         Libjxl { cfl_newton_libjxl_parity = true, content_class_auto_classify = false, adaptive_buttloop_iters = IterMode::Off, },
//!         Zenjxl { cfl_newton_libjxl_parity = false, content_class_auto_classify = true, adaptive_buttloop_iters = IterMode::Auto, },
//!         LeanFaster { cfl_newton_libjxl_parity = false, content_class_auto_classify = false, adaptive_buttloop_iters = IterMode::Off, },
//!         Aggressive { cfl_newton_libjxl_parity = false, content_class_auto_classify = true, adaptive_buttloop_iters = IterMode::Auto, },
//!     }
//!
//!     gates {
//!         /// W44-184: ...
//!         cfl_newton_libjxl_parity: bool {
//!             env_hook = "JXL_W44_184_FORCE_LIBJXL_NEWTON" => bool_one,
//!         },
//!         content_class_auto_classify: bool {},
//!         adaptive_buttloop_iters: IterMode {
//!             env_hook = "JXL_W44_168_MODE" => parse_iter_mode,
//!         },
//!     }
//! }
//! ```
//!
//! ### What's generated
//!
//! For `name = Prototype` the macro emits (roughly):
//!
//! ```ignore
//! pub enum IterMode { Off, Auto, SmoothSkip }
//!
//! #[derive(Clone, Debug, PartialEq)]
//! pub struct PrototypeEncoderImprovements {
//!     pub cfl_newton_libjxl_parity: bool,
//!     pub content_class_auto_classify: bool,
//!     pub adaptive_buttloop_iters: IterMode,
//! }
//!
//! impl Default for PrototypeEncoderImprovements { /* matches Zenjxl */ }
//!
//! #[derive(Clone, Debug, PartialEq)]
//! pub struct PrototypeResolvedImprovements { /* same fields, pub(crate) */ }
//!
//! impl PrototypeResolvedImprovements {
//!     pub fn libjxl() -> Self { /* ... */ }
//!     pub fn zenjxl() -> Self { /* ... */ }
//!     pub fn lean_faster() -> Self { /* ... */ }
//!     pub fn aggressive() -> Self { /* ... */ }
//!     pub fn from_custom(c: &PrototypeEncoderImprovements) -> Self { /* ... */ }
//! }
//!
//! #[derive(Clone, Debug, Default, PartialEq)]
//! pub enum PrototypeEncoderStrategy {
//!     Libjxl,
//!     #[default]
//!     Zenjxl,
//!     LeanFaster,
//!     Aggressive,
//!     Custom(Box<PrototypeEncoderImprovements>),
//! }
//!
//! impl PrototypeEncoderStrategy {
//!     pub fn resolve(&self) -> PrototypeResolvedImprovements { /* ... + env fallback */ }
//! }
//!
//! fn apply_prototype_env_var_fallbacks(r: &mut PrototypeResolvedImprovements) {
//!     // For each gate with `env_hook = "NAME" => parser`:
//!     // - read the env var; if unset, no-op.
//!     // - if the resolved field equals its Default value, apply parser
//!     //   to override; otherwise the explicit caller setting wins.
//! }
//! ```
//!
//! Env-var lookups run **at resolve-time only** (typically once per
//! encode), never in the encode hot path. The `apply_*_env_var_fallbacks`
//! function is `#[cfg(feature = "std")]`-gated by the consumer of the
//! generated code (the consumer must wrap the call in the same `cfg`); we
//! emit the call unconditionally so callers can decide.
//!
//! ## Env-hook parser convention
//!
//! Each `env_hook = "NAME" => parser` clause names a parser fn-item with
//! signature `fn(env_value: &str) -> Option<T>` where `T` is the field's
//! type. Returning `None` keeps the resolved value at its current
//! (post-overrides) value. Two helpers are exposed by the consumer crate:
//!
//! - `bool_one(s: &str) -> Option<bool>` — returns `Some(true)` if `s == "1"`,
//!   else `None`.
//! - `parse_f32(s: &str) -> Option<f32>` — wraps `f32::from_str` returning
//!   `Option`.
//!
//! Custom parsers (one per enum) live in the consumer crate next to the
//! enum definition.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Attribute, Expr, Ident, ItemEnum, LitStr, Path, Token, Type, braced, parse_macro_input};

// ──────────────────────────────────────────────────────────────────────
// AST
// ──────────────────────────────────────────────────────────────────────

/// The whole `strategy_def! { ... }` input.
struct StrategyDefInput {
    name: Ident,
    default_strategy: Ident,
    enums: Vec<ItemEnum>,
    strategies: Vec<StrategyVariant>,
    gates: Vec<GateDef>,
}

/// One strategy variant declaration: `Zenjxl { gate1 = expr, gate2 = expr, ... }`.
struct StrategyVariant {
    attrs: Vec<Attribute>,
    name: Ident,
    /// Field assignments: `gate_name = expr`. Must cover every gate in
    /// the `gates {...}` block; checked at expansion time.
    values: Vec<(Ident, Expr)>,
}

/// One gate metadata block: `name: ty { env_hook = "..." => parser, ... }`.
struct GateDef {
    attrs: Vec<Attribute>,
    name: Ident,
    ty: Type,
    env_hook: Option<(LitStr, Path)>,
    /// Free-form metadata kept for the W44-194 build-script that
    /// auto-generates the divergence table. We capture-and-pass-through.
    divergence_section: Option<LitStr>,
    divergence_row_ref: Option<LitStr>,
}

// ──────────────────────────────────────────────────────────────────────
// Parser
// ──────────────────────────────────────────────────────────────────────

impl Parse for StrategyDefInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name: Option<Ident> = None;
        let mut default_strategy: Option<Ident> = None;
        let mut enums = Vec::new();
        let mut strategies: Option<Vec<StrategyVariant>> = None;
        let mut gates: Option<Vec<GateDef>> = None;

        while !input.is_empty() {
            let lookahead = input.lookahead1();
            if lookahead.peek(Ident) {
                let key: Ident = input.parse()?;
                let key_str = key.to_string();
                match key_str.as_str() {
                    "name" => {
                        input.parse::<Token![=]>()?;
                        name = Some(input.parse::<Ident>()?);
                        input.parse::<Token![;]>()?;
                    }
                    "default_strategy" => {
                        input.parse::<Token![=]>()?;
                        default_strategy = Some(input.parse::<Ident>()?);
                        input.parse::<Token![;]>()?;
                    }
                    "enums" => {
                        let content;
                        braced!(content in input);
                        while !content.is_empty() {
                            enums.push(content.parse::<ItemEnum>()?);
                        }
                    }
                    "strategies" => {
                        let content;
                        braced!(content in input);
                        let mut list = Vec::new();
                        while !content.is_empty() {
                            list.push(parse_strategy_variant(&content)?);
                            // Optional trailing comma between strategies.
                            let _ = content.parse::<Token![,]>();
                        }
                        strategies = Some(list);
                    }
                    "gates" => {
                        let content;
                        braced!(content in input);
                        let mut list = Vec::new();
                        while !content.is_empty() {
                            list.push(parse_gate_def(&content)?);
                            let _ = content.parse::<Token![,]>();
                        }
                        gates = Some(list);
                    }
                    other => {
                        return Err(syn::Error::new(
                            key.span(),
                            format!(
                                "strategy_def!: unknown top-level key `{}` (expected one of \
                                 `name`, `default_strategy`, `enums`, `strategies`, `gates`)",
                                other
                            ),
                        ));
                    }
                }
            } else {
                return Err(lookahead.error());
            }
        }

        let name = name.ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "strategy_def!: missing `name = <Ident>;` top-level key",
            )
        })?;
        let default_strategy = default_strategy.ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "strategy_def!: missing `default_strategy = <Ident>;` top-level key",
            )
        })?;
        let strategies = strategies.ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "strategy_def!: missing `strategies { ... }` block",
            )
        })?;
        let gates = gates.ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "strategy_def!: missing `gates { ... }` block",
            )
        })?;

        // Cross-check: every strategy must list every gate (no missing,
        // no extras).
        let gate_names: Vec<&Ident> = gates.iter().map(|g| &g.name).collect();
        let gate_name_set: std::collections::BTreeSet<String> =
            gate_names.iter().map(|i| i.to_string()).collect();
        for strat in &strategies {
            let provided: std::collections::BTreeSet<String> =
                strat.values.iter().map(|(i, _)| i.to_string()).collect();
            // Report the first missing gate (if any). All-or-nothing is
            // fine here — the user fixes one and re-runs; we don't need
            // to enumerate every missing gate in one pass.
            if let Some(missing) = gate_name_set.difference(&provided).next() {
                return Err(syn::Error::new(
                    strat.name.span(),
                    format!(
                        "strategy_def!: strategy `{}` is missing gate `{}` \
                         (every strategy must provide a value for every gate)",
                        strat.name, missing
                    ),
                ));
            }
            // Report the first extra (undeclared) gate.
            if let Some(extra) = provided.difference(&gate_name_set).next() {
                let bad = strat
                    .values
                    .iter()
                    .find(|(i, _)| ident_matches(i, extra))
                    .expect("set membership: `extra` came from `provided` which is built from `strat.values`");
                return Err(syn::Error::new(
                    bad.0.span(),
                    format!(
                        "strategy_def!: strategy `{}` lists unknown gate `{}` \
                         (not declared in `gates {{ ... }}`)",
                        strat.name, extra
                    ),
                ));
            }
        }

        // Verify `default_strategy` names a real strategy.
        if !strategies.iter().any(|s| s.name == default_strategy) {
            return Err(syn::Error::new(
                default_strategy.span(),
                format!(
                    "strategy_def!: `default_strategy = {}` does not name any of the declared \
                     strategies",
                    default_strategy
                ),
            ));
        }

        Ok(Self {
            name,
            default_strategy,
            enums,
            strategies,
            gates,
        })
    }
}

fn parse_strategy_variant(input: ParseStream) -> syn::Result<StrategyVariant> {
    let attrs = input.call(Attribute::parse_outer)?;
    let name: Ident = input.parse()?;
    let content;
    braced!(content in input);
    let mut values = Vec::new();
    let assignments: Punctuated<GateAssign, Token![,]> = Punctuated::parse_terminated(&content)?;
    for GateAssign { name, expr } in assignments {
        values.push((name, expr));
    }
    Ok(StrategyVariant {
        attrs,
        name,
        values,
    })
}

struct GateAssign {
    name: Ident,
    expr: Expr,
}

impl Parse for GateAssign {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let expr: Expr = input.parse()?;
        Ok(Self { name, expr })
    }
}

fn parse_gate_def(input: ParseStream) -> syn::Result<GateDef> {
    let attrs = input.call(Attribute::parse_outer)?;
    let name: Ident = input.parse()?;
    input.parse::<Token![:]>()?;
    let ty: Type = input.parse()?;

    let content;
    braced!(content in input);

    let mut env_hook: Option<(LitStr, Path)> = None;
    let mut divergence_section: Option<LitStr> = None;
    let mut divergence_row_ref: Option<LitStr> = None;

    while !content.is_empty() {
        let key: Ident = content.parse()?;
        let key_str = key.to_string();
        content.parse::<Token![=]>()?;
        match key_str.as_str() {
            "env_hook" => {
                let env_name: LitStr = content.parse()?;
                content.parse::<Token![=>]>()?;
                let parser: Path = content.parse()?;
                env_hook = Some((env_name, parser));
            }
            "divergence_section" => {
                divergence_section = Some(content.parse()?);
            }
            "divergence_row_ref" => {
                divergence_row_ref = Some(content.parse()?);
            }
            other => {
                return Err(syn::Error::new(
                    key.span(),
                    format!(
                        "strategy_def!: unknown gate metadata key `{}` \
                         (expected one of `env_hook`, `divergence_section`, \
                          `divergence_row_ref`)",
                        other
                    ),
                ));
            }
        }
        // Comma between metadata entries; tolerated trailing too.
        let _ = content.parse::<Token![,]>();
    }

    Ok(GateDef {
        attrs,
        name,
        ty,
        env_hook,
        divergence_section,
        divergence_row_ref,
    })
}

// ──────────────────────────────────────────────────────────────────────
// Expansion
// ──────────────────────────────────────────────────────────────────────

/// Compare a `syn::Ident` against a `&str` without allocating an
/// owned `String` for the comparison. `Ident::to_string` allocates; we
/// often only need equality.
fn ident_matches(ident: &Ident, needle: &str) -> bool {
    ident == needle
}

/// Convert `CamelCase` strategy name to a `snake_case` constructor name.
/// Mirrors the hand-written convention (`Libjxl` → `libjxl`,
/// `LeanFaster` → `lean_faster`, `Aggressive` → `aggressive`).
fn snake_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i != 0 {
                out.push('_');
            }
            for low in c.to_lowercase() {
                out.push(low);
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn expand(input: StrategyDefInput) -> TokenStream2 {
    let StrategyDefInput {
        name,
        default_strategy,
        enums,
        strategies,
        gates,
    } = input;

    let custom_struct = format_ident!("{}EncoderImprovements", name);
    let resolved_struct = format_ident!("{}ResolvedImprovements", name);
    let strategy_enum = format_ident!("{}EncoderStrategy", name);
    let env_fn = format_ident!("apply_{}_env_var_fallbacks", snake_case(&name.to_string()));

    // ── field declarations (Custom struct: pub fields) ─────────────
    let custom_fields = gates.iter().map(|g| {
        let GateDef {
            attrs, name, ty, ..
        } = g;
        quote! {
            #(#attrs)*
            pub #name: #ty,
        }
    });

    // ── field declarations (Resolved struct: pub(crate) fields) ────
    let resolved_fields = gates.iter().map(|g| {
        let GateDef { name, ty, .. } = g;
        quote! {
            pub(crate) #name: #ty,
        }
    });

    // ── strategy constructors ───────────────────────────────────────
    // For each strategy, build a `fn <snake_name>() -> Self { Self { ... } }`.
    let strategy_ctors = strategies.iter().map(|s| {
        let snake = format_ident!("{}", snake_case(&s.name.to_string()));
        let doc = format!(
            "Generated by `strategy_def!` for strategy variant `{}`.\n\n\
             Set per-gate values are listed inline.",
            s.name
        );
        // Order the field initializers by the gate declaration order so
        // the generated code is deterministic and diff-friendly.
        let inits = gates.iter().map(|g| {
            let assign = s
                .values
                .iter()
                .find(|(i, _)| i == &g.name)
                .expect("checked in parser");
            let field = &g.name;
            let expr = &assign.1;
            quote! { #field: #expr, }
        });
        quote! {
            #[doc = #doc]
            pub fn #snake() -> Self {
                Self { #(#inits)* }
            }
        }
    });

    // ── from_custom ────────────────────────────────────────────────
    let from_custom_inits = gates.iter().map(|g| {
        let f = &g.name;
        quote! { #f: c.#f.clone(), }
    });

    // ── EncoderStrategy enum: one variant per strategy + Custom ────
    let strategy_variants = strategies.iter().map(|s| {
        let var_name = &s.name;
        let attrs = &s.attrs;
        let is_default = var_name == &default_strategy;
        let default_attr = if is_default {
            quote! { #[default] }
        } else {
            quote! {}
        };
        quote! {
            #(#attrs)*
            #default_attr
            #var_name,
        }
    });

    // resolve() match arms
    let resolve_arms = strategies.iter().map(|s| {
        let var_name = &s.name;
        let snake = format_ident!("{}", snake_case(&var_name.to_string()));
        quote! {
            Self::#var_name => #resolved_struct::#snake(),
        }
    });

    // ── env-var fallback layer ─────────────────────────────────────
    // For each gate with `env_hook = "NAME" => parser`, emit a block
    // that:
    //   - reads the env var via std::env::var
    //   - calls the parser; if Some(val), overwrites the resolved field
    //     ONLY when the field equals its current Default::default value.
    // Cached at resolve-time (called once per encode). Std-only.
    let env_blocks: Vec<TokenStream2> = gates
        .iter()
        .filter_map(|g| {
            let (env_name, parser) = g.env_hook.as_ref()?;
            let field = &g.name;
            let ty = &g.ty;
            Some(quote! {
                // Per-gate env-var fallback. Only applies when the
                // resolved field equals its type-level Default (so
                // explicit caller settings via `Custom` payload always
                // win over the env-var).
                if r.#field == <#ty as ::core::default::Default>::default()
                    && let Ok(__s) = ::std::env::var(#env_name)
                    && let Some(__v) = #parser(__s.as_str())
                {
                    r.#field = __v;
                }
            })
        })
        .collect();

    let env_fn_body = if env_blocks.is_empty() {
        quote! { let _ = r; }
    } else {
        quote! { #(#env_blocks)* }
    };

    // ── Default impl for Custom struct ─────────────────────────────
    // Use the user-declared `default_strategy` to populate defaults.
    let default_strategy_snake = format_ident!("{}", snake_case(&default_strategy.to_string()));
    let default_inits = {
        let strat = strategies
            .iter()
            .find(|s| s.name == default_strategy)
            .expect("checked in parser");
        gates.iter().map(|g| {
            let f = &g.name;
            let assign = strat
                .values
                .iter()
                .find(|(i, _)| i == &g.name)
                .expect("checked in parser");
            let expr = &assign.1;
            quote! { #f: #expr, }
        })
    };

    // ── Local enum re-emits (so consumers don't have to declare twice) ──
    let enum_emits = enums.iter().map(|e| quote! { #e });

    // ── Divergence metadata (kept as compile-time consts for W44-194 build-script consumer) ──
    let divergence_consts = gates.iter().filter_map(|g| {
        let section = g.divergence_section.as_ref()?;
        let row_ref = g.divergence_row_ref.as_ref();
        let const_name = format_ident!(
            "__{}_DIVERGENCE_{}",
            name.to_string().to_uppercase(),
            g.name.to_string().to_uppercase()
        );
        let row_ref_str = match row_ref {
            Some(r) => r.value(),
            None => String::new(),
        };
        let combined = format!("section={} ; row_ref={}", section.value(), row_ref_str);
        Some(quote! {
            #[doc(hidden)]
            #[allow(non_upper_case_globals, dead_code)]
            pub const #const_name: &str = #combined;
        })
    });

    quote! {
        // Local enums (re-emitted verbatim).
        #(#enum_emits)*

        // Public custom-improvements struct.
        #[derive(Clone, Debug, PartialEq)]
        pub struct #custom_struct {
            #(#custom_fields)*
        }

        impl ::core::default::Default for #custom_struct {
            fn default() -> Self {
                Self {
                    #(#default_inits)*
                }
            }
        }

        // Crate-internal resolved-improvements struct.
        #[allow(dead_code)]
        #[derive(Clone, Debug, PartialEq)]
        pub(crate) struct #resolved_struct {
            #(#resolved_fields)*
        }

        impl ::core::default::Default for #resolved_struct {
            fn default() -> Self {
                #resolved_struct::#default_strategy_snake()
            }
        }

        impl #resolved_struct {
            #(#strategy_ctors)*

            /// Copy every field from the Custom struct.
            pub fn from_custom(c: &#custom_struct) -> Self {
                Self {
                    #(#from_custom_inits)*
                }
            }
        }

        // Strategy enum.
        #[derive(Clone, Debug, Default, PartialEq)]
        pub enum #strategy_enum {
            #(#strategy_variants)*
            /// Caller-defined per-field overrides. Always reachable.
            Custom(::std::boxed::Box<#custom_struct>),
        }

        impl #strategy_enum {
            /// Resolve the strategy enum to a `ResolvedImprovements`
            /// struct. Env-var fallbacks are applied at the end (only
            /// for gates whose resolved value equals its type-level
            /// `Default::default()`).
            pub fn resolve(&self) -> #resolved_struct {
                let mut __resolved = match self {
                    #(#resolve_arms)*
                    Self::Custom(c) => #resolved_struct::from_custom(c.as_ref()),
                };
                #env_fn(&mut __resolved);
                __resolved
            }
        }

        /// Per-gate env-var fallback applied by `EncoderStrategy::resolve`.
        ///
        /// Std-only (the macro currently emits an unconditional
        /// `std::env::var` call — wrap in `#[cfg(feature = "std")]` at
        /// the call site if a `no_std` build is supported).
        #[allow(dead_code)]
        fn #env_fn(r: &mut #resolved_struct) {
            #env_fn_body
        }

        // Divergence-table metadata consts (consumed by the W44-194
        // build-script that auto-generates docs/LIBJXL_DIVERGENCES.md).
        #(#divergence_consts)*
    }
}

// ──────────────────────────────────────────────────────────────────────
// Public proc-macro entry
// ──────────────────────────────────────────────────────────────────────

/// Declarative gate registry for an encoder strategy bundle.
///
/// See the [crate-level docs](crate) for the syntax + example output.
///
/// # Errors
///
/// Emits `compile_error!` on:
/// - Missing top-level keys (`name`, `default_strategy`, `strategies`,
///   `gates`).
/// - Unknown top-level key.
/// - A strategy declaration that omits a declared gate, or lists an
///   unknown one.
/// - `default_strategy = ...` not matching any declared strategy.
/// - Unknown gate-metadata key.
///
/// All error messages carry a `Span` pointing at the offending token so
/// IDE diagnostics highlight the right location.
#[proc_macro]
pub fn strategy_def(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as StrategyDefInput);
    expand(parsed).into()
}
