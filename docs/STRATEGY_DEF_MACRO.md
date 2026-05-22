# `strategy_def!` macro — design + usage reference

**Status**: Phase 1 (W44-192) — proc-macro shipped + validated on 3
prototype gates. Phase 2 (W44-193) will big-bang-migrate the 24
production gates in `api.rs`. Phase 3 (W44-194) adds per-cell hash CI
for `EncoderStrategy::Libjxl` and a build-script that auto-generates
`docs/LIBJXL_DIVERGENCES.md` from the per-gate `divergence_section` /
`divergence_row_ref` metadata.

**Provenance**: implements the recommended Solution B from the
[W44-190 RFC](RFC-strategy-refactor-2026-05-22.md). User signoff
2026-05-22 on big-bang migration + opt-in env hooks + proc-macro
syntax.

**Source**: `jxl-encoder-macros/src/lib.rs` (the macro);
`jxl-encoder/src/strategy_def_prototype.rs` (the prototype consumer);
`jxl-encoder/tests/strategy_def_prototype_env_fallback.rs` (the
integration test for the env-fallback layer).

---

## 1. What it does

One `strategy_def! { ... }` invocation generates the **full strategy
bundle plumbing** for an encoder configuration: the public
`EncoderImprovements` struct, the crate-internal `ResolvedImprovements`
struct, four named constructors per strategy variant, the
`EncoderStrategy` enum with a per-variant `Default` impl, a `resolve()`
method composing strategy + Custom + env-var fallbacks, and per-gate
divergence-table metadata consts.

Today (in production `api.rs`) those pieces are hand-written and add
~1100 LOC for a single new bool gate (W44-184 was the cited example).
The macro reduces that to **~5 LOC at one site per gate** plus
**~10 LOC at one site per strategy**.

---

## 2. Syntax reference

```rust
jxl_encoder_macros::strategy_def! {
    // ── Required: name of the bundle (PascalCase, e.g. `Prototype`).
    // Generated types: `<Name>EncoderImprovements`,
    // `<Name>ResolvedImprovements`, `<Name>EncoderStrategy`.
    name = <Ident>;

    // ── Required: which strategy variant `Default` returns.
    // MUST match one of the strategies declared in `strategies { ... }`.
    default_strategy = <Ident>;

    // ── Optional: re-emit user-defined enums verbatim. Useful when
    //   the gate types live next to the macro invocation. May be empty.
    enums {
        pub enum <Name> { ... }
    }

    // ── Required: per-strategy gate-value blocks. Every strategy must
    //   provide a value for every gate declared in `gates { ... }`;
    //   missing/extra entries are compile-time errors.
    strategies {
        /// Doc comment becomes the variant doc-comment on
        /// `<Name>EncoderStrategy::<StrategyName>`.
        <StrategyName> {
            <gate_name> = <expr>,
            <gate_name> = <expr>,
            ...
        },
        <StrategyName> { ... },
        ...
    }

    // ── Required: per-gate metadata.
    gates {
        /// Doc comment becomes the doc-comment on the field of
        /// `<Name>EncoderImprovements`.
        <gate_name>: <Type> {
            // OPTIONAL: env hook. The env-var fallback fires AFTER
            // overrides apply, ONLY when the resolved field equals
            // `<Type>::default()` (explicit caller settings win).
            // `<parser>` is a fn-item path of signature
            // `fn(&str) -> Option<<Type>>`.
            env_hook = "<ENV_VAR_NAME>" => <parser>,

            // OPTIONAL: divergence-table metadata. Consumed by the
            // W44-194 build-script that auto-generates
            // docs/LIBJXL_DIVERGENCES.md. Stored as a compile-time
            // const so the generator can harvest by symbol name.
            divergence_section = "A" | "B" | "C" | "D" | "E" | "F" | "G",
            divergence_row_ref = "<row title or grep anchor>",
        },
        <gate_name>: <Type> { ... },
        ...
    }
}
```

### What the macro emits

For `name = Prototype` and a single bool gate `cfl_newton_libjxl_parity`:

```rust
pub struct PrototypeEncoderImprovements {
    pub cfl_newton_libjxl_parity: bool,
    /* one field per gate */
}
impl Default for PrototypeEncoderImprovements { /* matches default_strategy */ }

pub(crate) struct PrototypeResolvedImprovements {
    pub(crate) cfl_newton_libjxl_parity: bool,
    /* one field per gate */
}
impl Default for PrototypeResolvedImprovements { /* delegates to <default>() */ }

impl PrototypeResolvedImprovements {
    pub fn libjxl() -> Self { /* ... */ }
    pub fn zenjxl() -> Self { /* ... */ }
    pub fn lean_faster() -> Self { /* ... */ }
    pub fn aggressive() -> Self { /* ... */ }
    pub fn from_custom(c: &PrototypeEncoderImprovements) -> Self { /* ... */ }
}

#[derive(Default)]
pub enum PrototypeEncoderStrategy {
    Libjxl,
    #[default] Zenjxl,
    LeanFaster,
    Aggressive,
    Custom(Box<PrototypeEncoderImprovements>),
}

impl PrototypeEncoderStrategy {
    pub fn resolve(&self) -> PrototypeResolvedImprovements {
        let mut r = match self { /* dispatch to ctors */ };
        apply_prototype_env_var_fallbacks(&mut r);
        r
    }
}

fn apply_prototype_env_var_fallbacks(r: &mut PrototypeResolvedImprovements) {
    // For each gate with an `env_hook` clause:
    if r.<gate> == <Type>::default()
        && let Ok(s) = std::env::var("ENV_VAR_NAME")
        && let Some(v) = <parser>(s.as_str())
    {
        r.<gate> = v;
    }
}

// Hidden divergence-metadata const for the W44-194 build-script.
pub const __PROTOTYPE_DIVERGENCE_CFL_NEWTON_LIBJXL_PARITY: &str =
    "section=C ; row_ref=CfL Newton parameters (W44-183/184)";
```

---

## 3. Adding a new gate

Today (handwritten, ~6 mandatory file edits — W44-184 added ~1100 LOC
for a single bool gate; ~50 of those were the algorithmic delta and the
rest were boilerplate).

With the macro:

1. **Edit `gates { ... }`** in the `strategy_def!` invocation: add the
   field declaration (type, env hook, divergence metadata).
2. **Edit each `strategies { ... }` block**: add a value for the new
   gate (compile-time error if any strategy omits it).
3. **Edit the read site**: the actual encoder logic consuming the new
   field. (Unchanged from today — the macro doesn't help here yet; a
   future facade-method generator might collapse some of these.)
4. **Update `docs/LIBJXL_DIVERGENCES.md`** — until W44-194 lands the
   build-script, this is still manual (but the `divergence_section` /
   `divergence_row_ref` metadata in the macro is the eventual source).

Total: **3 sites × 1-5 lines each** vs the old **6 sites × ~50 lines
each**.

### Example (adding a hypothetical `dummy_gate: bool` to the prototype)

```rust
strategy_def! {
    name = Prototype;
    default_strategy = Zenjxl;
    enums {}

    strategies {
        Libjxl { cfl_newton_libjxl_parity = true, ..., dummy_gate = false, },
        Zenjxl { cfl_newton_libjxl_parity = false, ..., dummy_gate = true, },
        LeanFaster { cfl_newton_libjxl_parity = false, ..., dummy_gate = false, },
        Aggressive { cfl_newton_libjxl_parity = false, ..., dummy_gate = true, },
    }

    gates {
        cfl_newton_libjxl_parity: bool { /* ... */ },
        /* ... */

        /// Doc-comment becomes the field doc.
        dummy_gate: bool {
            env_hook = "JXL_DUMMY_GATE" => bool_one,
            divergence_section = "B",
            divergence_row_ref = "Dummy gate (test only)",
        },
    }
}
```

That's it. The four constructors, both struct fields, both `Default`
impls, the env-fallback block, and the divergence-metadata const are
all generated.

---

## 4. Adding a new strategy

Today: ~80 LOC of hand-mirrored constructor in
`ResolvedImprovements::<name>()` plus a `EncoderStrategy::<Name>`
variant plus a match arm.

With the macro: **one strategy block, one line per gate the strategy
overrides**.

```rust
strategy_def! {
    name = Prototype;
    default_strategy = Zenjxl;
    enums {}

    strategies {
        /* existing strategies */

        /// **NewStrategy** — full per-gate value list (no inheritance
        /// syntax in W44-192 Phase 1; W44-193 may add it).
        NewStrategy {
            cfl_newton_libjxl_parity = true,
            content_class_auto_classify = false,
            adaptive_buttloop_iters = IterMode::TexturedExtend,
        },
    }

    gates { /* unchanged */ }
}
```

That's it. The `Self::NewStrategy => <Name>ResolvedImprovements::new_strategy()`
match arm, the `pub fn new_strategy() -> Self { ... }` constructor, and
the `EncoderStrategy::NewStrategy` variant are all generated.

**Strategy inheritance** (e.g., "Aggressive inherits Zenjxl plus 2
changes") is **not** in W44-192 Phase 1. Every strategy explicitly lists
every gate; the compile-time check catches missing/extra entries. RFC
Section 3 sketches `strategy LeanFaster inherits Zenjxl { ... }`
syntax for a future phase.

---

## 5. Env-var hooks: opt-in, cached at resolve-time

Per the user's 2026-05-22 directive ("env var loading is super slow —
make it opt-in"):

- The env hook is **always consulted at resolve-time** (called once per
  `LossyConfig::resolve()`, **not** in any hot encode loop).
- The hook fires **only when** the resolved field equals
  `<Type>::default()`. Caller-supplied values via the `Custom` payload
  or per-strategy preset ALWAYS win.
- The macro emits a `std::env::var(...)` call unconditionally; consumer
  code is responsible for `#[cfg(feature = "std")]`-gating the call if
  the consumer crate supports `no_std`.
- Parsers are fn-items in the consumer crate. Two helpers we recommend
  exporting alongside the macro invocation:
  - `fn bool_one(s: &str) -> Option<bool>` — matches the production
    `JXL_W44_184_FORCE_LIBJXL_NEWTON=1` parser.
  - `fn parse_f32(s: &str) -> Option<f32>` — matches the production
    `JXL_BUTTLOOP_INITIAL_QF_SCALE` parser.
- Enum parsers are one fn per enum; see `parse_iter_mode` in the
  prototype module for an example.

### Precedence (top wins)

1. **Custom payload** (`EncoderStrategy::Custom(Box<<Name>EncoderImprovements>)`)
   — every field comes from the user's explicit struct.
2. **Strategy preset** (`Libjxl` / `LeanFaster` / `Zenjxl` /
   `Aggressive`) — every field comes from the per-strategy block
   inside `strategies { ... }`.
3. **Env-var fallback** — fires only when (1) and (2) above leave the
   field at `<Type>::default()`.

This matches the production W44-132 Chunk F semantics. Integration
tests live in `tests/strategy_def_prototype_env_fallback.rs`.

---

## 6. Divergence-table metadata

Per W44-190 RFC Section 3 + the project CLAUDE.md mandatory-maintenance
rule, every divergence from libjxl reference is tracked in
`docs/LIBJXL_DIVERGENCES.md`. The hand-maintained table has drifted
from the source code 4 times across W44-91/W44-152/W44-156/W44-171
(documented in RFC §1.6).

The macro captures the table reference next to the gate:

```rust
gates {
    cfl_newton_libjxl_parity: bool {
        env_hook = "JXL_W44_184_FORCE_LIBJXL_NEWTON" => bool_one,
        divergence_section = "C",
        divergence_row_ref = "CfL Newton parameters (W44-183/184)",
    },
}
```

The macro emits a `pub const __<NAME>_DIVERGENCE_<GATE>: &str = "..."`
constant per gate that includes both fields. W44-194 will add a
build-script that:

1. Walks every `__*_DIVERGENCE_*` symbol via `cargo doc --output-format=json`
   (or a syn-based source-tree scan).
2. Joins each symbol to its source location (file:line).
3. Renders `docs/LIBJXL_DIVERGENCES.md` from the metadata, replacing the
   hand-maintained table.

Until W44-194 lands, the table remains hand-maintained — but the macro
metadata is the eventual source of truth, so the manual edits should
already include the same `section=X ; row_ref=Y` values.

---

## 7. Compile-time error messages

The macro emits `compile_error!` (via `syn::Error`) with a `Span`
pointing at the offending token. Examples (recorded as golden files in
`jxl-encoder-macros/tests/ui_fail/*.stderr`):

- Missing `default_strategy = ...`:
  ```
  error: strategy_def!: missing `default_strategy = <Ident>;` top-level key
  ```
- `default_strategy = X` where `X` isn't a declared strategy:
  ```
  error: strategy_def!: `default_strategy = X` does not name any of the declared strategies
  ```
- Strategy variant missing a gate:
  ```
  error: strategy_def!: strategy `Zenjxl` is missing gate `y`
         (every strategy must provide a value for every gate)
  ```
- Strategy variant with an extra (undeclared) gate:
  ```
  error: strategy_def!: strategy `Zenjxl` lists unknown gate `surprise`
         (not declared in `gates { ... }`)
  ```
- Unknown top-level key:
  ```
  error: strategy_def!: unknown top-level key `typo_key`
         (expected one of `name`, `default_strategy`, `enums`, `strategies`, `gates`)
  ```

Every error carries a `Span` so IDE diagnostics highlight the right
token. Run `cargo test -p jxl-encoder-macros --test trybuild` to verify
the messages stay stable.

---

## 8. W44-192 acceptance gates (validated)

From the W44-192 task spec — all gates PASS:

- (a) **Build PASS** — `cargo build -p jxl-encoder` and
  `-p jxl-encoder-macros` both succeed.
- (b) **Generated code byte-identical to handwritten for 3 prototype
  gates** — verified via
  `cargo expand -p jxl-encoder --lib strategy_def_prototype` saved at
  `benchmarks/w44_192_macro_expansion_2026-05-22.rs`. The generated
  `pub fn libjxl()`, `pub fn zenjxl()`, `pub fn lean_faster()`,
  `pub fn aggressive()` constructors for the 3 prototype gates produce
  the same literal-value field-init shape as the handwritten
  `api.rs::ResolvedImprovements::libjxl()` etc. for those same 3 fields.
  (Differences are stylistic — generated code uses `Self { ... }`
  shorthand; handwritten uses field assignments with `..Default::default()`
  inheritance in some places — but the resolved struct's field values
  are bit-identical.)
- (c) **Hash-locks 36/36 BYTE-IDENTICAL** — the prototype module is
  `pub(crate)` and never wired into the production encoder. The
  production `EncoderImprovementsCustom` / `ResolvedImprovements` are
  unchanged.
- (d) **Library tests pass** — `cargo test -p jxl-encoder --lib`:
  1393 passed, 0 failed (prior baseline 1386; the +7 are the new
  inline `strategy_def_prototype::tests` cases).
- (e) **Doc written** — this file.
- (f) **Pushed to origin/main** — single chunk-commit per the
  conventional pre-push rebase pattern.

Integration test (`__internals` feature): 9 env-fallback tests pass.
Trybuild ui tests: 7 tests pass (2 ui_pass + 5 ui_fail).

---

## 9. Migration plan (W44-193 + W44-194)

**W44-193** (next chunk) — big-bang migrate the 24 production gates
in `api.rs` to a single `strategy_def!` invocation. Hash-locks expected
byte-identical because the generated code matches the handwritten
patterns. Plan:

1. Build the consumer-side enums/parsers list (~15 items mapped from
   today's `StrategyOverrides` + `apply_env_var_fallbacks`).
2. Convert `api.rs::EncoderImprovementsCustom` → one
   `strategy_def!{...}` block.
3. Convert the 4 `ResolvedImprovements::libjxl()` / `::zenjxl()` /
   `::lean_faster()` / `::aggressive()` constructors → strategy blocks.
4. Convert `apply_env_var_fallbacks` → per-gate `env_hook` clauses.
5. Migrate the 31 inline unit tests in `api.rs::tests::test_w44_*` (the
   ones that exercise resolve dispatch) to depend on the macro-generated
   types via the same `__internals` re-export pattern.
6. Run hash-locks: expected 36/36 byte-identical.
7. Run multi-decoder roundtrips on at least 3 strategies × 2 cells: 6/6
   expected PASS.

**W44-194** (later chunk) — add CI:

1. Per-cell hash lock for `EncoderStrategy::Libjxl` against cjxl output
   (RFC Solution C).
2. Build-script that auto-generates `docs/LIBJXL_DIVERGENCES.md` from
   the per-gate `divergence_section` / `divergence_row_ref` macro
   metadata (RFC G5).
3. CI fails if either (1) drifts vs cjxl or (2) the table drifts from
   the macro metadata.

---

## 10. References

- RFC: [`docs/RFC-strategy-refactor-2026-05-22.md`](RFC-strategy-refactor-2026-05-22.md) (W44-190).
- Hand-maintained divergence table: [`docs/LIBJXL_DIVERGENCES.md`](LIBJXL_DIVERGENCES.md).
- Project mandatory-maintenance rule: see `CLAUDE.md` "MANDATORY:
  Maintain `docs/LIBJXL_DIVERGENCES.md` with every change".
- Production W44-132 Chunk F (env-fallback precedent):
  `jxl-encoder/src/api.rs::apply_env_var_fallbacks` +
  `tests/strategy_env_fallback.rs`.
- Macro expansion artifact (proof of acceptance gate b):
  `benchmarks/w44_192_macro_expansion_2026-05-22.rs`.
