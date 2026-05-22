# RFC: strategy-and-divergence refactor for rapid experimentation

**Status**: DRAFT — author W44-190, requested by user 2026-05-21 directive
"consider what refactor makes it easiest to rapidly iterate and experiment
different strategies while keeping some sets locked in like libjxl-matching"

**Date**: 2026-05-22
**Companion docs**: `docs/COMPATIBILITY_MODES.md` (the W44-125→133
EncoderStrategy v2 design), `docs/LIBJXL_DIVERGENCES.md` (the
authoritative divergence inventory), `CLAUDE.md` (mandatory
divergence-table maintenance rule)

---

## Summary

The W44-125→133 arc shipped `EncoderStrategy{Libjxl, LeanFaster, Zenjxl,
Aggressive, Custom}` and laid a clean foundation. The W44-164→184 follow-on
filled it with ~10 additional gates. The pattern works but the per-gate
cost is high: each new gate today touches ~6 files, adds ~50 boilerplate
lines (struct fields, default impls, four constructors, divergence-table
row, often a test), and creates drift potential between code and docs.

This RFC proposes a refactor that:
1. Moves the strategy bundle definitions into a **single declarative
   source** (a Rust-source `strategy!` macro file, NOT toml — see §5 Q1
   answer).
2. Replaces the four hand-written `ResolvedImprovements` constructors
   (`libjxl`, `lean_faster`, `zenjxl`, `aggressive`) with macro-generated
   functions that read from the declarative source.
3. Introduces a **gate registry macro** that emits the field declaration,
   the default value, the four-constructor entries, and a snippet of the
   divergence-table row — keeping code and docs in lockstep.
4. Adds **byte-parity CI** for `EncoderStrategy::Libjxl` against `cjxl`
   (currently we only have hash-locks against ourselves, which detect
   our-vs-ourselves drift but not vs-cjxl drift).
5. Migrates phased: declarative + macro layer lands additively in Phase 1
   alongside the existing system; gates migrate one-by-one in Phase 2;
   Phase 3 deletes the manual constructors; Phase 4 lights up byte-parity
   CI.

The refactor is doc-only at this stage (no source change). Acceptance
gate is user signoff on the Section 5 open questions before
implementation chunks start landing.

---

## Section 1: Pain points (audit)

### 1.1 Field count and surface area

As of `b8517c09a6d5` (W44-184) the `EncoderImprovementsCustom` struct
carries **24 `pub` fields**:

```
ScreenshotEntropyMulPolicy / HighDPhotoEntropyMulPolicy / Dct64SearchPolicy /
Dct32SearchPolicy / SmoothPhotoDct64Policy / ButtloopQfSeedPolicy /
AdaptiveQuantQfSeedPolicy / EpfSharpnessSeed / EpfDispatch /
PixelLossDispatch / SinglePassEntropyDispatch / PatchesDispatch /
EffortGate × 3 (cfl_two_pass / try_dct64 / epf_dynamic_sharpness) /
block_ctx_map_15_cluster / content_class_auto_classify /
photo_epf_seed_admit / photo_variant_z_admit / find_best_32_per_m3_lift /
adaptive_buttloop_iters / adaptive_buttloop_iters_narrow /
terminal_class_exclude / cfl_newton_libjxl_parity
```

`ResolvedImprovements` mirrors these one-for-one as `pub(crate)` fields.
Adding a new field today means editing both structs, all four constructors
(`libjxl()` / `lean_faster()` / `zenjxl()` / `aggressive()`), the
`from_custom()` copy function, plus the `Default` impls on both — **6
spots × 1 line each = 6 mandatory edits before any code reads the field**.
The W44-184 commit added `cfl_newton_libjxl_parity` and touched +153 lines
in `api.rs` alone, much of it boilerplate. The W44-166 commit (variant Z
admission) added +106 lines in `api.rs` for similar boilerplate before any
encoder logic.

### 1.2 Call-site read count

`grep "resolved\." jxl-encoder/src/` finds 87 hits across 5 files
(`api.rs`, `effort.rs`, `vardct/bitstream.rs`, `vardct/butteraugli_loop.rs`,
`vardct/encoder.rs`). The call sites read individual fields directly, which
is fine for readability but means a field-rename refactor would touch up to
87 spots. There is no facade method per gate (e.g., no
`resolved.is_w44_117_seed_active()`); the gate predicate is reconstructed at
each call site (e.g., `resolved.adaptive_buttloop_iters_narrow &&
target_distance >= 4.0 && target_distance <= 5.0 && effort >= 8 && ...`).
The same predicate is sometimes duplicated across two call sites (e.g., the
W44-176 terminal-class discriminator appears at both the eval point and the
documentation comment in `api.rs:4426`).

### 1.3 Effort-gate sites

`grep "if effort >= "` finds **20 sites** in jxl-encoder. Some are
historic constants (Section A), some are content-specific gates (W44-29
`HIGH_D_PHOTO_MIN_DISTANCE`), some compose into multi-clause predicates.
The `EffortGate` enum from W44-127 Chunk G models the Section A gates
cleanly but only 3 of the 20 effort-gate sites have been migrated. The
remaining 17 are inline `if effort >= N` literals scattered across `effort.rs`
and the vardct module.

### 1.4 Discriminator chains in vardct/encoder.rs

`grep "W44_[0-9]\+\|w44_[0-9]\+"` in `vardct/encoder.rs` finds 483 hits;
`vardct/butteraugli_loop.rs` has 192. These are a mix of constants,
predicates, comments, and dispatch sites. There are at least **14 named
W44-NN discriminator chains** active in production (W44-29, W44-65, W44-68,
W44-91, W44-96, W44-98, W44-99, W44-100, W44-117, W44-118, W44-120, W44-124,
W44-135/143, W44-176). They compose nested (W44-91 ⊂ W44-29; W44-98 ⊂
W44-96; W44-99 ⊂ W44-96; W44-124 ⊂ W44-65). The composition rules are
embedded in the call-site control flow (`if W44-96-fires { if W44-98-fires
{ HC-table } else if W44-99-fires { LC-table } else { Z-table } }`) and
there is no central place to query "what discriminators fire on this image
at this distance?".

### 1.5 Divergence-table row count

`docs/LIBJXL_DIVERGENCES.md` currently has **170 table rows** across
9 sections:
- A. Effort-gate divergences: 7 rows
- B. Content-aware discriminators: 17 rows
- C. Cost-model constants: 45 rows
- D. Algorithm choices: 18 rows
- E. Per-API behavior (opt-in): 32 rows
- F. KNOWN-BUG clusters: 15 rows
- G. RESOLVED (historical): 41 rows
- H. Maintenance checklist
- I. References

Adding a gate per the CLAUDE.md mandatory-maintenance rule means
manually adding one row with site, ours-value, libjxl-value, status,
commit SHA. The rule is enforced by social convention, not code; we
verify compliance via `git log --oneline -- docs/LIBJXL_DIVERGENCES.md`.
There is **no machine-checked link** between a row and the source-code
constant that implements it.

### 1.6 Drift potential (table vs source)

Observed historical drift incidents:
- **W44-152 row** (Section B): claims "narrows the distance band to
  [3.0, 5.0]" — the source constant is
  `W44_152_W44_151_MAX_DISTANCE = 5.0` but the field name carries the
  legacy W44-151 designation. Easy to miss when reading the row.
- **W44-156 row**: documents `W44_156_VARIANT_Z_D_HIGH_THRESHOLD = 5.5`
  as the high-distance split — that constant lives in `vardct/encoder.rs`
  but the row points readers to `vardct/encoder.rs` generically without
  a line number. A `grep` is required to find the actual definition.
- **W44-171 row**: changed `tree_kind = kLearn` gate from `effort >= 4`
  → `effort >= 8`. The row was updated in the same commit as the source.
  Per W44-172, the previous (pre-W44-171) row had cited
  `enc_modular.cc:1166` as parity — **that was a misread** of libjxl
  source. The row was wrong for 8 months before W44-171 caught it.
- The MEMORY.md "process_no_premature_wip_takeover_2026-05-20" note
  documents 3 incidents in one week where the divergence table was out
  of sync with main during concurrent chunk work.

These are not catastrophic — the table is correct most of the time —
but each drift incident costs an investigator hours to disambiguate
"is this a real divergence or a stale row?".

### 1.7 Env-var hooks have no API

There are **27 distinct `JXL_W44_*` / `JXL_BUTTLOOP_*` env-vars** read at
59 sites across the source tree. Of those:
- 4 were promoted to first-class fields by W44-132 Chunk F
  (`JXL_W44_117_DISABLE`, `JXL_W44_120_EPF_SEED_MIN_DISTANCE`,
  `JXL_BUTTLOOP_INITIAL_QF_SCALE`, `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE`)
- 23 remain env-only (`JXL_W44_166_VARIANT_Z_ADMIT_MODE`,
  `JXL_W44_167_MODE`, `JXL_W44_168_MODE`, `JXL_W44_142_*`,
  `JXL_W44_143_MIN_DISTANCE`, etc.)

The env-only knobs are typically used by harness sweeps but their
existence is undocumented in the public API. A user reading
`EncoderImprovementsCustom` cannot tell which gates have env-var hooks
without grepping the source.

### 1.8 Boilerplate cost per new gate

Concrete count from W44-184 (`cfl_newton_libjxl_parity`, the freshest
example) — the gate added a single `bool` to `ResolvedImprovements`.
Lines added per file:
- `api.rs`: +153 lines (struct field, doc comment, all 4 constructor
  entries, default value, override field, env-var fallback path, 3 unit
  tests)
- `effort.rs`: +122 lines (the `apply_section_c_cfl_newton_libjxl_parity`
  helper, profile field, integration into
  `effective_profile_for_image_with_smoothness`, 3 unit tests)
- `vardct/encoder.rs`: +15 lines (read site)
- `vardct/chroma_from_luma.rs`: +31 lines (3 call sites where the value is
  threaded through to the SIMD layer)
- `vardct/bitstream.rs`: +11 lines
- `jxl-encoder-simd/src/cfl.rs`: +264 lines (libjxl-parity constants and
  the actual Newton parameter switch)
- `docs/LIBJXL_DIVERGENCES.md`: ~10 lines (Section C row revision)

Total: 17 files touched, +1137 / -71 lines, **most of it boilerplate**.
The actual algorithmic delta is the ~50 lines of Newton-parameter
constants and one if-branch in the SIMD layer. The remaining ~1000
lines are plumbing.

W44-166 (variant Z admission) was similar: 8 files, +1279 / -7 lines,
the actual algorithmic delta was ~30 lines in `vardct/encoder.rs` and the
remainder was plumbing + a 502-line A/B-bench example + a 221-line
roundtrip test.

### 1.9 Test + hash-lock churn

Hash-lock fixtures live in `jxl-encoder/tests/hash_lock_expected.txt`
(36 fixtures, 13 lossless + 23 lossy). When a gate ships:
- If the gate fires on synthetic fixtures (rare — most fixtures are
  ≤ 48×48 px, below the `CONTENT_CLASS_MIN_PIXELS = 65,536` gate), the
  expected hashes need regen via `just update-hashes`. This is OK because
  the new hashes are deterministic.
- If the gate is Zenjxl-only (per the established pattern), the
  Libjxl-strategy hash-locks at `tests/strategy_libjxl_hash_locks.rs`
  must also be re-checked.
- Multi-decoder roundtrip tests are added per-gate (W44-164, W44-166,
  W44-176 each have dedicated `tests/w44_NNN_decoder_roundtrip.rs` files
  — 200-500 lines each).

### 1.10 Summary: what hurts

1. **6 mandatory edits before any encoder code runs** for a single bool gate.
2. **No link between divergence-table row and source-code constant**.
3. **Discriminator predicates duplicated in code, comments, and the table**.
4. **Strategy bundle constructors are 80+ lines each, hand-mirrored × 4**.
5. **Env-only hooks have no API surface** — invisible to users.
6. **No cjxl byte-parity CI** for `EncoderStrategy::Libjxl` — we trust the
   `_libjxl_hash_locks.rs` file but only against ourselves, not vs cjxl.

---

## Section 2: Design goals

(G1) **Adding a new gate ≤ 5 LOC at one site**, including the
divergence-table row. The boilerplate (field declaration, default, four
constructors, table row) must be macro-generated.

(G2) **Adding a new strategy ≤ 10 LOC at one site** (one declarative
block listing per-field overrides relative to a base strategy). Strategy
inheritance ("Aggressive inherits Zenjxl plus these 2 changes") should
not require copy-pasting an 80-line constructor.

(G3) **`EncoderStrategy::Libjxl` is locked-in cjxl byte-parity**, with
CI that catches drift on a fixed N-cell corpus. The CI baseline is
generated by running cjxl on the same inputs and hashing both outputs.
Any commit that breaks Libjxl strategy = CI fail.

(G4) **`EncoderStrategy::Custom` is easy to compose for experimentation**:
a caller can write `EncoderStrategy::experimental("zenjxl + dct64_search:
ForceSuppress")` in a one-liner without manually instantiating every field.

(G5) **The divergence table is auto-generated from code annotations** on
each gate. No drift possible because the table IS the source. A `cargo
doc` extension or a build-script post-pass produces
`docs/LIBJXL_DIVERGENCES.md` from `#[divergence(...)]` annotations.

(G6) **Backwards-compatible incremental migration** (no big-bang). The
declarative layer ships alongside the existing system in Phase 1;
gates migrate one-by-one in Phase 2; the manual constructors get
deleted in Phase 3 after every gate has migrated.

(G7) **Discriminator chains are explicit data structures** that can be
inspected, tested, and walked. A test harness should be able to feed a
synthetic image proxy and assert "discriminator X fires" without
running a full encode.

---

## Section 3: Proposed solutions (ranked)

Four candidate solutions, ranked by EV-vs-cost. Solution B is the
recommended winner; A is a sub-component; C is independent and
unconditionally recommended; D, E, F are deferred.

### Recommended (ship together as Phase 1+2+4):

**Solution B (gate registry macro)** + **Solution C (Libjxl-parity CI)**.

These two solve goals G1, G3, G5, G6 directly. Solutions A and E are
nice-to-have follow-ons; D and F are speculative.

---

### A. Declarative strategy source (Rust-source `strategy!` macro file)

**Shape**: a single file `jxl-encoder/src/strategies.rs` with a macro
invocation per strategy:

```rust
use crate::api::*;

strategies! {
    /// Strict libjxl-parity bundle. See LIBJXL_DIVERGENCES.md Section G.
    strategy Libjxl {
        // Section B (content-aware gates): all disabled
        screenshot_entropy_mul = ScreenshotEntropyMulPolicy::Disabled,
        high_d_photo_entropy_mul = HighDPhotoEntropyMulPolicy::Disabled,
        dct64_search_policy = Dct64SearchPolicy::ForceAllow,
        // ... full list ...

        // Section A effort gates: libjxl thresholds
        cfl_two_pass_min_effort = EffortGate::Libjxl,
        try_dct64_min_effort = EffortGate::Libjxl,
        epf_dynamic_sharpness_min_effort = EffortGate::Libjxl,

        // Section D KNOWN-BUG: deliberately re-enable
        block_ctx_map_15_cluster = true,

        // Section C CfL Newton: bit-exact libjxl params
        cfl_newton_libjxl_parity = true,
    }

    /// Production default. Inherits Zenjxl from Default.
    strategy Zenjxl inherits Default {
        // Per-field overrides (none needed — Zenjxl == every type's default).
    }

    /// Drops heavy per-image content gates.
    strategy LeanFaster inherits Zenjxl {
        screenshot_entropy_mul = ScreenshotEntropyMulPolicy::Disabled,
        high_d_photo_entropy_mul = HighDPhotoEntropyMulPolicy::Auto,
        dct64_search_policy = Dct64SearchPolicy::ForceAllow,
        // ... drop list ...
        content_class_auto_classify = false,
        photo_epf_seed_admit = false,
        photo_variant_z_admit = false,
        // ... etc
    }
}
```

The `strategies!` macro generates the existing `fn libjxl()`, `fn zenjxl()`,
etc. functions on `ResolvedImprovements`. No callers change.

**Pros**:
- One file to read to see all strategy presets side-by-side.
- Per-strategy diffs become obvious (today they're spread across
  four 60-80 line constructors that are 95% copy-paste).
- Inheritance avoids the `..Default::default()` trick used in
  `lean_faster()` today.

**Cons**:
- The macro must understand field types (enum vs bool) for the inheritance
  diff to work cleanly.
- Adds a layer of indirection. Stepping through `EncoderStrategy::resolve`
  in a debugger goes through generated code.

**Why a Rust-source macro file, NOT toml/yaml**: per project conventions
(§5 Q1 below), keeping the source-of-truth in Rust avoids parsing a
config file at build time, keeps type-checking on (you can't typo a
policy name), and keeps the documentation tooling (`cargo doc`) able to
follow the references. Toml/yaml would buy us cross-language portability
but jxl-encoder has no other-language consumers of the strategy data.

**Effort**: ~3 days. The macro is ~150 LOC, the existing four
constructors become one `strategies!{}` block of ~200 LOC. Net LOC
change is roughly zero, but the new code is declarative.

**Verdict**: nice-to-have but NOT critical. Defer to Phase 3.

---

### B. Gate registry macro (RECOMMENDED, Phase 1+2)

**Shape**: one macro invocation per gate. The macro emits everything:
- The field on `EncoderImprovementsCustom` and `ResolvedImprovements`
- The entry in the `from_custom()` copy function
- The default value used by each strategy preset (Libjxl / LeanFaster /
  Zenjxl / Aggressive)
- The divergence-table row text (consumed by a doc-build pass)
- Optionally: the env-var hook name and parser

Example registry file (`jxl-encoder/src/divergences/registry.rs`):

```rust
use crate::api::*;

register_gates! {
    // ── Section B content-aware gates ──────────────────────────────
    gate cfl_newton_libjxl_parity {
        ty: bool,
        default: false,
        defaults: {
            Libjxl: true,
            LeanFaster: false,
            Zenjxl: false,
            Aggressive: false,
        },
        env_var: Some(("JXL_W44_184_FORCE_LIBJXL_NEWTON", parse_force_bool)),
        divergence: {
            section: "C",
            row_title: "CfL Newton parameters (W44-183/184)",
            ours: "eps=1, max_iters=10, LS warm-start, LS fallback",
            libjxl: "eps=100, max_iters=20, start x=0, no LS fallback",
            status: "RESOLVED-via-opt-in",
            commit: Some("b8517c09a6d5"),
            notes: "W44-183 measurement: flipping at default-path regresses 25/27 photo cells. Salvaged via Libjxl strategy where the rest of the cost model is also flipped.",
        },
    }

    gate adaptive_buttloop_iters_narrow {
        ty: bool,
        default: true,
        defaults: {
            Libjxl: false,
            LeanFaster: false,
            Zenjxl: true,
            Aggressive: true,
        },
        env_var: None,
        divergence: {
            section: "B",
            row_title: "W44-169 distance-narrowed SmoothSkip",
            // ...
        },
    }
}
```

The macro produces:
1. Augments `EncoderImprovementsCustom` with the field.
2. Augments `ResolvedImprovements` with the field.
3. Auto-generates the `from_custom()` line.
4. Auto-fills the four constructors with the per-strategy defaults.
5. Wires `apply_env_var_fallbacks()` if `env_var: Some(...)`.
6. Emits an entry into a `static DIVERGENCE_ROWS: &[DivergenceRow]` table
   that a `build.rs` post-pass renders into `docs/LIBJXL_DIVERGENCES.md`.

**Adding a new gate** then becomes:
```rust
gate my_new_gate {
    ty: bool,
    defaults: { Libjxl: false, LeanFaster: false, Zenjxl: true, Aggressive: true },
    divergence: { section: "B", row_title: "...", ... },
}
```
~15 lines in one file. Plus the actual encoder-logic edit (the read
site). **Total: ~15 + 5 lines vs the W44-184 ~1137 lines of boilerplate.**

**Pros**:
- Solves G1 (boilerplate) and G5 (table vs code drift) directly.
- A linter can warn "this gate field is read at no call site" (because
  every read is `resolved.GATE_NAME`).
- The table generator can include the SOURCE LINE NUMBER automatically.
- Removing a gate is one-edit: delete the `gate {...}` block. All four
  constructors, the field on both structs, the divergence row, and the
  env-var hook are removed atomically.
- Cross-section composition checks become possible: the macro can verify
  that every gate marked `defaults: { Libjxl: <non-default> }` produces
  a corresponding Section A/B/C/D row.

**Cons**:
- Macro complexity. Requires a `proc_macro` crate (~600 LOC).
- The macro spans crates: it has to know about all the policy enums
  (`ScreenshotEntropyMulPolicy`, etc.). This is fine because they all
  live in `jxl-encoder/src/api.rs`; the macro just emits names.
- One-time learning cost for new contributors.

**Effort**: ~5-7 days for the macro + ~3 days to migrate the 24 existing
gates one-by-one. Each migration is a single PR that converts one gate
and verifies hash-lock parity.

**Verdict**: SHIP. Highest EV-vs-cost solution. Solves G1 (boilerplate)
and G5 (table-source drift) in one stroke.

---

### C. Libjxl-parity hash-lock CI (RECOMMENDED, Phase 4)

**Shape**: a small N-cell harness (5-10 fixtures) that:
1. Encodes each input with `EncoderStrategy::Libjxl`, hashes the bytes.
2. Encodes the same input with cjxl (CLI) at the matching effort/distance,
   hashes the bytes.
3. Pins both hash sets in `tests/strategy_libjxl_vs_cjxl.txt`.
4. CI fails if either the our-hash or the cjxl-hash diverges from the pin.

This is the **only existing test that catches Libjxl-strategy drift vs
the actual reference**. Today `tests/strategy_libjxl_hash_locks.rs`
pins our outputs against ourselves — which catches an accidental edit
to the Libjxl constructor but does NOT catch a divergence in some other
file that happens to break the parity claim.

**Composition with `cjxl-rs`**: the CI harness runs both the upstream
`cjxl` binary (installed in the CI runner) and our `cjxl-rs`
(`EncoderStrategy::Libjxl`), and asserts they match byte-for-byte on
fixtures where parity is claimed.

**Fixtures**: 10 cells covering the regression-prone axes:
- 2 photos at d=1.0 e7 (CID22 1418519, CID22 1025469)
- 2 photos at d=4.0 e7 (CID22 1531677, CID22 1189261)
- 2 screenshots at d=1.0 e5 (gb82-sc imac_g3, gb82-sc terminal)
- 2 screenshots at d=4.0 e8 (gb82-sc codec_wiki, gb82-sc imac_dark)
- 1 grayscale at d=2.0 e6
- 1 RGBA at d=2.0 e7

Total CI time: ~30 seconds of encoding × 2 codecs = ~1 minute on a CI
runner. Acceptable.

**What about cells that aren't byte-parity?** Section A effort-gate
divergences and Section D KNOWN-BUG cells are by design NOT
byte-identical. The CI harness pins our hash separately for those (mark
as "expected-divergent"), and the bench gates Libjxl against cjxl on
metric deltas (size/SSIM2/butteraugli) within a tolerance band. This
catches "we accidentally improved the divergence by 2%" which would
otherwise look like a positive regression.

**Pros**:
- Solves G3 directly. Gives the user the cjxl-strict-mode warranty.
- Catches multi-file regressions that hash-lock-only CI cannot.
- Cheap (5-10 fixtures, ~1 min CI time).

**Cons**:
- Requires cjxl binary in CI. Easy: `apt install libjxl-tools` or build
  from source pinned to a libjxl tag.
- cjxl version pinning is its own discipline. Bumping libjxl-tools
  version invalidates the pinned cjxl hashes. We need a documented "bump
  cjxl version" runbook.

**Effort**: ~1-2 days. The test framework already exists
(`tests/hash_lock_expected.txt` pattern); we copy it and add a second
hash column for cjxl output.

**Verdict**: SHIP. Independent of solution B. Unconditional benefit.
Bundle into Phase 4.

---

### D. Per-strategy entry points (NOT RECOMMENDED)

**Shape**: `lossy_encode_libjxl()` / `lossy_encode_zenjxl()` /
`lossy_encode_leanfaster()` as separate top-level functions. Each
function does no runtime strategy dispatch — the strategy is encoded in
the function name. Shared logic via generics or traits.

**Pros**:
- Zero runtime branches. Best-case codegen.
- The compiler can const-evaluate every gate.

**Cons**:
- Code bloat. Each strategy variant gets its own monomorphization of every
  internal function it transitively calls.
- The four-strategy matrix means 4× the binary size for any non-inlined
  hot path.
- Custom strategies (the most useful for experimentation) can't be
  pre-monomorphized — they fall back to dynamic dispatch anyway, defeating
  the point.
- Refactoring nightmare. Adding a fifth strategy = N new entry points.

**Verdict**: REJECT. The runtime-branch cost is negligible (the dispatch
happens once per encode, not per pixel). Code-bloat cost is significant.

---

### E. DiscriminatorChain data structure (RECOMMENDED for Phase 5)

**Shape**: an explicit composable chain of discriminator predicates:

```rust
pub struct DiscriminatorChain {
    pub steps: Vec<DiscriminatorStep>,
}

pub enum DiscriminatorStep {
    Outer(Box<dyn Fn(&ImageProxies, &EncodeParams) -> bool>),
    Nested {
        parent: usize,  // index into steps
        predicate: Box<dyn Fn(&ImageProxies, &EncodeParams) -> bool>,
    },
    Action(Action),
}
```

The W44-29/91/96/98/99/100/152/156/166 nested chain becomes:

```rust
let high_d_photo_chain = DiscriminatorChain {
    steps: vec![
        // W44-29 outer
        DiscriminatorStep::Outer(Box::new(|px, p|
            p.target_distance >= 3.0 && px.mask1x1_median < 50.0)),
        // W44-91 admit 1189261-class inside W44-29
        DiscriminatorStep::Nested {
            parent: 0,
            predicate: Box::new(|px, _|
                px.m_colourfulness >= 80.0 && px.fcbr < 0.01),
        },
        // W44-96 admit {1420710, 1531677} inside W44-29 + above
        DiscriminatorStep::Nested {
            parent: 0,
            predicate: Box::new(|px, p|
                p.target_distance >= 4.5 && px.edge_density >= 0.7 && px.fcbr < 0.01),
        },
        // ... etc
    ],
};
```

The chain is then walked at the call site and the matching action is
applied. A test harness can construct synthetic `ImageProxies` and
assert which step fires.

**Pros**:
- Solves G7 directly.
- Discriminator predicates become testable in isolation.
- "What fires on this image?" becomes a query, not an investigation.
- Composition rules are data, not control flow.

**Cons**:
- Per-encode dispatch cost (vec walk + dyn Fn calls). Probably ~1 µs,
  but measurable on tiny images.
- Refactor risk: 14+ nested chains to migrate, each with its own
  composition rules. Easy to get the parent index wrong.
- Need a custom serialization layer to dump a chain ("for image X,
  steps fired: 0, 2, 5") for sweep harness debugging.

**Effort**: ~5-7 days for the framework + ~2-3 days per chain to
migrate. ~30 days total if we migrate all 14 chains.

**Verdict**: DEFER. Real value but not urgent. Land after the gate
registry (Solution B) so the chains can be expressed declaratively
against the registry's gate names.

---

### F. Compile-time strategy specialization (NOT RECOMMENDED)

**Shape**: `EncoderStrategy` as a const-generic or feature-flag,
allowing per-strategy code paths to be eliminated at compile time.

**Pros**:
- Zero-cost abstraction.

**Cons**:
- `EncoderStrategy::Custom(Box<...>)` doesn't fit a const-generic.
- Rust's const-generic story for enums is still limited (2026).
- Feature-flag approach means shipping multiple builds.
- The dispatch cost we'd save is, again, negligible.

**Verdict**: REJECT. Same reasoning as D.

---

### Recommendation summary

| Solution | Goals satisfied | Effort | Recommendation |
|---|---|---|---|
| A. Declarative strategy DSL | G2 partial | 3d | Defer to Phase 3 |
| B. Gate registry macro | G1, G5, G6 | 5-10d | **SHIP Phase 1+2** |
| C. Libjxl-parity CI | G3 | 1-2d | **SHIP Phase 4** |
| D. Per-strategy entry points | G4 partial | 5d | REJECT |
| E. DiscriminatorChain | G7 | 30d | Defer to Phase 5 |
| F. Compile-time specialization | none | 10d+ | REJECT |

---

## Section 4: Migration plan

Phased plan that does not break shipping. Each phase is a discrete chunk
that lands on main with hash-lock parity before the next phase starts.

### Phase 1 (1 week): introduce the gate registry alongside existing system

**Deliverables**:
- New crate `jxl-encoder-divergence-macro` with the `register_gates!`
  proc-macro (~600 LOC).
- New file `jxl-encoder/src/divergences/registry.rs` with the
  `register_gates!{}` invocation listing EXISTING gates (one entry per
  gate, mirroring the current state).
- Build-script post-pass that emits a new file
  `docs/LIBJXL_DIVERGENCES_GENERATED.md` alongside the hand-maintained
  one. The two files must agree exactly (CI check).
- ZERO changes to encoder logic. ZERO call-site rewrites. The macro emits
  the same code that's hand-written today.

**Acceptance**:
- All 36 hash-locks BYTE-IDENTICAL.
- All existing tests pass.
- `cargo doc` builds with no warnings.
- The generated divergence table matches the hand-maintained one
  (modulo whitespace).
- The `register_gates!{}` block compiles and is unused (`#[allow(dead_code)]`
  for now).

**Risk**: low. The macro is additive; no existing code depends on it.

### Phase 2 (1 week per ~5 gates, ~5 weeks total): migrate gates

Per gate:
1. Move the field declaration from manual `EncoderImprovementsCustom` /
   `ResolvedImprovements` to the macro registry.
2. Delete the now-redundant entries from the four hand-written
   constructors.
3. Verify hash-locks BYTE-IDENTICAL.
4. Update the divergence-table row (the macro now owns it).

**Order**: easiest first. Start with the bool-typed Smart-Zenjxl gates
(W44-164, W44-165, W44-166, W44-167, W44-168, W44-169, W44-176, W44-184)
since they all share the same shape. Move to the enum-typed gates
(`ScreenshotEntropyMulPolicy` etc.) once the macro handles enums cleanly.

**Acceptance per gate**:
- Hash-lock parity.
- One CI run.
- Diff is `+N lines macro / -M lines manual` where `N << M`.

**Risk**: low per gate (atomic migration). Cumulative risk: if a macro bug
ships, it could subtly break ~24 gates at once. Mitigated by Phase 1's
"build the macro first, validate against existing code without
deleting" gate.

### Phase 3 (1 week): delete the manual constructors

Once every gate has migrated to the macro:
- Delete the hand-written `fn libjxl()`, `fn lean_faster()`,
  `fn zenjxl()`, `fn aggressive()`, `fn from_custom()` implementations
  on `ResolvedImprovements`.
- Delete the hand-maintained `docs/LIBJXL_DIVERGENCES.md` — the
  generated one becomes the source of truth (the build script now
  writes to `docs/LIBJXL_DIVERGENCES.md` directly).
- Update CLAUDE.md to reference the macro-registry workflow instead of
  the hand-edit workflow.

**Acceptance**:
- All hash-locks BYTE-IDENTICAL.
- `git log --oneline -- docs/LIBJXL_DIVERGENCES.md` shows the file is
  now updated by the build script (CI commit) rather than human commits.

**Risk**: medium. This is the irreversible step. If a gate was missed
during Phase 2, this is where we discover it. Mitigated by Phase 2's
per-gate hash-lock verification.

### Phase 4 (1 week): cjxl byte-parity CI

- Add `cjxl` binary to CI runner (`apt install libjxl-tools` pinned to a
  specific version, e.g., libjxl 0.12.0).
- New test file `tests/strategy_libjxl_vs_cjxl.rs` per Solution C.
- Pin baseline hashes in `tests/strategy_libjxl_vs_cjxl_expected.txt`.
- Document the "bump cjxl version" runbook in
  `docs/STRATEGY_LIBJXL_PARITY_CI.md`.

**Acceptance**:
- New CI job runs in ~1 minute.
- All 10 baseline fixtures pass.

**Risk**: low. Pure additive test.

### Phase 5+ (defer): DiscriminatorChain refactor (Solution E)

Land after Phase 4 stabilizes. Per-chain migration as a 2-3 day chunk
each. Optional.

### Phase 6+ (defer): Declarative strategy DSL (Solution A)

After Phase 3 deletes the manual constructors, the declarative DSL is a
nice-to-have on top of the macro registry. Reduces strategy bundle
definitions from ~200 LOC of `register_gates!{}` per-strategy `defaults:`
entries to a single `strategies!{}` block. Optional.

---

## Section 5: Open questions (user decides before implementation)

### Q1: DSL syntax — toml vs yaml vs Rust-source macro file?

**Recommendation**: Rust-source macro file.

**Rationale**:
- jxl-encoder has no other-language consumers of the strategy data
  (no Python binding reads the strategy bundle).
- A Rust-source macro file keeps type-checking on (you can't typo a
  policy name like `ScreenshtEntropyMulPolicy`).
- Toml/yaml require a build-script parser, adding a build-time cost
  and a non-Rust dependency.
- Documentation tooling (`cargo doc`) can follow references in a Rust
  macro; it cannot follow string references in a toml file.

**Alternative**: toml — buys cross-language portability we don't need.

**User decision needed**: yes / no on the Rust-source macro file
approach.

### Q2: Libjxl-strategy lock granularity — per-fixture byte-parity or per-strategy aggregate hash?

**Recommendation**: per-fixture byte-parity (10 fixtures × 2 codecs ×
hash = 20 expected values).

**Rationale**:
- Per-fixture catches a regression on cell N=3 even if cells 1-2 and 4-10
  still pass. An aggregate hash would hide single-cell drift.
- The maintenance cost is 10 hashes vs 1, which is trivial.

**Alternative**: aggregate hash (1 expected value).

**User decision needed**: per-fixture (10 hashes) vs aggregate (1 hash)?

### Q3: Custom strategy — runtime-configurable per-encode-call or also DSL-defined?

Today `EncoderStrategy::Custom(Box<EncoderImprovementsCustom>)` is set
per-encode-call via builder methods. The DSL (Solution A) currently
defines named strategies (`Libjxl`, `Zenjxl`, etc.) but does not have a
way to define a Custom variant declaratively.

**Recommendation**: keep Custom runtime-only. Named strategies in the
DSL; Custom remains a hand-built `EncoderImprovementsCustom` value.

**Rationale**:
- Named strategies are shipping defaults; Custom is for exploratory
  experimentation. The two have different lifecycles.
- A DSL-defined Custom would conflict with the goal "Custom is easy to
  compose for experimentation" — you don't want to edit a file and rebuild
  for each experiment.

**Alternative**: define `experimental!()` macro that produces a Custom
value from a one-liner spec at the call site. This is a nice ergonomic
follow-on but doesn't need to land in Phase 1.

**User decision needed**: keep Custom runtime-only (yes / no)? Land
`experimental!()` macro in Phase 1 or defer?

### Q4: Migration — big-bang or gradual?

**Recommendation**: gradual (per-gate migration in Phase 2).

**Rationale**:
- Big-bang risks a single broken commit that requires a rollback.
- Per-gate migration is reviewable as a small PR each.
- Hash-lock CI catches per-gate breakage immediately.

**Alternative**: big-bang in one giant PR. Reviewer-unfriendly.

**User decision needed**: yes / no on gradual approach.

### Q5: Should env-var hooks be retired entirely once promoted to API?

There are 23 env-only hooks today (not promoted by W44-132). Should
Phase 2 promote them all to first-class fields and delete the env-var
fallback, or should the env-var layer stay forever as a sweep
convenience?

**Recommendation**: KEEP env-var fallback for the gates that have one.
Document the env-var name as part of the gate registry's
`env_var: Some(...)` field. Generate a `docs/ENV_VARS.md` from the
registry so users can find them.

**Rationale**:
- Sweep harnesses pre-date the API and rely on env-vars.
- Removing them breaks `examples/w44_*_ab.rs` bench files.
- Keeping them documented (via registry) solves the "invisible" complaint
  from §1.7.

**Alternative**: promote all 23 and delete the env-var layer. Breaks
benches.

**User decision needed**: keep env-var fallback (yes / no)?

### Q6: Sub-agent prompts and CLAUDE.md updates

CLAUDE.md currently says "every chunk that touches a gate MUST update
LIBJXL_DIVERGENCES.md". With Solution B + Phase 3, the divergence-table
file becomes machine-generated. The CLAUDE.md instruction should be
updated to "every chunk MUST update the matching `gate {...}` block in
`src/divergences/registry.rs`".

**Recommendation**: update CLAUDE.md as part of Phase 3 cutover.

**User decision needed**: confirm CLAUDE.md gets updated in Phase 3.

### Q7: Backward compatibility for the `EncoderImprovementsCustom` struct

Today external Custom callers use struct-update syntax:
```rust
let custom = EncoderImprovementsCustom {
    screenshot_entropy_mul: ScreenshotEntropyMulPolicy::ForceOn,
    ..Default::default()
};
```

With Solution B, the struct fields are macro-generated. The struct is
still a `pub struct` with `pub` fields, so struct-update syntax keeps
working — but renaming a field via the macro now requires updating
external callers. Is this OK?

**Recommendation**: yes — we have no external users (per CLAUDE.md
"no backwards compatibility required"). Bump the 0.x version on any
field rename and document in CHANGELOG.

**User decision needed**: confirm "no external users, can rename
freely" still holds.

---

## Section 6: Risks and counter-arguments

### 6.1 Macro overhead

A proc-macro adds ~3 seconds to `cargo build` cold and ~0.5 seconds
warm. Given jxl-encoder's existing ~2 minute cold build, this is
negligible (~2.5%).

### 6.2 Debugger experience

Stepping through generated code is harder than stepping through
hand-written code. Mitigation: the macro emits standard struct
declarations and function bodies that look the same as the hand-written
equivalents (`cargo expand` shows the generated code clearly).

### 6.3 Macro error messages

`proc_macro` error spans can be confusing. Mitigation: write a battery
of compile-fail tests (`#[test] fn missing_field_in_strategy_compile_fails()`)
so contributors hit good error messages, not raw `proc_macro` panics.

### 6.4 What if cjxl ships a regression upstream?

Phase 4's CI pins our hash AND cjxl's hash. If libjxl ships a regression
that changes cjxl's output, the CI fails until we either:
(a) accept the new cjxl output as the new pin (libjxl-parity intent
preserved), or
(b) document why we deliberately diverge from the new cjxl behavior
(treat as a new Section A or B row).

Either path is healthy. The current state — where we don't check vs cjxl
at all — silently lets us drift.

### 6.5 What if Zenjxl moves SO far from Libjxl that they share no code?

Unlikely. Even today, ~80% of the encoder code is shared between
strategies; only the gate predicates and a few constants differ.
Solutions B+C scale to "Libjxl is a deeply different code path" if
needed (every gate just declares per-strategy defaults; if they all
differ, the macro generates entirely separate constructors).

### 6.6 Time-to-value

Phase 1+2+3+4 is ~8 weeks of compounded work. Is it worth it given
each gate's marginal cost is "only" 50 boilerplate lines?

**Argument for shipping it**: the W44 gates have been accumulating at
~3 per week for the last 6 months. At 50 lines per gate, that's 150
boilerplate lines per week, ~7800 lines per year. Solution B drops
that to ~225 lines per year (15 per gate). Payback is ~5 weeks.

---

## Section 7: Acceptance gates for this RFC

Before implementation starts:
1. User signs off on Section 5 open questions (Q1-Q7).
2. User confirms Phase ordering (the recommended Phase 1+2+3+4 sequence,
   deferring Phase 5+6).
3. Author files implementation issues:
   - Issue: "Phase 1 — gate registry macro infrastructure"
   - Issue: "Phase 2 — migrate gates A through F"
   - Issue: "Phase 3 — delete manual constructors"
   - Issue: "Phase 4 — Libjxl byte-parity CI"

Each issue references this RFC and the relevant Section 5 decisions.

---

## Section 8: References

- `docs/COMPATIBILITY_MODES.md` — W44-125 v2 EncoderStrategy design (the
  foundation this RFC builds on)
- `docs/LIBJXL_DIVERGENCES.md` — the inventory that Solutions B and C
  reorganize
- `jxl-encoder/src/api.rs:3740-5128` — the current `EncoderStrategy`
  and `EncoderImprovementsCustom` implementations (Solution B replaces
  most of this)
- `jxl-encoder/src/effort.rs:1983-2046` — the `apply_section_a_effort_gates`
  and `apply_section_c_cfl_newton_libjxl_parity` patterns Solution B
  generalizes
- W44-127→133 series commits: `4f52af16` (Chunk A), `9d45e999` (Chunk B),
  `a9e6763d..4766e229` (Chunk C, 8 commits), `207ae055..9cd9739c`
  (Chunk D, 6 commits), `133897d9` (Chunk E), `49093cef` (Chunk F),
  `d6330c26..8691aafc` (Chunk G, 4 commits)
- W44-184 commit `b8517c09a6d5` — the freshest example of per-gate
  boilerplate cost (17 files, +1137/-71 lines for a single bool gate)
- W44-166 commit `14cc95b2fc6a` — similar (8 files, +1279/-7 lines for
  a single bool gate)
- CLAUDE.md `MANDATORY: Maintain docs/LIBJXL_DIVERGENCES.md with every
  change` — the drift problem Solution B + Phase 3 eliminates

---

## Appendix A: Sample `register_gates!{}` block (concrete shape)

This appendix shows what the registry would look like for the 4 most
recent gates (W44-164, W44-176, W44-184, W44-169) to illustrate the
boilerplate reduction concretely.

```rust
// jxl-encoder/src/divergences/registry.rs

use crate::api::*;

register_gates! {
    // ── W44-164 (Smart-Zenjxl chunk 1, 2026-05-21) ───────────────
    gate content_class_auto_classify {
        ty: bool,
        defaults: {
            Libjxl: false,
            LeanFaster: false,
            Zenjxl: true,
            Aggressive: true,
        },
        env_var: None,
        divergence: {
            section: "B",
            row_title: "W44-164 auto_classify_content_class_from_layout",
            ours: "fcbr >= 0.35 → Screenshot; fcbr < 0.10 AND m3 >= 5 → Photo",
            libjxl: "no per-image classifier",
            status: "INTENTIONAL",
            commit: Some("ccae9089c08d"),
            min_pixels: 65_536,
            notes: "Discriminator calibrated from 10 GB82-SC + 41 CID22.",
        },
    }

    // ── W44-176 (Smart-Zenjxl follow-on, 2026-05-21) ─────────────
    gate terminal_class_exclude {
        ty: bool,
        defaults: {
            Libjxl: false,
            LeanFaster: false,
            Zenjxl: true,
            Aggressive: true,
        },
        env_var: Some(("JXL_W44_176_DISABLE", parse_force_bool_inverted)),
        divergence: {
            section: "B",
            row_title: "W44-176 terminal-class exclude sub-discriminator",
            ours: "luma_var ∈ [1500, 2200] AND fcbr ≥ 0.70 → suppress W44-109 lift",
            libjxl: "no per-image classifier (W44-109 itself is not in libjxl)",
            status: "INTENTIONAL",
            commit: Some("a75d8c389043"),
        },
    }

    // ── W44-184 (CfL Newton libjxl-parity salvage, 2026-05-22) ───
    gate cfl_newton_libjxl_parity {
        ty: bool,
        defaults: {
            Libjxl: true,
            LeanFaster: false,
            Zenjxl: false,
            Aggressive: false,
        },
        env_var: Some(("JXL_W44_184_FORCE_LIBJXL_NEWTON", parse_force_bool)),
        divergence: {
            section: "C",
            row_title: "CfL Newton parameters (W44-183 measurement, W44-184 salvage)",
            ours: "eps=1, max_iters=10, LS warm-start, LS fallback",
            libjxl: "eps=100, max_iters=20, start x=0, no LS fallback",
            status: "RESOLVED-via-opt-in",
            commit: Some("b8517c09a6d5"),
            notes: "Salvaged via Libjxl strategy only (W44-183 found default-path port regressed 25/27 photo cells).",
        },
    }

    // ── W44-169 (distance-narrowed SmoothSkip, 2026-05-21) ────────
    gate adaptive_buttloop_iters_narrow {
        ty: bool,
        defaults: {
            Libjxl: false,
            LeanFaster: false,
            Zenjxl: true,
            Aggressive: true,
        },
        env_var: None,
        divergence: {
            section: "B",
            row_title: "W44-169 distance-narrowed SmoothSkip iter reduction",
            ours: "effort >= 8 AND d ∈ [4.0, 5.0] AND mask_median > 95 → iters - 1",
            libjxl: "fixed per-effort schedule (e≤7=0, e8=2, e9=4, ...)",
            status: "INTENTIONAL",
            commit: Some("3dfd53e9cc3b"),
        },
    }
}
```

**Size comparison**:
- Current shape (4 gates × ~50 boilerplate lines each, distributed across
  `EncoderImprovementsCustom` + `ResolvedImprovements` + 4 constructors +
  `Default` impls + env-var fallback + divergence-table rows): ~200 lines
  scattered across 4 files.
- New shape: ~80 lines in one file (the macro generates everything else).

For the full 24-gate registry, the projected file size is ~500 lines
declarative (replacing ~2400 lines of scattered boilerplate today).

---

## Appendix B: What does NOT change

To set expectations:
- **The public API surface stays the same**. `EncoderStrategy`,
  `EncoderImprovementsCustom`, `LossyConfig::with_strategy`, all the
  policy enums — same shapes. The macro just generates the
  `EncoderImprovementsCustom` field list instead of hand-writing it.
- **The encoder logic doesn't change**. Call sites still read
  `resolved.GATE_NAME` exactly as today.
- **Hash-locks stay binding**. Phase 1+2+3 preserve every fixture
  BYTE-IDENTICAL.
- **The discriminator predicates stay where they are** (Solution E moves
  them later, but that's Phase 5+ and explicitly deferred).
- **The strategy preset semantics stay the same**. Libjxl is still
  all-divergence parity; Zenjxl is still production default; LeanFaster
  is still drop-heavy-gates; Aggressive is still forward-compat slot.

What DOES change:
- Adding a gate becomes ~15 lines in one file (vs ~1100 lines across 17
  files today).
- The divergence table becomes machine-generated (vs hand-edited per
  CLAUDE.md mandatory rule).
- CI starts catching Libjxl-vs-cjxl drift (vs only catching
  Libjxl-vs-ourselves drift today).

---

*End of RFC.*
