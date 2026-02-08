# Module Reorganization Plan

## Goal

Rename `tiny/` to `vardct/`, fix backwards dependencies (modular→tiny), remove dead
code, and give files obvious names. The public API is unchanged. Hash lock tests
remain byte-exact.

---

## Current Problems (ranked)

### 1. `tiny/` is a lie (HIGH)

The module is 30k lines, 35 files — the primary VarDCT encoder. "Tiny" comes from the
libjxl-tiny origin, but CLAUDE.md explicitly says "libjxl-tiny was a stepping stone,
NOT the reference." The struct is `TinyEncoder` with 271 references across 17 files.
Should be `vardct/` with `VarDctEncoder`.

### 2. Shared code trapped inside `tiny/` (HIGH)

Three files in `tiny/` are used by `modular/` — creating a backwards dependency where
the lossless path depends on the lossy module:

| File | Used by outside tiny/ | Lines |
|------|----------------------|-------|
| `tiny/entropy_code.rs` | modular/improved.rs, modular/section.rs | 1712 |
| `tiny/token.rs` | modular/improved.rs, modular/section.rs (as AnsToken) | 259 |
| `tiny/icc_codec.rs` | api.rs (write_icc) | 851 |

Additionally, `tiny/lz77.rs` (1177 lines) exports `Lz77Method` which is re-exported
as public API through `api.rs → crate::tiny::Lz77Method`.

The correct home for these is `entropy_coding/` (for entropy infrastructure) and
top-level `icc.rs` (for ICC profile encoding).

### 3. 865 lines of dead code in `modular/` (MEDIUM)

| File | Lines | Evidence |
|------|-------|---------|
| `modular/encoder.rs` | 279 | Types `ModularEncoder`, `ModularEncoderOptions`, `EncodedModularData` — re-exported in mod.rs but zero imports from any other file |
| `modular/minimal.rs` | 382 | Not even exported in mod.rs, zero imports from outside |
| `modular/token.rs` | 204 | Types `Token`, `TokenList`, `Histogram` — only consumer is dead `encoder.rs` |

### 4. `frame/` is misleading (MEDIUM)

Contains only `FrameEncoder` which does modular frame assembly. VarDCT frame assembly
is in `tiny/encoder.rs` and `tiny/frame.rs`. The name implies general frame handling
but it's modular-specific. Should fold into `modular/`.

### 5. `improved.rs` is meaningless (LOW)

It's the main modular encoding pipeline (2553 lines). "Improved" was relative to
`minimal.rs` which is dead code. Should be `encode.rs`.

### 6. `encoder_tests.rs` orphaned name (LOW)

Was `#[path = "encoder_tests.rs"]` inside the deleted `encoder.rs`. Now included from
`lib.rs`. Tests the public API — name should reflect that.

### 7. Two token types causing confusion (LOW)

`tiny/token.rs::Token` (with `is_lz77_length` flag) is used by both VarDCT and modular.
`modular/token.rs::Token` (simpler) is dead code. After cleanup, only one Token type
remains — the confusion is self-resolving.

### 8. Duplicate Histogram/clustering types (INFO — NOT a fix target)

`tiny/cluster.rs` has a `Histogram` type with fixed `[u32; ALPHABET_SIZE]` arrays, used
by `tiny/context_tree.rs`. `entropy_coding/cluster.rs` has a general-purpose `Histogram`
from `entropy_coding/histogram.rs`. These serve different needs (fixed-size VarDCT
contexts vs dynamic distributions). Don't merge them.

---

## Target Structure

```
jxl_encoder/src/
├── api.rs                    — Public API (unchanged)
├── lib.rs                    — Crate root
├── icc.rs                    — ICC profile encoding (moved from tiny/icc_codec.rs)
├── vardct/                   — VarDCT (lossy) encoder (renamed from tiny/)
│   ├── mod.rs                — Updated module declarations
│   ├── encoder.rs            — VarDctEncoder (was TinyEncoder)
│   ├── dct.rs                — DCT transforms (unchanged)
│   ├── bitstream.rs          — VarDCT bitstream assembly
│   ├── ac_strategy.rs        — AC strategy selection
│   ├── transform.rs          — Quantization/dequantization
│   ├── adaptive_quant.rs     — Perceptual masking
│   ├── reconstruct.rs        — Encoder-side reconstruction
│   ├── quant.rs              — Quant weight tables
│   ├── epf.rs                — Edge-preserving filter
│   ├── noise.rs              — Noise synthesis
│   ├── frame.rs              — VarDCT frame header writing
│   ├── rate_control.rs       — Butteraugli quant loop
│   ├── cluster.rs            — VarDCT-specific histogram clustering
│   ├── context_tree.rs       — Block context tree
│   ├── dc_coding.rs          — DC coding
│   ├── dc_tree_learn.rs      — DC tree learning
│   ├── static_codes.rs       — Static prefix codes
│   ├── ac_context.rs         — AC entropy contexts
│   ├── ac_group.rs           — AC group assembly
│   ├── coeff_order.rs        — Custom coefficient ordering
│   ├── afv.rs                — Corner DCT
│   ├── chroma_from_luma.rs   — CfL
│   ├── gaborish.rs           — Gaborish inverse
│   ├── common.rs             — Shared utilities
│   ├── debug_log.rs          — Debug logging macros
│   ├── tile_distmap.rs       — Butteraugli distmap
│   ├── precomputed.rs        — Rate-control precomputed data
│   └── tests.rs              — Module-level tests
├── modular/                  — Modular (lossless) encoder
│   ├── mod.rs                — Updated (dead code removed)
│   ├── encode.rs             — Main pipeline (renamed from improved.rs)
│   ├── frame.rs              — Modular frame assembly (moved from frame/frame_encoder.rs)
│   ├── section.rs            — Global/group section writing
│   ├── channel.rs            — ModularImage, Channel types
│   ├── predictor.rs          — 14 predictors including Weighted
│   ├── tree.rs               — Decision tree types
│   ├── tree_learn.rs         — MA tree learning
│   ├── rct.rs                — Reversible color transform
│   ├── squeeze.rs            — Haar wavelet transform
│   └── palette.rs            — Palette transform
├── entropy_coding/           — Unified entropy coding
│   ├── mod.rs                — Updated with new re-exports
│   ├── ans.rs                — ANS encoder core (existing)
│   ├── ans_decode.rs         — ANS decoder for verification (existing)
│   ├── cluster.rs            — General histogram clustering (existing)
│   ├── context_map.rs        — Context map encoding (existing)
│   ├── histogram.rs          — General Histogram type (existing)
│   ├── huffman_tree.rs       — Huffman tree building (existing)
│   ├── hybrid_uint.rs        — HybridUint config (existing)
│   ├── encode.rs             — Token→bitstream pipeline (moved from tiny/entropy_code.rs)
│   ├── token.rs              — Token type for entropy coding (moved from tiny/token.rs)
│   └── lz77.rs               — LZ77 backward references (moved from tiny/lz77.rs)
├── headers/                  — File + frame header writing (unchanged)
├── bit_writer.rs             — Bitstream writer (unchanged)
├── color/                    — XYB conversion (unchanged)
├── container.rs              — EXIF/XMP box wrapping (unchanged)
├── image/                    — Image buffer types (unchanged)
├── error.rs                  — Error types (unchanged)
├── trace.rs                  — Tracing macros (unchanged)
├── test_helpers.rs           — Decoder wrappers (unchanged location)
├── encoder_tests.rs          — API tests (rename to api_tests.rs in later step)
└── tests/
    └── baselines.rs          — (unchanged)
```

Changes deleted: `frame/` directory (merged into modular/), `modular/encoder.rs`,
`modular/minimal.rs`, `modular/token.rs`.

---

## Execution Steps

Each step is one commit. Run verification after each.

### Step 1: Delete dead code (zero risk, 865 lines removed)

**Delete 3 files:**
- `modular/encoder.rs` (279 lines) — dead: types re-exported but never imported
- `modular/minimal.rs` (382 lines) — dead: not exported, not imported
- `modular/token.rs` (204 lines) — dead: only consumer was `encoder.rs`

**Edit `modular/mod.rs`:**
- Remove `pub mod encoder;`
- Remove `pub mod minimal;`
- Remove `pub mod token;`
- Remove re-exports: `pub use encoder::{...}`, `pub use token::Token`

**Verification:**
```bash
cargo fmt && cargo clippy -- -D warnings && cargo test -p jxl_encoder --lib
cargo test -p jxl_encoder --test hash_lock_features
```

**Commit:** `refactor: delete 865 lines of dead code in modular/ (encoder, minimal, token)`

---

### Step 2: Extract shared code from `tiny/` to `entropy_coding/` (fixes backwards deps)

**Move 3 files:**
```
tiny/entropy_code.rs → entropy_coding/encode.rs
tiny/token.rs        → entropy_coding/token.rs
tiny/lz77.rs         → entropy_coding/lz77.rs
```

**Edit `entropy_coding/mod.rs`** — add 3 new module declarations and re-exports:
```rust
pub mod encode;    // was tiny/entropy_code.rs
pub mod token;     // was tiny/token.rs
pub mod lz77;      // was tiny/lz77.rs

pub use lz77::Lz77Method;
```

**Edit `tiny/mod.rs`** (soon to be `vardct/mod.rs`):
- Remove `pub(crate) mod entropy_code;`
- Remove `pub(crate) mod token;`
- Remove `mod lz77;`
- Remove `pub use lz77::Lz77Method;`

**Fix internal imports within the 3 moved files:**

In `entropy_coding/encode.rs` (was `tiny/entropy_code.rs`):
- `super::lz77::` → `super::lz77::` (stays same — both now in entropy_coding/)
- `super::token::` → `super::token::` (stays same)
- `crate::entropy_coding::ans::` → `super::ans::` (now same module)
- `use crate::debug_log` → `use crate::vardct::debug_log` (or rework)
- `super::cluster::` (test-only) → `crate::vardct::cluster::`

In `entropy_coding/token.rs` (was `tiny/token.rs`):
- `super::common::floor_log2_nonzero` → `crate::vardct::common::floor_log2_nonzero`

In `entropy_coding/lz77.rs` (was `tiny/lz77.rs`):
- `super::token::` → `super::token::` (stays same — both in entropy_coding/)

**Fix imports in 9 files within `tiny/` that imported from the moved files:**

Files importing `super::entropy_code`:
- `tiny/static_codes.rs` — change to `crate::entropy_coding::encode::`
- `tiny/icc_codec.rs` — change to `crate::entropy_coding::encode::`
- `tiny/dc_coding.rs` — change to `crate::entropy_coding::encode::`
- `tiny/coeff_order.rs` — change to `crate::entropy_coding::encode::`
- `tiny/bitstream.rs` — change to `crate::entropy_coding::encode::`
- `tiny/encoder.rs` — change to `crate::entropy_coding::encode::`
- `tiny/ac_group.rs` — change to `crate::entropy_coding::encode::`
- `tiny/cluster.rs` — change to `crate::entropy_coding::encode::`
- `tiny/context_tree.rs` — change to `crate::entropy_coding::encode::`

Files importing `super::token`:
- `tiny/icc_codec.rs` — change to `crate::entropy_coding::token::`
- `tiny/dc_coding.rs` — change to `crate::entropy_coding::token::`
- `tiny/coeff_order.rs` — change to `crate::entropy_coding::token::`
- `tiny/dc_tree_learn.rs` — change to `crate::entropy_coding::token::`
- `tiny/bitstream.rs` — change to `crate::entropy_coding::token::`
- `tiny/encoder.rs` — change to `crate::entropy_coding::token::`
- `tiny/ac_group.rs` — change to `crate::entropy_coding::token::`
- `tiny/context_tree.rs` — change to `crate::entropy_coding::token::`

Files importing `super::lz77`:
- `tiny/tests.rs` — change to `crate::entropy_coding::lz77::`
- `tiny/bitstream.rs` — change to `crate::entropy_coding::lz77::`
- `tiny/encoder.rs` — change to `crate::entropy_coding::lz77::`

**Fix imports in `modular/` (the backwards dependency disappears):**
- `modular/improved.rs:22-23` — `crate::tiny::entropy_code::` → `crate::entropy_coding::encode::`
                                  `crate::tiny::token::Token as AnsToken` → `crate::entropy_coding::token::Token as AnsToken`
- `modular/section.rs:19-20` — same changes

**Fix imports in `api.rs`:**
- `pub use crate::tiny::Lz77Method` → `pub use crate::entropy_coding::Lz77Method`

**Total: ~51 import line changes across ~15 files.**

**Verification:**
```bash
cargo fmt && cargo clippy -- -D warnings && cargo test -p jxl_encoder --lib
cargo test -p jxl_encoder --test hash_lock_features
cargo test -p jxl_encoder_cli
```

**Commit:** `refactor: extract entropy_code, token, lz77 from tiny/ to entropy_coding/`

---

### Step 3: Move `tiny/icc_codec.rs` to top-level `icc.rs`

**Move 1 file:**
```
tiny/icc_codec.rs → icc.rs
```

**Edit `lib.rs`:** Add `pub mod icc;`

**Edit `tiny/mod.rs`:** Remove `pub(crate) mod icc_codec;`

**Fix imports within `icc.rs`:**
- `super::entropy_code::` → `crate::entropy_coding::encode::`
- `super::token::` → `crate::entropy_coding::token::`
- `crate::bit_writer` and `crate::error` stay the same

**Fix 2 callers:**
- `api.rs:938` — `crate::tiny::icc_codec::write_icc` → `crate::icc::write_icc`
- `tiny/bitstream.rs` — `super::icc_codec::write_icc` → `crate::icc::write_icc`

**Verification:**
```bash
cargo fmt && cargo clippy -- -D warnings && cargo test -p jxl_encoder --lib
cargo test -p jxl_encoder --test hash_lock_features
```

**Commit:** `refactor: move icc_codec.rs from tiny/ to top-level icc.rs`

---

### Step 4: Rename `tiny/` → `vardct/` and `TinyEncoder` → `VarDctEncoder`

**Rename directory:** `mv jxl_encoder/src/tiny jxl_encoder/src/vardct`

**Edit `lib.rs`:** `pub mod tiny;` → `pub mod vardct;`

**Rename struct:** In `vardct/encoder.rs`, rename `pub struct TinyEncoder` → `pub struct VarDctEncoder`

**Update `vardct/mod.rs`:**
- Module doc: update "Tiny JPEG XL encoder" → "VarDCT (lossy) encoder"
- `pub use encoder::TinyEncoder;` → `pub use encoder::VarDctEncoder;`

**Update `debug_log.rs` macro path:**
- `$crate::tiny::debug_log::write_debug_log` → `$crate::vardct::debug_log::write_debug_log`
- `$crate::tiny::debug_log::flush_debug_log` → `$crate::vardct::debug_log::flush_debug_log`

**Global find/replace (in this order to avoid partial matches):**

1. `crate::tiny::` → `crate::vardct::` (19 occurrences across codebase)
2. `jxl_encoder::tiny::` → `jxl_encoder::vardct::` (integration tests + examples)
3. `TinyEncoder` → `VarDctEncoder` (271 occurrences across 17 files)

**Files requiring `TinyEncoder` → `VarDctEncoder`:**
- `vardct/encoder.rs` — struct definition + impl blocks
- `vardct/mod.rs` — re-export
- `vardct/bitstream.rs` — function parameters
- `vardct/transform.rs` — function parameters
- `vardct/reconstruct.rs` — function parameters
- `vardct/rate_control.rs` — function parameters
- `vardct/tests.rs` — test usage
- `api.rs` — construction site
- `test_helpers.rs` — detection logic
- `jxl_encoder_cli/src/main.rs` — CLI usage
- `tests/clic2025.rs` — integration tests
- `tests/dct4x8_diagnostic.rs` — integration tests
- `tests/llf_invariants.rs` — integration tests (many occurrences)
- `tests/rate_control.rs` — integration tests
- `examples/distance_sweep.rs` — example
- `examples/fresh_encode.rs` — example
- `examples/multigroup_gallery.rs` — example

**Also update in vardct/ files that now use `crate::entropy_coding::` for moved files:**
- These were already updated in Step 2, but their `super::` references to other vardct
  files stay as `super::` (they're still in the same module, just renamed).

**Verification:**
```bash
cargo fmt && cargo clippy -- -D warnings && cargo test -p jxl_encoder --lib
cargo test -p jxl_encoder --test hash_lock_features
cargo test -p jxl_encoder_cli
cargo test -p jxl_encoder --test llf_invariants
cargo test -p jxl_encoder --test clic2025 -- --ignored 2>/dev/null; true  # some may be slow
```

**Commit:** `refactor: rename tiny/ → vardct/, TinyEncoder → VarDctEncoder`

---

### Step 5: Move `frame/frame_encoder.rs` → `modular/frame.rs`

**Move 1 file:**
```
frame/frame_encoder.rs → modular/frame.rs
```

**Delete `frame/` directory** (both `frame/mod.rs` and now-empty dir).

**Edit `lib.rs`:**
- Remove `pub mod frame;`

**Edit `modular/mod.rs`:**
- Add `pub mod frame;`
- Add `pub use frame::{FrameEncoder, FrameEncoderOptions};`

**Fix imports within `modular/frame.rs`:**
- `crate::modular::channel::` → `super::channel::`
- `crate::modular::improved::` → `super::encode::` (already renamed in Step 6)
- `crate::modular::section::` → `super::section::`
- All other `crate::` imports stay the same

**Fix 1 caller:**
- `api.rs:891` — `use crate::frame::{FrameEncoder, FrameEncoderOptions};`
              → `use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};`

Wait — `modular/improved.rs` tests also create `FrameEncoder` instances. But these use
`crate::frame::FrameEncoder` which would become `super::frame::FrameEncoder`. Actually
since improved.rs is already in modular/, it can use `super::frame::FrameEncoder`.

**Integration tests:** `hash_lock_features.rs` doesn't import FrameEncoder. No
integration test imports FrameEncoder directly.

**Verification:**
```bash
cargo fmt && cargo clippy -- -D warnings && cargo test -p jxl_encoder --lib
cargo test -p jxl_encoder --test hash_lock_features
```

**Commit:** `refactor: move FrameEncoder from frame/ into modular/frame.rs`

---

### Step 6: Rename `modular/improved.rs` → `modular/encode.rs`

**Rename 1 file:**
```
modular/improved.rs → modular/encode.rs
```

**Edit `modular/mod.rs`:**
- `pub mod improved;` → `pub mod encode;`
- All `pub use improved::` → `pub use encode::`

**Fix imports in `modular/section.rs`:**
- `super::improved::` → `super::encode::`

**Fix imports in `modular/frame.rs` (was frame_encoder.rs):**
- `crate::modular::improved::` → `super::encode::`

**Verification:**
```bash
cargo fmt && cargo clippy -- -D warnings && cargo test -p jxl_encoder --lib
cargo test -p jxl_encoder --test hash_lock_features
```

**Commit:** `refactor: rename modular/improved.rs → modular/encode.rs`

---

### Step 7: Rename `encoder_tests.rs` → `api_tests.rs` (optional, low priority)

**Rename 1 file:**
```
encoder_tests.rs → api_tests.rs
```

**Edit `lib.rs`:**
```rust
#[cfg(test)]
#[path = "api_tests.rs"]
mod api_tests;
```

**Verification:**
```bash
cargo test -p jxl_encoder --lib
```

**Commit:** `refactor: rename encoder_tests.rs → api_tests.rs`

---

## Dependency Graph: Before → After

### BEFORE (backwards dependency)
```
modular/improved.rs ──→ tiny/entropy_code.rs ──→ entropy_coding/ans.rs
modular/section.rs  ──→ tiny/entropy_code.rs
modular/section.rs  ──→ tiny/token.rs
api.rs              ──→ tiny/icc_codec.rs
api.rs              ──→ tiny/Lz77Method
```

### AFTER (clean layer ordering)
```
modular/encode.rs   ──→ entropy_coding/encode.rs ──→ entropy_coding/ans.rs
modular/section.rs  ──→ entropy_coding/encode.rs
modular/section.rs  ──→ entropy_coding/token.rs
api.rs              ──→ icc.rs
api.rs              ──→ entropy_coding::Lz77Method

vardct/* ──→ entropy_coding/* (clean downward dependency)
modular/* ──→ entropy_coding/* (clean downward dependency)
Neither vardct/ nor modular/ imports from the other.
```

---

## Risk Assessment

| Step | Risk | Mitigation |
|------|------|------------|
| 1. Delete dead code | Zero | No code calls it |
| 2. Extract to entropy_coding/ | Medium | Many import path changes. Compiler catches all errors. |
| 3. Move icc_codec | Low | Only 2 callers |
| 4. Rename tiny→vardct | Low (mechanical) | Global find/replace. 271 TinyEncoder refs are text-only. |
| 5. Move FrameEncoder | Low | Only 2 callers (api.rs + modular tests) |
| 6. Rename improved→encode | Zero | 1 file rename + mod.rs update |
| 7. Rename test file | Zero | 1 file rename + lib.rs path |

**Total estimated import changes:** ~80 lines across ~20 files.
**Functional code changes:** Zero. This is purely structural.
**Hash lock tests:** Must remain byte-exact (encoder output is unchanged).

---

## What This Does NOT Change

- Public API types (`LosslessConfig`, `LossyConfig`, `EncodeRequest`, `PixelLayout`, etc.)
- Encoding behavior or output bytes
- `entropy_coding/` existing files (ans.rs, huffman_tree.rs, etc.)
- `headers/` module
- `color/` module
- `bit_writer.rs`, `error.rs`, `trace.rs`, `container.rs`, `image/`
- Any algorithm or constant

---

## Verification Commands (run after every step)

```bash
# Must-pass: compilation, lint, unit tests, hash lock
cargo fmt && cargo clippy -- -D warnings && cargo test -p jxl_encoder --lib
cargo test -p jxl_encoder --test hash_lock_features

# Should-pass: CLI and integration tests
cargo test -p jxl_encoder_cli
cargo test -p jxl_encoder --test llf_invariants
cargo test -p jxl_encoder --test ans_roundtrip
cargo test -p jxl_encoder --test minimal_ans
cargo test -p jxl_encoder --test dct4x8_diagnostic
cargo test -p jxl_encoder --test sixteenbit_metadata
```

The full `just rd-regression` should pass too but is slow (~3min) — run after all steps.
