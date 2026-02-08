# Context Handoff: Module Rename & Reorganization

## Task

Design and execute a rename/reorganization of the jxl_encoder crate's internal module
structure to be obvious, intuitive, and correct. The public API (LosslessConfig, LossyConfig,
EncodeRequest, PixelLayout, etc.) is unchanged — this is purely internal reorganization.

## What Just Happened (Phase 4 — completed)

Three commits deleted the old `Encoder`/`EncoderOptions`/free functions (`encoder.rs`, 301 lines).
`api.rs::encode_lossless()` now calls `FrameEncoder` directly. All 609 lib tests and 36 hash
lock tests pass. The old encoder was the last wrapper layer — now there are exactly two
encoding paths: `TinyEncoder` (VarDCT/lossy) and `FrameEncoder` (modular/lossless).

## Current Module Map

```
jxl_encoder/src/
├── api.rs              — Public API (1262 lines). LosslessConfig, LossyConfig, EncodeRequest.
│                         Calls FrameEncoder for lossless, TinyEncoder for lossy.
├── lib.rs              — Crate root (51 lines). Module declarations, re-exports.
├── tiny/               — VarDCT (lossy) encoder (30k lines, 33 files)
│   ├── encoder.rs      — TinyEncoder struct (1621 lines). The main lossy entry point.
│   ├── frame.rs        — VarDCT frame header writing, DistanceParams
│   ├── ac_strategy.rs  — AC strategy selection (2538 lines)
│   ├── dct.rs          — DCT transforms (3421 lines)
│   ├── entropy_code.rs — Huffman+ANS entropy for VarDCT AND modular (1712 lines) ← SHARED
│   ├── token.rs        — Token type used by both VarDCT and modular
│   ├── lz77.rs         — LZ77 backward references (1177 lines)
│   ├── bitstream.rs    — VarDCT bitstream assembly (1665 lines)
│   ├── transform.rs    — Quantization/dequantization (1596 lines)
│   ├── adaptive_quant.rs — Perceptual masking (1016 lines)
│   ├── reconstruct.rs  — Encoder-side reconstruction (1261 lines)
│   ├── quant.rs        — Quant weight tables (1304 lines)
│   ├── epf.rs          — Edge-preserving filter (872 lines)
│   ├── noise.rs        — Noise synthesis (953 lines)
│   ├── icc_codec.rs    — ICC profile encoding (used by api.rs too)
│   ├── rate_control.rs — Butteraugli quant loop
│   └── ... (15 more small files)
├── frame/              — Modular frame assembly (1 file + mod.rs)
│   └── frame_encoder.rs — FrameEncoder struct. Writes modular frames with TOC.
├── modular/            — Modular (lossless) encoder (7.7k lines, 13 files)
│   ├── improved.rs     — Main modular pipeline (2553 lines). Multi-context, tree, ANS.
│   ├── section.rs      — Global/group modular section writing
│   ├── tree_learn.rs   — MA tree learning (921 lines)
│   ├── predictor.rs    — 14 predictors including Weighted (766 lines)
│   ├── tree.rs         — Decision tree types and traversal
│   ├── channel.rs      — ModularImage, Channel types (482 lines)
│   ├── rct.rs          — Reversible color transform
│   ├── squeeze.rs      — Haar wavelet transform
│   ├── palette.rs      — Palette transform
│   ├── encoder.rs      — DEAD CODE (ModularEncoder, never used outside itself)
│   ├── minimal.rs      — DEAD CODE (write_minimal_modular_stream, never called)
│   ├── token.rs        — Modular token types
│   └── mod.rs
├── headers/            — File + frame header writing (6 files)
├── entropy_coding/     — ANS, Huffman, clustering, HybridUint (5.3k lines, 8 files)
│                         Used by: tiny/entropy_code.rs, modular/improved.rs, modular/section.rs
├── bit_writer.rs       — Bitstream writer
├── color/              — XYB conversion
├── container.rs        — EXIF/XMP box wrapping
├── image/              — Image buffer types
├── error.rs            — Error types
├── trace.rs            — Bitstream tracing macros
├── encoder_tests.rs    — 3944 lines of tests (orphaned name from deleted encoder.rs)
├── test_helpers.rs     — Decoder wrappers (jxl-rs, djxl, jxl-oxide)
└── tests/              — baselines.rs
```

## Problems to Fix

### 1. `tiny/` is a terrible name (HIGH PRIORITY)
- 30k lines, the primary VarDCT encoder. "Tiny" comes from the libjxl-tiny origin.
- CLAUDE.md explicitly says "libjxl-tiny was a stepping stone, NOT the reference."
- Should be `vardct/` — that's what JXL calls this encoding mode.
- `TinyEncoder` → `VarDctEncoder`

### 2. Shared code trapped inside `tiny/` (HIGH PRIORITY)
The modular path imports from `tiny/`:
- `modular/improved.rs` → `crate::tiny::entropy_code::{OwnedAnsEntropyCode, build_entropy_code_ans, write_tokens_ans}`
- `modular/section.rs` → `crate::tiny::entropy_code::{...}`
- `modular/tree_learn.rs` → `crate::tiny::token::Token`
- `api.rs` → `crate::tiny::icc_codec::write_icc`

This creates a backwards dependency: the lossless path depends on the lossy module.
The shared pieces should live in a neutral location (entropy_coding/ or a new shared/ module).

Specifically:
- `tiny/entropy_code.rs` (1712 lines) — The unified Huffman+ANS entropy encoder. Both VarDCT
  and modular use it. Should move to `entropy_coding/` or become a shared module.
- `tiny/token.rs` — Token type used everywhere. Should be in `entropy_coding/` or crate root.
- `tiny/icc_codec.rs` — ICC profile encoding. Used by api.rs. Not VarDCT-specific.
- `tiny/lz77.rs` — LZ77 methods. Used by both paths. Re-exported via `api.rs` as `Lz77Method`.

### 3. `frame/` is misleadingly named (MEDIUM)
- Contains only `FrameEncoder` which does modular frame assembly.
- VarDCT frame assembly is in `tiny/encoder.rs` and `tiny/frame.rs`.
- `frame/` implies it's the general frame layer but it's modular-specific.
- Options: fold into `modular/`, or rename to clarify scope.

### 4. `modular/improved.rs` — "improved" over what? (MEDIUM)
- It's the main modular encoding pipeline (2553 lines).
- The name is relative to `modular/minimal.rs` which is dead code.
- Should be `modular/encode.rs` or `modular/pipeline.rs`.

### 5. Dead code in modular/ (LOW — quick cleanup)
- `modular/encoder.rs` — `ModularEncoder`, `ModularEncoderOptions`, `EncodedModularData` —
  not used anywhere outside itself. 279 lines of dead code.
- `modular/minimal.rs` — `write_minimal_modular_stream` — not called anywhere. 382 lines dead.
- These were early implementations superseded by `improved.rs`.

### 6. `encoder_tests.rs` orphaned name (LOW)
- Was `#[path = "encoder_tests.rs"]` inside the now-deleted `encoder.rs`.
- Now `#[path = "encoder_tests.rs"]` in lib.rs.
- Tests the public API. Should be renamed or moved to `tests/`.

### 7. Two token types (LOW — but confusing)
- `tiny/token.rs::Token` — used by both VarDCT and modular (via `crate::tiny::token::Token as AnsToken`)
- `modular/token.rs::Token` — separate type for modular
- The modular code imports tiny's Token as `AnsToken`. This indirection is confusing.

### 8. `frame/frame_encoder.rs` redundant naming (LOW)
- `frame/frame_encoder.rs` inside `frame/` — the "frame" is repeated.
- Could just be `frame/mod.rs` directly (it's the only file) or `frame/encoder.rs`.

## Dependency Graph (current cross-module imports)

```
api.rs ──→ tiny/TinyEncoder (lossy path)
api.rs ──→ frame/FrameEncoder (lossless path)
api.rs ──→ tiny/icc_codec (ICC encoding)
api.rs ──→ tiny/Lz77Method (re-export)

frame/FrameEncoder ──→ modular/improved (write_global/group_modular_section)
frame/FrameEncoder ──→ modular/section (write_global_modular_section_with_tree)
frame/FrameEncoder ──→ modular/channel (ModularImage)
frame/FrameEncoder ──→ headers/ (FrameHeader, ColorEncoding)

modular/improved ──→ tiny/entropy_code (ANS encode/decode) ← WRONG DIRECTION
modular/improved ──→ tiny/token (Token type) ← WRONG DIRECTION
modular/improved ──→ entropy_coding/ (huffman, hybrid_uint)
modular/section ──→ tiny/entropy_code ← WRONG DIRECTION
modular/section ──→ entropy_coding/hybrid_uint
modular/tree_learn ──→ tiny/token ← WRONG DIRECTION

tiny/entropy_code ──→ entropy_coding/ (ANS core, clustering, histogram, ans_decode)
tiny/bitstream ──→ entropy_coding/hybrid_uint
```

## Proposed Target Structure

```
jxl_encoder/src/
├── api.rs              — Public API (unchanged)
├── lib.rs              — Crate root
├── vardct/             — VarDCT (lossy) encoder (renamed from tiny/)
│   ├── encoder.rs      — VarDctEncoder (renamed from TinyEncoder)
│   ├── frame.rs        — VarDCT frame header writing
│   ├── ... (all other tiny/ files except shared ones)
│   └── mod.rs
├── modular/            — Modular (lossless) encoder
│   ├── encode.rs       — Main pipeline (renamed from improved.rs)
│   ├── frame.rs        — Modular frame assembly (moved from frame/frame_encoder.rs)
│   ├── ... (existing files minus dead code)
│   └── mod.rs
├── entropy/            — Unified entropy coding (merged from entropy_coding/ + tiny shared bits)
│   ├── ans.rs          — ANS encoder/decoder
│   ├── huffman.rs      — Huffman tree building (renamed from huffman_tree.rs)
│   ├── encode.rs       — Token→bitstream (moved from tiny/entropy_code.rs)
│   ├── token.rs        — Token type (moved from tiny/token.rs)
│   ├── lz77.rs         — LZ77 methods (moved from tiny/lz77.rs)
│   ├── cluster.rs      — Histogram clustering
│   ├── hybrid_uint.rs  — HybridUint config
│   ├── context_map.rs  — Context map encoding
│   └── mod.rs
├── icc.rs              — ICC profile encoding (moved from tiny/icc_codec.rs)
├── headers/            — File + frame header writing (unchanged)
├── bit_writer.rs       — Bitstream writer (unchanged)
├── color/              — XYB conversion (unchanged)
├── container.rs        — EXIF/XMP (unchanged)
├── image/              — Image buffer types (unchanged)
├── error.rs            — Error types (unchanged)
├── trace.rs            — Tracing macros (unchanged)
└── tests/              — All test files
    ├── api_tests.rs    — Public API tests (renamed from encoder_tests.rs)
    ├── test_helpers.rs — Decoder wrappers
    └── baselines.rs
```

## Suggested Execution Order

1. **Delete dead code** — `modular/encoder.rs`, `modular/minimal.rs` (661 lines, zero risk)
2. **Extract shared code from tiny/** — Move `token.rs`, `entropy_code.rs`, `icc_codec.rs`,
   `lz77.rs` to neutral locations. This fixes the backwards dependency. Mechanically heavy
   (many import path changes) but semantically simple.
3. **Rename `tiny/` → `vardct/`** and `TinyEncoder` → `VarDctEncoder`. grep/sed job.
4. **Move FrameEncoder into modular/** — it only does modular frames.
5. **Rename `improved.rs` → `encode.rs`** — the main modular pipeline.
6. **Rename/move test files** — `encoder_tests.rs` → `tests/api_tests.rs`.

Steps 2-3 are the high-value changes. Steps 1, 4-6 are small cleanups.

## Key Files to Read First

- `jxl_encoder/src/lib.rs` — module declarations
- `jxl_encoder/src/api.rs` — public API, see `encode_lossless()` and `encode_lossy()`
- `jxl_encoder/src/tiny/mod.rs` — tiny module structure and re-exports
- `jxl_encoder/src/modular/mod.rs` — modular module structure and re-exports
- `jxl_encoder/src/frame/frame_encoder.rs` — FrameEncoder (only consumer of modular pipeline)
- `CLAUDE.md` — project instructions, resolved bugs, current status

## Verification After Each Step

```bash
cargo fmt && cargo clippy -- -D warnings && cargo test -p jxl_encoder --lib
cargo test -p jxl_encoder --test hash_lock_features  # 36 byte-exact tests
cargo test -p jxl_encoder_cli
```

## Constraints

- No public API changes (LosslessConfig, LossyConfig, EncodeRequest, PixelLayout unchanged)
- Hash lock tests must remain byte-exact (they test output bytes, not internal paths)
- `#![forbid(unsafe_code)]` must remain
- Commit incrementally — one logical change per commit
- The `tiny::TinyEncoder` is referenced by CLI (`jxl_encoder_cli/src/main.rs`) for the
  rate control path — update that too when renaming
- Integration tests in `jxl_encoder/tests/` reference `jxl_encoder::tiny::TinyEncoder` and
  `jxl_encoder::tiny::dct::` — these need updating when renaming tiny/
