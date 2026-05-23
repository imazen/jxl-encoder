# Tuning Relations

Canonical graph of every tunable VarDCT numeric const + its owners, neighbors,
and shipped-win patterns. Auto-generated from W44-210-A/B/C/D/E audits
(2026-05-22) and maintained going forward per the MANDATORY rule below.

This file complements [`LIBJXL_DIVERGENCES.md`](LIBJXL_DIVERGENCES.md):
- `LIBJXL_DIVERGENCES.md` is the per-divergence ledger (rows = our values vs
  libjxl's values, status, commit refs).
- `TUNING_RELATIONS.md` (this file) captures the COMPOSITION rules between
  rows — which constants share a discriminator, which compose additively,
  which antagonise each other, which were superseded by another.

## MANDATORY maintenance rule

When a W44-* commit adds or changes a tunable numeric constant that is not
spec-required (i.e. has tuning headroom), the commit MUST update this file's
relevant sections AND, if the change introduces a new edge to an existing
const, MUST update the Edges section. Mirror of the
[`LIBJXL_DIVERGENCES.md`](LIBJXL_DIVERGENCES.md) maintenance rule.

What "tunable numeric constant" means:
- File-level `pub const` / `const` in `jxl_encoder/src/**/*.rs` or
  `jxl-encoder-simd/src/cfl.rs`
- `EffortProfile` struct fields with per-effort values
- `EntropyMulTable` per-strategy values
- `gate_registry::strategy_def!` per-strategy values
- Inline numeric literals at gate sites (effort thresholds, distance
  thresholds, mask thresholds, fcbr/m3/edge_density thresholds)

What is OUT of scope (these do NOT trigger updates):
- Spec-mandated values (JXL signature bytes, wire-format constants like
  `BLOCK_DIM`, `GROUP_DIM`, `EPF_DEFAULT_SHARPNESS`, `NUM_DC_CONTEXTS`)
- Buffer sizes / capacity hints / scratch dims
- Test `EXPECTED_HASH` constants
- Pure structural constants (`NUM_VALID_STRATEGIES`, etc.)

Sub-agent prompts spawning code-change chunks MUST include this file in
the "inputs to read FIRST" list AND require updates to the relevant
section(s) before commit.

---

## Table of contents

- [Section 0: Canonical access paths (W44-211)](#section-0-canonical-access-paths-w44-211) — `crate::tuning::*` re-export hub
- [Section 1: Const inventory](#section-1-const-inventory) — 180-entry index from W44-210-A
- [Section 2: Mechanism layers](#section-2-mechanism-layers) — 5 orthogonal composition layers
- [Section 3: Discriminator chains](#section-3-discriminator-chains) — 5 content discriminators + shared-threshold map
- [Section 4: Edges](#section-4-edges) — 58 directed const-interaction edges
- [Section 5: Shipped-win patterns](#section-5-shipped-win-patterns) — repeatable templates
- [Section 6: Cross-arc connections](#section-6-cross-arc-connections) — how W44-N corrected earlier W44-M
- [Section 7: DO NOT (binding for future agents)](#section-7-do-not-binding-for-future-agents)
- [Section 8: Empirical coupling structure (W44-217)](#section-8-empirical-coupling-structure-w44-217) — pointer to `PARAM_INTERACTIONS.md`

---

## Section 0: Canonical access paths (W44-211)

Every tunable in this file is also reachable through the
[`jxl_encoder::tuning`](../jxl-encoder/src/tuning.rs) module, organised
into 14 submodules that mirror the W44-210-A section structure.

| `tuning::<module>` path | source-of-truth file | what's covered |
|---|---|---|
| `tuning::discriminator_thresholds` | `vardct/encoder.rs` + duplicates | per-image content discriminator thresholds (mask/m3/edge_density/fcbr/distance windows) |
| `tuning::entropy_mul_tables` | `effort.rs` | `EntropyMulTable` per-strategy variants |
| `tuning::buttloop` | `vardct/butteraugli_loop.rs` | buttloop QF seed, EPF sharpness seed, adaptive_quant qf pre-scale, kPow / max-increase deviation, terminal-class exclude |
| `tuning::coeff_orders` | `vardct/coeff_order.rs` | order-bucket / permutation-context counts (W44-82 cost-gate constants stay inline) |
| `tuning::epf` | `vardct/epf.rs` | EPF sharpness search constants |
| `tuning::patches` | `vardct/patches.rs` | detection + cost-benefit guards |
| `tuning::splines` | `vardct/splines.rs::detect_params` | spline auto-detection thresholds |
| `tuning::gaborish` | `vardct/gaborish.rs` | gaborish sharpening + adaptive params |
| `tuning::noise` | `vardct/noise.rs` | sensor physics constants |
| `tuning::cfl` | `vardct/chroma_from_luma.rs` + `jxl_simd` re-exports | CfL Newton tuning (4-const subset exposed at simd crate root) |
| `tuning::quant_weights` | `vardct/quant.rs` | parametric DCT quant-weight bands (libjxl-spec; READ-ONLY for sweep, decoder mandated) |
| `tuning::ac_strategy` | `vardct/ac_strategy.rs` | cost-model exponents + channel offsets (libjxl-spec; READ-ONLY for sweep) |
| `tuning::dc_tree` | `vardct/bitstream.rs` | DC tree effort gates |
| `tuning::gates` | `effort.rs` | top-level effort / pixel-count / distance gate constants |
| `tuning::squeeze` | `vardct/encoder.rs` | modular alpha extra-channel squeeze quantizer |
| `tuning::runtime` (opt-in `tuning-override`) | `src/tuning.rs` | runtime override `RuntimeTuning` + `install` / `install_from_postcard_file` for the future sweep runner |

### Shared-value aliases

Two semantic clusters that occurred 4× each in the inventory are exposed
as canonical aliases (with compile-time `const _ : () = assert!(…)`
gates ensuring every original site agrees):

| alias | value | replaces 4 sites |
|---|---|---|
| `tuning::discriminator_thresholds::SMART_ZENJXL_PHOTO_MASK_P25_MIN` | 85.0 | `W44_166_*`, `W44_150_*`, `W44_151_*`, `W44_168_SMOOTH_*` |
| `tuning::discriminator_thresholds::SCREENSHOT_MEDIAN_THRESHOLD` | 95.0 | `CONTENT_AWARE_*`, `buttloop::SCREENSHOT_*`, `W44_168_SCREENSHOT_*`, `splines::SCREENSHOT_*` |

These aliases are intentionally `pub const` (the only NEW `pub const`s
W44-211 added). Sweep runners SHOULD reference these aliases when
expressing the *semantic* threshold; the original per-owner constants
remain for back-compat and per-W44 commit traceability. The
`tuning_drift` golden test (`src/tuning.rs::tests`) protects every
alias against drift from its constituent sites.

### Production-binary safety guarantee

The `tuning` module is purely re-exports. Production source still reads
each const through its source-of-truth path. The `cargo build` artifact
is byte-identical pre-vs-post W44-211. Verified by `hash_lock_features`
36/36 + `strategy_libjxl_hash_locks` 5/5.

### Opt-in `tuning-override` feature

When enabled, `crate::tuning::runtime` exposes a postcard-deserialisable
`RuntimeTuning` struct mirroring the const paths. `install()` is
single-shot per process; `get(|t| t.field)` returns the installed value
or the default. Production builds without the feature pay ZERO cost —
the runtime layer compiles to nothing and consumers read the const
directly so the compiler inlines.

### Downstream consumers (W44-212+)

| consumer | path | reads | enables `tuning-override`? |
|---|---|---|---|
| Production encoder (`cjxl-rs`, library API) | every source file | each const through its source-of-truth path | NO — production binaries pay zero cost for the runtime layer |
| W44-212 [`zenjxl-tuning-runner`](../zenjxl-tuning-runner/) | per-cell sweep worker | calls `tuning::runtime::install_from_postcard_file(<blob>)` once per cell, emits `params_blob` Parquet column | YES (`--features tuning-override`) |

**W44-213 wiring (SHIPPED 2026-05-22)**: the 6 `RuntimeTuning` fields
listed below are now wired through the
[`runtime_or_default!`](../jxl-encoder/src/tuning.rs) macro at every
production consumer site. With `--features tuning-override` disabled
(default for production builds) the macro expands to the const
reference (zero overhead, compiler inlines). With the feature enabled
(sweep-runner builds) the macro calls
[`tuning::runtime::get_or_default`](../jxl-encoder/src/tuning.rs)
which short-circuits to the default const when no override is
installed (single atomic-OnceLock load + branch).

Wired-and-proven fields (W44-213, verified by
[`w44_213_runtime_tuning_wiring`](../jxl-encoder/tests/w44_213_runtime_tuning_wiring.rs)
integration test that installs a non-default override and asserts
encoded bytes change):

| RuntimeTuning field | source-of-truth const | call sites | wired |
|---|---|---|---|
| `smart_zenjxl_photo_mask_p25_min` | W44_168_SMOOTH_MASK_P25_MIN, W44_151_HIGH_MASK_P25_MIN, W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN (= 85.0 in 4 sites) | `vardct/encoder.rs:929, 2832, 2880, 2883, 4417, 4543, 4546` (5 hot-path sites: `w44_168_is_smooth`, W44-151 admit ×2, W44-166 admit ×4) | YES |
| `screenshot_median_threshold` | CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD, butteraugli_loop::SCREENSHOT_MEDIAN_THRESHOLD, W44_168_SCREENSHOT_MEDIAN_MIN (= 95.0 in 3 sites) | `vardct/encoder.rs:929, 2755, 3920, 4277, 5055` (5 hot-path sites: `w44_168_is_smooth`, W22-1 screenshot lift ×2, W44-109 adaptive_quant pre-scale, W39-2 buttloop HIGH-regime classify) | YES |
| `buttloop_default_screenshot_qf_seed_scale` | DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE (= 4.0) | `vardct/butteraugli_loop.rs:1361` (W44-105 buttloop QF seed scale gate) | YES |
| `buttloop_qf_seed_scale_min_distance` | BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE (= 3.5) | `vardct/butteraugli_loop.rs:758, 1348` (W44-107 distance gate × 2: adaptive_quant + buttloop) | YES |
| `adaptive_quant_screenshot_qf_seed_scale_e5_e6` | DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6 (= 2.0) | `vardct/butteraugli_loop.rs:793` (W44-109 per-effort scale) | YES |
| `adaptive_quant_screenshot_qf_seed_scale_e7` | DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7 (= 3.0) | `vardct/butteraugli_loop.rs:790` (W44-109 per-effort scale) | YES |

**Verification**: with the wiring proof test running, doubling
`buttloop_default_screenshot_qf_seed_scale` from 4.0 → 8.0 on a 512×512
screenshot at e8 d=4 produces a +30.85% byte delta vs the default-tuning
baseline. The byte change is FAR above the float-precision drift floor
(~0.001%), proving the production code path reads the runtime override
when the feature is enabled. Hash-locks 36/36 + Libjxl byte-locks 5/5
stay BYTE-IDENTICAL at the `tuning-override` feature default (i.e. no
override installed) — `RuntimeTuning::default()` matches every
source-of-truth const exactly, enforced by the
`tuning::runtime::tests::default_matches_production_consts` unit test.

**Remaining unwired fields**: future `RuntimeTuning` extensions for
other tunables (e.g. `W44_91_M3_COLOURFULNESS_MIN`,
`W44_124_DCT32_KEEP_*`, `LIBJXL_INIT_MUL`, `DEFAULT_CUR_POW_LOW`, the
EPF sharpness thresholds) follow the same wiring pattern: (1) add the
field to `RuntimeTuning` (with serde-default helper + `Default` impl
update), (2) wrap the consumer site with
`crate::runtime_or_default!(const, field)`, (3) update this table.
The W44-213 macro infrastructure is the reusable plumbing.

---

## Section 1: Const inventory

180 distinct tunable values across `vardct/`, `effort.rs`, `api.rs`, and
`jxl-encoder-simd/src/cfl.rs`. Source: W44-210-A inventory + W44-210-C
chronological add/change table. Each row links to its owning W44-* chunk
(see [Section 4](#section-4-edges) for the relation graph) and to its
libjxl-comparison bucket (see [Section 5](#section-5-shipped-win-patterns)
and W44-210-E).

Columns:
- **name** — exact identifier
- **value** — current (post-W44-206) value
- **owner** — most-recent W44-* ticket that touched it
- **libjxl bucket** — SAME / LIBJXL-PARITY-LOCKED / DEVIATED / NOT-IN-LIBJXL (from W44-210-E)
- **room** — tuning headroom: LOCKED / low / medium / high

### 1.1 `vardct/encoder.rs` — per-image content-aware discriminators

| name | value | owner | bucket | room |
|---|---|---|---|---|
| `SQUEEZE_QUALITY_FACTOR_CONST` | 0.35 | — | NOT-IN-LIBJXL | high |
| `SQUEEZE_LUMA_FACTOR_CONST` | 1.1 | — | NOT-IN-LIBJXL | high |
| `SQUEEZE_LUMA_QTABLE` | libjxl tiny table (16) | — | SAME (libjxl tiny) | low |
| `CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD` | 95.0 | W22-1 | NOT-IN-LIBJXL | high |
| `W44_65_DCT_SUPPRESS_MEDIAN_THRESHOLD` | 99.5 | W44-65 | NOT-IN-LIBJXL | high |
| `HIGH_D_PHOTO_SMOOTH_THRESHOLD` | 50.0 | W44-29 | NOT-IN-LIBJXL | high |
| `HIGH_D_PHOTO_W44_91_MASK_UPPER` | 80.0 | W44-91 | NOT-IN-LIBJXL | high |
| `HIGH_D_PHOTO_W44_91_MAX_DISTANCE` | 5.0 | W44-91 | NOT-IN-LIBJXL | high |
| `W44_91_M3_COLOURFULNESS_MIN` | 80.0 | W44-91 | NOT-IN-LIBJXL | high |
| `W44_91_FCBR_MAX` | 0.01 | W44-91 | NOT-IN-LIBJXL | high |
| `W44_96_EDGE_DENSITY_MIN` | 0.7 | W44-96 | NOT-IN-LIBJXL | high |
| `W44_96_FCBR_MAX` | 0.01 | W44-96 | NOT-IN-LIBJXL | high |
| `W44_96_VARIANT_Z_MIN_DISTANCE` | 4.5 | W44-96 | NOT-IN-LIBJXL | high |
| `W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN` | 85.0 | W44-166 | NOT-IN-LIBJXL | high |
| `W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN` | 25.0 | W44-98 | NOT-IN-LIBJXL | high |
| `W44_156_VARIANT_Z_D_HIGH_THRESHOLD` | 5.5 | W44-156 | NOT-IN-LIBJXL | medium |
| `W44_124_DCT32_KEEP_M3_MIN` | 60.0 | W44-124 | NOT-IN-LIBJXL | high |
| `W44_124_DCT32_KEEP_EDGE_DENSITY_MAX` | 0.05 | W44-124 | NOT-IN-LIBJXL | high |
| `W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE` | 1.4 | W44-143 | NOT-IN-LIBJXL | medium |
| `W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE` | 3.5 | W44-135 | NOT-IN-LIBJXL | high |
| `W44_168_SMOOTH_MASK_P25_MIN` | 85.0 | W44-168 | NOT-IN-LIBJXL | high |
| `W44_168_SCREENSHOT_MEDIAN_MIN` | 95.0 | W44-168 | NOT-IN-LIBJXL | high |
| `W44_168_TEXTURED_EDGE_DENSITY_MIN` | 0.5 | W44-168 | NOT-IN-LIBJXL | high |
| `W44_168_TEXTURED_ITERS_AT_E7` | 2 | W44-168 | NOT-IN-LIBJXL | high |
| `W44_169_NARROW_MIN_DISTANCE` | 4.0 | W44-169 | NOT-IN-LIBJXL | high |
| `W44_169_NARROW_MAX_DISTANCE` | 5.0 | W44-169 | NOT-IN-LIBJXL | high |
| `SINGLE_PASS_ENTROPY_SMOOTH_PHOTO_MAX_MEDIAN` | 50.0 | W44-87 | NOT-IN-LIBJXL | high |
| `SINGLE_PASS_ENTROPY_MAX_EFFORT` | 5 | W44-87 | NOT-IN-LIBJXL | high |
| `SINGLE_PASS_ENTROPY_MAX_DISTANCE` | 1.0 | W44-87 | NOT-IN-LIBJXL | high |
| `HIGH_D_PHOTO_MIN_DISTANCE` | 3.0 | W44-78 | NOT-IN-LIBJXL | high |
| `W44_150_PHOTO_W44_117_MASK_P25_MIN` | 85.0 | W44-150 | NOT-IN-LIBJXL | high |
| `W44_150_PHOTO_W44_117_MIN_DISTANCE` | 4.0 | W44-150 | NOT-IN-LIBJXL | high |
| `W44_151_HIGH_MASK_P25_MIN` | 85.0 | W44-151 | NOT-IN-LIBJXL | high |
| `W44_152_W44_151_MIN_DISTANCE` | 3.0 | W44-152 | NOT-IN-LIBJXL | high |
| `W44_152_W44_151_MAX_DISTANCE` | 5.0 | W44-152 | NOT-IN-LIBJXL | high |
| `PATCHES_DISPATCH_BLOCK_MASK_THRESHOLD` | 60.0 | W41-2 | NOT-IN-LIBJXL | high |
| `PIXEL_LOSS_DISPATCH_MEDIAN_THRESHOLD` | 80.0 | W44-90 | NOT-IN-LIBJXL | high |

### 1.2 `vardct/butteraugli_loop.rs` — buttloop + EPF seed + adaptive_quant qf

| name | value | owner | bucket | room |
|---|---|---|---|---|
| `LIBJXL_INIT_MUL` | 0.6 | — | SAME | low |
| `DEFAULT_CUR_POW_LOW` / `_HIGH` | 0.2 / 0.2 | — | SAME | low |
| `DEFAULT_MAX_INCREASE_LOW` / `_HIGH` | 100.0 / 100.0 | — | SAME | low |
| `DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT` | 100.0 | W22-1 | NOT-IN-LIBJXL (kept for future bisect) | high |
| `SCREENSHOT_MEDIAN_THRESHOLD` | 95.0 | W22-1 | NOT-IN-LIBJXL | high |
| `DEFAULT_DISTANCE_SPLIT` | 2.0 | — | SAME | low |
| `DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE` | 4.0 | W44-105 | NOT-IN-LIBJXL | high |
| `BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE` | 3.5 | W44-107 | NOT-IN-LIBJXL | high |
| `BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE` | 2.0 | W44-108 | NOT-IN-LIBJXL | high |
| `BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX` | 30.0 | W44-108 | NOT-IN-LIBJXL | high |
| `W44_176_TERMINAL_CLASS_LUMA_VAR_MIN` | 1500.0 | W44-176 | NOT-IN-LIBJXL | high |
| `W44_176_TERMINAL_CLASS_LUMA_VAR_MAX` | 2200.0 | W44-176 | NOT-IN-LIBJXL | high |
| `W44_176_TERMINAL_CLASS_FCBR_MIN` | 0.70 | W44-176 | NOT-IN-LIBJXL | high |
| `ADAPTIVE_QUANT_QF_SEED_SCALE_MAX_EFFORT` | 7 | W44-109 | NOT-IN-LIBJXL | high |
| `W44_120_EPF_SEED_MIN_DISTANCE` | 1.0 | W44-120 | NOT-IN-LIBJXL | medium |
| `W44_140_EPF_SEED_FADE_MAX` | 1.5 | W44-140 | NOT-IN-LIBJXL | medium |
| `W44_142_EPF_SEED_SUPPRESS_M3_MIN` | 60.0 | W44-142 | NOT-IN-LIBJXL | high |
| `W44_142_EPF_SEED_SUPPRESS_EDGE_DENSITY_MAX` | 0.05 | W44-142 | NOT-IN-LIBJXL | high |
| `W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE` | 1.5 | W44-142 | NOT-IN-LIBJXL | medium |
| `DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6` | 2.0 | W44-109 | NOT-IN-LIBJXL | high |
| `DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7` | 3.0 | W44-109 | NOT-IN-LIBJXL | high |
| `W44_145_PER_BLOCK_MASK_LOW` | 70.0 | W44-145 | NOT-IN-LIBJXL (dead_code, kept for e8+ follow-up) | high |
| `W44_145_PER_BLOCK_MASK_HIGH` | 99.5 | W44-145 | NOT-IN-LIBJXL (dead_code) | high |
| `K_BUTTERAUGLI_ACCEPT_FACTOR` | 1.05 | — | SAME | low |
| `K_TILE_NORM` | 1.2 | — | SAME | low |
| `K_ORIGINAL_COMPARISON_ROUND` | 1 | — | SAME | low |

### 1.3 `vardct/coeff_order.rs` — Lehmer cost model + cost-benefit gate

| name | value | owner | bucket | room |
|---|---|---|---|---|
| `NUM_ORDER_BUCKETS` | 13 | — | SAME (spec) | low |
| `NUM_PERMUTATION_CONTEXTS` | 8 | — | SAME (spec) | low |
| `STRATEGY_TO_BUCKET` | libjxl `kStrategyOrder` | — | SAME (spec) | low |
| inline `log2(size)` end-marker cost | log2(size) | W44-82 | DEVIATED (per-bucket SKIP via W44-201/205 is the fix, not the heuristic) | medium |
| inline `0.5` zero-entry cost | 0.5 | W44-82 | DEVIATED | medium |
| inline `1.5 + log2(v+1)` nonzero cost | 1.5 + log2(v+1) | W44-82 | DEVIATED | medium |
| inline `1 bit/zero` savings estimate | 1 bit/zero (empirical ~0.3-0.5) | W44-82 | DEVIATED | medium |

### 1.4 `vardct/adaptive_quant.rs` — mask1x1 + low-pass mask

All values are bit-exact libjxl-spec; the only "tunable" shape is
`K_MUL_BASE` + `K_MUL_ADD` which the picker could theoretically retune
but doesn't (libjxl calls them "constants" too).

| name | value | owner | bucket | room |
|---|---|---|---|---|
| `K_MUL_BASE` | `[0.125, 0.1, 0.09, 0.06]` | — | SAME | low |
| `K_MUL_ADD` | `[0.0, -0.1, -0.09, -0.06]` | — | SAME | low |
| `K_TOTAL` | 0.29959705784054957 | — | SAME | low |
| `K_MUL` (reciprocal numerator) | 1.0 | — | SAME | low |
| `K_OFFSET` (reciprocal denominator) | 0.001 | — | SAME | low |
| `W_R`/`W_D`/`W_R2`/`W_L`/`W_D2` (kFilterMask1x1) | spec | — | SAME (spec) | low |
| inline `K_AC_QUANT = 0.765` | 0.765 | — | SAME | low |

### 1.5 `vardct/ac_strategy.rs` — EstimateEntropy cost model

| name | value | owner | bucket | room |
|---|---|---|---|---|
| `MASK_CHANNEL_OFFSET` | `[12.0, 0.0, 4.0]` | — | SAME (spec) | low |
| `CHANNEL_MUL` | `[2.088e7, 1.0, 1.267]` | — | SAME (spec) | low |
| `K_BIAS` | 0.13731743 | — | SAME (spec) | low |
| `K_POW_INFO_LOSS` | 0.33677807 | — | SAME (spec) | low |
| `K_POW_ZEROS_MUL` | 0.5099093 | — | SAME (spec) | low |
| `K_POW_COST_DELTA` | 0.3670294 | — | SAME (spec) | low |
| `COEFF_DOMAIN_CONSTANTS` | `(138.0, 5.3359, 7.5651)` | — | SAME (libjxl-tiny) | low |
| `K_INFO_LOSS_MULTIPLIER2` | 50.4684 | — | **NOT-IN-LIBJXL** | high |
| `K_COST2` | 4.462815 | — | **NOT-IN-LIBJXL** | high |
| `K_LIMIT` / `K_MUL` (X-channel penalty) | 1.54138 / 0.56391 | — | SAME | low |

### 1.6 `vardct/ac_strategy_search.rs` — per-strategy `(mul1, mul2, base)` cost triples

All currently match libjxl exactly. The 3-copy duplication of the 32×32 /
32×16 triples is a refactor candidate (hoist to `EffortProfile`).

| name | value | owner | bucket | room |
|---|---|---|---|---|
| `k8x8mul1/mul2/base` (pixel-domain) | -0.4 / 1.0 / 1.4 | — | SAME (spec) | low |
| `profile.k8x8` (coef-domain legacy, `pixel_domain_loss=false` only) | (-0.4125, 0.8052, 1.4) | — | DEVIATED (legacy path) | medium |
| `k8x16/profile.k16x8` | -0.55 / 0.902 / 1.6 | — | SAME / DEVIATED | low / medium |
| `k16x16/profile.k16x16` | -0.65 / 0.88 / 1.8 | — | SAME / DEVIATED | low / medium |
| `k32x32mul1/mul2/base` (3 copies) | -0.75 / 1.2 / 2.0 | — | SAME | low |
| `k32x16mul1/mul2/base` (3 copies) | -0.70 / 1.1 / 2.0 | — | SAME | low |
| `k64x64mul1/mul2/base` | -0.80 / 1.3 / 2.5 | — | SAME | low |
| `k64x32mul1/mul2/base` (2 copies) | -0.75 / 1.2 / 2.5 | — | SAME | low |

### 1.7 `vardct/quantize.rs` — AdjustQuantBlockAC dead-zones + bias

| name | value | owner | bucket | room |
|---|---|---|---|---|
| `BIAS` (`kDefaultQuantBias`) | `[0.9453, 0.9299, 0.9501, 0.145]` | — | SAME (spec) | low |
| `QUANT_MAX` | 256 | — | SAME (spec) | low |
| `K_LIMIT` (per-quadrant dead-zone limit) | `[0.46; 4]` | — | SAME | low |
| `K_MUL` (per-quadrant dead-zone mul) | `[0.9999; 4]` | — | SAME | low |
| `K_MUL1` / `K_MUL2` (per-(quadrant, channel)) | libjxl table | — | SAME | low |
| `K_QUANT_NORMALIZER` | 2.294270834328472 | — | SAME | low |
| `ERROR_DIFFUSION_FACTOR` | 0.25 | — | DEVIATED (opt-in only; libjxl is no-op) | medium |
| `Y` thresholds | `[0.56, 0.62, 0.62, 0.62]` | — | SAME | low |
| `X/B` thresholds | `[0.58, 0.62, 0.62, 0.62]` | — | SAME | low |

### 1.8 `vardct/epf.rs`, `vardct/dot_detection.rs`, `vardct/frame.rs`, `vardct/quant.rs`, `vardct/chroma_from_luma.rs`

All bit-exact libjxl-spec apart from W37-2's `EPF_AUTO_SMOOTH_MASK_THRESHOLD = 60.0`
(auto-EPF dispatch on smooth content) and `K_FAVOR_NO_SMOOTHING = 0.99` (EPF
cost-model multiplier — likely OURS, no libjxl reference found).

| name | value | owner | bucket | room |
|---|---|---|---|---|
| `EPF_AUTO_SMOOTH_MASK_THRESHOLD` | 60.0 | W37-2 | NOT-IN-LIBJXL | high |
| `K_FAVOR_NO_SMOOTHING` | 0.99 | — | NOT-IN-LIBJXL (likely) | high |
| `DC_QUANT_POW` / `_QUANT` / `_MUL` | 0.83 / 1.095924 / 0.3 | — | SAME (spec) | low |
| `K_INV_COLOR_FACTOR` (CfL) | 1.0/84.0 | — | SAME (spec) | low |
| `CFL_FIXED_POINT_PRECISION` | 11 | — | SAME (spec) | low |
| `DEFAULT_COLOR_FACTOR` | 84 | — | SAME (spec) | low |
| `JPEG_CFL_ZERO_BIAS_DEFAULT` | `[0.5; 3]` | — | SAME (spec) | low |
| `INV_DC_QUANT` | `[4096.0, 512.0, 256.0]` | W44-8 (distance-derived for patches) | SAME (spec) | low |
| `DCT*_PARAMS` / `DCT*_BAND_PARAMS` / `AFV_*` / `IDENTITY_WEIGHTS` / `DCT2_WEIGHTS` | libjxl spec | — | SAME (spec, decoder-locked) | LOCKED |

### 1.9 `vardct/bitstream.rs` — DC tree gates

| name | value | owner | bucket | room |
|---|---|---|---|---|
| `DC_TREE_VARIABLE_TRIAL_MIN_EFFORT` | 8 | W44-171 | SAME (post-W44-171) | low |
| `DC_TREE_VARIABLE_PREDICTOR_FULL_MIN_EFFORT` | 9 | W44-172 | SAME (post-W44-172) | low |

### 1.10 `vardct/patches.rs` + `vardct/splines.rs` + `vardct/noise.rs` + `vardct/gaborish.rs`

Patches: 14 file-level + 4 fn-local constants; most spec-matched. The four
`SAVINGS_BYTES_PER_PIXEL*` / `SAFETY_MULTIPLIER` / `SAFETY_DIVISOR` are
imazen-tuned cost-benefit guards (NOT-IN-LIBJXL, libjxl uses a fixed
`if (n_patches >= N) admit` gate).

Splines: opt-in via `LossyConfig::with_splines()`; 10 NOT-IN-LIBJXL
discriminator + sigma constants. Sole fully picker-exposed const set in
the encoder.

Noise: opt-in via `--noise`; libjxl-spec camera-sensor constants.

Gaborish: `K_GABORISH` spec kernel + 5 adaptive-gaborish fn-locals (LOW/HIGH
mask thresholds, MIN_MUL/MAX_MUL multiplier range).

| name | value | owner | bucket | room |
|---|---|---|---|---|
| `SAVINGS_BYTES_PER_PIXEL_LOSSLESS` (patches) | 0.35 | — | NOT-IN-LIBJXL | high |
| `SAVINGS_BYTES_PER_PIXEL` (patches lossy) | 0.78 | — | NOT-IN-LIBJXL | high |
| `SAFETY_MULTIPLIER` / `SAFETY_DIVISOR` (1.5×) | 3 / 2 | — | NOT-IN-LIBJXL | high |
| `splines::MIN_GRAD_MAG` | 0.15 | — | NOT-IN-LIBJXL | high |
| `splines::MIN_EIG_RATIO` | 5.0 | — | NOT-IN-LIBJXL | high |
| `splines::MIN_POLYLINE_LEN` / `MAX_POLYLINE_LEN` | 32 / 1024 | — | NOT-IN-LIBJXL | high |
| `splines::TARGET_CONTROL_POINTS` | 8 | — | NOT-IN-LIBJXL | high |
| `splines::MAX_SPLINES` | 64 | — | NOT-IN-LIBJXL | high |
| `splines::INIT_SIGMA` / `SIGMA_MIN` / `SIGMA_MAX` | 1.0 / 0.6 / 4.0 | — | NOT-IN-LIBJXL | high |
| `splines::COST_BENEFIT_MARGIN` | 2.0 | — | NOT-IN-LIBJXL | high |
| `SCREENSHOT_MEDIAN_MASK_THRESHOLD` (splines) | 95.0 | — | NOT-IN-LIBJXL | high |

### 1.11 `jxl-encoder-simd/src/cfl.rs` — Newton CfL tuning (Libjxl-parity locked)

| name | default | Libjxl-strategy | owner | bucket | room |
|---|---|---|---|---|---|
| `NEWTON_EPS_DEFAULT` | 1.0 | 100.0 (`NEWTON_EPS_LIBJXL`) | W44-184 | LIBJXL-PARITY-LOCKED | LOCKED |
| `NEWTON_MAX_ITERS_DEFAULT` | 10 | 20 (`NEWTON_MAX_ITERS_LIBJXL`) | W44-184 | LIBJXL-PARITY-LOCKED | LOCKED |
| Newton starting `x` | `ls_x` (warm-start) | 0 | W44-184 | LIBJXL-PARITY-LOCKED | LOCKED |
| Newton fallback | `ls_x` | last `x` | W44-184 | LIBJXL-PARITY-LOCKED | LOCKED |
| `TOWARDS_ZERO` | 2.6 | 2.6 | — | SAME | low |
| `NEWTON_CLAMP` | 20.0 | 20.0 | — | SAME | low |
| `NEWTON_COEFF` | 1.0/3.0 | 1.0/3.0 | — | SAME | low |
| `NEWTON_THRES` | 100.0 | 100.0 | — | SAME | low |
| `NEWTON_STABILIZER` | 0.85 | 0.85 | — | SAME | low |
| `NEWTON_CONVERGENCE` | 3e-3 | 3e-3 | — | SAME | low |

### 1.12 `effort.rs` — picker thresholds + `EffortProfile` fields

Top-level picker thresholds:

| name | value | owner | bucket | room |
|---|---|---|---|---|
| `SMALL_IMAGE_PIXEL_THRESHOLD` | 1_000_000 | — | NOT-IN-LIBJXL | high |
| `LARGE_IMAGE_PIXEL_THRESHOLD` | 4_000_000 | — | NOT-IN-LIBJXL | high |
| `LARGE_E9_TREE_MAX_BUCKETS` | 192 | — | DEVIATED (perf wall workaround) | medium |
| `LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD` | 500_000 | W44-34 | NOT-IN-LIBJXL | high |
| `LOSSY_LOW_DISTANCE_THRESHOLD` | 2.0 | W44-34 | NOT-IN-LIBJXL | high |
| `CONTENT_CLASS_MIN_PIXELS` | 65_536 | W44-164 | NOT-IN-LIBJXL | high |

`EffortProfile` per-effort fields (~30 fields × 5 effort tiers); all are
picker-tunable but currently match libjxl per-effort defaults. Key
cost-model fields:

| field | type | per-effort | bucket | room |
|---|---|---|---|---|
| `k_info_loss_mul_base` | f32 | 1.2 (`lossy_experimental` = 1.3) | SAME | low |
| `k_zeros_mul_base` | f32 | 9.309 | SAME | low |
| `k_cost_delta_base` | f32 | 10.833 | SAME | low |
| `k_ac_quant` | f32 | 0.765 | SAME | low |
| `k_favor_2x2` | f32 | -0.4 | SAME | low |
| `k_avoid_transforms_base` | f32 | 0.5 | SAME | low |
| `initial_q_numerator` | f32 | 0.39 / 0.79 | SAME | low |
| `entropy_mul_table` | `EntropyMulTable` | per-strategy variants | mixed (see §1.13) | mixed |
| `tree_threshold_base` | f32 | `75 + 14·speed_tier` | DEVIATED | medium |
| `tree_max_samples_fixed` (e≤4) | u32 | 65_000 | DEVIATED | medium |
| `tree_num_properties` | u8 | 3..16 per effort | SAME | low |
| `tree_max_buckets` | u16 | 32..256 per effort | SAME | low |
| `tree_sample_fraction` | f32 | 0.15..0.65 per effort | SAME | low |
| `nb_rcts_to_try` | u8 | 0..19 per effort | SAME | low |
| `wp_num_param_sets` | u8 | 0 / 2 / 5 per effort | SAME | low |
| `cfl_pass2_ls_at_low_effort` | bool | false (true under Libjxl) | LIBJXL-PARITY-LOCKED | LOCKED |

### 1.13 `EntropyMulTable` variants

Per-strategy entropy multipliers. `reference()` matches libjxl exactly.
8 variant tables ship under `EncoderStrategy::Zenjxl/Aggressive`; `Libjxl`
strategy always resolves to `reference()`.

| table fn | dct16x16 | dct32x32 | dct16x32 | gate | owner |
|---|---|---|---|---|---|
| `reference` | 1.34 | 1.48 | 1.49 | always | — |
| `screenshot_suppressed` | (ref) | (ref) | (ref) | identity=1.85, dct2x2=1.15, afv=0.95, dct4x8=0.98 | W22-1 |
| `high_d_photo_smooth_suppressed` | 1.27 | 1.34 | 1.35 (scaled) | W44-29: mask<50 AND d>=3 | W44-29 |
| `high_d_photo_smooth_suppressed_z` | 1.27 | 1.22 | 1.228 (scaled) | W44-96: + ed>=0.7, fcbr<0.01, d>=4.5 | W44-148/154 |
| `..._z_high_colour` | 1.27 | 1.22 | 1.30 (decoupled) | W44-98: + m3>=25 | W44-98 |
| `..._z_low_colour` | 1.27 | 1.22 | 1.23 | W44-99: + m3<25 | W44-100 |
| `..._z_d_high` | 1.27 | 1.20 (back to pre-W44-148) | 1.208 (scaled) | W44-156: + d>5.5 | W44-156 |
| `..._z_high_colour_d_high` | 1.27 | 1.20 | 1.30 | W44-156 + m3>=25 | W44-156 |
| `..._z_low_colour_d_high` | 1.27 | 1.20 | 1.23 | W44-156 + m3<25 | W44-156 |
| `experimental` (PR #4506) | (ref) | (ref) | (ref) | dct4x4=0.88, identity=0.88, afv=0.75 | — |

### 1.14 `gate_registry.rs` — per-strategy gate values

24 macro-generated gates × 4 strategies = 96 values. See
[`docs/LIBJXL_DIVERGENCES.md`](LIBJXL_DIVERGENCES.md) for the canonical
per-strategy values. Summary:

- `Libjxl`: bit-for-bit cjxl mirror across Sections A+B+D
- `LeanFaster`: fast preset, skips per-image discriminators
- `Zenjxl` (DEFAULT): all content-aware lifts ON
- `Aggressive`: same as Zenjxl plus admit-more deltas (rare)

Env-hookable for A/B (4 of 24 gates):
- `JXL_BUTTLOOP_INITIAL_QF_SCALE` (W44-105 buttloop_qf_seed)
- `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE` (W44-109 adaptive_quant_qf_seed)
- `JXL_W44_117_DISABLE` / `JXL_W44_120_EPF_SEED_MIN_DISTANCE` (W44-117 buttloop_epf_sharpness_seed)
- `JXL_W44_184_FORCE_LIBJXL_NEWTON` (W44-184 cfl_newton_libjxl_parity)
- `JXL_W44_201_DISABLE_BUCKETS` (W44-201 coeff_orders_disable_large_buckets)
- `JXL_W44_205_FORCE_LEGACY_MEDIUM_BUCKETS` (W44-205 coeff_orders_disable_medium_buckets)

### 1.15 Cross-section shared-value clusters

Multiple constants share identical values. These represent (a) intentional
re-use of a discriminator threshold (e.g. mask_p25=85 across W44-149/150/151/166/168)
or (b) coincidental same-value. Refactor candidates noted.

| value | members | refactor? |
|---|---|---|
| `mask1x1_p25 = 85.0` | `W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN`, `W44_150_PHOTO_W44_117_MASK_P25_MIN`, `W44_151_HIGH_MASK_P25_MIN`, `W44_168_SMOOTH_MASK_P25_MIN` | **YES** — hoist to single `SMART_ZENJXL_PHOTO_MASK_P25_MIN` |
| `mask1x1_median = 95.0` | `CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD`, `BUTTLOOP::SCREENSHOT_MEDIAN_THRESHOLD`, `W44_168_SCREENSHOT_MEDIAN_MIN`, `splines::SCREENSHOT_MEDIAN_MASK_THRESHOLD` | **YES** — hoist to single shared const |
| `mask1x1_median = 50.0` | `HIGH_D_PHOTO_SMOOTH_THRESHOLD`, `SINGLE_PASS_ENTROPY_SMOOTH_PHOTO_MAX_MEDIAN` | conditional — verify both consumers want the same value |
| `mask1x1_median = 60.0` | `EPF_AUTO_SMOOTH_MASK_THRESHOLD`, `PATCHES_DISPATCH_BLOCK_MASK_THRESHOLD` (also `m3 = 60.0` in `W44_124_DCT32_KEEP_M3_MIN`, `W44_142_EPF_SEED_SUPPRESS_M3_MIN`) | NO — mask vs m3 semantics differ; coincidence |
| `m3 = 60.0` | `W44_124_DCT32_KEEP_M3_MIN`, `W44_142_EPF_SEED_SUPPRESS_M3_MIN` | already explicit (W44-142 cites W44-124) |
| `m3 = 25.0` | `W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN` (used as both HC admit AND LC reject splitter) | already correctly shared |
| `edge_density = 0.05` | `W44_124_DCT32_KEEP_EDGE_DENSITY_MAX`, `W44_142_EPF_SEED_SUPPRESS_EDGE_DENSITY_MAX` | already explicit (W44-142 cites W44-124) |
| `fcbr = 0.01` | `W44_91_FCBR_MAX`, `W44_96_FCBR_MAX` | already explicit (W44-96 inherits W44-91) |

### 1.16 3-copy const families that should hoist to EffortProfile

These cost-model triples are inline-duplicated in 2-3 places in
`ac_strategy_search.rs`. Hoisting to `EffortProfile` lets the picker tune
them per-content-class without touching production sites.

| triple | inline copies | proposed EffortProfile slot |
|---|---|---|
| `k32x32mul1/mul2/base` | 3 copies | `EffortProfile.k32x32` (NEW) |
| `k32x16mul1/mul2/base` | 3 copies | `EffortProfile.k32x16` (NEW) |
| `k64x32mul1/mul2/base` | 2 copies | `EffortProfile.k64x32` (NEW) |

---

## Section 2: Mechanism layers

The variant Z arc (W44-148→W44-166) discovered the encoder has 5
orthogonal mechanism layers. Composition rules:
- Two gates on the SAME layer COMPETE — require a discriminator that
  strictly partitions firing.
- Two gates on DIFFERENT layers can COMPOSE additively — but require
  explicit measurement (W44-119 chain-disable A/B pattern).

### Layer 1: Adaptive-quant qf scale

- **Consts**:
  - `DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE = 4.0` (W44-105, e>=8)
  - `BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE = 3.5` (W44-107)
  - `BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE = 2.0` (W44-108)
  - `BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX = 30.0` (W44-108)
  - `W44_176_TERMINAL_CLASS_*` (W44-176)
  - `DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6 = 2.0` (W44-109)
  - `DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7 = 3.0` (W44-109)
  - `ADAPTIVE_QUANT_QF_SEED_SCALE_MAX_EFFORT = 7` (W44-109)
- **Discriminator**: `is_screenshot` (`median(mask1x1) > 95`) + distance + m3
- **Mechanism**: scales the initial quant_field_float seed before buttloop
  (e>=8 via W44-105) or before adaptive_quant materialization (e<8 via W44-109).
  Same fix-class, different pipeline stages.
- **Known interactions**:
  - ADDITIVE with Layer 4 (W44-117 EPF seed) — both target buttloop recon
    bias, address orthogonal mechanisms. W44-119 chain-disable A/B verified
    chain CANNOT be retired (SSIM2 -1.85 to -5.58 on every screen cell).
  - REDUNDANT with future W44-138 Phase-2 buttloop-recon root-cause fix.
- **DO NOT**: remove the chain without W44-138 Phase-2 fix.

### Layer 2: Outer entropy_mul table

- **Consts**: `EntropyMulTable::high_d_photo_smooth_suppressed` (W44-29)
- **Discriminator**: `mask1x1_median < 50` AND `distance >= 3` (W44-78
  widened from 4.0 → 3.0)
- **Mechanism**: swaps `EffortProfile.entropy_mul_table` from `reference()`
  to a content-suppressed variant for high-distance smooth photos.
- **Known interactions**:
  - COMPOSES with Layer 3 (inner variant Z) — different table layer
  - ADDITIVE with W44-91/96 sub-discriminators that escalate to variant Z
- **DO NOT**: widen outer admission past mask<50 without a zenanalyze
  sub-discriminator (W44-91/96/151).

### Layer 3: Inner variant Z entropy_mul table

- **Consts**: `..._z` / `..._z_high_colour` / `..._z_low_colour` /
  `..._z_d_high` / `..._z_high_colour_d_high` / `..._z_low_colour_d_high`
  (W44-96/98/99/100/148/154/156)
- **Discriminator**: nested inside Layer 2 firing
  - `edge_density >= 0.7 AND fcbr < 0.01 AND distance >= 4.5` (W44-96)
  - `m3 >= 25` → high_colour (W44-98); `m3 < 25` → low_colour (W44-99)
  - `distance > 5.5` → d_high variant (W44-156)
- **Mechanism**: swaps to variant Z table inside Layer 2 dispatch when the
  nested predicate fires.
- **Known interactions**:
  - COMPOSES with Layer 2 (orthogonal — outer cost-model table)
  - ADDITIVE with W44-152 OUTER admit (mask_p25>=85)
  - ANTAGONIST with W44-150 (Layer 4 EPF seed at same content class —
    mechanisms competed; W44-165 honest-stop replaced by W44-166)
- **DO NOT**: re-bisect variant Z dct32x32 outside {1.20, 1.22} — W44-148/154/156
  measurement is conclusive.

### Layer 4: Recon-signal EPF seed

- **Consts**:
  - `W44_120_EPF_SEED_MIN_DISTANCE = 1.0` (W44-120)
  - `W44_140_EPF_SEED_FADE_MAX = 1.5` (W44-140)
  - `W44_142_EPF_SEED_SUPPRESS_M3_MIN = 60.0` (W44-142)
  - `W44_142_EPF_SEED_SUPPRESS_EDGE_DENSITY_MAX = 0.05` (W44-142)
  - `W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE = 1.5` (W44-142)
- **Discriminator**: `is_screenshot` (W44-118) + distance >= 1.0 (W44-120)
  + per-block fade in [1.0, 1.5] (W44-140) + codec_wiki suppress (W44-142
  reuses W44-124 m3+ed predicate)
- **Mechanism**: seeds buttloop's EPF sharpness map with the production-
  computed map instead of uniform-4, eliminating buttloop-recon-vs-decoder
  divergence.
- **Known interactions**:
  - ADDITIVE with Layer 1 qac chain (W44-119 verified)
  - ANTAGONIST with W44-150 mask_p25>=85 photo-admit (mechanism class
    wrong — only recovers ~10% of 1418519 cluster deficit; W44-166
    replaced with Layer 3 variant Z mechanism)
- **DO NOT**: re-attempt Mechanism A admission via W44-117 alone (W44-150).

### Layer 5: Buttloop iteration schedule

- **Consts**:
  - `W44_168_SMOOTH_MASK_P25_MIN = 85.0` (W44-168 SmoothSkip)
  - `W44_168_SCREENSHOT_MEDIAN_MIN = 95.0`
  - `W44_168_TEXTURED_EDGE_DENSITY_MIN = 0.5` (dead_code, env-only)
  - `W44_168_TEXTURED_ITERS_AT_E7 = 2` (dead_code)
  - `W44_169_NARROW_MIN_DISTANCE = 4.0` / `_MAX_DISTANCE = 5.0` (W44-169)
- **Discriminator**: `is_smooth` (mask>95 OR mask_p25>=85) + narrow
  distance band [4.0, 5.0]
- **Mechanism**: decrement butteraugli_iters at e>=8 on smooth photos in
  narrow d-band. Cuts wall time on 1418519 e8 d=4/5 by 6-8% without
  SSIM2 cost.
- **Known interactions**:
  - ANTAGONIST with W44-166 (Layer 3) on 1418519 d=6 — destroyed +0.45
    SSIM2 win in W44-168 broad mode; W44-169 narrow band salvaged
  - CO-GATED with W44-166 (gate excludes d=6 where W44-166 ships its win)

---

## Section 3: Discriminator chains

5 distinct content discriminators. Many constants share the same
discriminator, creating implicit chains where tuning one threshold
affects every downstream constant gated on it.

### Chain 1: mask1x1 (median / p25 / p10) — "smooth content" axis

| threshold | sites |
|---|---|
| `median(mask1x1) > 95` | W22-1 screenshot_lift_hint, W44-118 is_screenshot, W44-168 smooth |
| `median(mask1x1) > 99.5` | W44-65 try_dct64=false, W44-68 try_dct32=false |
| `median(mask1x1) < 50` | W44-29 high_d_photo_smooth_suppressed, W44-87 single_pass_entropy |
| `median(mask1x1) in [50, 80)` | W44-91 widen to ambiguous band (needs zenanalyze sub-discriminator) |
| `median(mask1x1) < 60` | W37-2 EPF auto-skip, W41-2 patches dispatch |
| `mask_p25 >= 85` | W44-149/150/151/152/166/168 — all 1418519-class photo admit |

**Tuning interaction**: lowering `W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE`
(W44-143 = 1.4) interacts with `W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE`
(=1.5) because both gates target codec_wiki at low-d. Raising W44-118's
`SCREENSHOT_MEDIAN_THRESHOLD` past 95 simultaneously disables W44-105/107/108/109/117/120/140/142
chain on all borderline content.

### Chain 2: m3_colourfulness — "Hasler-Süsstrunk M3"

| threshold | sites |
|---|---|
| `m3 >= 80` | W44-91 photo admit (1189261-class) |
| `m3 >= 60` | W44-124 dct32_keep auto, W44-142 EPF SUPPRESS (codec_wiki class) |
| `m3 < 50` | W44-163 B2 PixelLossDispatch nested (proposed) |
| `m3 < 30` | W44-108 W44-105 sub-discriminator (codec_wiki reject) |
| `m3 >= 25` | W44-98 variant Z' HIGH-colour |
| `m3 < 25` | W44-99/100 variant Z'' LOW-colour |

**Tuning interaction**: lowering W44-98 threshold (=25.0) would admit
1531677 (m3=12.30) which W44-99 was specifically introduced to gate AWAY
(because dct16x32=1.30 lift regresses 1531677 SSIM2 by -0.34 to -0.93).
Raising W44-124 threshold (=60.0) past 67 would exclude imessage
admission (designed boundary).

### Chain 3: edge_density — "Sobel luma gradient density"

| threshold | sites |
|---|---|
| `edge_density >= 0.7` | W44-96 variant Z (1420710/1531677-class) |
| `edge_density >= 0.5` | W44-168 textured (dead_code) |
| `edge_density < 0.05` | W44-124 dct32_keep, W44-142 EPF SUPPRESS |
| `edge_density < 0.15` | W44-35 smooth_photo_dct64 (legacy) |

**Tuning interaction**: W44-124 and W44-142 are HARD-LINKED by the
ed<0.05 threshold (W44-142 memo cites W44-124 directly). Changing one
without the other risks codec_wiki cluster discrimination.

### Chain 4: fcbr (flat_color_block_ratio) — "uniformity of 8x8 blocks"

| threshold | sites |
|---|---|
| `fcbr >= 0.35` | W44-164 auto-classify Screenshot |
| `fcbr < 0.10` | W44-164 auto-classify Photo |
| `fcbr < 0.01` | W44-91, W44-96 (nested photo admit) |
| `fcbr < 0.60` | W44-35 smooth_photo_dct64 flat_solid_max (legacy) |
| `fcbr >= 0.70` | W44-176 terminal_class match |

**Tuning interaction**: W44-91 and W44-96 fcbr threshold (=0.01) is
load-bearing — raising past 0.05 admits 297394 (fcbr=0.096) which sits
at the boundary of the photo-vs-screenshot split.

### Chain 5: ZenanalyzeProxies (composite proxy struct)

Introduced by W44-91, extended by W44-96/108/124/142/149/166/168/176.
Carries (m3_colourfulness, flat_color_block_ratio, edge_density,
luma_var) to every gate site. Computed once at encode entry on 8-bit
sRGB layouts (Rgb8/Rgba8/Bgr8/Bgra8).

**Coupling**: any new content-aware gate inherits the streaming/animation
gap (proxies need raw sRGB u8 source bytes, not in scope on streaming
`LossyEncoder` or per-frame animation paths). Callers needing the lift
on those paths must set the corresponding hint explicitly via
`LossyConfig::with_*_hint`.

---

## Section 4: Edges

58 directed const-interaction edges captured from W44-210-D scan of 141
memos. Edge types:

- **ADDITIVE** — A and B both improve same axis on same cells; combined
  exceeds (or matches) sum of either alone.
- **REDUNDANT** — B subsumes A. If B fires, A is a no-op.
- **ANTAGONIST** — A regresses what B fixed (or vice versa). Cannot ship
  both without a discriminator gating one out.
- **CO-GATED** — A only fires when B's predicate is true (or false).
- **SHARED-MECHANISM** — A and B target the same code path (entropy_mul
  table, EPF seed, qac scale) but operate on different dimensions.
- **CONTENT-COUPLED** — A and B both predicate on the same content
  discriminator. Tuning one threshold affects the other.
- **SUPERSEDED** — A was the intended fix; B replaced it after measurement.

### 4.1 Layer 1 (adaptive-quant qf) chain

| A | B | type | mechanism |
|---|---|---|---|
| W44-105 buttloop_qac_seed_scale (×4 at d>=2) | W44-107 BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE (=3.5) | CO-GATED | Raises distance gate from d>=2 → d>=3.5 to close codec_wiki e8 d=3 regression |
| W44-105 buttloop_qac_seed_scale | W44-108 m3<30 sub-discriminator | CO-GATED | Nests m3<30 inside W44-107's distance gate to recover 8 W44-105 d=2..3 wins |
| W44-105 (e>=8 only) | W44-109 adaptive_quant_qf_seed_scale (e5/e6/e7) | SHARED-MECHANISM | Both apply screenshot-class qac scaling at different pipeline stages |
| W44-105/107/108/109 qac chain | W44-117 EPF sharpness seed | ADDITIVE | Chain-disable A/B regresses every screen cell -1.85..-5.58 SSIM2 |
| W44-105/107/108/109 qac chain | W44-138 buttloop-recon root-cause fix (FUTURE) | REDUNDANT | If Phase-2 production fix lands, chain becomes obsolete |
| W44-108 m3<30 sub-discriminator | W44-176 LUMA_VAR + FCBR terminal exclude | SHARED-MECHANISM | W44-176 nests terminal-class sub-exclude inside W44-108's m3 sub-gate |

### 4.2 Layer 4 (EPF seed) chain

| A | B | type | mechanism |
|---|---|---|---|
| W44-117 EPF sharpness seed | W44-118 is_screenshot gate (median>95) | CO-GATED | W44-117 admits ONLY when is_screenshot=true (avoids 1025469 photo regression -0.85 SSIM2) |
| W44-117 | W44-120 EPF_SEED_MIN_DISTANCE (=1.0) | CO-GATED | Lower-distance gate closes terminal d=0.8 over-correction -1.87 SSIM2 |
| W44-117 | W44-140 EPF_SEED_FADE band (d in [1.0, 1.5]) | SHARED-MECHANISM | Linear blend of seed vs uniform-4 in narrow band |
| W44-117 | W44-142 codec_wiki SUPPRESS sub-gate (m3>=60 AND ed<0.05 AND d<1.5) | CO-GATED | Reuses W44-124's exact m3+ed predicate; closes codec_wiki e9 d=1.2 -0.599 SSIM2 |
| W44-117 EPF seed | W44-150 mask_p25>=85 photo-admit | ANTAGONIST | Pairing only recovers ~10% of 1418519 deficit; W44-166 ships W44-148 variant Z instead |

### 4.3 Layers 2+3 (outer + inner entropy_mul) chain

| A | B | type | mechanism |
|---|---|---|---|
| W44-29 outer entropy_mul lift | W44-91 zenanalyze auto-dispatch (m3>=80 AND fcbr<0.01) | CO-GATED | Wires zenanalyze proxies into W44-29's outer gate for 1189261-class photos |
| W44-29 W44_29_lower gate | W44-96 variant Z (high_d_photo_smooth_suppressed_z) | CO-GATED | Sub-discriminator (ed>=0.7 AND fcbr<0.01 AND d>=4.5) inside W44-29 |
| W44-96 variant Z | W44-98 variant Z' high_colour (m3>=25, dct16x32=1.30) | CO-GATED | m3>=25 escalates to high_colour variant Z' (1420710 m3=32.93) |
| W44-96 variant Z | W44-99 variant Z'' low_colour (m3<25, dct16x32=1.22) | CO-GATED | Mirror of W44-98 — m3<25 keeps milder dct16x32=1.22 for 1531677 (m3=12.30) |
| W44-98 variant Z' (HC m3>=25) | W44-99 variant Z'' (LC m3<25) | SHARED-MECHANISM | Two sibling tables splitting variant Z on m3 |
| W44-99 variant Z'' (dct16x32=1.22) | W44-100 variant Z'' micro-bisect (dct16x32=1.23) | SUPERSEDED | Last 1531677 e5 d=5 OPEN cell closer |
| W44-29 outer | W44-148 variant Z dct32x32 1.20→1.24 raise | SHARED-MECHANISM | W44-148 raises dct32x32 across ALL 3 variant Z tables |
| W44-148 (dct32x32=1.24) | W44-154 micro-bisect to 1.22 | SUPERSEDED | Recovers 5 of 6 W44-148 boundary FIXED→OPEN flips |
| W44-148 (dct32x32=1.22) | W44-156 d_high tables (dct32x32=1.20 at d>5.5) | SHARED-MECHANISM | Distance-aware split; 3 new d-high tables |
| W44-29 outer admission (mask<50) | W44-151 mask_p25>=85 admission widen | CO-GATED | W44-151 widens W44-29 with OR-branch; W44-152 narrows to d in [3.0, 5.0] |
| W44-152 outer table choice | W44-166 INNER variant Z admit (mask_p25>=85) | ADDITIVE | Compose orthogonally (outer + inner on different layers) |
| W44-152 outer admission | W44-165 EPF seed photo admit (mask_p25>=85) | ANTAGONIST | Competed at recon-signal layer — overshoot -0.105 SSIM2 |
| W44-150 mask_p25>=85 discriminator | W44-166 photo_variant_z_admit | CONTENT-COUPLED | W44-166 reuses W44-150 predicate (originally paired with W44-117 EPF seed, honest-stopped) |

### 4.4 DCT32/64 suppression chain

| A | B | type | mechanism |
|---|---|---|---|
| W44-65 try_dct64=false (mask>99.5) | W44-68 try_dct32=false (mask>99.5) | ADDITIVE | Both suppress large-DCT search on saturated-screenshot content |
| W44-65/W44-68 suppression gate | W44-122 admit-DCT64 on codec_wiki d=3 | ANTAGONIST | W44-122 honest-stopped — "DCT64 picks land on blank regions; SSIM2 cliff is in TEXT regions" |
| W44-65 try_dct64=false | W44-123 with_dct32_keep_hint (codec_wiki) | CO-GATED | W44-123 opts OUT of W44-65's try_dct32 portion (keeps try_dct32=true when try_dct64=false fires) |
| W44-123 (opt-in) | W44-124 auto-discriminator (m3>=60 AND ed<0.05) | SUPERSEDED | W44-124 wires W44-123 as default via zenanalyze proxies |
| W44-124 m3+ed discriminator | W44-135 W44_124_DCT32_KEEP_AUTO_MIN/MAX_DISTANCE (=2.0..3.5) | CO-GATED | W44-135 narrows W44-124 firing to d in [2.0, 3.5] |
| W44-135 distance gate | W44-143 W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE 2.0→1.4 widen | SUPERSEDED | Lower bound bisect-optimized to 1.4 |
| W44-124 dct32_keep predicate | W44-142 EPF SUPPRESS predicate | CONTENT-COUPLED | Both reuse (m3>=60 AND ed<0.05) constants |
| W44-93 try_dct64 effort widening (e>=5) | W44-65 try_dct64=false gate | ANTAGONIST | W44-93 attempted libjxl-parity gate widening; broke W44-65 mechanism balance |
| W44-104 admit-DCT64-terminal | W44-122 admit-DCT64-codec_wiki d=3 | SHARED-MECHANISM | Both honest-stopped on with_dct_suppress_hint(Some(false)) bypass |

### 4.5 W44-82 / coeff_orders cost-gate chain

| A | B | type | mechanism |
|---|---|---|---|
| W44-77 find_best_32x32 entropy_mul tightening | W44-94 find_best_32x32 widen variants W/X/Y/Z | ANTAGONIST | "1420710 and 1531677 want OPPOSITE adjustments at d=5" — global table cannot satisfy both |
| W44-77 entropy_mul cost-gate | W44-82 custom_orders cost-benefit gate | SHARED-MECHANISM | Both implement libjxl-parity cost-benefit gating |
| W44-82 coeff_orders cost-benefit gate | W44-201 DCT32 family coeff_orders skip | SUPERSEDED | W44-201 adds per-bucket SKIP within W44-82's framework; closes 219 Pareto-losers |
| W44-201 large_buckets gate (DCT32) | W44-205 medium_buckets gate (DCT16-family) | ADDITIVE | Per-bucket Lehmer probe confirms independent contributions; combined -1.85% |
| W44-201+W44-205 per-bucket gates | W44-206 savings_factor multiplier (f=0.3) | REDUNDANT | Gates deliver -1.85%, single-scalar 6.4× smaller; per-bucket SKIP structurally necessary |
| W44-167 per-m3 find_best_32 INNER widen | W44-94 find_best_32 widen | SHARED-MECHANISM | Different table layer, same lever class — W44-167 honest-stopped on INNER |
| W44-204 C1 (extend W44-201 to buckets 4/5) | W44-201 DCT32 bucket gate | SHARED-MECHANISM | Generalizes per-bucket SKIP to DCT16x8/16x16 buckets |

### 4.6 BlockCtxMap clustering chain

| A | B | type | mechanism |
|---|---|---|---|
| W44-71 15-cluster default | W44-73 LZ77-in-non-simple writer fix | ADDITIVE | Both fix block-context map encoding; both required for cluster-map wins |
| W44-71 | W44-80 / W44-84 BlockCtxMap threshold | SHARED-MECHANISM | All in BlockCtxMap discriminator family |

### 4.7 DC tree / `kLearn` chain (perf-arc)

| A | B | type | mechanism |
|---|---|---|---|
| W44-54 DC tree LearnTree at e>=4 (MISREAD libjxl line) | W44-171 raise to e>=8 (libjxl-parity) | SUPERSEDED | W44-171 corrects W44-54 misread of `enc_modular.cc:1166` vs `:1591`; 17-21× speedup |
| W44-171 (Variable trial at e>=8) | W44-172 (Predictor::Best at e=8, Variable at e>=9) | CO-GATED | Libjxl `enc_modular.cc:1593` kKitten parity; 3.20-3.30× speedup terminal d=0.5 e8 |

### 4.8 CfL Newton chain (W44-184 salvage)

| A | B | type | mechanism |
|---|---|---|---|
| W44-178 DCT64X64 recon hypothesis | W44-181 DC quant precision probe | SUPERSEDED | 4-cycle falsification chain on clic_097cb426 right-column SSIM2 deficit |
| W44-182 CfL Newton 4-param divergence | W44-183 default-flip falsified | ANTAGONIST | W44-29..W44-172 downstream cost-model calibrated against broken-CfL baseline; correcting in isolation regressed 25/27 cells |
| W44-182 CfL Newton bug | W44-184 Libjxl-only salvage opt-in | CO-GATED | Ships W44-182 port behind EncoderStrategy::Libjxl ONLY |
| W44-184 Libjxl Pass-2 Newton | W44-195 Libjxl Pass-1 Newton | ADDITIVE | Together resolve Section C CfL Newton divergence (Libjxl strategy only) |

### 4.9 Smart-Zenjxl arc (W44-164..169)

| A | B | type | mechanism |
|---|---|---|---|
| W44-118 is_screenshot gate | W44-164 auto-classify content_class | CONTENT-COUPLED | W44-164 generalizes W44-118's median>95 to 3-class fcbr+m3 classifier |
| W44-168 adaptive_buttloop_iters (broad) | W44-166 variant Z photo admit | ANTAGONIST | W44-168 broad mode destroyed W44-166's +0.45 SSIM2 win on 1418519 e8 d=6 |
| W44-168 (forward-compat field) | W44-169 narrow d in [4.0, 5.0] iter-skip | ADDITIVE | W44-168 = forward-compat hook; W44-169 = narrow distance band salvage |
| W44-169 narrow iter-skip | W44-166 variant Z photo admit (PROTECT) | CO-GATED | W44-169 gate uses [4.0, 5.0] to avoid d=6 where W44-166 ships |

### 4.10 Honest-stops + ruled-outs

| A | B | type | mechanism |
|---|---|---|---|
| W22-1 screenshot_lift_hint | W23-2 lift-table bisect | ANTAGONIST | Mask>95 too coarse — every lifted tuple regresses windows95 +30-33% bfly at d=0.5 |
| W44-87 single-pass entropy dispatch | W44-88/89 polish levers | SUPERSEDED | All in e5 perf cluster nulled by variance artifact |
| W44-90 PixelLossDispatch default-flip | W44-163 B2 nested PixelLossDispatch | CONTENT-COUPLED | W44-90 mask>80 too coarse (windows95 catastrophic); B2 tightens with nested m3 + distance |
| W44-102 cfl_two_pass widen to e>=5 | W44-101 audit prediction | SUPERSEDED | W44-101 predicted 10-20 wedge closures; W44-102 measured 0/4 + 4 cells over -0.3 SSIM2 |

### 4.11 Refactor / consolidation

| A | B | type | mechanism |
|---|---|---|---|
| W44 strategy_def! macro (W44-192/193) | W44-127→W44-133 EncoderStrategy enum | SHARED-MECHANISM | Both replace `with_*_hint(Option<bool>)` proliferation with structured presets |
| W44-29/91/96/98/99/100 cluster | W44-148/152/154/156/166/167 cluster | SHARED-MECHANISM | All hang off `with_high_d_photo_hint`; outer + inner layer composition |
| W44-188 audit (Section C/D/E missing dct16x32 row) | W44-98/99/100 variant Z constants | SHARED-MECHANISM | Audit added dct16x32 row to Section C of LIBJXL_DIVERGENCES.md |

### 4.12 Meta-directive (binding)

| A | B | type | mechanism |
|---|---|---|---|
| W44-66 user-correction "never FMA precision" | W44-178→W44-184 + all subsequent honest-stops | SHARED-MECHANISM | User binding rule cited in 30+ memos; affects how subsequent chunks frame SSIM2/byte deltas |

---

## Section 5: Shipped-win patterns

Repeatable templates extracted from W44-210-B/C arc analysis. Each
pattern lists its prototype chunk, expansion chunks, and "when to reach
for it" guidance.

### Pattern 1: zenanalyze-discriminated entropy_mul lift

- **Prototype**: W44-29 (mask1x1<50)
- **Expansions**:
  - W44-78 widen distance gate 4.0 → 3.0
  - W44-91 add m3>=80 + fcbr<0.01 in ambiguous band [50, 80)
  - W44-96 add edge_density>=0.7 + fcbr<0.01 (variant Z)
  - W44-98 split high-colour (m3>=25, dct16x32=1.30)
  - W44-99 split low-colour (m3<25, dct16x32=1.22)
  - W44-100 micro-bisect LC 1.22→1.23
- **Shipped wins**: closes 3-7 OPEN cells per chunk; 0 OPEN at W44-100
- **When to reach for**: when a single image-class wants a parameter
  shift that other images would regress on. Cost is one new
  `EntropyMulTable` variant + one nested predicate inside an existing
  gate.
- **DO NOT**: ship a global lift without per-image discriminator
  (W44-28/31/77/94/95 all honest-stopped).

### Pattern 2: cost-model recalibration via per-bucket SKIP

- **Prototype**: W44-77 find_best_32x32 entropy_mul tightening (honest-stop
  on widening attempts)
- **Expansions**:
  - W44-201 coeff_orders bucket-skip for DCT32 family (buckets 3+6)
  - W44-205 extend to DCT16 family (buckets 2+4)
  - W44-206 RULED OUT recalibrating savings_factor globally
- **Shipped wins**: W44-201 closed 219 Pareto-losers (-19255 B / -0.65%)
- **When to reach for**: when the cost model's underlying assumption is
  wrong (e.g. 1 bit/zero vs empirical 0.3-0.5), not just a parameter
  miscalibration. Per-bucket SKIP is structurally necessary.
- **DO NOT**: replace per-bucket gates with a single scalar multiplier
  (W44-206 falsified — 6-8× smaller EV).

### Pattern 3: distance-window narrowing

- **Prototype**: W44-107 (tighten W44-105 d>=2 → d>=3.5)
- **Expansions**:
  - W44-120 (W44-117 EPF seed min_distance=1.0)
  - W44-135 (W44-124 dct32_keep window [2.0, 3.5])
  - W44-143 (lower W44-135 min to 1.4)
  - W44-152 (narrow W44-151 to [3.0, 5.0])
  - W44-156 (distance-split variant Z at d>5.5)
  - W44-169 (narrow W44-168 iter-skip to [4.0, 5.0])
- **Shipped wins**: closes 1-5 cells per chunk by excluding the
  distance subrange where the mechanism over-applies
- **When to reach for**: when a mechanism ships strong wins on a
  distance subrange but regresses on adjacent distances. Bisect the
  band; narrow until the regression closes.
- **Bisect grid**: typically {min, min+0.2, min+0.5, min+1.0}; W44-107
  through W44-156 all used 3-5 point bisects.

### Pattern 4: opt-in API → auto-discriminator wire-up

- **Prototype**: W44-63 (with_dct_suppress_hint API) → W44-65 (auto
  median>99.5)
- **Expansions**:
  - W44-123 with_dct32_keep_hint → W44-124 auto (m3>=60 AND ed<0.05)
  - W44-79 high_d_photo discriminator docs → W44-91 auto-wire
  - W44-166 photo_variant_z_admit API + auto (Zenjxl=true)
- **Shipped wins**: opt-in coverage extended to auto via measured
  discriminator
- **When to reach for**: when a high-EV mechanism ships behind an opt-in
  API; the cardinal rule "leave nothing unported" requires pairing
  every opt-in with an auto-discriminator.

### Pattern 5: mechanism-layer classification

- **Prototype**: W44-165 honest-stop (paired W44-150 discriminator with
  W44-117 EPF seed — both pushed recon in compatible direction →
  overshoot)
- **Lesson**: classify which of the 5 mechanism layers a new gate
  operates on (see [Section 2](#section-2-mechanism-layers)). Same-layer
  composition requires strict-partition discriminator; different-layer
  composition can be additive but needs measurement.
- **Application**: W44-166 succeeded where W44-165 failed by switching
  from Layer 4 (EPF seed) to Layer 3 (inner variant Z).

### Pattern 6: per-stream trial-and-pick (DC tree)

- **Prototype**: W44-57 (trial both Variable + kWPFixedDC DC trees, pick
  cheaper)
- **Expansion**: W44-171 corrected the effort gate from `>=4` to `>=8`
  (libjxl-parity); W44-172 added Predictor::Best at e=8 (libjxl-parity
  for kKitten)
- **When to reach for**: when two cost-model branches each win on
  different content; build both, pick by per-stream cost estimate.

### Pattern 7: content-aware iter-count adjustment

- **Prototype**: W44-168 (SmoothSkip mode — admit forward-compat hook)
- **Expansion**: W44-169 narrow distance-band [4.0, 5.0] for iter-skip
  on smooth photos
- **Shipped wins**: W44-169 cuts wall 6-8% on 1418519 e8 d=4/5 without
  SSIM2 cost
- **When to reach for**: when wall time on a narrow class of cells is
  the wedge, not bytes or SSIM2.
- **DO NOT**: extend without explicit measurement that broader admission
  doesn't regress other classes (W44-168 broad mode regressed W44-166
  PROTECT cells).

### Pattern 8: Libjxl-strategy-only opt-in for parity-locked deviations

- **Prototype**: W44-184 (CfL Newton parity port wired behind
  `EncoderStrategy::Libjxl` only)
- **Expansion**: W44-195 (Pass-1 Newton on Libjxl), W44-197 (Pass-2 LS at
  e=5/6 on Libjxl)
- **When to reach for**: when libjxl-parity is desired but default-flip
  would regress downstream cost-model calibration. Ship the parity port
  behind `EncoderStrategy::Libjxl` to preserve byte-identical
  hash-locks for Zenjxl/Aggressive/LeanFaster.

---

## Section 6: Cross-arc connections

How W44-N corrected or built upon earlier W44-M:

- **W44-148/154/156 variant Z dct32x32** trail starts from
  W44-96/98/99/100 (which had set dct32x32=1.20 on a narrow 4-cell
  measurement set). The W44-148 broader-corpus refresh REVERSED that
  direction (1.20 → 1.24, then W44-154 micro-bisect to 1.22, then
  W44-156 distance-aware split: 1.22 at d<=5.5, 1.20 at d>5.5).

- **W44-176 terminal_class_exclude** is a refinement on W44-108 (which
  itself recovered W44-105 d=2..3 wins that W44-107 sacrificed). The
  mechanism stack W44-91 → W44-108 → W44-176 evolved per-image
  discriminator precision over 3 chunks.

- **W44-201 DCT32 coeff_orders gate** fixes the W44-82 cost-model that
  has been load-bearing since the prior arc. W44-204 audit identified
  W44-82's `savings_factor=1.0` assumption as a structural mismatch with
  cjxl's behaviour (308 vs 5 Lehmer codes is 61.6× ratio).

- **W44-184 libjxl-parity CfL Newton** is the salvage of W44-183
  honest-stop. The downstream W44-29 → W44-172 cost-model tuning is
  calibrated against the effective LS-warm-start baseline; W44-184
  wires the parity port behind `EncoderStrategy::Libjxl` to make both
  worlds available.

- **W44-171 DC tree gate** corrected a W44-54 misread of libjxl
  `enc_modular.cc:1166` (real gate is `:1591`, effort >= 8). The fix is
  libjxl-parity AND removes the perf wedge — both at once.

- **EncoderStrategy 7-chunk arc** (W44-127→W44-133) consolidated all
  per-divergence dials from W44-1..100 era (W44-29, W44-65, W44-91,
  W44-96/98/99/100, W22-1, W44-34/35, W44-105/107/108, W44-117/118/120,
  W44-109, W44-123/124) into top-level presets via `ResolvedImprovements`.
  The W44-128/129 rewires touched ~10 historical gate sites.

- **W44-178 → W44-184 4-cycle falsification** on clic_097cb426 right-
  column SSIM2 deficit: DCT64X64 hypothesis → DC quant precision → DC
  CfL → DC dequant FMA → CfL Newton 4-param divergence IDENTIFIED →
  default-flip FALSIFIED → Libjxl-only opt-in SHIPPED.

---

## Section 7: DO NOT (binding for future agents)

Compiled from the W44-210-A/B/C/D/E DO-NOT lists. Any future tuning
chunk that wants to revisit these MUST first re-read the cited memo.

### Discriminator widenings

- **DO NOT widen W22-1 `mask>95`** to `mask>90` without a per-image
  sub-discriminator — W22-1/W23-2/Smart-Dispatch-Chunk-1 honest-stopped
  on windows95-class regression.
- **DO NOT widen W44-29 outer admission past `mask<50`** without
  zenanalyze proxies — W44-91/96/151 demonstrate discriminator is
  load-bearing.
- **DO NOT lower `W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN` below 25.0** —
  W44-99 measurement: 1531677 (m3=12.30) regresses SSIM2 -0.34 to -0.93
  under m3>=25 admission to dct16x32=1.30.
- **DO NOT raise `W44_124_DCT32_KEEP_M3_MIN` past 67** — would exclude
  imessage admission (designed boundary).
- **DO NOT raise `W44_91_FCBR_MAX` past 0.05** — admits 297394 (fcbr=0.096)
  which sits at photo-vs-screenshot split.
- **DO NOT widen `W44_152_W44_151_MAX_DISTANCE` beyond 5.0** — W44-151
  honest-stop measured d=6 over-fire conclusively.
- **DO NOT widen `HIGH_D_PHOTO_W44_91_MAX_DISTANCE` beyond 5** — W44-79
  trial saw +560 B regression at d=6 on 1189261.

### Effort gates

- **DO NOT widen `try_dct64` effort gate** — W44-93 RULED OUT (19 cells
  with SSIM2 drops >=0.3, OOM on imac_g3 e9 d=3).
- **DO NOT widen `cfl_two_pass` to e>=5 in default** — W44-102 RULED
  OUT (4 cells over -0.3 SSIM2 budget); shipped as Libjxl-strategy-only.
- **DO NOT lower `DC_TREE_VARIABLE_TRIAL_MIN_EFFORT` below 8 or raise
  above 8** without re-running W44-171 78%-of-CPU bench. Both
  directions break things.
- **DO NOT lower `DC_TREE_VARIABLE_PREDICTOR_FULL_MIN_EFFORT` below 9
  or raise above 9** — W44-172 measurement conclusive.

### Mechanism choices

- **DO NOT remove the W44-105/107/108/109 qac chain** — W44-119
  chain-disable measured catastrophic regression (SSIM2 -1.85 to -5.58
  on every screen cell). REDUNDANT only if W44-138 Phase-2
  buttloop-recon root-cause fix lands.
- **DO NOT pair a new mechanism with an existing chain member without
  classifying its mechanism layer** (1-5 in [Section 2](#section-2-mechanism-layers)).
  W44-165 honest-stop taught this rule.
- **DO NOT re-attempt Mechanism A admission via W44-117 EPF seed alone**
  — measured ~10% closure of 1418519 cluster deficit (W44-150).
  Pair the W44-149 discriminator with a DIFFERENT mechanism layer
  (W44-166 succeeded via Layer 3 variant Z).
- **DO NOT default-flip `cfl_newton_libjxl_parity` to true for Zenjxl**
  — W44-183 conclusive (25/27 cells regressed -0.25 to -13 SSIM2 +
  +7.82% bytes).
- **DO NOT replace W44-201/W44-205 per-bucket coeff_orders gates with
  a single `savings_factor` multiplier** — W44-206 ruled out (6-8×
  smaller EV).
- **DO NOT re-attempt 3637739 zenjxl Pareto-loser via discriminator-
  and-lift** — W44-198 probe found NO clean discriminator axis. Cluster
  is structural.
- **DO NOT re-bisect variant Z dct32x32 outside {1.20, 1.22}** —
  W44-148/154/156 conclusive across multiple bisects.
- **DO NOT cite "FMA precision" for any byte/SSIM2 delta** — binding
  user directive 2026-05-19, cited in 30+ memos.

### Refactor / structural

- **DO NOT remove the W44-145 helper consts** (`W44_145_PER_BLOCK_MASK_LOW`/`HIGH`)
  even though they're `#[allow(dead_code)]` — retained for the e8+
  per-block bimodal qac follow-up.
- **DO NOT remove the W44-168 `TEXTURED` consts** — reachable via
  `JXL_W44_168_MODE=C` env hook.
- **DO NOT hoist `mask1x1_median = 95.0` to a single shared const**
  without verifying every consumer agrees on the semantic (screenshot
  predicate vs high-confidence smooth predicate).
- **DO NOT change `LIBJXL_INIT_MUL = 0.6`** — libjxl wire-compatible
  value baked into the seed table.
- **DO NOT touch the `vardct/quant.rs` parametric quant-weight tables**
  (`DCT8_PARAMS` etc) — decoder agreement required (LOCKED).
- **DO NOT touch `vardct/ac_strategy.rs` `K_BIAS` / `K_POW_*`** —
  libjxl-spec distance scaling exponents; picker tunes BASE values
  via `EffortProfile`, not these exponents.

---

## Section 8: Empirical coupling structure (W44-217)

The 6 W44-213-wired [`crate::tuning::runtime::RuntimeTuning`] fields have
been empirically characterized via Tier-1 numerical analysis on the
W44-216 Stage B sweep corpus (4,938 cells, 27 images, 5 efforts, 7
distances, 2 strategies, 13 params blobs).

**See [`PARAM_INTERACTIONS.md`](PARAM_INTERACTIONS.md)** for the full
analysis: per-outcome ANOVA decomposition, 15 marginal PDP surfaces × 2
outcomes, 8 per-stratum PDPs, mutual information matrices, conditional
cross-term regressions across 31 strata, classification of each pair as
ADDITIVE / SUPPRESSIVE / SYNERGISTIC / GATED / SHARED-DISCRIMINATOR /
WEAKLY-COUPLED, and Tier-2 knob design recommendations.

### Top empirical findings

- The 6 RuntimeTuning fields have ZERO effect on `EncoderStrategy::Libjxl`
  (W44-213 wiring deliberately routes only through the zenjxl dispatches).
  Bytes CV across all 13 param blobs is ≤ 0.03 % on libjxl strategy.
- Pairwise interaction terms DOMINATE main effects: single-param effects
  explain 0.3–5.4 % of variance; pairwise interactions explain 6–22 % each.
- Marginal PDPs are ADDITIVE; conditional PDPs on `class=screen +
  dist_band=very_high` show strong SUPPRESSIVE/SYNERGISTIC structure with
  cross-term magnitudes up to 0.26 (normalized by σ_y).
- A Tier-2 knob set of 3 (`smoothness_bias`, `screen_quant_lift`,
  `buttloop_screen_d_gate`) covers the structure modulo (p1, p3) which
  is structurally mutually-exclusive (no image fires both dispatches).

### Coupling-skeleton module

Empirical findings are encoded as function skeletons in
[`crate::tuning::coupling`](../jxl-encoder/src/tuning.rs#L387) (the
new module added by W44-217). Each fn has a doc-comment hypothesis +
expected mechanism + Tier-2 reparameterisation.

**W44-218 status (shipped 2026-05-22)**: all 7 per-pair skeletons now
have closed-form ridge implementations. 15 round-trip / range /
saturation / composition tests assert byte-exact default round-trip,
W44-216 LHS envelope coverage, and saturation cap engagement. Per-pair
response R² fits were attempted but came in below the 0.5 acceptance
gate (best ~0.08) — corpus is too sparse in param dimension. Ridges
are calibrated from empirical envelope, with HONEST-STOP documented
for the response-fit deferred to W44-220 after the W44-219 denser sweep.

**W44-220 status (HONEST-STOP 2026-05-22)**: per-pair response R²
REFIT attempt on the 21×-denser W44-216+W44-219 combined corpus
(267 blobs, 13991 rows) HONEST-STOPPED below the 0.5 gate. 0 of 7
pairs clear with linear+cross-term; 0 of 7 with GBR-pair-only; 3 of 14
(pair, outcome) cells with GBR-all-6 (all on `log_bytes_resid` at
`class=screen/dist_band=very_high`, shared across p3_p6/p4_p5/p4_p6).
**The algebraic forms are wrong, not the corpus** — the structural
ceiling on the highest-signal stratum is `ssim2 R² ≈ 0.41`,
`log_bytes R² ≈ 0.44`. Re-derivation queued as W44-221+. The W44-218
geometric-calibration ridges REMAIN SHIPPED.

See [`PARAM_INTERACTIONS.md`](PARAM_INTERACTIONS.md) "W44-220 status"
and [`benchmarks/sweeps/w44-219-densify/analysis/w44_220/README.md`](../benchmarks/sweeps/w44-219-densify/analysis/w44_220/README.md)
for the per-pair gate failure measurement and W44-221+ re-derivation
candidates.

The `expand_knobs_to_runtime` fn (W44-222 scope) composes the per-pair
fns into the 6-vector the production encoder consumes; it remains
`unimplemented!()` until W44-222 lands.

### Analysis artefacts

`benchmarks/sweeps/w44-216-stage-b/analysis/`:
- `scripts/` — 7 Python scripts (`prep_data.py`, `anova_analysis.py`,
  `mi_analysis.py`, `pdp_analysis.py`, `conditional_analysis.py`,
  `stratum_pdp.py`, `sanity_check.py`)
- `anova_<outcome>.tsv` × 5 + `anova_summary_per_param.tsv`
- `mi_param_outcome.tsv`, `mi_feature_outcome.tsv`,
  `mi_param_x_feature_<outcome>.tsv` × 2
- `pdp_<pi>_x_<pj>_<outcome>.png` × 30 (15 marginal pairs × 2 outcomes)
- `stratum_pdp/pdp_*_class*_<outcome>.png` × 8 (top conditional couplings)
- `coupling_classification.tsv`, `stratum_interactions.tsv`,
  `interaction_ranking.tsv`
- `params_blob_decode.json` — 13 params blob sha256 → (p1..p6) decoding

### Successor work

- **W44-218** (SHIPPED 2026-05-22): per-pair ridges through default,
  calibrated from empirical envelope. 7 of 7 skeletons replaced with
  closed-form fns; 15 unit tests; hash-locks byte-identical.
  Per-pair response R² fits HONEST-STOPPED below 0.5 acceptance gate
  pending W44-219 denser sweep.
- **W44-219** (queued): design a follow-up sweep targeting the open
  questions in `PARAM_INTERACTIONS.md` §9 (50+ LHS samples, 100+
  images, denser low/high distance bands, content-class-stratified LHS).
- **W44-220** (HONEST-STOP 2026-05-22): refit attempted on
  W44-216+W44-219 combined corpus (267 blobs, 21× density). 0/7
  pairs cleared the 0.5 R² gate with linear+cross forms; 3/14
  (pair, outcome) cells cleared with GBR-all-6 (all
  log_bytes_resid on screen/very_high). The W44-218 algebraic
  forms are STRUCTURALLY underfit, not corpus-sparse. W44-218
  geometric ridges RETAINED. See
  [`benchmarks/sweeps/w44-219-densify/analysis/w44_220/`](../benchmarks/sweeps/w44-219-densify/analysis/w44_220/)
  for measurement and W44-221+ re-derivation candidates.
- **W44-221+** (next): pick a re-derivation direction (six-knob
  expansion, per-class formula families, RD-theoretic derivation,
  or Tier-3 MLP conditioning). Drop the per-pair-ridge approach
  per W44-220 findings.
- **W44-222** (queued): implement `expand_knobs_to_runtime` —
  compose the 7 per-pair ridge fns into the full 6-vector
  `RuntimeTuning` consumed by the production encoder.

---

## Provenance

| input | source | scope |
|---|---|---|
| const inventory | W44-210-A memo | 180 const rows across 13 sections |
| W44-1..100 history | W44-210-B memo | 88 tags, 6 clusters |
| W44-101..206 history | W44-210-C memo | 317 commits, 92 chunks, 4 phases |
| edge graph + chains | W44-210-D memo | 58 edges + 5 mechanism layers + 5 chains |
| libjxl comparison | W44-210-E memo | ~120 consts vs libjxl, bucket assignment |
| existing divergence ledger | `docs/LIBJXL_DIVERGENCES.md` | 170 rows |
| residual cluster ranking | W44-204 audit memo | Smart-Zenjxl held-back list |

Consolidated by W44-210-consolidate (2026-05-22). Future maintenance per
the mandatory rule at the top of this file.
