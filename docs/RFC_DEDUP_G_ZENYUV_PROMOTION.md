# RFC: zenyuv promotion to top-level shared YCbCr crate

**Status**: DRAFT (DEDUP-G scoping audit, 2026-05-26)
**Owner**: TBD (queued chunk)
**Reference**: `~/work/zen/zensim/benchmarks/dedup_inventory_master_2026-05-26.md` Tier-0 #1
**Risk classification**: HIGHEST correctness urgency (zero-tolerance color precision per CLAUDE.md)

## TL;DR

Three independent RGB↔YCbCr SIMD implementations carry overlapping color-transform math
across the codec workspace. The audit ranks this as the **single highest-correctness-urgency
extraction** because two independent matrix+gamma implementations *will* round differently at
boundaries, and there is no cross-check today. Promoting `zenyuv` to a top-level shared crate
and routing `zenavif` (and optionally the small `zenjxl-decoder` YCbCr stage) through it
closes the precision-divergence shipping-bug surface.

This RFC scopes the work into 3 phases. **Phase 1 alone is shippable** as a useful expansion
of zenyuv's public API; Phase 2 is the actual migration (multi-day, golden-test-heavy); Phase 3
is a smaller audit that may decide to leave zenjxl-decoder's stage alone.

## Inventory (verified 2026-05-26)

| Component | Repo path | LOC | Direction | SIMD backend | Depends on zenyuv? |
|---|---|---:|---|---|:--:|
| **zenyuv** (canonical) | `~/work/zen/zenjpeg/zenyuv/` (workspace member of zenjpeg) | 5363 | Both | archmage + magetypes (`#[arcane]` x86_64 AVX2, NEON, WASM SIMD128, generic) | (self) |
| **zenavif** YCbCr stack | `~/work/zen/zenavif/src/yuv_convert*.rs` + `convert.rs` + `strip_convert.rs` (7 files) | 3425 | YCbCr→RGB only | hand-rolled magetypes + libyuv ports | **NO** |
| **zenjxl-decoder** YCbCr stage | `~/work/zen/zenjxl-decoder/zenjxl-decoder/src/render/stages/ycbcr.rs` | 153 | YCbCr→RGB only | `jxl_simd` (its own dispatch wrapper) | **NO** |

The zenjxl-decoder YCbCr stage is much smaller because JXL's YCbCr↔RGB inverse runs on
planar f32 (no subsampling, no quant). zenavif's 3425 LOC handles u8/u16 + 4:2:0/4:2:2/4:4:4
+ alpha-add + streaming strip — a richer surface.

## Gap analysis — zenyuv public API vs zenavif requirements

### What zenyuv exposes today (`pub` surface)

```
pub struct YuvContext;
  pub fn new(range: Range, matrix: Matrix) -> Self
  pub fn encode_444_u8(...)        // RGB → YCbCr 4:4:4
  pub fn encode_420_u8(...)        // RGB → YCbCr 4:2:0
  pub fn encode_420_y_only_u8(...) // Y-only (grayscale)
  pub fn encode_420_f32(...)       // float variant
  pub fn encode_sharp_420_u8(...)  // sharp chroma downsample
  pub fn encode_sharp_420_f32(...)
pub struct SharpYuvConfig;
pub enum Matrix { Bt601, Bt709, Bt2020 }
pub enum Range { Full, Limited }
```

Decode-side (`yuv*_to_rgb`, `yuv*_to_rgb_with`, `yuv*_to_rgb_bilinear*`) exists in
`src/decode.rs` and `src/decode_generic.rs` (167+274 LOC) but is `pub(crate)` only.
The kernels are there; the public surface isn't.

### What zenavif needs (from `yuv_convert.rs` + `strip_convert.rs`)

```
pub fn yuv420_to_rgb8(y, cb, cr, strides, dims, matrix, range) -> ImgVec<RGB8>
pub fn yuv422_to_rgb8(...)
pub fn yuv444_to_rgb8(...)
pub fn yuv420_to_rgb8_strip(y, cb, cr, strides, dims, matrix, range, strip_y, strip_h) -> Vec<RGB8>
pub fn yuv420_to_rgba8_strip(...)  // adds opaque alpha=255
pub fn yuv422_to_rgb8_strip(...)
pub fn yuv422_to_rgba8_strip(...)
pub fn yuv444_to_rgb8_strip(...)
pub fn yuv444_to_rgba8_strip(...)
// Plus 16-bit u16 variants for 10/12-bit AV1 (in yuv_convert_fast.rs / yuv_convert_libyuv*)
```

### The 4 API gaps

1. **Direction**: zenyuv's pub surface is encode-only; zenavif needs decode. The kernels
   exist privately — just need pub re-exports.
2. **Bit depth**: zenyuv is u8-only on the pub surface (plus encode_420_f32 for floats);
   zenavif needs u16 paths for 10-bit / 12-bit AV1.
3. **Output format**: zenyuv writes to slices; zenavif returns `ImgVec<RGB8>` or
   `Vec<Rgba<u8>>` with alpha-fill. Trivial wrappers, but the convention difference
   matters for callers.
4. **Stride / strip API**: zenavif's `*_strip(strip_y, strip_h)` enables streaming
   decode; zenyuv's encode API is full-image only. Decode side needs strip support added.

### zenjxl-decoder gap (Phase 3 candidate, lower priority)

The 153 LOC stage uses `jxl_simd` (its own dispatch wrapper, NOT magetypes). It operates
on planar f32, in-place, in the render pipeline. Migrating it would require either:
- (a) Adding a magetypes-based planar-f32 in-place YCbCr→RGB to zenyuv (~150 LOC), then
  rewriting the render stage to call it. Same algorithm in two SIMD frameworks (jxl_simd
  vs magetypes) — the precision-divergence risk is real here too.
- (b) Leaving the render stage alone (153 LOC is small, the SIMD framework choice is local
  to zenjxl-decoder's pipeline) and accepting that this one ~3-coefficient YCbCr matrix
  is implemented in two places.

**Recommendation: defer to a follow-on chunk after Phase 2 lands**, with a measurement
gate (golden-pixel cross-check between jxl_simd dispatch and magetypes dispatch on
representative JXL test inputs).

## Phased migration plan

### Phase 1 — Expand zenyuv pub API (1-2 days, low-risk, ships independently)

Promote decode kernels to `pub`, add u16 variants, add strip/stride variants.

**Concrete deliverables:**
- `pub fn YuvContext::decode_420_u8(y, cb, cr, rgb_out, w, h)` matching the encode-side shape
- `pub fn YuvContext::decode_422_u8(...)` + `pub fn YuvContext::decode_444_u8(...)`
- u16 variants: `decode_420_u16(y_u16, cb_u16, cr_u16, rgb_out_u8, w, h, bit_depth)` with
  appropriate range scaling
- Strip variants: `decode_420_u8_strip(y_row, cb_row, cr_row, rgb_out_row, w)` that
  process one (or N) row(s) — composable into zenavif's strip API
- RGBA output: add an `_rgba` family that writes 4-channel output with alpha=255 (or a
  callback). Considered: a single trait-based `OutputFormat` (Rgb | Rgba) — defer to v0.2
  if it adds surface.

**Acceptance gates:**
- All new pub fns have golden roundtrip tests vs `yuv` crate (already used in zenyuv tests
  for cross-validation, max_abs_err <= 1, mean_abs_err < 0.05).
- 16-bit decode tested at 10-bit and 12-bit input ranges.
- `cargo bench rgb_to_yuv_bench` still passes regression gate (no perf regression on
  encode-side).
- Ship as `zenyuv 0.2.0` with full CHANGELOG entry.

### Phase 2 — Migrate zenavif (3-4 days, high-risk, golden-test-heavy)

Add `zenyuv = { path = "../zenjpeg/zenyuv" }` to zenavif Cargo.toml. Rewire
`yuv_convert.rs` + `strip_convert.rs` to delegate to zenyuv. Keep zenavif's existing public
surface (no breaking change) — zenavif's enums (`YuvRange`/`YuvMatrix`/`ChromaSubsampling`)
become thin From-impls to zenyuv's enums.

**Concrete deliverables:**
- Bridge: `impl From<zenavif::YuvRange> for zenyuv::Range` etc.
- `yuv_convert.rs::yuv420_to_rgb8` body becomes 3 lines: call zenyuv decode, wrap in
  ImgVec, return.
- All 7 strip fns delegate via `YuvContext` reused per-call.
- DELETE: `yuv_convert_fast.rs` + `yuv_convert_libyuv*.rs` (after measuring that zenyuv's
  perf is competitive or better — likely true since zenyuv is magetypes-AVX-512-ready and
  yuv_convert_fast is hand-magetypes).
- Per-strip-size golden tests on representative AVIF test inputs (Kodim corpus 4:2:0 +
  4:4:4 + 10-bit HDR samples) — pixel-exact match against pre-migration output.
- Acceptance: zenavif decoder roundtrip tests pass identically; no SSIM2 / butteraugli
  regressions on the zenavif fuzz corpus.

**Risks:**
- zenavif's libyuv-port kernels (`yuv_convert_libyuv*.rs`) may produce slightly different
  rounding than zenyuv's per-matrix kernels. The audit's claim that "two matrix impls *will*
  round differently at boundaries" applies HERE — Phase 2 surfaces the divergence. Resolve
  by:
  - Picking ONE rounding convention (BT.601 Professional vs full BT.709 standard) and
    documenting which kernels match which.
  - Measuring max_abs_err vs both libyuv and `yuv` crate references on each migrated path.
  - If a path diverges by > 1 ULP at a boundary, document in a `KNOWN_DIVERGENCES.md` and
    add an env-var fallback to the libyuv-port for one release cycle.
- 16-bit roundtrips need explicit BT.2020 + PQ + HLG fixtures (HDR AVIF).
- zenyuv's max u16 input bit-depth needs to cover AV1's 10 + 12 bit; check current support.

### Phase 3 — Audit zenjxl-decoder YCbCr stage (0.5-1 day, decide-only)

**Measurement chunk, NOT a migration commitment.** Cross-check the 153-LOC
`zenjxl-decoder/.../ycbcr.rs` stage against the same matrix math via zenyuv on:
- 10 representative JXL fixtures (CID22 + screenshots) covering the YCbCr decode path
  (i.e., JXL files using ColorEncoding YCbCr, NOT XYB).
- Document max_abs_err in `~/work/zen/zenjxl-decoder/docs/YCBCR_PARITY.md`.

**Decision matrix:**
- If max_abs_err <= 1 ULP across all fixtures: leave both impls in place; one shared
  kernel doesn't pay back the migration cost on 153 LOC. Add a CI regression test that
  pins the current numeric agreement.
- If max_abs_err > 1: this IS the silent-precision-divergence shipping bug the audit
  warned about. Migrate immediately (rewrite the stage as a zenyuv call).

## Why this is highest correctness urgency (per CLAUDE.md)

> **ZERO TOLERANCE for image corruption, distortion, or precision loss.** Any code path
> that silently produces wrong pixels — even by 1 bit, even at boundaries, even "only in
> streaming mode" — is a shipping bug, not an acceptable tradeoff.

Two independent matrix+gamma implementations for the same color space *necessarily* round
differently somewhere — the question is only where, by how much, and whether the audit
detects it before a user reports a wedge. Phase 2's golden-pixel cross-test surfaces those
boundaries; Phase 3 surfaces the zenjxl-decoder analog.

The audit is right to rank this above the more visible work (target-loop dedup, recipe
consolidation) because color-precision bugs ship silently and the math-divergence surface
grows every time another codec gets its own YCbCr fast path.

## Out of scope (NOT this RFC)

- Promoting zenyuv into its own top-level git repo. zenyuv is currently a workspace member
  of zenjpeg. Moving it out is a separate decision (logistics, releases, CI). For now, the
  recommendation is **publish zenyuv to crates.io separately** (it already has its own
  Cargo.toml + version + license + repo URL) and consume via crates.io — no workspace move
  required.
- `zenpixels-convert`'s ICC / gamut / oklab role. That crate owns a different color
  domain (perceptual, HDR tone-mapping). zenyuv is the codec YCbCr fast path. No overlap.
- `garb`'s byte-swizzle role. Already shared, no overlap.

## Acceptance checklist (post-Phase-2)

- [ ] zenyuv 0.2.0 published with pub decode + u16 + strip API + Phase 1 CHANGELOG
- [ ] zenavif depends on zenyuv (path or version) + 3425 LOC of yuv_convert* deleted (or
  reduced to thin enum-bridge + delegation)
- [ ] zenavif decoder roundtrip tests pass pixel-exact vs pre-migration baseline (golden
  artifacts committed to fuzz corpus)
- [ ] HDR AVIF fixtures (PQ + HLG, 10-bit + 12-bit) verified
- [ ] Pre-existing fuzz corpus in zenavif passes without regression
- [ ] `KNOWN_DIVERGENCES.md` documents any rounding deltas with rationale
- [ ] CHANGELOG entries in BOTH zenyuv and zenavif under `[Unreleased]`
- [ ] Single (or 2-commit) PR per repo, assigned to `lilith` per CLAUDE.md
- [ ] Phase 3 audit doc shipped to zenjxl-decoder docs/ (decision documented)

## Open questions for owner

1. Does zenyuv's existing AVX-512 / NEON dispatch cover u16 paths, or does the kernel-side
   need to grow magetypes lane-width support before Phase 1 can ship clean?
2. zenavif's `auto-tune` feature pulls in zenanalyze + zenpredict — does Phase 2 keep that
   chain unaffected? (Yes, it's orthogonal to the YCbCr decode path.)
3. zenavif's libyuv-port kernels may have specific numeric agreements with libyuv-using
   downstream consumers (browsers, Chromium, libavif). Worth checking if any user depends
   on byte-exact libyuv output before deleting `yuv_convert_libyuv*.rs`.

---

*This RFC is a read-only scoping audit. No source modified. Phase 1 is the next
actionable chunk once owner is assigned.*
