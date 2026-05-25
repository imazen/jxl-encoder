# RFC: zensim audit vs the `RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md` contract

**Author**: Lilith River (with Claude scaffolding)
**Started**: 2026-05-25
**Status**: SCOPING (analysis-only; informs `RFC_ZENSIM_FORK_PLAN.md`)

This document audits the zensim crate's current API surface (CPU at `~/work/zen/zensim/zensim/` + GPU at `~/work/zen/zenmetrics/crates/zensim-gpu/`) against the 14 + 6 gates in `RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md` §9. The shape mirrors `docs/CVVDP_W44_GATE_TRANSFER.md` — a row per requirement with current state + gap + effort estimate.

The user statement that motivated this audit (2026-05-25): "zensim's latest version is comparable to ssim2 + cvvdp". The audit verifies that claim against the source code, then identifies what the buttloop integration needs that zensim doesn't currently expose.

## §1. zensim repository inventory

### §1.1. CPU crate (`~/work/zen/zensim/zensim/`)

Library entry: `zensim/src/lib.rs:1-306`.

Pub surface (extracted from the module's `pub use` statements):

```rust
// --- Primary API ---
pub use error::ZensimError;
pub use metric::{FeatureView, Zensim, ZensimResult,
    dissimilarity_to_score, score_to_dissimilarity};
pub use codec_calibration::{CalibrationAffine, CodecCalibration};
pub use profile::ZensimProfile;
pub use source::{AlphaMode, ColorPrimaries, ImageSource, PixelFormat,
    RgbSlice, RgbaSlice, StridedBytes};

// --- Diffmap API ---
pub use diffmap::{DiffmapOptions, DiffmapResult, DiffmapWeighting};

// --- Streaming / batch API ---
pub use streaming::{PrecomputedReference, ZensimScratch};

// --- Feature-gated ---
#[cfg(feature = "classification")] pub use metric::{AlphaStratifiedStats, ...};
#[cfg(feature = "training")] pub mod cvvdp_features;
#[cfg(feature = "training")] pub mod xyb_lms_features;
#[cfg(feature = "zenpixels")] pub use zenpixels_compat::ZenpixelsSource;
```

**Zensim type** (`metric.rs:986-1047`):

```rust
pub struct Zensim {
    profile: ZensimProfile,
    parallel: bool,
    max_pixels: Option<usize>,
}
impl Zensim {
    pub fn new(profile: ZensimProfile) -> Self;
    pub fn with_parallel(self, parallel: bool) -> Self;
    pub fn with_max_pixels(self, max_pixels: usize) -> Self;
}
```

**`ZensimProfile` variants** (`profile.rs:25-265`):

- `PreviewV0_1`, `PreviewV0_2` (legacy, 218k synthetic baseline)
- `PreviewV0_3` (current default, 372-feature extended + IW pool path; CID22 SROCC 0.9367, JND CV-MAE 0.0078)
- `PreviewV0_4` (228→64→1 MLP, mixed-supervision)
- `PreviewV0_5` (alias for `PreviewV0_5Balanced`)
- `PreviewV0_5Balanced`, `PreviewV0_5Compression`, `PreviewV0_5Ensemble`, `PreviewV0_5Tuner` (two-trail SOTA framework)

Profile-axis: `PreviewV0_5Tuner` is the **fine-tuner** profile (per `profile.rs:218-265` ranking table — 0.9278 SROCC, 0.0044 CV-MAE, best of all on JND). `PreviewV0_3` remains the legacy default. The default in `Zensim::new` is whatever the caller passes; there's no auto-default profile.

**Scalar score API** (`metric.rs:1054-1133`):

```rust
pub fn compute(&self, src: &impl ImageSource, dst: &impl ImageSource)
    -> Result<ZensimResult, ZensimError>;
pub fn compute_with_codec_hint(&self, src, dst, codec_hint: Option<&str>)
    -> Result<ZensimResult, ZensimError>;
pub fn compute_extended_features(&self, src, dst)
    -> Result<ZensimResult, ZensimError>;
```

**Warm-reference + batch API** (`metric.rs:1152-1474`):

```rust
pub fn precompute_reference(&self, src: &impl ImageSource)
    -> Result<PrecomputedReference, ZensimError>;
pub fn compute_with_ref(&self, pre: &PrecomputedReference, dst: &impl ImageSource)
    -> Result<ZensimResult, ZensimError>;
pub fn compute_with_ref_into(&self, pre, dst, scratch: &mut ZensimScratch)
    -> Result<ZensimResult, ZensimError>;
pub fn precompute_reference_linear_planar(
    &self, planes: [&[f32]; 3], width: usize, height: usize, stride: usize,
) -> Result<PrecomputedReference, ZensimError>;
```

**Diffmap API** (`diffmap.rs:608-684, 706-795, 806-815`):

```rust
pub fn compute_with_ref_and_diffmap(
    &self, pre: &PrecomputedReference, dst: &impl ImageSource,
    options: impl Into<DiffmapOptions>,
) -> Result<DiffmapResult, ZensimError>;

pub fn compute_with_ref_and_diffmap_linear_planar(
    &self, pre: &PrecomputedReference,
    planes: [&[f32]; 3], width: usize, height: usize, stride: usize,
    options: impl Into<DiffmapOptions>,
) -> Result<DiffmapResult, ZensimError>;

pub fn compute_with_diffmap(
    &self, src, dst, options,
) -> Result<DiffmapResult, ZensimError>;
```

**`DiffmapResult`** (`diffmap.rs:551-590`):
```rust
pub struct DiffmapResult {
    result: ZensimResult,
    diffmap: Vec<f32>,
    width: usize,
    height: usize,
}
impl DiffmapResult {
    pub fn result(&self) -> &ZensimResult;
    pub fn score(&self) -> f64;
    pub fn diffmap(&self) -> &[f32];           // row-major W×H, no padding
    pub fn into_parts(self) -> (ZensimResult, Vec<f32>, usize, usize);
    pub fn width(&self) -> usize;
    pub fn height(&self) -> usize;
}
```

**`DiffmapOptions`** (`diffmap.rs:36-141`):
- `weighting: DiffmapWeighting` — `Trained` (default, per-scale profile-tracked) / `Balanced` ([0.15, 0.70, 0.15] X/Y/B) / `Custom([f32; 3])`
- `masking_strength: Option<f32>` — contrast masking suppression (typical 2.0-8.0)
- `sqrt: bool` — compresses dynamic range (similar to butteraugli's `sqrt(dc_masked + ac_masked)`)
- `include_edge_mse: bool` — accumulates edge artifact / detail loss / MSE per pixel
- `include_hf: bool` — accumulates HF energy loss / magnitude loss / energy gain (features 10-12)

**`ZensimScratch`** (`streaming.rs:2024-2080`):
```rust
pub struct ZensimScratch {
    pub(crate) dst_planes: [Vec<f32>; 3],
}
impl ZensimScratch {
    pub fn new() -> Self;
}
```

Persists 3 distorted-side XYB plane buffers across calls. Avoids per-call ~25 MB (1080p) / ~99 MB (4K) re-allocation. **Critical: only `compute_with_ref_into` uses it; the diffmap API does NOT have a `_into` variant.**

**Score semantics** (`metric.rs:704-748, 806-815`):

```rust
pub fn score(&self) -> f64;       // 0..100, 100 = identical, HIGHER = BETTER
pub fn raw_distance(&self) -> f64; // unbounded, LOWER = MORE SIMILAR
pub fn dissimilarity(&self) -> f64; // (100 - score) / 100, 0 = identical, higher = worse
pub fn approx_ssim2(&self) -> f64;   // direct power-law fit, MAE 4.4
pub fn approx_dssim(&self) -> f64;
pub fn approx_butteraugli(&self) -> f64;  // 2.365 × raw_distance^0.613
```

Direction: **higher = better** for `score()`, **lower = better** for `raw_distance()`. The `approx_butteraugli` direct mapping has Pearson r=0.713 only (weak correlation per `metric.rs:782-785`); raw_distance with calibrated power-law is the right buttloop input.

**Input formats** (`source.rs:44-63`):

```rust
pub enum PixelFormat {
    Srgb8Rgb,            // tight u8 [r,g,b]
    Srgb8Rgba,           // tight u8 [r,g,b,a]
    Srgb8Bgra,           // tight u8 [b,g,r,a]
    Srgb16Rgba,          // tight u16 [r,g,b,a]
    LinearF32Rgba,       // tight f32 [r,g,b,a] - linear light, INTERLEAVED
}
```

**Linear-planes input** (`metric.rs:1441-1474` + `diffmap.rs:706-795`):
- `precompute_reference_linear_planar([&[f32]; 3], width, height, stride)` — accepts SEPARATE planes with arbitrary stride. **Maps directly to the encoder's `padded_width` pattern.**
- `compute_with_ref_and_diffmap_linear_planar(pre, planes, w, h, stride, options)` — same shape.

**This is a major win.** zensim's linear-planes API matches the cvvdp-gpu W44-PHASE3-B4 shape almost exactly — no sRGB pack roundtrip needed.

### §1.2. GPU crate (`~/work/zen/zenmetrics/crates/zensim-gpu/`)

Module layout (`zensim-gpu/src/lib.rs`):

```rust
pub enum ZensimFeatureRegime { Basic, Extended, WithIw }
pub fn simd_padded_width(width: usize) -> usize;
pub fn score_from_features(features: &[f64], weights: &[f64]) -> f64;
pub enum Error { ... }
```

**`Zensim<R>` runtime-generic** (`pipeline.rs:121-251`):

```rust
impl<R: Runtime> Zensim<R> {
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Result<Self>;
    pub fn new_with_memory_mode(client, w, h, memory: MemoryMode) -> Result<Self>;
    pub fn new_with_regime(client, w, h, regime: ZensimFeatureRegime) -> Result<Self>;
    pub fn new_with_regime_budget(client, w, h, regime, budget) -> Result<Self>;

    pub fn compute_features(&mut self, ref_srgb: &[u8], dist_srgb: &[u8])
        -> Result<[f64; TOTAL_FEATURES]>;
    pub fn compute_features_vec(&mut self, ref_srgb: &[u8], dist_srgb: &[u8])
        -> Result<Vec<f64>>;

    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> Result<()>;
    pub fn clear_reference(&mut self);
    pub fn has_cached_reference(&self) -> bool;
    pub fn compute_with_reference(&mut self, dist_srgb: &[u8])
        -> Result<[f64; TOTAL_FEATURES]>;
    pub fn compute_with_reference_vec(&mut self, dist_srgb: &[u8])
        -> Result<Vec<f64>>;
}
```

**`ZensimOpaque` runtime-agnostic facade** (`opaque.rs:264-540`):

```rust
impl ZensimOpaque {
    pub fn new(...) -> Result<Self>;
    pub fn new_with_memory_mode(...) -> Result<Self>;

    pub fn compute_features_srgb_u8(...) -> Result<[f64; TOTAL_FEATURES]>;
    pub fn compute_features_pixels(...) -> Result<[f64; TOTAL_FEATURES]>;
    pub fn compute_features_vec_srgb_u8(...) -> Result<Vec<f64>>;
    pub fn compute_features_vec_pixels(...) -> Result<Vec<f64>>;

    pub fn compute_with_reference_srgb_u8(&mut self, dis_rgb: &[u8])
        -> Result<Vec<f64>>;
    pub fn compute_with_reference_pixels(&mut self, d: PixelSlice<'_>)
        -> Result<Vec<f64>>;

    pub fn compute_srgb_u8(...) -> Result<Score>;
    pub fn compute_srgb_u8_with_codec(...) -> Result<Score>;
    pub fn compute_pixels(...) -> Result<Score>;

    fn score_from_profile_vec(...) -> Score;  // (internal)
    fn score_from_linear(&self, [f64; TOTAL_FEATURES]) -> Score;  // (internal)
}
```

**Critical gaps in zensim-gpu**:

1. **NO diffmap output.** Every API returns either a 228/300/372-element feature vector OR a scalar `Score`. The CPU side's `DiffmapResult` has no GPU counterpart.
2. **NO linear-planes input.** Every API takes `&[u8]` sRGB-u8 packed format. The CPU side's `precompute_reference_linear_planar` has no GPU counterpart.
3. **NO warm-reference path with diffmap.** `set_reference(ref_srgb)` exists, but the subsequent `compute_with_reference_*` returns only features/score — no per-pixel signal.

These three gaps are the structural blockers Phase 1 of the zensim fork must close.

## §2. Per-requirement audit against `RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`

| § | Requirement | zensim CPU has? | zensim GPU has? | Gap | Effort | Phase |
|---|---|---|---|---|---|---|
| §1.1 | Direction convention (smaller=better at trait boundary) | ✓ `raw_distance()` direction OR negate from `score()` | ✓ `Score` carries the same direction surface | Wrap in backend impl; mirror cvvdp's `(10.0 - jod).clamp` shape | small (LOC: ~5 lines per backend) | 3 |
| §1.2 | Numerical range/finite/f64 | ✓ `score()` returns `f64`, range [0, 100] | ✓ same | None | 0 | — |
| §1.3 | Per-distance target table | absent | absent | Build `vardct/zensim_targets.rs` (mirror cvvdp_targets.rs shape; 7-entry seed table from baseline sweep) | medium (~1 week incl. sweep) | 4 |
| §1.4 | Identity → zero scalar | likely ✓ (need unit test) | likely ✓ | Confirm via unit test in backend impl | small (1 test) | 3 |
| §2.1.1 | Identity → zero diffmap | ✓ via DiffmapResult (need unit test to confirm 1e-7) | **MISSING — no diffmap API** | CPU: add invariant test. GPU: add diffmap kernels. | small (CPU) + large (GPU, ~2 weeks) | 1 |
| §2.1.2 | Non-negative diffmap | ✓ (SSIM error is intrinsically non-negative; `include_hf` features use `max(0, _)`) | **MISSING** | Confirm via unit test | small | 1, 3 |
| §2.1.3 | Monotone in distortion | likely ✓ (per design); confirm with synthetic-distortion unit test | **MISSING** | Add unit test in backend impl (mirror cvvdp's invariant test) | small | 3 |
| §2.1.4 | Spatial localization | likely ✓ (multi-scale fusion preserves locality up to 88×88 — see `diffmap.rs:540-541`) | **MISSING** | Add unit test | small | 3 |
| §2.1.5 | Warm-ref invariance | ✓ via `precompute_reference` → `compute_with_ref_and_diffmap` chain | **MISSING (no warm diffmap)** | Add unit test | small | 1, 3 |
| §2.2 | Strict invariant (`score == minkowski_norm(diffmap)`) | NO (zensim is multi-scale-fused; this is the same honest-stop as cvvdp) | NO | Document as aspirational, NOT a ship gate | 0 | — |
| §3.1 | Per-block reducer constant table (`ZENSIM_BLOCK_CONSTANTS`) | absent | absent | Phase 1 ships placeholder `k_tile_norm: 1.2` (butter parity); Phase 4 fits via Phase 8g-style harness | medium (~3 days incl. tile_dist capture) | 4, 8g |
| §3.2 | Bad-rate parity verification | absent (no capture harness wired) | absent | Mirror `examples/cvvdp_phase8g_tile_dist_capture.rs` for zensim | medium (~1 day) | 4 |
| §4 | Diffmap renormalization scale | absent | absent | Phase 1 ships `1.0` (no renorm); Phase 4 fits via Phase 8c-style harness | medium (~3 days) | 4 |
| §5.1 | Trait surface (`PerceptualBackend`) | satisfied by wrapping in `vardct/zensim_backend.rs` (mirror cvvdp_backend.rs) | satisfied by wrapping | New file ~400 LOC + 5 unit tests | medium (~1 week) | 3 |
| §5.2 | Warm-ref per-cell amortization | ✓ via `PrecomputedReference` (built once per cell, reused per iter) | ✓ via `set_reference(ref_srgb)` + `compute_with_reference` | None on CPU; on GPU the path returns features-only — need diffmap variant of `compute_with_reference` | small (CPU) + large (GPU; covered by §2.1.1 gap closure) | 1, 3 |
| §5.3 | Linear-planes input | ✓ `precompute_reference_linear_planar` + `compute_with_ref_and_diffmap_linear_planar` | **MISSING — only sRGB-u8** | Add `score_from_linear_planes` family to `Zensim<R>` + `ZensimOpaque`, mirroring cvvdp-gpu's `8b658b4` commit shape | large (~1.5 weeks) | 1 |
| §5.4 | `&mut Vec<f32>` diffmap output | ⚠ DiffmapResult OWNS the diffmap Vec — no `_into` variant for buffer recycling | **MISSING** | Add `compute_with_ref_and_diffmap_linear_planar_into` (CPU) that fills caller-owned `&mut Vec<f32>`. GPU: covered by §2.1.1 closure (build the API right from the start). | small (CPU) + bundled (GPU) | 1, 3 |
| §5.5 | Stride convention (`padded_width` distorted) | ✓ via `stride: usize` arg on linear-planar APIs | depends on GPU API design | Wrap to pass stride through | small | 3 |
| §6.1 | Wall-time per-iter budget | TBD — bench needed @ 1024² | TBD — bench needed @ 1024² | Phase 1 closing chunk benches the CPU+GPU paths; if > 250 ms, OPT_IN_ONLY default | bench-only (~1 day) | 1 |
| §6.3 | Diffmap overhead measurement | absent | absent | Mirror `crates/cvvdp-gpu/benchmarks/diffmap_overhead_2026-05-24.tsv` shape | bench (~1 day) | 1 |
| §7.1 | `catch_unwind` GPU init | N/A (CPU only) | absent (need new init path) | Bundle into Phase 1 GPU diffmap API | small (~1 day, bundled) | 1, 3 |
| §7.2 | Silent CPU fallback chain | N/A | N/A | Phase 3 extends `construct_backend` dispatch with zensim branch (mirror cvvdp Phase 3 + 5) | small (~1 day) | 3 |
| §7.3 | `EncoderStrategy::Libjxl` bypass | N/A | N/A | Phase 3 adds `LossyConfig::resolve_zensim_loop` with Libjxl short-circuit | small (~half-day) | 3 |
| §8 | Pareto-position-pct ≥ 85% | N/A | N/A | Phase 6 5-or-6-backend tracking sweep; Phase 8-style refit if < 85% | large (~2-3 weeks) | 6, 8 |
| §9.1 | All 14 opt-in gates | partial (CPU has scalar + diffmap; GPU has scalar only) | partial (scalar only) | Phase 1-3 close G1-G14 | (sum of above) | 1-3 |
| §9.2 | All 6 default-flip gates | TBD until Phase 6 | TBD until Phase 6 | Decision rule per cvvdp Phase 6 + 8 arc | depends on Phase 6 verdict | 6, 7, 8 |

## §3. Detailed gap analysis

### §3.1. The 3 zensim-gpu gaps (Phase 1 work)

#### §3.1.1. NO diffmap output

**Most important blocker.** Without a per-pixel diffmap, zensim-gpu cannot drive the buttloop's per-block 8×8 reducer.

The CPU side's diffmap fusion logic (`zensim/src/diffmap.rs::compute_zensim_streaming_with_ref_and_diffmap`):

1. Per-scale SSIM error maps from 4-scale pyramid.
2. Coarser scales bilinear-upsampled to base resolution.
3. Channel weights from `DiffmapWeighting` (per-scale trained, or balanced, or custom).
4. Multi-scale blend with `scale_blend` weights.
5. Optional `include_edge_mse` + `include_hf` post-processing.
6. Optional masking via `apply_contrast_masking` + `sqrt`.

**Direct map to a GPU kernel chain**: same 4-scale pyramid (already exists for the feature path), per-scale SSIM error compute (already exists for the feature path), bilinear-upsample-and-blend at base resolution (new kernel — mirror cvvdp-gpu's `kernels/diffmap.rs`).

**Effort estimate**: ~2 weeks of zenmetrics-side work for an agent familiar with cubecl + the zensim-gpu existing pyramid. Mirrors cvvdp-gpu's Phase 1 work scope (`cvvdp_gpu_diffmap_api_shipped_2026-05-24.md`, ~6-day chunk per the memo).

#### §3.1.2. NO linear-planes input

**Same shape as cvvdp's Phase 1 + W44-PHASE3-B4 work.** The GPU pipeline currently expects sRGB-u8 packed bytes; the encoder hands it linear-f32 planes. Two paths:

1. Host-side LUT pack: linear-f32 → sRGB-u8 → upload. Documented cost: 5-15 ms/iter at 1 MP (W44-PHASE3-B1 dead-code retained at `perceptual_backend.rs:1110-1164`). Workable but slow.
2. Native linear-planes upload: skip the LUT entirely. Mirror butteraugli-gpu W44-PHASE3-B4 (11-21% wall savings).

**Recommendation**: Phase 1 implements path 2 directly. The CPU side's `precompute_reference_linear_planar` API has been shipped for ages; the GPU side just needs to catch up.

**Effort estimate**: ~1.5 weeks, bundled into the §3.1.1 diffmap kernel work. Same engineer, same crate, overlapping scope.

#### §3.1.3. NO warm-reference diffmap path

Already covered by §3.1.1 — once the GPU has a diffmap kernel chain, exposing it via `set_reference(ref_planes); compute_with_warm_ref_from_linear_planes(dist_planes, Some(&mut diffmap_vec))` is mechanical. Mirror cvvdp-gpu's `8b658b4` API shape: 7 new pub methods on `Zensim<R>`, 7 mirrored on `ZensimOpaque`.

### §3.2. The 1 zensim-CPU gap (Phase 1 / 3 hybrid work)

#### §3.2.1. NO `_into` variant for the diffmap API

`compute_with_ref_into` (no diffmap) takes `&mut ZensimScratch` and recycles dst_planes. But `compute_with_ref_and_diffmap` and `compute_with_ref_and_diffmap_linear_planar` BOTH return an owned `DiffmapResult { diffmap: Vec<f32>, ... }`.

The buttloop pattern at `vardct/perceptual_loop.rs` keeps `diffmap_vec: Vec<f32>` alive across iters. Calling zensim's diffmap API would force per-iter `Vec<f32>` allocation + GC.

**Two options**:

A. **Add `compute_with_ref_and_diffmap_linear_planar_into(pre, planes, w, h, stride, options, scratch: &mut ZensimScratch, diffmap_out: &mut Vec<f32>) -> Result<ZensimResult, _>`** to zensim's API. Same shape as cvvdp-cpu's diffmap path.

B. **Wrap inside the backend impl**: cache a `ZensimScratch` + a local `diffmap_buf: Vec<f32>` in `CpuZensimBackend`. Call the existing API, then copy `result.diffmap()` into the caller's `diffmap_out` via `diffmap_out.clear(); diffmap_out.extend_from_slice(result.diffmap())`. Adds one `O(W*H)` copy per iter — ~1-2 ms at 1024², negligible vs metric compute.

**Recommendation**: Ship option B in Phase 1 (zero zensim API churn); file option A as a zensim-side follow-on (cleaner but adds an API surface the user isn't currently asking for).

### §3.3. The single biggest content-class structural concern

zensim's design is **multi-scale SSIM with per-scale trained weights** (lib.rs:183-194). It's tuned on a 344k-pair synthetic training set across 6 codecs (JPEG, WebP, AVIF, JXL, PNG, mozjpeg) with codec-specific affine calibration (`compute_with_codec_hint`).

The buttloop's per-block 8×8 reducer was calibrated to butteraugli's near-pointwise max-norm distribution. The cvvdp arc showed that a different metric's per-pixel distribution (cvvdp's Laplacian-pyramid + per-band CSF) needs `K_TILE_NORM` 7.5× smaller (1.2 → 0.16).

**zensim's diffmap is also multi-scale fused** — likely closer in distribution shape to cvvdp than to butteraugli. Expect `ZENSIM_BLOCK_CONSTANTS::k_tile_norm` to land somewhere in the same band as cvvdp's 0.16; the actual value comes from Phase 4 fitting per RFC #1 §3.2.

### §3.4. HDR not supported (out-of-scope)

zensim's input format set is sRGB / BT.709 / linear with sRGB-aware gamut mapping (`source.rs:16-30`). HDR (PQ, HLG) is **rejected** with `UnsupportedFormat`. No `intensity_target` knob.

The encoder's HDR buttloop path (W44-RECON-DEEP/A10 intensity_target dispatch) won't have zensim as an option. zensim opt-in must short-circuit when the active `HdrLoss` is non-SDR. Pattern: `LossyConfig::resolve_zensim_loop` returns false when `resolved_hdr_loss() != HdrLoss::Sdr` (or equivalent).

This is the **same constraint as butteraugli** before W44-RECON-DEEP/A10 — butteraugli also has limited HDR support. Not a zensim-specific blocker.

## §4. Wall-time projections (TBD until Phase 1 bench)

zensim's published CLAUDE.md statement: "Fast psychovisual image similarity metric combining ideas from SSIMULACRA2 and butteraugli." With AVX2/AVX-512 SIMD + rayon. The streaming pipeline allocates `~14 × pixels × 4 B` (per `Zensim::with_max_pixels` doc).

No published wall-time numbers in the source. Phase 1 closing chunk MUST bench:

1. `Zensim::compute_with_ref_into` (score-only, CPU) at 256² / 512² / 1024² / 2048² on photo + screenshot fixtures.
2. `Zensim::compute_with_ref_and_diffmap_linear_planar` (score + diffmap, CPU) at same fixtures.
3. zensim-gpu's `Zensim<CudaRuntime>::compute_with_reference` (score-only, GPU) at same fixtures.
4. (After Phase 1 GPU diffmap lands) zensim-gpu with diffmap at same fixtures.

**Hypothesis** (based on the published "AVX2/AVX-512 SIMD throughout via archmage" + 344k-pair calibration design):

- CPU score-only: likely 50-150 ms @ 1024² (between butter CPU 150 ms and cvvdp CPU 222 ms).
- GPU score-only: likely 15-30 ms @ 1024² (similar to cvvdp GPU's 20-25 ms warm).
- Diffmap overhead: likely +20-40% on top of score-only (similar to cvvdp).

If the hypothesis holds, zensim could be a viable default-on backend per RFC #1 §6.2 threshold (50 ms/iter). If wall is closer to cvvdp-cpu's 222 ms, it stays opt-in-only.

**Phase 1 acceptance gate (binding)**: bench TSV `benchmarks/zensim_wall_baseline_<date>.tsv` captures all 4 sizes × 4 fixtures × 4 backends; report mean + p95 per cell. Same shape as `cvvdp_phase8b_*.meta` provenance.

## §5. Profile selection — which `ZensimProfile` for the buttloop?

The buttloop needs ONE profile per encode (the metric's score interpretation must stay stable across iters). Per `profile.rs:218-265` ranking table (CID22 SROCC / JND CV-MAE):

| Profile                | CID22 SROCC | JND CV-MAE | Notes                                                |
|---                     |---          |---         |---                                                   |
| `PreviewV0_3` (legacy) | **0.9367**  | 0.0078     | 372-feature extended + IW pool path                  |
| `PreviewV0_5Tuner`     | 0.9278      | **0.0044** | Two-trail SOTA framework, ~2× cost                   |
| `PreviewV0_5Ensemble`  | 0.8611      | 0.5733     | Larger ensemble                                      |
| `PreviewV0_5Balanced`  | 0.7800      | 0.7556     | Balanced trail                                       |
| `PreviewV0_5Compression` | 0.7189    | 0.7033     | Compression-tuned                                    |
| `PreviewV0_2`          | not in table | —         | Legacy, 218k synthetic baseline                      |

**Recommendation for Phase 3 default**: `PreviewV0_3`. Higher CID22 SROCC than V0_5Tuner; the buttloop primarily wants accurate inter-image quality discrimination at the same content (which CID22 measures). The 0.0034-pp lower JND CV-MAE on V0_5Tuner doesn't justify the 2× wall cost for a per-iter compute.

`LossyConfig::with_zensim_profile(profile: ZensimProfile)` would let callers override (mirror cvvdp's lack of per-config profile choice — cvvdp ships only one profile from upstream). Recommend exposing this as Phase 3 API surface.

## §6. Decision: is zensim a viable third PerceptualBackend?

**Yes, with caveats clearly identified.**

### §6.1. Strong points

- **CPU diffmap API is already shipped** — `compute_with_ref_and_diffmap_linear_planar` at `zensim/src/diffmap.rs:706`. Direct fit for the buttloop's `set_reference + compare_with_reference` pattern.
- **Linear-planes input on CPU is already shipped** — `precompute_reference_linear_planar`. Skips the sRGB roundtrip.
- **CPU score + diffmap + 4-scale fusion + 5 invariants likely all hold** (need unit tests to confirm; no fundamental API gaps).
- **Score direction normalization is mechanical** — `(100.0 - score)` or `raw_distance` directly.
- **Trained on 6-codec 344k pairs INCLUDING JXL** — better calibrated for codec output than cvvdp (which was tuned against general distortions).
- **Per-codec affine calibration available** via `compute_with_codec_hint("jxl")` — could ship as a Phase 3 option for tighter calibration.

### §6.2. Weak points / blockers

- **NO diffmap output on zensim-gpu** — Phase 1 work (large, ~2 weeks).
- **NO linear-planes input on zensim-gpu** — Phase 1 work (large, bundled).
- **NO `_into` variant for the CPU diffmap API** — Phase 1 work (option B: wrap inside backend, small).
- **NO per-distance target table or per-block reducer constants** — Phase 4 calibration work (medium, ~1 week of bench + fit).
- **Wall-time unknown** — Phase 1 bench required before opt-in/default-flip decision.
- **HDR not supported** — Phase 3 must short-circuit on non-SDR HdrLoss (same as butteraugli's current constraint).
- **Pareto-position-pct unknown until Phase 6 sweep** — could land at 85%+ (Phase 8-free) or below (Phase 8-style refit needed).

### §6.3. Aggregate effort estimate

| Phase | Scope                                                | Effort       | Crate(s)              |
|---    |---                                                   |---           |---                    |
| 1     | zensim-gpu diffmap + linear-planes API; CPU `_into` wrap path | ~2-3 weeks   | zenmetrics + jxl-encoder |
| 2     | PerceptualBackend trait alignment (likely no-op)     | ~1 day       | jxl-encoder           |
| 3     | `ZensimBackend` (CPU + GPU) impl + opt-in API        | ~1 week      | jxl-encoder           |
| 4     | Buttloop wiring + target table + initial bench       | ~1 week      | jxl-encoder           |
| 5     | optional CPU/GPU split refinement (likely tiny)      | ~1-2 days    | jxl-encoder           |
| 6     | 6-backend tracking sweep + Pareto decision           | ~1 week      | jxl-encoder + corpus  |
| 7     | Docs closeout (if OPT_IN_ONLY)                       | ~1 day       | jxl-encoder           |
| 8 (cond) | Pareto refit (renorm + block constants)           | ~2-3 weeks   | jxl-encoder           |

**Total**: ~5-7 weeks if Phase 6 lands OPT_IN_ONLY (no Phase 8 needed); ~8-10 weeks if Phase 8 refit needed (likely, based on the cvvdp arc precedent).

## §7. Comparison with the cvvdp arc

| Aspect | cvvdp at Phase 1 start | zensim at Phase 1 start |
|---    |---                     |---                     |
| Existing scalar API     | yes (cvvdp-gpu shipped) | yes (CPU + GPU shipped) |
| Existing diffmap API    | NO (Phase 1 added it on CPU + GPU)  | yes on CPU (`compute_with_ref_and_diffmap_linear_planar`); NO on GPU |
| Existing linear-planes  | NO on CPU + GPU (Phase 1 added both) | yes on CPU; NO on GPU |
| Existing warm-ref       | yes on GPU; NO on CPU (Phase 5 added it) | yes on CPU (`PrecomputedReference`); yes on GPU (`set_reference(ref_srgb)`) |
| In-house team familiarity | high (zenmetrics owners) | high (zensim crate is internally owned) |
| Codec-specific calibration | no | yes (`compute_with_codec_hint`) |
| Test suite for invariants | NO at Phase 1 start | partial (CPU side has `DiffmapResult` unit tests; GPU has feature parity tests) |

**Net**: zensim is at a STARTING POINT closer to where cvvdp landed at Phase 4. The CPU diffmap API + linear-planes input + warm-ref + codec-specific calibration are all already there. The Phase 1 work is concentrated on the GPU side (diffmap kernels + linear-planes upload) plus the integration scaffolding inside jxl-encoder.

If the Phase 1 GPU work lands in 2-3 weeks, the rest of the arc (Phases 2-6) is the same shape and ~3-4 weeks of work, plus 0-3 weeks of Phase 8 refit. Total budget: 5-10 weeks. **Plannable.**

## §8. Out of scope (for this audit)

- Refactoring zensim's profile selection (V0_3 vs V0_5Tuner) — keep V0_3 default.
- Adding HDR support to zensim — separate multi-month effort.
- Changing zensim's score direction (100 = identical convention is established in the public API).
- Modifying zensim's training corpus / weights / profile — orthogonal to the buttloop integration.
- Per-image content-class metric dispatch (e.g. "use zensim for screenshots, butteraugli for photos") — RFC #3 may surface this.
