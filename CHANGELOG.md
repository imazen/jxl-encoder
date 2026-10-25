# Changelog

## [Unreleased]

- Fixed (#128): ICC profiles that LZ77 compresses past 256:1 (zero or repeated tag payloads, from about 300 KB up) produced files djxl v0.12 rejects as a "Corrupted stream"; libjxl's ICC reader checks the ratio every 65536 decoded symbols. The writer now checks every such prefix and falls back to coding without LZ77, then to a code with at least one bit per symbol. Profiles whose predicted stream is 64 KiB or smaller are byte-identical. Backport from main.
- Fixed: lossless `with_source_gamma` was ignored in the codestream header, which kept saying sRGB. One-shot and streaming lossless encodes now write the gamma, as lossy encodes do. Backport from main.

## [0.3.2] - 2026-10-24

Maintenance release on the 0.3 line. Builds again under fresh dependency
resolution and carries the security/decodability fixes that apply to 0.3.x.
No public API changes. Encoded bytes can differ from 0.3.1 only where a fix
applies (LZ77 match-length cap, e9 lossless property buckets, alpha-bearing
LF/patch/progressive frames, lossless colour-encoding header).

### Fixed

- `jxl-encoder-simd`: build with `--no-default-features` (no_std). Scalar
  paths called std-only `f32` methods; they now go through a `scalarmath`
  veneer that uses the inherent methods under `std` and `libm` otherwise
  (backport of 4705cb6, #38).
- `jpeg-reencoding` feature: build against zenjpeg 0.8.4. zenjpeg 0.7.1
  fails to compile with magetypes 0.9.30. The coefficient copy now uses
  checked arithmetic for attacker-controlled SOF block counts and errors
  when the decoded buffer is shorter than the SOF implies (from d6ec9a7).
- Effort-9 encodes hung on periodic content (optimal LZ77 match extension
  was unbounded). Match length is capped at 1024 in both the optimal and
  greedy paths (from 45a63c3, #27).
- Out-of-bounds index panics in LZ77 chain following and the patches BFS
  (backport of 6796417, #30).
- Streaming encoders validate pixel dimensions like the one-shot path;
  tree learning and the JPEG SOS parser are hardened against malformed
  input (backport of 071adbb, #35).
- Panics on internal-consistency failures in ANS writing, Huffman depth
  overflow, and oversized/empty ICC profiles now return errors; patch
  bin-packing and perceptual-loop iteration counts are bounded (from
  45a63c3, #33).
- `ColorEncoding::write` coded `rendering_intent` as raw bits, so Saturation
  and Absolute produced a header no decoder accepts. In 0.3.1 this was
  reachable only through the public `headers` writers; with the lossless
  colour-encoding fix below it is reachable from `with_color_encoding` too.
  The field is now coded as an enum (backport of 157ebfb).
- Effort-9 multi-group lossless emitted streams no decoder accepts: the
  ref-property stride was cut mid 4-slot group, and tree property 1
  (`group_id`) used ad-hoc ids instead of the spec stream id (backport of
  5eefe5f and 329f207, #68).
- Effort-9 property quantisation kept up to 256 thresholds while bucket
  indices are `u8`, so the top bucket wrapped to 0. The tree learner then
  grew invalid nodes up to its cap and the validator repaired them one at a
  time: a 259x133 RGBA lossless e9 encode went from 36 s / 7,460 bytes to
  0.41 s / 4,154 bytes (e8 unchanged). Thresholds are now capped at 255.
- Lossless `GrayAlpha8`/`GrayAlpha16` (one-shot, streaming and animation)
  wrote a header without the alpha extra channel; every such file was
  undecodable (backport of c6e20f6).
- Lossy RGBA with progressive AC on a single-group (<=256x256) image moved
  alpha out of LfGlobal, producing an undecodable file (placement fix from
  0ca49f0).
- `with_lf_frame(true)` and patches (lossy e7+, lossless e5+) on images with
  alpha wrote LF / patch-reference frames without the declared extra
  channel, and patch positions without per-extra-channel blend modes; the
  files were undecodable. Those frames now carry zero-filled placeholder
  extra channels, as libjxl does.
- aarch64 and wasm32 builds failed with magetypes 0.9.27+ (retired
  per-platform SIMD structs); kernels use the generic API and the
  archmage/magetypes floor is 0.9.27 (backport of 4c09d53, #137).
- Lossless encodes ignored `with_color_encoding` and always signalled sRGB
  (backport of 52976f1, #17).
- `jxl-encoder-cli` no longer enables `unsafe-performance` by default.

## [0.3.1] - 2026-05-02

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->
- `EffortProfile` and `EntropyMulTable` will become `#[non_exhaustive]`
  so we can grow them additively without breaking external struct-literal
  constructions. Callers that construct via struct literal must switch
  to `EffortProfile::lossy(effort, mode)` /
  `EffortProfile::lossless(effort, mode)` /
  `EntropyMulTable::reference()` / `EntropyMulTable::experimental()`
  and mutate fields as needed. Already in main; held for next minor bump.
- The crate-root `EffortProfile` re-export is now `#[doc(hidden)]`. New
  expert callers must use `LossyInternalParams` / `LosslessInternalParams`
  via the segmented `with_internal_params` setters instead.

### Added

- Picker / sweep escape hatch behind new `__expert` cargo feature
  (eebd561, 6bdab0b, 25bb80f and follow-up; renamed from
  `unstable-tuning-knobs` for cross-codec consistency with
  zenavif/zenwebp/zenravif). The double-underscore prefix signals
  "private — do not depend on this in production code." Default API
  surface is unchanged when the feature is off.
- **Segmented expert surface**: `LossyInternalParams` and
  `LosslessInternalParams` structs (gated `__expert`) replace the single
  `EffortProfile` knob bag. Each carries `Option<T>` fields for the knobs
  the corresponding encode mode actually reads, applied via
  `LossyConfig::with_internal_params(LossyInternalParams)` and
  `LosslessConfig::with_internal_params(LosslessInternalParams)`.
  - **Why**: the type system enforces mode-correctness — lossy-only knobs
    (AC strategy gates, CfL, cost-model constants) cannot be passed to
    the lossless setter, and modular-only knobs (RCT search, WP scan,
    tree-learning shape) cannot be passed to the lossy setter. Pickers
    can train per-mode independently because the input space is
    disjoint by construction. Matches the segmented `InternalParams`
    pattern used in zenavif / zenwebp / zenravif.
  - **`LossyInternalParams` fields** (13): `try_dct16`, `try_dct32`,
    `try_dct64`, `try_dct4x8_afv`, `fine_grained_step`,
    `k_info_loss_mul_base`, `entropy_mul_table`, `cfl_two_pass`,
    `chromacity_adjustment`, `patch_ref_tree_learning`, `non_aligned_eval`,
    `enhanced_clustering_vardct`, `k_ac_quant`.
  - **`LosslessInternalParams` fields** (7): `nb_rcts_to_try`,
    `wp_num_param_sets`, `tree_max_buckets`, `tree_num_properties`,
    `tree_threshold_base`, `tree_sample_fraction`,
    `tree_max_samples_fixed`.
  - Both structs are `#[non_exhaustive]` and `Default`; field sets may
    grow additively between minor versions. `with_effort()` preserves
    the params across effort-level changes (the underlying
    `EffortProfile` snapshot is retained).
- `EntropyMulTable` re-exported at crate root (used by
  `LossyInternalParams::entropy_mul_table`).
- Examples (`lossless_pareto_calibrate` / `lossy_pareto_calibrate`)
  rewired through the segmented surface; see imazen/jxl-encoder#24.
- `effort_expert_tests` module gated on `__expert`: per-knob OAT
  (one-at-a-time) coverage for the lossy and lossless internal-params
  surfaces, override-roundtrip checks, and default-baseline
  byte-equivalence tests asserting that an all-`None`
  `LossyInternalParams::default()` / `LosslessInternalParams::default()`
  override produces byte-identical output to the no-override path at
  the same effort + distance.
- `validate()` methods on `LossyConfig`, `LosslessConfig`, and (gated
  `__expert`) `LossyInternalParams` / `LosslessInternalParams`. Returns
  `Result<(), ValidationError>` with one variant per failure mode
  (`DistanceOutOfRange`, `EffortOutOfRange`, `IterCountOutOfRange`,
  `QualityLoopMutuallyExclusive`, `FineGrainedStepOutOfRange`,
  `KInfoLossMulBaseInvalid`, `KAcQuantInvalid`, `NbRctsToTryOutOfRange`,
  `WpNumParamSetsOutOfRange`, `TreeMaxBucketsZero`,
  `TreeNumPropertiesOutOfRange`, `TreeThresholdBaseInvalid`,
  `TreeSampleFractionOutOfRange`, …). `ValidationError` is
  `#[non_exhaustive]`. Existing encode paths still clamp out-of-range
  values; `validate()` is opt-in for batch jobs that prefer fail-fast
  over silent coercion. Cross-param: catches stacking of butteraugli /
  ssim2 / zensim quality loops (mutually exclusive). New `validation`
  module + 37-test coverage matrix (one test per error variant + happy
  paths + cross-param).

### Changed

- **`EffortProfile` becomes an internal type** for back-compat. The
  crate-root re-export is `#[doc(hidden)]`; existing callers continue
  to compile, but new code should reach for `LossyInternalParams` /
  `LosslessInternalParams` via the `with_internal_params` setters.
- **Removed `with_effort_profile_override`** from both `LossyConfig`
  and `LosslessConfig`. Replaced by the segmented
  `with_internal_params(LossyInternalParams)` /
  `with_internal_params(LosslessInternalParams)` setters. Never
  published — `__expert` was renamed before any release shipped — so
  no migration path is needed for external callers; internal harnesses
  (calibrate examples) were rewired in the same change.
- Expanded `EffortProfile` field-level theory docs: pipeline stage,
  override rationale, mechanism (with src/-relative line refs),
  and effort-level interaction now documented for the
  cost-model constants (`k_*`), tree-learning shape
  (`tree_num_properties`, `tree_max_buckets`, `tree_threshold_base`,
  `tree_max_samples_fixed`, `tree_sample_fraction`), modular search
  knobs (`nb_rcts_to_try`, `wp_num_param_sets`),
  coefficient-domain multipliers (`k8x8`/`k16x8`/`k16x16`/`k4x8`/`k4x4`),
  and quantization thresholds (`fixed_thresholds_y`,
  `adjust_thresholds`).

## [0.3.0] - 2026-04-16

### Added

- Custom white point and custom primaries encoding for `ColorEncoding`
  (`WhitePoint::Custom`, `Primaries::Custom`). New `CIExy` and `CustomPrimaries`
  types with convenience constructors `with_custom_white_point()`,
  `with_custom_primaries()`, `with_custom_white_point_and_primaries()`. Bit-level
  U32 encoding follows libjxl's `Customxy::VisitFields`. 24 new tests including
  three roundtrips verified with jxl-rs (8732d1c).

### Changed

- `with_threads(0)` now uses the ambient rayon pool instead of creating a fresh
  `ThreadPool` on every encode. `threads=1` is sequential; `threads>=2` creates
  a dedicated pool. Lets orchestrators control thread count externally via
  `pool.install(|| ...)` (ad7a100).
- Parallelized EPF (steps 0/1/2 and candidate sharpness search), XYB conversion,
  gaborish inverse, and noise denoise across strips and channels under the
  `parallel` feature. Bit-exact vs serial at all thread counts. 1.32x faster on
  CID22 2048x2048 effort=7 q=80 (795 -> 601 ms at 32 threads) (90c9daa).
- Further parallelized XYB bottom-row padding (three independent channels via
  `rayon::join`) and `PixelStatsForChromacityAdjustment::calc` (64-row strips,
  max-reduction). Gated at height >= 256 so short images keep the serial
  early-exit. Cumulative speedup 1.39x vs pre-easy-stack baseline (1a4664e).
- Removed the no-op `safe-mode` feature flag from both crates, CI, justfile,
  README, and examples. All multi-group VarDCT paths are covered by tests (2d71d84).

### Fixed

- Decode failure for images wider than 2048 pixels (more than one DC group). The
  encoder wrote a static context tree while collecting tokens with the WP tree's
  contexts, causing decoders to read wrong histograms. The WP tree's root
  splitval is now dynamic (`num_dc_groups`). Fixes imazen/jxl-encoder#3 (3e2f1eb).
- Display P3 and BT.2020 primaries are now transformed to sRGB before XYB
  conversion. The XYB opsin matrix is defined for sRGB/BT.709 primaries;
  feeding wide-gamut linear RGB directly produced wrong colors. Adds
  `P3_TO_SRGB` and `BT2020_TO_SRGB` 3x3 matrices to both the main and
  rate-control XYB paths. Fixes #7 (2c87854).
- Custom white point and custom primaries paths returned `Error::NotImplemented`
  instead of panicking via `todo!()` on valid-but-uncommon color profiles. Now
  superseded by the full implementation above; the intermediate fix avoided
  runtime panics while the feature was in progress (7649ac1).

## [0.2.0] — 2026-04-01

### Quality — At parity with cjxl e7

Size parity (grand average -0.0% vs cjxl e7) across 41 CID22 images × 9 distances.
Butteraugli and SSIM2 metrics within ±1% at most distances.

**Key quality fixes:**
- Compute adaptive quant on pre-gaborish XYB (was post-gaborish, inflating masking)
- Match libjxl ties-to-even rounding (`round_ties_even()` vs `round()`)
- Fix merge sub-cost entropy_mul adjustments (kFavor2X2 discount was missed)
- Fix EPF sharpness integer division to match libjxl exactly
- Fix global_scale formula to use effort-matched fixed q values
- Remove AC strategy distance gates (match libjxl effort-level gating)
- Correct AdjustQuantBlockAC effort gating (effort >= 5, not <= 5)

### New features

- **Zensim quantization loop** (`--zensim-iters N`, `--features zensim-loop`):
  Alternative to butteraugli loop using zensim psychovisual metric.
  ~2x faster than butteraugli loop with comparable quality improvement.
- **SSIM2 quantization loop** (`--ssim2-iters N`, `--features ssim2-loop`):
  Alternative loop using SSIMULACRA2 for per-block quality refinement.
- **HDR/non-sRGB color encoding** (`with_color_encoding()`):
  Signal custom transfer function, primaries, and white point.
- **LfFrame** (`--lf-frame`): Separate DC frame for progressive display.
- **Progressive encoding** (`--progressive`, `--qprogressive`):
  2-pass or 3-pass coefficient splitting for incremental decode.
- **Splines** (API: `LossyConfig::with_splines()`):
  Gaussian-blurred parametric curves for thin features.
- **Patches/dictionary** (default-on, `--no-patches` to disable):
  Auto-detect repeated patterns in screenshots/UI. 33-47% savings on screenshots.
- **Lossy delta palette** (`--lossy-palette`):
  Near-lossless with error diffusion for palette-like images.
- **Grayscale lossy** encoding.
- **16-bit and float pixel input** (Rgb16, Rgba16, Gray16, GrayAlpha16,
  RgbLinearF32, RgbaLinearF32, GrayLinearF32, GrayAlphaLinearF32).

### Performance

- **2.5x overall speedup** on 1024×1024 photos at effort 7 (release build).
- SIMD (AVX2 + NEON + WASM SIMD128) for 14 hot kernels: DCT/IDCT, XYB,
  quantize, dequant, entropy, gaborish, mask1x1, pixel_loss, block_l2, EPF.
- Parallel transform+quantize, AC tokenization, CfL, AC strategy search.
- 86x faster tree learning (incremental entropy, count_increase buckets, nlog2n LUT).
- Token struct compacted from 12 to 8 bytes. Two-phase re-tokenization eliminates
  AC token storage.
- Fast powf (libjxl fast_math port) replaces libm powf throughout.
- Pre-sized allocations, buffer pooling, early memory release.

### Lossless

- **Beats cjxl e7** on CLIC photos. Average: -0.7% (7 of 8 images smaller).
- Tree learning with 14 predictors, 50% pixel sampling, 256 quantization buckets.
- RCT selection (best of 7 candidates) for multi-group images.
- Per-histogram HybridUint config optimization.
- LZ77: RLE (e7), greedy (e8), optimal Viterbi DP (e9+).
- Squeeze transform (Haar wavelet) opt-in via `.with_squeeze(true)`.
- Lossless patches: 37% savings on screenshots, zero overhead on photos.
- Palette transform with auto-detect.

### Entropy coding

- ANS: 28-config HybridUint optimization, RLE logcount encoding,
  flat distribution cost baseline, precise population cost for shift selection.
- LZ77 for ICC profiles.
- Non-simple context map encoding for >8 histograms.
- Max histogram clusters increased from 64 to 128.
- Content-adaptive block context map (QF-based splitting).

### Bug fixes

- U64 varint encoding for values >= 273.
- Container box headers for >4GB payloads.
- F16 Inf/NaN/overflow rejection.
- ZeroIfNegative clamp in XYB conversion.
- Intensity target scaling in XYB.
- Custom coefficient orders limited to buckets ≤ 6.
- LZ77 distance cost table extended to 139 entries.
- Palette transform bit widths corrected (u2S selectors).
- ANS alias table log_alpha_size consistency across distributions.
- Predictor formulas 10-13 corrected (AverageWest/NorthWest, AverageAll, etc.).

### Dependencies

- archmage 0.9, magetypes 0.9
- butteraugli 0.9
- zensim 0.2 (optional, for zensim-loop feature)
- fast-ssim2 0.7 (optional, for ssim2-loop feature)

## [0.1.3] — 2026-02-14

Initial public release on crates.io. VarDCT lossy + Modular lossless encoder
with ANS entropy coding, 19/27 AC strategies, adaptive quantization,
chroma-from-luma, gaborish, noise synthesis, and butteraugli quantization loop.
