# Changelog

## [Unreleased]

### Added

- **`__internal_recon_hook` cargo feature** (f73765ff, Layer-1 drift invariant):
  process-global hook on the butteraugli loop's final-iteration internal
  reconstruction (planar linear RGB the loop measures butteraugli against,
  cropped to image dims). Re-exported as `vardct::__recon_hook` with
  `set_capture_enabled` / `take_last` / `InternalRecon`. Backs the new
  `tests/buttloop_recon_parity.rs` Layer-1 test that compares the
  buttloop's internal recon vs jxl-rs decode of the SHIPPED bitstream;
  initial run shows max-abs-diff = 0.183 in linear RGB on a CID22 photo
  at d=2.0 e8 (threshold 1e-3, fails by 184×). Test is `#[ignore]` —
  documents the e8 quality-targeting drift root cause from
  memory/quality_drift_investigation_2026-05-15.md, ships green CI.
  Off by default; not stable; debug instrumentation only.
- **Layer-2 buttloop target-distance parity test** (Chunk 2 of the drift
  investigation): `tests/buttloop_target_parity.rs` asserts that for each
  (image, distance) cell at effort 8, the measured Rust butteraugli of
  (encode → jxl-rs decode → linearize → compare) is within +10% of the
  requested `--distance` (libjxl's calibration intent: distance N means
  "max butteraugli ≈ N"). Sweeps the same 3 photos × 4 distances grid as
  the Layer-1 test (clic2025/02809272, cid22/1025469, gb82-sc/graph at
  d=0.5/1.0/2.0/4.0). Initial run: 7 of 12 cells exceed the +10% bound
  (worst: smooth_photo @ d=0.5 measured 0.80 vs target 0.55, ratio 1.6).
  Failure pattern matches the Layer-1 internal-recon divergence: low-d
  cells fail hardest (the buttloop's optimism translates directly into
  bit under-investment). Test is `#[ignore]` — CI passes; the failure
  is the regression target for Chunk 3's fix. Gated behind the default
  `butteraugli-loop` feature; no production behavior change.
- **Dot detection** (closes #19, 8bff5247 + 6dec363d + 14872a54 +
  6c667f6b + 98adc2d4 + 05dd7695): full port of libjxl's
  `enc_detect_dots.cc` star-field / specular-highlight detector.
  Pipeline: weighted XYB energy image (Gaussian-0.65 vs 2×Gaussian-3
  background) → 7-neighbor flood-fill connected components (cap
  1000 px / 5×5 window) → 2D anisotropic Gaussian ellipse fit
  (1st/2nd central moments + 2×2 eigendecomposition + LSQ
  intensity refit) → quality filter (l2/custom losses, intensity,
  centroid alignment). Surviving dots promoted to a fresh
  `PatchesData` via new `from_dots()` and routed through the
  existing patches subtract → quantize → reconstruct pipeline.
  Default off (`LossyConfig::with_dot_detection(true)`); auto-gates
  at effort >= 7 + distance >= 3.0 like libjxl. Niche feature
  (astronomy / specular-on-dark content).
- **CfL for JPEG recompression** (closes #16, ff54ef1f): full port
  of libjxl's `enc_frame.cc:855-941` JPEG-CfL search. New
  `vardct/chroma_from_luma::jpeg_cfl_search` builds a per-tile
  histogram of YtoX/YtoB multipliers that zero each chroma AC
  coefficient (after subtracting `RatioJPEG(factor) * Y` in fixed
  point), picks the multiplier with most zeros above the
  offset_sum baseline. Wired into `jpeg/encode.rs` for 4:4:4 YCbCr
  3-component JPEGs; other shapes (4:2:0, 4:2:2, grayscale) keep
  the zero map (libjxl behavior). Targets the 1-3% savings the
  issue described. Gated behind the `jpeg-reencoding` feature.
- **Extra channel types beyond alpha** (closes #9, 79dd06b7 +
  3cb79b80 + 6f5f0ff7 + this commit): new public `ExtraChannel<'a>`
  type with `from_alpha_buf` / `depth` / `spot_color(color)` /
  `selection_mask` / `thermal` / `cfa(idx)` constructors.
  `EncodeRequest::with_extra_channels` builder. **Both** the lossless
  modular path and the lossy VarDCT path now thread arbitrary extras
  end-to-end. Lossy single-group + 1+ non-alpha extras and lossy
  multi-group + N extras-beyond-alpha both encode and decode through
  djxl. New `VarDctEncoder::encode_with_extras(...)` accepts an
  arbitrary `&[ExtraChannel<'_>]`; the existing
  `encode(... alpha: Option<&[u8]>)` becomes a thin wrapper. Internal
  `vardct/extras.rs` module + `VardctExtra<'a>` view make the alpha
  sub-bitstream writer generic over N channels (u8 + u16, dim_shift =
  0). Pending run is flushed at every channel boundary so a uniform
  end-of-channel doesn't leak into the next channel's residuals.
  `FrameEncoder::num_extra_channels` derivation widened from
  alpha-only (`if has_alpha { 1 }`) to channel-count-based
  (`channels.len() - num_color`). Lossy + extras + `resampling > 1`
  rejects up front (extras at the original dims while RGB downsamples
  is a follow-up); lossy + Alpha-typed extra + Alpha pixel layout
  rejects to avoid silent double-alpha. Tests cover RGB+Depth (lossless
  + lossy), Gray+Spot, RGBA+Depth, RGBA+SpotColor,
  RGBA+Depth+SpotColor (6 channels), lossy multigroup
  RGB+Depth (300×300), lossy multigroup RGBA+Depth+Spot (300×300),
  resampling rejection, double-alpha rejection.

- **`LossyConfig::with_perceptual_optimizations(bool)`**: convenience
  switch toggling all encoder-side perceptual heuristics in one
  call. Mirrors libjxl's `cparams.disable_perceptual_optimizations`
  (`enc_heuristics.cc:215`, `enc_frame.cc:282`,
  `enc_patch_dictionary.cc:637`). `false` disables gaborish,
  patches, dot detection, noise, pixel-domain loss in one go;
  `true` resets to libjxl-faithful defaults. Per-knob settings
  called *after* still win. Useful for decoder testing,
  reproducibility, and picker-training without perceptual
  confounds. New `LossyConfig::patches()` and `dot_detection()`
  getters added (the others already existed).

- **`LossyConfig::with_already_downsampled(bool)`**: tells the encoder
  the input is already at the post-resampling resolution; skips the
  internal downsample but still writes the matching `upsampling`
  factor in the bitstream. Mirrors libjxl's
  `cparams.already_downsampled`. Use case: GPU pipeline produces a
  downsampled image at the target encode resolution and wants the
  encoder to honour it (write `upsampling=N`, decoder upsamples,
  file header advertises original dims = `input_dims * N`). Without
  this flag, `with_resampling(N)` would downsample the input again.
  No-op when `effective_resampling() == 1`.

- **`LosslessConfig::with_force_rct(Some(rct))`**: forces a specific
  Reversible Color Transform colorspace, skipping the per-effort RCT
  search. Mirrors libjxl's `cparams.colorspace`. `None` (default)
  keeps the per-effort search; `Some(rct)` applies the given RCT
  directly. Useful for known-best content classes (e.g.
  `RctType::YCOCG` for screenshots), reproducibility, and runtime
  picker output. Threaded through both `select_best_rct` and
  `select_best_rct_at` (handles the post-ChannelCompact case).
  `EffortProfile.forced_rct` + `LosslessInternalParams.forced_rct`
  also exposed for `__expert` picker plumbing.

- **`LossyConfig::with_quant_ac_rescale(Some(r))`**: post-compute
  multiplier on the AC quantiser's `global_scale`. Mirrors libjxl's
  `cparams.quant_ac_rescale` (`enc_cache.cc:99` →
  `Quantizer::ScaleGlobalScale`). `r < 1.0` shrinks `global_scale`
  → finer AC quant → larger files but higher quality; `r > 1.0` is
  the inverse. Useful as a fine-grained quality nudge on top of a
  fixed `distance` (e.g. picker output: "encode at d=1.0 but quant
  AC 5 % finer for this content"). Doesn't change the bitstream's
  reported butteraugli distance — encoder-side tweak only. New
  `DistanceParams::apply_quant_ac_rescale(r)` exposes the
  underlying mechanic. Threaded through all three `api.rs` encode
  call sites (one-shot, streaming, animation).

- **`LossyConfig::with_manual_noise_lut(Some(lut))`**: caller-supplied
  8-point noise LUT, third noise source alongside content estimation
  and photon-noise simulation. Mirrors libjxl's `cparams.manual_noise`.
  Priority order matches libjxl `enc_frame.cc:680-689`:
  `with_photon_noise_iso` > `with_manual_noise_lut` > `with_noise`
  (content estimation) > no noise. Values are clamped to
  `[0.0, ~0.9995]` so the 10-bit writer can't trip its debug-assert;
  all-zero LUTs are silently dropped (no noise header emitted, output
  matches no-noise baseline byte-for-byte). Useful when the caller
  has its own noise model (film grain emulation, calibrated sensor
  noise from downstream metadata).

- **`LossyConfig::with_original_distance(Some(orig))`**: caller-supplied
  source-image butteraugli distance for re-encode pipelines. Mirrors
  libjxl's `cparams.original_butteraugli_distance` (`enc_frame.cc:100`).
  When set, distance-based heuristics that compare against source
  quality — primarily `x_qm_scale` (`enc_frame.cc:658`, ramped vs
  `[2.5, 5.5, 9.5]` thresholds) — use the caller-supplied source
  distance instead of the target. Useful when re-encoding an
  already-lossy JPEG / JXL: the encoder needs to know the source's
  existing error budget so it doesn't aggressively chroma-quantize as
  if the source were pristine. `None` (default) keeps the existing
  ground-truth-source behaviour. New `DistanceParams::compute_for_profile_with_original`
  exposes the underlying entry point. Threaded through all three
  call sites (one-shot, streaming, animation).

- **`LossyConfig::with_photon_noise_iso(Some(iso))`**: synthesise noise
  parameters from a camera ISO value instead of estimating from
  content. Faithful port of libjxl's `SimulatePhotonNoise`
  (`enc_photon_noise.cc`); matches the `--photon_noise=ISO` CLI flag.
  Closes the libjxl photon-noise feature-parity gap.
  Useful for re-encoding **denoised** photographs (or CGI / HDR
  content) where the caller wants controlled grain matching a target
  camera ISO instead of preserving the source's natural noise.
  Constants match libjxl: 35 mm full-frame sensor, daylight spectrum,
  effective QE 0.2, PRNU 0.5 %, read noise 3 e⁻ RMS. Takes priority
  over `with_noise` (both flag the noise header); negative / NaN /
  zero ISO values are quietly ignored.

- **`LosslessConfig::with_tree_learning_sample_fraction(f)`** (refs #23):
  public knob to dial back the tree-learning sample fraction at e7+
  for a smoother time/size trade between e6 (no tree) and e7
  (full-strength tree). The effort cliff is real — at e7 tree
  learning first turns on and adds ~28× encode time for ~38% size
  win on a single illustration. Lowering the sample fraction (e.g.
  `0.15` instead of the effort-7 default `0.50`) lets callers tune
  between those two extremes without picker / `__expert` access.
  Clamped to `[0.0, 1.0]` so a stray caller can't trip the
  validator. No-op when `tree_learning` is disabled.

- **`estimate_peak_memory_bytes` on both Config types** (refs #11):
  conservative upper bound on the encoder's peak working-set RSS for
  a given (width, height, layout) pair. Models the major
  dimension-driven buffers — linear_rgb, XYB planes, quant_ac, alpha
  — plus a 25 % overhead for unmodelled scratch. Lossless variant
  also accounts for tree-learning state at effort >= 7 and squeeze
  residuals when enabled. Useful for capacity planning and (once #11
  lands) comparing one-shot vs streaming encode cost. Returns
  `Option<u64>` and propagates overflow via `None`.

- **DCT 4×4 / 4×8 / 8×4 NEON + WASM128 dispatch — closes #2**: 12
  new `_neon` and `_wasm128` entry points (one per direction × 3
  shapes × 2 archs) wire the small-block transforms onto the
  cross-platform dispatcher. The 4×4-class kernels stay on the
  scalar body (LLVM auto-vectorises the fixed-index value-returning
  helpers well at this granularity), but they're now reached through
  `#[archmage::arcane]` with the right NEON / WASM128 token, so the
  caller's target_feature context survives the call. Removes the
  last x86_64-only branch from the SIMD module structure. **#2 is
  now fully closed**: every DCT / IDCT shape (4×4, 4×8, 8×4, 8×8,
  16×8, 8×16, 16×16, 32×32, 32×16, 16×32, 64×64, 64×32, 32×64) has
  AVX2 + NEON + WASM128 + scalar paths. If profiling later
  identifies one of the 4×4 shapes as hot enough for hand-written
  per-arch SIMD (a pixel-art / text-on-flat workload that picks
  DCT4×4 frequently), the entry point is ready — only the body
  needs a rewrite. All 6 `dct4::tests::*` pass on x86_64, aarch64
  (NEON, via `cross`), and wasm32 (WASM128, via `wasmtime`).

- **DCT/IDCT 64×64, 64×32, 32×64 NEON + WASM128 SIMD** (refs #2):
  six new SIMD functions in `jxl-encoder-simd` mirror the existing
  AVX2 paths but at 4-wide (f32x4). Same butterfly, same constants,
  same `dct1d_64_batch_*` / `idct1d_64_core_batch_*` recursion into
  the 32-point batch (which itself recurses into the 16-point batch
  — both already have NEON + WASM coverage from the prior tick).
  Dispatcher in `dct_64x64` / `dct_64x32` / `dct_32x64` /
  `idct_64x64` / `idct_64x32` / `idct_32x64` now selects AVX2 → NEON
  → WASM128 → scalar. Closes the second of the three remaining gaps
  in #2 (DCT/IDCT 64×64). Leaves DCT 4×4 (17 funcs) for follow-up.
  All 15 `dct64::tests::*` + `idct64::tests::*` pass on x86_64,
  aarch64 (NEON), and wasm32 (WASM128). Also lifts pre-existing
  `INV_WC64` x86_64-only cfg gate.

- **DCT/IDCT 32×32, 32×16, 16×32 NEON + WASM128 SIMD** (refs #2):
  six new SIMD functions in `jxl-encoder-simd` mirror the existing
  AVX2 paths but at 4-wide (f32x4) rather than 8-wide. Same butterfly,
  same constants, same `dct1d_32_batch_*` recursion into the 16-point
  batch. Dispatcher in `dct_32x32` / `dct_32x16` / `dct_16x32` /
  `idct_32x32` / `idct_32x16` / `idct_16x32` now selects AVX2 → NEON
  → WASM128 → scalar. Closes the largest of the three remaining gaps
  in #2 (DCT/IDCT 32×32). Leaves DCT/IDCT 64×64 + DCT 4×4 (17 funcs)
  for follow-up ticks. All 16 `dct32::tests::*` + `idct32::tests::*`
  pass on x86_64, aarch64 (NEON, via `cross`), and wasm32 (WASM128,
  via `wasmtime`). Also lifts pre-existing `INV_WC32` x86_64-only
  cfg gate and rewrites two `(MASKING_K_MUL * 1e8_f32).sqrt()`
  call sites in `adaptive_quant.rs` to use the
  `crate::scalarmath::sqrt_f32` veneer (was blocking no_std wasm
  builds — `f32::sqrt` is std-only, the veneer dispatches between
  std and `libm` based on cargo features).

- **2×/4×/8× input resampling for high-distance encoding** (closes #12,
  46b4b78 + 5ecc0c1 + c3a9b5d + 4e4d186): new
  `LossyConfig::with_resampling(factor)` accepts 1/2/4/8; the encoder
  downsamples input via box filter (4×/8×) or libjxl's 12×12 sharper
  kernel (2×) before encoding, signals the decoder to upsample after
  rendering, and reports original dimensions in the file header.
  `LossyConfig::with_auto_resampling(bool)` (default on) engages 2×
  sharper at distance ≥ 10 with internal distance scaled to
  `d * 0.25 + 0.25`, matching libjxl `enc_frame.cc:103-115`.
  Effective values queryable via `effective_resampling()` /
  `effective_distance()`.
- **Center-first AC group permutation** (closes #14, 7f6cb30 + d864de4):
  `LossyConfig::with_center_first(true)` reorders multi-group AC
  sections in concentric-square order from the image center via
  Lehmer-coded TOC permutation, so progressive renderers display image
  centers first. No-op for single-group images. libjxl
  `cparams.centerfirst`.
- **Brotli-compressed metadata boxes (`brob`)** (closes #15, 7ffec89 +
  9574429): new `with_brotli_metadata(bool)` builder on `LossyConfig`
  / `LosslessConfig`; EXIF / XMP attachments larger than the
  break-even threshold are wrapped in `brob` container boxes when
  enabled. Gated behind new `brotli-metadata` cargo feature.
- **Per-component PQ / HLG / BT.709 inverse OETF input**
  (closes #17, 6d7ff63 + 6c7233e + 2d0dbfd + 4fd6dbf + 8f63649 +
  457e5bb): `EncodeRequest` accepts u8, u16, and Gray / GrayAlpha
  variants for ST 2084, BT.2100 HLG, and Rec. BT.709-6 transfer
  functions; the encoder linearizes per-pixel before XYB conversion.
  Streaming path matches one-shot bit-exact.
- **`PixelLayout::*LinearF16` (FP16) inputs** (closes FP16 portion of
  #18, cc6cf23): new layouts accept half-precision linear RGB / RGBA
  / Gray / GrayAlpha; converted to f32 at the boundary.
- **`EncodeRequest::with_row_stride`** (closes #18, 7d5fbff):
  non-tightly-packed input buffers — caller specifies stride in bytes
  per row, the encoder unpacks into a tightly-packed scratch buffer
  before processing. Preserves the existing tightly-packed fast path.
- **Configurable `bits_per_sample`** (closes bits_per_sample portion
  of #18, 85a95d3 + c8b0c85): `EncodeRequest::with_bits_per_sample`
  signals 10/12/14-bit input precision in the codestream `BitDepth`
  header (vs. the layout-derived 8 or 16). Streaming + lossless
  paths covered.
- **HDR signaling on `EncodeRequest`** (closes #21, 2d71e76):
  `with_intensity_target(nits)` and `with_min_nits(nits)` now
  reachable from the convenience encode path; previously required
  the metadata struct.
- **`ColorEncoding::bt2100_hlg()` preset constructor** (closes #22,
  1d6d749): companion to `bt2100_pq()` for HLG content.
- **Premultiplied alpha round-trip** (closes #13, 1601177 + ed03980 +
  76a1f05): `EncodeRequest::with_premultiplied_alpha(true)` signals
  the codestream's `alpha_associated` bit and unpremultiplies the
  input pre-XYB; the decoder re-premultiplies on output. Lossless +
  lossy + streaming paths covered.
- **`SimplifyInvisible` pre-pass for RGBA lossy encodes** (closes
  #10, 6f7c9fa): smears color values in alpha=0 pixels to a weighted
  average of visible neighbors before XYB conversion, reducing
  high-frequency DCT energy from arbitrary garbage in transparent
  regions. 5–20% smaller files on sprites / icons; near-zero cost on
  photos with mostly-opaque alpha. Default-on; toggle via
  `LossyConfig::with_simplify_invisible(false)`.
- **`__internals` cargo feature for downstream parity testing**
  (c82e05c): exposes selected internal types for jxl-encoder-gpu's
  pre-quantized AC entry points and equivalent crates.

- **`VarDctEncoder::encode_from_precomputed_with_extras`** (8322ab9):
  new public method on `VarDctEncoder` (gated `__pre_quantized`) that
  threads caller-supplied alpha / depth / spot color / selection mask /
  thermal / CFA channels through the precomputed-AC entry point.
  Validates `dim_shift = 0` and `sample-count = width * height` at the
  boundary. The legacy `encode_from_precomputed` now delegates with
  `&[]` for source-compatibility. Closes the long-standing TODO at
  `vardct/encoder.rs:2063` where the precomputed entry silently dropped
  any caller-supplied extras.

- **`VarDctEncoder::encode_from_pre_quantized_ac_with_extras`**
  (b32ed29): companion to `encode_from_precomputed_with_extras` for
  the deeper GPU fast path where DCT + quantize run on the GPU and
  only the per-block coefficient buffers cross the wire. Same boundary
  validation; the legacy `encode_from_pre_quantized_ac` delegates with
  `&[]`. Gated `__pre_quantized`.

- **`VarDctEncoder::encode_from_pre_quantized_ac` entry point**
  (9cdd29e): new top-level entry that skips `transform_and_quantize`
  (forward DCT + quantize + nzeros + float_dc) and goes straight to
  `encode_two_pass`. Caller is responsible for producing per-channel
  `TransformOutput`-shaped data matching what `transform_and_quantize`
  would have emitted. Designed for the GPU encoder fast path; saves
  ~50 ms at 12 MP / d=1.0 vs running `transform_and_quantize` again on
  the CPU. Adds `DCT_BLOCK_SIZE` to `__pre_quantized` exports. Gated
  `__pre_quantized`.

- **`__pre_quantized`: `INV_DC_QUANT`, `quant_weights_dct8`,
  `default_thresholds_dct8`** (1802b31): re-exports for the GPU
  pre-quantized AC producer to build per-channel constants without
  reimplementing libjxl tables. Gated `__pre_quantized`.

- **`__pre_quantized`: `TransformOutput` + `transform_and_quantize_for_test`**
  (7bfbeb1): re-exports the per-group transform-output struct and a
  test helper that drives `transform_and_quantize` end-to-end, so
  downstream callers can produce parity-test fixtures without
  reimplementing the inner pipeline. Gated `__pre_quantized`.

- **`__pre_quantized`: `refine_cfl_map`** (e03cff1): re-export of the
  per-tile CfL refinement helper for downstream pipelines (notably
  jxl-encoder-gpu) that compute encode-side CfL on the GPU and want
  the second-pass refinement on the host. Gated `__pre_quantized`.

- **`__pre_quantized`: `adjust_quant_field_with_distance`** (6e25844):
  re-export of the post-`AdjustQuantBlockAC` quant-field rescaler so
  downstream callers can match the CPU `compute_quant_field_float`
  →`adjust_quant_field_with_distance` two-step exactly. Gated
  `__pre_quantized`.

- **`__pre_quantized`: patches detection + `EncoderPrecomputed::with_patches_data`**
  (e23a1b2): exposes the libjxl-parity patches detect/subtract pipeline
  (`find_and_build_patches`, `PatchesData`) and a setter on
  `EncoderPrecomputed` to attach pre-built patches data when the GPU
  pipeline runs detection on the host (case-1 routing per libjxl
  `enc_frame.cc`). Gated `__pre_quantized`.

- **EPF dynamic sharpness wired into `encode_from_precomputed`**
  (16d4356): the GPU pre-quantized entry was passing `None` for
  `sharpness_map`, leaving the bitstream emitting uniform `sharpness=4`
  on the GPU fast path. Now mirrors the CPU `encode_image_lossy`
  path — gated on `params.epf_iters > 0 && distance >= 0.5 &&
  profile.epf_dynamic_sharpness`, falls back to `compute_mask1x1`
  when `EncoderPrecomputed.mask1x1` is `None`. Closes Gap B from the
  GPU buttloop RD-gap chase. CPU bitstream byte-identical.

- **Patches detect/subtract on PRE-gaborish XYB in
  `compute_with_budget` + `encode_from_precomputed`** (f41d59c +
  0c463ec): patches detection now runs on pre-gaborish XYB so the
  detected pattern roundtrips correctly through the decoder pipeline
  (IDCT → gaborish → EPF → patches per libjxl `dec_cache.cc:148-194`).
  Bonus rate-control CLI gaborish gate fix mirrors `api.rs:3842`'s
  `distance > 0.5` check. Screenshot ratios at d=0.5: terminal
  1.327→1.094, codec_wiki 0.927→0.857, windows95 1.354→1.136, imac_g3
  0.574→0.551 — all BEAT the default API path. Default-path bitstream
  byte-identical (hash_lock 36/36 green); RD regression 18/18 photos
  pass.

- **`ExtraChannel::with_dim_shift`** (ddb07b9): builder method to
  declare an extra channel at a downsampled resolution (depth maps at
  1/2, 1/4, …). `dim_shift` enters the bitstream as the channel's
  per-channel resolution shift; the lossless modular path serialises
  the channel at the matching dimensions.

- **16-bit extra channels** (54ae465): new `ExtraChannelBuf` enum
  (`U8(&[u8])` / `U16(&[u16])`), `ExtraChannel::depth_u16` constructor,
  and `ModularImage::push_extra_channel_u16` so depth / spot / thermal
  / CFA extras can carry full 16-bit precision instead of being capped
  at 8 bits. Lossless modular path threads `u16` end-to-end.

- **CLI: 6 libjxl-parity knobs surfaced on `cjxl-rs`** (4a8b876 +
  391058f): new flags wire the new API additions into the CLI.
  - `--photon-noise-iso ISO` → `with_photon_noise_iso`
  - `--original-distance D` → `with_original_distance`
  - `--quant-ac-rescale R` → `with_quant_ac_rescale`
  - `--force-rct {none|ycocg|…}` → `with_force_rct`
  - `--no-perceptual-optimizations` → `with_perceptual_optimizations(false)`
  - `--tree-learning-sample-fraction F` →
    `with_tree_learning_sample_fraction`
  Threaded through both lossless animation and one-shot paths.

### Performance

- **Parallel DC + AC entropy code build via `rayon::join`** (ade20b4):
  the DC entropy code build and the per-pass AC entropy code builds in
  `encode_two_pass_to_writer` are independent (disjoint token streams,
  distinct outputs) but ran sequentially. Wraps both into closures
  joined by `rayon::join` (sequential fallback when `parallel` is off).
  Adds `parallel_join` helper to `crate::parallel` and env-var-gated
  phase timing (`__JXL_ENC_PHASE_TIMING`). Measured at 12 MP / d=1.0:
  `build_codes` ~84→68 ms, u8 path median 572→491 ms (-81 ms).

- **Parallel-reduce token accumulation across groups** (4da4039):
  `build_entropy_code_ans_from_token_groups` Phase A (per-context
  histogram + value-frequency accumulation) was sequential over input
  token groups (~30-40 ms single-threaded at 12 MP). Now `par_iter`s
  over groups, builds a per-group accumulator on each worker, and
  reduce-merges via the existing associative `AccumulatedAnsData::merge`.
  Sequential fallback when `parallel` is off or there's only one group.
  Measured at 12 MP / d=1.0: `build_codes` ~68→30 ms (-38 ms),
  end-to-end median 486→450 ms.

- **Horizontal-band parallel reduce of `count_zero_coefficients`**
  (55ef5ba): the per-encode coefficient-zero counter was a sequential
  double loop over `xsize_blocks × ysize_blocks` (~20 ms single-threaded
  at 12 MP). Now splits the y-axis into up to 16 horizontal bands;
  per-band accumulate into a fresh counts grid; reduce-merge at the end.
  Safe to split on arbitrary y boundaries because `is_first` only
  matches at the top-left sub-block of a multi-block strategy. Measured
  at 12 MP / d=1.0: phase 20→5 ms, encode_two_pass total 70→55 ms,
  u8 end-to-end median 450→444 ms.

- **Flat `Box<[T]>` per-group result storage in transform**
  (348a467): `GroupTransformResult` previously held `[Vec<Vec<T>>; 3]`
  for `quant_dc` / `quant_ac` / `nzeros` / `raw_nzeros` — ~400 mallocs
  per 32×32 group at full size, ~80 000 small allocations per encode at
  12 MP. Now `[Box<[T]>; 3]` flat-indexed as `[ly * width + lx]` —
  one allocation per field per channel per group, ~5× fewer mallocs
  total. Allocator pressure drops materially. Updates 30+ access sites
  in `transform.rs` and `quantize_ac_block`.

- **`scalarmath` uses inherent `f32` methods under `std`** (7dda253):
  the no-`std` `libm` veneer added in #38 (`f15b90c`) had been
  routing `floor` / `sqrt` / `mul_add` / `round` / `round_ties_even`
  through `libm` even on `std` builds, missing hardware FMA on x86_64
  / aarch64. Now dispatches via cargo features: `std` builds use the
  inherent methods (LLVM emits `vfmadd*` etc.); `no_std` keeps `libm`.
  Zero behaviour change; measurable speedup in the SIMD math hot paths.

### Fixed

- **CI clippy/lint cleanup from the `__pre_quantized` API expansion this
  week** (refs e23a1b2, 7bfbeb1, 348a467, 6e25844, e03cff1, f41d59c):
  five workspace clippy errors broke `cargo clippy --workspace -- -D warnings`
  on main. `TransformOutput::new` exposed `pub(crate) MemoryBudget` in
  its `pub` signature (`private_interfaces`); now `pub(crate)` — the
  struct itself stays `pub` for `__pre_quantized` re-export and downstream
  callers obtain instances via `transform_and_quantize_for_test`.
  `compute_mask1x1` is `pub` for `__pre_quantized` re-export but has no
  default-features non-test caller; gated with
  `#[cfg_attr(not(any(test, feature = "__pre_quantized")), allow(dead_code))]`.
  `coeff_order::merge_into`'s outer `&mut Vec<Vec<Vec<i64>>>` parameter
  is index-only (no resize/push/pop on the outer Vec); changed to
  `&mut [Vec<Vec<i64>>]`. `GroupTransformResult` doc had a `+` continuation
  the new clippy parsed as a list item; reworded to "plus" so the
  paragraph reads cleanly without indent gymnastics. `transform_and_quantize`
  takes 11 args; added `#[allow(clippy::too_many_arguments)]` with a comment
  explaining why packing into a struct would force per-call unpacking on
  the per-group parallel reduce (internal hot path, three call sites all
  in this crate).

- **Gaborish ordering in animation-frame path** (fb26368): the
  animation-frame entry point `encode_frame_to_writer` in
  `vardct/bitstream.rs` applied `gaborish_inverse` BEFORE
  `compute_quant_field_float_with_budget`, opposite of both
  still-image paths and of libjxl `enc_heuristics.cc:1117-1142`.
  Effect: gaborish sharpens edges → inflates per-block masking →
  adaptive-quant produces different quant values than the still-image
  paths, so animation-frame encodes diverged from same-pixel
  still-image encodes. Reordered to mirror the still-image paths
  exactly: `compute_quant_field_float_with_budget` on PRE-gaborish XYB
  (with `distance_for_iqf = distance * 0.62` when gab is off),
  `quantize_quant_field`, then `gaborish_inverse`. CLAUDE.md "Gaborish
  ordering (1af2202)" had documented the equivalent still-image bug;
  only the animation path had been missed.

- **Cross-group AC strategy OOB panic in
  `vardct/transform.rs`** (6001b74): `AcStrategyMap::set` silently
  wrote multi-block strategies (DCT64×64, DCT32×32, …) past 32×32-block
  pass-group boundaries in release builds — the existing
  `debug_assert` was a no-op outside debug. The group transform
  pipeline then OOB'd at `transform.rs:544` with `index out of bounds:
  the len is 1024 but the index is 1048` when writing per-block DC
  values. The in-tree per-tile strategy search satisfies the invariant
  naturally (tiles align with groups), but downstream callers of
  `__pre_quantized::EncoderPrecomputed::from_parts` (e.g.
  jxl-encoder-gpu's strat-search injector) can supply an
  `AcStrategyMap` whose entries straddle a group / image boundary, and
  untrusted producers shouldn't crash the encoder. Repro at
  `tests/transform_oob_repro.rs` hand-crafts a DCT64×64 placement at
  `(bx=25, by=25)` on a 64×64-block grid (= 2×2 groups).

- **`refine_cfl_map` accumulator OOB clamp** (4400284): the per-tile
  coefficient accumulator (`coeffs_yx` / `coeffs_x` / `coeffs_yb` /
  `coeffs_b`) is sized at `TILE_DIM_IN_BLOCKS² × DCT_BLOCK_SIZE = 4096`
  floats — same as libjxl's `kColorTileDim²`. The libjxl heuristic
  that gates on cumulative size (`enc_chroma_from_luma.cc:304`) checks
  `covered + tile_origin > tile_end` against the TILE start, not the
  current block's `(bx, by)`. Multi-block first-blocks near the
  tile-end edge therefore aren't filtered out and contribute their
  full `(covered_x × covered_y × 64)` coefficients to *this* tile.
  In pathological `ac_strategy` configurations the cumulative sum
  exceeds 4096 — libjxl writes past via SIMD stores and treats the
  tail as undefined; we panic in release with `index out of bounds:
  the len is 4096 but the index is 4096`. Found while wiring CfL
  pass 2 into the GPU buttloop. Fix: clamp writes to remaining
  capacity, label the outer block-loop and break out once full. CfL
  is a least-squares fit; dropping the small tail past the
  accumulator is benign relative to the panic.

- **`--features __pre_quantized` build regression** (acc7502):
  `compute_quant_field_float_free` and `EncoderPrecomputed::from_parts`
  were re-exported from `pub mod __pre_quantized` (commit 83253aa)
  but the underlying functions only lived on the unmerged
  `feat/pre-quantized` branch. `cargo build --features __pre_quantized`
  had been failing on main since 2026-05-11. Both functions are now
  on main with the same signatures as the side branch (gated
  `#[cfg(feature = "__pre_quantized")]`, `#[doc(hidden)]`, unstable
  API) so downstream consumers (notably jxl-encoder-gpu) can target
  main rather than the side branch. Also brought
  `--features rate-control` back to building after the lossy +
  extras-beyond-alpha refactor changed `encode_two_pass`'s signature
  from `Option<&[u8]>` to `&[VardctExtra<'_>]`. 905 default + 954
  all-feature lib tests pass.

- **`num_extra_channels` size coder spec** (refs #9, 6f5f0ff7):
  selector 2 was `Val(2)` instead of `Bits(4) + 2` per jxl-rs
  `#[size_coder(implicit(u2S(0, 1, Bits(4) + 2, Bits(12) + 1)))]`,
  shifting every subsequent header field by 4 bits. Manifested as
  `InvalidFloat` deep in `tone_mapping` / `color_encoding` parse for
  any image with 2+ extra channels. Now decodes cleanly via
  jxl-oxide.
- **Modular `num_color_channels` derivation** (refs #9, 3cb79b80):
  `should_use_palette` (palette.rs) and ChannelCompact in
  `write_modular_stream_with_tree` (encode.rs) used
  `if has_alpha { len - 1 } else { len }`. For RGBA + 1 extra (5
  channels), this would treat the spot/depth/etc as a color
  channel and try to palette-encode 4 channels — wrong. Now uses
  base color set: 1 (gray) or 3 (RGB), regardless of how many
  extras follow.
- **`color_encoding` wired into lossless file header** (closes #17,
  3f8b89b): `LosslessConfig` / `LosslessEncoder`'s `color_encoding`
  override was being silently dropped; the file header is now built
  with the override before write.
- **`row_stride` validated up front** (a2c915d): bad strides
  (`stride < width * bytes_per_pixel`, or `height * stride` overflow)
  are now rejected at `validate_pixels` before any allocation rather
  than later inside `unpack_strided_pixels`. The error message
  shape is preserved; only fail-fast timing changed.
- **EXIF / XMP / ICC metadata size capped + parity across paths**
  (7ab560d): a single `validate_metadata_sizes` helper applies a
  ~1 GB defensive cap on each of ICC, EXIF, and XMP buffers and is
  now wired into `EncodeRequest::encode_inner`,
  `LossyEncoder::finish_inner`, and `LosslessEncoder::finish_inner`
  (previously only ICC was checked, only on the one-shot path).
  Pathological multi-GB metadata previously reached
  `Vec::with_capacity` in the container wrapper and exhausted system
  memory at write time. Empty ICC also remains rejected with a
  clear error message.
- **Tone-mapping validated up front** (29103ed): bad values for
  `with_intensity_target` / `with_min_nits` (NaN, Inf, negative,
  zero peak, peak > f16 max ≈ 65504, min > peak) are now rejected
  with a clean `EncodeError::InvalidInput` at the API surface
  rather than failing deep inside `f32_to_f16_bits` in the file-
  header writer. Wired into all three paths via a new
  `validate_tone_mapping` helper.
- **`source_gamma` + `intrinsic_size` validated up front**
  (c8bcfb7): bad `with_source_gamma` values (NaN, Inf, ≤ 1/255, > 1)
  and `with_intrinsic_size(0, 0)` / above-spec dims now reject at
  the API surface. `source_gamma` matches libjxl's accepted range
  exactly so codestreams round-trip through cjxl/djxl unchanged;
  previously, out-of-range values silently produced garbage encodes
  via overflow in the gamma LUT (`inv_gamma = 1.0 / gamma`).
- **`cfg.validate()` is now auto-invoked on every encode path**
  (5ecc8e6 + 3e133ea): `LossyConfig::validate()` /
  `LosslessConfig::validate()` used to be opt-in; only callers who
  remembered to call them got the full validation. The encode
  pipeline now invokes them automatically at
  `EncodeRequest::encode_inner`, `LossyEncoder::finish_inner`,
  `LosslessEncoder::finish_inner`, and the two
  `encode_animation_*` paths, so distance / effort / iter-count /
  mutual-exclusivity checks fire for every encode regardless of
  caller. New `From<ValidationError>` for `EncodeError`. The
  streaming path in particular was previously silent on
  `LossyConfig::new(50.0)` (above DISTANCE_MAX); now all paths
  reject identically.
- **4 latent serialization bugs in non-alpha extra-channel paths**
  (closes #8, 4cb33e8): enum coder, F16 vs F32 alpha range, CFA
  channel distribution, name-length distribution. Alpha encodes were
  unaffected (covered by the alpha-only fast path); other channel
  types now serialize correctly.

### Changed (security)

- **Post-#30 security follow-ups + bug-masking fixes** (#33,
  125984a): additional bounds checks at entropy-coding hot paths
  surfaced by the #30 audit; previously-silent bug-masking removed in
  favor of explicit error returns.
- **Per-encode allocation budget plumbed through encoder hot paths**
  (#32, d1c01c2): the working-set budget added in 0.3.2 now reaches
  internal allocators, surfacing `EncodeError::AllocationLimit` when
  individual hot-path allocations would exceed the cap rather than
  only at the up-front estimate.

### Fixed (build)

- **`cargo build --no-default-features` now succeeds** (closes #38,
  f15b90c). The `jxl-encoder-simd` crate has `#![no_std]`
  unconditionally but used 35 inherent `f32` methods (`floor`,
  `sqrt`, `mul_add`, `round`, `round_ties_even`) that only exist
  under `std`. New crate-internal `scalarmath` module wraps
  `libm 0.2.16` (`floorf` / `sqrtf` / `roundf` / `roundevenf` /
  `fmaf`); call sites switched. Adds one tiny pure-Rust dep, zero
  measurable cost in `std` builds (LLVM inlines through). Required
  for WASM and embedded targets that disable `std`.

### Removed

- **`unsafe-performance` cargo feature** (#37, 1972037): unused
  perf-only path that opened up SIMD `unsafe` blocks; the safe SIMD
  path covers all production deployments. No public API change.

### Documentation

- **`Lz77Method::Optimal` at e9+ + the jxl-rs decoder bug**
  (refs #29, 674b0a5): in-source comment in `effort.rs` documents
  why we keep `Optimal` as the lossy default at e9+ despite tripping
  a latent jxl-rs decoder bug (5× regression on synthetic gradients
  if we switched to `RLE`; only zenjxl-decoder is affected).
- **`LosslessConfig::with_effort` e6→e7 cliff warning**
  (refs #23, 6b5cdf5): in-source comment surfaces the ~28× encode-time
  jump from e6 to e7 for ~38% size win on typical photos.
- **README**: dropped stale `unsafe-performance` mention (removed
  in #37); refreshed test-count claim from "940+" to "850+" for
  the workspace README (c8913279). The published per-crate README
  is unchanged pending author review.

### Internal (tests + CI)

- **`concurrency: cancel-in-progress` on the CI workflow**
  (061cfe66): rapid push bursts no longer stack 10+ full matrices
  in the runner queue; only the head commit's CI runs for any given
  branch. PR runs use the PR number to keep concurrent reviews
  isolated.
- **Up-front no-default-features build step in CI**
  (cb329ba): catches future regressions of the kind that closed
  #38 (inherent `f32::method()` calls reintroduced into
  `jxl-encoder-simd`).
- **Clippy + format cleanup** (a9fdb0fb + e1d793bd + 83253aad +
  61e5c31a + f508b54f): workspace `excessive_precision = "allow"`
  (libjxl-port heritage), `iter().any` → `contains`, `Range::contains`
  for `0.0..1e-3`-style bounds checks, fold loop-var-only-used-as-
  index, drop two stale clippy warnings (unused mut, redundant
  parens), drop three stale `#[allow(dead_code)]` on `f16` /
  `vardct::epf` / `vardct::reconstruct`, gate `xyb_to_linear_rgb`
  /`xyb_to_linear_rgb_planar` / `apply_epf` on the right
  `cfg(any(test, feature = ...))` so non-loop builds stay clean.
- **Stale-`#[ignore]` test triage** (c5eeaab + f002702e + da2b4bb3
  + 6fe6dcf8): un-ignored 3 lossy-roundtrip tests that pre-dated
  recent encoder fixes (`test_roundtrip_lossy_rgb_d1`,
  `test_roundtrip_lossy_rgb_d2`, `test_dct32x16_16x32_roundtrip`,
  `test_afv_strategy_roundtrip`, `test_tiny_encoder_decode`);
  removed `test_decode_libjxl_tiny_reference` entirely (libjxl-tiny
  is no longer the reference per CLAUDE.md); migrated two
  corpus-using `patches::tests` from buried `if !path.exists()`
  silent-skip to proper `#[cfg_attr(not(feature = "corpus-tests"),
  ignore = "...")]` + `crate::skip_without_corpus!()`. Lib test
  count: 837 → 853 (+16); ignored: 34 → 28 (-6).
- **Hash-lock sidecar entry** for `lossy_rgba_32x32` at 638 bytes
  (61e5c31a): the SimplifyInvisible commit (#10, 6f7c9fa) silently
  changed the byte count from 636 to 638 without updating
  `hash_lock_expected.txt`. CI's "Build native (Linux)" + "Coverage"
  jobs were silently failing; appended the new hash entry.
- **Regression test for `--rate-control` gaborish gate**
  (`jxl-encoder-cli/tests/rate_control_gaborish_gate.rs`, e03c4947):
  invokes the actual `cjxl-rs` binary on a center-crop of the
  committed `frymire.png` fixture and asserts that
  `bytes(--rate-control -d 0.4)` equals
  `bytes(--rate-control -d 0.4 --no-gaborish)` (gate forces gaborish
  off internally below d=0.5, making `--no-gaborish` a no-op).
  Discriminating against the pre-f41d59c "always on at effort >= 3"
  state — verified by reverting the gate locally and observing the
  new test fail at d=0.4. Adds `image = "0.25"` (default-features =
  false, png) as a `dev-dependency` on `jxl-encoder-cli` for runtime
  PNG cropping.

## [0.3.2] - 2026-05-06

### Fixed (security)

- **Two OOB index DoS vectors** in encoder hot paths (#30, 1498053):
  LZ77 chain follows in `entropy_coding/lz77.rs` now masked with
  `window_mask`, and patches.rs flood-fill BFS gained defensive
  bounds checks at queue-pop. Both panics had bit-30 set in the
  failing index (0x40000000 pattern), suggesting a shared upstream
  cause; the fixes are defensive at the panic sites.
- **Hardened encoder DoS surface** across multiple components
  (499ac75): bounded transform-tree growth, capped quant-iteration
  in butteraugli/ssim2 loops, additional bit-reader guards.
- **NaN/Inf sanitization + dimension arithmetic** (f178000): float
  inputs now sanitized at the boundary; width × height × channel
  arithmetic uses checked multiplies to prevent overflow into
  small-allocation paths.
- **Silent defenses made loud + quant-iter cap aligned with
  validator** (3767210): defenses that previously degraded silently
  now surface `EncodeError`, and the per-component quant-iteration
  cap matches the validator-side limit to prevent inconsistent
  reject/accept behavior.

### Changed

- **Up-front working-set precheck against memory cap** (061862f):
  `Limits::with_max_memory_bytes(n)` is now enforced at
  `EncodeRequest::encode_inner` via an estimate of peak working-set
  (~40 bytes/pixel). Encodes that would exceed the cap return
  `EncodeError::LimitExceeded` immediately rather than allocating.
  Default cap is `DEFAULT_MAX_MEMORY_BYTES = 2 GB` when `Limits` is
  unset. Internal `MemoryBudget` type added (`pub(crate)`) for
  per-allocation accounting; no public API change.

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
