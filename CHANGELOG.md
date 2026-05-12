# Changelog

## [Unreleased]

### Added

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

### Fixed

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
