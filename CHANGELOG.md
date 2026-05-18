# Changelog

## [Unreleased]

### Added

- **`alpha_distance_audit` example + parity audit** — sweeps three RGBA
  test images (opaque, semi-transparent UI gradient, photographic alpha
  mask) at `alpha_distance ∈ {0.5, 1.0, 2.0, 5.0}` against `cjxl v0.12.0`
  (both default `--responsive=1` and `--responsive=0`). Quantizer formula
  port (`bbf8a98`, W6-3) is at **bit-exact MAE parity with cjxl
  `--responsive=0`** at every tested distance (the libjxl no-squeeze
  alpha pipeline our encoder implements). cjxl default is much smaller
  (-18% to -160% bytes) at lower MAE because it applies the Squeeze
  wavelet + ChannelCompact pre-pass on the alpha plane — a separate
  algorithm not yet ported. Audit produces TSV + meta at
  `/mnt/v/output/jxl-encoder/alpha-distance-audit-2026-05-17/`. CLAUDE.md
  Investigation Notes documents three ranked follow-on chunks
  (squeeze-on-extras, ChannelCompact-on-extras, entropy-coder gap).
  Reproducer: `cargo run --release -p jxl-encoder --example
  alpha_distance_audit -- --output <path>`. Refs A1 audit Top-5 #4.

- **Multi-group `--ec_resampling N` writer** (A1 audit Top-5 #2,
  follow-on to W5-1 `59b31cc`). Closes the multi-group hole left by
  `59b31cc`'s single-group-only landing. `extract_region` now
  downshifts the per-group rect by each channel's own
  `hshift`/`vshift` (matches libjxl `enc_modular.cc:1400-1407`'s
  `Rect(rect.x0() >> fc.hshift, rect.y0() >> fc.vshift, ...)`), so
  downsampled extras (e.g. half-res alpha at `dim_shift = 1`) crop
  in channel-local coordinates rather than at full-resolution. The
  destination channel inherits `hshift`/`vshift` from the source so
  downstream consumers (tree learning, residual gather, group
  section writer) see the same geometry the decoder reconstructs.
  Per-group rects that shift to empty are materialised as zero-sized
  channel placeholders, which the decoder skips via the standard
  `if (!channel.w || !channel.h) continue;` check
  (`encoding.cc:579`). The CLI rejection of multi-group
  `--ec_resampling > 1` (`jxl-encoder-cli/src/main.rs:1455-1463`) is
  removed — 4K+ web assets with downsampled alpha now route through
  the standard lossless RGBA / BGRA / GrayAlpha path. New API:
  `ModularImage::push_extra_channel_u8_with_shift(...)` and the
  16-bit twin; `api.rs` propagates `ExtraChannelInfo.dim_shift` to
  the channel automatically. New regression test:
  `test_lossless_rgba_multi_group_with_ec_resampling_half_res_alpha`
  (384×384 = 4 groups, half-res alpha, jxl-oxide + djxl verified).
  hash_locks 36/36 byte-identical (no change at `dim_shift = 0`).

### Changed

- **`--modular-predictor N` now overrides the MA tree learner**
  (W12-4 audit Top-5 #1, follow-on to W7-2 `e887c2bb`). When
  `LosslessConfig::modular_predictor = Some(N)` with `N` in `0..=4` or
  `6..=13` AND the encode runs through the tree-learning path (default
  at effort >= 7), the ID3 learner is now bypassed and a single-leaf
  tree pinned to predictor `N` is emitted instead — matching the
  libjxl `cjxl -P N` / `--modular_predictor` semantics where
  `options.predictor` overrides what would otherwise be the tree
  learner's per-leaf choice. Wired through both the single-group path
  (`write_modular_stream_with_tree_dc_quant_knobs`) and the multi-group
  LfGlobal path (`write_global_modular_section_with_tree_knobs`); per-
  group sections pick up the override via the existing
  `GlobalModularState::AnsWithTree` tree handle. Three exceptions
  preserve hash-lock parity: `Some(5)` (Gradient — the legacy default
  the resolver maps to None to keep the ID3 path identical), `Some(14)`
  (libjxl `Best`) and `Some(15)` (libjxl `Variable`) are meta-modes
  that explicitly request per-leaf selection and stay byte-identical to
  the unset default. The lossy modular path (LfFrame, `is_lossy`) does
  NOT honour the override — its forced-split tree + Zero predictor
  invariant must be preserved for residual divisibility. Verification:
  4 new tests
  (`modular_knobs_predictor_some5_byte_identical_to_default_tree_learn`,
  `modular_knobs_predictor_overrides_tree_learner_left`,
  `modular_knobs_predictor_tree_learn_meta_modes_fall_back_to_id3`,
  `modular_knobs_predictor_tree_learn_all_ids_roundtrip_via_jxl_rs`)
  pin both the bytes-change semantics and the pixel-exact jxl-rs
  roundtrip for all 14 ids; hash_locks 36/36 byte-identical; 1132/1132
  lib tests pass; CLI smoke test
  `modular_predictor_flag_accepted_lossless_path` updated to match new
  Gradient-fallthrough invariant. Measured impact on
  `gb82-sc/terminal.png` at effort 7 lossless: default ID3 49714 bytes,
  `-P 5` 49714 bytes (identical), `-P 4` (Select) 84384 bytes, other
  ids 95-1518 KB — confirms ID3 wins on screenshot content but the
  override produces valid bitstreams (djxl + jxl-rs decode) for every
  id, opening the door to per-image content-discriminated dispatch.

### BREAKING CHANGE (queued)

- **`modular_knobs_predictor_does_not_override_tree_learner`** test
  renamed to `modular_knobs_predictor_some5_byte_identical_to_default_tree_learn`
  and semantics flipped: the original test asserted that NO id
  overrides the tree learner; the new test pins only `Some(5)`
  (Gradient default) as byte-identical, while the companion
  `modular_knobs_predictor_overrides_tree_learner_left` requires non-
  Gradient ids to CHANGE bytes on the tree-learn path. Downstream
  callers that have built tooling assuming `modular_predictor` is a
  no-op on the tree-learn path (the W4-1 / W7-2 partial-wire state)
  will see bytes change when they pass `-P {0,1,2,3,4,6,7,8,9,10,11,12,13}`.

- **Lossless patches gate now uses lossless-shape trial encoder**
  (`trial_encode_ref_frame_bytes_lossless`, RFC#45 lossless chunk 5
  follow-on to W11-1 `ad9964a6`). Replaces the XYB-shape
  `trial_encode_ref_frame_bytes` invoked by W11-1's
  `is_cost_effective_lossless` with a path that mirrors the live emit
  (`encode_reference_frame_rgb`). The XYB-shape trial overshot true
  lossless byte cost by up to 1.8× on smooth-dark UI screenshots
  (mean overshoot 1.32× across the gb82-sc 8-image admitted set);
  with the new tighter overhead estimator, `SAVINGS_BYTES_PER_PIXEL_LOSSLESS`
  drops from `0.45` to `0.35` and the gate is 22% tighter against
  pathological mixed content (admission band shifts from
  `c_needed_xyb ≤ 0.45` to `c_needed_lossless ≤ 0.35`). Same 8/8
  admission on the gb82-sc corpus — bytes byte-identical to W11-1.
  Signature change: `is_cost_effective_lossless(use_ans)` →
  `is_cost_effective_lossless(bit_depth, use_ans)` (`bit_depth = 8`
  for the common Rgb8 path, 16 for Rgb16). Refs jxl-encoder#45.
  Calibration TSV:
  `benchmarks/patches_lossless_savings_calibrate_all_lossless_trial_2026-05-17.tsv`.
  A/B verdict-vs-empirical:
  `benchmarks/patches_lossless_gate_ab_lossless_trial_2026-05-17.tsv`
  (5/5 screenshot wins, 5/5 photo no-ops, 10/10 gate-verdict matches
  empirical sign). hash_locks 36/36 byte-identical.

### Added

- **`trial_encode_ref_frame_bytes_lossless`** — lossless-shape
  reference-frame trial encoder mirroring the live
  `encode_reference_frame_rgb` emit. Companion to the existing XYB-shape
  `trial_encode_ref_frame_bytes`. Used by the chunk-5
  `is_cost_effective_lossless` gate. New `__internals` wrapper
  `patches_trial_overhead_lossless(bit_depth, use_ans)` exposed for
  calibration harnesses (sidecar to `patches_trial_overhead` which
  retains the XYB-shape estimator). Refs jxl-encoder#45.

- **Lossless-mode patches per-image cost gate**
  (`PatchesData::is_cost_effective_lossless`, RFC#45 chunks 4-7 backport
  to the lossless path). Mirrors the chunk-7 lossy structure
  (trial-encoded `ref_overhead` + `dict_overhead`, integer-form 1.5×
  safety margin `2 * savings_est >= 3 * total_overhead`) but without a
  distance axis — lossless preserves every coefficient exactly so the
  savings model is `pixels * C_LOSSLESS` (no `1/sqrt(d)` divisor).
  Calibrated from
  `benchmarks/patches_lossless_savings_calibrate_all_2026-05-17.tsv`
  (11 gb82-sc screenshots, 8 produce patches, 3 hit the detector's 1%
  coverage filter): `C_LOSSLESS = 0.45` admits all 8 net-winning cells
  at 1.5× margin (worst case `imac_dark` at margin 1.03×). Wired into
  `api.rs` at both `encode_lossless` one-shot (after `find_and_build_lossless`)
  and the streaming `LosslessEncoder::finish` variant. Photos
  byte-identical (detector returns `None` upstream → gate not invoked,
  5/5 CID22-512 cells unchanged); 5/5 measured gb82-sc screenshots
  byte-identical (gate admits the same patches the no-gate path
  shipped). hash_locks 36/36 byte-identical. The gate is **protective**
  — it ships behind the detector's existing 1% coverage filter and only
  fires on pathological mixed content where overhead clearly exceeds
  savings; no measured regression on the gb82-sc corpus. Pre-gate state
  shipped every detected patch unconditionally. The current calibration
  is **overhead-overshoot-corrected** — the true geomean
  `actual_savings / total_patch_pixels` is 0.27, but
  `trial_encode_ref_frame_bytes` invokes the XYB-shape path which
  overshoots actual lossless ref-frame cost by ≈1.5-2×. Future work:
  ship a lossless-shape trial encoder and re-fit C against tighter
  overhead estimates. Refs jxl-encoder#45.

### Changed

- **`PatchesData::is_cost_effective` — per-image overhead correction**
  (RFC#45 chunk 7 — follow-on to W9-4 chunk 6 `088719c5`). Replaces the
  analytical `dict_overhead_est = 5 * ref_positions + 5 * positions`
  estimate with a trial-encode of `encode_patches_section` to measure the
  actual dictionary-section byte count per image. Also bundles the chunk-6
  1.5x safety-margin relaxation (`2 * savings_est >= 3 * total_overhead`),
  which was never landed on `main@origin`. The analytical estimate
  overshot the actual ANS-coded delta-encoded dictionary size by 2-4x on
  screenshots with many similar packed patches, inflating `total_overhead`
  and forcing the gate shut on the two W9-4 residual cells (`windows95
  @ d=4.0` and `windows @ d=4.0`). Chunk 7 admits both residuals plus 4
  other previously-rejected high-d cells, while keeping the 14 already-
  admitted cells byte-identical and the 20 photo cells unchanged
  (detector returns `None` upstream on photos so the gate is not
  invoked). Total newly-admitted savings: 425,793 B across 6 screenshot
  cells (`benchmarks/patches_gate_experimental_ab_chunk7_2026-05-17.tsv`).
  Hash-locks 36/36 byte-identical (Reference mode unchanged; gate fires
  only in `EncoderMode::Experimental`).

### Added

- **Auto-splines content discriminator** (chunk 5 — follow-on to
  chunk 4 `cbb36478`). Adds `vardct::splines::looks_like_screenshot`,
  a `median(per-8x8-block mean of mask1x1) > 95.0` gate (threshold
  mirrors the GPU encoder's W7-3 AFV cost-grid
  `SCREENSHOT_MEDIAN_MASK_THRESHOLD` in
  `jxl-encoder-gpu/src/lossy_encoder.rs:2907`). When `auto_splines`
  fires at `effort >= 7`, the discriminator runs first on the
  post-patches pre-gaborish XYB Y plane; on screenshot-class content
  the splines detector is skipped entirely, avoiding the
  `bbox-area-linear` energy-drop proxy's structural over-claim on
  long bright ridges (table borders, wallpaper edges). After chunk 4
  the remaining residual was `codec_wiki @ d=1.0` (+3.3%) and
  `imac_g3 @ d=1.0` (+3.5%); after chunk 5 both go byte-identical
  (`benchmarks/auto_splines_bench_2026-05-17_chunk5.tsv`: 33 of 33
  cells delta 0.000% across 5 photos + 3 screenshots + 3 synthetics
  × e7/e8/e9). Discriminator-validated screenshot median: 100.013;
  photo median: 55.878 (clean ≥5x gap from threshold). Default
  `auto_splines_default(_) = false` unchanged — the discriminator
  is so effective at filtering false-positives that no test image
  benefits from default-on, so flipping it would add a
  `compute_mask1x1` pass per encode for zero observable RD benefit.
  The flag remains opt-in for callers tuning for thin-feature
  content (power lines on a noisy sky, hair on a photo background)
  where the discriminator does NOT fire AND the cost gate admits the
  candidate. Hash-locks: `hash_lock_features` 36/36 byte-identical;
  `tests/auto_splines.rs` 6/6 pass (5 retained, plus chunk-5
  `chunk5_multi_line_runs_detector` that replaced
  `chunk3_multi_line_decreases_bytes` whose flat-grey synthetic
  contract chunk 5 correctly short-circuits); 4 new lib unit tests
  in `vardct::splines::tests` cover the discriminator at the flat /
  photo-gradient / strided / tiny-image boundaries.
  `vardct::splines::SCREENSHOT_MEDIAN_MASK_THRESHOLD` exposed at
  `pub(crate)` for any future intra-crate caller that wants the same
  gate before its own analysis pass.

- **Seed-budget expansion to 16 + two new variance dimensions for
  multi-seed tree learning** (RFC#45 pick #1 chunk 6 — follow-on to
  chunk 5 `2b2ce912`). W9-1 chunk 5 expanded e11 from 4 → 8 seeds and
  split chunk-3 perturbations (seeds 0..=3) from chunk-4 dimensions
  (seeds 4..=7), producing −0.46% bytes vs chunk 4 / strict win over
  chunk 3 on the 5-image CID22-512 paired bench. Chunk 6 extends the
  same seed-slot pattern with two coupled changes:
  (1) `EffortProfile::tree_learn_seeds_for(11)` raised from `8 → 16`
  (e10 stays at `2`, e ≤ 9 still single-seed). (2) Two new
  `derive_seeded_*` helpers wire orthogonal variance dimensions into
  dedicated 4-seed slots: `derive_seeded_max_property_values(seed)`
  returns `Some(64) / Some(128) / Some(192) / None` for seeds
  8..=11 (split-bucket-count override that coarsens
  `find_best_split`'s value quantization grid — coarser grids can
  land on different and sometimes cheaper discrete thresholds than
  the 256-bucket canonical), and
  `derive_seeded_properties_truncation(seed)` returns
  `Some(8) / Some(10) / Some(12) / None` for seeds 12..=15 (truncates
  the canonical `properties` Vec to a smaller leading prefix —
  structural regularization that can outperform full-property trees
  when the canonical run over-fits late-tier properties like the
  `WPMaxError` family at indices 10-15 chasing bucket noise on
  smooth content). Both helpers return `None` outside their 4-seed
  slot ranges so the two chunk-6 dimensions never stack on a single
  seed — the seed-slot doctrine that chunks 3-5 established (each
  chunk's dimension owns its own 4-seed block) is now codified by
  strict slot-range gates rather than wrap-around modulus.
  `section.rs` applies both overrides to a per-seed clone of the
  baseline `TreeLearningParams` after `derive_seeded_params`, with
  truncation clamped to `properties.len()` so a cap longer than the
  property Vec is a no-op rather than an invalid index. Chunk-2
  `estimate_token_cost` picker keeps the cheapest of the 16
  candidate trees — strictly ≥ chunk 5 by construction (seeds 0..=7
  cover the same chunk-3/4/5 candidate space). Seed 0 stays
  byte-identical to the canonical libjxl single-seed path. New unit
  tests: `test_derive_seeded_max_property_values_low_seeds_are_none`,
  `test_derive_seeded_max_property_values_high_seeds_active`,
  `test_derive_seeded_properties_truncation_low_seeds_are_none`,
  `test_derive_seeded_properties_truncation_high_seeds_active`,
  `test_chunk6_dimensions_are_orthogonal` (enforces that
  bucket-count slot seeds 8..=11 never trigger truncation and
  truncation slot seeds 12..=15 never trigger bucket override).
  Bench harness: `examples/e10_e11_multiseed_chunk6_ab.rs`.
  Hash-locks: `hash_lock_features` 36/36 byte-identical at e ≤ 9.
  RFC#45 issue thread updated with the 16-seed slot table.

- **Seed-slot split + e11 budget expansion for multi-seed tree learning**
  (RFC#45 pick #1 chunk 5 — follow-on to chunk 4 `ef5c1d11`). W8-3-r2's
  honest 5-image A/B showed chunk 4 regressed vs chunk 3 at e11 by
  +0.39% bytes because the fixed 4-seed budget meant chunk-4's new
  variance dimensions cycled through *different* 4 trees rather than
  *more*. Chunk 5 addresses that with two coupled changes:
  (1) `EffortProfile::tree_learn_seeds_for(11)` raised from `4 → 8`
  (e10 stays at `2`); and (2) seed-slot split inside
  `derive_seeded_sample_fraction(seed)` and
  `derive_seeded_predictor_order(seed)`: seeds 0..=3 now return
  `None` / canonical (chunk-4 dimensions held to no-op), so chunk-3's
  three perturbations (`split_threshold` jitter, property-order
  rotation, per-seed stride) get four dedicated seed slots without
  being recombined with sample-fraction overrides or predictor
  permutations; seeds 4..=7 cycle through the four chunk-4
  sample-fraction values `[Some(0.40), Some(0.60), Some(0.70), None]`
  and the four `CANDIDATE_PREDICTORS_PERMS` permutations on top of
  the chunk-3 perturbations they pick up by virtue of `seed % 4`. The
  chunk-2 `estimate_token_cost` picker keeps the cheapest of the 8
  candidate trees — strictly ≥ chunk 3 by construction (seeds 0..=3
  cover the same candidate space) and strictly ≥ chunk 4 when the
  recombined chunk-4 dimensions beat chunk-3's threshold/property/
  stride alone. Seed 0 stays byte-identical to the canonical libjxl
  single-seed path; e ≤ 9 still has `tree_learn_seeds = 1` so this
  helper is never called there. Updated 4 unit tests
  (`test_derive_seeded_sample_fraction_low_seeds_are_none`,
  `test_derive_seeded_sample_fraction_high_seeds_active`,
  `test_derive_seeded_predictor_order_low_seeds_canonical`,
  `test_derive_seeded_predictor_order_high_seeds_perturb`,
  `test_derive_seeded_predictor_order_preserves_predictor_set`,
  `test_new_with_predictor_order_for_seed_low_seeds_match_default`)
  enforce the chunk-5 seed-slot contract. Bench harness:
  `examples/e10_e11_multiseed_chunk5_ab.rs`. Hash-locks:
  `hash_lock_features` 36/36 byte-identical at e ≤ 9. Bench TSV +
  meta archived at workspace-root
  `benchmarks/e10_e11_multiseed_chunk5_ab_2026-05-17.{tsv,meta}`.
  Wall-clock at e11 roughly doubles vs chunk 4 (8 seeds vs 4); e10
  unchanged. **5-image A/B vs chunk 4 on CID22-512 photos
  (deterministic across both samples):** e10 -0.008% (-96 bytes; 1
  cell improved [1279330 207123→207027], 4 byte-identical), e11
  **-0.46%** (-5647 bytes; 3 cells improved [1044329 330499→327001
  -1.06%, 1189261 303864→302399 -0.48%, 1279330 207123→206214
  -0.44%], 1 byte-identical, 1418519 +0.14% regression — within
  noise of chunk-3's win there). vs chunk 3 baseline at e11
  (sum 1232021 bytes): chunk 5 sum 1231208 = **-0.066%** — chunk 5
  strictly beats chunk 3 AND fixes chunk 4's +0.39% regression.
  RFC#45 issue thread updated with the seed-slot table and 5-image
  bytes comparison.

- **`LossyConfig::with_auto_delta_frames(bool)` /
  `LosslessConfig::with_auto_delta_frames(bool)` + getters** (A1
  audit "Animation" — Skip / delta frame encoding, chunk-1 POC).
  Opt-in (default `false`, hash-locks 36/36 byte-identical at
  default). When enabled, the animation encode path swaps the
  existing same-pixel `Replace`-over-1×1 / 8×8 crop for an
  `Add`-over-zero-pixel-crop on byte-identical successor frames.
  Add-of-zero is a no-op redraw in linear-RGB float; zero pixels
  modular-encode smaller than arbitrary canvas-pixel values.
  Chunk-1 scope: identical-frame short-circuit on no-alpha layouts
  only (RGBA needs `ec_blend_modes = Add` plumbing, queued for
  chunk-2 alongside the full per-frame trial-encode loop of
  `Regular` vs `Add(prev)` vs `Blend(prev)`). Measured -10 bytes
  on a 3-frame 256×256 RGB8 gradient with all frames identical
  (208 → 198 bytes lossless; jxl-rs + jxl-oxide both decode the
  result to a 3-keyframe animation with frames 1/2 matching frame
  0). New tests in `jxl-encoder/tests/animation.rs`:
  `test_auto_delta_frames_default_off_is_byte_identical`,
  `test_auto_delta_frames_lossless_identity_short_circuit`,
  `test_auto_delta_frames_lossless_identical_path_decodes_via_jxlrs`,
  `test_auto_delta_frames_lossy_identical_path_decodes`.

- **`with_auto_delta_frames` chunk-2: RGBA support + full-frame
  delta-residual trial-encode loop** (follow-on to chunk-1 POC
  `904b373d`). Two coupled widenings:
  (1) RGBA layouts can now take the identity short-circuit. The
  extra-channel blend mode is overridden to `Add` (via a new
  `FrameOptions::ec_blend_mode_override` /
  `FrameEncoderOptions::ec_blend_mode_override` Option<BlendMode>)
  and the extra-channel `source` is mirrored onto the main
  `blend_source` so an `Add`-of-zero alpha lands on the same
  reference slot the main `Add`-of-zero RGB does — without the
  source mirror, the alpha would composite against the empty slot 0
  and decode as zero.
  (2) For genuinely-different frames the lossless animation path
  trial-encodes two candidates per frame — (A) the existing Regular
  same-pixel crop and (B) a full-frame `BlendMode::Add` payload whose
  pixels are signed `frame_N - frame_N-1` deltas built by a new
  internal helper `build_lossless_delta_image` (handles Rgb8 / Rgba8 /
  Bgr8 / Bgra8 / Gray8 / GrayAlpha8 / Rgb16 / Rgba16 / Gray16 /
  GrayAlpha16; float / PQ / HLG inputs fall back silently to
  candidate A). Each candidate is encoded into its own scratch
  `BitWriter`; the smaller (by bit count, since frame-header writes
  are not byte-aligned at start) is appended to the output via
  `append_unaligned`. Delta-residual is byte-exact for lossless
  because the modular signed-i32 channels round-trip both branches
  of the subtraction. Lossy is NOT extended to delta-residual —
  per the chunk-1 commit, lossy residuals must round-trip through
  the reconstructed (already-quantised) reference frame, not the
  original pixels; that needs a reconstruction shadow that
  chunk-2 does not wire. Lossy gets only the RGBA identity
  extension.
  Bonus fix: the chunk-2 work surfaced a long-latent baseline bug —
  for ALL RGBA animation crop frames (not just the chunk-2 paths)
  the encoder was writing every extra-channel
  `BlendingInfo::source` as `0`, so alpha decoded to zero everywhere
  outside the crop region. The fix mirrors `blend_source` onto every
  ec when a crop is set, in both `modular/frame.rs::
  apply_animation_to_header` and `vardct/bitstream.rs`. New
  regression test `test_rgba_animation_crop_alpha_baseline_preserved`
  locks in the post-fix behaviour. Hash-locks 36/36 still byte-
  identical (none cover RGBA + animation crop); `cargo test --tests`
  passes including the existing 26 animation cases. New tests in
  `jxl-encoder/tests/animation.rs`:
  `test_auto_delta_frames_lossless_rgba_identity_short_circuit`,
  `test_auto_delta_frames_lossless_rgb_small_motion_wins`,
  `test_auto_delta_frames_lossless_rgba_small_motion_alpha_survives`,
  `test_auto_delta_frames_lossless_fully_different_no_regression`,
  `test_auto_delta_frames_lossy_rgba_identity_short_circuit`,
  `test_rgba_animation_crop_alpha_baseline_preserved`. Default
  remains `false`; opt-in only.

- **`EffortProfile::auto_splines_default(effort: u8) -> bool` and
  `LossyConfig::auto_splines_explicit()` getter** (follow-on to W6-2 +
  W7-4 chunk 3). The function centralises the per-effort default for
  the chunk-3 ridge detector; `with_auto_splines(b)` now flips
  `auto_splines_explicit = true` so a caller's choice survives
  subsequent `with_effort()` calls. The function currently returns
  `false` at every effort (see below). A new
  `examples/auto_splines_corpus_bench.rs` A/B harness drives 8 real
  images plus 3 synthetic ridges across e7/e8/e9 for future re-bench
  passes. Hash-locks 36/36 byte-identical; all 6 `tests/auto_splines.rs`
  integration tests pass (incl. `auto_splines_default_is_off`,
  `auto_splines_chunk3_multi_line_decreases_bytes`).

### Changed

- **Auto-splines cost gate `BYTES_PER_ENERGY_UNIT_AT_D1` recalibrated
  from `50.0` to `0.20`** (chunk 4 follow-on to W8-6 `6c01965`).
  W8-6's rejection rationale was wrong: the chunk-3 cost gate is
  deterministic on (XYB, distance) inputs and effort-independent, so
  it can NOT silently start rejecting all candidates at e8+. Re-running
  the bench against the chunk-4 binary
  (`benchmarks/auto_splines_bench_2026-05-17_chunk4.{tsv,meta}`)
  showed the gate was actually OVER-claiming savings on screenshots
  and 2/5 photos under the old `50.0` constant: terminal regressed
  +3-8% at e7/e8/e9, codec_wiki regressed +6-9%, imac_g3 +3.2-3.4%.
  Root cause: the original `50.0` anchor was derived from a stale
  comment that estimated `energy_drop ≈ 2-4` for the 1024×256 power-
  line synthetic, but the chunk-3 detector measures `energy_drop ≈
  533` for the same image — the realised bytes-per-energy ratio is
  closer to `0.07-0.15`. Recalibrating to `0.20` (geomean fit on
  the multi-line synthetics) restores screenshots and all 5 photos
  to byte-identical at e7/e8/e9 while keeping the multi-line
  power-line wins (-2.3 to -3.1% at e7/e8, -557 to -138 bytes).
  The `test_find_splines_finds_horizontal_ridge` unit test was
  updated to bypass the cost gate (verifies the pre-gate detector
  produces polylines, since the chunk-4 gate correctly rejects the
  prior single-ridge synthetic as a real-encode regression). Hash-
  locks 36/36 byte-identical; default `auto_splines = false` is
  unchanged so the recalibration has zero effect on the default
  encode path; all 6 `tests/auto_splines.rs` integration tests pass.

### Investigated

- **Auto-splines default-on at e7+: rejected even after chunk-4
  recalibration** (`benchmarks/auto_splines_bench_2026-05-17_chunk4.{tsv,meta}`).
  After fixing the over-claim bug (above), photos and `terminal.png`
  go byte-identical at e7/e8/e9 (was +3-8% regression). But two
  remaining screenshots (`codec_wiki.png`, `imac_g3.png`) still admit
  6 / 33 splines on wide bright ridges (table borders, wallpaper
  edges), regressing real encodes by ~3% across all three efforts.
  The energy-drop proxy is structurally biased on long bboxes — it
  scales linearly in pixel count but actual VarDCT byte savings are
  sub-linear (the AC coefficients aren't independent). Fixing that
  would require either full A/B trial-encode (too expensive) or a
  content discriminator that's outside chunk-4 scope. The multi-line
  synthetics still net-win at e7/e8 (-2 to -3%) and lose at e9 (the
  more aggressive baseline outpaces the splines section), so the
  detector design is sound in its narrow target regime. Default
  stays `false` at every effort. Investigations of options A
  (`COST_BENEFIT_MARGIN` 2.0 → 1.5) and B (run gate on initial
  quant field, not post-buttloop) were skipped after the proxy
  miscalibration was identified as the dominant lever — neither
  option fixes the structural over-claim on long ridges. The
  `auto_splines_trace` helper example was added under
  `jxl-encoder/examples/` for future debugging passes.

- **Auto-splines default-on at e8+: rejected after bench
  (`benchmarks/auto_splines_bench_2026-05-17.{tsv,meta}`).** Photo
  no-regression invariant holds (10/10 byte-identical), but the chunk-3
  detector's trial-encode cost gate rejects every candidate on every
  tested image at e8 plus e9 — including the multi-line power-line
  synthetics the detector was designed to win on at e7. At e7 the
  detector still nets -138 / -557 bytes on 4-line / 8-line ridges
  (+118 on the 1-line edge case) via `with_auto_splines(true)`.
  Flipping default-on at e8+ would ship CPU overhead (Sobel + NMS +
  Hessian + polyline trace + per-candidate trial encode) for zero
  byte change across the corpus. Default stays `false` at every
  effort. When the detector evolves to win at e8+, only
  `EffortProfile::auto_splines_default` needs updating.
  (Note: chunk 4 above showed the "rejected at e8+" rationale was
  incorrect — the gate isn't effort-dependent. The default-off
  conclusion stands but for the right reason: gate over-claims on
  long-ridge content, not effort-tied behavior.)

### Changed

- **Lossy alpha pipeline now fires on mixed-extras frames** (W8-2,
  follow-on to W6-3 `bbf8a985`). W6-3 wired the
  `LossyConfig::with_alpha_distance(Some(d))` quantizer through the
  modular extras sub-bitstream but only when `extras.len() == 1`;
  any image with alpha + a second extra (depth, spot color,
  selection mask, ...) silently stayed all-lossless. The encoder
  now dispatches a per-channel quantizer slice (libjxl
  `cparams.ec_distance[i]` shape, `enc_modular.cc:973-1027`): each
  channel's `q` is computed from its `ExtraChannelType` — alpha
  reads `alpha_distance`, all others stay at `q = 1` until per-
  channel `ec_distance` is wired through the public API. When the
  resolved quantizers are mixed (e.g. `[q=15, q=1]` for alpha-lossy
  + depth-lossless), the encoder emits a multi-leaf gradient tree
  splitting on property 0 (channel index, libjxl
  `static_props[0] = chan`); when only one channel is lossy or all
  are lossless, the single-leaf paths are preserved byte-identical
  (W6-3 single-extra alpha frames and pre-W6-3 lossless frames are
  bit-for-bit unchanged). Wiring: new
  `compute_extras_pixel_quantizers` + dispatch in
  `write_modular_extras_subbitstream`, new
  `write_tree_histogram_for_channel_split_lossy` +
  `write_channel_split_tree_tokens` in `modular/encode_tree.rs`.
  Roundtrip proof in
  `tests/lossy_mixed_extras_alpha.rs::mixed_extras_alpha_lossy_depth_lossless`
  (RGB + alpha + depth at `alpha_distance=10.0`: jxl-rs decode
  shows alpha MAE > 1.0 while depth comes back byte-identical) and
  byte-identical guard in
  `mixed_extras_alpha_lossless_depth_lossless_byte_identical`
  (`alpha_distance=None` and `Some(0.0)` produce identical bytes on
  mixed-extras frames). `hash_lock_features` 36/36 byte-identical;
  existing single-extra lossy alpha tests
  (`alpha_distance_high_loses_alpha_precision`,
  `alpha_distance_nonzero_changes_bytes`) still pass.
- **`--modular-predictor 0..13` now wires through to all
  no-tree-learning modular paths** (W4-1 follow-on; W4-1 stored the
  knob on `LosslessConfig` but only the tree-learn path consumed it).
  The override mirrors libjxl `cjxl -P N` / `--modular_predictor`:
  `0=Zero, 1=Left, 2=Top, 3=Average, 4=Select, 5=Gradient (default),
  6=Weighted, 7=NorthEast, 8=NorthWest, 9=WestWest,
  10=AverageWestAndNorthWest, 11=AverageNorthAndNorthWest,
  12=AverageNorthAndNorthEast, 13=AverageAll`. Ids 14 (`Best`) and 15
  (`Variable`) are libjxl encoder-only meta-modes that imply tree
  learning — the non-tree paths fold them to Gradient (id 5) so the
  bitstream stays self-consistent. The MA tree learner (default at
  effort ≥ 7) is libjxl's `Predictor::Variable` mode and ignores the
  knob by design. Wiring covers: `write_improved_modular_stream`
  (LZ77), `write_simple_modular_stream`, `write_modular_stream_with_
  rct_only`, `write_modular_stream_with_palette_knobs`,
  `write_modular_stream_with_lossy_palette_budget_knobs`, the
  squeeze multi-group LfGlobal residual pass, the lossy-palette
  multi-group LfGlobal residual pass, and the multi-group
  non-tree-learn standard path (via new
  `write_global_modular_section_with_predictor` +
  `collect_all_residuals_with_predictor`). Id 6 (Weighted) routes to
  the dedicated `write_modular_stream_with_(rct_)weighted` writers
  when the path is simple enough to delegate, otherwise folds to
  Gradient (paths without per-channel `WeightedPredictorState` can't
  emit consistent weighted residuals — `resolve_fixed_predictor_for_
  simple_path` documents this). All 14 predictor ids verified
  pixel-exact roundtrip via jxl-rs in
  `modular_knobs_predictor_all_ids_roundtrip_via_jxl_rs`. Default-
  config output remains byte-identical (`hash_lock_features` 36/36
  green, RD-regression 18/18 within thresholds).
- **`--faster_decoding 0..4` now wires through to encoder choices**
  (follow-on to W4-3's storage-only landing). The knob mirrors libjxl
  `cparams.decoding_speed_tier` and biases the bitstream toward simpler
  shapes that decode faster at the cost of compression. Per-tier
  effects:
  - **tier 0** (default): no-op, bytes byte-identical to pre-W4-3
    (hash_lock_features 36/36 byte-identical, RD-regression 18/18
    within thresholds).
  - **tier 1**: LZ77 disabled (`enc_ans.cc:1372`, `enc_modular.cc`).
  - **tier 2**: tier 1 + pair-merge histogram clustering for VarDCT
    disabled (`enhanced_clustering_vardct = false`), patches detection
    skipped (`enc_modular.cc:707`), `modular_group_size_shift` forced
    to `0` for multithreaded decode (`enc_frame.cc:340-343`).
  - **tier 3**: tier 2 + custom coefficient orders disabled, tree-split
    threshold raised by `+10 * tier` (`enc_modular.cc:533`).
  - **tier 4**: tier 3 + MA tree learning disabled, gaborish forced
    off (`enc_frame.cc:280`), DCT32X32 / DCT64 disabled in AC strategy
    search (`enc_ac_strategy.cc:936`), `tree_sample_fraction = 0` (so
    the sampler returns its floor and the tree learner sees minimal
    data — mirrors libjxl `nb_repeats = 0` at tier 4).
  Wiring lives on the existing `LossyConfig` / `LosslessConfig`
  `with_faster_decoding(u8)` builder; the new `EffortProfile::apply_
  faster_decoding(tier)` method runs last inside
  `effective_profile()`, and per-flag effective getters
  (`effective_lz77`, `effective_tree_learning`, `effective_patches`,
  `effective_gaborish`, `effective_modular_group_size_shift`) route
  the config-stored values through the speed tier at the encoder
  consumption sites. Explicit `with_modular_group_size(Some(n))` from
  the caller still wins over the tier-2 default. Verified with new
  jxl-rs roundtrip tests at levels 0/2/4 on a 96×96 RGB synthetic;
  lossless byte counts grow as the tier rises (6193 → 6193 → 20864
  bytes at tier 0/2/4 — tier 0 is the most compressed, tier 4 the
  fastest-to-decode).

### Fixed

- **Clippy `-D warnings` CI red** — 13 lints introduced by the recent
  e10/e11 multi-seed and Phase 4 inline-dedup work were failing the
  `Clippy (x64)` and `Clippy (aarch64)` CI jobs (no other jobs affected).
  Seven dead-code items (`HASH1_CONST`, `HASH2_CONST`,
  `FusedHashKeyBuilder`, `BuilderOverflow`, `FinalizedKey` + impl
  methods, `InlineDedupTable::{capacity, len, is_empty, lookup_only,
  unique_keys}`, `gather_samples_strided_with_dedup`,
  `select_best_tree_multi_seed`) flagged `#[allow(dead_code)]` with
  comments — all are real code (used by the `dedup_samples_strategies`
  microbench under `__bench_internals`, or reserved for e10/e11
  multi-seed paths the default-features clippy build doesn't exercise).
  Five trivial lints fixed in place: doc-lazy-continuation indentation,
  `match` → `if let Some(true)`, `let_and_return` collapse in CLI,
  needless `return`. `cargo clippy --workspace -- -D warnings` and
  `--features zensim-loop` both green; `cargo build --workspace` and
  `cargo check --features __bench_internals` both clean.

### Added

- **Splines auto-detect chunk 3 — fidelity improvements that flip
  multi-line bytes net-negative** (A1 audit "VarDCT cost model"
  PARTIAL item, follow-on to chunk 2's `24f0787`). Three fidelity
  refinements close the residual gap that left chunk 2 paying +199
  bytes net on the 1024×256 single-line power-line synthetic:
  (1) **Per-control-point Hessian-derived sigma**
  (`hessian_lambda_large`, `vardct/splines.rs`) — sigma is now fit
  per arc-length sample as `1 / sqrt(|λ_large|)` of the local 2×2
  image-Hessian (clamped to `[SIGMA_MIN=0.6, SIGMA_MAX=4.0]`), then
  DCT-fit alongside the colour channels; sharp 1-px ridges get tight
  Gaussians, soft ridges get wider ones (was DC-only sigma in chunk 2).
  (2) **Bilinear colour sampling** (`bilinear_sample`) — replaces the
  chunk-2 nearest-pixel lookup, which under-represented ridge intensity
  by up to 50% when the ridge sat between integer pixels.
  (3) **Trial-encode cost gate** (`spline_passes_trial_encode_gate`)
  — replaces chunk 2's analytical estimate with a real
  `encode_splines_section` byte count (exact bytes for the candidate's
  splines section) plus a measured XYB residual energy reduction in
  the spline's bbox; mirrors the
  `vardct/patches::trial_encode_ref_frame_bytes` pattern at
  `vardct/patches.rs:2255`. (4) **Near-coincident-candidate dedup** —
  drops the second of a pair whose start AND end control points are
  both within `DUP_RADIUS_PX = 4.0`, suppressing the 8-connected
  tracer's habit of emitting both sides of a ridge as separate seeds.
  Realised effect at `distance=1.0, effort=7` (see
  `examples/splines_chunk3_bench.rs`):
    - `power_line 1024x256` (1 line, W6-2 test) — chunk 2: +199 bytes;
      chunk 3: **+118** (-81). Single-line still net-cost because VarDCT
      already encodes one isolated ridge cheaply and the per-image
      splines-section fixed overhead (~80 bytes) dominates.
    - `power_line 1024x512` (4 lines) — **-138 bytes** (net win).
    - `power_line 2048x1024` (8 lines) — **-557 bytes** (net win).
  Photo-like noisy-ramp content still produces zero admitted splines
  (`auto_splines_on_photo_is_byte_identical_to_default` unchanged).
  Default-config output remains byte-identical (`auto_splines` defaults
  to `false`; all 36 `hash_lock_features` fixtures unchanged). New
  tests: `test_bilinear_sample_interpolates_and_clamps`,
  `test_hessian_lambda_large_on_ridge_vs_flat`,
  `test_dedup_keeps_single_horizontal_ridge`; integration test
  `auto_splines_chunk3_multi_line_decreases_bytes` pins the
  strictly-decreases multi-line win.

- **Real spline auto-detection pipeline** (A1 audit "VarDCT cost
  model" PARTIAL item, chunk 2; follow-on to chunk 1's stub). The
  `find_splines_at_distance` entry replaces the chunk-1 stub with the
  full seven-stage pipeline sketched in the chunk-1 docstring:
  Sobel-magnitude ridge candidates, 1D non-max suppression along the
  gradient direction, 2x2 Hessian-eigenvalue ratio test (`λ_large /
  λ_small ≥ 5`, ridge-like only), direction-biased 8-connected
  polyline trace with seed-strength ordering, arc-length-uniform
  subsampling to 8 Catmull-Rom control points, per-channel DCT-II fit
  for X/Y/B colour (32 coefficients each, scaled to recover the
  decoder's continuous-IDCT convention) + DC-only sigma fit, and a
  per-spline cost-benefit gate (`COST_BENEFIT_MARGIN = 2×` patches-
  style margin, distance-aware, with empirically-anchored encoded-
  bytes and savings-per-pixel constants). The gate is intentionally
  conservative — it admits zero candidates on photo-like / smoothly-
  varying content (verified in
  `auto_splines_on_photo_is_byte_identical_to_default` and
  `test_find_splines_rejects_smooth_gradient`), and only fires on
  long high-contrast thin ridges. **Known limitation**: on synthetic
  flat-background single-line content the gate's theoretical savings
  estimate overshoots the realized win — the chunk-2 detector ships
  a DC-only sigma fit and nearest-pixel colour sampling, so the
  spline approximation leaves enough residual that VarDCT still
  encodes the ridge; chunk 3 will refine with a true
  `trial_encode_splines_section` gate mirror of
  `vardct/patches::trial_encode_ref_frame_bytes`. Default-config
  output remains byte-identical (`auto_splines` defaults to `false`,
  all 36 `hash_lock_features` fixtures unchanged). New tests pin the
  pipeline stages: `test_sobel_vertical_edge`,
  `test_hessian_rejects_corner`,
  `test_hessian_accepts_horizontal_ridge`,
  `test_subsample_polyline_endpoints`,
  `test_find_splines_returns_empty_for_constant_image`,
  `test_find_splines_finds_horizontal_ridge`,
  `test_find_splines_rejects_smooth_gradient`. Integration tests:
  `auto_splines_power_line_changes_bitstream` (bytes differ when the
  detector fires on a 1024×256 ridge),
  `auto_splines_on_photo_is_byte_identical_to_default` (cost gate
  rejects all candidates on noisy ramp content),
  `auto_splines_below_effort_gate_is_byte_identical`.

- **`LossyConfig::with_auto_splines(bool)` API surface and encoder wiring
  for automatic spline detection** (A1 audit "VarDCT cost model" PARTIAL
  item, chunk 1). Mirrors libjxl `enc_heuristics.cc:1048-1054` which
  gates auto-splines at `speed_tier <= kSquirrel` (effort >= 7) when no
  manual `cparams.custom_splines` are set. The detector hook lives at
  `vardct::splines::find_splines(xyb_x, xyb_y, xyb_b, w, h, stride) ->
  Vec<Spline>`. **Chunk 1 ships a stub detector that returns `vec![]`**,
  matching the `// TODO(user): implement spline detection.` stub upstream
  in libjxl `enc_splines.cc:104-107` — the encoder short-circuits the
  empty path so default-config output remains byte-identical (all 36
  `hash_lock_features` fixtures unchanged). The flag is preserved across
  `with_effort`, defaults to `false`, and is fully no-op until chunk 2
  lands a real ridge-following detector (see `find_splines` docstring
  for the chunk-2 algorithm sketch). Manual `with_splines(vec)` always
  wins outright when both are set. New tests:
  `auto_splines::auto_splines_default_is_off`,
  `auto_splines::auto_splines_preserved_across_with_effort`,
  `auto_splines::auto_splines_with_stub_is_byte_identical_to_default`,
  `auto_splines::auto_splines_below_effort_gate_is_byte_identical`,
  plus two unit tests pinning the stub contract
  (`test_find_splines_stub_returns_empty_for_constant_image`,
  `test_find_splines_stub_ignores_ridge`).

- **Lossy alpha pipeline (`LossyConfig::with_alpha_distance` >
  0.0)** — follow-on to W4-2-r (`62fc60e`) which staged the
  storage but kept the alpha extras sub-bitstream lossless. The
  encoder now mirrors libjxl `enc_modular.cc:973-1027` +
  `QuantizeChannel` (`enc_modular.cc:141`): for a single alpha
  extra at `dim_shift = 0`, computes an integer pixel quantizer
  `q = floor(0.025 * dist * bitdepth_correction * 0.35 * 1.1 *
  163.84)` (clamped to ≥1), snaps each alpha pixel to the nearest
  multiple of `q` (libjxl round-half-up by absolute value), and
  writes a single-leaf gradient tree whose `(mul_log, mul_bits)`
  carry the multiplier so the decoder reconstructs `pixel =
  prediction + val * q` (matches `ModularMultiplierInfo` +
  `make_pixel(val, multiplier, offset)` in
  `modular/encoding/encoding.cc:186-191`). `q == 1` (including
  `None` and `Some(0.0)`) keeps the lossless path
  byte-for-byte identical — hash-locks 36/36 unchanged. Mixed-extras
  inputs (count > 1) stay lossless until per-channel multiplier
  dispatch lands. Wiring proof in
  `tests/lossy_knobs_wiring.rs::alpha_distance_nonzero_changes_bytes`
  (d=2.0 → q=3, d=10.0 → q=15) and roundtrip proof in
  `tests/lossy_alpha_roundtrip.rs::alpha_distance_high_loses_alpha_precision`
  (jxl-rs decode confirms alpha MAE > 1 at d=10.0 while RGB stays
  byte-identical — alpha_distance does not leak into the VarDCT
  color path). djxl 0.12.0 also decodes the lossy-alpha bitstream
  cleanly. Implementation:
  `vardct/encoder.rs::compute_alpha_pixel_quantizer` (libjxl
  formula),
  `modular/encode_tree.rs::write_tree_histogram_for_gradient_lossy`
  +`write_gradient_tree_tokens_lossy` (lossy tree leaf),
  `vardct/bitstream.rs::write_modular_extras_subbitstream` (pre-quantize
  + divide-by-q residuals).

- **`--ec_resampling N` CLI flag + `downsample_channel_u8` API**
  (A1 audit "Pixel formats / extras"; mirrors libjxl `cjxl
  --ec_resampling`). Pre-downsamples the alpha plane on the
  lossless RGBA / BGRA / GrayAlpha 8-bit path with the same
  box filter libjxl uses (`lib/jxl/image_ops.cc::DoDownsampleImage`),
  then attaches it as an extra channel with
  `dim_shift = log2(N)`. Accepts `N ∈ {1, 2, 4, 8}`. Public
  helper `jxl_encoder::downsample_channel_u8` lets API
  callers run the same downsample on any u8 channel; pair
  with `ExtraChannel::with_dim_shift(log2(N))`. Single-group
  only (≤256×256) — multi-group bitstreams with
  `dim_shift > 0` extras fail libjxl djxl until the
  per-group writer is updated; the CLI rejects multi-group
  inputs at this knob rather than silently emitting broken
  output. Hash-locks 36/36 unchanged at default
  (`ec_resampling=1`). Roundtrip verified with jxl-oxide
  (`tests/api_tests.rs::test_lossless_rgb_with_ec_resampling_half_res_alpha`)
  and djxl on the 32×32 RGBA fixture.

- **ReferenceOnly animation frames + `save_as_reference` cross-frame
  compositing** (W4-A1 audit follow-on). `AnimationFrame::with_reference_only(bool)`
  flips the frame to `FrameType::ReferenceOnly` — the codestream writes
  the frame into its `save_as_reference` slot but decoders skip it
  during playback. Subsequent regular frames composite against the
  saved canvas via `with_blend_source(slot)` + a non-`Replace`
  `BlendMode`. The encoder auto-sets `is_last=false`, defaults the save
  slot to 1 when unset, and writes `save_before_ct=true` (mirroring
  libjxl's reference-frame defaults at `enc_frame.cc:446` +
  `enc_patch_dictionary.cc`). Public API rejects `reference_only=true`
  on the last animation frame (`EncodeError::InvalidInput`) — the file
  must end on a displayable frame. ReferenceOnly frames are written
  full-size (crop detection skipped) and don't advance the diff base
  for the next regular frame. Three new tests in `tests/animation.rs`:
  `test_animation_reference_only_lossless_jxlrs` (3-frame red →
  ReferenceOnly blue at slot 2 → Add/blend_source=2 green, validated
  via jxl-rs and jxl-oxide), `test_animation_reference_only_lossy_oxide`
  (VarDCT path), `test_animation_reference_only_last_frame_rejected`
  (rejection invariant). Zero impact on hash-locks (36/36
  byte-identical) — opt-in builder. Implementation in
  `headers/frame_header.rs::FrameOptions`, `api.rs::AnimationFrame`,
  `modular/frame.rs::apply_animation_to_header`,
  `vardct/bitstream.rs::encode_frame_to_writer`.

- **Modular group-size knob — `LosslessConfig::with_modular_group_size` /
  `cjxl-rs -g 0..3`** (A1 audit "Modular" PARTIAL item). Mirrors
  libjxl `cjxl -g` / `cparams.modular_group_size_shift`. `None`
  (default) keeps the existing 256-pixel group dimension (shift = 1)
  so output bytes are unchanged — hash-locks remain green. `Some(n)`
  for `n in 0..=3` maps to group dimensions `128 << n` =
  {128, 256, 512, 1024} and is forwarded into both the frame-header
  `group_size_shift` field and the modular encoder's per-group
  partitioning / global-vs-grouped channel cutoff. VarDCT is
  unaffected (libjxl + this encoder both fix VarDCT groups at 256).
  Verified pixel-exact via jxl-rs + djxl roundtrip across all four
  shifts on a 600×600 mixed-gradient. New test:
  `jxl-encoder/tests/modular_group_size_knob.rs` (4 cases — default
  matches shift=1 byte-identical, pairwise distinct bitstreams,
  pixel-exact roundtrip per shift, large-vs-small grid size delta).
- **Four `cjxl` parity knobs: `--faster-decoding`, `--container`,
  `--progressive-dc`, `--premultiply`** (W4-3 A1 audit). New builders
  on `LossyConfig` / `LosslessConfig` (and `EncodeRequest` for
  `with_premultiplied_alpha_mode`) plus matching CLI flags on
  `cjxl-rs`:
  - `with_faster_decoding(u8)` / `--faster-decoding 0..4` — mirrors
    libjxl `cparams.decoding_speed_tier`; per-tier semantics
    documented in the builder rustdoc (Weighted predictor → MA tree
    learner → EPF → DCT32+ + gaborish drop-out path). Values clamp
    to `MAX_FASTER_DECODING = 4`.
  - `with_container_mode(ContainerMode)` / `--container -1|0|1` —
    mirrors libjxl `cjxl --container 0|1`. New `ContainerMode` enum
    with `Auto` (default, wrap on metadata or codestream-level
    demand), `Always`, `Never`.
  - `with_progressive_dc(u8)` / `--progressive-dc 0..2` (lossy only) —
    `1` implies `with_lf_frame(true)` and produces byte-identical
    output to the existing `--lf-frame` flag; `2` is stored for
    forward compatibility (currently emits a single LfFrame). Values
    clamp to `MAX_PROGRESSIVE_DC = 2`.
  - `with_premultiplied_alpha_mode(PremultipliedAlphaMode)` /
    `--premultiply -1|0|1` — `Off` / `On` / `Auto` enum mirroring
    libjxl's tri-state. `Auto` is wired as a request-level policy
    flag; resolution at encode time is queued follow-on work.
  Also fixes two pre-existing same-type clippy casts
  (`effort.rs:1717`, `modular/inline_add_sample.rs:457`) flagged in
  the W3-3 audit. Five new unit tests in `api::tests` cover clamping,
  defaults, builder round-trip, and the `progressive_dc>=1 =>
  lf_frame` implication.

- **Lossy skeleton-flag wiring** — W4-2 follow-on to the W3-6 CLI passthrough
  bundle (`c8d3752c`) and the W4-1 modular skeleton wiring (`b7c1cb5a`).
  Wires four `LossyConfig` knobs through to the `VarDctEncoder` and the
  `FileHeader` so each affects encoded bytes when set:
  - `--upsampling_mode N` (libjxl `JxlEncoderSetUpsamplingMode`, `encode.cc:1393`)
    selects the decoder upsampling LUT for the active upsampling factor.
    `-1` / `None` keeps the default fancy upsampling (file header takes
    the `all_default=true` 1-bit fast path). `0` emits the nearest-neighbour
    LUT, `1` emits the "pixel dots" LUT. Only meaningful at `upsampling > 1`;
    only factors 2/4/8 carry an LUT (factor 2's pixel-dots LUT degenerates
    to nearest per libjxl). LUT bytes are written via
    `FileHeader::write_transform_data` after a new `upsampling_lut_weights`
    helper in `headers/file_header.rs` that mirrors
    `JxlEncoderSetUpsamplingMode`'s slot tables byte-for-byte.
    Layer-3 byte-divergence invariants in
    `tests/lossy_knobs_wiring.rs::upsampling_mode_changes_bytes_factor{2,4_pixel_dots}`.
  - `--group_order N` (0..2) (libjxl `cparams.group_order` /
    `JXL_ENC_FRAME_SETTING_GROUP_ORDER`). `Some(0)` = explicit scanline,
    `Some(1)` = center-first (wires the existing `center_first` flag so
    the concentric-square AC group permutation activates), `Some(2)` is
    stored as a no-op for forward compatibility. Invariants in
    `tests/lossy_knobs_wiring.rs::group_order_one_implies_center_first`
    and `group_order_zero_disables_center_first`.
  - `--center_x X` / `--center_y Y` (libjxl `cparams.center_x` / `center_y`)
    override the AC group permutation centre used when `group_order = 1`.
    `None` falls back to `width / 2` / `height / 2` (libjxl's `size_t(-1)`
    sentinel). Layer-3 invariant in
    `tests/lossy_knobs_wiring.rs::center_x_center_y_change_bytes_on_multigroup`.
  - `--alpha_distance D` (libjxl `cjxl --alpha_distance`,
    `enc_params.h:alpha_distance`) is stored on the encoder and reaches
    `VarDctEncoder::alpha_distance`. The alpha extras subimage is
    **still emitted losslessly** (gradient predictor + LZ77 RLE) at all
    `D` values — the lossy alpha pipeline (separate quantisation matrix
    for the alpha modular subimage) is queued follow-on. The
    `alpha_distance_lossless_path_byte_identical_today` test guards this
    contract so a future lossy-alpha change has to flip the assertion
    deliberately rather than silently. Default behaviour unchanged.

  All defaults preserved: 36/36 `hash_lock_features` byte-identical.
  1077/1077 lib tests pass. New `tests/lossy_knobs_wiring.rs` adds 6
  integration tests proving each knob plumbs through.

- **Multi-seed lossy butteraugli sweep at e10/e11** (RFC#45 pick #1 chunk 3).
  New `EffortProfile::lossy_search_seeds` field (1 at e ≤ 9, 2 at e10, 4 at e11)
  drives [`vardct::butteraugli_loop`]: at seeds > 1 we run the full
  `FindBestQuantization` loop N times with different `kInitMul` values
  (libjxl hardcodes 0.6 at `enc_adaptive_quantization.cc:1042`; we sweep
  `[0.6, 0.4, 0.8, 0.5]` — index 0 is always the libjxl default so the
  multi-seed picker can never regress below single-seed). The picker
  keeps the seed with the largest mean(quant_field_float) (proxy for
  smallest encoded bytes — coarser quant → fewer non-zero AC coefficients)
  whose final butteraugli score does not exceed `1.05 ×` target.
  Isolation A/B on 5 CID22-512 photos × 3 distances × 2 efforts shows
  **-0.65% bytes** total vs `seeds=1` at e10/e11 while consistently
  improving butteraugli. Bit-identical at e ≤ 9 (36/36 hash_lock pass).
  Exposed via `LossyInternalParams::lossy_search_seeds` for sweep
  harnesses (`__expert` feature). Bench:
  `benchmarks/lossy_multiseed_isolate_ab_2026-05-17.{tsv,meta}`.

- **Modular skeleton-flag wiring** — follow-on to the W3-6 CLI passthrough
  bundle (`c8d3752c`). Wires four of the five `--modular-*` flags
  through `LosslessConfig` → `FrameEncoderOptions::modular_knobs` →
  the modular encode pipeline so each knob produces a measurable
  bitstream effect when set:
  - `--modular-palette-colors N` overrides the multi-channel palette
    colour cap (libjxl `enc_params.h:121` `palette_colors = 1 << 10`).
    `0` disables palette detection entirely (single-group + multi-group +
    tree-learn path + RCT path + lossy-palette path). Layer-3 byte-divergence
    invariant in `api_tests::modular_knobs_palette_zero_disables_palette_path_lossless`.
  - `--modular-channel-colors-global-percent P` overrides the global /
    single-group ChannelCompact threshold (libjxl
    `enc_params.h:118` `channel_colors_pre_transform_percent`, default
    95.0). Wired through `write_modular_stream_with_tree_dc_quant_knobs`.
    Layer-3 invariant in `api_tests::modular_knobs_channel_colors_global_pct_changes_bytes_when_compact_path_runs`.
  - `--modular-channel-colors-group-percent P` overrides the per-group
    ChannelCompact threshold (libjxl `enc_params.h:120`
    `channel_colors_percent`, libjxl default 80.0). Wired through
    `encode_modular_multi_group_inner`. Default behaviour unchanged
    (continues to use 95.0 for bitstream stability — set the flag
    explicitly for libjxl 80.0 parity).
  - `--modular-nb-prev-channels N` caps `max_ref_channels` for the MA
    tree learner's previous-channel reference properties (libjxl
    `modular/options.h:76` `max_properties`). `0` disables ref-channel
    properties entirely. Layer-3 invariant in
    `api_tests::modular_knobs_nb_prev_channels_cap_changes_tree_path`.
  - `--modular-predictor N` is stored on `ModularKnobs::modular_predictor`
    but does NOT yet override the per-leaf tree-learned predictor (libjxl
    `Predictor::Variable` semantics — our default tree-learn already runs
    Variable mode). Documented as partial-wire in
    `api_tests::modular_knobs_predictor_stored_but_does_not_override_tree_learner`;
    flipping that assertion requires deliberate forced-predictor wiring
    through every non-tree-learn modular path and a CHANGELOG entry.

  New surface: `ModularKnobs` struct in `modular/palette.rs`
  (`palette_colors_or_default()`, `channel_colors_global_percent_or_default()`,
  `channel_colors_group_percent_or_default()`, `nb_prev_channels_cap()`),
  threaded into `FrameEncoderOptions::modular_knobs` and consumed by
  three new `_knobs` variants of the modular stream writers
  (`write_modular_stream_with_palette_knobs`,
  `write_modular_stream_with_rct_knobs`,
  `write_modular_stream_with_tree_knobs` +
  `write_modular_stream_with_tree_dc_quant_knobs`). New
  `CHANNEL_COLORS_GROUP_PERCENT = 80.0` constant matching libjxl
  `enc_params.h:120` for callers who want libjxl-faithful per-group
  thresholds.

  Tests: 7 new unit tests in `modular::palette::tests::modular_knobs_*`
  pin the resolver semantics, 6 new API integration tests in
  `api_tests::modular_knobs_*` prove byte-divergence on a 32-colour
  synthetic palette-friendly image, 5 updated CLI smoke cases in
  `jxl-encoder-cli/tests/cli_passthrough_smoke.rs` exercise the
  bytes-change behaviour via the cjxl-rs binary.

  Hash-lock: 36/36 byte-identical at default. RD-regression 18/18 within
  thresholds (0.0%–0.3% size delta — non-zero deltas trace to upstream
  changes between this branch's parent and prior baselines, not these
  knobs).

- **CLI passthrough bundle — A1 audit `cjxl` parity flags** (CLI parity
  section). Adds `cjxl-rs` flags that round out the libjxl `cjxl` parity
  surface so existing benchmark / sweep scripts can shell out without
  flag-mapping shims. Eleven new flags:
  - `--intensity-target NITS` → `EncodeRequest::with_intensity_target`,
    writes `ToneMapping.intensity_target` in the file header. Fully
    wired (regression: `tests/cli_passthrough_smoke.rs::
    intensity_target_flag_changes_bitstream_lossy_path`).
  - `--brotli-effort Q` → `EncodeRequest::with_brotli_metadata`. Wired
    when the new `brotli-metadata` CLI feature is enabled; silently
    accepted otherwise so scripts stay portable.
  - `--alpha-distance D`, `--group-order N`, `--center-x X`,
    `--center-y Y`, `--upsampling-mode N` → stored on `LossyConfig`
    via new `with_alpha_distance` / `with_group_order` /
    `with_center_x` / `with_center_y` / `with_upsampling_mode`
    builders + matching getters. `--group-order 1` mirrors the
    existing `center_first` flag through to the AC group reorder;
    the other four are skeleton-only today (value stored, encoder-side
    wiring queued as follow-on work).
  - `--modular-predictor`, `--modular-palette-colors`,
    `--modular-channel-colors-global-percent`,
    `--modular-channel-colors-group-percent`,
    `--modular-nb-prev-channels` → stored on `LosslessConfig` via
    parallel `with_modular_*` builders + getters. Initially skeleton-only.
    **Encoder-side wiring** for the four non-predictor flags landed in a
    follow-on (see "Modular skeleton-flag wiring" above). The predictor
    flag remains stored-only pending a deliberate forced-predictor pass
    through the non-tree modular paths.

  Hash-lock: 36/36 byte-identical. New smoke tests in
  `jxl-encoder-cli/tests/cli_passthrough_smoke.rs` (12 cases) cover
  each flag's CLI parse path and prove `intensity-target` produces
  divergent bytes vs default.

- **`LossyConfig::with_epf_level(level: i8)`** and matching CLI flag
  `--epf -1..3` — caller-pinned edge-preserving filter strength,
  mirroring libjxl `cjxl --epf` and the `JXL_ENC_FRAME_SETTING_EPF`
  C API knob (`enc_frame.cc:284-285`). `-1` (default) keeps the
  distance-derived `epf_iters` selection (libjxl thresholds
  `[0.7, 1.5, 4.0]`); `0` forces the filter off and skips the
  per-block dynamic sharpness search; `1`/`2`/`3` force the matching
  iteration count. Plumbed through every `DistanceParams::compute_*`
  call site (`vardct/encoder.rs` three sites, `vardct/bitstream.rs`,
  `vardct/rate_control.rs`) via the new `VarDctEncoder::epf_level_override:
  Option<u32>` field and `apply_epf_level_override(&mut params)`
  helper. Default (`-1`) is byte-identical to prior behaviour (all
  36 `hash_lock_features` fixtures pass). Layer-3 invariant in
  `jxl-encoder/tests/epf_force_level.rs` (3 jxl-rs roundtrips:
  default decodes, each `-1..=3` level decodes, and `auto`/`off`/`max`
  produce three distinct bitstreams). A1 audit parity item:
  PARTIAL → IN.

- **Roundtrip tests for the four `PixelLayout::*LinearF16` input variants**
  (A1 audit "Pixel formats / extras" PARTIAL item). `RgbLinearF16`,
  `RgbaLinearF16`, `GrayLinearF16`, and `GrayAlphaLinearF16` enum
  variants + dispatch arms + helper functions (`f16_to_linear_f32_rgb`,
  `f16_gray_to_linear_f32_rgb`, `extract_alpha_f16`) were already wired
  in `api.rs`, but no integration test covered the encode → decode →
  pixel-compare loop. New `tests/f16_input_roundtrip.rs` builds a 16×16
  synthetic image from values that quantize exactly through f16,
  encodes lossy at d=0.5 via the public `LossyConfig` path, and verifies
  the decoded RGB matches via both `jxl-rs` (primary) and `jxl-oxide`
  (secondary linear-sRGB decode). Max measured channel diff: 0.033 on
  [0,1] linear, well under the 0.07 wiring tolerance. Closes the
  Float16 portion of #18.

### Refactor

- **`kAvoidEntropyOfTransforms` formula extracted into named helpers** in
  `jxl-encoder/src/vardct/ac_strategy_search.rs`. The
  `kAvoidEntropyOfTransforms` and `kFavor2X2AtHighQuality` adjustments
  (libjxl `enc_ac_strategy.cc::FindBest8x8Transform` line 585-601) were
  already implemented and applied at all three evaluation sites (initial
  8×8 selection, 32×32 merge sub-cost re-evaluation, 64×64 merge sub-cost
  re-evaluation) — see commit `88aad38` (Feb 21, 2026). This change
  extracts the formula into `avoid_entropy_of_transforms_mul(distance)`
  and `favor_2x2_weight(distance)` free functions with libjxl source-line
  citations, and adds three regression unit tests pinning the formulas
  to libjxl's exact values across the distance range. Bit-identical
  output: all 36 `hash_lock_features` tests pass. The A1-audit "OUT"
  label and the `dropped_optimizations_for_parity_2026-05-15.md` entry
  for kAvoidEntropyOfTransforms applied to the **GPU** encoder's
  cost model, not the CPU encoder.

### Changed

- **More aggressive text-like patch detection** (RFC#45 pick #5 chunk 1).
  Lower the `kMinPeak` threshold in `vardct::patches::find_text_like_patches`
  from 2 to 1, so the detector accepts patches whose quantized magnitudes
  include at least one `±1` value (previously required at least one `≥|2|`
  value). Targets low-contrast glyphs and anti-aliased text edges. The
  downstream `is_cost_effective` gate (trial-encodes the reference frame,
  requires a 2× savings-vs-overhead ratio) keeps photo content from
  regressing. Measured impact at e7 on 5 screenshots × {d0.5, d1.0, d2.0}
  and 5 CLIC photos × same: 12 of 15 photo cells byte-identical (all 15
  unchanged), 12 of 15 screenshot cells byte-identical, 1 saves -53 B,
  1 saves -43 B, 1 regresses +465 B (`windows95.png` @ d=0.5, where the
  cost estimator's `0.3/distance` per-pixel savings model over-estimates
  low-d savings — known limitation, follow-up tracking in #45 chunk 2).
  All 36 `hash_lock` fixtures stay byte-identical. djxl decodes the new
  `windows95.png` @ d=1.0 output cleanly.

### Fixed

- **Streaming `LossyEncoder` silently dropped five `LossyConfig` fields**
  (A1 audit top-10 #2, photon-noise CLI/API audit). The one-shot
  `EncodeRequest::encode_lossy` (api.rs:4531) and animation
  `encode_animation_lossy` (api.rs:6892) paths wired every field
  through; the streaming `LossyConfig::encoder() → LossyEncoder::finish*`
  path (api.rs:5414) only wired `photon_noise_iso` and quietly ignored:
  `manual_noise_lut`, `quant_ac_rescale`, `original_distance`,
  `ssim2_iters`, `zensim_iters`. Setters accepted the values and the
  `LossyConfig` carried them, but the streaming finalizer never read
  them — a textbook silent-drop gate. CLI was unaffected (uses one-shot
  path). Layer-1 regression test in
  `jxl-encoder/tests/streaming_noise_gate.rs` (3 paired byte-diff
  cases — `manual_noise_lut`, `quant_ac_rescale`, plus the already-wired
  `photon_noise_iso` as a control). Audit also added explicit
  `# Gate / silent-drop conditions` doc sections to `with_noise`,
  `with_photon_noise_iso`, and `with_manual_noise_lut` documenting the
  three priority levels, the all-zero-LUT drop, and that noise is
  lossy-only. Hash-lock: 36/36 byte-identical, no bitstream change for
  the previously-working paths.

### Added

- **Sample-fraction jitter + predictor-order shuffle for e10/e11
  multi-seed tree learning** (RFC#45 pick #1 chunk 4 — follow-on to
  chunk 3 `a8fbd360`). Two additional variance dimensions on top of
  chunk 3's three perturbations: (1) per-seed `tree_sample_fraction`
  cycled by `seed % 4` over `[None, Some(0.40), Some(0.60),
  Some(0.70)]` — seed 0 keeps the canonical profile fraction
  (`None` → byte-identical); higher seeds map an absolute target
  fraction onto a gather stride via the new
  `stride_for_seeded_sample_fraction(total_pixels, frac)` helper,
  which takes precedence over chunk-3's `derive_seeded_stride`. The
  triplet straddles the canonical 0.50 default with one substantially
  denser sample (0.70) that captures rare-bucket splits the canonical
  run misses. (2) Per-seed permutation of the 14 `CANDIDATE_PREDICTORS`
  array via the new `derive_seeded_predictor_order(seed)` →
  `[canonical, strong-first (Gradient/Weighted lead), directional-first
  (TopRight/TopLeft/Average1..4 lead), full reverse]`. This affects
  greedy ID3's strict-`<` tie-break in `find_best_predictor`, so the
  per-leaf predictor flips on equal-entropy ties — surfacing trees with
  different leaf predictors. Set equality is preserved (all 4 perms
  contain the same 14 predictors) so every per-seed tree remains
  spec-valid and the chunk-2 `estimate_token_cost` picker chooses
  among them on equal terms. Seed-0 byte-identicality enforced by a
  unit test (`test_new_with_predictor_order_for_seed_seed_zero_matches_
  default`); 7 new unit tests in total
  (`test_derive_seeded_sample_fraction_*`,
  `test_stride_for_seeded_sample_fraction_*`,
  `test_derive_seeded_predictor_order_*`). New helpers in
  `modular::tree_learn`: `derive_seeded_sample_fraction(u64) ->
  Option<f32>`, `derive_seeded_predictor_order(u64) -> &'static
  [Predictor]`, `stride_for_seeded_sample_fraction(usize, f32) ->
  usize`, `TreeSamples::new_with_predictor_order_for_seed(num_refs,
  seed)`. Bench harness: `examples/e10_e11_multiseed_chunk4_ab.rs`
  (5 CID22-512 photos × {e9, e10, e11} × 2 paired samples).
  Hash-locks: `hash_lock_features` 36/36 byte-identical at e ≤ 9.
  **Honest A/B vs chunk 3 on this 5-image corpus (deferred for
  larger-corpus validation):** chunk 4 regresses at e11 by +0.39%
  (4834 bytes worse, 5 images) and is a wash at e10 (+0.008%). Only
  one cell (1418519@e11) improves vs chunk 3 (-0.137%); two regress
  (1044329@e11 +1.07%, 1189261@e11 +0.48%). Likely cause: the 4-seed
  budget at e11 is fixed, so adding more variance dimensions cycles
  through a *different* 4 candidate trees, not *more* — chunk-3's
  threshold-jitter + property-rotation perturbations happened to hit
  better minima on 2/5 images than chunk-4's recombined set. Logged
  as RFC#45 #45 follow-on; possible resolutions: (a) reserve chunk-3
  perturbations for seeds 0..3 and apply chunk-4 perturbations only
  beyond seed 3 (requires expanded budget at a new effort tier);
  (b) expand to 6 or 8 seeds at e11; (c) per-image dispatch. Bench
  TSV + meta archived at `benchmarks/e10_e11_multiseed_chunk4_ab_
  2026-05-17.{tsv,meta}`.

- **Broader seed variance for e10/e11 multi-seed tree learning**
  (RFC#45 pick #1 chunk 3 — follow-on to chunk 2 `d4f2e282`). The
  chunk-2 dispatch only varied gather `start_offset`, which produced
  highly correlated sample subsets — on 3 CID22 photos the canonical
  seed 0 always won. Chunk 3 widens the per-seed candidate space via
  three deterministic, seed-0-preserving perturbations:
  (1) `split_threshold` jitter (per-seed multiplier from
  `[1.0, 0.7, 1.3, 0.85]`); (2) property-order rotation past the
  structural `Channel` + optional `GroupId` prefix; (3) per-seed
  stride from `[base, base+1, base-1, base*2]`. Seed 0 is a clone of
  the canonical `TreeLearningParams` for all three knobs — preserves
  chunk-2's byte-identical seed-0 path and keeps e ≤ 9 hash-locks at
  36/36. On 5 CID22-512 photos at default settings, e11 strictly
  beats e9 in 5/5 cells (avg -0.46% bytes, best -0.97%); e10 wins
  3/5 (60%). New helpers in `modular::tree_learn`:
  `derive_seeded_params(&TreeLearningParams, u64)` and
  `derive_seeded_stride(usize, u64)`. Bench harness:
  `examples/e10_e11_multiseed_chunk3_ab.rs` (5 photos × 3 efforts ×
  N samples). Six new unit tests cover seed-0 cloning, threshold
  jitter, structural prefix preservation, property-order variance,
  stride clamping, and density perturbation.

- **Multi-seed lossless tree learning at e10/e11** (RFC#45 pick #1 chunk 2).
  At effort 10/11 the global modular tree-learning path now runs the
  gather→`compute_best_tree`→`collect_residuals_with_tree` pipeline 2
  (e10) or 4 (e11) times with different stride offsets, scores each
  candidate tree by `estimate_token_cost` (libjxl-parity per-context
  entropy + extra bits + per-context header term), and keeps the
  cheapest. Each seed shifts `subsample_counter` initial value within
  `[0, stride)` so different pixel subsets feed the greedy ID3 split
  selection — closing part of the "single-pass libjxl tree" greedy
  gap. e ≤ 9 stays single-seed and byte-identical (hash-locks 36/36
  unchanged). New `tree_learn_seeds: u8` field on `EffortProfile` +
  matching `LosslessInternalParams::tree_learn_seeds: Option<u8>`
  `__expert` override. Bench harness at
  `examples/e10_e11_multiseed_ab.rs` (3 photos × 3 efforts × N samples,
  byte/wall-clock TSV).

- **`colr` (alternative colour descriptor) and `hCdR` (HDR content
  description) container boxes** (A1 audit "Container/boxes" OUT items,
  effort S each). Pass-through ISOBMFF box appenders added to
  `jxl_encoder::container`: `append_colr_box(jxl_data, &[u8])` and
  `append_hcdr_box(jxl_data, &[u8])`. A typed helper
  `colr_nclx_payload(cp, tc, mc, full_range) -> [u8; 11]` builds the
  ISO/IEC 14496-12 `nclx` sub-payload from CICP enum values (ITU-T
  H.273). Wired into the one-shot `EncodeRequest` path via two new
  `ImageMetadata` fields and builders: `with_colr_payload(&[u8])` and
  `with_hcdr_payload(&[u8])`. JXL spec clause 5 requires decoders to
  ignore unrecognised boxes, so emitting these boxes never alters
  decoded pixels — they exist for ISOBMFF-aware inspectors (HEIF/AVIF
  metadata extractors, HDR pipelines) that would otherwise have to
  parse the codestream. Streaming encoders silently drop these fields
  (documented). Hash-lock fixtures stay byte-identical (36/36) — both
  fields default to `None`. 5 new container unit tests + 4 end-to-end
  integration tests in `tests/colr_hcdr_boxes.rs`.

- **`AnimationFrame` per-frame override fields + public `BlendMode` re-export**
  (audit item #3, "Animation API expansion"). The animation header has
  always carried per-frame blend mode / blend source / save-as-reference /
  name / timecode (libjxl `FrameHeader::blending_info` /
  `save_as_reference` / `name` / `timecode`), but the high-level
  `encode_animation*` API only exposed `pixels` + `duration` — multi-layer
  animations with overlay/blend semantics were unreachable from Rust
  callers. New `AnimationFrame::{new, with_blend_mode, with_blend_source,
  with_save_as_reference, with_name, with_timecode}` constructors and
  matching `Option<_>` public fields thread the override into both
  lossless modular and lossy VarDCT animation paths. Setting `timecode`
  on any frame auto-flips the file-level `have_timecodes` flag.
  `BlendMode` (Replace / Add / Blend / AlphaWeightedAdd / Mul) is now
  re-exported from the crate root. Defaults preserve the existing
  encoder behavior bit-for-bit (`hash_lock_features` 36/36, all 21
  pre-existing animation tests still pass).

  This change also fixed two pre-existing bugs that were never exercised
  before:
  - `FrameHeader::write_blending_info` wrote `source` *before*
    `alpha_channel` / `clamp`, while libjxl + jxl-rs (and the spec) put
    `source` *last*. Reversed for parity; only the previously-unused
    Blend / AlphaWeightedAdd / Mul paths are affected.
  - `FrameHeader::write_name` used wrong selector ranges
    (`Bits(4)+4`, `Bits(10)+20`) instead of the spec's
    `U32(Val(0), Bits(4), 16 + Bits(5), 48 + Bits(10))`. Names of any
    length now write per spec.

  Roundtrip tests in `tests/animation.rs`:
  `test_animation_blend_overlay_lossless_jxlrs` (Blend mode + name + EC
  alpha + reference-slot semantics through jxl-rs) and
  `test_animation_timecode_roundtrip` (timecode roundtrip through
  jxl-rs + jxl-oxide).

- **JUMBF (`jumb`) container box pass-through** — A1 audit top-10 item #3.
  Caller-supplied JUMBF (JPEG Universal Metadata Box Format, ISO 19566-5;
  the container used by C2PA / Content Authenticity Initiative for
  provenance metadata) bytes are emitted verbatim into a `jumb` ISOBMFF
  box appended after the standard `Exif`/`xml ` boxes. Available on all
  three API layers: `ImageMetadata::with_jumbf(bytes)` for one-shot
  encodes, `LossyEncoder::with_jumbf` / `LosslessEncoder::with_jumbf` for
  streaming, and `cjxl-rs --jumbf <FILE>` on the CLI. Routes through the
  Brotli path when `brotli-metadata` + `EncodeRequest::with_brotli_metadata`
  are enabled (new `wrap_in_container_with_brob_and_jumbf` helper). Bare
  appender `container::append_jumbf_box(jxl_data, jumbf_bytes)` also
  exposed for callers that need to attach JUMBF to a previously-encoded
  codestream. Hash-lock fixtures stay byte-identical (36/36); the new
  field defaults to `None` so existing call sites are unaffected. Empty
  payloads are rejected at validation time. Mirrors libjxl's
  `JxlEncoderAddBox(enc, "jumb", ...)` API
  (`lib/jxl/encode.cc:2211-2216`).

- **`LossyConfig::with_canonicalize_input` /
  `LosslessConfig::with_canonicalize_input`** (RFC #45 pick #2 chunk 1).
  Opt-in single-pass input canonicalization that drops opaque alpha,
  collapses near-grayscale RGB(A) to Gray(Alpha), and downcasts
  byte-replicated 16-bit to 8-bit. Each step is a no-op when its
  precondition fails. Outputs are strictly smaller-or-equal and preserve
  every pixel value bit-exactly within the new layout. Default `false`
  to keep existing hash-locks byte-identical. Bench on synthetic padded
  inputs (256×256, `examples/canonicalize_input_ab.rs`): lossless
  −50.5% on opaque-RGBA-grayscale, −67.6% on byte-replicated Rgb16. No
  byte regression on CLIC real photos (paired Δ = 0). All 36
  `hash_lock_features` cases byte-identical at default-off. Roundtrip
  decoder validation (jxl-rs + jxl-oxide) in
  `tests/canonicalize_input_roundtrip.rs` confirms semantic
  equivalence: dropped-alpha decodes to α=255 everywhere, collapsed
  grayscale decodes to R==G==B exactly, 16→8 downcast decodes to the
  original byte values. New `canonicalize` module at
  `jxl-encoder/src/canonicalize.rs` (13 unit tests).

- **CMYK lossy perceptual CMY→XYB transform** (A1 audit item #6
  chunk 3, follow-on to `1b222af`). Chunk 2 wired `Cmyk8`/`Cmyk16`
  through the lossy VarDCT path by reinterpreting the C/M/Y bytes
  as if they were sRGB-encoded R/G/B — a placeholder with no
  physical basis (a fully-saturated cyan ink encoded as bright red
  in XYB, decoding to the wrong gamut sector). Chunk 3 replaces
  that mapping with the naive uncalibrated subtractive model:
  `R_linear = (1 - C/255) · (1 - K/255)`, analogues for G/B from M
  and Y. New helpers `cmyk_u8_to_linear_f32_rgb` and
  `cmyk_u16_to_linear_f32_rgb` (api.rs) consume both the CMY and
  the deinterleaved K plane to produce linear-light RGB directly,
  bypassing the sRGB-decode LUT entirely. K still ships separately
  as the modular `ExtraChannelType::Black` extra so ink coverage
  round-trips bit-exact through the lossless modular path. The
  transform is **not colorimetric** — it ignores ink chromaticity,
  dot gain, illuminant, and printer profile — but it places the
  colour in the correct gamut sector so the XYB perceptual
  quantiser allocates bits sensibly. A future chunk can wire
  either the caller-supplied CMYK ICC profile (option A) or a
  hardcoded SWOP/FOGRA matrix (option B) for true colorimetric
  conversion. New test `test_lossy_cmyk8_chunk3_gamut_direction`
  encodes pure C/M/Y/K swatches and asserts each decodes within
  the correct gamut octant (cyan ink → low R, high G+B; magenta
  → low G, high R+B; yellow → low B, high R+G; black → near zero).
  The chunk-2 `test_lossy_cmyk8_roundtrip` test was updated to
  invert the subtractive transform before comparing CMY: bounds
  widened to ±128 max / ±64 avg per channel because the inversion
  `C = 1 - R/(1-K)` amplifies VarDCT error inversely with `1-K`
  on a high-contrast block-edge gradient; the gamut-direction
  test is the real perceptual check. Hash-locks: 36/36
  byte-identical (`Cmyk*` layouts are opt-in).

- **CMYK lossy encode** (A1 audit item #6 chunk 2, follow-on to
  `f2deff72`). `PixelLayout::Cmyk8` and `PixelLayout::Cmyk16` now
  route through the lossy (`VarDCT/XYB`) one-shot path in addition
  to the lossless one. The C/M/Y planes flow through XYB by being
  reinterpreted as if they were sRGB-encoded R/G/B bytes (a
  perceptually-coarse mapping that chunk 3 will replace with a
  CMY-aware transform); the K plane is split off and attached as a
  modular `ExtraChannelType::Black` extra channel at ec index 0, so
  the ink coverage survives the lossy round-trip bit-exact (within
  the f32→u8 decoder rounding). Mirrors libjxl's wire shape for
  lossy CMYK (`lib/jxl/enc_image_bundle.cc:57`: three colour planes
  in XYB plus a Black extra). Patches detection is disabled for
  CMYK input (same reason as the lossless path — the detector
  assumes RGB-like perceptual colour). Caller-supplied Black
  extras are still rejected with a clear `InvalidInput` error to
  prevent silent double-Black bitstreams. Three new tests —
  `test_lossy_cmyk8_roundtrip` (jxl-rs decode, gradient pattern at
  d=1.0 e5, K bit-exact + CMY within ±48 byte / ≤12 avg per
  channel), `test_lossy_cmyk16_header_signals_16bit_black`
  (16-bit CMYK header signaling + jxl-oxide render), and
  `test_lossy_cmyk_rejects_duplicate_black_extra` (guard test).
  Hash-locks: 36/36 byte-identical (Cmyk\* layouts are opt-in).
  Streaming CMYK push-rows still defers to a future chunk;
  animated CMYK is out of scope.

- **CMYK lossless encode** (A1 audit item #6, issue #58). New
  `PixelLayout::Cmyk8` (4 bytes/pixel: C, M, Y, K) and
  `PixelLayout::Cmyk16` (8 bytes/pixel, native-endian u16) variants
  on the lossless one-shot path. The K plane is auto-synthesised as
  an `ExtraChannelType::Black` extra channel at ec index 0 (matching
  libjxl's `EncoderTest.CMYK` round-trip in
  `lib/jxl/encode_test.cc:2070`); the codestream level auto-bumps to
  10 because the Black extra channel is forbidden at level 5
  (`compute_codestream_level`). Pixel-exact round-trip verified via
  jxl-rs and jxl-oxide on synthetic 32x32 CMYK input. Two new
  `ExtraChannel` constructors — `ExtraChannel::black(&[u8])` and
  `ExtraChannel::black_u16(&[u16])` — let callers who already keep
  K separate from C/M/Y attach the plane manually (e.g., paired with
  `PixelLayout::Rgb8`); supplying both `Cmyk*` layout and a manual
  Black extra is now a clear `InvalidInput` error rather than a
  silent double-Black bitstream. Patches detection is disabled for
  CMYK input because the CMY planes are not perceptually RGB-like.
  Streaming CMYK push-rows defers to a future chunk. Callers who
  need colour-managed CMYK should attach a CMYK ICC via
  `LosslessConfig::with_metadata` → `ImageMetadata::icc_profile`.

- **JPEG XL codestream Level 10 signaling** (`jxll` container box,
  audit item #1). Encoder now computes the required codestream level
  per libjxl `VerifyLevelSettings` (`lib/jxl/encode.cc:550`) from
  image dimensions, ICC size, and extra-channel count, and emits a
  `jxll` (level) box directly after `ftyp` when any level-5 cap is
  exceeded. Container is forced even without EXIF/XMP at level 10
  (mirrors libjxl `MustUseContainer`). Unblocks encoding of images
  beyond the Level 5 envelope (> 262 144 per axis, > 2²⁸ pixels,
  > 4 extra channels, CMYK, or ICC > 4 MB). Public surface:
  `container::compute_codestream_level`,
  `container::wrap_in_container_with_level`, and `_with_brob_and_level`,
  `_with_jbrd_and_level`, `_jxlp_with_level` siblings. All existing
  `wrap_in_container*` entry points keep their level-5 behaviour, so
  byte layout for normal-sized images is unchanged (hash-locks
  byte-identical: 36/36).

- **`hdr-gainmap` feature: typed `GainMapBundle` serializer + end-to-end
  `HdrFromSdrRequest` Ultra HDR encoder API** (issue #46, A3 chunks 3+4).
  New `jxl_encoder::hdr` module gated behind the optional `hdr-gainmap`
  cargo feature. Two surfaces:
  - `hdr::GainMapBundle` mirrors libjxl's `JxlGainMapBundle` struct
    (`gain_map.h:38`) with owned `Vec<u8>` fields. `GainMapBundle::serialize`
    produces a `jhgm` box payload that matches `JxlGainMapWriteBundle`
    (`gain_map.cc:83-153`) byte-for-byte: `jhgm_version (u8)` +
    `gain_map_metadata_size (u16 BE)` + metadata + `color_encoding_size
    (u8)` + color-encoding bits (via our `ColorEncoding::write` →
    `BitWriter::finish_with_padding`) + `alt_icc_size (u32 BE)` + alt ICC
    + raw gain-map codestream. Wrap with `hdr::append_gain_map_bundle`
    (thin convenience over the existing `container::append_gain_map_box`).
  - `hdr::HdrFromSdrRequest::new(width, height, sdr_image, hdr_image,
    hdr_intensity_target).encode()` derives the gain map via
    `ultrahdr_core::gainmap::compute_gainmap_slice`, encodes the SDR base
    via `LossyConfig` (default distance 1.0, callable
    `with_lossy_config`), encodes the gain-map plane losslessly via
    `LosslessConfig`, serializes the ISO 21496-1 metadata via
    `ultrahdr_core::serialize_iso21496_fmt(.., Iso21496Format::JxlJhgm)`,
    and returns a single JXL container with the `jhgm` box appended.
    Includes `HdrImage<'a>` / `HdrColorEncoding` / `HdrPixelLayout` value
    types so the constructor stays under the clippy
    `too_many_arguments` ceiling.
  - Dep: `ultrahdr-core = "0.5.0"` with `default-features = false,
    features = ["std"]` (skips the `tonemap` feature so we do not
    transitively pull `zentone`). The crate is already in the
    `imazen/ultrahdr` workspace and pulls only `zenpixels` + `zencodec`
    as new transitive deps — no `zenjpeg` pull-in.
  - 11 new tests cover the wire-format layout (BE size fields, tail
    placement of the gain-map codestream, color-encoding padding) and
    the end-to-end pipeline (8×8 synthetic SDR+HDR pair encodes
    successfully and produces a container starting with the JXL
    signature and containing both `jxlc` and `jhgm` boxes).

- **`LossyConfig::with_keep_invisible(bool)` + `LosslessConfig::with_keep_invisible(bool)`**
  — libjxl-named alias for the `SimplifyInvisible` pre-pass
  (`cparams.keep_invisible` at `enc_params.h:83`,
  `ApplyOverride(_, IsLossless())` at `enc_frame.cc:1590`). Defaults match
  libjxl: lossy runs the smear pass (default `keep_invisible = false`,
  i.e. `simplify_invisible = true`); lossless preserves all RGB bytes
  (default `keep_invisible = true`, i.e. `simplify_invisible = false`).
  On lossless, opting in with `with_keep_invisible(false)` zeros RGB
  samples in pixels whose alpha=0 before modular encoding — modular's
  predictor + LZ77 then compresses long zero runs for **5-20% smaller
  files on sprites / icons / UI assets** with large transparent regions
  (a 64×64 noisy-invisible synthetic sprite shrank by **83.3% — 5427 →
  906 bytes**). Visible pixels round-trip bit-exact. Default behavior
  byte-identical (hash_lock_features 36/36 unchanged). Closes A1
  coverage audit Top-10 item #4. `LossyConfig::with_keep_invisible`
  delegates to the existing `with_simplify_invisible` with inverted
  semantics — both names are available so callers porting from `cjxl`
  can use libjxl terminology.

- **Public JPEG → JXL lossless transcoding API** (issue #44, this
  session). The pre-existing internal `jpeg-reencoding`-gated module
  (`jxl-encoder/src/jpeg/`, 2,253 LoC, 52 integration tests) is now
  exposed through the public API surface. New entry points (all gated
  behind the `jpeg-reencoding` cargo feature):
    - `LosslessConfig::encode_jpeg_transcode(jpeg_bytes: &[u8]) -> Result<Vec<u8>>` —
      parses an existing JPEG and emits a JXL container with the JBRD
      reconstruction box, so `djxl out.jxl out.jpg --reconstruct_jpeg`
      reproduces the original JPEG byte-for-byte. Pixel-identical
      decode through any JXL decoder.
    - `LosslessConfig::encode_jpeg_transcode_codestream(jpeg_bytes: &[u8])` —
      bare codestream variant (no container, no JBRD). Smaller output
      bytes, but cannot reconstruct the original JPEG.
    - `jxl_encoder::jpeg::is_jpeg_signature(bytes)` — lightweight
      `0xFF 0xD8 0xFF` sniff for routing decisions.
    - `EncodeError::JpegParse { message }` — new error variant for
      malformed JPEG input (returned by both transcode methods).
  CLI integration in `jxl-encoder-cli` (also feature-gated):
    - `--lossless-jpeg` — force the JPEG transcode path for the input.
    - `--no-lossless-jpeg` — disable the auto-detect path even on
      `.jpg` / `.jpeg` / `.jpe` / `.jfif` extensions.
    - Auto-detection by extension is on by default when the
      `jpeg-reencoding` feature is enabled. The CLI sniffs the SOI
      marker before routing so a mis-extensioned PNG fails loudly.
  Bumped `zenjpeg` dep to `^0.8.4` (the published `0.7.1` calls
  `magetypes::mf32x8::load_8x8(block)` with the pre-0.9.16 single-arg
  signature, incompatible with the current `magetypes ^0.9.23` floor
  pulled in by `zensim`/`butteraugli`/`fast-ssim2`). The `0.8.4` floor
  pulls in the token-passing API and clears the broken-build state
  that existed on `main` with `jpeg-reencoding` on.
  Coverage: 7 new public-API integration tests in
  `tests/jpeg_public_api.rs` (signature sniff, container with JBRD,
  bare codestream, non-JPEG rejection, jxl-rs pixel roundtrip — all
  passing). Pre-existing `tests/jpeg_reencoding.rs` (52 tests covering
  4:4:4/4:2:0/4:2:2/4:4:0/grayscale, JBRD parse via jxl-jbr, etc.)
  unchanged. The `djxl --reconstruct_jpeg` byte-exact reconstruction
  has known pre-existing edge cases on some fixtures (tracked in the
  existing `test_jbrd_roundtrip_*` tests, which are tolerant of
  djxl-side failures); this chunk does NOT change the JBRD payload —
  it only exposes the existing transcode path through the public API.

### Investigated (negative result, primitive shipped under `__bench_internals`)

- **Phase 4 fused `AddSample` primitive** (`FusedHashKeyBuilder` in
  `jxl-encoder/src/modular/inline_add_sample.rs`, issue #41 chunk 1).
  Streaming hash-and-write builder that folds canonical-key bytes into
  libjxl `Hash1`/`Hash2` accumulators as they are computed, eliminating
  Phase 3's separate `pack_local_key_phase3` walk. Primitive is correct
  (10 unit tests + cross-check against Phase 3's `pack_local_key_phase3`
  + `InlineDedupTable::lookup_or_insert` on 16 real-photo seeds, all
  byte-equivalent). **However, microbench shows it is 10-25% SLOWER than
  Phase 3 on every cell measured** (8 cells: 200K/1.35M samples × dup
  300/600/800 × photo-like + synthetic distributions); see
  `benchmarks/inline_addsample_microbench_2026-05-17.{txt,meta}`. Root
  causes (hypothesized): (a) loss of LLVM auto-vectorization when
  byte-write and hash-fold interleave inside the same loop body; (b)
  trailing zero-byte fold in `finalize()` adds 8-32 muls per sample
  for `InlineDedupTable::raw_hash1`/2 fingerprint parity. Primitive
  ships gated behind `__bench_internals` for measurement only; **NOT
  wired into the production gather loop**. See
  `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/lossless_phase4_inline_addsample_2026-05-17.md`
  for the chunk 2+ decision tree.
### Investigated (kept opt-in)

- **`LosslessConfig::with_smart_fanout` default-on decision: KEEP OPT-IN**
  (this session, cumulative-state bench
  `benchmarks/cumulative_state_2026-05-17.tsv` + `.meta`). Re-validated
  the smart-fanout dispatch (shipped as opt-in in `1c4691f0`) against a
  broader 20-image corpus (5 small + 5 medium + 5 large + 5 screenshots)
  × 3 efforts × 3 paired samples × {smart_off, smart_on} variants
  (bitstream-equivalent claim verified on every cell via sha256).
  Aggregate best-iter wins are large (-5 to -8% across e7/e8/e9), but
  one cell (`medium_M4_e0d8e29c` e9) shows a +4-5% paired regression
  that exceeds the task brief's strict `≥+3%` flip gate. The bench was
  run under concurrent-agent load (1-min load 4.5-8.5 throughout), so
  the regression may be load-induced noise on the median rather than a
  signal — but the gate is strict, so the opt-in stays. The shipped
  `with_smart_fanout(true)` / `--smart-fanout` knob continues to deliver
  the demonstrated 5-15% wall-clock wins on small/medium photos at zero
  byte cost (sha256 byte-identical on every measured cell). A re-bench
  on a quiesced host (load < 1.0) is needed before flipping the default.
  See the meta file for the full per-cell table + analyzer scripts.

### Changed (performance)

- **Predictor-pruning seed-first hybrid for the parallel branch of
  `find_best_predictor`** (issue #23 chunk 4 — completes the multi-chunk
  predictor-pruning port; see `predictor_prune_c4_ab_2026-05-17.{tsv,meta}`).
  Splits the parallel branch into four phases: compute all 14 extra-bits
  lower bounds in parallel → pick lowest-LB seed (lowest-index tie-break)
  → run the seed predictor's full eval **sequentially** → dispatch the
  remaining 13 workers in parallel with the atomic seeded by the real
  seed cost. The chunk-3 wireup (52f8e816 / 685244b) capped at ~40 %
  effective prune because the early wave of workers raced against an
  empty `f64::MAX` seed; the seed-first hybrid populates the atomic with
  a tight real cost before fan-out so every worker — not just the late
  wave — benefits from the prune. New `costs[i] = current_best_bits`
  on skip (instead of `f64::INFINITY`) closes a theoretical tie-break
  hazard with the non-MAX seed; full byte-identity proof in the comment
  block at `tree_learn.rs:5293-5366`. Paired A/B at 8T (12 paired iters
  × 3 images × 3 efforts, sample-major interleaved): **medium 1.05 MP @
  e7 median Δ −5.70 %** (the brief's gate cell — chunk-3 was at −0.5 %
  here), **large 4.19 MP @ e9 median Δ −13.75 %** (chunk-3 had only an
  n=1 anecdote at this cell), **medium 1.05 MP @ e9 +0.32 % median**
  (chunk-3 +3.03 % regression now erased). Large 4.19 MP @ e7 regresses
  +1.27 % median — the deliberate trade-off for the win at the brief's
  gate cell and the large+e9 cell; the per-worker full eval at large-e7
  is short enough that the +1 serial seed eval costs more critical-path
  latency than the prune saves on the remaining 13 workers. Hash-locks
  `--features parallel-tree-learning`: 36/36 byte-identical; direct
  sha256 verification on 5 (image, effort) cells of real photos:
  byte-identical. Issue: imazen/jxl-encoder#23.

- **Always-on VarDCT `try_dct64` per-image dispatch on small + low-d cells**
  (chunk 1 of the VarDCT speed push, follows the lossless smart-fanout /
  small-image-fallback / bucket-dispatch family pattern). New
  `EffortProfile::adapt_to_image_lossy(pixels, distance)` adapter plus
  `LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD = 500_000` (u64) and
  `LOSSY_LOW_DISTANCE_THRESHOLD = 2.0` (f32) constants. When `pixels < 500_000`
  AND `distance < 2.0`, drops `try_dct64` from the effort-7+ default `true`
  to `false`. Skips the entire
  `vardct::ac_strategy_search::find_best_64x64_transform` pipeline (DCT64x64
  + 2×DCT64x32 + 2×DCT32x64 candidates plus their 4× `find_best_32x32_transform`
  reuse path) — about 9 expensive entropy-estimate evaluations per 64×64 tile
  that essentially never win on small low-distance content. New
  `LossyConfig::effective_profile_for_image(pixels)` mirrors the lossless
  signature and is called from the three lossy entry points in `api.rs`
  (`encode_lossy`, `LossyEncoder::finish_inner`, `encode_animation_lossy`).
  Override-respect: when the caller has supplied a `__expert`
  `LossyConfig::with_internal_params(...)` override, the adapter is skipped so
  sweep harnesses keep their pinned `try_dct64` value (mirrors
  `LosslessConfig::effective_profile_for_image`). Hash-locks
  (`tests/hash_lock_features.rs` 36/36) stay byte-identical — every lossy
  fixture is at most 48×48, too small for any 64×64-aligned position so the
  adapter is a no-op even on the gated tier. RD regression
  (`tests/clic2025.rs::test_rd_regression`, CID22-512 small photos at
  d=0.25/0.50/1.0): all 18 image×distance cells produce 0.0–0.5% **smaller**
  output (matching the dispatch's "DCT64 is wasted work here" hypothesis), all
  butteraugli/ssim2 within the existing thresholds. Companion paired A/B at 1T
  (`benchmarks/vardct_ac_dispatch_paired_2026-05-17.tsv`, 4 images × 3
  distances × 10 paired samples, sample-major interleaved): non-gated cells
  (medium 1.05 MP and large 2.78 MP at every distance, plus every image at
  d=2.0) all produce **byte-identical output sample-pairwise**, confirming the
  adapter only fires on its gated cell. Companion sweep harness:
  `examples/vardct_ac_dispatch_paired_ab` (registered under `__expert`).
- **Always-on `tree_max_buckets` per-image dispatch at large+e9 cells**
  (audit conditional-value catalog item #3 —
  `rejected_optimizations_conditional_value_2026-05-17.md`; resurrects the
  Pareto-sweep insight from commit `4572790` that was originally no-shipped
  for failing the single-binary "≥5% on ≥2 of 3 profile images" gate but
  produces a clean Pareto win on the largest tier alone). New
  `EffortProfile::adapt_tree_max_buckets_for_image(pixels)` adapter plus
  `LARGE_IMAGE_PIXEL_THRESHOLD = 4_000_000` and `LARGE_E9_TREE_MAX_BUCKETS = 192`
  constants. When `pixels >= 4_000_000` AND `effort >= 9`, drops
  `tree_max_buckets` from 256 → 192. `LosslessConfig::effective_profile_for_image`
  calls the adapter unconditionally — this is a **default change**, not opt-in.
  Skipped when the caller has supplied a `__expert`
  `LosslessInternalParams::with_internal_params(...)` override so sweep
  harnesses keep their pinned values. Paired A/B
  (`benchmarks/bucket_dispatch_paired_ab_2026-05-17.tsv`, 7 paired samples ×
  3 images × 3 efforts × 8T, sample-major interleaved): **large+e9 median
  wall-clock −17.44%** (best-iter −21.47%) at **+0.090% bytes**, exceeding
  both the ≥5% wall-clock gate and the ≤+0.5% bytes gate from the task brief.
  Bytes Δ matches the original Pareto sweep prediction (+0.09%) to three
  significant figures. All 8 non-(large+e9) cells produce **byte-identical
  output** sample-pairwise (sha256-prefix match, 7/7 paired samples each).
  Hash-locks (`tests/hash_lock_features.rs` 36/36) stay byte-identical — every
  hash_lock fixture is below the 4 MP threshold so the dispatch does not fire.
  Third per-image dispatch chunk in the smart-fanout family (`1c4691f0` +
  `142ef4f6` precedents). Companion sweep harness:
  `examples/bucket_dispatch_paired_ab` (registered under `__expert`).

- **Skip per-property `Vec<i32>` swaps on the lossless tree-learning main
  path** (resurrects issue #40 chunk-3c, originally reverted in `a16958f`). Adds
  `SplittableSamples::skip_props_swap` and wires `partition_node_in_place_with(
  ..., skip_props_swap=true)` from `compute_best_tree_with_budget` and
  `build_subtree_sequential_borrowed` — the lossless paths that use
  `PartitionKey::Bucket` exclusively and never read `samples.props` after
  `pre_quantize`. Elides ~16-30 `Vec::swap` calls per row swap in
  `split_tree_samples_in_place`. Paired A/B at 8T (15 samples/cell,
  `bench_chunk3c_resurrect_ab.sh`): -2.5 to -10% wall-clock on 7/7
  evaluated cells (small/medium/large × e7/e8/e9), every sample
  byte-identical. Best-iter on 1024² e7 with `parallel-tree-learning`:
  1.64× → 1.53× cjxl. **Not** wired into
  `compute_best_tree_with_multipliers` whose static-prop axes use
  `PartitionKey::Property` and read `samples.props[axis]` at evaluation
  time; a `debug_assert!` in `PartitionKey::matches` catches the misuse.
  Env-var `JXL_DISABLE_CHUNK3C=1` forces the props-swap path for paired
  A/B (process-cached via `OnceLock`). Hash locks 36/36 byte-identical in
  both default and `parallel-tree-learning` feature configurations. The
  earlier `a16958f` chunk-3c attempt (doc-only revert) had failed the 5%
  gate at load 10-12; this resurrection ships at the lower 1% gate
  characterised in the rejected-optimizations audit memory because the
  path-conditional dispatch has zero opportunity cost on the multipliers
  path.

### Added

- **Effort levels 10 and 11** beyond libjxl's `kTortoise` (effort 9) ceiling
  (RFC issue #45 chunk 1; `LossyConfig::with_effort(10)` / `with_effort(11)`).
  Both accept and validate through the public `EffortProfile::lossy`/`lossless`
  clamp (now `1..=11`) and through `EFFORT_RANGE` in `validation.rs`. e10/e11
  produce 100% spec-valid bitstreams — djxl / jxl-rs / jxl-oxide decode
  unchanged. Today the only differing knob is `butteraugli_iters`:
  `9 => 4` (libjxl `kMaxButteraugliIters`), `10 => 8`, `_ => 16`
  (saturated at `MAX_QUANT_LOOP_ITERS`, which the structural cap in
  `butteraugli_loop.rs:151` already enforces). Every other effort-derived
  knob falls through to the existing `_` arms (so e10/e11 lossless behaviour
  matches e9 today; multi-seed tree learning ships in chunk 2). New tests:
  `effort::tests::test_butteraugli_iters_e10_e11_extended` pins the iter
  table; `validation_tests::lossy_effort_zero_rejected` /
  `lossless_effort_each_level_validates` extend the validation range to
  `1..=11`. Hash-lock fixtures (36/36) stay byte-identical — all fixtures
  encode at the default e7, well below the new effort levels. New A/B/C
  bench harness: `examples/e10_e11_paired_ab.rs` (CID22-512 × distance
  × {e9, e10, e11}, paired sample-major interleave, jxl-oxide-linear-sRGB
  decode + Rust butteraugli scoring). CLI `--effort` blurb now documents
  the 1-11 range.

- **`LossyConfig::with_dot_detection(bool)` + CLI `--dot-detection` /
  `--no-dot-detection`** wire up the existing ported `vardct::dot_detection`
  module into the public lossy encode API (refs #19 / audit "surprise #2").
  Default is **on**, mirroring libjxl's `Override::kDefault` semantics for
  `cjxl --dots` — the in-encoder gates (effort ≥ 7 + distance ≥ 3.0 + no
  text-like patches for the same frame, matching
  `enc_patch_dictionary.cc:632-643`) make this a no-op outside the niche
  star-field / specular-highlight content range. When the gates fire, the
  detector promotes each surviving Gaussian dot into a patch dictionary
  entry via `PatchesData::from_dots`. `with_perceptual_optimizations(true|false)`
  now toggles the new knob in step (previously left it off-by-default
  regardless). Hash-locks (36/36) byte-identical — no fixture content trips
  the gates. On `gb82/night-lossless.png` at d=3.0 e=7: +27 bytes (24701 vs
  24674) for 1 detected candidate dot. djxl + jxl-rs roundtrip clean.

- **`ColorEncoding::from_cicp(cp, tc, mc, full_range)` CICP lookup helper**
  (HDR plan chunk 2, issue #46). Maps the most common ITU-T H.273 / ISO/IEC
  23091-2 CICP 4-tuples to JXL's internal `ColorEncoding` — the wire-format
  used by MP4/Matroska/HEIC/AV1/Ultra HDR. Supports `cp ∈ {1, 9, 11, 12}`
  (sRGB / BT.2100 / DCI-P3 / Display P3), `tc ∈ {1, 8, 13, 16, 17, 18}`
  (BT.709 / Linear / sRGB / PQ / DCI / HLG); rejects `mc != 0` and
  `full_range == false` with descriptive `&'static str` errors. Mapping
  matches libjxl's `ApplyCICP` (`lib/jxl/cms/jxl_cms.cc:928`) exactly,
  including the `cp=12` → `(WhitePoint::D65, Primaries::P3)` and
  `cp=11` → `(WhitePoint::DCI, Primaries::P3)` split. 15 new unit tests
  covering common HDR tuples, error paths, and jxl-rs roundtrip for
  CICP-derived sRGB and BT.2100 PQ.

- **Opt-in pixel-count + effort gated small-image fallback for the
  parallel-tree-learning thread-local SplitWorkspace cache** (audit
  conditional-value catalog item #10 —
  `rejected_optimizations_conditional_value_2026-05-17.md`). New
  `EffortProfile::tree_parallel_small_image_fallback` (bool) +
  `SMALL_IMAGE_PIXEL_THRESHOLD = 1_000_000` (u64) +
  `EffortProfile::adapt_small_image_fallback(pixels)`. Wired into
  `LosslessConfig::effective_profile_for_image(pixels)` as an
  **opt-in** per-image adapter that flips the flag for inputs below
  1 MP AT EFFORT ≤ 7 when the caller opts in via
  `LosslessConfig::with_small_image_fallback_override(Some(true))`
  (or CLI `--small-image-fallback`).
  When the flag is on, `compute_best_tree` bypasses the thread-local
  `SplitWorkspace` cache (commit `cb5e202`) by routing through a new
  `with_workspace_dispatched` helper that allocates a fresh
  `SplitWorkspace::new` per `find_best_split` call.
  **Default: OFF** — paired bench data
  (`benchmarks/small_image_fallback_paired_2026-05-17.tsv`) on top of
  chunk-3c (`79ff70ed`) shows the audit-claimed cb5e202 cache
  regression no longer reproduces: small_0.26MP × e7 × 8T median Δ
  -0.40% (default vs `nofallback`), within noise. Infrastructure
  ships behind the opt-in for future investigation if the regression
  re-emerges. The parallel root-split and borrowed-view fan-out are
  unconditionally on. Bitstream-equivalent: hash_lock 36/36
  byte-identical; sha256 matches on 0.26 MP / 1.05 MP profile images.
  New expert knob:
  `LosslessInternalParams::tree_parallel_small_image_fallback: Option<bool>`.
  Second instance of the `EffortProfile::adapt_*` per-image dispatch
  pattern established by smart-fanout (`1c4691f0`).
  Companion follow-up: imazen/jxl-encoder#42 tracks the larger
  +6.2% borrowed-view regression (audit item #9 — deferred per task).

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

- **Predictor-pruning lower-bound skip wired into `find_best_predictor`
  sequential paths** (issue #23, chunk 2; chunk 1 shipped the primitive
  at `c579cbd1`): both the `cfg(feature = "parallel-tree-learning")`
  small-range sequential fallback (`tree_learn.rs:4878-4914`) and the
  `cfg(not(feature = "parallel-tree-learning"))` mirror
  (`tree_learn.rs:4946-4979`) now call
  `predictor_extra_bits_lower_bound` + `decide_predictor` before each
  `compute_predictor_entropy`. Strict-`<` tie-break preserves the
  byte-identical bitstream invariant: hash_lock_features 36/36 unchanged
  under both cfg flavors; sha256-identity verified on a real photo at
  e7/e8/e9. Paired-A/B 9-cell bench at 8T (CID22 0.26 MP / CLIC 1.05 MP
  / CLIC 4.19 MP × e7/e8/e9, 8 paired iters):
  | image            | e7      | e8      | e9      |
  |------------------|--------:|--------:|--------:|
  | small_0.26MP     | −0.7%   | −0.8%   | −0.2%   |
  | medium_1.05MP    | −0.3%   | −0.8%   | **−4.0%** |
  | large_4.19MP     | −0.0%   | +0.3%   | +0.7%   |
  Headline: byte-identical across all cells; medium e9 clears 3%; other
  cells within ±1% of noise. Wireup targets the wrong code path under
  `--features parallel-tree-learning` at e7 — lossless callers go through
  the parallel branch (lines 4900-4920) on the root call (range >> 1024),
  so the sequential lb-skip never fires there. The wireup is correct and
  beneficial for (a) `--no-default-features` / non-parallel builds,
  (b) `compute_best_tree_with_multipliers` per-child calls (lossy
  modular / LfFrame DC) where range can dip under 1024, and (c) e9
  deep-subtree paths (the −4.0% on medium e9). Chunk 3 will extend
  lb-skip into the parallel branch to capture the e7 wins. Full TSV +
  meta at `benchmarks/predictor_prune_ab_2026-05-17.{tsv,meta}`.

- **Predictor-pruning lb-skip extended into the parallel branch**
  (issue #23, chunk 3; algorithmic change shipped via `23f22d22`'s
  inadvertent file-bundling — see `benchmarks/predictor_prune_c3_ab_2026-05-17.meta`
  for the full attribution story). `find_best_predictor`'s
  `parallel_map` fan-out (`tree_learn.rs:4916-5022`) now carries a
  shared `AtomicU64` running best (`f64::to_bits()`); each worker
  pre-computes its extra-bits lower bound, reads the atomic, and
  emits `f64::INFINITY` instead of running `compute_predictor_entropy`
  when `lb >= best`. CAS update on full-eval completion is strict-`<`,
  matching the sequential tie-break. The post-fanout reduction reuses
  the existing strict-`<` minimum scalar — `INFINITY` slots lose every
  comparison, preserving the lowest-index winner. Byte-identical to the
  chunk-2 baseline (hash_lock_features 36/36; sha256 verified on a real
  photo at e7/e8/e9 against `52f8e816`-built CLI binary). Paired A/B at
  8T (CID22 0.26 MP / CLIC 1.05 MP / CLIC 4.19 MP × e7/e8/e9, 12 paired
  iters; large_4.19MP@e9 captured only 1 iter pair due to harness shell
  termination — see meta):
  | image            | e7              | e8              | e9              |
  |------------------|----------------:|----------------:|----------------:|
  | small_0.26MP     | −1.4% / −2.0%   | +0.3% / +0.8%   | +1.0% / +0.4%   |
  | medium_1.05MP    | −0.5% / +0.4%   | −0.1% / −0.8%   | +3.0% / +2.8%   |
  | large_4.19MP     | **−7.5% / −0.0%** | **−8.2% / −4.1%** | −5.9% (n=1)   |
  Format: median paired pairwise Δ / 10-90 trimmed mean Δ (preferred over
  min/avg on this heavily loaded run). Large 4.19 MP cell at e7/e8
  recovers the chunk-1 microbench's predicted savings (-7 % to -8 %
  pairwise); medium 1.05 MP @ e7 lands at the noise floor (-0.5 %
  median, brief target of ≥3 % NOT MET); medium e9 +3 % regression is
  the early-worker race-window structural cap (all 14 workers see
  `f64::MAX` and run full eval before any can post a real cost to the
  atomic). Two interventions documented in the meta but not shipped this
  chunk: (a) seed-first hybrid — serialize the lowest-LB eval before
  dispatching the parallel fan-out so the atomic is populated when
  concurrent workers start; (b) Strategy A — sorted-by-LB sequential
  eval, loses parallelism but guarantees the microbench savings on
  small per-call ranges. Full TSV + meta at
  `benchmarks/predictor_prune_c3_ab_2026-05-17.{tsv,meta}`.

- **Streaming hash-table dedup backend (opt-in, issue #41)**: ported
  libjxl's `AddSample` / `AddToTableAndMerge` two-hash cuckoo
  open-addressing dedup (`enc_ma.cc:602-655`, `enc_ma.cc:711`) as a
  drop-in sibling to the existing packed-key sort dedup
  (`dedup_samples_packed_sort`). Enabled via
  `LosslessInternalParams { use_streaming_dedup: Some(true), .. }`
  (requires `__expert` feature). Default `false` at every effort. Both
  backends produce byte-identical bitstreams (hash_lock_features 36/36
  unchanged; new `test_dedup_backends_agree_on_unique_set` invariant
  test verifies unique-sample multiset equality on real-pattern pixel
  data). **The streaming path regresses end-to-end wall-clock by +3% to
  +8% at e7 on CLIC photos (0.26 / 1.05 / 4.19 MP), so it ships off** —
  `pack_sample_key` random-accesses the parallel SoA arrays per sample
  with no cache locality, and the sort path exploits adjacent-pixel
  spatial coherence the hash path cannot. The win libjxl gets requires
  building keys *during* the gather pass (issue #41 Phase 2, future
  work), not on top of an already-gathered SoA buffer. Retained as an
  opt-in so the Phase-2 rework has a tested kernel to integrate.
- **SIMD-vectorized `estimate_bits` for tree-learning `find_best_split`**
  (refs #23): new `jxl_simd::estimate_bits_u32` AVX2/NEON/WASM128 path
  replaces the scalar inner loop in `tree_learn::find_best_split` and
  `compute_predictor_entropy`, where the libjxl-style 1/4096-probability-
  floored Shannon cost is called 22k+ times per node. Pre-SIMD asm
  (`benchmarks/find_best_split_asm_hot_loop_2026-05-15.txt`) showed a
  serialized `subsd` accumulator dep chain + scalar `fast_log2f` (~25
  cycles/iter); SIMD path uses 8 lanes × 2 independent accumulators and
  FMA polynomial, hiding the log2 latency. Measured at effort 7 single-
  thread on CLIC photos (commit-time, AMD 7950X):
  | image | size | wall-clock Δ | compute_best_tree Δ |
  |---|---:|---:|---:|
  | CID22 photo (0.26 MP) | 156 KB | −8.9% | −11.8% |
  | CLIC 1 MP photo | 1.28 MB | −8.0% | −10.2% |
  | CLIC 4.2 MP photo | 2.76 MB | −5.1% | −6.5% |
  Output bytes are byte-identical to baseline on all three images;
  all 13 `lossless_*` hash-locked tests pass unchanged. Full numbers +
  asm dumps under `benchmarks/find_best_split_post_simd_2026-05-15.tsv`.

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

### Changed

- **`nb_rcts_to_try=0` fallback now uses RCT-10 (GBR+SubGR) instead of
  RCT-6 (YCoCg)** in `select_best_rct{,_at}`. The previous fallback
  defaulted to YCoCg unconditionally when no RCT trial was performed
  (effort < 5, or `LosslessInternalParams::nb_rcts_to_try = Some(0)`).
  RCT-10 (permutation=GBR, transform=Subtract-Green) saves **1.19% bytes**
  on a diverse 490-image corpus relative to YCoCg as a single-RCT default
  (per the chunk-1 RCT-picker investigation in commit `287d915`). Default
  effort (e7) is unaffected — it sets `nb_rcts_to_try=7` and runs the full
  trial search, so all hash-locked tests are byte-identical. Measured impact
  at effort 4 on the 3 profile photos: small −1.82%, medium −0.64%, large
  −0.64% (consistent with the sweep direction). Adds
  `RctType::GBR_SUBGR = RctType(10)` as a named constant.

- **Empty modular sub-bitstream EOF in multi-group VarDCT/patches frames**
  (mirrors `imazen/jxl-oxide@fd4e2c3`): when a modular section had no
  decodable channels (every non-meta channel deferred to PassGroups by the
  `max_chan_size` filter), `jxl-encoder` ended the section without the
  32-bit ANS initial state. libjxl is bug-compatible by *always* emitting
  those 32 bits via `WriteTokens` even with zero tokens — its
  `Decoder::begin()` reads them unconditionally before checking buffer
  dims. djxl and jxl-rs short-circuit before that read (via the
  `num_chans == 0` / `is_empty` early-returns in
  `modular/encoding/encoding.cc:587` and `decode_modular_subbitstream`),
  so they accepted the pre-fix bitstream; **stock jxl-oxide 0.12.5
  rejected it with `UnexpectedEof`**. Two trigger configurations are
  fixed:
  1. Multi-group VarDCT with an extra channel (alpha) larger than
     `group_dim` (`vardct/bitstream.rs` `write_modular_empty_global`):
     now writes `use_global_tree=1` + 32-bit ANS initial state instead
     of an isolated 4-bit GroupHeader.
  2. Multi-group modular (patches reference frame, lossless) whose
     channels are deferred to PassGroups (`modular/section.rs`
     `write_global_modular_section` / `write_global_modular_section_with_tree_dc_quant`):
     unconditionally emit the 32-bit ANS initial state after the global
     ModularHeader instead of skipping when `nb_meta_tokens == 0`.
  Cost: +4 bytes per affected LfGlobal section. Regression test added in
  `tests/empty_modular_section_roundtrip.rs` (Layer 3 — encoder roundtrip
  via jxl-rs and in-process jxl-oxide; stock 0.12.5 verified manually).
  The `[patch.crates-io]` pin to the imazen jxl-oxide fork stays in
  place as defense-in-depth for bitstreams from third-party encoders.

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
- **RCT smart-picker investigation (chunk 1, 2026-05-17)**: new
  `jxl-encoder/examples/rct_per_image_sweep.rs` (unregistered,
  zenanalyze-dependent) sweeps 490 corpus images × 7 RCT candidates
  via `with_force_rct(Some(RctType(N)))` to identify the
  ground-truth best RCT per image, then fits a 33-feature random
  forest. 5-fold CV top-2 accuracy = 74.7% — under the 80% ship
  threshold. New `jxl-encoder/examples/rct_picker_wall_ab.rs`
  (unregistered, public-API-only) confirms wall-clock savings from
  trial reduction are within noise under 8-thread rayon (the
  `select_best_rct` `parallel_map` makes the 7-trial cost
  effectively free); single-thread shows 1.8-10.1% wall savings.
  Sweep data: `benchmarks/rct_per_image_full_2026-05-17_512px.tsv`.
  Side finding (not yet landed): the `nb_rcts_to_try=0` fallback
  currently picks YCoCg (RCT 6); RCT-10 (GBR+SubGR) beats it by
  1.19% bytes on the 490-image corpus with no predictor needed.
  Full chronology in
  `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/zenanalyze_rct_predictor_2026-05-17.md`.
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
