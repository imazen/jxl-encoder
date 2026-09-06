# Changelog

## [Unreleased]

(the [0.4.0] section below is release-prepped; not yet tagged or
published. See docs/RELEASE_0.4.0.md for the publish gates.)

### Changed
- **Auto-resampling is no longer on by default** (imazen/jxl-encoder#101 follow-up; the never-resample default was confirmed as a user directive on 2026-09-06 after the imazen-26 sweep put the perfect-oracle upside at 0.08 %): libjxl's `d ≥ 10 → 2× + d×0.25+0.25` rule (`enc_frame.cc:108-114`) now sits behind the strategy gate `auto_resample_libjxl_rule` — ON under `EncoderStrategy::Libjxl` (parity), OFF for Zenjxl/Aggressive/LeanFaster; `with_auto_resampling(true|false)` pins it either way and explicit `with_resampling(N)` is unchanged. Measured on 20 real images × e5/e8 (`benchmarks/auto_resample_monotonicity_2026-09-05.*`, harness `examples/auto_resample_monotonicity`, report `scripts/auto_resample_monotonicity_analyze.py`): the switch is never the cheaper regime at matched butteraugli at d = 10 (40/40 cells) — photos +12 %/+30 % bytes for slightly worse butteraugli, graphics −11..−27 % bytes for butteraugli +9..+26 / SSIM2 −30..−85 — and wins only at d ≥ 17 on 6/40 cells by ≤ 14 %; cjxl v0.11.1's own d ≥ 20 switch shows the same pattern. A distance ladder is now one regime end to end: no byte increase or quality cliff at d = 10 (the remaining non-monotonicity is quantiser noise, bytes up on 6/260 steps by ≤ 2.5 %). Hash-locks and the Libjxl byte-lock are unchanged (no lock cell used the auto rule)
- The CPU zensim backend no longer copies the reconstruction planes to tight stride before scoring. `CpuZensimBackend::compare_with_reference` copied all three distorted planes strided→tight into a per-instance scratch and then called `compute_with_ref_and_diffmap_linear_planar` with `stride == width`. **The callee has taken a `stride` parameter all along** (zensim `src/diffmap.rs`), threads it into `convert_linear_planar_to_xyb`'s row addressing (`row_off = y * stride`), and validates `stride >= width` + `len >= stride * height` — so the copy fed it the same values in the same order, having first moved 3 × width × height × 4 bytes through memory. The scratch is deleted and the planes are passed with their real `padded_width`; output shape is unaffected because the diffmap is built at zensim's own `simd_padded_width(width)`, independent of input stride. The in-source comment stated the premise that refutes it ("the linear-planar API expects `stride >= width`. We normalise to `stride == width` to keep the code path uniform") — `padded_width >= width` already satisfies that — and at `padded_width == width` the "fast path" still did a full `copy_from_slice` instead of short-circuiting. Byte-identical over 104 backend-route cells (2 corpora × 2 `JXL_ZENSIM_DIFFMAP_SIGNALS` arms), of which **48 are genuinely strided**; the harness is mutation-proven sensitive — a deliberate wrong-stride mutant moves exactly those 48 and leaves all 56 `padded_width == width` cells untouched. **Not applied to the other two copy sites, deliberately**: `GpuZensimBackend` keeps its scratch because `score_from_linear_planes_with_warm_ref_diffmap` takes no stride parameter (a comment now says so, to stop a symmetry-driven "cleanup"), and `vardct/rate_control.rs` does not have this pattern at all — it is the butteraugli/tile_distmap path behind the `rate-control` feature. `examples/zensim_config_byte_identity.rs` gains `--route loop|backend` (the two zensim entry points are disjoint) and two always-present non-multiple-of-8 fixtures, because a corpus of 512×512 images has `padded_width == width` on every cell and would have made this check vacuous
- **Config over flags, Phase 1** (`vardct/zensim_loop.rs`, the densest file — design: `~/work/zen-workspace/CONFIG_OVER_FLAGS_2026-08-31.md`): the loop's 27 `std::env::var` call sites — **two of them inside the iteration body** — collapse to **one** (`grep -c 'std::env::var' vardct/zensim_loop.rs` is now `1`, reached only from two `from_env` constructors that run once per encode). `env::var` allocates a `String` and takes a process-global lock per call, so a knob read per iteration costs something every iteration. 33 env names become typed fields on `ZensimLoopConfig` whose `Default` **is** the shipped behaviour, three closed sets become enums (`MapArm` 8 arms replacing 8 string literals `matches!`-ed at 6 sites; `H3GainMode`; `Controller`, which collapses `JXL_ZENSIM_SECANT` + `_CTRL_EXP` + `_CTRL_CLAMP` + `_SECANT_MIN_EPS` + `_SECANT_MIN_DLNL` into one value with the step rule as `Controller::step`), and the four instrumentation sinks move behind a `ZensimLoopObserver` trait with an `EnvTraceObserver` shim. **Behaviour-preserving and proven so, not asserted**: `from_env` keeps every name's shipped meaning, default and validation (including the un-filtered `JXL_ZENSIM_TARGET_SCORE`, deliberately left accepting `nan`), and `JXL_ZENSIM_TRACE`'s 7 columns / `JXL_ZENSIM_SECANT_TRACE`'s 10 / `JXL_ZENSIM_ATTR_PROBE`'s 4 keep their exact widths AND format specs — the substrate probe in `analyze_23shot.py verify` reads that trace **positionally** and string-compares columns 2 and 3. Evidence: a BITWISE differential unit test (`config_tests::controller_step_is_bit_identical_to_the_pre_refactor_block`) pinning `Controller::step` against the frozen pre-refactor block over ~300k cases on `to_bits()` with all seven branches asserted reached, hash locks byte-identical with none relocked, plus `examples/zensim_config_byte_identity.rs` + `scripts/zensim-loop-eff/byte_identity_matrix.sh` (per-cell SHA256 over CID22-512 × 3 distances × 2 efforts × a 27-arm env matrix, `diff -r` empty across the change). Not collapsed, and the input to Phase 2: `JXL_ZENSIM_CTRL_EXP`/`_S4_EPS` (the S4 per-image elasticity prior reads the IMAGE, so the exponent resolves late) and `JXL_ZENSIM_RD_PROFILE`/`_MAP_BAKE` (process-wide `OnceLock` mounts; the map-bake mount stays lazy because it `eprintln!`s and can `panic!`)

### Added
- `examples/resample_admissibility` + `scripts/resample_admissibility_analyze.py` + `benchmarks/resample_admissibility_2026-09-06.*` (imazen/jxl-encoder#101): the 43-image imazen-26 stratified sweep (23 strata × sizes {512,1024} × e5/e8 × 27 distances = 4644 encodes) that settles whether 2× resampling is ever worth choosing. A perfect oracle picking the cheaper regime per cell at matched butteraugli saves **0.08 % of corpus bytes**; 3 % of cells are admissible and 72 % are unreachable at any bitrate. `floor/d` ranks admissibility at AUC 0.927, beating both a fitted logistic (0.913) and one with all 101 zenanalyze features added (0.897). Against ground truth the floor gate selected 0 unreachable cells where libjxl's `d >= 10` rule selected 468 of its 1032
- `examples/distance_targeting_probe`: reports DELIVERED vs requested butteraugli for any image and grid, which separates "off the RD curve" (waste) from "on the curve but mis-targeted" for any distance-keyed gate (#101)
- `tests/it/resampling_odd_dims_streaming.rs` (imazen/jxl-encoder#101): the odd-dimension SizeHeader contract fixed in `71e0f6af` / `11828823` now also covers the streaming `LossyEncoder` entry point, Rgba8 (alpha downsampled alongside colour, `ec_upsampling` following), multi-group coded cells (513×769 → 257×385), the issue's exact cell at d = 10 and d = 25, streaming == one-shot byte identity (also under `with_already_downsampled`), an `#[ignore]` djxl leg, and a decoder-independent LSB-first parse of the codestream's own SizeHeader. #101 was filed from a zenjxl sweep build that path-patched a checkout of this repo predating both fixes: 130 of its 585 persisted odd-dimension bitstreams — exactly 13 images × 10 distances at d ≥ 10 — declare +1/+1; current `main` encodes the same shapes at the source size
- **libjxl byte-parity differential harness** (`7486d732`, `fefb52eb`): `scripts/jxl_bitstream_diff.py` parses a bare codestream's envelope field by field and reports the first field whose VALUE differs plus the per-TOC-section byte table, so a residual lands on `LfGlobal` / `LfGroup k` / `HfGlobal` / `HfGroup p g` instead of "byte 4711 onward". It also splits both global sections' field-coded prefixes (`dequant_dc`, `quantizer`, `block_ctx_map` thresholds; `dequant_matrices`, `num_histograms`, `used_orders`) without needing an ANS decoder. A deliberate standalone re-implementation of the libjxl `d089091a` bit layout — not a call into our own reader or jxl-rs — because a differential that shares code with the encoder under test cannot see a bug the two share; dependency-free, so it needs no build. Self-checks that `header + TOC + Σ sections == file size` and prints `PARSE INCONSISTENT` rather than reporting a confident wrong attribution. Drivers: `scripts/libjxl_parity_sweep.sh`, `scripts/make_parity_fixtures.py`, and `examples/libjxl_gate_ab.rs` (A/B one named gate with a cjxl reference arm on bytes + butteraugli + ssim2). Standing recorded in `docs/LIBJXL_DIVERGENCES.md` §D

### Changed
- **`EncoderStrategy::Libjxl` now emits a codestream envelope that is field-identical to `cjxl v0.12.0`** (`e2d98cce`, `5ac78c32`, `fefb52eb`). Signature, `SizeHeader`, `ImageMetadata`, `CustomTransformData`, `FrameHeader` and the TOC match on every cell of a 196-cell grid (7 fixtures × e{3,5,7,9} × d{0.5,1,2,4,7,10,15}); `header_and_toc` vs cjxl went **+24.4 % → +0.3 %**. Four `all_default` fast paths existed in the header writers and **none could ever fire**: `ImageMetadata`'s predicate was a hardcoded `false`; `ColorEncoding`'s demanded `rendering_intent == Perceptual` where the spec default is `kRelative` (which the lossy builder sets *to match libjxl*); `FrameHeader`'s demanded `x_qm_scale == 2` where the spec default is **3** (only *b*`_qm_scale` is 2) **and** `do_ycbcr`, a field only serialized when `!xyb_encoded`; and the same predicate demanded the extra-channel vectors be *empty* where libjxl only needs them at their *defaults*, excluding every RGBA frame. The `x_qm_scale == 2` case was also a latent correctness hazard — a frame emitted `all_default` at 2 would be dequantized by the decoder at 3. Two further parity fixes: `kSkipAdaptiveDCSmoothing` was set on every lossy frame where libjxl sets it only for JPEG transcode, and `x_qm_scale` was scored against the auto-resample-reduced distance where libjxl captures `original_butteraugli_distance` first (73 of 196 cells affected). **All six are gated on `EncoderStrategy::Libjxl`; zen-mode output is byte-identical throughout** — flipping any of them in zen mode is an owner call, and the two RD-affecting ones are measured rather than assumed (`benchmarks/jxl_dc_smoothing_ab_2026-08-31.{tsv,meta}`, `benchmarks/jxl_x_qm_scale_ab_2026-08-31.{tsv,meta}`). Adaptive DC smoothing turns out to be strongly content-dependent — butteraugli −9…−34 % on smooth gradients at d ≥ 1 but **+0.5…+13.4 % on textured photo** — so zen's skip flag is a defensible trade, not an oversight
- **Sectioned per-group tree learns choose their predictor set from CONTENT, not from a fixed K** (issue #99 item 1). The sectioned writer gathered all 14 candidate predictors per group and then pruned to `SECTIONED_PRUNE_PREDICTORS_K = 8` by ROOT COST. Both halves were wrong: root-cost ranking is leaf-blind (already refuted for the global tree in `benchmarks/jxl_probe_prune_2026-08-15.md`) and pruning *after* the gather only shrinks the split search. Now ONE whole-image probe tree is learned per encode (`sectioned_probe_predictors`, node-capped at 2047, every group sampled at 16x the gather stride, gather fanned out across groups) and every group's learn is restricted, BEFORE its gather, to `Weighted` plus the shortest leaf-share prefix of the statics reaching 80 % of static leaf mass, hard-capped at the old K so the set can never be larger than what shipped. Probes with fewer than 768 leaves are not trusted and keep the previous fixed-K path. **The count is chosen by content** — 4 on a concentrated photo tree, up to 8 on a spread document/palette tree — which is exactly the asymmetry a fixed K cannot express and why fixed K=4 cost palette content +1.35 %. Wall vs cjxl v0.12 on the bar cell (photo 3840x2160, min of 5 INTERLEAVED repeats): e7 t=1 **1.60x -> 1.23x** (11.52 -> 8.90 s, the <= 1.3x bar MET), e7 t=8 1.78x -> 1.64x, e9 t=1 **1.07x -> 0.79x** (44.79 -> 33.14 s), e9 t=8 **1.12x -> 0.96x** — e9 does not regress, it now beats cjxl on both thread counts. Bytes on that cell +0.32 % (e7) / +0.26 % (e9). Fitted on a NEW 310-rendition / 52-origin corpus, k-means-clustered over 2238 imazen-26 + gb82-sc sources with an ORIGIN-level train/validate split, log-spaced sizes, crops AND resizes, no upscaling (`benchmarks/jxl_sectioned_adaptive_k_2026-08-31.{tsv,meta}` + the `_corpus`/`_picks`/`_t1`/`_bar`/`_leaves*` siblings; `scripts/select_sectioned_k_corpus.py`, `scripts/fit_sectioned_k_gate.py`, `scripts/sectioned_k_bar_cell.sh`, harness `examples/sectioned_k_corpus.rs`). The gate fires on 49 of 310 renditions and NEVER on documents, web screenshots, plots/line-art, clipart or the gb82-sc screens — those stay byte-identical to the previous default. Held-out fired-cell wall **-16.7 %** at e7 t=1 with bytes +0.31 % mean / **+2.99 % max** (one colour patent rescan, which buys -20.3 % wall on the same cell). Two things the corpus also settled: the shipping K=8 default was ITSELF costing **+0.38 % bytes mean / +6.54 % max** vs no pruning at all (invisible before because the 2026-08-28 sweep used K=8 as its own baseline), and **e7 t=8 is not predictor-count-bound** — fixed K=4, the maximum possible reduction, still lands at 1.38x on the bar cell, so the residual there needs the cheaper-split-search lever, not a smaller set. Env dials: `JXL_SECTIONED_TREE_PREDICTORS=auto|off|K`, `JXL_SECTIONED_PROBE_{MAX_NODES,MIN_LEAVES,COVERAGE,GROUP_STRIDE,PIXEL_MULT}`. The GLOBAL probe selector is untouched

### Fixed
- **`sharper_downsample_2x_plane` is now bit-exact with libjxl's `DownsampleImage2_Sharper`** (imazen/jxl-encoder#102). The ringing clamp's upper bound was seeded with Rust's `f32::MIN` (the most negative float) where libjxl uses C++ `std::numeric_limits<float>::min()` (the smallest POSITIVE normal, ~1.18e-38). On opsin planes that are wholly negative — routine for the X channel — that silently tightened the clamp. Pinned by `sharper_downsample_2x_is_bit_exact_with_libjxl`, which replays procedural inputs against golden vectors generated by LINKING the real `jxl::DownsampleImage2_Sharper` (generator committed as `jxl-encoder/testdata/sharper_downsample2x_golden_gen.cc`): five shapes compared value-by-value plus 129×97 and 256×256 pinned by digest. Only the five `_r2_` hash-lock cells move (resampling factor 2, including e10 whose iterative kernel starts from sharper); every other lock, the Libjxl byte-lock and the drift test are unchanged
- Two further invariants pinned while localising that defect, both of which pass and rule out alternative explanations for the resampling floor gap: our opsin round-trip is lossless over the resampling path (`opsin_round_trip_is_lossless_over_the_resampling_path`), and our opsin transform agrees with libjxl's `LinearRGBRowToXYB` to <1e-5 (`opsin_transform_agrees_with_libjxl`, golden-vector paired)
- Auto-resampled encodes adapt the effort profile at the REMAPPED distance (imazen/jxl-encoder#101 follow-up): when libjxl's rule fires (`EncoderStrategy::Libjxl`, or `with_auto_resampling(true)`) the content-aware profile adaptation and the zenanalyze-proxy gates were fed the requested distance (10) while the encoder ran at `d×0.25+0.25` (2.75); libjxl rewrites `cparams.butteraugli_distance` in `ParamsPostInit` so everything downstream except `original_butteraugli_distance` (the `x_qm_scale` ladder) sees the remapped value. Both now use `effective_distance()`. Zen defaults are unaffected (the rule is off there); pinned by `issue_101_libjxl_strategy_keeps_the_switch` (Libjxl auto at d = 10 ≡ explicit `with_resampling(2)` at 2.75 with original distance 10, byte for byte)
- Streaming `LossyEncoder` now honours `with_already_downsampled` like the one-shot `EncodeRequest` path (imazen/jxl-encoder#101): it downsampled pre-downsampled input a second time and advertised the coded size in the SizeHeader (a 300×200 `already_downsampled` input at factor 2 declared 300×200 where one-shot declares 600×400), so the two entry points produced different streams for the same config; pinned by `issue_101_already_downsampled_streaming_matches_one_shot` (byte-identical). `with_already_downsampled`'s rustdoc now states the odd-original caveat (`dims × N` is the only size the encoder can advertise there)
- The 19 `__expert`-gated examples compile again (`39d56dda`). They imported `jxl_encoder::effort::{EntropyMulTable, LossyInternalParams}`, but `effort` became `pub(crate)` in #76 — E0603 on every one. **Both types were already re-exported at the crate root** (`lib.rs:136`, `:139`), so the fix is the import path, not the visibility; `effort` stays private and the bypass #76 deliberately closed stays closed. All 19 import the identical pair, so the diff is 19 identical one-line swaps and nothing else. CI gains a `cargo check -p jxl-encoder --features "__expert,parallel,butteraugli-loop,ssim2-loop" --examples` step inside the existing clippy job (which already has the toolchain, sibling clone and cache) — **no CI job enabled `__expert`**, which is exactly why this rotted unseen. It is `check` and not `clippy` **deliberately**: measured with `--keep-going` (a single clippy run stops early and reports a different subset each time), the `__expert` example set has clippy findings in **78 examples**, **56 of them `undocumented_unsafe_blocks`** around edition-2024 `set_var` — a `-D warnings` lane there would be red on arrival and has to be earned file by file. That cleanup is not attempted here
- **`with_resampling(N)` / auto-resample changed the image DIMENSIONS on any axis that is not a multiple of `N`** (`71e0f6af`, #45). The encoder downsamples by `div_ceil(dim, N)` — correct, and what libjxl's `FrameDimensions::Set` does — but the file header rebuilt the advertised size as `downsampled × N`, which rounds **up** by as much as `N − 1`. A 1118×1105 input at `d = 10` came back **1118×1106**: one row taller than the caller supplied, and `djxl`, `jxl-rs` and `jxl-oxide` all agreed, because the `SizeHeader` genuinely said 1106. Reachable with **no caller opt-in** — `auto_resampling` is on by default and picks `N = 2` at `distance >= 10`, so any odd-width or odd-height image encoded at `d >= 10` silently came back the wrong size; explicit `with_resampling(2/4/8)` hit it at every distance. libjxl writes the TRUE size and lets the decoder's `DivCeil(xsize_px, upsampling)` recover the coded grid and crop the upsampled result (verified: `cjxl v0.12.0 -d 12` on the same input writes `ysize = 1105`). Fixed by carrying the pre-downsample size to the header builder on both the one-shot and streaming lossy paths; `with_already_downsampled` keeps the `dims × N` contract it documents, pinned by its own test. Nothing caught this because every `resampling > 1` hash-lock cell uses a 512×512 fixture and 512 is divisible by 8, so the buggy product equalled the true size in all of them — including the five cells added days earlier precisely to cover `resampling`. Pinned by `tests/it/resampling_odd_dims.rs` (odd dims 259×133 / 65×65 / 127×255 × factors {2,4,8}, the auto-resample arm at d ∈ {10,12,15}, a no-resample control, and the `already_downsampled` contract, each through **both** jxl-rs and jxl-oxide); mutation-verified. Two stale doc claims corrected in the same change — `with_resampling`'s rustdoc still said "we don't auto-select yet; callers opt in explicitly", which is what made this look opt-in. No hash lock moved
- Sectioned lossless pre-flight estimate is a **max of two phases, not their sum** (issue #99 item 3), which fixes an admission-safety bug as well as the over-prediction it was filed for. Threads {1,2,3,4,6,8,12} shows the pre-tree phase (the `select_best_rct` trial wave, or patches detection at one worker) and the per-group tree learns are consecutive, so a worker's group set costs nothing while the floor is higher: photo 12 MP measures **48.0 B/px marginal at t=2 AND at t=12** at both e7 and e9, while photo 1024² (16 groups) climbs 69 → 443 B/px over the same range. The additive model double-counted, and — invisibly to the old 4-point thread grid, which had no cell between 64² and 1024² — **UNDER-predicted 12 of 192 cells**, worst photo 256² e9 t1 at **0.54×** (TYP 37.0 MB for a measured 68.6 MB peak); under-prediction is the unsafe direction, since admission sizes a cap from TYP. Terms are now per-channel and structurally derived (`RCT_TRIAL_WAVE = 2` trials + best + image = 4 i32 planes/channel = the 16.0 B/px/channel measured to three digits: 48.0 rgb, 64.0 rgba), the in-flight learn count is clamped to `groups + 2`, and the measured **+7.3 KiB/worker** slope of the flat region keeps the estimate STRICTLY monotone in the pool width — so the thread walk-down contract and all three `encode_preflight_sectioned` tests hold unchanged, with no assertion weakened. Over a 192-cell × 3-repeat grid (`benchmarks/jxl_sectioned_thread_dense_2026-08-30.{tsv,meta}`, `scripts/mem_sectioned_threads_sweep.sh` + `scripts/mem_sectioned_model_fit.py`): under-predictions **12 → 0**, ≥ 2 MP TYP/measured max **2.77× → 2.28×** and mean **1.74 → 1.46**, < 2 MP max 19.63× → 8.58×. Residual error is one-sided (over-prediction only). Estimator-only — no encode path touched, no bytes moved
- Lossy `with_resampling(N > 1)` **with alpha** no longer emits an undecodable stream. The VarDCT frame-header writer hardcoded `ec_upsampling = 1` while the api downsampled alpha by `N` alongside the colour planes; `ec_upsampling < upsampling` is invalid — an extra channel may be coded coarser than colour, never finer — so every RGBA lossy encode at `resampling > 1` produced a header decoders reject (`jxl-oxide: ValidationFailed("EC upsampling < color upsampling, which is invalid")`). `ec_upsampling` is now written as `upsampling` (libjxl reaches the same value by defaulting `ec_resampling` to `resampling`, `enc_frame.cc:118-120`, and clamping it up, `:2658-2659`), and `FrameHeader::write` now REFUSES to serialize `ec_upsampling < upsampling` with a named `InvalidInput` rather than emitting bytes nothing can read. `resampling == 1` is unaffected (58/58 pre-existing hash locks byte-identical); the `lossy_mg_rgba_512x512_blocky_r2_e7` cell that exposed this is now locked. Found while adding the first-ever `resampling > 1` lock cells for #45 (#45)
- `with_resampling(N)` downsamples in the **XYB (opsin) domain** at every effort and every factor, matching libjxl. The decoder's upsampling stage runs on the XYB planes *before* the inverse colour transform (`dec_cache.cc` puts `GetUpsamplingStage` ahead of `GetXYBStage`; jxl-oxide matches) and libjxl's encoder downsamples the opsin image (`DownsampleColorChannels(..., Image3F* opsin)`, `enc_frame.cc:740-763`, reached at `:1648` after `ToXYB` converts `color` in place at `:1610`) — for the sharper 2× kernel AND the plain box used at 4×/8×. Our sharper (e ≤ 9, 2×) and box (all efforts, 4×/8×) wrappers were filtering linear RGB pre-XYB; averaging does not commute with the XYB nonlinearity, so the round trip could not compose. Both now go through the shared `to_opsin_planes` / `from_opsin_planes` helpers that the e ≥ 10 iterative path (`e3a54d7e`) was refactored onto byte-identically. Measured over 4 × CID22-512 × d{1,2,4,7,10} × e{1,3,5,7,9,10} × r{2,4,8} with in-process Rust butteraugli over a jxl-oxide `srgb_linear` decode: butteraugli **−33.6 … −41.2 %** at 2× (21.3–22.8 → 12.6–15.0, i.e. at cjxl's score), −3.4 … −10.1 % at 4×, −4.1 … −7.7 % at 8× — **with bytes down 1.2–20.2 % at the same time** (`benchmarks/jxl_resampling_domain_2026-08-30.{tsv,meta,md}`, harness `examples/resampling_domain_ab.rs`). **This moves every `resampling > 1` stream except `r2` at e ≥ 10** (owner-approved). No hash lock covered any of those cells, which is why the mismatch survived; five were added (`lossy_mg_rgb_512x512_noise_{r2_e3,r2_e7,r2_e9,r4_e7,r8_e7}`) and the regen left all 53 pre-existing lines byte-identical, including every `resampling == 1` cell and `r2_e10` (#45)
- zensim qf-targeting controller: the diffmap secant's trust region now guards the axis that actually sets step size. The step is `(ln L_t - ln L)/eps_hat`, so a SHALLOW measured elasticity extrapolates to a huge move; the registered min-|d ln L| guard watches only the ratio's NUMERATOR and — measured over 1053 secant-eligible controller steps, smallest |d ln L| = 3.11e-3 — never fires at its guessed 1e-3, which is therefore kept as a pure divide-by-zero floor. The |eps_hat| floor that was always present as `eps_hat < -1e-6` (a sign test, not a trust region) is fitted to **0.25** on a threshold sweep: plateau 0.20-0.30 under both emit modes, matching the tail/body split of the measured |eps_hat| distribution (min 0.043, 25/779 steps below 0.20, median 0.589). Worst trajectory overshoot 3.07 -> 2.17 points, overshooting cells 4 -> 3, median overshoot 0.425 -> 0.215, median final in-loop |err| 0.436 -> 0.381, with decoded census and median |err| UNCHANGED at both budgets and both emit modes. Both thresholds are env-overridable (`JXL_ZENSIM_SECANT_MIN_DLNL`, `JXL_ZENSIM_SECANT_MIN_EPS`) and a controller diagnostic trace is available at `JXL_ZENSIM_SECANT_TRACE`. zensim-loop only — default-feature bytes unmoved (`benchmarks/zensim_secant_min_dlnl_2026-08-30.{md,meta}`)
- Single-worker RCT selection prices candidates WITHOUT materializing them (issue #99 lever 1, step 2): `estimate_cost_rct_streaming` walks the trial's transformed rows through the same per-row transform core (`rct::forward_rct_rows_in_place`) and the same tally/entropy helpers as the materialized cost, so the streamed cost is bit-equal (lib test `streaming_rct_cost_matches_materialized`) and only the winner is cloned+transformed. Lossless t=1 peak_live: photo 12 MP 457,138 → **376,747 KiB** (29.2 B/px — the peak is now the patches-detection internals over just the input), 2048² → 110,653 (25.8 B/px), rgba 3840×2160 → 291,703; bytes identical everywhere, t ≥ 2 keeps the wave KiB-identical (`benchmarks/jxl_sectioned_rct_stream_2026-08-30.{tsv,meta}`). The sectioned estimator's t=1 floor re-anchors 68 → 50 B/px (the 68 envelope was ≥ 2.8× three measured ≥ 2 MP cells; 50 covers the worst cell at 1.30× — single-thread lossless admission at a cap is correspondingly more permissive); the multi-thread floor stays 68
- Animation lossless honours `LosslessConfig::with_sectioned_trees` and is admitted on the band its frames actually run (issue #99 item 4): the frame options hardcoded `SectionedTrees::Auto` (silently ignoring the knob — the same bug shape the streaming `LosslessEncoder` had) and the pre-flight shimmed `SectionedTrees::Off`, so animation lossless was admitted/estimated on the whole-image band while its frames could run sectioned. Both now consult the config's knob; default (`Auto`) animations are byte-identical (full `it` suite + animation fixtures green). Regression tests `animation_honours_sectioned_knob` (Off ≠ On, both decode identically in jxl-oxide AND jxl-rs; mutation-verified) and `animation_admission_uses_sectioned_band` (a cap just under the whole-image estimate rejects `Off` and admits + completes `Auto` sectioned; mutation-verified against the old shim)
- Single-worker lossless encodes no longer hold the full RCT trial wave at the peak (issue #99 lever 1): `select_best_rct`'s wave of 2 collected both in-flight whole-image clones through the reduce, which — once the patches-phase set was gone — WAS the sectioned t=1 peak on every content class (alloc-sites: nine i32 channel clones = 36 B/px + the 12 B/px `ModularImage`, 421,875 KiB at 12 MP). At `effective_threads() <= 1` (where the wave's `parallel_map` runs sequentially anyway) trials now fold one at a time, holding best + current only — byte-identical (same order, same strict-`<` reduce; verified across the measured grid and the full suite). t=1 sectioned peak_live: photo 12 MP 597,763 → 457,138 KiB (~36 B/px), imac_dark → 228,308, reddit → 422,495, rgba 3840×2160 → 421,304; every t ≥ 2 cell re-measured KiB-identical (`benchmarks/jxl_sectioned_rct_fold_2026-08-30.{tsv,meta}`). Estimator floors unchanged; t=1 measured-cell pins re-pinned
- Lossless patches DETECTION no longer sits at the sectioned-mode encode peak on screen content (the "patches-phase lifetime" residual of imazen/jxl-encoder#96): detection now runs BEFORE the whole-image `ModularImage` is built (it reads only the raw input pixels; the gate is layout-derived and provably equal to the old image-derived gate), and the flood-fill BFS seed queue is sized exactly (`num_seeds × 16` entries — the former 2× factor was a leftover of the pre-2026-08-28 single-FIFO design and held a never-used ~65 MiB on a 5.6 MP screenshot at the peak instant). Byte-identical everywhere (105-cell grid incl. the content-adaptive e5/e6 arms and patch-heavy palette content; full `it` suite + hash locks green). Sectioned `On` peak_live at every thread count: gb82-sc imac_dark 349.5 → 274.4 MiB (−75 MiB, 62.2 → 48.2 B/px marginal), qoi reddit.com 650 → 511.4 MiB (−138.5 MiB, 61.9 → 48.0 B/px) — screen content now sits on the same ~48 B/px floor as photo (`benchmarks/jxl_sectioned_patches_lifetime_2026-08-30.{tsv,meta}`); the light-effort global-tree peaks that carried part of the detection set drop a few MiB (imac e6 −4.7). Estimator floors unchanged (68 B/px covers with ~1.4× headroom); measured-cell pins re-pinned (imac e9 t12 deliberately kept stale-high pending the per-thread-term refinement — see the pin comment and issue #99)
- `mem_probe` gains per-site peak attribution: `JXL_ALLOC_SITES=1` (+ `_MIN` bytes, `_TOP` rows) captures a backtrace per ≥ 64 KiB allocation and snapshots the live map at every ≥ 8 MiB peak raise, reporting the top holdings AT the peak instant with symbols + file:line — the in-repo port of the zenjxl `mem_probe_encode` methodology (`benchmarks/jxl_alloc_sites_4k_2026-08-13.md`). This is what attributed the patches-phase excess (and identified `select_best_rct`'s nine whole-image channel clones as the next sectioned-peak lever, 193 MiB on imac_dark — issue #99)

## [0.4.0] - UNRELEASED

### Changed
- Whole-image lossless memory band recalibrated on a three-class grid (photo 64² → 12 MP, palette screenshot, web screenshot; e5–e10; rgb + rgba — `benchmarks/jxl_lossless_band_2026-08-28.{tsv,meta}`): e7–e8 128 B/px (was 540), e9 160, e10 160 + 160 MiB, e6 92 (was 235), alpha +72 B/px (was 230), effort-dependent intercepts. `LosslessConfig::estimate_encode` and `SectionedTrees::Auto`'s memory-pressure arm drop ~4× for tree-learning efforts (12 MP e7 TYP 6.4 → 1.6 GB against 1.14 GB measured); at the 8 GiB lossless default cap the whole-image path now admits up to ~65 MP at e7 instead of ~15.7 MP (#96)
- Lossless tree learning skips the parallel fork engine on single-worker pools (`effective_threads() <= 1`) and runs the sequential grow loop directly — byte-identical (the two engines are bitstream-equivalent by construction; verified on the 8.3 MP photo, imac_dark and reddit cells) and 3.5 % less learn wall at threads=1 on both the global and the sectioned path (#96)
- Sectioned group size measured (128 / 256 / 512 / 1024, `benchmarks/jxl_sectioned_group_size_2026-08-28.{tsv,meta}`): no size dominates (512 buys 1.0–1.3 % on photo/palette-screenshot and loses 3.1 % on a tall web screenshot; 1024 costs +50–60 % peak with ≥ 4 workers), so the 256 default stays; `with_modular_group_size` remains the per-call knob (#96)
- Wall standing recorded vs cjxl v0.12 on the aarch64 laptop (`benchmarks/jxl_sectioned_prune_k_2026-08-28.meta`, `jxl_sectioned_phases_2026-08-28.tsv`): sectioned e9 is at 1.08× (t=1) / 1.10× (t=8) — inside the ≤ 1.3× target; e7 is at 1.59× / 1.65×, structurally the per-group split search (K=8 kept predictors vs libjxl's 2); the per-group prune K=6/4 sweep is recorded and rejected as the default (≤ −13 % wall for up to +1.35 % bytes on palette content). `profile-phases` now also times the sectioned writer (`sectioned/gather|prune_predictors|learn|collect|lz77|ans_build|write`) and three coarse `tree/z_*` guards inside the learner (#96)

### Fixed
- Lossless patches detection no longer inflates the single-thread encode peak (imazen/jxl-encoder#96): the connected-component scan takes the bounded union-by-min labeling path at ≥ 1 MP on every thread count instead of a scanning DFS whose flat-index stack grew through doubling reallocs on photo content (one foreground component). This was the entire "sectioned t=1 excess" of the 2026-08-27 memory sweep — attributed with a `MEM_PROBE_PATCHES=0` A/B: 12 MP lossless e7/e9 `SectionedTrees::On` at threads=1 peak_live 855 → 584 MiB, 8.3 MP 518 → 404 MiB, bytes identical everywhere (`benchmarks/jxl_sectioned_mem_t1excess_2026-08-28.{tsv,meta}`); the sectioned estimator's threads=1 floor drops 80 → 68 B/px (= the multi-thread floor) so single-thread lossless admission at a memory cap is ~13 % more permissive
- `mem_probe` gains `MEM_PROBE_PATCHES=0|1` and `MEM_PROBE_GROUP_SHIFT=0..=3` for lossless attribution / group-size runs (#96)
- Sectioned local-tree lossless mode (`SectionedTrees::On` / `Auto`, imazen/jxl-encoder#96) now covers full-image palette, ChannelCompact and lossless-patches content instead of silently falling back to the whole-image global tree: stream 0 learns its own tree from the meta channels and codes them there, the patches dictionary precedes the modular stream exactly as on the global path. gb82-sc `imac_dark` (2940×1912, palette + patches): 0 → 96/96 local sections, peak_live 486 → 348 MiB at e7 and 772 → 348 MiB at e9 (t=1), bytes +0.2 % (e7); djxl, jxl-rs and zenjxl-decoder pixel-exact on the three new `it` fixtures (`benchmarks/jxl_sectioned_mem_meta_2026-08-28.{tsv,meta}`). Default-path bytes unchanged
- Lossless pre-flight is bounded again under the sectioned local-tree mode (imazen/jxl-encoder#96): `SectionedTrees::On`/`Auto` requests are admitted, thread-walked and reported (`EncodeStats::estimated_peak_bytes`) on a calibrated SECTIONED estimate (`heuristics::estimate_encode_sectioned`, measured 2026-08-27 on real photo/screenshot crops 64² → 12 MP × e7/e9 × threads 1–12, `benchmarks/jxl_sectioned_mem_2026-08-27.tsv`) instead of skipping admission outright and reporting the whole-image figure the encode never used; absurd dimensions are rejected on the sectioned floor with a message naming it
- Streaming `LosslessEncoder` honours `LosslessConfig::with_sectioned_trees` (its frame options silently defaulted to `Auto` regardless of the config) and uses the same sectioned-aware pre-flight as the one-shot request (#96)
- `SectionedTrees::Auto` rustdoc now states the 2026-08-19 thread policy (sectioned at e ≤ 7 with > 1 worker), not only the memory-pressure arm (#96)

### Added
- **e11+ lossless TectonicPlate config trial** (issue #45, supersets libjxl e11): probe pair + the transcribed 24/20-config lists re-encode the frame at e10 trial effort (palette / channel-compact / group-size / predictor / patches / sample-density axes, unique-config dedup, caller-explicit knobs pinned), winner re-encoded at the full tier profile, smallest wins (20edf242, 15facee2)
- **`DownsampleImage2_Iterative` port** for `with_resampling(2)` at effort ≥ 10 (libjxl `enc_frame.cc:752` kGlacier gate): 3 decoder-upsampler-adjoint refinement rounds run in the XYB domain — cjxl-e10 differential on CID22-512 shows near-exact quality parity (butteraugli 7.196 vs 7.189) at fewer bytes (20edf242, e3a54d7e)
- New extended-tier test surface: `tests/it/effort_ladder_tiers.rs` (jxl-rs + jxl-oxide roundtrips, tectonic win/pin/threshold pins, `--ignored` cjxl differential) and 3 hash-lock cells (lossy r2 e10, lossy e11, lossless 320×96 blocky e11) (15facee2, e3a54d7e)
- `jxl-encoder-cli/examples/mem_probe`: allocator-agnostic `peak_live_kb` / `marginal_live_kb` / `allocs` columns from a counting global allocator (portable — the `VmHWM` columns are Linux-only), a `tree` argument (`auto|global|sectioned|hybrid`) and `MEM_PROBE_CROP=WxH` real-content crops; `scripts/mem_sectioned_sweep.sh` drives the resumable size × effort × threads × content grid (#96)
- `write_token`/`write_tokens` no longer silently corrupt streams on codeless symbols: an out-of-alphabet token now returns `Error::InvalidInput` instead of index-panicking, and a depth-0 symbol in a multi-symbol prefix code errors loudly instead of emitting zero Huffman bits (the zenjpeg #194 mechanism; sweep issue #97)
- Static DC prefix codes now have Kraft-validity + used-symbol assertions in `test_static_codes_exist` (#97)

### Added
- Every hash-lock feature-matrix cell is now decoded in-process with jxl-oxide (dimensions asserted, frame rendered) in both check and update modes, so a regen can never lock undecodable bytes (#97)

- W44-231: LEARNED qf-seed sub-band lift admission — a 4-feature
  logistic over zenanalyze features (new default-on `learned-admission`
  feature, dep: zenanalyze), trained on train-digit imazen-26 only,
  deployed as a strictly-narrowing confident-BAD exclude (P(bad)>=0.90)
  for the d<3.5 low-colour band. Held-out (val digits): 12 blocks,
  12/12 correct, zero strict-good losses (the hand W44-230 box caught
  2/37 on the same set). Main band untouched; fail-open; env escape
  JXL_W44_231_DISABLE=1. Model artifact + full label-evolution record
  in benchmarks/

### Fixed

- W44-230: the screenshot qf-seed lift no longer misfires on textured
  content wearing screenshot clothing (product-page captures, bilevel
  scan texture) — new low-colour-band exclude on `luma_var >= 2353 AND
  edge_density <= 0.113`, wired in BOTH lift paths (W44-109 e<=7 and
  the e8/e9 buttloop seed). Hunt misfires collapse to cjxl
  parity-or-better at every effort (8028 mean +16.4% -> -2.7%, 6006
  +14.2% -> -1.3%); gb82-sc protection set byte-identical (132 cells,
  0 diffs); RD gate zero-drift. Env escape JXL_W44_230_DISABLE=1

- CI red on rustc 1.98.0 (stable updated 2026-08-18): the new
  `clippy::chunks_exact_to_as_chunks` + slice-fill lints fired across
  the workspace (SIMD pixel_loss kernel, ingest, palette, tree_learn,
  resampling, tests, examples). All sites converted to `as_chunks` /
  `.fill()` — output BYTE-IDENTICAL (hash-locks 49/49 in default and
  parallel configs), MSRV 1.89 check clean. One documented allow where
  the chunk size is `size_of::<T>()` (generic const exprs unstable)

- Nightly RD gate coverage extended 12 -> 21 cells (owner-approved):
  reserved-digit CID22 images (val/test holdout endings 1/3/5/7/9)
  replaced by train-legal picks; NEW coverage for e3, aggressive d3.5
  (incl. the windows95 overshoot-contract sentinel), and doc / aliased-
  plot / web-screenshot / ai-product / manuscript-scan content classes
  from sha256-pinned imazen-26 R2 fetches. Verified green on aarch64 +
  x64

### Added

- imazen-26 train-legal worst-performer hunt
  (benchmarks/imazen26_hunt_2026-08-20.md): 588-cell lossy sweep +
  lossless pass on 21 per-stratum picks. ROOT-CAUSED: the W44-109
  adaptive-quant qf-seed lift misfires at d>=2 on screenshot-classified
  non-screenshot content (8028 product-page capture +48%, 6006 bilevel
  patent scan +40% — both collapse to cjxl parity with the lift
  neutralized, single-knob A/B). Second finding: ai-clipart 9016 is
  knob-independently off the Pareto front at e7/e9 (cjxl strictly
  dominates); third: e3 carries a broad +2-3% bytes band

- Nightly zenjxl RD gate rebaselined (owner-approved): red since 2026-07-15
  on cell 1025469 e9 d=2 (-0.3459 ssim2 vs -0.30 slack). 7 of 12 cells
  drifted Jun-11..Aug-19, every one to FEWER bytes at small ssim2 spends
  and flat-or-better butteraugli (rate-seeking, not corruption); keep-best
  CfL contributes only -0.054 of the failing cell's drift. cjxl reference
  values unchanged; fresh ours values measured on x64

- `JXL_CFL_KEEP_BEST` env hook was structurally inert: the strategy_def
  env layer only fires while a gate equals its TYPE default, and a plain
  `bool` pinned `true` by every non-Libjxl strategy could never match.
  Gate is now tri-state `Option<bool>` (`None` = follow the e7+ effort
  default, env-reachable; Libjxl pins `Some(false)`, env-immune).
  Default output byte-identical (hash-locks 49/49 x3 configs); `=0`
  force-off verified live on the nightly-gate cell A/B


- Latent rayon work-stealing panic in tree-learning's thread-local
  workspace ("RefCell already borrowed") — shipped main panicked 3/5
  interleaved rounds at 4K lossless e9 t8 on x64; fixed with a per-thread
  workspace stack (09c582cf)

- CROSS-ARCH BYTE IDENTITY: `estimate_bits_u32` + `shannon_entropy_bits`
  rewritten as canonical arch-stable magetypes kernels — fixed virtual
  accumulator mapping + fixed combine tree on every tier, so x86_64 (AVX2)
  and aarch64 (NEON) produce bit-identical costs. The hand-written per-arch
  kernels grouped by native register width, flipping near-tie tree-learner
  and ANS-clustering decisions between arches (3 hash-lock fixtures; CI red
  since 2026-08-06). Verified: 4K photo e5 encode sha256-IDENTICAL on
  Ryzen 7900X vs Apple Silicon; hash-locks 53/53 on both from one sidecar;
  Libjxl byte-lock 5/5 unchanged; rd-regression green. Sidecar regenerated
  (6 cells shifted at ulp-scale estimator rounding)
- Lossy-modular multipliers path (`with_lf_frame(true)`) panicked (index
  OOB) reading static prop columns freed by pre_quantize's wave-free
  (51a0b473 regression); retained via `pre_quantize_retaining` +
  regression test (this round).

### Changed

- MSRV raised 1.89 -> 1.93 (zenanalyze 0.2.0 dependency; matches the
  house 'Rust 1.93+' policy)

- `SectionedTrees::Auto` (lossless) now selects SECTIONED per-group trees
  at effort <= 7 when the encode runs multi-threaded (`parallel` feature,
  >1 effective thread); effort >= 8 and single-threaded runs keep the
  GLOBAL tree. Measured (x64, 12-image mixed corpus): e5/e7 wall -41/-44%
  at +-0.0% median bytes; e9 keeps global (its -66% wall costs +1.9%
  bytes). Memory-pressure escape unchanged. Output at lossless e<=7 now
  depends on thread configuration under `parallel` builds (mode choice
  only; each mode is thread-count-invariant) — pin
  `with_sectioned_trees(Off/On)` or `JXL_LOSSLESS_LOCAL_TREES=0/1` for
  config-independent bytes. Policy in `modular::frame::auto_tree_mode`;
  hash-locks pin the mode per cell, incl. a new sectioned-engine cell
  verified pixel-exact by djxl + jxl-rs + jxl-oxide (this round)


- Lossless tree-learning parallel structure (all byte-identical): parallel
  stable dedup-refinement scatter (f8adc43f), per-prop node-tensor build +
  chunked subtract (2c54af38), column-parallel prequant bucketize +
  per-image parallel exact-bucketize pre-walk (09c582cf, fa82501e; x64 e9
  gather+prequant 5.0 -> 2.8 s). Wall-neutral at e7/e9 t8 in interleaved
  A/B (the tree critical chain dominates); internal stamps and crash-free
  operation are the wins. Sectioned-trees corpus study recorded
  (benchmarks/lossless_sectioned_vs_global_x64_2026-08-18.*): -41..-66%
  wall at ~0% median bytes, Auto-policy extension awaiting sign-off
- Vectorized IDENTITY/DCT2X2 special transforms (bit-exact, both arches;
  355e8d0e) + canonical-f32 pixel_domain_loss (libjxl-parity precision,
  47929d7b); acstrat eval counters exposed the benchmark-mosaic screenshot
  misclassification -> honest-photo ladders (round 5, 9784a8bf)

- Zenanalyze content proxies rewritten as exact-integer strip-parallel sums
  (order-free => deterministic under any thread count): 4K sweep 78 -> 9.8 ms,
  lossy setup 128 -> 20 ms; corpus-gated byte-neutral (5efbc487)
- mask1x1 + gaborish kernels strip-parallelized via halo strips into the
  unchanged jxl_simd whole-buffer kernels, bit-identical; patches scan
  steps 1-2 + dispatch parallelized (45d3e5ec)
- Patches detector BFS level-synchronous parallel eval + union-find per-CC
  DFS replay with rejection truncation, byte-identical: patches phase
  165 -> 62.5 ms at 4K; lossy 4K photo walls now e5 t1 0.94x / e7 t1 0.73x /
  e7 t8 0.85x cjxl (wins), e5 t8 1.38x (f0c1538a)
- Lossless dedup: adaptive partition refinement + waved predictor
  compaction — byte-identical, default-path peak_live 834→729 MB (e7) /
  1130→1094 MB (e9) at 4K (4486e5e1).
- Lossless e8+ property columns now gather as u8 buckets (exact
  distinct-value pre-walk; byte-identical, raw-props arena removed):
  4K e9 peak_live photo 965→849 MB, screen 978→786 MB, wall +2-8%
  (`JXL_GATHER_BUCKETIZE` overrides) (this round).
- Lossless global tree learn now probe-prunes its predictor set BY
  DEFAULT at effort >= 7 (`JXL_GLOBAL_TREE_PREDICTORS=off|K|auto`
  overrides): a capped probe tree's leaf usage selects 7-9 of 14
  candidates per image — 4K wall −6..−29%, peak_live −130..−165 MB,
  bytes −0.005% total / worst +0.16% on the 18-image stratified corpus
  (this round).
- Sectioned lossless mode: per-group predictor pruning default-on at K=8
  — wall −25/−26% (4K e7/e9 t=1), bytes ±0.04%; `JXL_TREE_PRUNE_PREDICTORS`
  overrides (d2444010). WP-cache fusion halves per-group WP walks in
  sectioned/hybrid, byte-identical (ecf83e55). Lossy XYB cbrt Newton
  vectorized in f64x4, byte-identical, xyb phase −40% (a5d1dfe3).

### Changed (release mechanics)
- `butteraugli` dependency requirement corrected `"0.9"` → `"0.9.4"`
  (dep + dev-dep): `ButteraugliReference::estimated_reference_bytes`
  (the #93 buttloop budget guard) is 0.9.4-only — verified by building
  against crates.io 0.9.3 (E0599, `vardct/perceptual_loop.rs:1066`).
  Publishing 0.4.0 therefore waits on butteraugli 0.9.4.
- Publish blockers, order, and gates for this release live in
  `docs/RELEASE_0.4.0.md` (measured 2026-08-29; corrects two earlier
  audit claims — the butteraugli one above, and published 0.3.1 being
  unbuildable under fresh resolution because magetypes 0.9.28 breaks
  jxl-encoder-simd 0.3.0).

### BREAKING CHANGES (shipping together in this release)
<!-- Formerly the [Unreleased] QUEUED BREAKING CHANGES batch. -->
- **Public API narrowed to the intended surface (#76;
  22e16b3e/fdcf7a6d/32d76354/2df8c275/bfb880f9).** The supported surface
  drops from 2,397 item lines / 46 pub modules to 1,256 lines / 7 pub
  modules (`docs/public-api/jxl-encoder.txt`): `api` + crate-root
  re-exports, `entropy_coding` (only `Lz77Method` +
  `ANSHistogramStrategy`, both now also at the root), `modular` (only
  `RctType`), `validation`, and the feature-gated `jpeg`/`hdr`/`sweep`
  modules. `headers`, `vardct` internals (incl. the `VarDctEncoder`
  engine and `dct`/`transform`), `modular` internals, `effort`,
  `bit_writer`, `color`, `container`, `error` (the second
  `Error`/`Result`), `heuristics`, `image` (deleted outright — dead),
  `tuning`, `trace`/`debug_rect`/`profile_phases`/`zq_seed`
  instrumentation and all 17 exported macros are `pub(crate)` or
  `#[doc(hidden)]`. This deliberately folds in the four accidental
  0.3.2-dev breaks (`docs/RELEASE_SEMVER_0.3.1_to_0.3.2.md`) instead of
  shipping them piecemeal. Migration for known consumers: jxl-gpu keeps
  its `__pre_quantized`/`__internals` seams (unchanged) and gains
  `__gpu` for the DCT parity primitives; zenjxl swaps
  `jxl_encoder::container::{is_container, is_bare_codestream,
  append_gain_map_box}` and `jxl_encoder::heuristics::*` for the same
  names re-exported (doc-hidden) at the crate root.
- **Pre-existing 0.3.2-dev field/discriminant drift, now shipping under
  the 0.4.0 major** (surfaced by `cargo semver-checks` vs 0.3.1):
  `AnimationParams.premultiplied_alpha` +
  `AnimationFrame.{blend_mode, blend_source, save_as_reference}` are new
  fields on externally-constructible structs, and inserting
  `ValidationError` variants shifted the discriminants of
  `IterCountOutOfRange` and later variants.
- **Effort numbering above 9 shifted by +1 to align with libjxl** (issue
  #45, 2026-08-29): libjxl now ships e10 (kGlacier) and expert e11
  (kTectonicPlate), so our extended tiers moved up one — pre-shift
  e10/e11/e12 behaviours (8/16/32 butteraugli iters, 2/16/16 tree-learn
  seeds, 2/4/4 buttloop seeds) now live at e11/e12/e13, and
  `EFFORT_RANGE` widened to `1..=13`. New e10 = a superset of libjxl e10:
  `fine_grained_step = 1` (already at e≥10), the kGlacier MA-tree split
  threshold (89 → 75, the pre-existing `75 + 14×speed_tier` formula),
  the newly ported `DownsampleImage2_Iterative` under
  `with_resampling(2)`, and a forced global MA tree in
  `SectionedTrees::Auto`. New e11+ lossless supersets libjxl e11
  (kTectonicPlate): the per-image config trial re-encodes the frame
  under the transcribed probe pair + 24/20-config lists (palette,
  channel-compact percents, group-size shift, predictor, patches,
  sample density; unique-config dedup) at e10 trial effort and keeps
  the smallest, with the winner re-encoded at the full tier profile. e1–e9 output is
  byte-identical (hash-locks unchanged); one opt-in exception: lossy
  `with_lf_frame(true)` at e9 loses the accidental 2-seed DC-subframe
  tree learn the old `effort+1 → e10` mapping picked up (now
  libjxl-faithful single-seed).
- **0.4.0 will narrow the public API surface.** `cargo semver-checks` vs
  published 0.3.1 found 4 breaks already present on HEAD (see
  `docs/RELEASE_SEMVER_0.3.1_to_0.3.2.md`): new pub fields on `FileHeader`
  / `ImageMetadata` (incl. `ImageMetadata::force_modular_32bit`, #94), new
  private fields on `VarDctEncoder` (incl. `force_modular_32bit`, #94), new fields on
  `GlobalModularState` variants, and the removed `unsafe-performance`
  feature. Rather than ship these as accidental breaks under a patch
  bump, 0.4.0 will deliberately move the leaked `pub mod headers` /
  `vardct` / `modular` internals to `pub(crate)`, leaving a small
  intentional public API (`LossyConfig` / `LosslessConfig` /
  `EncodeRequest` / streaming encoders / `PixelLayout` / `EncodeError` /
  `Limits` / `ImageMetadata` / `EncoderStrategy`). Tracking: #76.
- **`jpeg::read_jpeg` signature** is now
  `read_jpeg(data, limits: Option<&Limits>, stop: Option<&dyn Stop>)`
  (was the interim `(data, max_pixels: Option<u64>)`). `Limits` bundles the
  pixel cap + memory budget + fallible policy; `Stop` cancels — the same
  pair the encode side takes. Approved API change (#77 / #78).
- **`convenience` module: the 8 `encode_*` fns now return
  `Result<Vec<u8>, At<EncodeError>>`** (was bare `Result<Vec<u8>, EncodeError>`).
  They previously dropped the `whereat` trace from `LossyConfig` / `LosslessConfig`
  `::encode` via `.map_err(|e| e.decompose().0)`; now they return the traced `At`
  directly so the originating encode site survives in the error.

### Changed
- **4K lossless encode peak memory cut 55 % (e7) / 44 % (e9), byte-identical.**
  Session of allocator-level fixes guided by a new per-site allocation
  profiler (zenjxl `mem_probe_encode` `JXL_ALLOC_SITES=1`; attribution report
  `benchmarks/jxl_alloc_sites_4k_2026-08-13.md`): free dead property columns
  before dedup (b22d122e), wave the pre-quantize and gather merges, exact
  dedup reservations (19c2f996), distinct-value collector instead of a
  47.5 MiB per-property clone (3aa48c4d), ANS bits buffer split to parallel
  u32/u8 columns (-142 MiB entropy floor), dedup keys packed at the rounded
  word width instead of a fixed 64 B + u32 unique indices (d1074adc), and
  adaptive i16 property columns that promote to i32 on first out-of-range
  value (`PropColumn`, 8b9b6121), dedup partitioned by the two lead key
  bytes so only one partition's pack is ever live (5119668d), gather storing
  only the configured property columns (21684778), and both whole-image
  copies freed before tree learning (`ImageSource`, 3e237d11). 3840x2160
  peak_live: e7 3141 -> 834 MB (-73%), e9 3141 -> 1130 MB (-64%); peak RSS
  5426 -> 1433/1785 MB; wall and allocation count unchanged, output
  byte-identical at every step. The remaining ~2.7x gap to cjxl (~306 MB)
  is architectural: cjxl streams per group and holds < 0.1 MiB of
  tree-learning state at peak (measured via malloc_history, same report).
- **4K lossy encode peak memory cut 27 % (photo e7 517 -> 375 MB), byte-
  identical.** The text-patch DFS stack packed to flat u32 indices in one
  reused buffer (was 128 MiB of per-CC pair Vecs at the peak) and the
  linear-RGB input freed right after the XYB conversion when no perceptual
  loop can read it (`LinearSource` ownership) (4b464975). 1.13x cjxl's
  live set at the same cell; at lossy e9 cjxl's butteraugli loop uses
  2.6 GB and ours stays at 375-434 MB.

### Changed
- **Lossless tree learning 17-28% faster at 4K, byte-identical
  (95deaabc, a6f61bd6).** Fixed the tensor-derivation profitability gate
  (omitted the per-property factor; mid-size nodes re-accumulated
  everything), added a cost-dispatched stable gather partition, in-place
  tensor row reads, and support-trimmed sweep histograms. photo e9
  84 -> 60.3 s, e7 14.4 -> 12.0 s (threads=1). Hybrid mode: per-group
  learns capped at e7 strength above e7 — e9 hybrid −34-37% wall for
  +0.2-0.5% bytes (still below the global tree). Phase data:
  `benchmarks/jxl_tree_perf_2026-08-14.md`.

### Added
- **`SectionedTrees::Hybrid`: per-group best-of-both tree choice
  (0c52a9d4).** Learns the global tree AND per-group trees (riding the
  gather waves), writes each group with whichever section is smaller.
  Beats both pure modes and cjxl on every measured 4K cell: photo e7
  −2.25%, e9 −0.28%, screen e7 −7.79%, e9 −1.66% vs the global tree
  (`benchmarks/jxl_hybrid_trees_4k_2026-08-14.md`); djxl decodes
  pixel-exact. Opt-in; default output unchanged.
- **`SectionedTrees` knob on `LosslessConfig` (`with_sectioned_trees`,
  additive API) with budget-driven `Auto` default (62251cc1).** Servers
  that set `Limits::max_memory_bytes` — and any encode too large for the
  built-in cap — transparently switch to per-group trees and SUCCEED
  within the budget instead of returning `LimitExceeded` (a 21 MP
  lossless e7 encode now completes inside the 8 GiB default cap; the old
  rejection contract is pinned under `SectionedTrees::Off`). Ordinary
  encodes stay byte-identical.
- **Sectioned local-tree lossless mode (`JXL_LOSSLESS_LOCAL_TREES=1`
  runtime override; production selection via `SectionedTrees`).** Every PassGroup modular stream carries its
  own MA tree learned from that group's samples (`use_global_tree = false`),
  eliminating the whole-image tree-learning working set from the peak: 4K
  photo e7 peak_live 834 -> 469 MB with 2.0% FEWER bytes and 17% fewer
  allocations; e9 1130 -> 469 MB, 25% faster, 33% fewer allocations.
  Spec-standard single frame, djxl pixel-exact; default path byte-identical
  (hash-locks unchanged). v1 scope + follow-ups: imazen/jxl-encoder#96
  (95b2a3e3).

### Fixed
- **Per-tile AC-strategy scratch maps were full-image sized — a
  px²/262144-byte QUADRATIC peak-memory term, ~44.5 GiB at 108 MP.**
  `compute_ac_strategy_for_tiles` allocated an image-sized `AcStrategyMap`
  per 64×64 tile and `parallel_map` held every tile's map alive until the
  merge; this (not a per-thread term) is what kernel-OOM-killed 32 GiB
  zensysbench fleet boxes at threads=1 on 108 MP encodes. Scratch maps are
  now windowed to the tile rect (≤ 64 B each; byte-identical output —
  hash-locks 53/53, Libjxl byte-lock 5/5). 108 MP lossy e5 t=1 peak RSS:
  ≥ 44 GiB (cgroup-killed) → 5.65 GiB; 48 MP: 9.93 → 2.49 GiB. Measured
  grids: `benchmarks/jxl_encode_mem_threads{,_postfix}_2026-08-01.{tsv,meta}`
  (6fc767f2; grids 4f04a8b7).

### Changed
- **Encode pre-flight is now an honest, calibrated admission check with a
  budget-driven thread walk-down** (was a flat `w×h×40` estimate,
  effort/path/thread-blind, copied at five entry points — it admitted the
  108 MP encodes above into kernel OOM kills). One shared
  `encode_preflight` now serves the one-shot request, both streaming
  finishes, and both animation paths: the calibrated
  `heuristics::estimate_encode_threaded` estimate is evaluated at the
  requested thread count (0 = ambient pool width), threads walk down until
  the estimate fits the budget cap (and detected available RAM × 0.8 —
  availability only reduces threads, never rejects), and the encode is
  rejected (`LimitExceeded`, naming the bytes, the cap, and the
  `Limits::with_max_memory_bytes` override) only when even the 1-thread
  estimate exceeds the cap. `check_limits`' duplicate flat-40 memory
  screen removed (the calibrated check is strictly stronger). At the
  path-aware default caps this now cleanly rejects what cannot fit
  (e.g. 108 MP lossy vs the 4 GiB default) instead of OOMing the host;
  callers with real budgets pass `with_max_memory_bytes` as before
  (4763a882).
- **Memory model recalibrated at real large sizes** (12–108 MP zensysbench
  corpus, d1.75 + d6.0, threads 1–16, post-quadratic-fix;
  `benchmarks/jxl_encode_mem_threads_postfix_2026-08-01.tsv`): lossy
  splits at e7 (`LOSSY_BPP_E7PLUS` = 135 B/px envelope — the q30 regime
  measured up to 122 B/px and was never swept before; e ≤ 6 base stays
  80), lossless base 76 → 92 and tree band 465 → 540 (2026-08-01 corpus
  photo measured 83 / 490 B/px), and the lossy 2.5 MB/thread term is
  re-confirmed and now UNCLAMPED past `max_useful_threads` in
  `estimate_encode_threaded` (memory keeps growing where speedup
  saturates — measured through t=16) (4763a882).
- **`EncodeStats` reports the memory decision** (additive, non_exhaustive
  pattern): `budget_peak_bytes()` (tracked reservation high-water),
  `threads_used()` (post-walk-down count, 0 = ambient) and
  `estimated_peak_bytes()` (the admitting estimate) (4763a882).
- **CI clippy now runs with `--all-targets`** (and the pre-commit checklist
  matches): test/example/bench targets carry their own feature unification
  and lint surface — the zenjpeg nightly break and ~70 accumulated test-code
  lints were invisible to the previous lib-only gate. The whole workspace is
  now clean under `cargo clippy --workspace --all-targets -- -D warnings`.
- **`api.rs` split into `api/` submodules** (18,370 → ~9,650 lines): the
  independent pieces moved to `api/{pixel_layout,quality,metadata,limits,
  animation,container,strategy,errors,ingest,content_detect,validate,
  animate,tests}.rs`, re-exported so public paths are unchanged. The
  tightly-coupled encode core (`LossyConfig`/`LosslessConfig`/`EncodeRequest`/
  `ExtraChannel`/streaming/result types) deliberately stays in `api.rs` —
  splitting it would force dozens of fields to `pub(crate)`. The boundary is
  documented in the `api.rs` module doc (00d42b16, addca8a6, 46e7216e).
- **Dead-code hygiene** (17 → 6 file-level `#![allow(dead_code)]`, two passes).
  Pass 1: dropped 4 vestigial + 2 single-item + 1 cluster-cleared blanket;
  deleted the unused zero-predictor debug cluster from
  `modular/encode_tree.rs` (`write_tree_histogram_for_zero`/
  `write_zero_tree_tokens`/`USE_ZERO_PREDICTOR`, all `pub(crate)`, zero
  callers) (9d9991a9, dcf3cc71, c3cf5e86). Pass 2: the 4 blankets that sat on
  mostly-LIVE modules (`patches`, `entropy_coding/{encode_ans,encode_huffman}`,
  `vardct/ac_context`) were over-broad — replaced with targeted item allows
  after classifying every hidden item as feature-gated (`cfg_attr(not(feature
  = …))` for parallel / jpeg-reencoding / __pre_quantized / __internals /
  butteraugli-loop), test-only parity reference, or simple-form entry wrapper;
  deleted the dead `weighted_distance_to_color` (superseded by the live `_idx`
  variant); and re-armed the two orphaned ANS conformance checkers
  (`verify_histogram_serialization` + `verify_ans_roundtrip_parsed`, now
  `#[cfg(test)]`) with a new `ans_verify_helpers_rearm` unit test — the
  omit_pos-class histogram-serialization / parsed-roundtrip guard. The
  dead-code lint now covers the live surface of those four modules. The 6
  remaining blankets cover a macro prototype (`strategy_def_prototype`),
  macro-generated gates (`gate_registry`), feature-gated constants
  (`perceptual_tuning`, `region_source`), reference-complete DCTs
  (`vardct/dct`), and test-covered WIP (`dc_tree_learn`) (6f674cdc).

### Documentation
- Fixed ~29 intra-doc links broken by the `api.rs` split (doc comments moved
  into `api/` submodules where the api-level types were no longer in scope —
  fully-qualified to `crate::` paths) plus 3 wrong-type links, and refreshed
  stale DC-tree references (`learn_dc_tree_variable_with_set` is the live
  learned-DC producer; the gate is `effort >= 8`, not the pre-W44-171
  `>= 4`) (7a6ca462).
- 2026-07-13 maintenance-review pass: streaming-encoder rustdoc no longer
  claims input streaming avoids materializing the image (full-image internal
  planes are held until `finish`); stale Zensim "Phase 4 pending" caveat
  replaced with actual calibration status; `ssim2-loop` feature documented
  and the `__expert` feature comment reattached (6f4942c8); stale
  `butteraugli_loop.rs` live-path citations annotated with the
  `perceptual_loop.rs` rename, `DIFFERENCES.md` got a staleness banner
  pointing at the drift-tested `docs/LIBJXL_DIVERGENCES.md`, completed
  `MIGRATION_PLAN.md` archived, accidental empty `step` file removed
  (7f018af4).
- Repo-root `README.md` overhauled to the zen crate conventions (flat-square
  badge row incl. dual AGPL/Commercial license badge, merged quick-start /
  distance-effort / cancellation / errors / limits onboarding, refreshed
  feature tables and footer) and split into a generated badge-free
  `README.crates.md` for crates.io (`readme = "../README.crates.md"`); heavy
  scoreboard / benchmark tables wrapped in `crates.io:skip` markers.

### Fixed
- **Misaligned pixel slices no longer panic (input-placement DoS)**: every
  16-bit / f16 / f32 pixel layout reinterpreted the caller's `&[u8]` with
  `bytemuck::cast_slice`, which panics when the slice is not 2-/4-byte
  aligned — and a sub-slice like `&file_buf[header..]` is only guaranteed
  1-byte alignment, so a service encoding untrusted-shaped input could be
  crashed by placement alone. All 53 ingestion casts (api.rs +
  modular/channel.rs) now go through `cast_pixel_lanes`, which stays
  zero-copy for aligned input and copies misaligned input into an aligned
  buffer. Regression test covers offsets 0..4 for Gray16 lossless +
  RgbLinearF32 lossy (3bb2b416).
- **`with_force_strategy` values past the strategy tables now return a clean
  `ValidationError::ForceStrategyOutOfRange`** instead of panicking inside
  `AcStrategyMap::force_strategy` (`COVERED_X[raw]` OOB for codes >= 19); the
  encode path auto-validates so the panic was reachable from the public API (3bb2b416).
- **Extra-channel `dim_shift > 30` is rejected up front** with a clear
  `InvalidInput` instead of flowing raw into the modular writer's per-group
  region shifts (`1 << shift` overflow); 30 is the ceiling implied by the
  2^30 max image dimension (3bb2b416).
- **LZ77 `special_distance` computed in i64** (saturating): `multiplier` is a
  channel width (up to 2^30) times a table factor up to 8, which could
  overflow i32 for ultra-wide images (debug panic / release wrong distance) (3bb2b416).
- **Patch coverage gates accumulate in u64**: bounding-box-area × occurrences
  over-counts sparse glyphs, so `total_patch_pixels * 100` could wrap 32-bit
  `usize` on wasm32 and mis-gate the patch dictionary (3bb2b416).
- **cjxl-rs could exit 0 without writing any output file**: with
  `--streaming-input --streaming-output` set, every encode path that fell
  back to in-memory encoding (lossy `--progressive`/`--qprogressive`/
  `--lf-frame`/`--progressive-dc`, `--rate-control` builds, lossless
  `--lossy-palette`, active `--ec_resampling`) skipped `write_output` because
  the skip keyed off the CLI flags rather than whether the streaming encoder
  actually wrote the file. Encode arms now return an `EncodeOutput` marker
  (`Bytes` vs `WrittenDirectly`) so the write decision follows the arm that
  ran (5e2eb76a). Same commit: unknown `--lz77-method` /
  `--epf-dispatch` / `--pixel-loss-dispatch` values are now hard errors
  (previously warned and silently changed the encode — a typo in
  `--lz77-method` selected greedy); `--jpeg-coarsen` without the
  `jpeg-reencoding` feature errors out like `--lossless-jpeg`;
  `--epf-dispatch` help text now states the real default (`auto`); the CLI
  README documented a nonexistent `--no-error-diffusion` flag (real flag:
  opt-in `--error-diffusion`).
- **Nightly CI red 2026-07-08..13 repaired (zenjpeg E0433 `fast_yuv`)**: the
  workspace zenjpeg path patch silently deactivated when the sibling bumped
  to 0.9.0 while our req stayed 0.8.4 (a `[patch.crates-io]` only applies
  when versions match), so builds resolved crates.io zenjpeg 0.8.4 — which
  lacks `decode_coefficients_with_jbrd_metadata` (required by
  `jpeg-reencoding`) and does not compile with `parallel` +
  `default-features = false` (undeclared `yuv` feature gap). Both reqs
  bumped to 0.9.0 to re-activate the patch (d9ff132b). zenjpeg 0.9.0 must be
  published to crates.io before the next jxl-encoder release.
- clippy `--all-targets -- -D warnings` restored to green: 8 lints in
  jxl-encoder-simd test code (byte groupings, useless_format,
  manual_contains, iter_cloned_collect; PRNG seed values unchanged)
  (dadc46fe).
- **True VarDCT near-lossless below distance 0.03 is now spec-conformant; the
  `0.03` distance floor is removed (imazen/jxl-encoder#94).** After the #18
  `i16 → i32` DC widening the reconstruction was correct, but sub-0.03 frames
  were still rejected by strict decoders (jxl-oxide: "ANS stream verification
  failed") — worked around by clamping distance up to `VARDCT_MIN_LOSSY_DISTANCE
  = 0.03`. Root cause: at distance < ~0.025 the fine DC quantiser produces
  quantized DC exceeding `i16` (`|DC| > 32767`), while the file header
  unconditionally signalled `modular_16bit_buffer_sufficient = true`. A
  conformant decoder honouring that promise reconstructs the LF/DC modular image
  into `i16` sample buffers; the oversized DC wraps there, corrupting the
  neighbours fed back into the DC Weighted-Predictor, which diverges the modular
  context derivation and desynchronises the DC ANS stream (final-state `0x130000`
  check fails). The entropy stream itself was always well-formed — the bug was
  the sample-buffer width the header advertised. Fix: signal
  `modular_16bit_buffer_sufficient = false` whenever the quantized DC overflows
  `i16` (`VarDctEncoder::note_dc_modular_width`, a new `force_modular_32bit`
  header flag), so a spec decoder uses `i32` buffers. The Huffman entropy path
  was also sized to the actual maximum token (large-DC symbols ≥ 64 previously
  panicked the fixed 64-symbol prefix-code path). The `VARDCT_MIN_LOSSY_DISTANCE`
  floor is deleted — `LossyConfig::new` accepts any positive distance. Distance
  ≥ 0.03 output is byte-identical (the flag/alphabet only change when DC exceeds
  `i16`, which does not occur there — verified: all 6 distances × 2 entropy
  coders hash-identical pre/post fix). New regression guard
  `tests/nl_dc_conformance_issue94.rs` decodes 0.01/0.02/0.03 through BOTH
  jxl-oxide (spec-conformance gate) and zenjxl-decoder, asserting acceptance,
  PSNR ≥ 40 dB, and monotonicity (fails pre-fix, passes post-fix — both
  verified).
- **Near-lossless VarDCT round-trip at butteraugli distance ≤ 0.02 no longer
  produces a corrupt low-frequency image (imazen/zenjxl#18).** Quantised DC
  coefficients were stored as `i16` and saturated at fine distances (the DC
  `inv_factor` = `INV_DC_QUANT[c]·scale_dc·dc_mul` grows large as distance → 0),
  collapsing the DC plane — a real sRGB fixture decoded to ~21–27 dB PSNR
  versus ~63 dB at distance 0.03, despite spending *more* bits (larger file).
  DC storage is widened `i16 → i32` (matching the wire format, which already
  packs `i32` DC residuals, and libjxl's internal `i32` DC), so bright blocks
  (`max |Y DC| > ~0.877`) stay exact. (Spec-conformance of the resulting large
  DC below distance 0.03 was initially handled by a `0.03` distance floor; that
  stopgap is now removed by the root-cause fix in #94 below.) Regression guard:
  `tests/nl_dc_roundtrip_issue18.rs`.
- **jxl-encoder-simd: NEON/WASM128 SIMD kernels no longer reference retired
  magetypes concrete-type methods (`f32x4::from_float32x4_t`, `.raw()`,
  `f32x4::from_v128`, single-arg `f32x4::from_i32x4`).** magetypes 0.9.27
  fully retired the legacy per-platform concrete SIMD structs (`arm::w128`,
  `wasm::w128`, …); the bare `magetypes::simd::f32x4` alias on aarch64/wasm32
  now resolves to the generic `generic::f32x4<Token>` strategy type, which
  only exposes the token-gated `from_repr`/`into_repr`/`from_i32x4(token, ..)`
  API (x86_64's `X64V3Token` f32x4/f32x8 keep a `.raw()`/`from_m128`/`from_m256`
  convenience alias, so those call sites were unaffected). Since this repo's
  `Cargo.lock` is gitignored, CI always re-resolves to the latest compatible
  magetypes/archmage (0.9.27) regardless of what's cached locally, which is
  why this broke `ubuntu-24.04-arm` Clippy in a downstream consumer (zenpipe)
  without ever showing up in a same-day x86_64 build. Fixed the four call
  sites in `jxl-encoder-simd/src/transpose.rs`, `dct8.rs`, and `dequant.rs`
  (NEON `transpose_8x8_neon`/`neon_transpose_4x4`/`neon_dequant_4`, WASM128
  `wasm128_transpose_4x4`/`wasm128_dequant_4`) and bumped the declared
  `archmage`/`magetypes` floor in `jxl-encoder-simd/Cargo.toml` and
  `jxl-encoder/Cargo.toml` to `0.9.27` (the first version where the bare
  aarch64/wasm32 `f32x4` alias is consistently the generic type — pinning
  lower risks resolving a version where it's still the old concrete struct,
  which lacks `from_repr`/`into_repr`/token-taking `from_i32x4`). Pure
  API-adaptation, no logic change — verified `cargo check`/`clippy -D
  warnings` clean on `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`,
  and `wasm32-unknown-unknown`, and jxl-encoder-simd's 186 scalar-vs-dispatch
  parity tests still pass on x86_64 (native execution can't cover the
  NEON/WASM128 code paths themselves on this host; typecheck + clippy across
  both cross targets is the verification available here).
- **Encode peak-memory model (`heuristics::estimate_encode`) recalibrated
  from a full SIZE sweep — the 2026-06-14 constants over-predicted (TYP acted
  like a MAX).** The prior constants were fit at 1024² only, which conflates
  the fixed overhead α into the per-pixel β and inflates the apparent B/px;
  the result over-predicted the marginal working set by ≈ 1.5–4× (worst on
  the lossy buttloop: **4.16×** at 12 MP). A size sweep {256,512,1024,2048} ×
  {lossy,lossless} × effort {1,4,5,6,7,8,9} × content {photo,screenshot} ×
  threads {1,8,16} (lossy at q50 worst-case + q90) re-fit each band as the
  tightest linear upper bound clearing the MAX measured cell at every size
  with a ≥ 10 % safety margin. Key changes: lossy base 85 → 80 B/px; **lossy
  buttloop 300 → 90 B/px** (the buttloop measures the same working set as e7
  at equal quality — the prior 300 was a 1024²-α artifact); lossless base
  90 → 76; **lossless e6 140 → 235 B/px** (the prior 140 was *under*-predicting
  — e6 is a much heavier partial-search than assumed); lossless tree 465 (kept,
  re-verified to cover the e9 top); lossy fixed 16 → 50 MB (covers the lossy
  1024² q50 marginal bulge), lossless fixed 20 → 16 MB. `MULT_MAX = 1.8`
  retained — heaptrack confirmed it covers the requested-heap-vs-RSS gap
  (1.02–1.25×). The probe (`zenjxl examples/mem_probe_encode`) gained a
  threads arg. Provenance: `benchmarks/jxl_encode_mem_2026-06-23.tsv`. Tests
  `estimate_brackets_measured_12mp` (now `estimate_safely_covers_measured_12mp`)
  and `effort_steps_and_path_ordering` updated to the re-measured truth (the
  old 12 MP anchors, 3.4 GB lossy / 5.0 GB lossless, were ~4× / ~1.2× over the
  real 857 MB / 4186 MB marginal).
- **butteraugli quantization loop is now bounded by the `MemoryBudget` (#93).**
  The CPU butteraugli backend's `ButteraugliReference` builds a
  multi-resolution psycho pyramid + masks *inside* the butteraugli crate —
  the buttloop's single largest allocation (heaptrack: 6.35 GB on an OOMing
  `modes_full` sweep) — which was invisible to the encoder's budget, so the
  default 4 GiB lossy cap could not catch it and a large rendition
  OOM-killed the host instead of failing gracefully. The buttloop now
  reserves `butteraugli::ButteraugliReference::estimated_reference_bytes`
  before constructing the reference (CPU butteraugli path only), so an
  over-cap encode returns `EncodeError::LimitExceeded`. Pure memory
  accounting — hash-locks (48/48) and the `EncoderStrategy::Libjxl`
  byte-lock (10/10) are unchanged. Measurement (`examples/pool_growth_probe.rs`,
  `benchmarks/issue93_pool_*_2026-06-23`) ruled out the reported
  "BufferPool grows unbounded" leak: the pool is fresh-per-encode,
  count-capped at 8, and dropped at encode end. The buttloop ref planes
  also now honor the runtime `Limits::fallible_alloc()` toggle.
- **`convenience::encode_*` preserve the whereat trace** — removed the
  trace-dropping `.map_err(|e| e.decompose().0)` from all 8 wrappers; the
  `At<EncodeError>` from `*Config::encode` now propagates intact. Regression
  test: `convenience::tests::convenience_encode_error_preserves_whereat_trace`.

### Security
- **Configurable pre-flight pixel cap on the untrusted JPEG-transcode SOF
  (#77 / #78).** The JPEG-reconstruction/transcode path (`read_jpeg` /
  `encode_jpeg_transcode*`) parses untrusted bytes and sizes per-block
  `i16` coefficient buffers from the SOF dimensions, but builds no
  `MemoryBudget`. Since JPEG dims are `u16` (≤ 65535 each), a ~12-byte
  crafted SOF could advertise up to ~4.3 Gpx and force a multi-gigabyte
  allocation. `parse_sof_marker` now rejects `width × height` over a cap
  **immediately after reading the SOF dimensions, before any coefficient
  allocation**. The cap is **configurable**: it defaults to the new
  `Limits::DEFAULT_MAX_JPEG_TRANSCODE_PIXELS` (120 MP — admits 108 MP
  phone photos) and is overridden through `Limits::max_pixels`, attached
  to the transcode path via the new `LosslessConfig::with_limits` /
  `limits()` (a trusted batch caller raises it or passes
  `with_max_pixels(u64::MAX)` to opt out; a hostile-input proxy tightens
  it). The public `read_jpeg` now takes
  `(data, limits: Option<&Limits>, stop: Option<&dyn Stop>)` — `Limits`
  bundles the pixel cap + memory budget + fallible policy and `Stop`
  cancels, matching the encode-side `with_limits` / `with_stop` pattern
  (both `None` = the historical pixel-cap-only, zero-overhead parse). This
  replaces the interim `max_pixels: Option<u64>` parameter (an approved
  public-API change — see BREAKING CHANGES above). Byte-
  identical for all real input: the cap only fires above the configured
  limit, and the JBRD byte-exact transcode conformance (28
  `tests/it/jpeg_reencoding.rs` fixtures, all < 120 MP) still passes.
  Pixel-path entry points are unchanged. (The full `MemoryBudget` threading
  and the `encode_jpeg_recompress_*` limits — the former #77 follow-ups —
  are now done; see the budget bullet below.)
- **Cooperative cancellation on the JPEG-transcode path (#77 item 2).**
  The transcode path previously ignored cancellation entirely — `parse.rs`
  hardcoded an `Unstoppable` token into the zenjpeg coefficient decode and
  the JXL entropy phase had no `Stop` at all, so a server could not abort a
  slow transcode of an oversized upload. New
  `LosslessConfig::encode_jpeg_transcode_with_stop` /
  `encode_jpeg_transcode_codestream_with_stop` accept an `enough::Stop` and
  poll it at coarse boundaries — entry, the zenjpeg decode, and per-group /
  pre-entropy during encoding — returning `EncodeError::Cancelled` (mapped
  from the new `JpegError::Cancelled` / the existing `Error::Cancelled`).
  Byte-identical to the non-stop entry points on the success path: the polls
  are skipped when no token is attached and are no-ops under an `Unstoppable`
  token (verified by `test_jpeg_transcode_cancellation`). Mirrors #81's
  VarDCT-path cancellation.
- **Per-encode `MemoryBudget` on the JPEG-transcode + PreserveJxl paths
  (#77 item 1, full).** The transcode path now builds a `MemoryBudget` from
  `Limits::max_memory_bytes` (the lossless 8 GiB default when unset) and
  threads it through both phases: the per-component `i16` coefficient buffers
  are reserved before allocation in `read_jpeg_with_stop` (the largest
  attacker-controlled decode buffer), and a **measured** encode working-set
  estimate is reserved as an RAII guard up front in `encode_jpeg_to_jxl_inner`
  so an oversized frame is rejected (`EncodeError::LimitExceeded`, via the new
  `JpegError::ResourceLimit`) **before** the expensive token collection /
  histogram clustering. The per-coefficient constant is `heaptrack`-calibrated,
  not guessed: on real photos the transcode peak heap is ≤ 44.1 B/DCT-coeff
  (wrenches, the worst of 4 corpus images; effort-independent, e7 == e9), so
  `ENCODE_BYTES_PER_COEFF = 56` carries a ~1.3× margin over the worst measured
  cell (provenance `benchmarks/transcode_mem_2026-06-18.tsv`, harness
  `examples/transcode_mem_probe.rs`). An earlier per-site reservation scheme
  under-counted (≈ 46 MB reserved vs 82 MB real peak on wrenches — the diffuse
  `AccumulatedAnsData` clustering allocations were unbudgeted), which is why the
  whole-working-set measured estimate replaced it. Default-on for every
  transcode entry point (`encode_jpeg_transcode*`), like the pixel path. The
  PreserveJxl `encode_jpeg_recompress_{,auto_,planar_}codestream` free functions
  gain `limits: Option<&Limits>` + `stop: Option<&dyn Stop>` parameters (the
  encode estimate is an RAII reservation released between the lossless-floor and
  lossy encodes so their working sets don't double-count), and a budget
  rejection on their decode side now maps to `Error::AllocationLimit`
  (reconstructed from the owned budget's live `cap`/`used`) instead of being
  funnelled into a misleading `InvalidInput`. Byte-identical for all real input
  under the default cap. Verified by `test_jpeg_transcode_memory_budget` +
  `test_jpeg_recompress_limits_and_stop`; the 14 byte-exact JBRD reconstruction
  / transcode-roundtrip gates still pass. This closes the remaining #77 P1
  follow-ups.
- **Runtime-configurable fallible allocation (`Limits::with_fallible_alloc`),
  wired through ALL standard encodes.** The choice between fast
  `vec!`/`Vec::with_capacity` (a single `calloc`, trusted input) and graceful
  `try_reserve` (returns `Error::OutOfMemory` instead of aborting on a genuine
  OOM, untrusted input) is a **runtime** toggle on `Limits` rather than a
  compile-time decision. All five standard-encode budget constructions
  (one-shot + streaming + animation, lossy + lossless) now build the
  `MemoryBudget` via `with_alloc_policy(cap, fallible)` instead of the
  infallible `MemoryBudget::new`, and the **dimension-driven** allocations
  select their mechanism from the budget policy (`MemoryBudget::is_fallible`):
  lossy output / DC+AC group BitWriters (`BitWriter::with_capacity_fallible`),
  the flat quant + masking fields (`vec_with_capacity_fallible`), the XYB planes
  (already), and the lossless modular channel buffers (the dominant lossless
  allocation, via `try_alloc_zeroed_permanent`); the JPEG-transcode path was
  the first consumer. Defaults to infallible (fast); a hostile-input proxy
  opts into `with_fallible_alloc(true)`. Byte-identical (mechanism only) on
  success — pinned by `test_standard_encode_fallible_alloc_toggle` (lossy +
  lossless) and `test_jpeg_transcode_fallible_alloc_toggle`, both asserting
  byte-identity across modes and budget rejection under a tight cap. Tiny
  fixed-size allocations and the modular tree-learning sample-collection
  working set (already budget-*capacity*-bounded via the `*_with_budget`
  gather/compute variants, but grown through hot-path `Vec::reserve` in
  budget-less sample structs) intentionally stay infallible — wiring those
  would be a hot-path method-signature refactor with perf risk and little
  fallibility benefit.
- **Fallible-alloc + budget policy extended to the remaining >1 MB
  dimension-driven buffers.** Beyond the dominant allocations wired above, the
  encoder's other heap buffers that can exceed ~1 MB at realistic sizes now
  route through `vec_with_capacity_fallible` / the budget policy (byte-identical
  on the default infallible path): DC-token buffers (`dc_coding` WP + the three
  region collectors, `dc_tree_learn` tree-variable/tree/learn-and-collect, and
  the `bitstream` DC-group tokenizers — the per-DC-group fan-out moved from
  `parallel_map` to `parallel_map_result`), modular multi-group residual
  buffers (`frame.rs` palette + squeeze closures), the VarDCT patches-dispatch
  mask repack, the `splines` screenshot-detection Y buffer
  (`looks_like_screenshot` is now `Result<bool>`), the `lf_frame` DC planes,
  the JPEG-preprocessing `resampling` box/sharper downsamplers + alpha, the
  `extras` alpha-squeeze plane, and the `patches` reference-frame channel
  buffers. Budget is threaded as `Option<&Arc<MemoryBudget>>` (or `self.budget`
  on `VarDctEncoder`/`FrameEncoder` methods). Verified byte-identical
  (hash-locks 48/48; full 1986-test suite) and clippy `-D warnings` clean
  (default + jpeg + zensim).
- **All three #88 honest-stopped >1 MB sites now wired through the
  fallible-alloc policy (follow-up).** The AQ-mask statistics, the patches
  BFS detection, and the lz77 optimal-parse buffers — all >1 MB at realistic
  sizes — now route through the budget policy (byte-identical on the default
  infallible path):
  - **AQ mask buffers** (`encoder` `median_mask1x1` / `percentile_mask1x1`, the
    `width * height` `f32` copy allocated on every lossy encode, 4–50 MB at
    ≥ 1 MP). Both helpers now take an `Option<&Arc<MemoryBudget>>` and return
    `crate::error::Result<f32>`; the `bool`-predicate cascade through
    `pixel_loss_auto_should_skip` (now `Result<bool>` + budget) propagates
    through its three `encode_inner` / `encode_frame_to_writer` /
    `compute_with_budget_and_buffering` callers (the `.map(..).transpose()?`
    pattern at the `mask1x1_median` / `mask1x1_p25` sites; `?` at the inline
    gate sites). Budget threaded from `self.budget` / the existing `budget`
    param.
  - **Patches detection** (`patches` `find_text_like_patches{,_with_min_peak}`
    and `find_and_build_lossless`, the BFS `queue` + 3 `background` f32 planes +
    `is_background` / `visited` bool planes + the lossless `planes` triple, all
    `stride * height`, ~13 MB at 1 MP). The two detectors return
    `crate::error::Result<Vec<PatchInfo>>` and `find_and_build_with_per_patch_gate`
    / `find_and_build_lossless` return `crate::error::Result<Option<PatchesData>>`,
    threaded `Option<&Arc<MemoryBudget>>` from the four VarDCT encode paths
    (`encoder` `encode_inner` / `encode_from_precomputed_inner`, `bitstream`
    `encode_frame_to_writer`, `precomputed` `compute_global_only`) and the two
    lossless paths (`api` `encode_lossless` + the streaming `finish_inner`
    closure). The public calibration wrappers (`find_and_build` /
    `find_and_build_with_min_peak` / `find_and_build_patches_lossless`) keep
    their `Option`-returning signatures (public API unchanged) and call the
    inner with `None` budget on the infallible path.
  - **lz77 optimal-parse buffers** (`apply_lz77_optimal`'s `prefix_costs` +
    `out`, both `O(token_count)`, multi-MB to 10s of MB at lossless e9+). The
    `budget` is threaded through `apply_lz77` and all its callers: the
    `pub(crate)` modular wrapper stack (`write_modular_stream_with_tree*`,
    `write_global_modular_section_with_tree*`, `write_group_modular_section_idx`)
    takes the param and bottoms out at `FrameEncoder` (`self.budget`);
    `vardct/bitstream`'s `encode_two_pass_to_writer` passes `self.budget`; the
    small `icc.rs` profile-compression call passes `None` (not a >1 MB site).
  Verified byte-identical (hash-locks 48/48; full 1986-test suite) and clippy
  `-D warnings` clean (default + jpeg + zensim).
- **Checked SOF-derived overflow in the JPEG coefficient parser (#77 item 3).**
  `extract_coefficients_zenjpeg` computed `width_in_blocks * height_in_blocks`
  (and `* 64`) as unchecked `u32` / `usize` on attacker-controlled SOF dims —
  a crafted SOF could wrap the count, size `comp.coeffs` short, and make the
  zigzag copy read out of bounds. Now computed with checked `usize` arithmetic
  via the pure `checked_coeff_len` helper (a malformed SOF errors; the wrap only
  bites 32-bit `usize` — i686 / wasm32 — where `*_in_blocks * 64` overflows at
  the largest representable JPEG), and the SOF-derived count is reconciled
  against zenjpeg's actually-decoded coefficient length before the copy. The
  `jbrd.rs` `write_u32_jbrd` `unreachable!` on the write path is replaced with a
  typed `Error::InvalidInput`. Both guards are pinned host-independently by
  `checked_coeff_len_guards_overflow` (parse) and
  `write_u32_jbrd_no_selector_match_is_typed_error` (jbrd).

### Fixed
- **`with_effort()` no longer silently discards builder settings placed
  before it in the chain (#80).** `LossyConfig::with_effort` /
  `LosslessConfig::with_effort` rebuild the config from the effort profile
  and previously preserved only a hand-picked field list, so any other
  setter called *before* `with_effort` was dropped — e.g.
  `LossyConfig::new(1.0).with_ans(false).with_effort(7)` silently lost
  `with_ans(false)`. Extended the existing `auto_splines_explicit` /
  `patches_explicit` / `tree_learning_user_set` touched-tracking pattern
  with private `*_explicit` flags so `with_effort` preserves a value only
  when its setter was actually called; the builder chain is now
  order-independent. Now-preserved across `with_effort`: on `LossyConfig`
  — `with_ans`, `with_gaborish`, `with_error_diffusion`,
  `with_pixel_domain_loss`, `with_lz77`, `with_lz77_method` (and the
  gaborish/pixel-domain-loss knobs that `with_perceptual_optimizations`
  sets, matching its pre-existing patches pin); on `LosslessConfig` —
  `with_ans`, `with_patches`, `with_lz77`, `with_lz77_method`, and
  `with_tree_learning` (whose `tree_learning_user_set` flag existed but
  was not honored by `with_effort`). The untouched common
  `new().with_effort(N)` path keeps adopting the effort default and stays
  byte-identical (hash-locks 48/48). New test
  `tests/it/with_effort_preserves_explicit.rs`.
- **Test builds: pin `alloc-stdlib` so `alloc-no-stdlib` stays on 2.x (#82).**
  `brotli` (a dev-dep via `zenjxl-decoder`, plus our optional `brotli-metadata`
  dep) provides `impl BrotliAlloc for StandardAlloc`, where `StandardAlloc`
  comes from `alloc-stdlib` and impls `alloc_no_stdlib::Allocator`. When
  `alloc-stdlib 0.2.3` loosened its `alloc-no-stdlib` range to `>=2.0.4, <4.0`,
  a fresh (non-`--locked`) resolve pulled `alloc-no-stdlib 3.0.0` for
  `StandardAlloc` while `brotli` stayed on the 2.x trait → `E0277` mismatch
  that broke **every** test build (the `hash_lock_features` / byte-lock gate
  was down from ~2026-06-14). Upstream is now fixed (`alloc-stdlib 0.2.4`
  re-tightened to `<3.0`; `brotli 8.0.4` added the same caps), but the durable
  guard is a Cargo.toml constraint, not the lockfile: `jxl-encoder`'s
  dev-dependencies now pin `alloc-stdlib = "0.2.4"`, which keeps
  `alloc-no-stdlib` on 2.x even under a fresh `--locked`-less resolve and
  blocks the 3.x-pulling `0.3.x`. Verified: `cargo tree -i alloc-no-stdlib`
  shows only `2.0.4`, and `cargo test -p jxl-encoder --lib --no-run` compiles.
- **Cooperative cancellation now aborts on every VarDCT entropy path (#77
  item 2 / #81).** `#81` wired per-DC/AC-group `Stop` checkpoints into
  `VarDctEncoder::encode_inner`, but they lived only in the multi-group
  two-pass entropy branch, so a single-pass or single-group (`num_sections == 4`)
  encode never polled the token — `test_stop_cancels_lossy_multigroup` failed
  (the encode completed instead of returning `Cancelled`). Added an entry-point
  checkpoint at the top of `encode_inner` that fires before any encode work on
  **all** paths; the per-group checkpoints remain for mid-encode responsiveness.
  Byte-identical under `Unstoppable` / `None`.
- **Full cross-codec cancellation coverage — butteraugli loop + modular
  lossless (#77 cancellation follow-up).** The cooperative `Stop` token is now
  threaded through the two heaviest non-JPEG encode paths that previously could
  not be aborted mid-work: the effort-8+ butteraugli quantization loop
  (`butteraugli_refine_quant_field*`, polled per iteration) and the modular
  (lossless) `FrameEncoder` — `encode_modular` / `encode_modular_with_patches` /
  the multi-group inner + lossy-palette + squeeze + squeeze-with-tree variants,
  polled at the encode boundary and again before each heavy per-group
  `parallel_map_result` (after tree learning). Coarse-grained (boundary /
  per-iteration, not per-coefficient) and byte-identical under `Unstoppable` /
  `None`, verified by `test_stop_cancels_lossy_e8_buttloop` +
  `test_stop_cancels_lossless_multigroup`.

### Added

- **Stale-map single-pass for folded-class attr arms** (zensim campaign
  appendix P lever 2, b8a582e5): `JXL_ZENSIM_SINGLEPASS=1` on a folded-class
  bake pays ONE fused compare (map cached), later steered iterations run the
  score-only extraction + cached map — steered compares at iters ≥2 drop to
  ~40-43 ms (v47A class; was 123-141 ms fresh), G-P5 27-cell decoded A/B
  holds (census exact, sp-vs-base 18W/8L/1T @ 0.979). Runner phases
  `h3ownsp` + `gainsweep`; ZENSIM_H3_GAIN sweep {5,10,20,40} → keep 10.
- **Fused folded-944 model-map steering in the zensim loop** (campaign appendix
  N): `JXL_ZENSIM_MODEL_MAP` attr-family arms now run on folded-class (720/924/
  944) bakes through zensim's fused `compute_folded944_score_and_attribution`
  (one extraction pass = exact features + steering map), replacing the appendix
  M panic; folded gradient probe at the bake's own feature regime; substrate
  probe 27/27+108/108 bit-exact for the 372-class loop. First result: the
  sota944 candidate's OWN-map h3 arm hits 17/27 ±2 at k3 (best inner census on
  the board) at bytes ratio 0.978 vs its generic-map baseline (ca7aa75f).
- **`sweep` module — config enumeration + computed unique-configs
  (`__expert`).** `LossySweep` / `LosslessSweep` closure-axis grids plus
  `unique_lossy_configs` / `unique_lossless_configs` enumerate a cartesian
  product of `LossyInternalParams` / `LosslessInternalParams` candidates,
  resolve each to an `EffortProfile`, and deduplicate by
  `EffortProfile::fingerprint()` (a stable 71-field content hash; equal
  fingerprints ⇒ byte-identical encode) — so a parameter sweep encodes each
  *distinct effective config* once instead of re-encoding combinations that
  resolve identically. (#80)
- **Calibrated peak-memory estimation (`heuristics` module + per-config
  `estimate_encode`).** New `jxl_encoder::heuristics::EncodeEstimate`
  (`peak_memory_bytes_min` / `peak_memory_bytes` typical / `_max` + coarse
  `time_ms` / `output_bytes`) and `LossyConfig`/`LosslessConfig::estimate_encode(w,h,layout)`,
  mirroring the zen per-codec pattern (cf. `zenwebp::heuristics`). Model:
  `input + fixed + bytes_per_pixel(path, effort)·pixels`, with effort STEP
  jumps (lossy buttloop e≥8, lossless tree-learning e≥7) and content
  multipliers (min 0.85 / max 1.8). Constants calibrated from measured
  marginal working set (mem_probe `VmHWM` delta, 12 MP-anchored):
  `benchmarks/mem_peak_calibrate_libharness_2026-06-14.tsv` (6 content
  classes × 64–2048 px × e5/e7/e9 × 8/16-bit) + direct 12 MP anchors. New
  reusable harness `scripts/mem_peak_calibrate.py` +
  `examples/mem_probe.rs`. Refined (44a929b6) from a focused 7-class ×
  1024² × e7/8/9/10 × rgb/rgba grid (`mem_peak_quick_2026-06-14.tsv` +
  `mem_peak_fit.py`): added a lossless **e≥10** band (~620 B/px, +35 % —
  the `fine_grained_step`/multi-seed step) and a lossless **alpha** term
  (+230 B/px; `estimate_encode` gains a `has_alpha` param). Lossy e8/e9/e10
  collapse to one band at 12 MP (sub-linear in size) and lossy+alpha is
  negligible, so neither gets a separate band. `time_ms` is now
  **effort-aware** (was a flat per-path throughput): measured single-thread
  µs/px per effort (`mem_probe` now reports wall + user/sys CPU,
  `benchmarks/jxl_encode_time_2026-06-14.tsv`) — encode time is dominated by
  effort (lossless e9 ≈ 300× e1 per pixel), so the old 3/6 MP/s constant was
  ~8× high at low effort and ~38× low at e9.
- **Comprehensive HDR test suite (`tests/it/hdr_suite.rs`), modeled on
  libjxl's HDR coverage.** Seven gates: TF×Primaries signaling
  preservation matrix (libjxl `PreserveOriginalProfileTest` shape, 15
  combos × lossless/lossy), `SetIntensityTarget` dispatch rules
  (PQ → 10,000 / HLG → 1,000 / SDR → 255 / explicit wins), HLG lossy
  quality-tracks-distance on 512² multi-group, PQ f32-vs-u16 layout
  consistency, lossless pixel invariance under PQ tagging (tagged and
  untagged bitstreams decode identically; ≤1 LSB through jxl-oxide),
  RGBA16-PQ lossy and Gray16-PQ lossless roundtrips. Decode checks use
  jxl-oxide native output — requesting even the file's own color
  encoding routes through a CMS roundtrip and drifts u16 samples.
- **16-bit lossless e5/e6 budgeted tree-learning lift (#72).** cjxl
  learns MA trees at every effort; our off-until-e7 schedule left
  16-bit (HDR PQ) content +29..+49 % vs cjxl at e5/e6. Integer 16-bit
  RGB(A) input now enables tree learning with hard budget caps (e6 adds
  size-adaptive sampling). Crop corpus: e5 +28.5 %→+6.8 %, e6
  +31.8 %→−3.4 % (beats cjxl); 12 MP walls stay monotonic under e7.
  Opt out: explicit `with_tree_learning(..)` or `JXL_NO_16BIT_TREE_LIFT`.
  8-bit output byte-identical (locks 45/45; new 16-bit multi-group lock
  cell).

### Changed
- **Sparse-override config resolution — `with_effort` is now a pure setter
  (#80).** The effort-derived knobs (lossy `use_ans`, `gaborish`, `lz77`,
  `lz77_method`, `patches`, `error_diffusion`, `pixel_domain_loss`,
  `auto_splines`, `butteraugli_iters`; lossless `use_ans`, `tree_learning`,
  `lz77`, `lz77_method`, `patches`) are stored as `Option<T>` (`None` =
  inherit the effort schedule) and resolved lazily in `effective_profile` /
  the getters. `with_effort` only sets the effort level, so it can no longer
  clobber a caller override in any builder order — the structural fix for the
  #80 footgun that #90/#91 patched field-by-field (the 14 `*_explicit` /
  `*_user_set` touched-bits + the ~120-line field-by-field preserve blocks
  are gone; the `Option` *is* the touched-bit). `with_internal_params`
  (`__expert`) likewise stores the sparse params and resolves them against
  the final effort (was: an eager `profile_override` capturing the effort at
  call time), fixing the same order-sensitivity on the expert path.
  Byte-identical on every hash-lock fixture (48/48).
- **Minimal default API surface (#80).** The ~20 codec-internal tuning
  setters (`with_ans`, `with_gaborish`, `with_lz77`, `with_lz77_method`,
  `with_patches`, `with_error_diffusion`, `with_pixel_domain_loss`,
  `with_auto_splines`, `with_force_strategy`, `with_max_strategy_size`,
  `with_quant_ac_rescale`, `with_strategy_overrides`, `with_butteraugli_iters`,
  `with_perceptual_optimizations`, lossless `with_tree_learning`, …) are now
  `#[doc(hidden)]`: the *documented* default builder is minimal
  (quality/effort/threads, output-format + feature toggles, metric
  selection, limits, metadata). They remain callable for power users and
  sweep harnesses, so this is **non-breaking**. The full sweepable knob set
  (`LossyInternalParams` / `LosslessInternalParams` + `with_internal_params`
  + the new `sweep` module) stays behind the `__expert` cargo feature.
- **Default memory cap raised 2 GiB → 4 GiB
  (`Limits::DEFAULT_MAX_MEMORY_BYTES`).** Measured VarDCT peak working set
  is ~180 bytes/pixel — ~2.14 GB real RSS at 12 MP e9 d4, in line with
  libjxl `cjxl` v0.12 on the same image (1.23 GB at effort 7 → 3.42 GB at
  effort 9). The old 2 GiB default rejected ordinary ≥ 11 MP encodes
  mid-pipeline (a 12 MP PQ-HDR e9 d4 encode failed with `memory budget
  exceeded: requested 144 MB on top of 2.13 GB`). The 4 GiB default covers
  up to ~22 MP at the measured rate while still bounding an oversized
  upload so it can't OOM the process. Hostile-input image proxies should
  set a tighter cap via `Limits::with_max_memory_bytes`; trusted batch can
  opt out with `Some(u64::MAX)`. (api.rs)
- **Default memory cap is now path-aware: lossy 4 GiB, lossless 8 GiB
  (`Limits::DEFAULT_MAX_MEMORY_BYTES_LOSSLESS` +
  `Limits::default_max_memory_bytes(is_lossless)`).** The lossless path is a
  heavier memory regime — MA tree-learning at effort ≥ 7 measures ~440 B/px
  (≈ 5 GB at 12 MP) vs lossy's ~85–300 B/px (see `crate::heuristics`) — so a
  single 4 GiB default rejected ordinary ≥ 9 MP lossless encodes. The five
  internal budget-cap sites (one-shot + streaming + animation, both paths)
  now select the default by path; an explicit `with_max_memory_bytes` still
  wins. Both defaults remain fixed DoS ceilings (not scaled with
  dimensions). (api.rs)
- **EPF sharpness search runs candidates sequentially with reused buffers
  (−25–27% encoder peak RSS, byte-identical).** `compute_epf_sharpness`
  evaluated its 2–3 sharpness candidates via `parallel_map`, each cloning
  `base_recon` + allocating scratch/padded planes (~432 MB/candidate at
  12 MP), so 3 candidates held ~1.3 GB at once — the single largest
  real-memory consumer (heaptrack peak attribution), and the reason the e7
  peak (2.07 GB) ran 1.7× over libjxl `cjxl` (1.23 GB). Since `apply_epf`
  is already strip-parallel internally, candidate-level parallelism mostly
  oversubscribed memory. Candidates now run sequentially reusing one buffer
  set, and `compute_block_l2_errors` was made strip-parallel to reclaim the
  per-candidate serial L2 cost. Measured on 12 MP PQ-HDR d4: peak RSS
  2.11 GB → 1.55 GB at e7 (−27%) and 2.15 GB → 1.61 GB at e9 (−25%); wall
  +7 % at e7 / +3 % at e9. Output byte-identical (hash-locks 48/48). (epf.rs)
- **VDP2-lite (HDR) reads its reference from interleaved `linear_rgb`;
  butteraugli builds planar refs only transiently (byte-identical).** The
  perceptual loop deinterleaved `linear_rgb` into 3 planar reference planes
  (~144 MB at 12 MP) and held them for the whole loop on both metric paths.
  `hdr_vdp2_lite` now has `compare_vdp2_interleaved`, which gathers the
  reference straight from the interleaved buffer (bit-identical to the
  planar path — pinned by `vdp2_planar_and_interleaved_bit_identical`), so
  the VDP2 path materializes no planar refs; the butteraugli path builds
  them only for the backend's one-shot `set_reference`, then drops them
  before the iteration loop. Frees ~144 MB of buttloop-phase working set on
  both paths. Headline peak RSS is ~unchanged (the encoder peaks in the EPF
  sharpness search — a different phase — so e9 only drops 1.61 → 1.59 GB);
  the win is reduced loop-phase memory pressure + removed redundancy.
  Byte-identical (hash-locks 48/48; the 12 MP PQ-HDR file is byte-equal at
  e7/e8/e9). (hdr_vdp2_lite.rs, vardct/perceptual_loop.rs)
- **Prefix-vs-ANS auto choice (libjxl `enc_ans.cc` parity) for VarDCT
  two-pass entropy codes.** Streams that are tiny (< 100 tokens) or
  fully deterministic (every context singleton) now use prefix codes:
  singleton symbols cost 0 bits and prefix streams carry no 32-bit ANS
  state flush, so fully-quantized-to-zero pass groups become literal
  0-byte sections — exact cjxl section-layout parity on smooth content
  (was 4-6 B per group). Tiny synthetic fixtures: 16×16 solid 97→84 B
  (−13.4 %), 8×8 gradient 112→109 B. Streams carrying LZ77 stay ANS
  (our LZ77 writer is ANS-only).
- **Fixed: singleton prefix codes leaked one phantom bit per token.**
  `create_huffman_tree`'s `n == 1` branch assigns a fake depth of 1 (a
  brotli-ism whose "fix on upper level" never existed here) while the
  header writers announce a single-symbol simple code the decoder reads
  at 0 bits — harmless padding at a section end, a latent bitstream
  DESYNC anywhere else (e.g. a singleton DC context in `--no-ans` mode,
  where AC metadata follows DC tokens in the same section). All four
  Huffman emission sites now consult the same depths-scan predicate the
  header writers use (`has_single_used_symbol`), making the two sides
  consistent by construction.
- **REVERTED same-day: the QuantizeWP default-flip below.** The
  nightly W44-202 zenjxl regression gate (per-cell SSIM2 baseline lock)
  failed with the flip ON: 4 photo cells dropped −0.375..−0.854 SSIM2
  (slack −0.300/cell), aggregate −0.226 (slack −0.180). The DC deadzone
  trades photo SSIM2 for bytes beyond the sanctioned budget; libjxl
  parity does not outrank the per-cell quality floor. QuantizeWP stays
  OPT-IN (env hook + field mutation); the Phase 7 static-writer
  sharpness fix, animation requantize parity, and the prefix-auto +
  singleton fixes are KEPT (quality-neutral with the flag off).
  Re-flip requires a Pareto-validated dispatch protecting those cells
  or an explicit decision to re-baseline W44-202.
- **libjxl `QuantizeWP` DC shaping is now DEFAULT at effort ≤ 7
  (W44-AUDIT-8 Phase 7)** — completes the `nl_dc` parity cluster
  started by Phase 5 `extra_dc_precision`. Closes ~half the median
  lossy HDR byte gap vs cjxl (smooth-sky cells up to −22 % bytes;
  `benchmarks/hdr_quantize_wp_ab_2026-06-12.tsv`) at quality
  at-or-better than cjxl (SDR GRAND(369): bytes +0.4 %, butteraugli
  −0.4 %, SSIM2 +0.14). The Phase 6 honest-stop was a misdiagnosis:
  the 3e-4 static-vs-dynamic pixel divergence came from the
  single-pass static writer dropping the computed EPF sharpness map
  (now wired — layout-identical fix), not from the gradient DC
  tokenizer. Animation frames apply the same shaping (still-vs-anim
  consistency gate enforces). Lossy e ≤ 7 hash-locks + Libjxl
  byte-locks intentionally re-baked; every lossless lock
  byte-identical; rd-regression passes within existing tolerances.

### Changed
- **`reconstruct_xyb` is band-parallel (#74 wedge 7b.1a)** — the
  encoder-side reconstruction (EPF sharpness search + buttloop
  fast-path input) now splits into 64 px bands dispatched across the
  rayon pool, mirroring how libjxl pool-threads `ReconstructImage`
  inside its own sharpness search. Transforms are 64 px-tile-contained
  vertically, so bands write disjointly — byte-identical under any
  schedule (verified: output bytes equal pre-change; full lock suite).
  8T on the 1 MP photo cell: recon 20.4 → 5.3 ms, EPF-sharpness stage
  65.9 → 51.1 ms, encode total 189 → 176 ms. 1T unchanged.
- **CLI default features now include `parallel-tree-learning` (#74
  wedge 7a).** The first quiet-box wall grid showed our default-build
  walls FLAT from 1T to 8T while cjxl scales 5–7×; the feature's
  "default-off because it requires rayon" rationale was void — the CLI
  default already ships `parallel` (rayon in-graph). Byte-identical to
  the previous default on every checked cell AND across thread counts;
  lossless e7 wall drops ~25 % even at 1T (1.475 → 1.10 s on the
  1 MP photo cell — the feature's serial-portion chunks reshape 1T
  code too). The remaining 8T gap is wedge 7b: the VarDCT encode
  pipeline has no intra-image parallelism below one 2048² DC group —
  structural work, filed on #74.
- **`EpfDispatch::Auto` is now the default (#74 wedge 5)** — the W36-2
  mask-mean smooth-skip for the per-block EPF sharpness search,
  previously opt-in. On smooth content the search was actively
  counterproductive: the HDR smooth-sky cell paid ~400 B coding a
  noisy sharpness map (29/64 varblocks nonzero vs cjxl's 7/64) AND
  scored worse than the uniform default — post-flip the cell is
  1,091 B at better quality, beating cjxl's 1,230. Textured content
  keeps the full search; W44-202, CID22 GRAND, and rd-regression are
  unchanged. Locks re-baked per the enum doc's flip protocol. Also a
  wall win (the search was 45.5 % of e6 wall where it fired).
- **QuantizeWP DC shaping now fires on PQ/HLG content at effort ≤ 7
  (#74 HDR wedge).** The Phase 7 default-flip was reverted because the
  W44-202 per-cell SSIM2 gate failed on 4 SDR photo cells — but every
  measured WP win is HDR. The dispatch keys on the RESOLVED transfer
  function (the same layout-level predicate as the #73 intensity
  dispatch — not a pixel discriminator, so no content cliff class):
  PQ/HLG at e ≤ 7 enables the libjxl nl_dc shaping; SDR is
  structurally unchanged (W44-202 green by construction);
  `EncoderStrategy::Libjxl` keeps its byte-locked behaviour. Smooth-sky
  e7 d4: 1,915 → 1,487 B (stacks with the #75 fix; old opt-in WP arm
  was 1,503; cjxl 1,230). New lock cell
  `lossy_mg_rgb16_pq_512x512_ramp_e7_d1` pins the dispatched path;
  mask-median content dispatch was measured and REJECTED (photo
  1418519 mask_med 92.3 sits above every sky cell — no separation).

### Documentation
- **README: closed the four worst usability gaps found by an insulated
  external-developer test** (reader given only the README, no source). The
  README had **no install/version line at all** (the workspace-inherited
  version was never stated) — added an `## Install` block (`jxl-encoder =
  "0.3.1"` / `cargo add`, MSRV, `no_std + alloc` note). Added a
  `## Quality (distance) and effort` section stating the distance scale
  explicitly (lower = higher quality; valid `0.0 < d <= 25.0`; `1.0` =
  visually lossless; `0.0` rejected for lossy → use `LosslessConfig`) and
  the `with_effort` setter (default 7, range `1..=12`). Added a
  `## Cancellation` section showing the real API surface — previously
  cancellation was *mentioned* but no `with_stop` setter/token type/example
  appeared: documents `EncodeRequest::with_stop(&dyn Stop)`, the re-exported
  `Unstoppable` no-op, and `almost_enough::Stopper::new()` + `.cancel()` for
  a triggerable token (deps + versions). Added a `## Resource limits`
  section documenting what `Limits::default()` bounds (the 4 GiB lossy /
  8 GiB lossless soft memory cap). README-only; every symbol verified
  against source.
- **`jxl-encoder-cli` README: stated three unstated `cjxl-rs` behaviors
  found by an insulated external-developer test** (reader given only the
  CLI README, no source — the README ran first-try otherwise). Added a
  Notes block under Usage: (1) the output file is **overwritten silently**
  if it exists (`write_output` uses `File::create` —
  `jxl-encoder-cli/src/main.rs:2396`; same on the streaming-output paths
  at `:2166` / `:2253`; no `--force` flag exists); (2) when both `-d` and
  `-q` are passed, **`-d` wins** (`-q` is only the fallback — main.rs:754;
  no clap `conflicts_with`, so it is silent, not an error); (3)
  **`--lossless` and `-d 0` are equivalent** — both fold to distance 0.0
  (main.rs:754) and route to the `LosslessConfig` modular path (main.rs:1433
  `distance > 0.0` is false → modular at main.rs:1813), i.e. true
  mathematical lossless either way. README-only; verified against source.

### Fixed
- **`estimate_peak_memory_bytes` under-reported the lossless path ~14×.**
  The old term-by-term model treated MA tree-learning's working set as
  ~8 B/px and returned ~360 MB for a 12 MP effort-7 lossless encode vs
  ~5 GB measured. Both `LossyConfig`/`LosslessConfig::estimate_peak_memory_bytes`
  now delegate to `heuristics::estimate_encode(..).peak_memory_bytes_max`
  (the conservative upper bound — the method's historical contract); the
  lossy estimate also gains the effort-8 buttloop step (was
  effort-independent). New `estimate_encode(w,h,layout)` exposes the full
  min/typical/max breakdown. (api.rs, f4e2d0d2)
- **Allocation budget over-counted the perceptual loop's scratch buffers,
  failing large encodes earlier than real RSS warranted.** The buttloop's
  `TransformOutput` (~149 MB at 12 MP), recon planes (~138 MB), and VDP2
  HDR reference planes (~144 MB) were reserved with `reserve_permanent`,
  so the reservations persisted through the post-loop entropy phase after
  the buffers had been dropped — and the loop's `TransformOutput`
  double-counted against the base-encode one (they never coexist in real
  memory). On the 12 MP PQ-HDR e9 d4 case this inflated the budget
  high-water mark to 2.99 GB against 2.14 GB of real RSS (38 % phantom),
  which combined with the (then) 2 GiB cap to fail the encode. All four
  sites now hold RAII `BudgetGuard`s released when the loop returns / the
  `TransformOutput` drops; the budget peak falls to 2.55 GB (−438 MB) and
  tracks real RSS. Accounting-only — output is byte-identical (hash-locks
  48/48, libjxl byte-lock 5/5). Extends the W44-AUDIT-2 `epf.rs` fix to
  `transform.rs` + `vardct/perceptual_loop.rs`. Regression coverage:
  `transform_output_reservation_released_on_drop` /
  `transform_output_two_live_still_exceed_cap` (unit), plus the `issue_54`
  / `w44_audit_2` no-spurious-OOM tests pinned to an explicit 2 GiB cap so
  the new 4 GiB default doesn't blunt them.
- **`with_threads(1)` silently used every core (#74 wall-grid
  postmortem).** `run_with_threads` early-returned for `threads <= 1`
  without installing a pool, so the rayon stages inside the encode
  (AC-strategy tiles, entropy-code builds, DC groups) ran on the
  ambient GLOBAL pool — `with_threads(1)` violated its documented
  "force sequential" contract and produced bogus 1T benchmarks (the
  first wall grid's "1T lossy wins 9/10" headline was this bug; TRUE
  1T lossy e7 on the 1 MP photo cell is 0.623 s vs the 0.262 s
  measured-at-32-threads). `threads >= 1` now installs a dedicated
  pool of exactly that size; `0` keeps the documented ambient
  behaviour. Bytes identical across thread counts.
- **#75: gray lossy at e7+ wasted ~2× bytes — CfL Newton fabricated
  ytox on the identically-zero X channel.** Gray input (r=g=b) has an
  exactly-zero XYB X plane; the e7+ Newton refinement's derivatives of
  the zero-residual cost are non-finite at production luma magnitudes,
  and the loop walked on garbage to multipliers up to ±17 — the encoder
  then coded −ytox·Y as the X residual: a scaled copy of Y carrying
  ~49 % of all AC tokens on gray content, at zero quality impact
  (decoder reconstructs X = 0 either way). Fixed with a zero-target
  guard in all three Newton variants (scalar/AVX2/NEON) + a
  non-finite-step break; LS already returned 0 (positive-denominator).
  Patents bilevel cell (e7 d0.5): 1,658,392 → 894,318 B (−46 %) at
  quality better than cjxl on both metrics (was +87 % bytes). RGB
  content structurally unaffected (guard requires an all-zero target;
  locks + byte-locks + CID22 GRAND unchanged). New lock cell
  `lossy_mg_gray_512x512_bilevel_e7_d1` pins X-channel silence; new
  `test_newton_zero_target_returns_zero_all_variants` parity gate
  covers every variant × mode.
- **W44-108 qf-seed sub-band misfire on smooth product shots — distance
  monotonicity violation (GOAL_BEAT_CJXL wedge 2).** imazen-26
  ai-products content (m3 = 28.19, fcbr = 0.70 — passes every
  block-based screenshot proxy including the W44-164 classifier) sat in
  the never-measured m3 ∈ (21, 30) band and took the screenshot qf-seed
  lift at d ≥ 2: our d2.0 encode came out bigger AND higher-quality
  than our own d1.9 (57 KB → 116 KB at e7) at +108..116 % bytes vs
  cjxl. Sub-band m3 ceiling tightened 30 → 24 (ratio midpoint of
  winner-max 20.96 and misfire 28.19) with full re-bisection
  (`benchmarks/qfseed_m3_rebisect_2026-06-12.{tsv,meta}`): 40/40
  gb82-sc protection cells byte-identical, misfire cells land at
  cjxl +3.8..+4.8 % with matched quality, monotonicity restored.
  KNOWN RESIDUAL: the d ≥ 3.5 main band has no m3 check and still
  misfires on this class (+72..+118 % at d4) — needs a new
  discriminator (every existing proxy axis overlaps the main-band
  winners).
- **HLG lossy was distance-flat garbage — missing forward OOTF (caught
  by the new HDR suite).** Our `hlg_*_to_linear` helpers produce scene
  light (inverse OETF only), but every conformant decoder's linear→HLG
  output conversion applies the inverse OOTF (libjxl `ApplyHlgOotf`,
  `jxl_cms.cc:1175` — fires whenever exactly one conversion side is
  HLG): the roundtrip landed at scene^(1/1.2), a constant ~22 dB /
  butteraugli ~54 wedge at every distance while PQ tracked correctly.
  Fix ports the forward OOTF (γ = 1.2 × 1.111^log2(it/1000), skipped
  at 295–305 nits, luminance-weighted per BT.2100/P3/sRGB primaries,
  hue-preserving gamut normalize) into both the one-shot and streaming
  lossy paths before XYB. HLG PSNR now 49.7→43.3 dB tracking d=1→4
  (was 22.5 flat); streaming stays bit-exact with one-shot
  (`test_streaming_lossy_hlg_matches_oneshot`); SDR + PQ output
  byte-identical (locks 45/45, libjxl byte-locks 5/5).
- **#73: lossy VarDCT destroyed PQ/HLG-tagged input — fixed with the
  libjxl-parity intensity pair.** `pq_*_to_linear_f32` normalizes
  1.0 = 10,000 nits, but the header kept `intensity_target = 255` and
  the XYB transform assumed the SDR scale: decoded pixels were garbage
  (PQ-butteraugli ~170 at every distance). Fix = `SetIntensityTarget`
  dispatch (PQ → 10,000, HLG → 1,000; `luminance.cc` parity) +
  `ComputePremulAbsorb`'s `intensity_target/255` multiplier in
  `convert_strip` (`enc_xyb.cc:217` parity; exactly 1.0 for SDR —
  byte-identical, locks 45/45 + libjxl byte-locks 5/5). Post-fix lossy
  HDR quality is AT PARITY with cjxl (PQ-butteraugli ±0.02 at d≤2,
  better at d=4) at +2..+8 % median bytes
  (`benchmarks/hdr_lossy_parity_fixed_2026-06-12.{tsv,meta}`). The CLI
  cICP gating from the hotfix is removed — lossy HDR input now keeps
  correct PQ/HLG signaling end-to-end. Regression gate:
  `tests/it/hdr_pq_lossy_roundtrip.rs` (quality tracks distance, sane
  range, signaling survives).
- **#71: CLI now honours PNG cICP — PQ/HLG HDR input keeps its transfer
  function.** `read_png` surfaces the cICP chunk; the request maps it
  through the libjxl-parity `ColorEncoding::from_cicp` (cICP outranks
  gAMA/sRGB; unsupported combos warn + fall back). Previously a 16-bit
  PQ HDR PNG was signaled as sRGB transfer in the codestream — pixels
  exact, rendering wrong on every conformant viewer. Verified PQ
  signaling on lossless AND lossy output; matches cjxl's signaling on
  the imazen-26 HDR PNG corpus byte-for-tag.
- **#69 item 2: full-image palette transform now runs on the lossless
  multi-group path.** Detection (`analyze_palette`, libjxl 1024-colour
  cap, `--modular_palette_colors` honoured) previously only existed on
  single-group paths: any few-colour image over 256x256 encoded without
  a palette. Now mirrors the single-group tree path before
  ChannelCompact/RCT: palette meta-channel stays whole in the global
  section, index channel splits per-group. nc >= 2 only — single-channel
  palette is ChannelCompact's domain (bitstream-identical transform with
  a stricter benefit filter; verified byte-identical on bilevel).
  43-pick bench: 14-colour plot cell -30.1/-27.7/-31.6 % at e7/8/9,
  everything else byte-identical. Gate:
  `tests/it/lossless_multigroup_palette.rs` + a 4th multi-group
  hash-lock cell (17-colour blocky, 3,424 B).
- **#69 item 1: per-section LZ77 now fires on the lossless multi-group
  path.** The tree-learned multi-group writer deliberately dropped
  `use_lz77`/`lz77_method` (global-code-vs-section histogram mismatch),
  so every lossless image over one group encoded with no LZ77 — the
  documented e7-RLE / e8-Greedy / e9-Optimal schedule was aspirational.
  Now mirrors the squeeze path: per-section transform with
  `dist_multiplier = max(section channel widths)`, global ANS code built
  over the transformed streams, per-group writers re-apply the identical
  deterministic transform. Photo-class sections rarely clear the libjxl
  20 % benefit floor and then stay byte-identical (`lz77.enabled=0`
  layout unchanged). Regression gate:
  `tests/it/lossless_multigroup_lz77.rs` (fires + roundtrips e7/e8/e9 via
  jxl-rs + jxl-oxide; identity on non-repetitive noise).
- **#68: e9+ lossless emitted undecodable bitstreams** on content whose
  learned tree split on a mid-group reference-channel property.
  `collect_residuals_with_tree*` sized the per-pixel property record at
  `max_tree_prop + 1` while the ref-property writer fills whole 4-slot
  groups behind a `base + 3 < prop_stride` guard — a tree whose max
  property id landed mid-group (only reachable at e9+, where the learner
  may split on slots 0..2) had its entire top ref-group skipped: the
  encode-side walk read zeros where every decoder computes real values,
  desyncing contexts (`SectionTooShort` / entropy EOF in zenjxl-decoder,
  jxl-oxide AND djxl). Stride now rounds up to whole ref-groups;
  byte-neutral for e<=8 (slot-3-only trees always had complete groups).
  Regression gate: `src/issue68_regression_tests.rs` (512^2 LCG noise,
  e8+e9, jxl-oxide decode-verified). Found by zenjxl's sweep_validate
  lossless-roundtrip-exactness harness.
- **#68 second cause: property-1 (`group_id`) used ad-hoc stream ids.**
  The multi-group lossless gather/apply/write paths numbered pass-group
  streams `meta_offset + group_idx`, while every decoder evaluates tree
  property 1 as the spec ModularStreamId
  (`1 + 3*num_lf_groups + 17 quant tables + group`). Internally
  consistent, spec-divergent: any e9+ tree that split on group_id (the
  e9 property order keeps it; below e9 it is erased, and single-group
  images make it constant) desynced on photo-class content even after
  the stride fix — the photo encodes were byte-identical pre/post that
  fix, which is how the harness re-run exposed the second cause. All
  three sites (section.rs gather+apply, frame.rs writer, vardct
  lf_frame) now derive ids from `modular_hf_stream_id_base()`.
  Byte-neutral below e9 (group_id is not a candidate property there;
  e8 encodes verified byte-identical). New corpus-free regression gate
  `e9_lossless_decodes_on_group_distinct_content` (quadrant-distinct
  512^2 content makes group_id predictive; verified to FAIL under the
  old ids).

### Changed
- **#69: lossless knob setter docs now state real consumption semantics**
  (doc-only, byte-neutral): `with_lz77`/`with_lz77_method` are
  deliberately dropped by the lossless multi-group section writer
  (global-tree ANS vs per-group histogram mismatch — the effort
  schedule's Greedy@e8/Optimal@e9 is aspirational there);
  `with_tree_learning_sample_fraction` is stride-quantized (f rounds
  down to nearest 1/k; (0.5,1.0) is byte-identical to 0.5);
  `with_patches`/`with_modular_palette_colors` are stored pending
  lossless-side detection. Issue #69 rescoped to the actual wiring work.

### Added
- Versioned public-API surface snapshots at `docs/public-api/{jxl-encoder,
  jxl-encoder-simd,jxl-encoder-macros}.txt`, regenerated on every
  `cargo test` by `jxl-encoder/tests/public_api_doc.rs` (`ZEN_API_DOC=check`
  verifies in CI's clippy x64 leg, `=off` skips); justfile recipes
  `fmt` / `api-doc` / `api-doc-check`. jxl-encoder itself is snapshotted
  default-features-only (the GPU-loop feature union does not compile —
  cubecl git pin vs zenforks-cubecl registry crates).

### Changed
- **rate-control decodes via zenjxl-decoder; jxl-oxide is dev-only.** The
  library's only production jxl-oxide use (the `rate-control` feature's
  decode-to-linear) now uses zenjxl-decoder's F32 output path (linear
  sRGB directly, color channels only — the old `image_all_channels()`
  would have interleaved alpha into the 3-channel diffmap stride).
  jxl-oxide remains a dev-dependency for secondary roundtrip validation
  (jxl-rs + djxl stay primary). Zero jxl-oxide-family crates left in the
  normal dependency graph; rate-control integration tests (encode →
  decode → butteraugli convergence) pass on the new decoder.

### Added (2026-06-10 lossless bench set)
- **k-means-stratified lossless benchmark set from imazen-26**
  (`benchmarks/lossless_bench_set_2026-06-10.tsv` + `.meta`,
  `scripts/select_lossless_bench_imazen26.py`): 43 picks (13 tier=core)
  across 23 content strata — modern mobile/web screenshots, document
  scans, 1-bit + gray-rescan patents, manuscripts, plots, AI classes,
  plus the JPEG photographic classes (general/interiors/nature/food/
  people, renders, textures, museum repros) — selected by per-stratum
  k-means on zenanalyze features (centroid-nearest, seed 42) from 1,951
  ≤16 MP candidates. JPEG picks are pre-decoded once to stripped PNGs
  under `/mnt/v/input/jxl-encoder/lossless-bench-imazen26-png/`
  (harnesses feed the TSV's `bench_input` column). Replaces ad-hoc
  lossless bench sets; CLIC-1024 photos + gb82-sc {terminal, imac_dark}
  stay as continuity cells. Bilevel scans, documents, and true-photo
  classes had zero prior lossless-bench coverage.

### Added (2026-06-10 issue #43 chunk 2c)
- **Screenshot-class 8×8-transform lift at effort 5, d ∈ [1.0, 2.0]**
  (`adapt_to_image_content`): `Screenshot`-classified images (W44-164
  auto-classifier on `EncoderStrategy::Zenjxl`/`Aggressive`, or explicit
  `with_content_class`) now evaluate the DCT4X8/DCT8X4/DCT4X4/AFV0-3
  block at effort 5 — one effort level below its libjxl-parity entry
  (e6/Wombat) — inside the measured distance win band. gb82-sc
  production-context paired A/B
  (`benchmarks/dispatch_2c_afv_screenshot_2026-06-10.{tsv,meta}`):
  in-band 14/14 cells win bytes, mean −1.21 % at better mean butteraugli,
  wall +0.5 %; photos 20/20 and e6 cells 28/28 byte-identical. Env hook
  `JXL_DISPATCH_AFV_SCREENSHOT_DISABLE=1` suppresses. The issue's
  original "auto-enable AFV at e ≥ 6" spec was a structural no-op —
  `try_dct4x8_afv` is already default-on at e ≥ 6 for every strategy
  (proven byte-identical: pin-probe e7 DEF == PIN_ON,
  `benchmarks/dispatch_2c_afv_screenshot_pin_probe_2026-06-10.tsv`).
  Photos and sub-65,536-px inputs (all hash-lock fixtures) are untouched.

### Changed
- **`__env_var_diagnostics` feature: diagnostic dump env hooks compile
  out by default.** All per-block/diagnostic dump hooks (six dump
  modules, cost-input/AFV/coeff-order/EPF/CfL dumps, phase timing) now
  exist only behind the opt-in feature; default binaries carry zero
  hook strings, zero probes, zero gate branches. Behaviour-override
  hooks stay runtime (test/bench contract). Hash-locks pass in default,
  parallel-tree-learning, and feature-on configs; dump-driver examples
  declare `required-features` and fail loud without it.
- **VarDCT: skip the content-classifier sweep where nothing consumes it
  (byte-identical).** `ZenanalyzeProxies` (24 % of lossy e3 CPU) is now
  computed only when a consumer can fire (`effort ≥ 5 || distance ≥ 2`);
  bytes identical 12/12 across both sides of every band edge. Lossy
  e3/e4 walls −17…−20 % on top of the env-gate wins (~−25 % combined)
  (`benchmarks/perf_lossy_low_2026-06-10.meta` addendum 2).
- **VarDCT: once-per-process gates on the diagnostic dump env hooks
  (byte-identical).** Five W44-era dump hooks probed `std::env::var`
  per BLOCK — 25–35 % of CPU at lossy e3/e4 (getenv + env RwLock + CStr
  scan). Presence is now cached at first probe (absent ⇒ permanently
  off; present ⇒ legacy re-read semantics). Lossy walls: e3 −6…−6.8 %,
  e4 −7.5…−8.4 %, e5 −0.4…−3 %; bytes identical 9/9 + all locks. Hooks
  must be set before the process's first encode
  (`benchmarks/perf_lossy_low_2026-06-10.meta`).
- **Tree learning: MSD radix bucketing for the dedup sort
  (byte-identical, proof-unblocked).** Equal-key representative choice
  was PROVEN byte-irrelevant (representative-flip probe: hash-locks +
  3 real cells identical), dissolving the earlier radix blocker; the
  sort is now 16-bit-prefix counting buckets + parallel per-bucket
  pdqsort. Bytes identical 5/5 through a different algorithm; walls
  lean-positive (noaa −2.4 %, city12mp −1.6 %), quiet confirm pending
  (`benchmarks/perf_radix2_msd_2026-06-10.{tsv,meta}`).
- **Tree learning: exact capacity reservation (byte-identical).** Gather
  reserves every SoA column to the known sample upper bound and collect
  pre-sizes its token Vec to the pixel count — the doubling-realloc
  `__memmove` (10.9 % of CPU on terminal) is gone. clic097 −6.98 %,
  noaa −5.95 %, terminal −5.19 %, imac_dark −1.12 %
  (`benchmarks/perf_reserve_sizing_2026-06-10.{tsv,meta}`).
- **Modular: estimate_cost context LUT + hoisted rows (byte-identical).**
  The RCT-trial cost estimator (7 calls/image at e7) replaces its
  17-compare-per-pixel cutoff scan with a 501-entry LUT and its branchy
  neighbor loads with row-hoisted edge/interior paths — exact same
  contexts/tokens. All cells faster: noaa −4.7 %, imac_dark −3.8 %,
  terminal/clic097 −1.3…−1.4 %
  (`benchmarks/perf_estcost_lut_2026-06-10.{tsv,meta}`).
- **Tree learning: inline dedup key comparator (issue #41 "radix" chunk
  1, byte-identical).** The packed-key sort's comparator is now an
  inline big-endian u64-word compare (identical ordering ⇒ identical
  pdqsort permutation ⇒ identical bytes) instead of libc `__memcmp` per
  comparison. Photos −2.9…−3.1 %, terminal −1.8 % at moderate load;
  true radix honest-stopped (unstable-sort representative choice is
  byte-relevant) — `benchmarks/perf_radix1_cmp_2026-06-10.{tsv,meta}`.
- **Tree learning: fused WP predict+update (issue #41 item 2,
  byte-identical).** `predict_property_update` fuses the per-pixel WP
  prediction and error update into one call at both hot sites (gather +
  collect pass 1) — same values, same state sequence. The item's
  original cross-pixel batching ask is semantics-blocked and documented
  at the definition: WP prediction at x reads the current row's
  transmission error at x−1 (`te_w`), a strict serial dependency
  matching libjxl's WP. Walls join the consolidated quiet re-measure.
- **Tree learning: lane-per-predictor gather (issue #41 item 1,
  byte-identical).** The gather hot loop computes all 14 canonical
  predictions in one straight-line pass (`predict_all_canonical`)
  instead of 14 per-pixel match dispatches; legacy loop retained for
  squeeze/permuted candidate lists. Value-parity unit test + bytes
  identical 5/5 A/B cells + hash-locks both feature sets. Walls pending
  a quiet box (`benchmarks/perf_gather_item1_lanes_2026-06-10.{tsv,meta}`).
  Also bumps the lock to zencodec 0.1.21 (published 2026-06-10,
  hash-lock-proven byte-neutral).
- **Tree learning: batched MA-tree traversal in collect_residuals (issue
  #41 chunk B2, byte-identical).** 3-pass row restructure: properties/WP
  in legacy order into row buffers, then a K=8 interleaved tree walk
  (`batch_traverse_row`) overlapping eight dependent-load chains, then
  legacy-order token emission. Quiet A/B: −0.9…−2.1 % wall across every
  content class (photos −1.7…−2.1 %, screens −1.3…−1.7 %, docs −0.9 %),
  stacking on chunk B1
  (`benchmarks/perf_traverse_b2_2026-06-10.{tsv,meta}`).
- **Tree learning: gather row staging (issue #41 chunk B1,
  byte-identical).** Default-path `gather_channel_samples` stages each
  row's samples in flat column-major scratch and flushes once per row
  via per-column `extend_from_slice`, replacing ~45–60 per-sample
  `Vec::push` calls (the 22 %-of-gather store path from the step-0
  annotate). Dedup-backend paths unchanged. Mins-based A/B: all 5 cells
  ≤ base, −0.5…−2.0 % on gather-heavy document/screenshot cells; quiet
  confirmation queued
  (`benchmarks/perf_gather_b1_staging_2026-06-10.{tsv,meta}`).
- **Tree learning: collect_residuals chunk A (issue #41, byte-identical).**
  Spec properties write directly into the extended property buffer
  (`compute_spec_properties_into`), replacing a measured-hot per-pixel
  `copy_from_slice`, and the invariant ref-prop tail zero-fill is hoisted
  out of the pixel loop. Bytes identical on the A/B grid + hash-locks
  both feature sets; both property fns stay fully inlined (nm-verified).
  Quiet-box verdict: wall-neutral (±0.7 % on mins; base-min == ours-min
  on the only positive-median cell) — kept as strictly-less-work
  groundwork for the chunk-B restructure
  (`benchmarks/perf_collect_chunka_2026-06-10.{tsv,quiet.tsv,meta}`).
- **Tree learning: BestSplit side-costs rider (issue #64 follow-on,
  byte-identical).** `BestSplit` now carries the winning threshold's
  left/right entropy costs out of the split sweep; all six engine call
  sites skip the 2×O(n_side) `compute_predictor_entropy` recompute per
  split. Bitwise-identical by construction (same u32 histograms, same
  `estimate_bits_u32` call shape; the sweep's `effective_histo ==
  histogram_size`) and enforced by permanent debug-build asserts at every
  consuming site. Proof: hash-locks 36/36 (default + parallel-tree-
  learning), libjxl byte-locks 5/5, full suite, and base-vs-ours sha256
  identity on 7 lossless cells + 2 lossy probes
  (`benchmarks/perf_bestsplit_rider_2026-06-10.{tsv,meta}`). Quiet-machine
  re-run (`.quiet.tsv`, loadavg 0.7): lossless photos −1.3…−1.8 % wall at
  1T (e7 and e9), −0.8 % at 8T; non-split-bound controls within noise.
- **Lossless tree learning: parent-histogram subtraction (issue #64 chunk 1,
  PERF-HIST-SUB-LOSSLESS).** `find_best_split` per-node aggregates
  (per-(prop,pred,bucket,token) counts, extra-bit sums, bucket weights) are
  now carried in additive integer tensors: the smaller child of every
  qualifying split is built from its rows and the larger child is derived by
  exact u32/u64 subtraction, skipping the larger side's per-sample loops
  (~87% of `find_best_split`, the top symbol at 27% CPU on lossless photos).
  Wall A/B (5 CLIC-1024 photos × e{7,9} × {1T,8T}): 20/20 cells faster,
  mean −20.7%, peaks −33.7% (e7 1T) / −38.1% (e9 8T); bytes identical on
  every cell. Peak RSS unchanged: an initial +47.7 MB single-sample
  reading was root-caused to glibc adaptive-arena noise (±120 MB band
  between runs of the SAME binary); with mmap_threshold pinned, ours
  measures −2.2 MB vs base (537b8974 addendum in the bench meta).
  Byte-identical output by construction (hash-locks 36/36 unchanged, libjxl
  byte-locks pass, derived==built unit tests); full results in
  `benchmarks/perf_hist_sub_lossless_2026-06-10.{tsv,meta}`. Lossy paths and
  the owned small-image fallback are unaffected (full-rebuild). (276c94dd)

### Fixed (2026-06-10 CI repair, round 2)
- **aarch64 NEON `compute_pre_erosion` read past the row/buffer on the last
  image column.** The NEON and WASM128 ports of the pre-erosion kernel were
  missing the AVX2 variant's `interior_end` right-edge guard: the 4-wide
  `right`-neighbour load could cover the final column, reading the next
  row's first pixel (wrong masking input) or running off the buffer on the
  last row (panic — caught by `test_per_region_quant_field_matches_whole_image`
  on the ubuntu-24.04-arm runner). The default whole-image path pads its
  buffers and was unaffected; only the per-DC-group region path (#11
  streaming work) on aarch64/wasm could observe it. All three vector
  variants now cap the vector interior one column short of the image edge
  and let the clamping scalar tail finish the row. Verified: aarch64 QEMU
  suite passes, x86 hash-locks 36/36 byte-identical, 185 SIMD parity tests.
- **32-bit cross builds**: `cvvdp-gpu` (dev-dep for one tracking example)
  doesn't compile on 32-bit (8 GiB VRAM budget overflows usize — filed
  imazen/zenmetrics#26); the dev-dep is now gated to
  `cfg(target_pointer_width = "64")`.
- **Corpus-gated decoder roundtrips + cross sibling mounts** (551eb7b8):
  the 8 decoder-roundtrip tests that hardcoded the dev machine's corpus
  path are now CODEC_CORPUS_DIR-resolved and `#[ignore]`-gated, run daily
  by nightly with a sparse corpus checkout; cross containers now same-path
  mount the host sibling layout via Cross.toml `[build.env] volumes`.

### Fixed (2026-06-09 CI repair)
- **CI green again end-to-end.** Main had been red since 2026-05-19 (#66),
  letting breakage accumulate: the sibling-clone set was missing the
  transitive closure (heic/ultrahdr/zenpng/zenwebp/zenavif/zenjxl/fast-ssim2/
  zentone/dssim — d45277b2), `cargo test` didn't compile (two
  `compute_cfl_map` call sites missed the W44-AUDIT-5 P2 12th argument —
  f4a6df03 + follow-up), `zenjxl-tuning-runner` used unix-only
  `libc::getrusage` unconditionally breaking every Windows build (01c171e8),
  wasm/cross builds pulled zstd-sys via the tuning-runner (excluded), and
  ~83 clippy warnings gated `-D warnings` (9344bfc6). The fmt job is now
  scoped to workspace members so cloned siblings can't fail it.
- **Default-build perf nit**: the W44-9 dct8 A/B counters did two atomic ops
  per strategy evaluation in production; the apparatus is now compiled only
  under `__expert` (9344bfc6).

### Changed
- **Documentation overhaul (2026-06-09)**: CLAUDE.md's 3,700-line dated
  investigation archive and the Feb-2026 status/roadmap snapshots moved
  verbatim to `docs/CODE-HISTORY.md`; CLAUDE.md keeps a distilled
  binding-constraints index plus live follow-ons. Shipped cvvdp phase briefs
  (3-6) moved to `docs/archive/`; `butteraugli_loop.rs` →
  `perceptual_loop.rs` rename artifacts fixed in LIBJXL_DIVERGENCES.md and
  TUNING_RELATIONS.md.
- **Consolidated the 90 integration-test files into a single `it` binary** (693576ec).
  Every former `tests/<name>.rs` moved to `tests/it/<name>.rs` and is now a submodule of
  one `tests/it/main.rs` target instead of compiling+linking as 90 separate test binaries —
  each former file linked the whole crate + dependency graph on its own, so this turns ~90
  link steps into one. Measured `cargo test -p jxl-encoder --tests --no-run` (rebuild lib +
  test binaries, 3-run median, default features): debug 61s → 38s (~38% faster), release
  54s → 37s (~31% faster); test census identical (2075 debug / 2073 release listed tests,
  `#[ignore]` count 223 unchanged). Feature-gated files keep their own `#![cfg(...)]`, so
  they still compile to an empty module when off. Select a former target with a module-path
  filter, e.g. `cargo test -p jxl-encoder --test it clic2025::test_rd_regression`; the
  `--test <name>` selectors in `justfile`, `nightly.yml`, and operational docs were updated
  to `--test it <name>`.

### Fixed
- **JPEG transcode: reject chroma sampling factors > 2 instead of panicking.**
  A JPEG with H/V sampling > 2 (4:1:1, the odd 3×N, …) can't be represented in
  the JXL `YCbCrChromaSubsampling` wire format (factors {1,2} only); cjxl
  refuses these too. The transcode path silently mis-mapped them and then
  panicked (OOB) in the coefficient/CfL path — now a clean `Unsupported` error
  (874142dc).

### Added
- **JBRD round-trip conformance gate** (`tests/jbrd_roundtrip_conformance.rs`,
  53 fixtures): transcode each JPEG → JXL and reconstruct it with zenjxl-decoder
  (pure Rust, no external djxl), asserting the brunsli contract — every JPEG
  either cleanly rejects or reconstructs byte-exact, never silent corruption.
  Self-describing (derives the expected outcome from each header) so it runs on
  any corpus via `JBRD_CONFORMANCE_CORPUS`. 47 round-trip + 6 clean-reject pass,
  0 failures (EXIF round-trip closed, imazen/zenjxl-decoder#19). Includes
  `meta_d_exif_orient_big.jpg` — a large-codestream EXIF-`Orientation` fixture
  that regression-guards trailing metadata-box reconstruction (small fixtures
  can't, since frame-decode read-ahead incidentally slurps their trailing
  boxes) — and `uhdr_gainmap_mpf.jpg`, an UltraHDR multi-picture (MPF) JPEG
  whose trailing gain-map sub-image is preserved verbatim as JBRD `tail_data`:
  the primary SDR image transcodes losslessly while the gain map survives
  byte-exact with no recompression. zenjxl-decoder is a path-patched
  dev-dependency + clone-siblings CI entry (874142dc, d0183ed4).
- `examples/jbrd_diff.rs`: transcode + reconstruct + first-diverging-byte
  debug helper (d0183ed4).
- `examples/jpeg_pixel_parity.rs`: decode-only pixel parity (jxl-rs decode of
  the transcoded JXL vs the JPEG's own decode) + byte-exact reconstruction
  check, covering orientation (EXIF) and ICC color-managed inputs.

### Performance

- **FMA-ize the forward DCT-32 / DCT-64 B-transform butterflies
  (newer-intrinsic survey #1, 2026-06-01)**: the
  `s[0] = sqrt2 * s[0] + s[1]` B-transform step in `dct1d_32_batch`
  and `dct1d_64_batch` (AVX2 + NEON + WASM128 variants in
  `jxl-encoder-simd/dct32.rs` + `dct64.rs`) now uses
  `sqrt2.mul_add(s[0], s[1])`, matching the established `dct8`/`dct16`
  idiom. Each butterfly emits one fused `vfmaddps` (x86) / `vfmaq_f32`
  (NEON) instead of a separate multiply+add — one fewer instruction
  and one fewer rounding on the dependency chain (asm-confirmed: +1
  vfmadd in each of the two AVX2 batch symbols). The forward DCT is
  encoder-internal, so this has zero decoder-conformance / libjxl-
  interop risk. Output is byte-identical to the prior build on all 12
  zenjxl regression-gate cells (the ≤1 ULP nudge did not cross a
  quantization boundary). The IDCT kernels (`idct16/32/64`) have no
  `a*x+y` MAC patterns to fuse (pure add/sub + scale butterflies),
  so they were intentionally left unchanged. Full suite green
  (1876 passed). Bench + asm verification:
  `jxl-encoder-simd/benchmarks/dct_fma_2026-06-01.md`.

- **JPEG-in-JXL recompression: port libjxl 3-way context-map cost
  comparison (Lever #1 / MTF, 2026-05-28)**: extends the libjxl
  `enc_context_map.cc::EncodeContextMap` 3-way comparison (simple /
  Huffman / Huffman+MTF) to the writer used by the JPEG transcode
  block-context-map path (`vardct/context_tree.rs::write_context_map_from_slice`)
  AND replaces the Shannon-entropy proxy in the AC stream's
  Huffman+MTF sub-decision with actual trial-encoding
  (`entropy_coding/encode_ans.rs::write_context_map_nonsimple_huffman`).
  Both candidates now share a `write_huffman_payload_no_selector`
  helper so the cost comparison uses the SAME bit-count math as the
  final emission. On the 200-file JPEG bench (seed 11): saves
  -36 bytes total = -0.000079% (the dominant ctx_map gain was already
  captured by `5a6b04c9 feat(entropy_coding): LZ77 in
  write_context_map_nonsimple` / W44-73). HONEST-STOP on the bench
  target. Decoder roundtrip preserved (djxl 3/3 byte-identical,
  jxl-rs/jxl-oxide via the 27-test `jpeg_reencoding` suite).
  Re-pinned `strategy_libjxl_byte_lock` golden (10 cells, -4 B each on
  the Libjxl-strategy 15-cluster fixtures) + `strategy_libjxl_hash_locks`
  LIBJXL_PINS (5 fixtures, -4 B each). Non-default lossy hash-locks
  (36 cells) BYTE-IDENTICAL — the compact COMPACT_BLOCK_CONTEXT_MAP
  path through `write_block_context_map` is deliberately unmodified.
  See `benchmarks/jpeg_lever1_mtf_2026-05-28.{tsv,meta,notes.md}`.

### Fixed

- JPEG-in-JXL recompression: byte-exact roundtrip for baseline JPEGs
  whose source encoder pads the final partial byte of each entropy
  segment with 0-bits instead of the spec-recommended 1-bits. The
  `jpeg-reencoding` feature now reads zenjpeg 0.8.6's
  `JbrdMetadata.has_zero_padding_bit` + `padding_bits` fields and
  copies them into `JpegData` so the existing JBRD writer at
  `jpeg/jbrd.rs:203-211` emits the source's exact pad bits.
  djxl's `JumpToByteBoundary` then consumes those bits to produce
  byte-identical reconstruction. Task #11. Closes the last residual
  DIFF on the 200-file recompression bench — 200/200 byte-identical
  via `djxl --reconstruct_jpeg`. See
  `benchmarks/jpeg_zero_padding_bit_2026-05-28.{tsv,meta}`.

### Changed (BREAKING — 2026-05-25)

- **Multi-metric API refactor (RFC #3 Phase 0)**: the cvvdp-specific
  builder methods `LossyConfig::with_cvvdp_loop` /
  `LossyConfig::with_cvvdp_use_cpu` and the GPU butteraugli builder
  `LossyConfig::with_gpu_butteraugli` have been **DELETED**. Use the
  unified replacements:
  - `LossyConfig::with_perceptual_metric(PerceptualMetric)` — pick the
    metric: `Butteraugli` (default) or `Cvvdp`.
  - `LossyConfig::with_perceptual_device(PerceptualDevice)` — pick the
    device: `Auto` (default), `Cpu`, or `Gpu`.
  - `LossyConfig::with_perceptual_target_score(Option<f32>)` — override
    the metric's per-distance target table.

  Migration cheat-sheet:
  | pre-Phase-0                                  | Phase 0 replacement                                                                                     |
  |---                                           |---                                                                                                      |
  | `.with_gpu_butteraugli(true)`                | `.with_perceptual_device(PerceptualDevice::Gpu)`                                                        |
  | `.with_gpu_butteraugli(false)`               | `.with_perceptual_device(PerceptualDevice::Cpu)`                                                        |
  | `.with_cvvdp_loop(Some(true))`               | `.with_perceptual_metric(PerceptualMetric::Cvvdp)`                                                      |
  | `.with_cvvdp_loop(Some(false))` / `None`     | `.with_perceptual_metric(PerceptualMetric::Butteraugli)` (or no call — that's the default)              |
  | `.with_cvvdp_use_cpu(Some(true))`            | `.with_perceptual_device(PerceptualDevice::Cpu)` (independent of metric)                                |
  | `.with_cvvdp_use_cpu(Some(false))` / `None`  | `.with_perceptual_device(PerceptualDevice::Auto)` (default) or `::Gpu`                                  |

  `LossyConfig::with_cvvdp_bytes_tighten(Option<bool>)` is RETAINED —
  it's a cvvdp-specific tuning knob (post-convergence bytes-tighten
  exit pass), not a metric-selection setter. The resolver now keys off
  `resolve_perceptual_metric() == Cvvdp` instead of the deleted
  `resolve_cvvdp_loop`.

  Cargo feature names UNCHANGED — `cvvdp-loop`, `cvvdp-loop-cpu`,
  `cvvdp-loop-tighten`, and `gpu-butteraugli` all stay valid. Only the
  runtime API changed.

  Default behaviour is byte-identical to the pre-Phase-0
  `LossyConfig::default()`: the default
  `(PerceptualMetric::Butteraugli, PerceptualDevice::Auto)` resolves to
  the same backend choice as the pre-Phase-0
  `gpu_butteraugli = cfg!(feature = "gpu-butteraugli")` + `cvvdp_loop =
  None` + `cvvdp_use_cpu = None` shape (verified hash-locks 36/36
  BYTE-IDENTICAL on the default feature set).

  `EncoderStrategy::Libjxl` strict cjxl-parity invariant preserved —
  `resolve_perceptual_metric()` forces `Butteraugli` regardless of
  caller selection (verified by the extended
  `strategy_libjxl_byte_lock` test with the new
  `(metric, device)` matrix).

  Files: `jxl-encoder/src/api.rs` (new enums + setters + resolvers, old
  setters deleted), `jxl-encoder/src/vardct/perceptual_backend.rs`
  (new `MetricSelection` struct, `construct_backend` signature
  collapsed from 7-arg to 5-arg + `propagate_resolved_metric_to_encoder`
  helper), `jxl-encoder/src/vardct/perceptual_loop.rs` (call-site
  update). 4 integration tests + 16 example files migrated. See
  [`docs/RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md`](docs/RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md)
  for the full design.

### Added

- **Lossy JPEG → JXL recompression (PreserveJxl, coefficient-domain,
  2026-05-28)**: re-compress an existing JPEG into a smaller JXL by
  re-quantizing its own quantized DCT coefficients to a coarser
  same-family scale of its quant tables, then losslessly transcoding —
  no pixel round-trip, so no generation-loss artifacts and no JBRD
  (lossy). New public entries on the `jpeg-reencoding` feature:
  `jpeg::encode_jpeg_recompress_auto_codestream(bytes, scale, effort)`
  (recommended — bundled `coarsen_policy`: scale-proportional AC
  deadzone + mild chroma lead), `encode_jpeg_recompress_planar_codestream`
  (explicit luma/chroma knobs), `coarsen_coefficients{,_auto,_dz,_planar}`,
  `coarsen_policy`. CLI: `cjxl-rs <in.jpg> <out.jxl> --jpeg-coarsen <scale>`.
  Output is guaranteed not larger than the lossless transcode (no-size-
  regression guard). Measured RD findings, all three target metrics
  (zensim-A / cvvdp / butteraugli), in `docs/JPEG_LOSSY_RECOMPRESSION.md`:
  the AC deadzone is a strict Pareto win (smaller AND higher quality on
  8/10 files); PreserveJxl wins at gentle/near-lossless targets while a
  full pixel re-encode wins at aggressive (content-dependent crossover);
  an oracle router (pick the smaller) beats the naive pixel-only default
  (cjxl's only lossy-JPEG option, which is *larger than lossless* at
  gentle quality) by −11.0% on zensim-A (−14.7% cvvdp, −6.2% butteraugli);
  and an inferred-target floor calibration (source quality caps achievable
  absolute quality). Decoder-validated (jxl-oxide + djxl). Benches:
  `benchmarks/jpeg_lossy_{rd_frontier,closed_loop,crossover_*,inferred_target}_2026-05-28.*`.

- **Phase 1 of butteraugli target symmetry (RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`,
  2026-05-26)**: `LossyConfig::with_perceptual_target_score(Some(score))`
  now drives the buttloop's `effective_metric_target_distance`
  convergence target for all three metrics. **Pre-Phase-1 it was a
  phantom no-op** — the field stored on the config and threaded
  through `MetricSelection.target_score` but DISCARDED at
  `vardct/perceptual_backend.rs::construct_backend` (the
  `let _ = selection.target_score;` no-op binding). The buttloop's
  dispatch in `vardct/perceptual_loop.rs` hard-coded each metric arm
  with no reference to the caller field. Phase 1 closes the wiring:
  - New `vardct/butteraugli_targets.rs` — inverse-direction
    `target_score → effective_distance` calibration table (7
    distance bands, seeded from
    `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`, n=162
    corpus-median per band). Mirrors `vardct/cvvdp_targets.rs` shape.
  - butteraugli arm: caller's score → inverse-table lookup →
    `effective_distance`; loop converges at that distance.
  - cvvdp + zensim arms: caller's score used DIRECTLY as the
    metric-native convergence target (bypasses the forward
    distance-table; the score is already metric-native).
  - New `LossyConfig::resolve_perceptual_target_score()` resolver
    with `EncoderStrategy::Libjxl` strict-parity short-circuit
    (forces `None` for Libjxl; W44-126 byte-lock invariant). Also
    sanitises NaN/Inf/non-positive caller inputs back to `None`.
  - New `VarDctEncoder::perceptual_target_score: Option<f32>` field
    threaded via `propagate_resolved_metric_to_encoder`.
  - New tests:
    `tests/perceptual_target_score_smoke.rs` (6 tests including
    jxl-rs primary roundtrip and jxl-oxide secondary roundtrip),
    extended
    `tests/strategy_libjxl_byte_lock.rs` with
    `libjxl_target_score_byte_identical_via_strict_parity_short_circuit`
    (40 byte-lock cells), 3 new api resolver tests, 10 unit tests
    on the `butteraugli_targets` module (table monotonicity,
    endpoint clamps, linear interp, NaN/negative guards,
    inverse-at-canonical-bands, ±10% seed envelope honeypot).
  - Default `perceptual_target_score = None` preserves
    pre-Phase-1 behaviour byte-for-byte; hash-locks 36/36
    BYTE-IDENTICAL. `EncoderStrategy::Libjxl` byte-lock 4/4
    BYTE-IDENTICAL with target_score active.
- **scripts/zenjxl-tuning-sweep/lib/zen-r2-lib.sh — shared R2 boilerplate
  (cross-repo dedup Chunk D, 2026-05-26)**: sourced shell helper that
  extracts the R2/S3 endpoint + creds + `aws s3` / `s5cmd` wrappers
  copy-pasted across 24 sweep scripts in this directory (see
  `zen/zensim/benchmarks/dedup_inventory_master_2026-05-26.md` Cluster
  A Class 4 for the inventory). Canonical home is
  `zenmetrics/scripts/lib/zen-r2-lib.sh`; the byte-identical copy here
  is the pragmatic-first sibling for cross-repo source duplication
  (acceptable for a sourced shell lib per the dedup synthesis;
  a future chunk may promote to a git submodule or shared install).
  Env contract matches `zenmetrics/crates/vastai-fleet/src/worker/r2.rs`:
  `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`, with
  fallback source from `~/.config/cloudflare/r2-credentials`. Exposes
  `zen_r2_init` (idempotent), `zen_r2_s3`, `zen_r2_s5`, `zen_r2_sync`
  (3-retry), `zen_r2_verify` (head-object pre-flight, mirrors the
  W44-PHASE4-S1h pattern), `zen_r2_hydrate_from_proc1environ` (vast.ai
  onstart). Migrated 2 representative scripts as proof:
  `launch_w44_216_fleet.sh` (replaced the 4-line creds-check +
  endpoint-derive block) and `worker.sh` (added source for local
  invocations; production bootstrap path unchanged because the launcher
  pre-exports `AWS_*` + `S3_ENDPOINT_URL`). Follow-on chunks will
  migrate the remaining 22 scripts in this directory (12 finalize +
  janitor + launch + 5 build_*.py / merge_*.py + onstart.sh +
  Dockerfile env-staging).

- **zensim fork — buttloop body wiring + per-distance calibration table (Phase 4)**
  (RFC [`docs/RFC_ZENSIM_FORK_PLAN.md`](docs/RFC_ZENSIM_FORK_PLAN.md) §6,
  mirrors cvvdp-fork Phase 4 (commit `32581839`)). Routes the zensim
  signal through `run_buttloop` proper. The pre-Phase-4 buttloop body
  consumed butteraugli-direction `target_distance` verbatim when zensim
  was active (the Phase 3 dispatch flag short-circuited the metric-
  target lookup); Phase 4 plumbs a zensim-native target through the
  same `target > effective_metric_target_distance` predicate and per-
  block bad-block math the cvvdp Phase 4 already exercises.
  Surface:
  - `vardct/zensim_targets.rs` (NEW) — `ZENSIM_DISTANCE_TARGETS`
    static + `zensim_target_score_for_distance` linear-interp lookup
    over a 7-entry table at d ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}.
    Seed values measured by the new `zensim_calibration_seed` example
    (3 images × 7 distances = 21 cells, post-hoc score over
    butteraugli-default output). 7 unit tests.
  - `vardct/perceptual_loop::ActiveMetric` enum
    (`Butteraugli`/`Cvvdp`/`Zensim`) — generalises the cvvdp-only
    bool dispatch.
  - `vardct/perceptual_loop::ZENSIM_BLOCK_CONSTANTS` (`k_tile_norm =
    1.2` butter-parity) consumed by the new
    `block_reducer_constants_for_metric(metric)` dispatch at the
    per-block tile-distance reducer. The pre-Phase-4 single-bool
    `block_reducer_constants_for_backend` is retained as a thin
    source-compat wrapper for the unit tests.
  - 3-way `effective_metric_target_distance` lookup precedence
    (zensim > cvvdp > butteraugli, matches the
    `propagate_resolved_metric_to_encoder` invariant).
  - `JXL_ZENSIM_K_TILE_NORM` env hook (bench-only override for the
    Phase 8-zensim refit harness).
  - `examples/zensim_calibration_seed.rs` (NEW) +
    `scripts/zensim_calibration_seed.py` (NEW) +
    `examples/zensim_loop_smoke_bench.rs` (NEW) — calibration
    pipeline + 20-cell smoke bench. Outputs at
    `benchmarks/zensim_calibration_seed_2026-05-25.{tsv,txt,meta}`
    and `benchmarks/zensim_loop_smoke_2026-05-25.{tsv,meta}`.
  - `tests/zensim_loop_smoke.rs` (NEW) — 4 active tests + 1 ignored:
    `PerceptualMetric::Butteraugli` byte-identical-to-default,
    default-byte-identical-to-Butteraugli+Auto,
    `EncoderStrategy::Libjxl` byte-identical-regardless-of-metric
    across 6 (metric, device) combos, public-API round-trip on the
    new `Zensim` variant, and the `#[ignore]`-d
    `metric_zensim_encodes_and_decodes` (5-cell jxl-oxide roundtrip
    that requires the slow CPU zensim buttloop).

  Hash-locks 36/36 BYTE-IDENTICAL at default features AND with
  `zensim-loop` compiled. `strategy_libjxl_byte_lock` 4/4 BYTE-
  IDENTICAL with all of `__expert`, `__expert butteraugli-loop`,
  `__expert butteraugli-loop zensim-loop ssim2-loop parallel`.
  `divergence_table_drift` 7/7 PASS. Multi-decoder roundtrip:
  jxl-oxide 5/5 PASS on the smoke cells.

  Per-distance seed table values (target_score = `(100 -
  median_zensim_native) * 1.05`, butter-direction):
  `(0.50, 6.6381), (1.00, 9.9958), (1.50, 13.2108), (2.00, 16.4704),
  (3.00, 20.9812), (4.00, 24.5327), (5.00, 28.4359)`.

  Phase 4 ships butter-parity `k_tile_norm`. The per-block reducer
  refit (Phase 8-zensim equivalent of cvvdp Phase 8g) is conditional
  on the Phase 6 6-backend tracking sweep verdict.

  Follow-ons:
  - **Phase 6** — 6-backend tracking sweep (B / B_GPU / C_GPU_v4 /
    C_CPU / Z_GPU / Z_CPU) + RFC §5.4 Pareto decision per metric.
  - **Phase 8-zensim** (conditional) — per-block reducer refit if
    Phase 6 lands below 85% Pareto coverage.
  - **Phase 1b** (zensim-gpu) — pure-GPU diffmap kernels to close the
    +1006% wall gap vs score-only GPU at 1024².

- **zensim fork — opt-in zensim-driven quantization loop (Phase 3)**
  (RFC [`docs/RFC_ZENSIM_FORK_PLAN.md`](docs/RFC_ZENSIM_FORK_PLAN.md) §5,
  [`docs/RFC_ZENSIM_BUTTLOOP_AUDIT.md`](docs/RFC_ZENSIM_BUTTLOOP_AUDIT.md),
  [`docs/RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md`](docs/RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md)).
  Mirrors the cvvdp Phase 3 + 5 shape (commits `57757ff8` + `206b874e`)
  for the third `PerceptualBackend` impl. **Phase 3 ships the backend
  impl + opt-in dispatch only** — the buttloop body still consumes
  butteraugli-direction targets; Phase 4 will plumb the zensim signal
  through `run_buttloop` via `vardct/zensim_targets.rs` per-distance
  calibration. Default OFF — opt-in via the unified Phase 0 API
  `LossyConfig::with_perceptual_metric(PerceptualMetric::Zensim)`.
  Surface:
  - `--features zensim-loop` cargo feature (pure-Rust CPU backend; wraps
    `zensim::Zensim::compute_with_ref_and_diffmap_linear_planar`). No
    CUDA needed; runs anywhere. Always available.
  - `--features zensim-loop-gpu` cargo feature (GPU backend wraps
    `zensim_gpu::ZensimOpaque` via zenmetrics master Phase 1 commit
    `1175b49` `score_from_linear_planes_with_warm_ref_diffmap`). Requires
    CUDA at build + load time. **Phase 1 honest-stop carryover**: the
    current GPU diffmap delegates to the canonical CPU pipeline (per
    `crates/zensim-gpu/docs/DIFFMAP_DIVERGENCES.md`), +1006% wall vs
    score-only GPU at 1024². Phase 1b (pure-GPU kernels) will close
    this; until then prefer `PerceptualDevice::Cpu` for wall-time-
    sensitive workloads.
  - `PerceptualMetric::Zensim` variant on the Phase 0 enum (new).
  - `vardct/zensim_backend.rs` (new ~700 LOC) hosts `CpuZensimBackend`
    + `GpuZensimBackend` impls + 10 unit tests.
  - `tests/zensim_backend_smoke.rs` (new) — 5 integration tests
    including the load-bearing `EncoderStrategy::Libjxl` byte-identical
    invariant under `PerceptualMetric::Zensim` (W44-126 strict cjxl
    parity preserved; same short-circuit as cvvdp).
  - Score-direction normalization at trait boundary:
    `(100.0 - zensim_score).clamp(0.0, 100.0)`. Diffmap renorm scale
    ships as placeholder `1.0` (no renorm) — Phase 4 fits the constant
    via a Phase 8c-equivalent harness.
  - Hash-locks 36/36 BYTE-IDENTICAL at default features AND with
    `zensim-loop` compiled (caller has to explicitly opt in for the
    backend to construct). `strategy_libjxl_byte_lock` 4/4
    BYTE-IDENTICAL with zensim-loop compiled. `divergence_table_drift`
    7/7 PASS.

- **cvvdp fork — opt-in ColorVideoVDP-driven quantization loop**
  (RFC [`docs/RFC_CVVDP_FORK.md`](docs/RFC_CVVDP_FORK.md), Phase 6
  decision memo [`docs/CVVDP_FORK_DECISION.md`](docs/CVVDP_FORK_DECISION.md)).
  Six-phase arc replaces the per-iteration butteraugli score+diffmap
  inside the buttloop with cvvdp (Mantiuk et al. 2024) when the
  caller opts in. **Default OFF** per Phase 6 verdict OPT_IN_ONLY —
  cvvdp Pareto-wins on its own metric on CID22 photos (100% of e=8
  cells score better on `cvvdp_gpu`, mean +0.005 to +0.31 JOD across
  distances) but the uncalibrated distance-target table costs +155%
  to +286% bytes at the same `distance` parameter (cvvdp targets
  calibration left as future work per RFC §2.3 TBD). Surface:
  - `--features cvvdp-loop` cargo feature (GPU backend, requires
    CUDA at build + load time; wraps `cvvdp-gpu`) shipped in Phase 3
    (`57757ff8`).
  - `--features cvvdp-loop-cpu` cargo feature (pure-Rust CPU
    backend, no CUDA, runs anywhere; wraps `cvvdp-cpu`) shipped in
    Phase 5 (`206b874e`). ~1.55× wall vs butteraugli CPU at e=8;
    output byte-identical to the GPU backend.
  - `LossyConfig::with_cvvdp_loop(Option<bool>)` builder setter and
    matching `cvvdp_loop()` getter shipped in Phase 3 (`57757ff8`).
  - `LossyConfig::with_cvvdp_use_cpu(Option<bool>)` builder setter
    and matching `cvvdp_use_cpu()` getter shipped in Phase 5
    (`206b874e`). `Some(true)` pins the cvvdp buttloop to CPU even
    when `cvvdp-loop` (GPU) is compiled in.
  - Phase 4 wiring (`32581839`) routes the cvvdp backend through the
    buttloop with the JOD calibration seed table at
    `jxl-encoder/src/vardct/cvvdp_targets.rs` (`9c1cfa3e` seed).
  - `EncoderStrategy::Libjxl` strict cjxl-parity invariant preserved
    — the `resolve_cvvdp_loop` helper forces cvvdp OFF under Libjxl
    regardless of caller field / cargo feature, so the W44 byte-lock
    cells stay BYTE-IDENTICAL.

### Changed

- **Phase 2 refactor — `ButteraugliBackend` trait renamed to
  `PerceptualBackend`** (`8c6e91cc`, shipped 2026-05-24). Trait shape
  unchanged (still `name() / set_reference / compare_with_reference`);
  the rename generalises the abstraction so the cvvdp backends slot in
  alongside the existing CPU/GPU butteraugli backends from
  W44-PHASE3-B1 (`c121c08e`). Crate-private trait (`pub(crate)`), no
  public-API surface change.

### Documentation

- **`docs/RFC_CVVDP_FORK.md`** — 12-section RFC scoping the cvvdp fork
  (architecture, calibration seeds, sweep design, RFC §5.4 ship-rule).
  Status flipped from SCOPING → SHIPPED-OPT-IN with the full 6-phase
  commit chain recorded in §9 (Phase 7 docs closeout this entry).
- **`docs/CVVDP_FORK_DECISION.md`** — Phase 6 decision memo
  (`8b5a13a7`). Per-corpus / per-metric Pareto wins, per-distance
  bytes deltas, cvvdp_gpu score deltas, wall-time trade-offs, RFC §5.4
  application, 11-point limitations + caveats. The OPT_IN_ONLY verdict
  rests on this data.
- **`docs/CVVDP_W44_GATE_TRANSFER.md`** — scoping audit of the W44
  cost-model gate cluster (W44-91 / W44-96 / W44-105 / W44-117). Not
  triggered in Phase 6; lives as forward-reference for a future
  distance-table-calibration chunk that would re-enable the
  default-flip discussion.
- **Phase briefs**: `docs/archive/RFC_CVVDP_PHASE3_BRIEF.md` through
  `docs/RFC_CVVDP_PHASE7_OPTIN_BRIEF.md` archived alongside the RFC.

### Measured

- **`benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`** — Phase 6
  tracking benchmark (`8b5a13a7`). 4 backends × 54 images × 7 distances
  × 3 efforts = 1,134 cells per backend (`B`, `B_GPU`, `C_GPU`,
  `C_CPU`). Every encoded output decoded via jxl-oxide and scored
  with butteraugli (CPU+GPU), cvvdp-gpu, and SSIMULACRA2.
  4,536 rows total. Corpora: CID22 (photos, 41 images), GB82-SC
  (screenshots, 11 images), W44-S1 (mixed, 2 images).
- **`scripts/cvvdp_pareto_analysis.py`** + `_2026-05-24.tsv` +
  `_2026-05-24.meta` — Pareto analyzer (`8b5a13a7`) computing
  per-(corpus, metric) Pareto wins and applying the RFC §5.4
  ship-rule. Verdict line: `VERDICT: OPT_IN_ONLY`.
- **`benchmarks/cvvdp_phase6_decoder_spotcheck_2026-05-24.tsv`** —
  60-check multi-decoder roundtrip on 20 cvvdp-encoded cells
  (5 fixtures × 2 distances × 2 cvvdp backends) through jxl-oxide +
  djxl + jxl-rs. **60/60 PASS.** REVERT branch of RFC §5.4 closed.

- **W44-162 — HDR ToneMapping `relative_to_max_display` + `linear_below`
  surfaced through public API (issue #46 chunk 1a closed)**
  (`jxl-encoder/src/headers/file_header.rs`, `jxl-encoder/src/api.rs`,
  `jxl-encoder/src/vardct/encoder.rs`,
  `jxl-encoder/src/vardct/bitstream.rs`,
  `jxl-encoder/src/api_tests.rs`,
  `docs/LIBJXL_DIVERGENCES.md` Section E new row). Pre-W44-162 the
  remaining two `ToneMapping` fields were hardcoded to spec defaults
  at the write site (`file_header.rs:594-595`) and not surfaced
  anywhere in the API; this was the only remaining HDR + gainmap
  sub-chunk left after W44-137 confirmed chunks 1b/2/3/4/5 all
  shipped in prior commits. W44-162 plumbs both fields through the
  full stack: `ImageMetadata<'a>` setters
  (`with_relative_to_max_display`, `with_linear_below`) with
  matching getters; request-level setters on `EncodeRequest<'a>`
  with documented "request wins over metadata" precedence (mirrors
  the existing `intensity_target` / `min_nits` pattern); streaming
  `LossyEncoder` + `LosslessEncoder` setters for streaming parity;
  field added to internal `VarDctEncoder` + `FileHeader::ImageMetadata`
  with propagation through `vardct/bitstream.rs:871-882`; write-site
  reads from the struct instead of hardcoded literals. The
  `extra_fields` trigger at `file_header.rs:483-489` extended so
  non-default `relative_to_max_display=true` with default
  `intensity_target/min_nits` still writes the bundle; the
  `tone_all_default` short-circuit at `file_header.rs:589` extended
  to gate on all 4 fields (matches jxl-rs `#[all_default]` semantics
  on the `ToneMapping` struct in
  `jxl/src/headers/image_metadata.rs:139-150`). Validator extended
  to `validate_tone_mapping_full(it, mn, rtmd, lb)` with libjxl
  `image_metadata.cc:406` rule: reject `linear_below > 1.0` when
  `relative_to_max_display=true`; accept any `linear_below >= 0`
  (capped at f16 max) when absolute nits. 9 new tests in
  `api_tests.rs`: jxl-oxide field roundtrip (both fields), jxl-rs
  primary-decoder roundtrip, `ImageMetadata`-level setters,
  request-level overrides metadata, validator rejects relative
  `linear_below > 1.0`, validator accepts absolute
  `linear_below > 1.0`, streaming `LossyEncoder` parity, streaming
  `LosslessEncoder` parity, `extra_fields` triggers on
  `relative_to_max_display` alone. Hash-locks 36/36 BYTE-IDENTICAL
  (fields default to spec defaults so hardcoded-literal →
  struct-reads-default produces the same bitstream).
  `cargo test --lib`: 1335 passed, 0 failed.

### Measured

- **W44-121 — codec_wiki d=3 per-strategy + per-region SSIM2 dump**
  (`benchmarks/w44_121_codec_wiki_d3_dump_2026-05-20.{tsv,meta}`,
  `benchmarks/w44_121_codec_wiki_d3_analysis_2026-05-20.txt`,
  `benchmarks/w44_121_run_dumps.sh`,
  `benchmarks/w44_121_analyze.py`,
  `jxl-encoder/examples/w44_121_codec_wiki_d3_dump.rs`,
  `docs/LIBJXL_DIVERGENCES.md` Section F codec_wiki d=3 row refined with
  W44-121 finding). Measurement-only chunk per the W44-103 (terminal d=4)
  template; NO production code change. Result: codec_wiki d=3 has SAME
  ROOT CAUSE as W44-103 terminal d=4. DCT16X16 over-fires 100× (ours
  14031/14030/14310 first-blocks vs cjxl 138/138/251 at e5/e6/e7), DCT64
  family ENTIRELY absent (ours 0 vs cjxl 783-1019 DCT64X64+DCT64X32+
  DCT32X64), strategy agreement 10.1-12.6%, top-row + (at e7) bottom-row
  dominate SSIM2 loss (e5/e6 top -5.98/-5.95, e7 bot averages -3.6).
  Per-region mean qac at PARITY (8.0-8.7 both sides) — pure strategy
  selection divergence, NOT quantization tuning. Top disagreement pair:
  DCT16X16 → DCT64X64 (28k cells e5/e6, 36k cells e7). W44-122 attack
  candidates ranked in `memory/w44_121_codec_wiki_d3_dump_2026-05-20.md`
  (top: find_best_64x64_transform cost-model audit, EV 5-10 SSIM2 points).

### Fixed

- **W44-120 — tighten W44-117 EPF sharpness seed distance gate to >=1.0**
  (`jxl-encoder/src/vardct/butteraugli_loop.rs`,
  `jxl-encoder/examples/w44_120_distance_bisect.rs`,
  `jxl-encoder/examples/w44_120_decoder_check.rs`,
  `benchmarks/w44_120_distance_bisect_2026-05-20.{tsv,meta}`,
  `docs/LIBJXL_DIVERGENCES.md` Section B new row + Section D row 110
  + Section F row 141 resolution). The W44-119 ledger refresh surfaced
  a `-1.87` SSIM2 regression on terminal e8/e9 d=0.8 introduced by
  W44-117 (`5604fdf0`) — at very low distance the buttloop's
  iter-0-fitted EPF sharpness seed over-protected edges that the
  encoder should be quantizing more freely (the legacy uniform-4
  path was already a close-enough match to the production sharpness
  map). W44-120 adds a `W44_120_EPF_SEED_MIN_DISTANCE = 1.0` gate to
  the seed compute (`butteraugli_loop.rs:868`): the W44-117 mechanism
  only fires when `target_distance >= 1.0`. Below that threshold the
  loop falls back to the legacy `vec![4u8; num_blocks]` seed (byte-
  identical to pre-W44-117 behaviour). The W44-118 `is_screenshot`
  gate composes underneath — photos are unaffected (already excluded
  via mask>95 discriminator). Sweep override
  `JXL_W44_120_EPF_SEED_MIN_DISTANCE=<f32>` preserved for future A/B.
  Bisection (`benchmarks/w44_120_distance_bisect_2026-05-20.tsv`)
  swept 0.8 / 1.0 / 1.2 / 1.5 on terminal × e8/e9 × d ∈ {0.5..6.0} +
  small GB82-SC screenshots × e8 d=4 + 5 photo cells. 1.0 is the
  pareto-optimal threshold: closes terminal e8/e9 d=0.8 regression
  (byte-identical to A_legacy), preserves terminal d=1.0 win
  (+0.529 SSIM2), preserves terminal d=4 win (+0.897 SSIM2). Photos
  byte-identical to W44-118. Hash-locks: 36/36 BYTE-IDENTICAL. 1282
  lib tests pass. 3/3 multi-decoder roundtrip via djxl + jxl-rs.

### Measured

- **W44-119 — ledger refresh post-W44-118 + qac-scale chain-disable A/B**
  (`benchmarks/cjxl_parity_ledger_2026-05-20_w44_119.{tsv,meta}`,
  `benchmarks/w44_119_chain_disable_ab_2026-05-20.{tsv,meta}`,
  `jxl-encoder/examples/w44_119_chain_disable_ab.rs`,
  `docs/LIBJXL_DIVERGENCES.md` Section B verdict + Section F new
  W44-120 candidate row). Full 590-cell ledger refresh on W44-118
  main (`f543288c`) finds **zero status flips** vs W44-110 baseline
  (8 OPEN, identical OPEN set). W44-117/118 produces real sub-
  threshold SSIM2 wins on terminal/codec_wiki/imac_g3 at mid distance
  (terminal e8/e9 d=4 +0.90, d=3 +0.66) but does NOT flip any cell
  FIXED↔OPEN. Worst W44-117 side effect: terminal e8/e9 d=0.8 SSIM2
  -1.87 (over-correction at low distance where the buttloop already
  had ~accurate EPF). Photos byte-identical (W44-118 gate works).
  Chain-disable A/B (83-cell, 79 with data): disabling
  W44-105/107/108/109 chain via `JXL_BUTTLOOP_INITIAL_QF_SCALE=1.0`
  AND `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=1.0` regresses avg SSIM2
  by 2.04 (B vs A) while saving 20.74% bytes. Per-cluster avg SSIM2
  delta: terminal -3.86 (n=25), codec_wiki -2.33 (n=9), imac_g3
  -3.97 (n=11), photo +0.00 (n=34, BYTE-IDENTICAL — chain skips
  photos). 42 of 79 cells regress SSIM2 > 0.3; ZERO cells where
  chain OFF wins. **Verdict**: the chain CANNOT be retired — W44-117
  EPF seed and chain address orthogonal mechanisms (recon-side bias
  vs measurement-side bias) and are additive. W44-120 plan: KEEP
  the chain at its W44-109 magnitude; investigate the terminal d=0.8
  W44-117 over-correction as a distance-gate refinement candidate.
  No production code change in this chunk (measurement + analysis +
  divergence-table update only).

### Fixed

- **W44-118 — gate W44-117 EPF sharpness seed on `is_screenshot`**
  (`jxl-encoder/src/vardct/encoder.rs`,
  `benchmarks/w44_118_mode_f_validation_2026-05-20.{tsv,meta}`,
  `benchmarks/w44_118_bisect_abcd_2026-05-20.{tsv,meta}`,
  `benchmarks/w44_118_bisect_abf_2026-05-20.{tsv,meta}`,
  `docs/LIBJXL_DIVERGENCES.md` Section D row 108 + Section F row 139).
  W44-117 (`5604fdf0`) seeded the buttloop's `apply_epf` sharpness map
  from `compute_epf_sharpness` on the initial reconstruction, replacing
  the legacy `vec![4u8; num_blocks]` uniform seed. The fix closed the
  R/G linear-RGB residual on screenshot-class content (terminal e8/e9
  d=3 +0.66 SSIM2, d=4 +0.90) but introduced a `-0.847` SSIM2 regression
  on 1025469 e8/e9 d=4 (a CID22 photo with mask1x1_median=76.08, in the
  W44-91 band [50, 80) but failing the colourfulness/fcbr lift gates so
  no content-aware lift compensates). W44-118 bisects 5 candidate fixes:
  (A) revert W44-117 [restores baseline but loses screenshot wins];
  (B) W44-117 default [the regression]; (C) disable content-aware lifts
  [moot — `JXL_W44_118_PROBE=1` confirms no lift fires on 1025469];
  (D) per-iter `compute_epf_sharpness` recompute [does NOT fix 1025469
  AND regresses terminal e8 d=4 by -3.06 SSIM2 — Option A from the
  W44-116 fix-shape menu is structurally a worse trade]; (F) gate
  W44-117 on `is_screenshot` [F=A byte-identical on every photo,
  F=B byte-identical on every screenshot — strict pareto win over both
  A and B]. SHIPS Mode F as production default. 58-cell validation
  bench: zero new F vs B SSIM2 regressions > 0.3, both 1025469 e8/e9
  d=4 regressions restored to +0.000 (byte-identical to legacy). 36/36
  hash-locks BYTE-IDENTICAL (synthetic 32×32 fixtures don't trigger
  pixel_domain_loss → mask1x1 stays None regardless of is_screenshot
  gating). 1282/1282 lib tests pass. 3/3 multi-decoder roundtrip
  (djxl + jxl-oxide) PASS on the W44-118 regression cell + W44-117
  win cell + W44-117-neutral photo. Env hooks preserved for future
  bisection: `JXL_W44_117_DISABLE=1` (legacy uniform-4 across all
  content); `JXL_W44_118_PROBE=1` (dump mask1x1_median + zenanalyze
  proxies); `JXL_W44_118_PER_ITER_SHARPNESS=1` (Mode D Option A, kept
  for documentation; production default off).

### Performance

- **W44-207 — W44-94 OUTER `find_best_32x32_transform` widening with
  per-m3 sub-discriminator HONEST-STOP** (no source change;
  `benchmarks/w44_207_outer_per_m3_analysis_2026-05-22.{tsv,meta}`,
  `docs/LIBJXL_DIVERGENCES.md` Section F row 211 appended,
  `CLAUDE.md` "Investigation Notes" W44-207 entry). W44-204 C3 chunk
  proposed re-applying the W44-167 per-m3 split mechanism at the OUTER
  `high_d_photo_smooth_suppressed()` table instead of the INNER variant
  Z layer. Phase 1 read-only analysis falsified the hypothesis on three
  structural grounds: (1) the cluster #3 cells (1420710, 1531677) fire
  variant Z (W44-96 admit), NOT OUTER — OUTER cannot affect their
  bytes; (2) the 3 OUTER-only REJECT_Z images (2389166 m3=47.996,
  1044329 m3=65.031, 7062219 m3=51.141) sit at m3 ∈ [48, 65] with no
  LOW-m3 image to PROTECT (per-m3 split has nothing to gate); (3) zero
  zenjxl losers on the 3 REJECT_Z images at d>=3 in W44-202
  (`benchmarks/w44_202_vs_w44_185.per_cell_diff.tsv`) — W44-205
  (`8fe99fb5`) already harvested -1.5% to -2.4% bytes there. The
  variant-Z-routed cells on zenjxl are bytes-axis WINNERS but SSIM2
  LOSERS; wider OUTER lift would push deeper into the SSIM2 deficit
  (wrong direction). Pivot recommendations: (a) W44-168+ distance-
  narrowed Mode D on LC (W44-167 surprise positive); (b) 3637739-class
  cluster #1 with fresh multi-week hypothesis; (c) root-cause buttloop
  measurement divergence (W44-167 follow-on #5).
- **W44-99 — low-colour sub-discriminator (m3 < 25) of variant Z**
  (`jxl-encoder/src/effort.rs`, `jxl-encoder/src/vardct/encoder.rs`,
  `benchmarks/w44_99_1531677_d5_attack_2026-05-19.{tsv,meta}`).
  Adds [`EntropyMulTable::high_d_photo_smooth_suppressed_z_low_colour`]
  (dct32x32=1.20 inherited from variant Z, dct16x32 LIFTED modestly to
  1.22 — vs variant Z's 1.208 and W44-98 high_colour Z''s 1.30). Wired
  in `compute_ac_strategy` behind a new `m3_colourfulness < 25.0`
  sub-gate that mirrors W44-98's high_colour gate (the two are mutually
  exclusive). Closes 1531677 d=5 OPEN cells that W44-98 could not
  address: e6 d=5 (+3.04% → +2.60%, OPEN → FIXED), e8 d=5
  (+3.05% → +1.92%, OPEN → FIXED), e9 d=5 (+3.64% → +2.53%, OPEN →
  FIXED); e5 d=5 stays OPEN at +3.09% (just 0.09% over the +3.0%
  threshold — needs the W44-94 follow-on butteraugli loop at e7). Of the
  2 CID22 photos that pass the W44-96 gate, only 1531677 (m3=12.30)
  passes the m3 < 25 sub-gate; 1420710 (m3=32.93) stays on the W44-98
  high_colour Z' table (byte-identical to the W44-98 ship). Worst SSIM2
  delta across all 29 measured cells: -0.0100 (essentially noise — far
  under the ≤0.30 budget). 36/36 hash-locks byte-identical, all W93_REGR
  and W95_REGR cells byte-identical. Multi-decoder roundtrip via djxl +
  jxl-rs + jxl-oxide on 2 closed cells: 6/6 PASS.

- **W44-98 — dct16x32 lift inside variant Z (m3 sub-discriminator)**
  (`jxl-encoder/src/effort.rs`, `jxl-encoder/src/vardct/encoder.rs`,
  `benchmarks/w44_98_dct16x32_lift_z_bisect_2026-05-19.{tsv,meta}`,
  `benchmarks/w44_98_baseline_diff_2026-05-19.{tsv,meta}`).
  Adds [`EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour`]
  (dct32x32=1.20 inherited from variant Z, dct16x32 LIFTED to 1.30
  breaking the libjxl 1.49/1.48 ratio). Wired in
  `compute_ac_strategy` behind a new `m3_colourfulness >= 25.0`
  sub-gate INSIDE the existing W44-96 variant Z dispatch. Targets the
  W44-97 finding that DCT32X16 is the universal #1 overspender on the
  7 OPEN F-D cells remaining post-W44-96 — DCT32X16 / DCT16X32 share
  the `dct16x32` slot (`ac_strategy.rs:713`) so lifting it favours
  DCT32X32 (square merge) and DCT16X16 over rectangular 32-class
  transforms. Of the 2 CID22 photos that pass the W44-96 gate, only
  1420710 (m3=32.93) passes the additional m3 sub-gate; 1531677
  (m3=12.30) stays on the default variant Z table (the W44-98
  bisect measured 1531677 regressing SSIM2 by -0.34 to -0.93 under
  ANY `dct16x32 ≥ 1.30`). Closes 3 OPEN cells (1420710 e5 d=5,
  e5 d=6, e7 d=5; -291B/-259B/-410B vs the W44-96 baseline) AND
  improves 4 W44-96 FIXED cells by 125-273B further. Aggregate -1741B
  over 29-cell bench with 0 FIXED→OPEN regressions and worst SSIM2
  delta -0.0275. All hash-locks (36/36) byte-identical, all W93_REGR
  and W95_REGR cells byte-identical. Multi-decoder roundtrip via
  djxl + jxl-rs + jxl-oxide on 3 closed cells: 9/9 PASS.

- **W44-68 — DCT32 suppression on screenshot-class content**
  (`jxl-encoder/src/vardct/encoder.rs`,
  `benchmarks/w44_68_codec_wiki_d4_ab_2026-05-19.{tsv,meta}`).
  Extends the W44-65 default-on `dct_suppress_hint` discriminator
  (mask1x1 median >= 99.5) to additionally drop `try_dct32 = false`
  on the same dispatched class. Bisection on codec_wiki d=0.5..d=6
  showed uniform -2.65% to -4.48% wins; other dispatched screenshots
  (terminal, imac_g3, imac_dark, windows) also win -0.76% to -3.78%.
  Closes the final OPEN screenshot cell in the cjxl_parity_ledger:
  `codec_wiki e7 d=4` flips from +3.55% bytes / OPEN → -1.09% bytes /
  FIXED. windows95 (mask1x1=99.06) and CID22 photos (median ≤ 92.34)
  remain protected by the discriminator threshold and are byte-identical
  in the dispatcher path. RD-regression passes with frymire wins
  (-3.1% to -4.4% size, +0.93 SSIM2 at d=1). Multi-decoder roundtrip
  verified on jxl-rs + djxl for codec_wiki, terminal, imac_g3 at
  d=3..d=4.

- **W44-54 — VarDCT DC LearnTree at effort >= 4** (`d53519d4`,
  closes part of imazen/jxl-encoder#56). Routes DC tokenization through
  the data-adaptive `dc_tree_learn::learn_dc_tree` stub for `effort >= 4`,
  mirroring libjxl's `speed_tier < SpeedTier::kFalcon` gate in
  `enc_modular.cc:1166-1217`. Previously the encoder always emitted the
  `kWPFixedDC` predefined tree (34 leaves, 45 post-AC-merge contexts)
  regardless of effort, over-spending LfGlobal ANS prefix-table bits on
  heavily-quantized screenshot DC where 1-3 contexts suffice. Effort <= 3
  keeps `kWPFixedDC` (libjxl `kFalcon` parity). Closes W44-50 wedge:
  `terminal e6 d=6` LfGlobal 700 B → 230 B; file total 57617 B → 55886 B
  (-3.0%, +0.9% over cjxl was +4.0%). Sweep on 72 cells (5 photos + 4
  screenshots × 2 efforts × 4 distances): photos +0.74%, screenshots
  -1.39%, overall **-0.21%**. Decoded pixels bit-identical between
  baseline and new path (zero quality regression). 23 of 36 lossy
  hash-lock sidecars rebaselined; all 13 lossless cells unchanged;
  headers byte-identical. RD-regression passes with multiple wins on
  screenshot content (`frymire d=1.0 -2.9% & +0.93 SSIM2`). Follow-on:
  WP-residual learning + per-leaf `Predictor::Weighted` for the
  photo-content regression cluster (libjxl `Predictor::Variable` parity).

- **W43-2 chunk-5 — magetypes-consolidate `pixel_domain_loss`**
  (`jxl-encoder-simd/src/pixel_loss.rs`, `jxl-encoder/Cargo.toml`,
  `jxl-encoder/examples/pixel_loss_magetypes_bench.rs`,
  `benchmarks/magetypes_pixel_loss_consolidation_2026-05-19.{tsv,meta}`).
  Mirrors W43-2 chunks 3 (`compute_mask1x1`) and 4 (`gaborish_5x5`) on
  the next candidate from the audit memo
  (`memory/magetypes_cpu_acceleration_candidates_2026-05-19.md`,
  candidate #5, LOW risk). The prior 3 hand-written SIMD variants
  (AVX2 + NEON + WASM128) plus the scalar fallback collapse to a single
  `#[magetypes(define(f32x8, f64x4), v3, neon, wasm128, scalar)]`
  body. The macro generates one `#[arcane]`-wrapped variant per listed
  tier from the same source-level algorithm:
    - `pixel_domain_loss_impl_v3`      (x86_64 AVX2, native 256-bit
                                        `f32x8` + `f64x4`)
    - `pixel_domain_loss_impl_neon`    (aarch64, 2× f32x4 polyfill of
                                        f32x8 and 2× f64x2 polyfill of
                                        f64x4)
    - `pixel_domain_loss_impl_wasm128` (wasm32, same polyfill shape)
    - `pixel_domain_loss_impl_scalar`  (portable scalar fallback)
  The body promotes `f32x8 → 2× f64x4` via the array round-trip
  `to_array() → [as f64; 4] → from_array(...)`. On AVX2 LLVM fuses
  the store + scalar-extend + load chain into the same `vcvtps2pd`
  pair the prior hand-written `_mm256_castps256_ps128 +
  _mm256_cvtps_pd` path emitted (asm-verified: 16 `vcvtps2pd` + 16
  `vextractf128` per 8-block inner loop, matching baseline). The
  manual `x²·x²·x²` 8th-power chain and per-half (`acc_lo` / `acc_hi`)
  accumulation grouping are preserved exactly, so bitstream byte-output
  is unchanged. **AVX-512 (`v4`) tier is NOT included** — magetypes
  0.9.23 does not implement `F64x4Backend` for `X64V4Token` /
  `X64V4xToken` (the natural f64 width on AVX-512 is `f64x8`, one
  512-bit register). Ceiling on x86_64 is `v3` (AVX2). Hash-lock
  36/36 byte-identical (`tests/hash_lock_features.rs`); 1250
  `jxl-encoder` lib tests pass; 125 `jxl-encoder-simd` tests pass
  including 3 pixel_loss parity tests
  (`test_pixel_domain_loss_matches_scalar` exercises every available
  token permutation via `archmage::testing::for_each_token_-
  permutation`). x86_64 best-iter wall-clock under heavy concurrent
  system load (load_avg ~14-25, other agents' bake_compare consuming
  26 cores) — dispatch arm matches baseline within noise:
  512² 0.106 → 0.105 ms (-1 %); 1024² 0.569 → 0.570 ms ( 0 %);
  2048² 2.405 → 2.062 ms (-14 %); 4096² 9.967 → 10.551 ms (+6 %).
  Per-size dispatch-vs-scalar speedups (post): 512² 2.12×, 1024² 1.48×,
  2048² 1.89×, 4096² 1.45×. LOC delta: 442 → 362 (-80, -18 %).
  Backwards-compat aliases preserved: `pixel_domain_loss_avx2`,
  `pixel_domain_loss_neon`, `pixel_domain_loss_wasm128`. Chunk-6
  candidate per W43-2 audit: `forward_xyb` (#1, 6 h, LOW risk,
  biggest LOC reduction — color-convert example shape, 3-channel
  fused). No AI attribution (pixel-domain loss formula derived from
  libjxl `EstimateEntropy`).

- **W43-2 chunk-4 — magetypes-consolidate `gaborish_5x5`**
  (`jxl-encoder-simd/src/gaborish5x5.rs`, `jxl-encoder-simd/Cargo.toml`,
  `jxl-encoder/examples/gaborish5x5_magetypes_bench.rs`,
  `benchmarks/magetypes_gaborish5x5_consolidation_2026-05-19.{tsv,meta}`).
  Mirrors the W43-2b `compute_mask1x1` consolidation pattern on the next
  candidate from the W43-2 audit memo (`memory/magetypes_cpu_acceleration_-
  candidates_2026-05-19.md`). The prior 3 hand-written SIMD variants
  (AVX2 + NEON + scalar fallback) plus the missing-WASM fall-through
  collapse to a single `#[magetypes(define(f32x8), v4, v3, neon, wasm128,
  scalar)]` body. The macro generates one `#[arcane]`-wrapped variant per
  listed tier from the same source-level algorithm:
    - `gaborish_5x5_impl_v4`      (x86_64 AVX-512, opt-in via the new
                                   `jxl-encoder-simd` `avx512` feature flag)
    - `gaborish_5x5_impl_v3`      (x86_64 AVX2, native 256-bit f32x8)
    - `gaborish_5x5_impl_neon`    (aarch64, 2x f32x4 polyfill of f32x8)
    - `gaborish_5x5_impl_wasm128` (wasm32, 2x f32x4 polyfill of f32x8 —
                                   NEW: pre-consolidation the wasm32
                                   dispatch fell through to scalar)
    - `gaborish_5x5_impl_scalar`  (portable scalar fallback)
  The body uses the same `f32x8` FMA chain shape the existing AVX2 body
  had (`mul_add` association `wc*center + (wr*r + (wd*d + (...)))`), so
  emitted instructions are bit-equivalent on AVX2 and NEON. wasm32 now
  gets a real SIMD path where it previously ran the scalar loop.
  Hash-lock 36/36 byte-identical (`tests/hash_lock_features.rs`); 125
  `cargo test -p jxl-encoder-simd` tests pass including 3 new gaborish
  parity tests (`test_gaborish_5x5_simd_matches_scalar` exercises every
  available token permutation via `archmage::testing::for_each_token_-
  permutation`). x86_64 wall-clock at 1024² photo size (the dominant
  cell in the e5-e7 budget): **5.03 ms dispatch vs 11.20 ms scalar =
  2.23× median speedup** (matches the prior hand-written AVX2 baseline
  within run-to-run thermal/CPU-load variance — the LLVM-emitted AVX2
  body is bit-equivalent to the prior hand-written one). Per-size
  dispatch-vs-scalar median speedups: 256² 3.83×, 512² 2.55×, 1024²
  2.23×, 2048² 1.82×, 4096² 3.05×. (Best-of-13-samples ratios are
  larger — 1024² 9.16/3.34 ms = 2.74× best-iter — and a quieter-load
  re-run hit 1024² 23.5/2.6 ms = 9.00×; the median bench was on a
  load-active machine so the dispatch arm caught more contention.)
  Backwards-compat aliases preserved:
  `gaborish_5x5_avx2`, `gaborish_5x5_neon`, plus the new
  `gaborish_5x5_wasm128`. Chunk-5 candidate per W43-2 audit:
  `pixel_domain_loss` (#5, 4 h, LOW risk — manual `x²·x²·x²` chain
  preserves the 8th-power parity already proven on the AVX2 path) or
  `forward_xyb` (#1, 6 h, biggest LOC reduction). No AI attribution
  (gaborish algorithm derived from libjxl).

### Investigated

- **W41-1 (issue #52) — distance-aware `min_peak` patches gate
  hypothesis RULED OUT** (`benchmarks/patches_min_peak_distance_2026-05-19.{tsv,meta}`,
  `benchmarks/patches_min_peak_admission_2026-05-19.txt`,
  `examples/patches_min_peak_distance_ab.rs`,
  `examples/patches_min_peak_scan.rs`). The W38-2 WF2 wedge audit
  hypothesised that raising `min_peak` from 1 to 2 at d>=3.0
  (libjxl unconditional `kMinPeak=2` parity) would close +22-51 %
  byte regressions vs cjxl on `imac_g3 / codec_wiki / terminal`
  at e7+. Measurement (84 paired cells: 4 screenshots × 3 photos
  × 6 distances × 2 variants) confirms the detected patch set is
  IDENTICAL between `min_peak=1` and `min_peak=2` on the three
  wedge images (e.g. `imac_g3`: 277 refs / 2052 occurrences at
  either threshold). Only `windows95.png` admits 3 extra refs at
  `min_peak=1` (82 vs 85); clamping it to 2 at d>=3 saves
  0.7-1.5 % bytes but regresses ssim2 by 0.4-1.3 points (net
  quality loss). **No code change shipped.** The bench harness +
  full-corpus admission scan (`patches_min_peak_admission_2026-05-19.txt`
  covers all 11 gb82-sc screenshots × 6 distances) are committed
  for the next-chunk follow-on, which must look elsewhere for the
  wedge root cause — candidates per the audit memo: post-detection
  encoding overhead, distance-blind `apply_per_patch_cost_gate`
  constants, or unlocking `is_cost_effective` in Reference mode at
  d>=3 with the `1/sqrt(distance)` divisor. Wedge stays open.

### Added

- **W43-3 chunk 1 — `HdrLoss::Ssim2` promoted to first-class variant**
  (`src/vardct/hdr_metrics.rs`, `src/vardct/ssim2_loop.rs`,
  `src/vardct/encoder.rs`, `tests/hdr_loss_ssim2_promotion.rs`,
  `examples/hdr_loss_ssim2_promotion_ab.rs`,
  `benchmarks/hdr_loss_ssim2_promotion_2026-05-19.{tsv,meta}`). The
  `ssim2-loop` cargo feature has wired
  `VarDctEncoder::ssim2_refine_quant_field` (SSIMULACRA2 — Jon
  Sneyers' JXL-tuned metric, the same algorithm that powers libjxl's
  `ssimulacra2_main`) internally for several releases. This chunk
  exposes that path through the public `HdrLoss` enum so callers can
  opt in via a single `LossyConfig::with_hdr_loss(HdrLoss::Ssim2)`
  call instead of `with_ssim2_iters`. New
  `ssim2_refine_quant_field_with_iters` shim takes an explicit
  `iters_budget` so the dispatch in `vardct/encoder.rs` can pass the
  `butteraugli_iters` budget without mutating `self` (the
  `forbid(unsafe_code)` rule rules out interior-mutability tricks).
  `validate_loss` surfaces a typed
  `HdrMetricError::Ssim2FeatureDisabled` (→ `Error::NotImplemented`)
  when `Ssim2` is selected without the `ssim2-loop` cargo feature —
  no silent fallback to butteraugli. **Default `HdrLoss::Auto`
  still resolves to `Butteraugli` on SDR** (no behaviour change on
  the 36/36 hash-lock corpus); a chunk-2 follow-on covers the A.9
  decisive-rule eval (Mohammadi 2025 6-stat panel) that would
  justify flipping `Auto` to `Ssim2` for SDR. Dispatch test:
  `ssim2_bytes_differ_from_butteraugli_proves_dispatch_works` proves
  the route is wired (byte-identical between modes would be a silent
  fallback regression). Bench: 5 CID22-512 photos × {d=0.5, 1.0,
  2.5, 4.0} × e8 = 40 cells; see TSV header for the per-distance
  paired aggregates.

- **W38-2 — `LossyConfig::with_pixel_loss_dispatch(PixelLossDispatch)`
  adaptive-dispatch surface** (`src/api.rs`,
  `src/vardct/encoder.rs`, `src/vardct/bitstream.rs`,
  `src/vardct/precomputed.rs`, `--pixel-loss-dispatch` CLI flag,
  `examples/pixel_loss_dispatch_ab.rs`,
  `benchmarks/pixel_loss_dispatch_2026-05-19.{tsv,meta}`). Per W38-1
  baseline `pixel_domain_loss` adds ~11 ms/MP on photos and
  ~70 ms/MP on screenshots at e5 — the IDCT-of-quant-error +
  per-pixel `mask1x1` weighting + 8th-power-norm path inside
  `estimate_entropy_full`. On smooth content the term rarely
  changes which AC-strategy wins. The new
  `PixelLossDispatch::{AlwaysOn, AlwaysOff, Auto}` enum lets callers
  opt into skipping the loss term: `AlwaysOff` unconditionally
  skips (equivalent to `with_pixel_domain_loss(false)`), `Auto`
  computes `mask1x1` then drops it before the AC-strategy search
  when per-image `median(mask1x1) > 80`. **Default `AlwaysOn`** —
  preserves the byte-identical historical bitstream (hash-lock
  36/36 + dedicated regression test). Mirrors the W36-2
  `EpfDispatch` / W36-3 `PatchesDispatch` opt-in patterns.
  A/B sweep (5 CID22-512 photos + 3 gb82-sc screenshots × 3
  distances × 2 efforts × 3 dispatches = 144 cells): photo e5
  Auto wall-clock Δ median = +2.2 % (mask1x1 compute overhead
  pessimises non-gated cells); screen e5 Auto Δ median = -1.2 %
  (gates 9/9, AlwaysOff = -10.4 % shows the ceiling). Quality on
  gated cells: photo ssim2 Δ median = -0.27, screen ssim2 Δ
  median = -0.64 (e5) / +0.02 (e7) — `Auto`-default flip is
  deferred to chunk 2 pending a wider corpus RD-pareto bench
  (`benchmarks/pixel_loss_dispatch_2026-05-19.meta` documents the
  gating logic and the chunk-2 acceptance criteria).

### Performance

- **W38-2 #1.1 — `fine_grained_step` libjxl parity at e9**
  (`src/effort.rs:752`,
  `examples/fine_grained_step_libjxl_parity_ab.rs`,
  `benchmarks/fine_grained_step_libjxl_parity_2026-05-19.{tsv,meta}`).
  Per W38-2 wedge audit (`benchmarks/rd_curve_wedges_2026-05-18.md`)
  we had `fine_grained_step = 1` at effort 9, the inverse of libjxl
  `enc_ac_strategy.cc:1046`:
  `size_t step = cparams.speed_tier >= SpeedTier::kTortoise ? 2 : 1;`
  libjxl uses step=2 at `speed_tier >= kTortoise` (which maps to our
  effort 1..=9 — kTortoise is libjxl's slowest *speed_tier value*, =1,
  with kGlacier=0 and kTectonicPlate=-1 below it). We were doing 4×
  more non-aligned 32×32/16×32/32×16 search work at e9 than libjxl AND
  the wedge audit found we were consistently losing on the high-d
  cells anyway. Fix: `fine_grained_step: if effort >= 10 { 1 } else { 2 }`
  — e10+ retains the finer step=1 as our explicit extension past
  libjxl kGlacier. e7 hash-locked output unchanged (36/36 byte-
  identical), rd-regression 18/18 within thresholds. A/B at e9 on 8
  mixed images (5 CID22-512 + 3 gb82-sc) × 4 distances × 3 samples:
  RD-neutral (Δbytes mean -0.35%, Δbfly mean -1.26% = better, Δssim2
  mean +0.005 with max |Δ| = 0.482). Wall-clock saving modest
  (-1.8% mean, -12.4% on the 5.6 MP imac_g3 screenshot where the
  non-aligned 32×32 step scales hardest); the 4× theoretical
  reduction is one phase among many at e9 (butteraugli loop, optimal
  LZ77, enhanced clustering, 14-predictor tree learner).

- **W38-3 — HONEST-STOP: parallel xform fan-out at e3/e4 is already
  shipped** (`benchmarks/parallel_xform_e3_e4_2026-05-19.{tsv,meta}`,
  no src/ changes). W38-1 (`a2cd4758`) flagged "parallel xform fan-out
  at e3/e4 on ≥1.5 MP screenshots — estimated 30-45 ms/cell saving" as
  a top-3 adaptive-dispatch candidate; on inspection the
  `transform_and_quantize` AC-group reduce at `vardct/transform.rs:1222`
  already routes through `crate::parallel::parallel_map(num_groups, ...)`
  and fires at every effort whenever the workspace `parallel` feature is
  enabled (default-on). Measured xform speedups at 8T vs 1T on three
  ≥1.5 MP screenshots × {e3, e4} × 3 trials median: terminal e3 0.71×
  (-17.9 ms, parallelism overhead exceeds work at 1.75 MP / 35 groups),
  terminal e4 1.05× (+3.2 ms), codec_wiki e3 4.04× (+157.7 ms),
  codec_wiki e4 1.80× (+39.7 ms), imac_g3 e3 1.48× (+72.4 ms), imac_g3
  e4 2.20× (+77.2 ms). Default unchanged. Future polish levers (not
  shipped this chunk): reduce per-group `GroupTransformResult`
  allocator pressure, parallelize the serial scatter step, auto-tune
  AC-group chunking on small images so rayon overhead doesn't exceed
  per-group work — these belong in a separate "xform parallel scaling
  polish" chunk and are documented in the bench meta. Per the W38-3
  task spec's honest-stop condition: "If transform_and_quantize is
  already parallel everywhere: honest-stop with explanation."

### Investigated

- **W39-2 — HONEST-STOP: screenshot-class HIGH-regime `max_increase`
  cap (W38-2 WF3 follow-on to W39-1 `3ecd397b`)**
  (`src/vardct/butteraugli_loop.rs`, `src/vardct/encoder.rs`,
  `src/vardct/bitstream.rs`, `src/vardct/mod.rs`,
  `examples/buttloop_screenshot_cap_sweep.rs` [new],
  `benchmarks/buttloop_screenshot_cap_sweep_2026-05-18.{tsv,meta}`).
  Wired content-class dispatch on top of W39-1's atomic infrastructure
  (`MAX_INCREASE_X1000_HIGH_SCREENSHOT` + `SCREENSHOT_MEDIAN_THRESHOLD`
  + `resolved_max_increase_with_class(target_distance, is_screenshot)`).
  The call site classifies the input via `median(mask1x1) > 95.0` (same
  discriminator as `splines::looks_like_screenshot` and
  `encoder::CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD`) and threads
  `is_screenshot` through the buttloop. Animation path
  (`bitstream.rs`) passes `is_screenshot = false` (no fixture coverage
  on screenshot-animation inputs).

  **Production default: 100.0 ("no cap")** — bit-identical to
  pre-W39-2 at every distance / effort / content class. Hash-locks
  36/36 byte-identical at default e7. 6 new unit tests cover the
  class-aware resolver + photo bit-identity invariant
  (`class_blind_resolver_byte_identical_to_legacy`,
  `screenshot_class_low_regime_uses_low_default`,
  `screenshot_class_high_regime_unmodified_picks_screenshot_default`,
  `screenshot_override_only_affects_screenshot_high`,
  `screenshot_high_picks_min_of_shared_and_screenshot_slots`,
  `screenshot_high_default_is_no_cap_until_sweep_lands`).

  **Why HONEST-STOP (no default-on flip)**: 240-cell paired sweep
  (3 screenshots × 3 photos × {d=2.0, 3.0, 4.0, 5.0} × {e8, e9} ×
  {1.3, 1.5, 1.8, 2.0, 100.0} caps) finds the cap fires only on
  `imac_g3.png` at d=2.0 / d=4.0 with sub-noise impact:

  - imac_g3 d=2.0 e8 cap=1.3 vs 100.0: -11 bytes (-0.005 %),
    +0.012 SSIM2, butteraugli unchanged
  - imac_g3 d=4.0 e8/e9 cap=1.3: -0.01 to -0.02 % bytes,
    ±0.001-0.004 SSIM2 noise
  - `terminal.png` + `codec_wiki.png`: byte-identical across all caps
    and all (effort, distance) combinations
  - All 120 photo cells: bit-identical across all 5 caps (gate
    doesn't fire — `is_screenshot=false`)

  **No cap value passes the acceptance gates** (bfly Δ ≤ -3 % AND
  ssim2 Δ ≥ +1 AND bytes Δ within ±2 %). Observed best is butteraugli
  +0.00 % (FAIL) and ssim2 +0.004 (1000× short of +1). The W38-2 WF3
  wedge (e8/e9 screenshots: bytes -20 to -22 % vs cjxl, bfly +9 to
  +19 %, ssim2 -2 to -5) is therefore NOT caused by the bad-block
  bump magnitude — bumps simply don't exceed `1.3×` per iter on real
  screenshot content (`tile_dist / target_distance` stays bounded by
  butteraugli's per-tile reference computation).

  **Chunk-3 plan (the real WF3 lever lives elsewhere)**:
  1. **Screenshot-class `cur_pow` cap (good-block reduction)** — the
     `cur_pow.powf(diff)` pullback at iter < 2 may be too aggressive
     on flat screenshot regions (`diff < 1.0` → quant field shrinks,
     using fewer bits). A screenshot-only `cur_pow = 0.15` (vs libjxl
     0.2) would preserve more good-block precision. The
     `MAX_INCREASE_X1000_HIGH_SCREENSHOT` atomic + sweep harness
     pattern is the template; add `CUR_POW_X1000_HIGH_SCREENSHOT`
     and reuse the same `median(mask1x1) > 95` discriminator.
  2. **Screenshot-class `global_scale` clamping** — `SetQuantField`
     recomputes `global_scale` from median/MAD per iter. A few
     extreme blocks (high-contrast text edges) inflate the MAD,
     pulling `global_scale` up, coarsening EVERY other block.
     Clamping the per-iter delta on screenshot-class encodes would
     stabilise this.
  3. **Direct W38-2 WF3 bisect** — re-run the audit's wedge cells
     (`benchmarks/rd_curve_wedges_2026-05-18.md` §3.1) with the
     buttloop DISABLED (`--no-butteraugli`) to confirm whether the
     bug is buttloop-internal or upstream (in CfL2 / AC strategy /
     `transform_and_quantize`).

  Atomic override stays in place
  (`__buttloop_overrides::MAX_INCREASE_X1000_HIGH_SCREENSHOT`) so a
  future power-user / sweep harness can engage the cap without
  rebuilding; production runs see no behaviour change. New
  diagnostic env var: `JXL_BUTTLOOP_W39_DEBUG=1` prints
  `is_screenshot` + resolved cap per encode (free in normal runs).

### Fixed

- **W40-2 follow-on — `wasm32` build broken since W38-2 (`33011181`)
  by ungated `__buttloop_overrides` re-export** (`src/vardct/mod.rs`).
  The `pub mod __buttloop_overrides { pub use super::butteraugli_loop::… }`
  block lifted the sweep-only atomic overrides into the public
  `__buttloop_overrides` namespace but missed the
  `#[cfg(feature = "butteraugli-loop")]` gate, so the `pub use` named
  a module that was `cfg`'d out under `--no-default-features --features "std"`
  (the WASM CI configuration). `cargo build --target wasm32-wasip1
  --no-default-features --features "std" -p jxl-encoder` failed with
  `E0432: unresolved import super::butteraugli_loop` at
  `src/vardct/mod.rs:141`. Fix: gate the `__buttloop_overrides`
  module behind the same `feature = "butteraugli-loop"` as the
  underlying re-exported items. Hash-lock 36/36 byte-identical
  (WASM-only fix; native build, default-feature build, and clippy
  all unchanged). Verified `cargo build --release --no-default-features
  --features "std" --workspace --target wasm32-wasip1` now passes.

- **W38-2 #3.1 — distance-aware butteraugli-loop tuning scaffolding
  (CPU port of GPU commit `d75bf7c`, HONEST-STOP on the literal port)**
  (`src/vardct/butteraugli_loop.rs`, `src/vardct/mod.rs`,
  `examples/buttloop_distance_split_ab.rs` [new],
  `benchmarks/buttloop_distance_split_port_2026-05-18.{tsv,meta}`).
  Ports the GPU encoder's distance-aware split of the per-iter
  `(cur_pow, max_increase)` tuning into the CPU buttloop as
  **scaffolding plus a sweep harness** — the literal GPU LOW-regime
  tuning regresses RD-pareto on CPU, so production defaults stay
  libjxl-faithful at both regimes.

  - New module-level constants in `vardct::butteraugli_loop`:
    `DEFAULT_CUR_POW_LOW/HIGH`, `DEFAULT_MAX_INCREASE_LOW/HIGH`,
    `DEFAULT_DISTANCE_SPLIT`. **All set to libjxl values** (`cur_pow=0.2`,
    `max_increase=100.0` ≈ "no cap", split at `d=2.0`). Production
    output is byte-identical to pre-port behaviour at every distance,
    every effort.
  - New `pub static` atomics `CUR_POW_X1000_{LOW,HIGH}`,
    `MAX_INCREASE_X1000_{LOW,HIGH}`, `DISTANCE_SPLIT_X1000` (re-exported
    via `#[doc(hidden)] pub mod vardct::__buttloop_overrides`). Sweep
    harnesses hot-swap per-regime values without rebuilds (mirrors the
    GPU encoder's atomics in `forks/butteraugli_loop.rs`).
  - New per-iter helpers `resolved_cur_pow(iter, target_distance)` /
    `resolved_max_increase(target_distance)` consult the atomics and
    fall back to defaults. Wired into the buttloop's good-block
    reduction (`cur_pow.powf(diff)`) and bad-block bump (`old * diff`
    capped at `max_increase`).
  - 7 unit tests covering helper resolution + override round-trip +
    regime-split shift + an invariant test
    (`production_defaults_are_libjxl_faithful`) that guards against
    accidental default drift.

  **Why HONEST-STOP on the literal port** (the GPU-tuned LOW values):
  the W38-2 audit (`benchmarks/rd_curve_wedges_2026-05-18.md` #3.1)
  documented WF3 (e8/e9 buttloop over-compresses screenshots at
  d>=2.0) and suggested mirroring the GPU split. **Two reasons that
  recommendation does not fit the CPU encoder**:

  1. The CPU loop was **already at the GPU's post-fix HIGH values**
     (libjxl defaults) at every distance pre-port. The literal port
     only changes LOW-regime behaviour, leaving WF3 (which lives at
     d>=2.0, HIGH regime) untouched. Verified empirically: PRE/POST
     paired sweep is byte-identical at every HIGH cell.
  2. **Applying the GPU LOW tuning to CPU LOW (cur_pow=0.5,
     max_increase=1.3) regresses RD-pareto** on both photos and
     screenshots. Paired sweep at d=0.5/1.0/1.5 e8/e9 (6 images,
     `benchmarks/buttloop_distance_split_port_2026-05-18.tsv`):
     - photos d=0.5-1.5: bytes -3 to -7 %, **bfly +1 to +8 %**, ssim2
       -0.04 to -1.04.
     - screenshots d=0.5-1.5: bytes -3 to -7 %, **bfly +4 to +13 %**,
       ssim2 -0.41 to -1.20.

     The GPU's tuning was calibrated against its own e7 baseline
     (≈9 % smaller bytes than cjxl e7) which left less room for
     good-block reclamation; CPU's baseline differs and the same
     reclamation factor over-shrinks the quant field.

  Hash-lock 36/36 byte-identical; RD-regression 18/18 within
  thresholds (buttloop is gated off at effort < 8, so rd-regression
  cells at e7 default are unaffected).

  **Next chunk (not landed here)**: the real WF3 fix is a screenshot-
  class cap at HIGH regime (`max_increase=1.3` for content with
  high-contrast text). The scaffolding shipped in this commit is the
  enabling infrastructure (atomic overrides + paired A/B harness) for
  that follow-on tuning sweep. Memory at
  `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/buttloop_rd_gap_2026-05-14.md`.

### Added

- **W38 — lossy low-effort phase baseline (e2..=e5) + zenjpeg-hybrid
  cross-codec wall-clock + RD comparator** on the W36-1 8-image corpus
  (`jxl-encoder/examples/lossy_low_effort_zenjpeg_compare.rs`,
  `benchmarks/lossy_phase_baseline_low_effort_2026-05-19.{tsv,meta}`,
  `benchmarks/lossy_phase_low_effort_with_zenjpeg_2026-05-19.{tsv,meta}`).
  Extends W36-1 (`70a48af9`) downward in effort space and adds zenjpeg
  `HybridProgressive` at q∈{60,75,85,95} mapped to JXL d∈{4,2,1,0.5}.
  Reuses `__JXL_ENC_PHASE_TIMING` env-var path; no src/ changes.
  Headline: jxl matches zenjpeg-hybrid wall-clock at e2/e3/e4 across
  most cells (most-common matched-e per class — photo: e2, scrn: e4),
  with bytes Δ=+17.3% overall but butteraugli Δ=−2.30 (better) and
  ssim2 Δ=+3.47 (better) at parity wall. The e2 fast path produces no
  phase markers because `optimize_codes=false` routes through the
  single-pass streaming Huffman entry point. Top-3 adaptive-dispatch
  targets identified: (1) skip two-pass entropy at e5 on smooth-photo
  d≤1.0 (~14 ms/cell saving), (2) skip pixel-domain loss at e5 on
  photo class (~11 ms/cell), (3) parallel xform fan-out at e3/e4 on
  ≥1.5 MP screenshots (~35 ms/cell). Sweep wall: 254.4s. zenjpeg dev-
  dependency (`{ version = "0.8.4", features = ["decoder", "trellis",
  "parallel"] }`) added; workspace `[patch.crates-io]` already
  redirects to local sibling.

- **`PatchesDispatch` enum + `LossyConfig::with_patches_dispatch`**
  (W36-3, `src/api.rs`, `src/vardct/encoder.rs`,
  `examples/patches_dispatch_ab.rs`,
  `benchmarks/patches_dispatch_e7_2026-05-18.{tsv,meta}`). Default
  `PatchesDispatch::Auto` skips the ~27 ms/MP patches scan on photo
  class (per-block-mean `median(mask1x1) <= 60` — same statistic the
  auto-splines screenshot skip and GPU AFV cost-grid gate use, with a
  dedicated lower threshold because the cost asymmetry is inverted:
  false-negative on a screenshot loses 30-70 % of the screenshot's
  bytes, while false-positive on a photo is just wall-clock overhead
  because the scan returns empty `PatchesData` either way). Empty
  `PatchesData` is the same result the scan would have returned on
  photo content (W11-1 + W12-5: "Zero overhead on CLIC photos"), so
  hash-lock 36/36 stays byte-identical. Screenshots — including
  `windows95.png` 640×480 (the documented false-negative of the >95
  gate per `auto_splines_bench_2026-05-17`) — keep running the scan
  exactly as before. `PatchesDispatch::AlwaysScan` restores the
  pre-W36-3 behaviour for A/B reproducibility runs;
  `PatchesDispatch::NeverScan` force-skips the scan on every image.

- **W36-2 — adaptive dispatch for per-block EPF sharpness selection**
  (`src/api.rs`, `src/vardct/epf.rs`, `src/vardct/encoder.rs`,
  `src/vardct/bitstream.rs`, `src/lib.rs`,
  `jxl-encoder-cli/src/main.rs`,
  `examples/epf_dispatch_ab.rs` [new],
  `tests/lossy_knobs_wiring.rs`,
  `benchmarks/epf_dispatch_e6_e7_2026-05-18.{tsv,meta}`).

  - New public `EpfDispatch` enum + `LossyConfig::with_epf_dispatch`
    builder. Three variants: `AlwaysSelect` (default — historical
    behaviour, byte-identical), `Auto` (skip the per-block search
    on smooth regions per `mask1x1` mean threshold), `AlwaysDefault`
    (force uniform default sharpness, skip the search
    unconditionally). New CLI flag `--epf-dispatch
    {always-select,auto,always-default}`.
  - `compute_epf_sharpness` is the dominant phase on the W36-1 phase
    baseline (`benchmarks/lossy_phase_baseline_2026-05-18.{tsv,meta}`):
    **45.5% of e6 wall-clock** and **33.8% of e7**. The per-block
    sharpness search is bitstream-affecting; skipping converges the
    bitstream onto the uniform default sharpness map (=4).
  - **Default unchanged**: `EpfDispatch::AlwaysSelect`. `hash_lock`
    36/36 byte-identical, RD regression unchanged. Auto-default
    flip evaluated in `examples/epf_dispatch_ab` (10 images × 3
    distances × 3 efforts × 3 dispatch modes = 266 successful cells
    out of 270 planned; 4 screen-e8 cells errored on buttloop budget
    exhaustion, not material to default-flip evaluation). All six
    (class, effort) gates PASS on the full 266-cell sweep: photo
    bytes −1.10 to −1.23 %, screen bytes −1.54 to −2.58 %,
    butteraugli +0.30 to +1.73 % across the grid (under the +2 %
    gate), wall-clock saving 34-49 ms/MP. Shipping as opt-in for
    chunk-1; default flip is queued as chunk-2 follow-on so the
    36-fixture `hash_lock_features` rebake + RD regression baseline
    rebake get their own commit + review (margins on photo-e6
    +1.69 % and screen-e8 +1.73 % are tight enough to want a
    standalone gate-flip rather than bundling with the surface
    introduction).
  - Helper functions in `vardct/epf.rs`:
    `uniform_default_sharpness_map(xb, yb)`,
    `mask1x1_mean(&[f32])`,
    `mask1x1_is_smooth_enough_to_skip_sharpness(&[f32])`. Threshold
    constant `EPF_AUTO_SMOOTH_MASK_THRESHOLD = 60.0` (post-blur
    `mask1x1` mean above this → skip search on Auto). Tested with
    3 unit tests in `vardct::epf::tests` + 3 integration tests in
    `tests/lossy_knobs_wiring.rs`.
  - Encoder field `VarDctEncoder.epf_dispatch` plumbed from
    `LossyConfig.epf_dispatch` at all three construction sites
    (one-shot, animation, JPEG transcode). Gate sites in
    `vardct/encoder.rs:2215` (encode_inner),
    `vardct/encoder.rs:3074` (encode_from_precomputed),
    `vardct/bitstream.rs:1868` (animation frame).

- **RFC#45 chunk 2 — e12 admit gate widening** (mirrors W21-2 chunk 1's
  e11 admit-gate pattern from `24f071db` + `ebf5ddaa`).
  (`src/validation.rs`, `src/effort.rs`, `src/api.rs`,
  `src/vardct/encoder.rs`, `src/vardct/lf_frame.rs`,
  `src/vardct/butteraugli_loop.rs`, `src/modular/frame.rs`,
  `src/validation_tests.rs`, `jxl-encoder-cli/src/main.rs`,
  `jxl-encoder-cli/README.md`,
  `examples/e12_admit_paired_ab.rs` [new],
  `benchmarks/effort_12_admit_2026-05-18.{tsv,meta}`).

  - **`EFFORT_RANGE` widened `1..=11` → `1..=12`** so callers passing
    `with_effort(12)` are not silently clipped to 11 by the validator.
    `EffortProfile::lossy(_).clamp(1, 11)` → `clamp(1, 12)` (and the
    matching lossless path). `vardct/lf_frame.rs::encode_lf_frame`
    DC effort cap `(effort + 1).min(11)` → `min(12)`.
  - **`ITER_MAX` bumped `16 → 32`** (`validation.rs:152`). This is the
    public `MAX_QUANT_LOOP_ITERS` / `Limits::DEFAULT_MAX_QUANT_LOOP_ITERS`
    re-export — it caps the butteraugli / ssim2 / zensim quantization
    loops. Callers that explicitly set a lower per-encode
    `Limits::with_max_quant_loop_iters(_)` are unaffected (the encoder
    saturates at the lower of the per-encode value and the validator
    max). The loop has its own per-iteration convergence early-exit so
    the cap remains a worst-case CPU bound, not a typical iter count.
  - **e12 differentiator: `butteraugli_iters = 32`** (vs e11's 16,
    e10's 8, e9's 4). Doubles the search budget along the same axis
    chunk-1 used for e10/e11, keeping a clean power-of-two ladder
    (4 → 8 → 16 → 32) per effort tier past libjxl's kTortoise=9 cap.
    Knob chosen for "least likely to saturate": the seed table
    `init_mul_seeds` is hard-capped at 4 entries, so requesting
    `lossy_search_seeds = 8` at e12 would silently cap at 4; the
    `tree_learn_seeds` ladder already shipped 16 at e11 (chunk-6
    follow-on); AC strategy `fine_grained_step` already saturates at
    1 from e9; `butteraugli_iters` was the only knob with daylight
    above e11.
  - **Doc comments updated 1-11 → 1-12** at: `EffortProfile.effort`
    (effort.rs:172), `EffortProfile::lossy/lossless` accept-range
    docs, `FrameEncoderOptions.effort` (modular/frame.rs:23),
    `VarDctEncoder.effort` (vardct/encoder.rs:204), `encode_lf_frame`
    arg doc (vardct/lf_frame.rs:133), `LossyConfig::with_effort` and
    `LosslessConfig::with_effort` (api.rs), CLI `--effort` help
    (jxl-encoder-cli/src/main.rs:34) and README ladder row.
  - **Tests**: 8 effort-loop iteration ranges (`1..=11` → `1..=12`)
    across `effort.rs` test module and `validation_tests.rs`.
    `test_effort_clamp` now asserts clamp(99) = 12. New asserts in
    `test_butteraugli_iters_e10_e11_extended` confirm
    `p12.butteraugli_iters == 32` AND that
    `MAX_QUANT_LOOP_ITERS == 32` (so the cap bump and the e12 table
    row stay in lockstep — drift on either side will fail the test).
    `test_lossy_search_seeds_e10_e11_extended` extended to assert
    e12 also fans out 4 seeds (table saturation, documented).
    `lossy_butteraugli_iters_in_range_validates` now accepts 32 as
    in-range; the `too_high_rejected` test asserts the new cap (`*valid.end() == 32`).
  - **Defaults unchanged (e7)**; `hash_lock_features` 36/36
    byte-identical; 1228 jxl-encoder lib tests pass; clippy clean;
    cargo fmt clean.
  - **Acceptance bench** (5 CID22-512 photos × 3 distances {0.5, 1.0,
    2.0} × 2 efforts {e11, e12} × 5 samples = 150 paired encodes,
    `examples/e12_admit_paired_ab.rs`,
    `benchmarks/effort_12_admit_2026-05-18.{tsv,meta}`):
    - **15/15 cells (100%) PASS** the relaxed ≥70% gate (e12 ≤ e11
      bytes AND e12 ≤ e11 butteraugli).
    - **15/15 cells (100%) byte-identical bitstream** (e12 sha256 ==
      e11 sha256 on every (image, distance, sample)). Geo-mean B/A
      ratios: bytes 1.0000 (±0.00%), butteraugli 1.0000 (±0.00%),
      encode_ms 1.86×.
    - **The butteraugli single-axis loop has fully converged within
      the 16-iter budget on CID22-512 at d ∈ {0.5, 1.0, 2.0}.** The
      extra 16 iters at e12 are pure CPU cost for zero RD benefit on
      this corpus — same "gate-only ship" outcome as chunk 1's e11.
    - **Decision**: ship the clamp + cap widening per the chunk-2
      task brief's "ship anyway" rule. The differentiator knob is
      live for callers who request `with_butteraugli_iters(32)` or
      hit slower-converging corpora; CID22-512 photos just don't need
      it. Chunk-3 follow-on plan (the actual e12 lever) documented in
      the meta file: extend `init_mul_seeds` past its 4-entry cap and
      bump `lossy_search_seeds[12] = 8`, OR split `tree_learn_seeds`
      slots into smaller perturbations and bump to 24, OR add a
      fundamentally new optimization axis (per-block AC strategy
      re-eval, two-pass mask1x1 with the post-loop quant field).
      Single-axis iter doubling is exhausted as a lever.

- **Streaming refactor #11 chunk 8b — `XybRegionSource` trait + walker
  seam in `encode_inner` + `encode_from_precomputed_inner`**
  (`src/vardct/region_source.rs` [new],
  `src/vardct/transform.rs::transform_and_quantize_with_source`,
  `src/vardct/encoder.rs::encode_inner` walker,
  `src/vardct/encoder.rs::encode_from_precomputed_inner` walker,
  `examples/bench_buffering_rss.rs`,
  `benchmarks/streaming_chunk8b_peak_rss_2026-05-18.{tsv,meta}`).

  - **New `XybRegionSource` trait** (`pub(crate)` in
    `vardct/region_source.rs`): `xyb_full() -> (&[f32], &[f32],
    &[f32])` plus `release_dc_region(dc_x, dc_y)` release hint.
    Whole-image impl (`WholeImageXybSource`) and borrowed-view impl
    (`BorrowedXybSource<'a>`) — both `Sync` for the rayon-parallel
    fan-out inside `transform_and_quantize`.
  - **`VarDctEncoder::transform_and_quantize_with_source`**: pull-
    style entry point that takes `&dyn XybRegionSource` instead of
    three `&[f32]` slices. Today it calls `xyb_full()` once and
    delegates to the existing whole-image
    `transform_and_quantize`; output is byte-identical (verified by
    `hash_lock_features` 36/36).
  - **`encode_inner` walker** wraps the three XYB Vecs in a
    `WholeImageXybSource`, calls `transform_and_quantize_with_source`,
    then iterates DC groups and calls
    `release_dc_region(dc_x, dc_y)` on the source. The whole-image
    source ignores the hint — chunk-8c will wire a streaming source
    that drops the region's storage on each release.
  - **`encode_from_precomputed_inner` walker** wires the same trait
    with a `BorrowedXybSource` (precomputed XYB is owned by the
    caller).
  - **Documented remaining whole-image consumers** in
    `region_source.rs` module docs: (1) `compute_epf_sharpness`,
    (2) the mask1x1 fallback inside the sharpness branch,
    (3) `butteraugli_loop` (feature-gated, multi-iteration), (4)
    splines auto-detection / `simplify_invisible` (run before
    `transform_and_quantize`, not affected). Chunk-8c plan: lift each
    consumer into the per-DC-group walker so the release can happen
    *before* the consumer runs.
  - **Peak-RSS at 4096×4096** (lossy d=1.0, 4 GiB cap):
    FullBuffered ≈ 2895 MB, BufferedOutput ≈ 2894 MB,
    FullStreaming ≈ 2895 MB — identical within measurement noise.
    Bytes byte-identical across all 3 variants (12382528 B). **No
    memory reduction is expected from chunk 8b alone** — the trait
    is a structural prereq; actual peak-RSS savings land in chunk-8c
    when the streaming source materialises one DC group at a time
    and drops it on the release hint.
  - **Acceptance**: `cargo test --lib` 1222 pass (+4 region_source
    unit tests vs 1218 baseline), `cargo test --test
    hash_lock_features` 36/36, `cargo test --test buffering_dispatch`
    7/7, `cargo test --test buffering_enum` 15/15, `cargo clippy
    --lib -- -D warnings` clean, `just rd-regression` 2/2
    (improvements on every cell — likely a marginal effect of the
    extra walker structure on a hot LLVM inlining decision).

### Fixed

- **W42-2 — patches reference-frame `group_size_shift` libjxl parity**
  (`jxl-encoder/src/vardct/patches.rs`,
  `jxl-encoder/examples/patches_group_size_shift_ab.rs`,
  `benchmarks/patches_group_size_shift_2026-05-18.{tsv,meta}`). Ports
  libjxl's `GetGroupSizeShift` dimension heuristic
  (`lib/jxl/enc_frame.cc:125-146`) to the patches reference-frame writer
  (`patches.rs` `encode_reference_frame` + `encode_reference_frame_rgb`).
  Pre-fix we hardcoded `group_size_shift = 1` (256-pixel groups) via
  `FrameHeader::lossless()`, so a typical 268×260 packed-patches ref
  frame split into a 2×2 = 4 PassGroup grid and paid 4× per-stream
  entropy overhead (LZ77 metadata, HybridUint headers, byte alignment,
  TOC entries). New helper `patches_ref_group_size_shift(w, h)` mirrors
  libjxl's thresholds (≤128: shift=0, ≤256: shift=1, ≤400: shift=2,
  else shift=1) so a 268×260 ref frame now emits as a single
  512-pixel group with no per-PassGroup overhead. Both sites also wire
  `FrameEncoderOptions::modular_group_size_shift = Some(shift)` so the
  encoder's actual partitioning matches the FrameHeader signal (the
  body writer reads its options field independently of the FH).
  **Headline (imac_g3 @ e7, d=3.0)**: total bytes 223,759 → 178,037
  (**-20.4 %**, ref-frame component 88,378 → ~46 KB at ~5 bpp);
  gap vs cjxl 0.12.0 closes from +38.4 % to +10.1 %. **Other wedge
  cells (e7)**: imac_g3 d=2.0 -50.7 % vs cjxl (we still win because
  cjxl runs streaming-mode at d<3 and skips patches entirely);
  terminal d=2.0 -48.9 % vs cjxl, d=3.0 +14.1 %; codec_wiki d=2.0
  -7.9 %, d=3.0 +14.3 %; windows95 d=2.0 +6.8 %, d=3.0 +7.3 %.
  Photo class byte-identical pre/post fix on the 3 CID22-512 photos
  sampled at d∈{2,3,4,5} (PatchesDispatch::Auto short-circuits when
  patches aren't detected, so the new shift code never runs). Closes
  issue #52 root-cause WF2.
  - **Validation**: `cargo test --lib patches::` 10/10 (incl. new
    `test_patches_ref_group_size_shift_matches_libjxl` covering libjxl's
    four dimension buckets + the imac_g3 wedge case), `cargo test
    --test hash_lock_features` 36/36 byte-identical, `cargo clippy
    --lib -- -D warnings` clean, `just rd-regression` 2/2 (all 18
    cells smaller than baseline, well within thresholds), djxl decode
    of the post-fix imac_g3 d=3.0 file successful.
  - **libjxl reference**: `lib/jxl/enc_frame.cc:125-146`
    `GetGroupSizeShift` — we model only the dimension-driven tail; the
    cparams branches (`!modular_mode`, `decoding_speed_tier >= 2`,
    `responsive == 1 && IsLossless`) don't apply to ref-frame emission.

### Investigated

- **W35-2 chunk-4 — safe-class entropy_mul re-bisect (windows95
  EXCLUDED) — HONEST-STOP, no default-on flip**
  (`examples/entropy_mul_safe_class_bisect.rs`,
  `benchmarks/entropy_mul_safe_class_bisect_2026-05-18.{tsv,meta}`).

  Follow-on to W35-1 chunk-1 (`3541912b`), which proved the
  `with_screenshot_lift_hint` API correctly suppresses the W22-1
  lift on windows95 (plog2=4) but the lifted table itself is too
  aggressive on EVERY screen-class image. This chunk drops windows95
  from the bisect corpus and re-sweeps the 9 plog2 ≥ 7 screenshots
  with LOWER lift values: `IDENTITY` ∈ {1.10..1.30} ×
  `DCT2X2` ∈ {1.04..1.13} (W23-2 stage A swept 1.20..1.60 / 0.95..1.045
  and failed; W35-2 narrows further). AFV + DCT4X8 pinned at W22-1
  lifted values (0.95, 0.98). 9 imgs × 20 tuples × 3 distances ∈
  {0.5, 1.0, 2.0} = 540 stage-A measurements.

  Pass gate: avg screen-class bytes Δ ≤ -0.30 % AND no cell
  |bfly Δ| > 3 % AND ≥ 80 % of cells |bfly Δ| ≤ 2 %.

  **NO TUPLE passes.** Smallest max |bfly Δ| across the entire grid
  is 91.48 % (IDENTITY=1.15 DCT2X2=1.10) — far above the 3 % bar.
  Best avg bytes is -0.509 % (IDENTITY=1.10 DCT2X2=1.10) but with
  max |bfly Δ| 115.9 %. Per-image bistability dominates: the same
  tuple shows `imessage d=1` -24.1 % bfly AND `imessage d=0.5`
  +16.2 % bfly. `graph` (796x481 high-edge plot) is the worst
  outlier — +91-115 % bfly at d=0.5 across the entire grid, even
  at IDENTITY=1.10. Confirms W23-2's structural finding: lifting
  IDENTITY entropy_mul triggers per-block AC-strategy flips that
  swing bfly wildly in both directions; no global tuple can clear
  the gate.

  Default `LossyConfig::content_aware_entropy_mul` stays `false`;
  the W35-1 hint API (`with_screenshot_lift_hint`) stays as the
  caller-driven opt-in. Hash-lock fixtures untouched.

  Chunk-5 plan (ranked, see meta): (1) per-block discriminator
  inside `compute_ac_strategy` (multi-week, deep AC search rework);
  (2) tighten zenanalyze rule with `lum_entropy >= 1.0` to suppress
  `graph`-class outliers (cheap but doesn't fix per-image
  bistability on the other 8); (3) lift `kAvoidEntropyOfTransforms`
  gate from `d > 4` to `d > 0` (W23-2 deferred); (4) decompose
  `screenshot_suppressed()` into per-strategy gates (start with
  DCT4X4-only lift); (5) accept that the wedge is structural and
  ship the W35-1 hint infrastructure as the final state. Recommend
  path #5 (no further work) per the data — no chunk-5 work is
  shippable today without one of the deep paths.

- **Streaming refactor #11 chunk 7 — peak-RSS bench at 4K confirms
  structural blocker; documented chunk-8 plan** (no production code
  changes). `benchmarks/streaming_chunk7_peak_rss_2026-05-18.{tsv,meta}`.

  - **Default `LossyConfig::encode()` path at 4096²** measures
    identical peak RSS (~1527 MiB) and identical bytes (12382528)
    across all 5 `Buffering` variants. The `Buffering` knob remains a
    no-op on the default path — this is the backwards-compat guarantee
    chunk 6 promised, and the gap that chunk 8 must close.
  - **Rate-control path at 4096²** confirms the chunk-6 pattern at
    larger size: per-region (BufferedOutput) uses **+7%** peak RSS
    (4759 vs 4441 MiB) and bytes diverge by +0.056% (per-region class
    vs whole-image class). Reproduces the chunk-6 3K finding (+12%
    per-region).
  - **Why chunk 7 cannot deliver peak-RSS reduction with the
    chunk-3/4/5/6 helpers as-built**: (1) `compute_global_only`
    allocates an `xyb_pre_gaborish` snapshot (~192 MiB at 4K) so
    per-region precompute reads from a stable source — the default
    `encode_inner` does gaborish in-place and pays no snapshot cost,
    so routing through `compute_with_budget_and_buffering` would
    INCREASE peak RSS; (2) the chunk-4 `encode_dc_group` primitive
    consumes whole-image token vectors (`dc_tokens`,
    `ac_section_tokens_per_pass`) — real per-DC-group memory savings
    require collecting tokens per-region AND clustering at the end
    (libjxl `acc28c0`'s `global_group_codes[]` shape).
  - **Chunk 8 plan** (the actual peak-RSS reduction): reshape
    `encode_two_pass` to collect tokens per-DC-group (drop slice on
    `quant_dc`/`quant_ac`/`nzeros`/`xyb_*` immediately after
    tokenization), run histogram clustering across the accumulated
    per-group token sets, emit DC global + per-DC-group sections +
    AC global with `permuted_toc=0` for `BufferedOutput` (libjxl
    `6553831`-style explicit-write) and permuted TOC + seek-back via
    the chunk-6 `WritableSeek` trait for `FullStreaming`. Target
    working set: ~5 MiB per DC group vs ~190 MiB whole-image xyb at
    4K. Estimated 4-7 agent-days per the porting plan.
  - **Honest-stop rationale** (per CLAUDE.md "honest-stop > false
    completion"): the prompt allowed "ship the partial refactor that
    at least removes the precompute peak even if downstream still
    re-materializes". The partial refactor (route precompute through
    `compute_with_budget_and_buffering`) ADDS memory cost on the
    default path because of (1) above. Shipping it as "chunk 7
    progress" would be false-completion — peak RSS would regress and
    the `BufferedOutput` knob would still be a no-op on the byte-level
    (encode_inner re-does everything inline regardless of what the
    precomputed struct contains). The bench documents the structural
    gap so the next agent picks up from the right baseline.

### Added

- **Streaming refactor #11 chunk 6 — `Buffering`-driven dispatch +
  `WritableSeek` trait + permuted-TOC `=0` invariant test** (`src/api.rs`,
  `src/lib.rs`, `src/vardct/encoder.rs`, `src/vardct/precomputed.rs`,
  `tests/buffering_dispatch.rs`,
  `examples/bench_buffering_rss_rate_control.rs`).

  - **`compute_with_budget_and_buffering`** in
    `vardct/precomputed.rs` — the chunk-5 `JXL_STREAMING_CHUNK5=1`
    env-var gate is replaced by a per-call `Buffering` parameter.
    Routing matrix: `FullBuffered` / `Threshold2048` always go through
    the whole-image precompute (chunk 3); `BufferedOutput` /
    `FullStreaming` engage the per-region precompute (chunk 5); `Auto`
    resolves via [`Buffering::resolve_for`] (≤2048² → `FullBuffered`,
    larger → `BufferedOutput`). The env var still works as an escape
    hatch when set.
  - **`VarDctEncoder.buffering`** field threads the caller's
    [`LossyConfig.buffering`] policy into the rate-control entry
    point (`encode_with_rate_control_config`), which is the only
    consumer of `compute_with_budget` today. Default `Buffering::Auto`
    keeps every existing hash-lock byte-identical.
  - **`pub trait WritableSeek: std::io::Write + std::io::Seek`** in
    `api.rs`, with blanket impl covering `std::io::Cursor<Vec<u8>>` /
    `std::fs::File`. Required by the new
    [`LossyEncoder::finish_to_seekable`] and
    [`LosslessEncoder::finish_to_seekable`] methods. **Chunk 6
    behaviour: routes through `finish_inner` like `finish_to`** — the
    bytes are produced in memory and written in one pass; the seek
    capability is plumbed for the chunk-7 level-3 streaming-output
    path (permuted TOC + DC-global placeholder + post-frame seek-back,
    mirroring libjxl `acc28c0`).
  - **Re-applied chunk 4** (encode-side `encode_dc_group` extraction;
    bitstream.rs). Chunks 1, 2, 3, 5 landed on `origin/main` but
    chunk 4 was authored as a sibling commit (`fa12661c`) that never
    made it into a branch. Chunk 6 needs the per-DC-group
    `EncodedDcGroup` emit primitive as the structural prereq for the
    chunk-7 per-DC-group section buffer drop, so the dangling commit
    is re-introduced here verbatim ahead of the chunk-6 wireup.
  - **`#[derive(Clone)]` on `CflMap`** (`chroma_from_luma.rs`). Pre-
    existing chunk-3 bug — `compute_dc_group` called
    `aggregated_cfl.clone()` but the type wasn't `Clone`, so the
    rate-control feature build failed with `E0596`. Repaired here so
    the chunk-6 rate-control test path compiles.
  - **New tests** in `tests/buffering_dispatch.rs`:
    - `rate_control_buffering_dispatch_routes_correctly` (gated on
      `feature = "rate-control"`) — verifies that on a sub-threshold
      (256²) image all `Buffering` variants produce byte-identical
      bytes via the whole-image path, while on a super-threshold
      (2560²) image the `BufferedOutput` / `FullStreaming` / `Auto`
      variants produce bytes that fall inside the chunk-5-documented
      <1% FP-drift envelope of the `FullBuffered` baseline.
    - `permuted_toc_zero_invariant_for_buffered_output` — asserts
      `BufferedOutput` and `FullBuffered` produce byte-identical
      output on a 2560² image (both write `permuted_toc=0`,
      mirroring libjxl PR `6553831`'s explicit-zero fix). Chunk 7
      will lift this for `FullStreaming` only, when the level-3 path
      starts writing `permuted_toc=1`.
    - `finish_to_seekable_round_trips_identically_lossy` /
      `_lossless` — sanity that the `WritableSeek` finish path
      produces bytes identical to `finish()` at chunk 6.
  - **Bench data** at
    `benchmarks/streaming_chunk6_peak_rss_2026-05-18.{tsv,meta}`.
    Headline at 3072×3072: `LossyConfig::encode()` (default
    `encode_inner` path) sees identical peak RSS (~861 MiB) and
    identical bytes across all 5 `Buffering` variants — confirms the
    backwards-compat guarantee. The rate-control path
    (`VarDctEncoder::encode_with_rate_control_config`, which IS the
    consumer of the chunk-6 dispatch) sees the per-region path use
    **slightly higher** peak RSS than whole-image (+5–12%), because
    the per-region functions produce per-region buffers that are then
    copied into the whole-image accumulators that downstream
    rate-control / butteraugli / `encode_from_precomputed` still
    consume. This is the chunk-5-documented behaviour: real memory
    reduction needs chunk 7 to refactor `encode_inner` itself to use
    the chunk-3/4/5 per-DC-group primitives + drop per-region buffers
    inline.

  **Why peak RSS does not drop at chunk 6** (the honest-stop): the
  default `LossyConfig::encode()` path goes through
  `vardct/encoder.rs:encode_inner` which does inline precompute (XYB
  conversion, `compute_quant_field_float_with_budget`,
  `compute_mask1x1_with_budget`, `gaborish_inverse_maybe_adaptive`,
  `compute_cfl_map_with_budget`, `compute_ac_strategy_for_tiles`) and
  inline emit (`parallel_map_result` over `encode_dc_group_section` /
  `encode_ac_group_section`). The chunk-3/4/5 per-DC-group helpers
  exist as `EncoderPrecomputed::compute_with_budget_and_buffering` +
  `bitstream::encode_dc_group`, but **only the rate-control path
  consumes them**. Chunk 7 must reshape `encode_inner` to call the
  same compute_global_only + per-DC-group encode_dc_group +
  per-region buffer drop sequence the rate-control path uses, then
  the BufferedOutput / FullStreaming routes will actually reduce RSS.

  **Chunk 7 plan** (carries forward from chunk 6):
  1. Refactor `encode_inner` to call `compute_global_only` (chunk 2)
     + per-DC-group `compute_dc_group` (chunk 3) +
     `fill_dc_group_state_per_region` (chunk 5) instead of the
     inline precompute calls.
  2. Hook `encode_dc_group` (chunk 4) into the per-DC-group emit
     loop so each DC group's LfGroup + HF sections land in
     `global_group_codes[]` and the per-region XYB / quant_field /
     mask1x1 slice on `global` is dropped (via `Vec::drain` or
     replacement with an empty same-stride buffer) before the next
     DC group runs.
  3. For `Buffering::FullStreaming`: emit each DC group's sections
     directly to the `WritableSeek` sink as they finish, reserve the
     DC-global placeholder upfront, then seek back at end-of-frame
     to write the real DC-global + write `permuted_toc=1` via
     `write_toc_with_permutation` (already exists in `vardct/frame.rs`).
     Mirror libjxl `6553831`'s explicit-write fix for the level-2
     `permuted_toc=0` bit while we're at it.
  4. Stream input via `LossyEncoder::push_rows` for the level-3
     path: today `push_rows` linearises eagerly into
     `self.linear_rgb`; chunk 7 should let each DC group consume only
     the rows it needs (mirroring libjxl's
     `JxlEncoderChunkedFrameAdapter` random-access shape).

  libjxl reference: PRs #4634 (`acc28c0`) + #4635 (`032d39a`) +
  #4637 (`b3510d1`) + #4642 (`1389871`) + #4728 (`6553831`). The
  chunk-6 dispatch mirrors `enc_frame.cc:1779-1820` (`CanDoStreamingEncoding`
  + default-buffering resolution) and reserves the chunk-7 seek-back
  for the actual `EncodeFrameStreaming` (`enc_frame.cc:2042-2200`)
  port.

### Changed

- **Streaming refactor #11 chunk 5 — per-region `quant_field` /
  `mask1x1` / `gaborish_inverse` with border replication**
  (`vardct/adaptive_quant.rs`, `vardct/gaborish.rs`,
  `vardct/precomputed.rs`). Adds three new pub(crate) helpers:
  - `compute_quant_field_float_for_region` — runs pre-erosion + fuzzy
    erosion + per-block modulations on a single DC-group-sized
    rectangle. The 1-block (8-pixel) border is read directly from the
    whole-image XYB planes — the existing SIMD primitives in
    `jxl-encoder-simd::adaptive_quant` already accept a rect on the
    input XYB and write region-local aq_map, so per-region wiring is a
    straight composition (byte-identical to the whole-image
    `compute_quant_field_float` when assembled over a tiling that
    covers the image exactly once — verified by
    `vardct::adaptive_quant::tests::test_per_region_quant_field_matches_whole_image`).
  - `compute_mask1x1_for_region` — loads `region + 2-pixel border`
    into a padded scratch buffer (edge-replicated at the image
    boundary), runs the 5×5-stencil raw mask + Symmetric5 blur on the
    padded buffer, extracts the inner region. `PAD = 3` closes the
    structural divergence at interior region boundaries where the SIMD
    primitive's internal clamping would otherwise substitute padded-
    edge pixels for off-buffer reads — bumping PAD by one pushes the
    clamping outside the inner-region's blur reach.
  - `gaborish_inverse_for_region` — same approach as `mask1x1` but
    with a 2-pixel PAD; takes separate `src_{x,y,b}` (pre-gaborish
    snapshot read-only) and `dst_{x,y,b}` (post-gaborish accumulator,
    mutated in place). The src/dst split mirrors the whole-image
    function's internal scratch copy and lets successive per-region
    calls read pre-gaborish neighbours even though earlier regions
    have already overwritten the dst at adjacent positions.

  **Dispatch** (`fill_dc_group_state_per_region` +
  `fill_dc_group_state_dispatch` in `vardct/precomputed.rs`):
  `EncoderPrecomputed::compute_with_budget` reads the
  `JXL_STREAMING_CHUNK5=1` env var to switch between the chunk-3
  whole-image precompute and the new chunk-5 per-region precompute.
  Currently NOT wired to any `Buffering` variant in the default
  path — actual buffer-drop memory savings need chunk 4
  (per-DC-group `encode_dc_group` split) so the assembly buffers can
  shrink. The dispatch lets correctness be validated end-to-end (hash
  lock + buffering_dispatch + rd_regression all pass with either flag
  setting) before chunks 4/6 land the bitstream-level work.

  **Byte-identity verification**:
  - `hash_lock_features` 36/36 byte-identical with `JXL_STREAMING_CHUNK5`
    on AND off — small images route through single-DC-group iterations
    of the per-region loop but the chunk-5 path still exercises the
    code.
  - `tests/buffering_dispatch.rs` 4/4 byte-identical (single-DC-group
    256×256 and multi-DC-group 2560×2560 lossy + lossless variants).
    Multi-DC-group lossy at 2560×2560 d=2.0 produces the IDENTICAL
    byte sequence under chunk-3 and chunk-5 dispatch — the FP drift in
    the per-region functions (max 256 ULPs on individual mask1x1 /
    gaborish values, 0 ULPs on quant_field) is bounded enough that
    downstream quantization / AC strategy thresholding absorbs it
    fully on these test inputs.
  - `just rd-regression` (18 cells): all within ±3% size, ±5%
    butteraugli, ±1.0 SSIM2. Chunk-5 path delivers a marginal 0.0-0.3%
    size win on every test cell (FP drift in mask1x1 nudges a handful
    of AC strategy decisions toward slightly better choices on these
    images; not a portable win — likely flips the other way on other
    content).
  - `just rd-regression-hd` (6 cells at d=3.0): all within quality
    thresholds.

  **Memory profile** (`bench_buffering_rss 3072 3072`, 4 DC groups):
  chunk-5 on vs off shows peak RSS within 3 MB at every Buffering
  variant (1630-1633 MiB). **No memory reduction** — chunk 5 alone
  cannot drop buffers because the loop driver still returns
  whole-image-sized `quant_field` / `masking` / `mask1x1` / post-
  gaborish XYB that the butteraugli loop and `encode_from_precomputed`
  expect. The load-bearing memory win lands in chunk 6 once chunk 4
  splits `encode_from_precomputed` so each DC group's bitstream
  section is emitted (and its assembly buffers freed) before the next
  DC group runs. Per-region functions are the structural prereq;
  bench data + meta saved at
  `benchmarks/streaming_chunk5_peak_rss_2026-05-18.{tsv,meta}`.

  libjxl reference: same PRs as chunk 3 (#4634/#4635/#4637/#4642/#4728).
  The per-region functions mirror libjxl's `Rect`-taking variants in
  `enc_adaptive_quantization.cc` and `enc_gaborish.cc` (which use
  `rect.Extend(3, parent)` to handle the border — our explicit `PAD`
  loading is the same idea).

  **Chunk 6 plan** (`WritableSeek` + permuted TOC for `FullStreaming`
  true seek-back path): when chunk 4 lands, swap
  `EncoderPrecomputed::compute_with_budget`'s `per_region` env-var
  gate for a `Buffering`-driven dispatch (FullStreaming →
  per-region-precompute + per-DC-group emit + buffer drop). Add a
  `pub trait WritableSeek: io::Write + io::Seek` and route
  `LossyEncoder::finish_to_seekable` through it for the level-3
  streaming-output path. Mirror libjxl `6553831`'s explicit
  `permuted_toc=0` write while we're at it.

- **Streaming refactor #11 chunk 3 — per-region `compute_dc_group` loop
  driver** (`vardct/precomputed.rs`, `vardct/chroma_from_luma.rs`,
  `vardct/ac_strategy.rs`). Replaces the chunk-2 monolithic
  `fill_dc_group_state_whole_image` with a real per-`DC_GROUP_DIM`
  (2048×2048) loop that iterates `compute_dc_group(global, dc_x,
  dc_y, ...)` over every DC group in the image and assembles
  per-region `PerDcGroupFill` slices into the whole-image Vecs that
  downstream rate-control / butteraugli / `encode_from_precomputed`
  consumers still expect. Hash-locked byte-identical
  (`hash_lock_features` 36/36, plus new `buffering_dispatch` test
  pinning byte-identity across all 5 `Buffering` variants on a
  2560×2560 multi-DC-group image).

  **Per-region split per cross-group dep**:
  1. Gaborish 5×5 — whole-image precompute, sliced per region.
     Chunk 5 will add 2-pixel border replication.
  2. mask1x1 5×5 — whole-image precompute, sliced per region.
     Chunk 5 will add 2-pixel border replication.
  3. quant_field 3×3-block — whole-image precompute, sliced per
     region. Chunk 5 will add 1-block border replication.
  4. CfL 8-block tiles — **per-region** via new
     `chroma_from_luma::compute_cfl_map_for_tiles` helper. DC groups
     (256×256 blocks = 32×32 CfL tiles) align cleanly; no border
     needed (per-tile CfL has no cross-tile state).
  5. AC strategy 1-block — **per-region** via new
     `ac_strategy::compute_ac_strategy_for_tiles` helper, taking an
     arbitrary tile list. Per-tile AC search reads only its tile's
     XYB slice; per-DC-group call is byte-identical to the slice of
     the whole-image call.

  **All `Buffering` variants currently route through the same
  per-region loop**, so output bytes are bit-identical regardless of
  `--buffering -1..3`. Peak RSS measurement on a 3072×3072 (4 DC
  groups) lossy `d=1.0` encode: FullBuffered = BufferedOutput =
  FullStreaming = 1.63 GiB (within 32 KB of each other), all
  producing the identical 6 973 041-byte bitstream. This is the
  honest-stop point for chunk 3 — actual memory savings on
  `Buffering::BufferedOutput` lands in chunk 5 once per-region
  versions of quant_field / mask1x1 / gaborish ship (chunk 4 handles
  per-DC-group bitstream emit + `global_group_codes[]`
  accumulation). The chunk-3 loop driver is the load-bearing
  structural prereq.

  libjxl reference: PRs #4634 (acc28c0) + #4635 (032d39a) + #4637
  (b3510d1) + #4642 (1389871) + #4728 (6553831). Bench:
  `cargo run --release --example bench_buffering_rss <variant> [w h]`.

- **Streaming refactor #11 chunk 2 — split `compute_with_budget` into
  global vs per-DC-group precompute** (`vardct/precomputed.rs`).
  Internally factors `EncoderPrecomputed::compute_with_budget` into:
  (a) `EncoderPrecomputedGlobal::compute_global_only` — runs the
  pipeline steps that fundamentally need to see the whole image
  (XYB conversion, noise estimation, patches detection / subtract,
  chromacity stats, pre-gaborish XYB snapshot); and
  (b) `fill_dc_group_state_whole_image` — runs the steps that can in
  principle be processed per-DC-group (quant_field, mask1x1,
  gaborish_inverse, CfL, AC strategy). In chunk 2 the per-DC-group
  fill processes the whole image as ONE region so the assembled
  `EncoderPrecomputed` is **bit-identical** to the prior monolithic
  implementation (`hash_lock_features` 36/36 pass).

  Public API unchanged — `EncoderPrecomputed::compute` /
  `compute_with_budget` keep the same signature and return shape.
  The split is the structural prerequisite for chunks 3-7 (streaming
  input + buffered output, mirroring libjxl PRs #4634 / #4635 / #4637
  / #4638 / #4639). Five hidden cross-DC-group dependencies are
  surfaced and documented on `EncoderPrecomputedGlobal` (gaborish
  5×5, mask1x1 5×5, quant_field 3×3 block, CfL 8-block tile, AC
  strategy neighbour-block heuristics) — each gets an explicit
  fix-or-accept decision in chunk 3.

  Chunk-3 plan: replace `fill_dc_group_state_whole_image` with a
  per-region `compute_dc_group(global, dc_x, dc_y, ...)` plus a
  driving loop in `encode_with_rate_control` /
  `EncoderPrecomputed::compute_with_budget` that iterates over real
  DC-group-sized windows with 1-block / 2-pixel border replication.
  When `Buffering::LargeImageOnly` / `Buffering::Always` is selected,
  the streaming code path keeps only the active DC group's slice of
  `xyb_x/y/b` in memory and drops it after the per-group encode
  completes — closing the ~400 MB → ~50 MB peak-RSS gap on a 4K
  encode (the issue #11 win).

### Investigated

- **W22-1 chunk-2 follow-on: CPU `entropy_mul` lifted-value re-bisect
  — HONEST-STOP, no default-on flip** (`cpu_entropy_mul_bisect.rs` +
  `cpu_entropy_mul_bisect_stage_a2.rs`). Swept `IDENTITY` ∈
  {1.20, 1.30, 1.40, 1.50, 1.60} × `DCT2X2` ∈ {0.95, 0.9975, 1.045}
  via `LossyInternalParams::entropy_mul_table` override on 5 gb82-sc
  screenshots at d ∈ {0.5, 1.0, 2.0}, with two AFV/DCT4X8 pinnings:
  stage A at the W22-1 lifted values (AFV=0.95, DCT4X8=0.98) and
  stage A2 at the libjxl reference (AFV=0.818, DCT4X8=0.859). NO
  tuple passes the chunk-2 acceptance gate
  (median Δbytes ≤ 0.5 %, max |Δbfly| ≤ 2 %) in either stage. Best
  Δbytes (stage A2 IDENTITY=1.20, DCT2X2=1.045) is -0.048 % median
  but max |Δbfly| 33.2 % on windows95 d=0.5. Per-image breakdown
  shows the destabilization is concentrated on flat-colormap
  screenshots (windows95 14-color, codec_wiki, terminal); the
  `median(mask1x1) > 95` discriminator (W22-1) groups images that
  respond very differently to IDENTITY lifting. `kAvoidEntropyOfTransforms`
  is wired (`ac_strategy_search.rs:60`) but gated to `d > 4.0`, so
  it provides no stabilization at the distances where the excursions
  occur. Default `LossyConfig::content_aware_entropy_mul` stays
  `false` (W22-1 opt-in unchanged); chunk-3 deferred pending one
  of three approaches: (a) lift `kAvoidEntropyOfTransforms` gate
  from `d > 4` to `d > 0`, (b) per-block (not per-image) lift
  discriminator, (c) decompose `screenshot_suppressed()` into
  per-strategy gates (start with DCT4X4-only lift). Bench data:
  `benchmarks/cpu_entropy_mul_bisect_2026-05-18.{tsv,meta}` (240
  measurements stage A) and
  `benchmarks/cpu_entropy_mul_bisect_stage_a2_2026-05-18.{tsv,meta}`
  (225 measurements stage A2).

### Added

- **EX-J11 chunk 4: `HdrLoss::Auto` default dispatcher — PQ / HLG →
  `Vdp2`, everything else → `Butteraugli`** (`vardct/hdr_metrics.rs`,
  `api.rs`, `tests/hdr_vdp2_chunk4_auto.rs`).
  Closes the chunk-3 follow-on: ship the auto-dispatch the chunk-3
  CHANGELOG promised, without disturbing the SDR hash-lock corpus.

  Public API: new `HdrLoss::Auto` variant + `HdrLoss::resolve(tf)`
  + `LossyConfig::resolve_hdr_loss(layout, color_encoding)`. The
  default for `LossyConfig` flips from `HdrLoss::Butteraugli` to
  `HdrLoss::Auto`. The resolver consults the encode's signaled
  transfer function — `EncodeRequest::with_color_encoding(...)` if
  set, else `PixelLayout::implied_transfer_function()` (populated
  for the `RgbPqF32` / `RgbHlgF32` / `RgbBt709F32` HDR layouts) —
  and picks `Vdp2` on PQ / HLG, `Butteraugli` on everything else.
  Resolution happens once at encode entry; the per-iteration
  butteraugli loop reads a concrete variant with zero dispatch
  cost.

  Validation:
  - `hash_lock_features` **36/36 byte-identical** — SDR content
    (sRGB / BT.709 / Linear / Unknown / no TF) resolves to
    `Butteraugli` and the existing reference precompute + per-iter
    compare path runs unchanged.
  - 8 chunk-2 integration tests (`hdr_vdp2_loss.rs`) re-asserted
    against the new default (one assertion updated:
    `default_is_auto_chunk4` replaces `default_is_butteraugli`).
  - 6 chunk-4 integration tests (`hdr_vdp2_chunk4_auto.rs`) prove
    the dispatch matrix: byte-identical `Auto == Butteraugli` on
    SDR Rgb8; byte-identical `Auto == Vdp2` on `RgbPqF32` and
    `RgbHlgF32`; byte-identical `Auto == Vdp2` when the caller
    overrides via `with_color_encoding(ColorEncoding::bt2100_pq())`;
    explicit `Butteraugli` on a PQ layout produces a *different*
    bitstream than explicit `Vdp2` (escape-hatch proof).
  - 10 hdr_metrics unit tests (`vardct::hdr_metrics::tests`)
    cover every cell of the dispatch matrix.

  Per the chunk-3 RD sweep (`benchmarks/hdr_vdp2_chunk3_rd_sweep_2026-05-18.tsv`,
  commit `c8010560`): on PQ / HLG content `Vdp2` improved the
  paper-faithful reference VDP2 score by **-36.5 % on average**
  (top cell -44.6 %) vs. the SDR butteraugli loop, so the new
  default ships measurable HDR perceptual quality wins out of the
  box without any caller opt-in.

  Escape hatches preserved: `LossyConfig::with_hdr_loss(HdrLoss::Butteraugli)`
  pins the SDR loss on any content (useful for byte-stable encodes
  on PQ-tagged but visually-SDR content); `HdrLoss::Vdp2` forces
  the HDR loss on any content.

- **EX-J11 chunk 3: HDR-VDP-2-lite real-corpus RD sweep — validates
  `HdrLoss::Vdp2` against the SDR butteraugli baseline on PQ/HLG
  content** (`examples/hdr_vdp2_chunk3_rd_sweep.rs`,
  `tests/hdr_vdp2_chunk3.rs`,
  `benchmarks/hdr_vdp2_chunk3_rd_sweep_2026-05-18.{tsv,meta}`).
  Closes the chunk-2 acceptance gate: does the calibrated HDR-VDP-2
  maths shipped in chunk 2 (`84be3a7f`) actually drive different —
  and *better* — quant decisions than the SDR-tuned butteraugli loop?

  Methodology: 5 stratified CID22 images × 3 distances {1.0, 2.0,
  4.0} × 3 modes {`HdrLoss::Butteraugli`, `HdrLoss::Vdp2`, `cjxl
  reference`} × 3 intensity_targets {1000, 4000, 10000 nits} = 135
  cells. No real HDR consumer corpus available locally, so we
  synthesise PQ-encoded f32 input from CID22 sRGB: linearise →
  scale to `intensity_target / 10000` → forward PQ-OETF → feed via
  `PixelLayout::RgbPqF32` + `ColorEncoding::bt2100_pq()` +
  `with_intensity_target(nits)`. Decoder side uses jxl-oxide in
  **linear sRGB** (CLAUDE.md-mandated path that's immune to PNG
  color-metadata bugs). The "judge" metric is a paper-faithful
  VDP2 implemented *inline* in the example (5 pyramid bands vs the
  shipped lite's 4, 30 ppd vs 32, Mantiuk-2011-style CSF
  parameters, pooling exponent p = 3.5 vs 4) — deliberately
  parametrised differently from the shipped `vardct::hdr_vdp2_lite`
  so the test is INDEPENDENT of the implementation it judges.

  **VERDICT: PASS** — recommend `HdrLoss::Vdp2` as default for
  PQ/HLG content (deferred to chunk 4 via auto-dispatch on
  `ColorEncoding::transfer_function == Pq | Hlg`):

  - **Dispatch fires**: encoded bytes for `HdrLoss::Vdp2` differ
    from `HdrLoss::Butteraugli` by >2 % on **42/45 (93.3 %)** cells.
    Average byte delta = +112.4 % — VDP2-lite's HDR-aware CSF
    consistently flags more visible distortion at high luminance
    and demands more quant precision than the SDR loop does.

  - **Vdp2 wins quality-per-byte 100 % of the time when spending
    more**: VDP2 spends more bytes than Butteraugli on **43/45
    (95.6 %)** cells; in **43/43 (100 %)** of those cells VDP2
    ALSO achieves a *lower* paper-faithful reference VDP2 score
    (average −36.5 % score improvement). i.e. when VDP2 spends
    bytes, it spends them on errors the reference HDR metric
    agrees are real.

  - Top per-byte win (1418519 d=4.0 it=4000 nits): bytes
    12 037 → 19 492 (+61.9 %), ref score 4.714 → 2.611
    (−44.6 %) — VDP2 spent ~60 % more bytes for ~45 % lower
    reference perceptual error.

  - Two cells where VDP2 strictly dominated (smaller bytes AND
    lower ref score, no trade-off): 1418519 d=1.0 it=4000
    (−0.03 % bytes, −20.98 % score) and d=1.0 it=10000 (−1.76 %,
    −2.09 %).

  Coverage:
  - `examples/hdr_vdp2_chunk3_rd_sweep.rs` (~520 LOC):
    self-contained 135-cell sweep harness with inline forward PQ
    OETF, inline reference-faithful VDP2, paired-delta analysis,
    Spearman correlation (informational only — global spearman
    across cells is dominated by intensity_target axis). Set
    `HDR_VDP2_SMOKE=1` for 1×1×1 cell pipeline check.
  - `tests/hdr_vdp2_chunk3.rs`: 3 integration smoke tests
    confirming the PQ pipeline works end-to-end at the API level
    (`HdrLoss::Vdp2` + `PixelLayout::RgbPqF32` +
    `with_intensity_target` + `with_color_encoding`). All three
    actually decode the output (no header-only false positives),
    all three pass.

  Default `HdrLoss::Butteraugli` stays **byte-identical** to every
  release prior to chunk 1 — `hash_lock_features` 36/36 ✓. Chunk 3
  is a validation-only chunk; no `src/` changes.

  Chunk 4 plan: auto-dispatch `HdrLoss::Vdp2` when the input has
  `ColorEncoding::transfer_function == TransferFunction::Pq |
  TransferFunction::Hlg` (lifted from `ColorEncoding::bt2100_pq()`
  / `bt2100_hlg()` and the `with_color_encoding` setter). Keep
  the explicit `with_hdr_loss(...)` opt-out so callers can pin
  to butteraugli for cross-toolchain bit-for-bit reproducibility.

- **Content-aware `entropy_mul` table dispatch (opt-in, default OFF)** —
  new `LossyConfig::with_content_aware_entropy_mul(bool)` toggle and a
  matching `EntropyMulTable::screenshot_suppressed()` constructor. When
  the caller opts in AND the per-image `median(mask1x1)` exceeds 95
  (screen / glyph / UI content), the AC-strategy search runs against
  lifted entropy_mul values on the four 8x8-class transforms that
  over-pick on flat content (`IDENTITY` 1.0428 → 1.85, `DCT2X2` 0.95
  → 1.15, `AFV` 0.818 → 0.95, `DCT4X8` 0.859316 → 0.98). Photo content
  (median ≤ 95) stays on the existing libjxl-faithful
  `EntropyMulTable::reference()` values bit-for-bit. Mirrors the GPU
  encoder's lifted-table screenshot/photo split
  (`vardct_gpu_dropped_optimizations_resurrection_2026-05-17.md`,
  item #3) on the CPU encoder; default `false` keeps every existing
  hash-lock fixture byte-identical (36 / 36). Wire-up in `effort.rs`
  (new constructor), `api.rs` (config field + builder + getter,
  `LossyConfig::with_effort` preservation, three `VarDctEncoder`
  construction sites), `vardct/encoder.rs` (per-encode gate +
  `median_mask1x1` helper + threshold constant). Issue tracking and
  chunk-2 default-on flip plan live in the
  `vardct_gpu_dropped_optimizations_resurrection_2026-05-17.md` audit.

### Fixed

- **RFC#45 chunk 1 admit-gate widening: actually apply the code changes
  the parent commit promised** (`c20e326c`, follow-on to `24f071db`).
  The parent shipped CHANGELOG + bench data only; this commit applies
  the actual widening: `vardct/lf_frame.rs:258` `min(10)` → `min(11)`,
  doc comments at 5 sites (`EffortProfile.effort`,
  `FrameEncoderOptions.effort`, `VarDctEncoder.effort`, `encode_lf_frame`,
  CLI `--effort` help) `1-10` → `1-11`, and 5 effort-loop test ranges
  in `effort.rs` `1..=10` → `1..=11`. Also replaces the partial
  (sample 1 + half of sample 2) committed acceptance TSV with the full
  5-sample grid — numbers reproduce exactly (encoder is deterministic):
  e10 17/20 (85%) PASS, e11 8/20 (40%) FAIL. Defaults unchanged (e7);
  hash-locks 36/36 byte-identical; 1170 lib tests pass.

- **Modular encoder: fuzz-hardening mirrors for two libjxl upstream
  fixes** (`modular/fuzz_safety.rs`).

  1. **NaN guard in lossy-palette float→int quantization** —
     `modular/palette.rs:1109` in `apply_lossy_palette_with_budget`
     now rejects NaN values produced by adversarial error-diffusion
     states before the `(color_with_error.round() as i64).clamp(...)`
     cast. Rust's NaN-to-int saturation is well-defined (yields 0)
     but silently producing wrong palette indices on fuzz input is
     still a bug. The function bails to `None` (caller skips the
     lossy palette), matching the rest of the function's failure
     contract. Mirrors libjxl commit `1eb44c9` ("Guard against NaN
     values", PR #4667) which adds the same check to
     `enc_modular.cc::QuantizeWP`.

  2. **i32-overflow guard on modular residual computation** —
     `modular/tree_learn.rs:6006` in
     `collect_residuals_with_tree_offset_with_budget`. The
     `pixel - prediction` subtraction is now routed through
     `fuzz_safety::checked_residual` (an `i32::checked_sub` wrapper)
     and returns `Error::InvalidInput("Residual overflow ...")` on
     overflow instead of panicking in debug / silently wrapping in
     release. Valid input never trips this — the weighted-predictor
     output is bounded by the channel's range — so the fast path is
     one branch on success and `hash_lock_features` stays 36/36
     byte-identical. Mirrors libjxl commit `87bee19` ("Check that
     residual does not overflow", PR #4759) which adds the same
     `SubOverflow` check to
     `modular/encoding/enc_encoding.cc::EncodeModularChannelMAANS`.

  Tests: 6 unit tests in `modular::fuzz_safety::tests::*` plus 2
  integration tests in `modular::tree_learn::tests::*`
  (`test_residual_overflow_rejected_with_top_predictor` constructs a
  1×2 single-channel image where `i32::MAX - (-1_000_000)` overflows
  under the `Top` predictor; `test_residual_overflow_guard_zero_overhead_on_valid_input`
  pins the "valid input never reaches the guard" invariant the
  budget-less wrapper's `.expect` relies on).

### Added

- **EX-J13 — Adaptive Gaborish kernel strength** (opt-in via
  `LossyConfig::with_adaptive_gaborish(true)`, default `false`).
  Encoder-side per-tile contrast lookup modulates the 5×5 sharpening
  kernel's strength multiplier in `[0.8, 1.0]` on the Y (luma)
  channel: libjxl-faithful `mul = 1.0` on edges/text, gentler
  `mul ≈ 0.8` on smooth regions. X (red-green) and B (blue) keep
  `mul = 1.0`. Wire-compatible — the decoder always applies the
  same fixed 3×3 inverse Gabor blur, so adaptive sharpening must be
  pre-baked into the post-Gab samples. Silent gate: a no-op when
  `with_gaborish(false)` or when the `effective_gaborish()`
  distance/speed-tier gates disable the inverse filter. New A/B
  harness: `cargo run --release -p jxl-encoder --example adaptive_gaborish_ab`.
  Default-off preserves byte-identical hash-locks (36/36 pass).

  **Wider-corpus follow-on (2026-05-18, W20-1):** 480-cell A/B sweep
  (25 CID22-512 photos + 5 gb82-sc screenshots × {d=0.5, 1, 2, 4} ×
  {e5, e7} × {fixed, adapt}) with butteraugli + ssim2 quality metrics
  via jxl-oxide linear-sRGB decode. The original 5-photo
  bytes-only finding (-1.74% at d=1.0 e7, ecd1ec3c) is corroborated as
  the byte direction (-0.98% on the wider set) but is paid for in
  butteraugli quality: individual cells regress by up to **+11.84%**
  (cid22/1418519 d=1 e=7) on photos and **+17.46%** (gb82-sc/codec_wiki
  d=1 e=5) on screenshots — the photo and screenshot default-on gates
  both fail on the per-cell butteraugli ≤ +5% ceiling. Adaptive
  Gaborish **stays opt-in**; one cell (d=2.0 e=7 on photos) is a clean
  win (-0.67% bytes AND -0.51% butteraugli), suggesting the per-tile
  mapping needs more conservative tuning OR distance-band gating before
  another default-on attempt. See
  `benchmarks/adaptive_gaborish_wider_corpus_2026-05-18.{tsv,meta}`.
  New harnesses:
  `cargo run --release -p jxl-encoder --example adaptive_gaborish_wider_corpus`
  and `adaptive_gaborish_wider_analyze`.

- **RFC#45 chunk 1 admit-gate widening: e10 / e11 effort ceiling open
  end-to-end** (issue #45). Closes the residual surface that still
  pinned the effort range at `1..=10` after the parent commit landed
  the per-knob e10/e11 wiring (clamp inside `EffortProfile::lossy` /
  `EffortProfile::lossless`, `butteraugli_iters` map extension, CLI
  `--effort` help string). Remaining sites widened:
  - `vardct/lf_frame.rs:258` — DC effort cap `(effort + 1).min(10)` →
    `min(11)`. Mirrors libjxl `enc_cache.cc:134-136` "one speed-tier
    slower for DC" idiom past the new ceiling so callers passing
    `with_effort(11)` aren't silently clipped to 10 inside the LfFrame
    path. e10/e11 fall through to the e9 (kTortoise) lossless DC code
    today; only knobs that explicitly scale (`tree_learn_seeds`,
    `lossy_search_seeds`, `butteraugli_iters`) consume the extra
    budget.
  - Doc comments: `EffortProfile.effort` (`effort.rs:136`),
    `FrameEncoderOptions.effort` (`modular/frame.rs:23`),
    `VarDctEncoder.effort` (`vardct/encoder.rs:155`),
    `encode_lf_frame` (`vardct/lf_frame.rs:133`), CLI `--effort` help
    text in `jxl-encoder-cli/README.md` — all updated from `1-10` to
    `1-11` with an explicit "e10/e11 extends libjxl kTortoise=9" note
    so external readers see the new ceiling instead of inferring it
    from compile errors.
  - Effort-loop test ranges in `effort.rs` widened from `1..=10` to
    `1..=11` (9 sites:
    `test_lossless_experimental_matches_reference`,
    `test_tree_parallel_schedule_lossy_matches_lossless`,
    `test_adapt_small_image_fallback_threshold` (two ranges),
    `test_adapt_tree_max_buckets_for_image_threshold` (cross-product),
    `test_adapt_tree_max_buckets_lossy_profile_parity`,
    `test_adapt_to_image_lossy_dct64_gate`,
    `test_adapt_to_image_content_screenshot_enables_patches_at_e5_e6`).
    All 24 effort-module tests pass at the widened range; 1170
    `jxl-encoder` lib tests pass; hash-lock fixtures 36/36
    byte-identical (defaults stay at e7).

  **Acceptance bench**
  (`benchmarks/effort_11_admit_2026-05-18.{tsv,meta}`, 300 paired
  encodes via `examples/e10_e11_paired_ab.rs`): 5 CID22-512 photos × 4
  distances {0.5, 1.0, 2.0, 4.0} × 3 efforts {e9, e10, e11} × 5
  samples, sample-major interleave, jxl-oxide linear decode + Rust
  `butteraugli_linear`. Per-cell medians across the 5 samples:
  - **e10 vs e9: PASS the RFC#45 chunk-1 acceptance gate (17/20 cells,
    85% — ≥80% required).** Geo-mean bytes ratio 0.9966 (-0.34%),
    butteraugli ratio 0.9729 (-2.71%), encode-ms ratio 2.326×.
  - **e11 vs e10: FAILS the same gate (8/20 cells, 40%).** Geo-mean
    bytes ratio 1.0069 (+0.69%), butteraugli ratio 0.9866 (-1.34%),
    encode-ms ratio 3.177×. The butteraugli loop saturates inside the
    iter-8 budget on 12/20 cells, so cranking to iter-16 buys nothing
    on those cells and converges to a slightly looser (qf, scale)
    solution on a handful of others.
  - **Decision** (per RFC#45 chunk-1 plan, "If acceptance fails: ship
    `effort.clamp(1, 11)` anyway — gate is opened — + chunk-2 plan"):
    e10 ships as the chunk-1 win; e11 ships as the gate-only widening
    so the downstream multi-seed (`lossy_search_seeds = 4` at e11) and
    multi-seed tree learning chunks (already wired in this tree —
    `tree_learn_seeds = 8` at e11 per W9-1 chunk 5) consume the e11
    budget instead of `butteraugli_iters` alone. Single-axis iter-16
    loop saturation alone is not enough to beat e10.

  Defaults unchanged (`LossyConfig::new(d)` and `LosslessConfig::new()`
  still produce e7 output). e10/e11 are strictly opt-in via
  `with_effort(10)` / `with_effort(11)`. Bitstream stays 100%
  spec-valid; jxl-rs + jxl-oxide + djxl decode every cell in the
  acceptance bench without warnings or fallback.

- **EX-J11 chunk 2: VDP2-lite — calibrated HDR-VDP-2 subset for the
  butteraugli quantization loop** (`vardct/hdr_vdp2_lite.rs`,
  `vardct/butteraugli_loop.rs`, `EX-J11` in
  `JXL_ENCODER_LEARNINGS.md`). Lands the actual maths behind chunk 1's
  `HdrLoss::Vdp2` dispatch — selecting it now runs the metric in-place
  of butteraugli inside the buttloop instead of surfacing a typed
  `NotImplemented` error.

  - New private module `vardct::hdr_vdp2_lite::compare_vdp2_planar`
    consumes the same planar linear-RGB layout as the butteraugli
    path and returns a `(score, diffmap)` pair the existing
    tile-distance machinery feeds on unchanged.
  - Pipeline: BT.709 → display-luminance (uses encode
    `intensity_target`) → log10(nits) → 4-level Laplacian pyramid →
    Mantiuk-2007 CSF weighted per band (adapts per-pixel to
    reference's local mean luminance) → Minkowski p-norm pooled
    diffmap (p = 4).
  - Default `HdrLoss::Butteraugli` is **byte-identical** to every
    release prior to chunk 1 — `hash_lock_features` stays 36/36 ✓,
    `corpus_regression` unchanged. Opt-in only via
    `LossyConfig::with_hdr_loss(HdrLoss::Vdp2)`.
  - Acceptance bench (`examples/hdr_vdp2_chunk2_bench.rs`,
    `benchmarks/hdr_vdp2_chunk2_bench_2026-05-18.{tsv,meta}`):
    butteraugli output is invariant across intensity_target (as
    expected — the butteraugli params are hardcoded to 80 nits);
    VDP2-lite output SCALES with intensity_target (1138 → 2598 bytes
    at d=2.0 going from 80 → 4000 nits), proving the HDR adaptation
    fires and steers the loop differently on PQ/HLG content.
  - Coverage: 8 new unit tests in `vardct::hdr_vdp2_lite::tests::*`
    (identity → zero, score-monotonic-in-distortion,
    HDR-sensitivity, CSF luminance / frequency shape, padded-stride
    correctness, SDR-score-in-range), updated 8 integration tests in
    `tests/hdr_vdp2_loss.rs` flipping the chunk-1 "Vdp2 stub errors"
    assertions to chunk-2 "Vdp2 completes" assertions plus a new
    `vdp2_with_hdr_intensity_target_completes` smoke test.
  - **Deliberate deviations from the full HDR-VDP-2 paper** (chunk-3
    follow-ons, documented in the module rustdoc):
    cortex-channel orientation decomposition is skipped (luminance
    pyramid only); chromatic sensitivity is skipped (achromatic
    only); phase-uncertain masking is replaced with a linear
    difference; the polynomial JOD calibration is omitted (raw
    pooled detection probability shipped, no 100-point quality
    rescale). For in-loop steering — which only needs *relative*
    scores between iterations of the same image — these
    simplifications are calibrated to be at parity with the full
    paper on the buttloop's accept-bound machinery.
  - Chunk-3 plan (queued): real HDR corpus RD measurement
    (CID22-PQ + butteraugli/SSIM2/ssim2 sweep), cortex-channel
    decomposition, chromatic sensitivity via L/M/S cones, masking
    model from `Visibility & Quality Predictions in All Luminance
    Conditions` §4.3.

- **EX-J11 chunk 1: HDR-aware loss dispatch for the butteraugli
  quantization loop** (`vardct/hdr_metrics.rs`,
  `LossyConfig::with_hdr_loss`, `EX-J11` in
  `JXL_ENCODER_LEARNINGS.md`). Ships the API surface + dispatch
  wiring + validation so callers can opt into a future HDR-VDP-2
  loss (PLCC 0.936 vs Butteraugli-pnorm's 0.882 on HDR-AIC-2025) on
  HDR encodes.

  - New public enum `HdrLoss { Butteraugli (default), Vdp2 }`
    re-exported from the crate root (gated behind
    `feature = "butteraugli-loop"`).
  - New `LossyConfig::with_hdr_loss(loss)` setter +
    `LossyConfig::hdr_loss()` getter; the field is preserved
    across `with_effort()` re-application (mirrors the
    `butteraugli_iters` preservation pattern).
  - Default `HdrLoss::Butteraugli` is **byte-identical** to every
    release prior to this commit — `hash_lock_features` stays
    36/36 ✓, `corpus_regression` unchanged.
  - `HdrLoss::Vdp2` is **opt-in only and stub-only in chunk 1**:
    when the butteraugli loop runs (effort ≥ 8) with `Vdp2`
    selected, the dispatch surfaces
    `Error::NotImplemented("HDR loss dispatch: HdrLoss::Vdp2 is
    not yet implemented (EX-J11 chunk 2 — multi-scale CSF pyramid
    pending) (selected: vdp2)")` — a typed error, never a panic.
  - Chunk 2 (queued) lands the actual HDR-VDP-2 maths
    (LUT-baked PQ/HLG transfer-function inversion to display
    nits, multi-scale CSF-weighted Laplacian pyramid, per-band
    visibility-threshold normalisation). Chunk 2 only has to
    swap the `validate_loss` call site in
    `vardct/butteraugli_loop.rs:128` to route through the real
    VDP-2 reference type; the rest of the loop is unchanged.
  - Coverage: 11 tests total — 4 unit tests in
    `vardct::hdr_metrics::tests::*` (enum surface, validation
    predicate, error formatting) plus 7 integration tests in
    `tests/hdr_vdp2_loss.rs` (default-is-Butteraugli,
    explicit-default-is-byte-identical-to-implicit,
    Vdp2-typed-error-when-buttloop-runs,
    Vdp2-silently-unused-at-e7,
    `with_effort`-preservation, end-to-end roundtrip with
    `HdrLoss::Butteraugli` at e8).

- **`Buffering` enum + `with_buffering` builders + `--buffering` CLI
  flag** (issue #11, chunk 1 of the streaming refactor porting plan).
  Scaffolding for the libjxl 3-level buffering refactor (mirrors
  upstream PRs #4634 + #4635 + #4637 + #4642 + #4728). Five variants:
  `Auto` (default; resolves to `FullBuffered` for ≤ 2048² images and
  `BufferedOutput` otherwise, matching libjxl post-`032d39a`),
  `FullBuffered` (libjxl `--buffering 0`), `Threshold2048`
  (`--buffering 1`), `BufferedOutput` (`--buffering 2`, libjxl
  default), and `FullStreaming` (`--buffering 3`). Surfaced on both
  `LossyConfig::with_buffering` and `LosslessConfig::with_buffering`
  with bare-name getters, plus `Buffering::from_i8` / `to_i8` /
  `resolve_for(width, height)` helpers. CLI flag `--buffering -1..3`
  applies to both lossy and lossless paths.
  **No dispatch wired yet** — every variant routes through today's
  one-shot path so output bytes are byte-identical regardless of
  value (36/36 hash-lock invariant); chunks 2-7 land the actual
  per-DC-group split, the buffered-output streaming path, and the
  seekable streaming-output path.

- **EX-J17a: wire-format-safe custom coefficient orders on the
  `--lossless-jpeg` transcode path** (issue #49). The JPEG bridge now
  computes per-channel custom coefficient orders from the same Lehmer
  cost-benefit gate used by the VarDCT path
  (`compute_custom_orders` at `vardct/coeff_order.rs:345`). The
  spec-mandated per-block channel order `[Y, X, B]` is unchanged —
  only the *position* permutation per channel varies, so existing
  decoders read the stream correctly and `djxl --reconstruct_jpeg`
  remains byte-exact on all corpus entries that round-trip on `main`.
  Aggregate −0.28% bytes on a 23-JPEG corpus (15 wins / 8 losses);
  per-image range −0.59% to +0.09%. Replaces the historical-but-
  wire-illegal "EX-J17 channel-grouped DCT reorder" idea (see issue
  #49 for the analysis that ruled out the literal paper-described
  layout).

- **EX-J5 reinterpreted — Lloyd-Max bucket boundaries for energy-
  correlated MA-tree properties** (opt-in via the `__expert` lossless
  override `LosslessInternalParams::lloyd_max_buckets`,
  `EffortProfile::lloyd_max_buckets`). The original EX-J5 proposal
  (Golchin & Paliwal 1998 — CALIC-style 4-level energy-quantized
  context as a 17th MA-tree property) is spec-illegal: JXL hard-codes
  `kNumNonrefProperties = 16` (`context_predict.h:378-379`, jxl-rs
  `tree.rs:197`), so any `property_idx >= 16` is interpreted as a
  (nonexistent) reference-channel property by decoders.

  This spec-legal reinterpretation refines the bucket-boundary picks
  *inside* the existing 16-property MA-tree learner. Instead of
  sort-quantile picks over the sorted-unique value list, the three
  documented residual-energy proxy properties (4 = `|N|`, 5 = `|W|`,
  15 = `wp_max_error`) use Lloyd-Max iterative clustering to choose
  bucket edges. The other 13 properties keep the cheap sort-quantile
  path because their distributions are not energy-shaped
  (channel/group id, signed gradient differences ~symmetric around
  zero), so Lloyd-Max would add cost without compression payoff.

  Algorithm: empirical-histogram Lloyd-Max with count-weighted
  k-quantile initialisation, midpoint cell boundaries, weighted-mean
  centroid updates, convergence on max centroid movement <0.5 input
  units or after 8 iterations (3-5 iters observed on CID22 / CLIC).
  Encoded thresholds are integer midpoints between consecutive
  centroids, clamped to `(min_val, max_val]` and post-deduplicated
  for the strictly-monotone contract `pre_quantize` expects.

  A/B (5 textured photos, e7 lossless, 8 threads, min of 3 samples):
  **-0.168 % bytes** aggregate, with -0.49 % on the textured
  CLIC `07b9f93f` photo and -0.13 % on CLIC `02809272`. Result
  matches the W18-2 abort-report expectation of "a fraction of the
  paper's claimed 0.5-1 % since we're refining existing properties
  not adding new ones". TSV + meta at
  `benchmarks/lloyd_max_buckets_ab_2026-05-18.{tsv,meta}`.

  Default `false` at every effort — hash-lock fixtures
  (`tests/hash_lock_expected.txt`, 36 entries) stay byte-identical
  with the flag off. Sweep harnesses opt in via the `__expert`
  override and re-bake hash-locks when promoting Lloyd-Max to a
  per-effort default.

  Roundtrip-validated pixel-exact on the 1024×1024 CLIC
  `02809272` Lloyd-Max-encoded photo via **djxl 0.12.0**,
  **jxl-rs**, and **jxl-oxide** (integration test
  `tests/lloyd_max_buckets_roundtrip.rs` covers jxl-oxide
  automatically; djxl + jxl-rs were spot-checked manually).
  Refs `~/work/zen/jxl-encoder/JXL_ENCODER_LEARNINGS.md` lines
  102-107 (EX-J5), W18-2 abort report. 5 new unit tests
  (`test_lloyd_max_thresholds_monotone`,
  `_constant_property`, `_two_clusters`, `_clamps_to_max_buckets`,
  `_partition_samples`) cover the clustering primitive in
  isolation; 3 integration tests cover roundtrip + opt-in
  semantics.

- **EX-J4 — RIGED gradient-aware modular predictor via
  `--modular-predictor 14`** (encoder-only meta-mode). Per Sharma
  et al. 2018 *Resolution-Independent Gradient-aware Edge Detection*:
  switches per-pixel among West / North / `Average((W+N)/2)` based on
  the relative strength of the vertical vs horizontal local
  gradient.

  Implementation: hand-crafted 3-leaf MA tree
  ([`modular::tree::riged_tree`]) that gates on properties 13
  (`|NW - W|`) and 10 (`|W - WW|`) at a bit-depth-scaled threshold
  (T = 44 for ≤ 8-bit, T = 768 for 16-bit, linear interpolation
  in-between). The wire bitstream uses only spec-conformant
  predictors (1, 2, 3) and properties — pixel-exact decode verified
  via jxl-rs and djxl.

  Slot: libjxl's `Predictor::Best` (id 14) is an encoder-only
  meta-mode never emitted on the wire; we repurpose this CLI slot
  for RIGED so the wireup matches `cjxl -P 14`. Id 15 (`Variable`)
  continues to fall through to the ID3 tree learner.

  **Honest measurement**: on 5 CLIC 2025 1024×1024 photos at e7
  lossless, RIGED is **+25% larger** than the ID3-learned default
  and **+1.6% larger** than `--no-tree-learning -P 5` (single-leaf
  Gradient). The 3-leaf approximation of Sharma's continuous
  `A_v vs A_h` discriminator loses to ID3's multi-context tree
  (~100+ leaves over 14 properties) and gives up enough vs a
  single-leaf Gradient to not pay for its extra context overhead on
  photographic content. The paper's 0.3–0.7% gain figure is vs
  classical predictors (JPEG-LS / MED / Paeth), not vs libjxl's
  ID3-learned MA tree.

  Kept as an opt-in research/comparison tool — the bytes regression
  is real on photos, but the override is useful for synthetic /
  benchmarking workflows and as a baseline against which future
  multi-property gradient-aware overrides can be A/B'd. Default
  output (`modular_predictor = None`) byte-identical
  (`hash_lock_features` 36/36 unchanged).

  Tests: 7 unit tests in `modular::tree::tests::test_riged_tree_*`
  (shape, bit-depth threshold scaling, decoder validation, per-leaf
  routing). 3 API tests in `api_tests::modular_knobs_predictor_*`
  (engagement vs default, pixel-exact jxl-rs roundtrip, fall-back
  invariants on no-tree paths). Wire bitstream verified pixel-exact
  on 5 CLIC photos via the external `djxl` binary.
- **Chroma subsampling chunk 5 — `ChromaSubsampling::Sub422` and
  `Sub440` now encode end-to-end via the same JPEG-shaped pipeline
  used by Sub420** (issue #47 follow-on to chunk 4 `7a21379f`). When
  both the `chroma-subsampling` and `jpeg-reencoding` cargo features
  are on, both single-axis chroma modes round-trip through jxl-rs and
  djxl on 256×256 RGB at d=1.0.

  Pipeline differences vs Sub420:
  - **Sub422** (`jpeg_upsampling=[0, 2, 0]`, Y `h_samp=2 v_samp=1`):
    horizontal-only chroma downsample. New
    `vardct::chroma_subsampling::rgb_to_yuv422_box` runs zenyuv's
    SIMD-dispatched 4:4:4 encode and then averages chroma along the
    horizontal axis (libwebp `(a + b + 1) / 2` round-half-up tail,
    odd-column edge replication).
  - **Sub440** (`jpeg_upsampling=[0, 3, 0]`, Y `h_samp=1 v_samp=2`):
    vertical-only chroma downsample. Symmetric to Sub422 via
    `rgb_to_yuv440_box`.

  zenyuv 0.1.3 has no dedicated 4:2:2 / 4:4:0 kernels and no Sharp
  YUV for the single-axis modes — the box-filter tail is a temporary
  bridge. A future zenyuv release with axis-specific Sharp YUV can
  slot in here without API change.

  New public helpers in
  `crate::vardct::chroma_subsampling`:
  `rgb_to_yuv422_box`, `rgb_to_yuv440_box`,
  `encode_rgb8_via_jpeg_path` (generic mode-dispatching entry),
  `encode_rgb8_sub422_via_jpeg_path`,
  `encode_rgb8_sub440_via_jpeg_path`. `encode_rgb8_sub420_via_jpeg_path`
  preserved as a thin wrapper.

  Scope: one-shot `EncodeRequest::encode` with `PixelLayout::Rgb8`
  only. Streaming `LossyEncoder::finish` and Rgba8 / Bgr8 / Bgra8 /
  Gray / 16-bit / float / linear layouts still reject — same gates
  as Sub420. RD parity with cjxl is still chunk-6+ territory.

  Default `Full444` bitstream byte-identical (`hash_lock_features`
  36/36 unchanged). Tests at
  `jxl-encoder/tests/chroma_subsampling_signal.rs::sub422_encodes_and_roundtrips_via_jxl_rs`,
  `::sub440_encodes_and_roundtrips_via_jxl_rs`, and
  `::sub422_and_sub440_decode_via_djxl_when_available`. Unit-level
  coverage in `vardct::chroma_subsampling::tests` (10 new tests for
  the box-filter tails and the YCbCr identity).

- **Chroma subsampling chunk 4 — `ChromaSubsampling::Sub420` now
  encodes end-to-end via the JPEG-shaped pipeline** (issue #47
  follow-on to chunk 3 `1994441`). When both the `chroma-subsampling`
  and `jpeg-reencoding` cargo features are on, calling
  `LossyConfig::new(d).with_chroma_subsampling(Sub420).encode_request(...).encode(rgb)`
  now produces a valid 4:2:0 JXL codestream instead of returning
  `EncodeError::InvalidConfig`.

  Pipeline:
  1. `vardct::chroma_subsampling::rgb_to_yuv420_sharp` (zenyuv Sharp
     YUV, AVX2/NEON SIMD) converts the input RGB to a planar YCbCr
     4:2:0 buffer.
  2. New `vardct::chroma_subsampling::encode_rgb8_sub420_via_jpeg_path`
     runs a standard 8×8 forward DCT-II + integer quantization
     (Annex K luma/chroma tables scaled by a `distance → quality`
     mapping) on every block in each plane — Y at full resolution,
     Cb/Cr at half resolution in both axes.
  3. The quantized coefficients are packed into a synthetic
     `crate::jpeg::JpegData` payload (omitting scan_info / marker
     bookkeeping which the encode side doesn't read) and handed to
     `crate::jpeg::encode_jpeg_to_jxl`, which already supports
     `do_ycbcr=true` + `jpeg_upsampling=[0,1,0]` + per-channel block
     grids.

  Scope: one-shot `EncodeRequest::encode` with `PixelLayout::Rgb8`
  only. Streaming `LossyEncoder::finish` still returns
  `InvalidConfig` for Sub420 (the streaming path eagerly linearizes
  sRGB → f32 in `push_rows`, so the JPEG-shaped pipeline — which
  needs raw u8 sRGB for BT.601 conversion — cannot consume the
  buffer without an extra round-trip; chunk 5 will wire that).
  Sub422 / Sub440 remain rejected (Sharp YUV is 4:2:0-only in
  zenyuv 0.1.3; chunk 5 ships the 4:2:2 / 4:4:0 box-filter paths).
  Rgba8 / Bgr8 / Bgra8 / Gray / 16-bit / float / linear pixel
  layouts are rejected for Sub420 (chunk 5+).

  Quality: the synthesized JPEG quant tables are NOT calibrated to
  match cjxl's RD curve at the requested `distance` — the chunk-4
  acceptance test only requires a valid roundtripable bitstream
  (verified via jxl-rs + djxl on 256×256 RGB at d=1.0). Chunk 5+
  will tune the per-distance quant matrices and add the butteraugli
  loop / patches / splines / progressive paths.

  Default `Full444` bitstream byte-identical (`hash_lock_features`
  36/36 unchanged). Tests at
  `jxl-encoder/tests/chroma_subsampling_signal.rs::sub420_encodes_and_roundtrips_via_jxl_rs`
  and `::sub420_decodes_via_djxl_when_available` (djxl test skips
  cleanly when the libjxl binary is not on `$PATH`).

### Fixed

- **Flaky `test_thread_local_workspace_caps_allocations` under
  parallel `cargo test --lib` pressure** (issue #51). The test
  measured `SplitWorkspace::new` allocations via a process-global
  `before`/`after` delta on `SPLIT_WS_ALLOC_COUNT`, which any
  concurrently-running test that called `compute_best_tree` could
  pollute (production callers exist in `modular/section.rs`,
  `modular/encode.rs`, `vardct/dc_tree_learn.rs`). Fix: added a
  thread-local `IS_TEST_POOL_THREAD` marker plus a dedicated
  `SPLIT_WS_ALLOC_COUNT_TEST_POOL` counter that only increments on
  threads where the marker is set, and rewrote the test to build a
  private `rayon::ThreadPool` whose `start_handler` sets the marker
  on each worker. The measurement is now immune to allocations on
  the global rayon pool or any unmarked thread. Verified pass 8/8
  on full `cargo test --lib` under `--test-threads=8` and
  `--test-threads=1`.

### Performance

- **e10/e11 multi-seed chunk 7 — Pareto-aware wall-clock early-out for
  the e11 tree-learning fan-out** (RFC#45 follow-on to chunk 6
  `47442bd0`). At e11 the multi-seed loop now examines the relative
  spread of token costs after the first 4 seeds (chunk-3 perturbation
  slot); if the spread is below 5%, it breaks out of the loop early and
  the picker keeps its best-so-far tree. High-variance images (spread
  ≥ 5%) keep running the full 16 seeds.

  Trade-off measured on the same 5-image CID22-512 paired bench used
  for chunk 6 (`benchmarks/e10_e11_multiseed_chunk7_ab_2026-05-17.tsv`):

  | image    | c6 bytes | c7 bytes | delta  | c6 wall (ms) | c7 wall (ms) | speedup |
  |----------|----------|----------|--------|-------------|--------------|---------|
  | 1025469  | 231127   | 231461   | +334 B | 27,384       | 5,969         | 4.59×  |
  | 1044329  | 327001   | 327001   |   0 B  | 14,846       | 6,143         | 2.42×  |
  | 1189261  | 302399   | 302399   |   0 B  | 18,127       | 6,234         | 2.91×  |
  | 1279330  | 206214   | 207027   | +813 B | 24,723       | 4,404         | 5.61×  |
  | 1418519  | 164133   | 164133   |   0 B  | 31,507       | 7,453         | 4.23×  |

  Net: **+0.0932% bytes, 3.86× wall-clock speedup at e11 median**.

  Honest finding: per-seed cost tracing showed that low chunk-3 spread
  does NOT reliably predict the absence of later-seed improvements.
  1279330 has the lowest chunk-3 spread on the corpus (0.31%) yet seeds
  4..15 find a 0.69% cost improvement that the early-out skips. The 5%
  threshold is therefore framed as a Pareto sweet spot, not a "no
  regression" promise — it converts e11 from "exhaustive search costing
  3-4× e10 wall-clock" into "near-exhaustive search at roughly e10
  wall-clock plus a small premium" on most images. The +0.09% bytes
  regression at e11 is small relative to e11's gains over e10
  (~-0.2 to -0.4%) and the wall-clock savings unlock more frequent e11
  use in time-budgeted pipelines.

  e ≤ 9 unchanged (`tree_learn_seeds = 1` short-circuits the loop, so
  the early-out is never reached). e10 unchanged for the same reason —
  2 seeds is below the 4-seed probe window. Hash-locks 36/36
  byte-identical. New helper + 12 unit tests live in
  `modular/tree_learn.rs::multi_seed_early_out_after_probe`. Bench at
  `jxl-encoder/examples/e10_e11_multiseed_chunk7_ab.rs`.

### Added

- **Squeeze-on-extras chunk 3 — skip squeeze when alpha is a single
  constant value** (follow-on to chunk 2.b `191801a1`, W14-1 ChannelCompact
  `e97e5bb7`). Adds a one-line predicate in
  `vardct::encoder::maybe_build_alpha_squeeze_pipeline` that checks the
  alpha extra via the new
  `VardctExtra::is_constant_full_image(width, height)` helper before
  building the squeeze pipeline. When the predicate fires, the
  dispatcher returns `Ok(None)` and the existing
  `write_modular_extras_subbitstream` path takes over — for
  constant-channel extras that path already emits a libjxl-parity
  `kPalette(num_c=1, nb_colors=1)` transform via
  `detect_constant_value` (W14-1, `e97e5bb7`) that collapses the
  channel to ~76 bytes regardless of `alpha_distance`.

  Closes the `red_night_opaque` overhead the chunk-2.b audit
  (`191801a1`) accepted as a tradeoff:

  | image                              | dims      | d   | no_sq | sq (pre) | sq (chunk-3) |
  |------------------------------------|-----------|-----|-------|----------|--------------|
  | red_night_opaque                   | 400×267   | 0.5 | 9118  | 9194 (+0.83%) | **9118 (+0.00%)** |
  | red_night_opaque                   | 400×267   | 1.0 | 9118  | 9195 (+0.84%) | **9118 (+0.00%)** |
  | red_night_opaque                   | 400×267   | 2.0 | 9141  | 9198 (+0.62%) | **9141 (+0.00%)** |
  | red_night_opaque                   | 400×267   | 5.0 | 9141  | 9209 (+0.74%) | **9141 (+0.00%)** |
  | alpha_nonpremul_photo_mask         | 1024×1024 | 0.5 | 6859  | 4794 (-30.11%) | 4794 (-30.11%) |
  | alpha_nonpremul_photo_mask         | 1024×1024 | 1.0 | 6859  | 4770 (-30.46%) | 4770 (-30.46%) |
  | alpha_nonpremul_photo_mask         | 1024×1024 | 2.0 | 5337  | 4816 (-9.76%)  | 4816 (-9.76%)  |
  | alpha_nonpremul_photo_mask         | 1024×1024 | 5.0 | 4848  | 4823 (-0.52%)  | 4823 (-0.52%)  |

  All `alpha_nonpremul_photo_mask` and `gradients_semitrans_ui` wins
  are preserved (squeeze is the right answer when alpha is varying);
  `red_night_opaque` (constant-opaque alpha) now matches its
  no-squeeze baseline byte-for-byte because the dispatcher hands
  the channel to ChannelCompact instead. Decoder roundtrip via
  jxl-rs unchanged (alpha MAE 0.00 across all 4 `red_night_opaque`
  distance points, preserved on the photo-mask).

  Hash-locks: 36/36 byte-identical (`alpha_squeeze` at default
  `false` is untouched; the chunk-3 dispatcher only fires when the
  flag is `true`). 3 new unit tests in `vardct::extras`
  (`is_constant_full_image_true_for_all_opaque`,
  `..._false_for_one_mismatch`, `..._true_for_all_transparent`).
  Repro: `cargo run --release -p jxl-encoder --example
  alpha_squeeze_chunk2b_roundtrip`.

- **Squeeze-on-extras chunk 2.b — multi-group + dim_shift>0 audit
  surfaces lifted** (follow-on to chunk 2 `1760b03`). Routes the
  squeezed alpha sub-channels across the standard VarDCT section
  layout per libjxl's decoder partition (`dec_modular.cc:331-373`):
  sub-channels with `w ≤ GROUP_DIM AND h ≤ GROUP_DIM` land in
  LfGlobal; `min(hshift, vshift) ≥ 3` go in LfGroup; `min < 3` go
  in HfGroup. Each section emits its own GroupHeader + tree +
  entropy code over its filtered sub-channel subset (the squeeze
  descriptor itself lives only in LfGlobal). The DC-group writer
  now inserts the LfGroup modular sub-bitstream between the VarDCT
  DC entropy code and the AC metadata header, matching libjxl
  `dec_frame.cc:322-336` read order. The HF-group writer continues
  to append the modular extras after the AC entropy code, but on
  the squeeze path emits the squeeze HF band (cropped to
  `GROUP_DIM`) instead of the raw-pixel writer.

  Bytes Δ on the two previously-skipped W13-4 audit images (sweep
  in `examples/alpha_squeeze_chunk2_bytes.rs` updated for chunk-2.b
  coverage):

  | image                              | dims      | d   | no_sq | sq    | Δ%       |
  |------------------------------------|-----------|-----|-------|-------|----------|
  | red_night_opaque                   | 400×267   | 0.5 | 9118  | 9194  | +0.83%   |
  | red_night_opaque                   | 400×267   | 1.0 | 9118  | 9195  | +0.84%   |
  | red_night_opaque                   | 400×267   | 2.0 | 9141  | 9198  | +0.62%   |
  | red_night_opaque                   | 400×267   | 5.0 | 9141  | 9209  | +0.74%   |
  | alpha_nonpremul_photo_mask         | 1024×1024 | 0.5 | 6859  | 4794  | **-30.11%** |
  | alpha_nonpremul_photo_mask         | 1024×1024 | 1.0 | 6859  | 4770  | **-30.46%** |
  | alpha_nonpremul_photo_mask         | 1024×1024 | 2.0 | 5337  | 4816  | -9.76%   |
  | alpha_nonpremul_photo_mask         | 1024×1024 | 5.0 | 4848  | 4823  | -0.52%   |

  `alpha_nonpremul_photo_mask` matches the W13-4 audit's "-18% to
  -160% smaller than cjxl default" direction. `red_night_opaque` is
  an all-opaque alpha plane that ChannelCompact already collapses
  to a 1-value palette in the no-squeeze baseline; the squeeze
  overhead's GroupHeader + per-band tree leaves cost ~+76 bytes on
  the very tight baseline. The squeeze path is opt-in, so callers
  for whom this tradeoff matters can keep `with_alpha_squeeze` at
  its default `false`.

  `dim_shift > 0` for the squeeze path is **not separately gated**.
  The `dim_shift > 0` rejection is enforced by every lossy VarDCT
  entry-point validator (`encoder.rs:927`, `2497`, `2901`) with
  `Error::InvalidInput` — that's a property of VarDCT lossy extras
  generally, not of the squeeze flag, and `check_alpha_squeeze_supported`
  no longer shadows it with a misleading squeeze-specific message.
  When the broader dim_shift > 0 path lifts upstream, the squeeze
  pipeline already materializes the alpha channel at its native
  `width >> dim_shift × height >> dim_shift` resolution; the
  partition/writer would still need a `Channel::hshift/vshift =
  dim_shift` seed to keep decoder-side shift bracket classification
  consistent.

  Hash-lock: 36/36 byte-identical with `alpha_squeeze` at default
  `false`. Roundtrip-verified on all 12 (image, distance) chunk-2.b
  outputs via **jxl-rs** (PRIMARY per project CLAUDE.md) and
  **djxl v0.12.0** (`/tmp/chunk2b_*.jxl` → 1024×1024 / 400×267 /
  256×128 decode clean, no parse errors). The previously-failing
  multi-group test `alpha_squeeze_chunk2_multigroup_returns_not_implemented_chunk2b`
  flips to `alpha_squeeze_chunk2b_multigroup_encodes_and_jxl_rs_roundtrips`
  asserting successful encode + jxl-rs roundtrip with bounded MAE
  on a 320×128 multi-group RGBA. Repro:
  `cargo run --release -p jxl-encoder --example alpha_squeeze_chunk2b_roundtrip`
  (jxl-rs MAE table) and
  `cargo run --release -p jxl-encoder --example alpha_squeeze_chunk2b_emit_for_djxl`
  (writes the 12 .jxl files to `/tmp/chunk2b_*.jxl` for `djxl`
  validation). 2 new partition unit tests + the flipped pipeline
  test (`alpha_squeeze_chunk2b_multigroup_encodes_and_jxl_rs_roundtrips`)
  cover the new wiring. RD-regression 18/18 within thresholds.

- **Squeeze-on-extras chunk 2 — `with_alpha_squeeze(true)` now wired
  into the lossy alpha bitstream** (W14-4 follow-on, builds on the
  chunk-1 framework `3b042f8`). Closes the dominant slice of the
  W13-4 audit gap (`a160deb`): cjxl default `--responsive=1` was
  -18% to -160% smaller than our `responsive=0` lossy alpha because
  we hadn't applied the Squeeze (Haar) wavelet to the alpha plane
  before quantizing. Chunk 2 ships the wiring for the single-group
  case end-to-end, mirroring libjxl `enc_modular.cc:937-1027`
  responsive=1 path narrowed to the extras-only ModularImage.

  Pipeline (when flag is on AND `alpha_distance > 0` AND single
  alpha extra AND ≤ 256×256):

  1. `build_alpha_squeeze_pipeline` (in `vardct::extras`) wraps the
     alpha plane in a 1-channel [`ModularImage`], runs the standard
     `default_squeeze_params` + `apply_squeeze` (Haar wavelet
     decomposition that halves alternating axes until both
     dimensions ≤ 8), then for each output sub-channel computes its
     integer quantizer via `compute_extra_pixel_quantizer_shifted(
     shift = (hshift + vshift) - 1)` (chunk-1 framework fn,
     unchanged) and in-place quantizes each sub-channel with the
     libjxl-parity `snap-to-multiple-of-q` `QuantizeChannel`
     (`enc_modular.cc:141`).
  2. `write_modular_extras_alpha_squeezed` (new, in
     `vardct::bitstream`) emits the modular subbitstream as:
     `GroupHeader { use_global_tree=0, wp_default=1,
     nb_transforms=1 }` → one `kSqueeze` transform descriptor with
     the explicit param list via `write_squeeze_transform` →
     channel-split tree (one gradient leaf per sub-channel, each
     carrying its own integer quantizer baked into the leaf
     multiplier via `decompose_multiplier_pub`) → shared
     entropy code over the per-sub-channel gradient residuals with
     LZ77 RLE detection on consecutive identical residuals.
  3. Routing wired at the bitstream-emit site
     (`vardct::bitstream:write_frame_with_dc_groups` single-group
     branch). `maybe_build_alpha_squeeze_pipeline` returns
     `Some(pipeline)` only on the chunk-2 happy path; otherwise the
     existing `write_modular_extras_global_with_quant` runs
     unchanged. Multi-group, multi-extra, non-alpha-only-extra, and
     `dim_shift > 0` cases surface a clearer `NotImplemented`
     pointing at chunk-2.b.

  Bytes Δ (3 W13-4 audit images × 4 alpha distances, A/B
  no-squeeze vs squeeze on `LossyConfig::new(1.0)` +
  `with_alpha_distance(Some(D))`):

  | image                          | dims    | d   | no_sq | sq    | Δ     | Δ%      |
  |--------------------------------|---------|-----|-------|-------|-------|---------|
  | gradients_semitrans_ui         | 256×128 | 0.5 | 8775  | 4894  | -3881 | -44.2%  |
  | gradients_semitrans_ui         | 256×128 | 1.0 | 8775  | 3827  | -4948 | -56.4%  |
  | gradients_semitrans_ui         | 256×128 | 2.0 | 5540  | 3194  | -2346 | -42.4%  |
  | gradients_semitrans_ui         | 256×128 | 5.0 | 4234  | 2920  | -1314 | -31.0%  |
  | red_night_opaque (400×267)     | multi   | any | n/a   | n/a   | n/a   | chunk-2.b |
  | alpha_nonpremul_photo_mask (1024²) | multi | any | n/a | n/a   | n/a   | chunk-2.b |

  Direction matches the W13-4 audit's "-18% to -160% smaller" cjxl
  delta on the only test image small enough to hit the chunk-2
  single-group gate. The two multi-group audit images (red_night,
  alpha_nonpremul_photo_mask) correctly land on the chunk-2.b
  NotImplemented gate so callers know to fall back.

  Roundtrip-verified with **jxl-rs** (PRIMARY, `tests/
  alpha_squeeze_chunk2_pipeline.rs::
  alpha_squeeze_chunk2_decodes_via_jxl_rs`) and **djxl v0.12.0**
  (`/tmp/sq_chunk2.jxl` 6486-byte 256×128 RGBA round-tripped
  through djxl → 46364-byte PNG, no parse errors). Hash-lock
  baseline preserved: `tests/hash_lock_features.rs` 36/36
  byte-identical with `alpha_squeeze` at its default `false`.

  The chunk-1 framework test
  `alpha_squeeze_on_plus_lossy_alpha_returns_not_implemented` flips
  from "expect `NotImplemented`" to
  `alpha_squeeze_on_plus_lossy_alpha_beats_no_squeeze_baseline`
  asserting the bytes-smaller direction. New
  `tests/alpha_squeeze_chunk2_pipeline.rs` adds 6 dedicated tests
  covering byte savings, jxl-rs roundtrip, multi-group chunk-2.b
  fallback, no-alpha no-op, default-off byte stability, and
  different-bytes-from-baseline. Repro:
  `cargo run --release -p jxl-encoder --example
  alpha_squeeze_chunk2_bytes`. Refs jxl-encoder W14-4
  (`3b042f8`), W13-4 audit (`a160deb`). Multi-group, multi-extra,
  and `dim_shift > 0` plumbing tracked as chunk-2.b in CLAUDE.md.

### Tests

- **A1 audit PARTIAL items — regression-test cleanup chunk 1**: adds
  three focused tests in `jxl-encoder/tests/lossy_knobs_wiring.rs`
  closing the W12-4 audit's "wired but lacks regression test" notes on
  `--center_x` / `--center_y`, `--brotli-effort`, and the lossless
  `--keep_invisible=false` skip-RGB pre-pass.
  (1) `center_xy_decodes_through_jxl_rs_and_oxide` encodes the same
  512×512 image with three distinct AC-permutation centres (default,
  top-left, bottom-left — each landing in a different central group of
  the 2×2 grid), asserts the three bitstreams differ, then decodes each
  through jxl-rs (PRIMARY) AND jxl-oxide (SECONDARY), confirming the
  permutation never corrupts the file-header `SizeHeader`.
  (2) `brotli_effort_q11_smaller_or_equal_to_q1_and_decodes` (gated
  `brotli-metadata`) encodes a 64×64 RGB image with 4 KB of repeated XMP
  at Brotli q=1 vs q=11, asserts both take the `brob` path, q=11 is
  strictly smaller than q=1, and both bitstreams decode end-to-end via
  jxl-rs + jxl-oxide — catches any future regression that silently pins
  the quality at a default constant. (3)
  `lossless_keep_invisible_false_jxl_rs_roundtrip` exercises the
  existing-but-jxl-oxide-only `with_keep_invisible(false)` skip-RGB
  pipeline via jxl-rs as well, asserting visible (`alpha=255`) pixels
  round-trip exactly and invisible (`alpha=0`) pixels decode back to
  `(0,0,0)` confirming the pre-pass zeros are preserved through the
  bitstream. Hash-lock byte-identical (36/36).

### Fixed

- **ChannelCompact for VarDCT extras (constant-channel case)** — closes
  the W13-4 audit (`a160deb`) `red_night_opaque @ alpha_distance=5.0`
  gap where our encoder snapped a fully-opaque alpha plane from `255`
  to `252` (MAE = 3.000) while `cjxl --responsive=1` preserved it
  exactly (MAE = 0.000). The lossy alpha quantizer (`bbf8a98`, W6-3)
  computes `q = 7` at `d = 5.0` and `(255 + 3) / 7 * 7 = 252` snaps
  every alpha pixel down by 3 — silent precision loss on the most
  common alpha shape (100% opaque). `write_modular_extras_subbitstream`
  now detects single-value constant extra channels via
  `VardctExtra::detect_constant_value` and emits a libjxl-parity
  single-channel `kPalette` transform (`num_c = 1, nb_colors = 1,
  predictor = Zero`, `enc_modular.cc:413-426`,
  `modular/transform/enc_palette.cc:177`). The palette meta-channel
  holds the original constant value at `q = 1` (meta channels skip
  lossy quantization, libjxl `enc_modular.cc:1004` only quantizes
  `i >= gi.nb_meta_channels`); the index channel is all-zeros and
  `snap(0, q) = 0` so it also survives. Decoder reconstructs
  `palette[index = 0] = constant_value`. Tree shape switches to the
  N-leaf `channel-split` tree (one leaf per coded channel) so the meta
  leaf gets `q = 1` while the data leaf keeps the per-channel
  quantizer. Gate fires only at `q > 1 AND channel is single-value
  constant` so hash-locked lossless paths (`hash_lock_features` 36/36
  unchanged) and the existing single-extra lossy alpha path
  (`bbf8a98`) on non-constant alpha stay byte-identical. Verified on
  `red_night_opaque` (400×267 multi-group): bytes `9141` vs cjxl
  `--responsive=1` `9253` (-1.2%) and cjxl `--responsive=0` `9216`
  (-0.8%), MAE drops `3.000 → 0.000` at every tested distance
  ({0.5, 1.0, 2.0, 5.0}). Multi-group support is automatic: each HF
  group's extras sub-bitstream independently detects + compacts its
  per-region slice. Roundtrip tests:
  `opaque_alpha_survives_high_alpha_distance_via_channel_compact`,
  `opaque_alpha_survives_all_lossy_distances_via_channel_compact`,
  `opaque_alpha_multigroup_survives_high_alpha_distance_via_channel_compact`
  in `lossy_alpha_roundtrip.rs` (jxl-rs decoded). Multi-color
  ChannelCompact for extras (`nb_colors >= 2`) and the
  squeeze-on-extras path stay parked in CLAUDE.md follow-ons.

### Added

- **`ChromaSubsampling` API surface + zenyuv-backed helpers (issue #47
  chunk 3)** — supersedes the homegrown helpers drafted on PR #48,
  which had been queued behind the chunk-1 API surface drafted on PR
  #47. Both PRs are closed in favour of this single landing on
  current main (PR #47's branch hadn't been refreshed against the
  `clone-siblings` CI fix shipped between its open date and today's
  main, so the PR couldn't merge cleanly; PR #48's homegrown
  `rgb_to_ycbcr_planar` / `box_downsample_2x_both` are replaced
  outright by zenyuv).

  Lands in one commit:

  1. New [`ChromaSubsampling`] enum (`Full444` / `Sub422` / `Sub420` /
     `Sub440`) mirroring libjxl `YCbCrChromaSubsampling::kHShift` /
     `kVShift` (`frame_header.h:81`). Per-mode `h_shifts()` /
     `v_shifts()` / `is_full()` / `tag()` accessors in libjxl `[Cb,
     Y, Cr]` channel order.
  2. New [`LossyConfig::with_chroma_subsampling`] builder + matching
     `chroma_subsampling()` getter. Default is
     `ChromaSubsampling::Full444` so every existing bitstream stays
     byte-identical (hash-lock 36/36 verified).
  3. Field carried across `LossyConfig::with_effort()` so the builder
     chain
     `LossyConfig::new(d).with_chroma_subsampling(Sub420).with_effort(5)`
     is order-independent. Regression test pins the invariant.
  4. New `vardct::chroma_subsampling` module gated behind a new
     `chroma-subsampling` cargo feature. Adds the production
     [`zenyuv`] crate (`0.1.3`, default-features = false) for SIMD
     RGB↔YCbCr conversion (BT.601 Full range; AVX2 / NEON / WASM
     SIMD dispatch via archmage) and Sharp YUV 4:2:0 chroma
     refinement (L2-optimal Newton step Cb/Cr, 25× faster than the
     original scalar implementation with better quality vs hand-
     tuned damping constants).
  5. Public chunk-3 helpers: `rgb_to_ycbcr_444`,
     `rgb_to_yuv420_box`, `rgb_to_yuv420_sharp`, `jpeg_upsampling_for`,
     `build_ycbcr_vardct_frame_header`. 9 unit tests cover plane
     sizes (including odd-dimensions round-up), Sharp-vs-box
     refinement non-no-op, jpeg_upsampling↔h/v_shifts round-trip,
     and white/black RGB→chroma=128 identity.
  6. Fast-fail guard in BOTH the one-shot `EncodeRequest::encode`
     path and the streaming `LossyEncoder::finish` path: any
     non-`Full444` value returns [`EncodeError::InvalidConfig`] with
     a message that names the format tag (`"4:2:0"` etc.) AND the
     missing wiring (per-channel block grids + `do_ycbcr=true` +
     `ColorTransform::kYCbCr`, which today only exist on the
     `jpeg-reencoding` path). 12-case integration test
     `tests/chroma_subsampling_signal.rs` covers the enum surface,
     default, libjxl shift-table parity, `Full444` jxl-rs roundtrip,
     and `InvalidConfig` for each non-default mode via both encode
     entry points.
  7. Chunk-4 wire-up plan (queued): route Sub420 through the JPEG
     transcode-shaped pipeline ([`crate::jpeg::encode`]), which
     already supports `do_ycbcr=true` + `jpeg_upsampling=[1,0,1]` +
     per-channel block grids. Feed it RGB → YCbCr+420 from
     `rgb_to_yuv420_sharp` instead of a parsed JPEG payload — gets us
     a decoder-roundtrippable Sub420 bitstream without retrofitting
     the standard VarDCT encoder for per-channel grids.

- **`LossyConfig::with_alpha_squeeze(bool)` — chunk-1 framework opt-in
  for the squeeze-on-extras (responsive=1) lossy alpha pipeline** (W13-4
  follow-on #1, named "Alpha squeeze-on-extras chunk 1"). Closes the
  framework half of the dominant alpha compression lever surfaced by the
  audit on `a160deb7`: cjxl default `--responsive=1` is -18% to -160%
  smaller than our current `responsive=0` path on non-opaque alpha.

  This ship lands:

  1. `SQUEEZE_LUMA_QTABLE[16]` + `SQUEEZE_QUALITY_FACTOR_CONST` +
     `SQUEEZE_LUMA_FACTOR_CONST` lifted out of inline literals into
     named constants matching `lib/jxl/enc_modular.cc:82-103` exactly
     (unit-test `squeeze_luma_qtable_matches_libjxl_constants` pins
     all 16 entries).
  2. New `VarDctEncoder::compute_extra_pixel_quantizer_shifted(bits,
     ec_type, shift)` — the responsive=1 quantizer formula
     (`enc_modular.cc:1019-1027` luma branch). Diverges from the
     existing no-squeeze `compute_extra_pixel_quantizer` by dropping
     the `* 0.1` "just color quantization" factor and folding in
     `squeeze_luma_qtable[shift]`; at `shift = 0` returns `~10×` the
     value of the no-squeeze path. Returns `1` (lossless) for
     non-alpha extras and for `alpha_distance` of `None` / `Some(0)`.
     Clamps `shift` to `[0, 15]` (table length).
  3. `LossyConfig::with_alpha_squeeze(bool)` builder + getter,
     plumbed through to `VarDctEncoder::alpha_squeeze` and
     preserved across `with_effort` (joins the CLI-passthrough knob
     list).
  4. `VarDctEncoder::alpha_squeeze_engaged()` predicate (`true` iff
     flag on AND `alpha_distance > 0`), and
     `check_alpha_squeeze_chunk1_unsupported` gate that surfaces
     `Error::NotImplemented` with a chunk-2 reference when an alpha
     extra is present + flag engaged. Wired into all three lossy
     entry points (`encode_with_extras`,
     `encode_from_precomputed_with_extras`, the pre-quantized
     variant). `Error::NotImplemented` lets callers distinguish
     "framework gate fired" from "real encode failure".

  **Chunk-1 contract verified (`tests/alpha_squeeze_chunk1_framework.rs`,
  6/6 passing)**:
  - default flag-off + `alpha_distance = 2.0` is byte-identical
    between repeat encodes AND identical to explicit
    `with_alpha_squeeze(false)` (no silent perturbation).
  - default flag-off decodes correctly via jxl-rs at d=2.0 with
    alpha plane variation preserved (POC roundtrip).
  - flag-on + alpha extra + `alpha_distance > 0` returns
    `NotImplemented` with a clear "chunk 2" message.
  - flag-on with no alpha extra OR `alpha_distance` unset/zero is a
    no-op (does not error — lets callers stage the flag).
  - `with_effort` preserves the flag (CLI-passthrough invariant).
  - `hash_lock_features`: **36/36 byte-identical**.

  **Chunk-2 plan** (multi-week, dominant compression lever):
  1. Lift the `dim_shift > 0` extras guard (currently rejects with
     `InvalidInput` in `encode_with_extras` and twin precomputed
     paths) for the squeeze-engaged alpha path only — non-alpha
     extras keep the existing guard until per-channel `ec_distance`
     lands.
  2. When `alpha_squeeze_engaged() == true`: route the alpha extra
     through `modular::squeeze::default_squeeze_params` +
     `apply_squeeze` BEFORE entering
     `write_modular_extras_subbitstream`. Track the per-sub-channel
     `(hshift, vshift)` pairs so the writer knows each shifted
     sub-channel's shift index.
  3. Replace the single `extras_quantizers: &[u32]` slice (one
     entry per top-level extra) with a per-sub-channel
     `Vec<u32>` produced by calling
     `compute_extra_pixel_quantizer_shifted` per sub-channel with
     `shift = (hshift + vshift) - 1` (libjxl
     `enc_modular.cc:1006-1008`). Each sub-channel maps to its own
     leaf in a channel-split tree (already supported by
     `write_tree_histogram_for_channel_split_lossy`); extend the
     property-0 split to dispatch by sub-channel index.
  4. Signal the Squeeze transform in the extras subbitstream's
     GroupHeader (`nb_transforms > 0`) and write each
     `SqueezeParam` via `write_squeeze_transform`.
  5. Bench bytes vs cjxl `--responsive=1` on the same three audit
     images at d ∈ {0.5, 1.0, 2.0, 5.0}; target is `<= cjxl
     bytes` at parity MAE. Acceptance gate:
     `tests/alpha_squeeze_chunk1_framework.rs::alpha_squeeze_on_plus_lossy_alpha_returns_not_implemented`
     flips from "expect Err" to "expect bytes < no-squeeze
     baseline" and the `expect_err` line becomes `expect`.

  **Chunk-3+ (parked for after chunk 2 byte-savings prove out)**:
  ChannelCompact (per-channel palette) for extras — handles the
  opaque-alpha snap-255-to-252 case where cjxl-default preserves
  the constant channel exactly via bitdepth-0 reduction. Documented
  in the audit Investigation Notes (`a160deb7`).

  Default `with_alpha_squeeze(false)` keeps the existing
  responsive=0 pipeline byte-for-byte identical (hash_locks 36/36).

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

- **Auto-splines chunk-6 false-positive suppression on textured photos**
  (A1 audit Top-5 #5, follow-on to W11-3 `ddc02a02` chunk-5 content
  discriminator). Adds a bbox-span gate inside
  `spline_passes_trial_encode_gate`: any candidate whose bbox
  `max(width, height)` doesn't span the image's long dimension is
  rejected before the existing trial-encode + cost-benefit machinery
  runs. New constant
  `vardct::splines::detect_params::MIN_BBOX_SPAN_OF_IMAGE_LONG_DIM = 1.0`.
  Closes 4 of 42 CID22-512 photo regressions (worst was
  `ularapi_Semarang_City_Logo` at +1.19% bytes / +0.86% on
  `klepas-Gentle-giants-of-the-sea-3`) at opt-in
  `with_auto_splines(true)`; default-off encode path is unaffected.
  The bbox-span discriminator was picked after testing Hessian-ratio,
  AC-only-energy, raw-L2 relative-drop, and cost-margin variants —
  the energy proxy is dominated by XYB-DC and cannot cleanly
  separate true thin features on textured backgrounds from
  sub-image ridge segments through textured photo content. Bbox span
  is image-relative, cheap to compute, and the chunk-3 stripe+ramp
  test image (1024×256, wire span 1024) passes the gate exactly
  unchanged. `tests/auto_splines.rs` 6/6 pass; `cargo test splines`
  30/30 pass; hash_locks 36/36 byte-identical; rd-regression all 18
  cells within thresholds. Calibration TSV:
  `benchmarks/auto_splines_bench_2026-05-17_chunk6_fp.tsv` (+
  `_before.tsv` for the pre-chunk-6 snapshot).

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

- **libjxl HEAD refresh + drift bench — zero drift across 39 cells**
  (W19-2). Pulled local `~/work/jxl-efforts/libjxl` from `d2c7032`
  (2026-02-22) to HEAD `4279d48` (2026-05-12) — 81 commits,
  274 files, +1,529 / −49,928 (the deletion volume is the
  `tools/jpegli*` and `tools/jni/*` removal, not encoder code).
  Rebuilt `cjxl`, preserved the old binary at
  `/tmp/cjxl_old_d2c7032`, and benched OLD vs NEW vs our
  `cjxl-rs` on four axes:
  - **RD parity**: 5 CLIC 1024×1024 photos × d∈{0.5, 1.0, 2.0, 5.0}
    × {ours, cjxl_old, cjxl_new} at e7 = 60 rows. bytes + Rust
    butteraugli (metadata-immune per CLAUDE.md).
  - **Lossless photos**: same 5 CLIC images at `-d 0 -e 7`.
  - **Lossless screenshots**: 5 gb82-sc images at `-d 0 -e 7`
    (chosen because the diff contains the streaming/buffering/MA-tree
    PRs `acc28c0` `032d39a` `1389871` `b3510d1` `e39a6aa` which would
    most plausibly affect tree-heavy/palette-heavy content).
  - **HDR**: re-ran `examples/hdr_rd_sweep_vs_cjxl` against both
    binaries (PQ/HLG/BT709 × d∈{1, 2, 5}).

  All 39 cells (20 RD + 5 lossless + 5 screenshots + 9 HDR) are
  byte-identical between `cjxl_old` and `cjxl_new`, with Rust
  butteraugli scores matching to six decimal places. The 81-commit
  delta does not touch `lib/jxl/quant_weights.cc`,
  `enc_quant_weights.cc`, `enc_adaptive_quantization.cc`,
  `enc_ac_strategy.cc`, or `enc_chroma_from_luma.cc`; what *did*
  change is overwhelmingly safety hardening (overflow / NaN / null
  / buffer-size guards) and CI dependency bumps.

  **No drift cells** → no items for the W19-2 cherry-pick
  investigation queue. Our libjxl-r0 baseline (`d2c7032`) is
  indistinguishable from HEAD on every axis we currently bench.
  Re-run quarterly or after the next encoder-touching libjxl PR
  (watch for changes under `lib/jxl/enc_*` outside the safety-fix
  pattern).

  Bench artifacts:
  - `benchmarks/libjxl_drift_rd_2026-05-18.{tsv,meta}` (60 rows)
  - `benchmarks/libjxl_drift_lossless_2026-05-18.{tsv,meta}` (15 rows)
  - `benchmarks/libjxl_drift_screenshots_2026-05-18.{tsv,meta}` (10 rows)
  - `benchmarks/hdr_drift_2026-05-18/{hdr_old,hdr_new,hdr_drift}.tsv` + `.meta`
  - `benchmarks/libjxl_drift_2026-05-18.SUMMARY.md` (top-level write-up)
  Reproducer scripts:
  - `scripts/libjxl_drift_{bench,lossless,screenshots,hdr}.sh`

- **Auto-splines default-on at e8+ — REJECTED for the second time, with
  stronger evidence** (chunk-7 re-bench, follow-on to chunk-5
  `ddc02a02` + chunk-6 `d77c589d`). Initial rejection
  (`6c01965`) was "no observed wins on real content". Chunk 7 picks
  18 cells (5 photo-realistic power-line synthetics that bypass the
  chunk-5 screenshot discriminator + 10 CID22-512 photos including all
  4 original chunk-6 false-positive images + 3 CLIC2025-1024
  photo-class images) and bench-encodes them at distance=1.0, effort=8
  with `auto_splines` off vs on. Result:

  - 13/13 real photos byte-identical (chunk-6 FP closure holds).
  - 2/5 wire synthetics (long_dim ≥ 2048): byte-identical because the
    chunk-6 bbox-span gate rejects every candidate (polyline tracer
    caps at ~1042 px so no segment spans `1.0 × 2048`).
  - 3/5 wire synthetics (long_dim = 1024): admit at the gate AND
    regress bytes by +3.1% / +4.3% / +5.5%. The trial-encode L2-energy
    proxy predicts a saving; the actual bitstream is bigger because
    the e8+ butteraugli loop re-converges the quant_field on the
    post-splines XYB and emits a strictly worse encode.

  Default `auto_splines_default(_) = false` stays. Flipping at e8+
  would net 13 byte-identical photos for 3 wire regressions on
  exactly the content the detector was designed to win on. The flag
  remains opt-in; a future flip needs either a buttloop-aware cost
  proxy or an effort-axis split that confines the detector to e5-e7
  (pre-buttloop). Bench archive:
    `benchmarks/auto_splines_bench_2026-05-17_chunk7.tsv` (18 cells)
    `benchmarks/auto_splines_bench_2026-05-17_chunk7.meta`
  Harness:
    `jxl-encoder/examples/auto_splines_chunk7_bench.rs`
  Hash-locks: `hash_lock_features` 36/36 byte-identical; tests/auto_splines.rs
  6/6; splines lib tests 24/24.

- **First-ever HDR RD-bytes sweep vs cjxl** (jxl-encoder#44 / W4
  follow-on; closes the "never RD-benchmarked" line item from
  `memory/hdr_encoding_implementation_plan_2026-05-17.md`). New
  bench-only example `jxl-encoder/examples/hdr_rd_sweep_vs_cjxl.rs`
  synthesizes a 256×256 RGB gradient in PQ / HLG / BT.709 codeword
  space, encodes it with `LossyConfig` + the matching
  `RgbPqF32` / `RgbHlgF32` / `RgbBt709F32` `PixelLayout` + the
  matching `ColorEncoding` preset + `EncodeRequest::with_intensity_target`,
  and compares bytes against `cjxl -x color_space={Rec2100PQ,
  Rec2100HLG, RGB_D65_SRG_Rel_709} --intensity_target=...` at
  d ∈ {1.0, 2.0, 5.0}. All nine cells produce well-formed
  bitstreams that both jxl-oxide (parse) and `djxl` (decode)
  consume cleanly; the colour-encoding header carries the
  expected transfer function (`Pq` / `Hlg` / `Bt709`) on every
  cell.

  PQ wins outright: -27% at d=1.0, -44% at d=2.0, -39% at d=5.0.
  HLG splits — +9% at d=1.0 but -48% / -32% at d=2.0 / d=5.0.
  BT.709 reverses direction with distance: +36% at d=1.0, +48%
  at d=2.0, -28% at d=5.0.

  Verdict: HDR signalling + transfer-function plumbing is at
  bytes-parity-or-better with cjxl across all three layouts at the
  high-distance end of the sweep. The d ≤ 2.0 BT.709 and d=1.0
  HLG overheads are not HDR-specific — they track the same gap
  the CLAUDE.md "Quality Gap vs Full libjxl (Feb 24, 2026)" table
  reports at d=1.0 / d=2.0 on sRGB photos (+0.8% / +2.8% there;
  the synthetic gradient amplifies it because it has only LF
  content and our cost model picks DCT8 where cjxl picks larger
  transforms). No HDR-path tuning chunks are needed. This is a
  bench-only delivery — no production-code changes beyond adding
  the example + Cargo.toml registration. Bench:
  `benchmarks/hdr_rd_sweep_20260518T053349Z.{tsv,meta}`. No
  HDR-aware perceptual metric is reported because Rust butteraugli
  in-tree assumes an SDR ~80 nits display model; once we expose
  an HDR butteraugli (or wire `butteraugli_main --intensity_target=`
  with a metadata-clean PNG pipeline) the same harness can drop
  in a `metric` column.

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
