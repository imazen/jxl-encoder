# libjxl alg-to-alg parity tracking — lossless tree learning

Source-grounded comparison of the modular MA-tree learning pipeline
against libjxl (shallow clone at `~/work/jxl-efforts/libjxl`, commit
d089091a, 2026-08-11). Started 2026-08-15 for the memory/wall/RD parity
goal. Update whenever either side's algorithm or measured numbers move.
Measured numbers: 3840x2160 mosaics, t=1, macOS M4 Pro
(benchmarks/jxl_probe_prune_2026-08-15.md + jxl_dedup_refine_2026-08-15.md).

## Per-unique-sample accumulator layout

| component | libjxl (enc_ma.h:24-138) | ours (tree_learn.rs TreeSamples) | parity |
|---|---|---|---|
| residual tokens | 2 B/pred: {u8 tok, u8 nbits}, extra bits NOT stored | 2 B/pred: tokens u8 + ebits u8 columns | ≈ parity per predictor |
| predictor count | **2** at all default lossless efforts (`Predictor::Best` = {Weighted, Gradient}, enc_modular.cc:642-644); 14 only at Glacier `Predictor::Variable` | 14 always (pre-2026-08-15); `JXL_GLOBAL_TREE_PREDICTORS=auto` probe-tree selection keeps 7-9 | **structural gap, now bridged by probe-tree pruning; our 7-9 > their 2 is an RD choice (we beat cjxl bytes 6-46%)** |
| properties | u8 PRE-QUANTIZED buckets at gather (≤256, QuantizeProperty LUT enc_ma.h:109-119); statics (channel, group) u32 | e8+: u8 buckets at gather (exact distinct-value pre-walk = byte-identical thresholds; raw columns never materialize — 2026-08-15); e≤7: raw i16/i32 → pre_quantize, waved free | **BRIDGED at e8+** (different mechanism, same effect: their subsampled pre-pass thresholds vs our exact pre-walk — ours keeps bytes identical) |
| dedup | streaming 2-position open hash at gather, u16 counts (saturate+evict) | post-gather packed-key sort (adaptive 2-byte+refined partitions), u32 counts | ours measured faster than our own streaming port (+3-8% wall); count width 2 B theirs vs 4 ours |
| per-unique bytes | e7 19 B, e9 28 B (P=2) | e7 ~35 B, e9 ~46 B at 14 preds; ~21-27 B at auto 7-9 | bridged by auto |

## Sampling density (GatherTreeData, enc_encoding.cc:105-207)

- libjxl: Bernoulli per-pixel gate (xorshift128+), pixel_fraction =
  nb_repeats: **e7 = 0.50, e9 = 0.65** (0.5 default × per-tier mult,
  enc_modular.cc:561-597); WP/property state updates every pixel.
- ours: fixed-stride subsample, e7 stride 2 = 0.50, e9 similar. DENSITY
  AT PARITY; the mechanism differs (their Bernoulli cannot alias; our
  fixed stride can — mitigated by the default-on cost-based self-repair,
  #24, which libjxl does not need).
- their threshold pre-pass (CollectPixelSamples, enc_ma.cc:967-1029):
  geometric-skip 10% of final density (5% of pixels at e7) feeding
  property quantization only.

## Split search (FindBestSplit, enc_ma.cc:167-500)

Both sides: per-(property, predictor) bucket histograms + one ascending
sweep maintaining above/below counts, entropy via SIMD estimate over i32
histograms; candidate splits = quantized bucket boundaries; samples
physically reordered per node (theirs swap-partition; ours swap OR
stable-gather by cost model). Their per-node cost ∝ 2 predictors; ours ∝
kept set. Their acceptance: cost + threshold < base_bits with
node_threshold = 75 + 14*speed_tier (+10*decoding_speed_tier) scaled by
sampled fraction; predictor-change penalty 800/(100+threshold); WP -eps,
Zero +eps nudges; fast_decode_multiplier preference for no-WP/static
splits. Ours: differs (no direct equivalent of the decode-speed
preferences) — NOT yet at parity, tracked as a future RD/decode-speed
item.

## Standing measured gaps (4K, default path, 2026-08-17)

Memory (t=1, heaptrack-verified ladder, benchmarks/jxl_wall_parity_2026-08-16.md):

| axis | cjxl 0.12 | ours 2026-08-14 | ours 2026-08-17 | status |
|---|---|---|---|---|
| lossless e7 photo peak_live | ~290-306 MB RSS | 834 MB | **548 MB** | 1.8× over; next: token-column narrowing + image floor |
| lossless e9 photo peak_live | ~306 MB | 1130 MB | **687 MB** (screen 676) | ebits columns eliminated (LUT), exact bucketize, probe-prune |
| lossless bytes | baseline | −6.2% photo, −36..46% screen | same | we win |

Wall (best-of-3, cjxl 0.12 same box; lossless from round 1c/2, lossy round 3):

| cell | cjxl | ours | ratio |
|---|---|---|---|
| lossless e7 t1 | 3.82 s | 6.51 s | 1.70× |
| lossless e9 t1 | 23.1 s | 38.2 s | 1.65× |
| lossless e7 t8 | 0.65 s | 2.74 s | 4.5× |
| lossless e9 t8 | 3.5 s | 18.2 s | 5.2× |
| lossy e3 t1 / t8 | 0.15 / 0.07 | 0.21 / 0.08 | 1.40× / 1.14× |
| lossy e5 t1 / t8 | 1.47 / 0.26 | **1.38** / 0.36 | **0.94× win** / 1.38× |
| lossy e7 t1 / t8 | 2.63 / 0.52 | **1.92** / **0.44** | **0.73× / 0.85× — wins** |

2026-08-28 sectioned standing on the aarch64 laptop (cjxl v0.12.0 NEON, photo
3840×2160 crop, `benchmarks/jxl_sectioned_prune_k_2026-08-28.meta`):

| cell (lossless, SectionedTrees::On) | cjxl | ours | ratio | bytes vs cjxl |
|---|---|---|---|---|
| e7 t=1 | 7.25 s | 11.5 s | 1.59× | +0.84 % |
| e7 t=8 | 1.12 s | 1.85 s | 1.65× | +0.84 % |
| e9 t=1 | 42.1 s | 45.3 s | **1.08×** (≤ 1.3× met) | +0.47 % |
| e9 t=8 | 6.03 s | 6.62 s | **1.10×** (≤ 1.3× met) | +0.47 % |

e7 phase split (`jxl_sectioned_phases_2026-08-28.tsv`, t=1, 11.5 s): per-group
tree learn 8.5 s (find_best_split 4.7 s, partition 1.2 s, dedup 0.6 s,
pre_quantize 0.5 s), gather 0.9 s, ANS build 0.6 s, collect 0.5 s, RCT 0.4 s,
write 0.25 s, LZ77 0.1 s, patches 0.1 s. The e7 gap is the split search over
K=8 kept predictors (libjxl learns over 2); K=6 / K=4 measured (−13 % / −27 %
wall on photo, but +0.20 % / +1.35 % bytes on imac_dark) and NOT defaulted.
Single-worker pools now bypass the fork engine (byte-identical, −3.5 % learn
wall on both paths). Group size 128–1024 measured: 256 stays
(`jxl_sectioned_group_size_2026-08-28.meta`).

2026-08-18 lossless-t8 update (rounds 6-8, all byte-identical): the
work-stealing RefCell crash fixed; dedup refinement scatter, tensor
build, prequant bucketize + pre-walk, and the per-group LZ77 transform
parallelized; FBS floors lowered; unbounded subtree forking (budget-slack
proof + sequential fallback). x64 4K e9 t8: 18.0 -> 11.3 s (7.3x -> 4.6x
cjxl); e7 t8 ~2.96 s (5.6x). Honest-photo lossy ladders (round 5): e5
1.11x t1 / 1.30x t8. Remaining structural items: (1) lossless t8 tree
CHAIN (giant skewed nodes; intra-node parallelism saturated) — the
sectioned/per-group mode is the measured lever (-41..-66% wall at ~0%
median bytes; Auto-policy extension awaiting owner sign-off); (2) u16
saturating dedup counts; (3) decode-speed-aware split preferences;
(4) lossy e5-t8 residue (acstrat kernel cost, candidate set at parity;
DCT4X4-at-e5 KNOWN-GAP needs its own RD study).

## Memory-model standing (2026-08-27, imazen/jxl-encoder#96 estimator arm)

Allocator-agnostic `peak_live` (counting global allocator, input buffer
included), real content, `benchmarks/jxl_sectioned_mem_2026-08-27.tsv`
(macOS laptop, jxl-encoder d7fc8f7e; `scripts/mem_sectioned_sweep.sh`):

| cell (RGB8, lossless, peak_live MiB) | global t=1 | sectioned t=1 | sectioned t≥4 |
|---|---|---|---|
| photo 3840×2160 e7 | 786 (96.4 B/px marginal) | **404** (47.9 B/px; was 518 before the 2026-08-28 patches-scan fix) | 404 (48.0 B/px) |
| photo 3840×2160 e9 | 1029 (127.0 B/px) | 404 (was 518) | 404 (t=12: 468) |
| photo 4000×3000 e7 | 1117 (94.6 B/px) | **584** (46.9 B/px; was 855) | 584 (48.0 B/px) |
| photo 4000×3000 e9 | 1517 (129.5 B/px) | 584 (was 855) | 584 |
| reddit.com 1313×8008 e7 | 888 | **511** (48.0 B/px since the 2026-08-30 patches-lifetime fix; was 650 / 61.8 B/px) | 512 (48.0 B/px) |
| imac_dark 2940×1912 e7/e9 | 475 / 754 (85.6 / 137.7 B/px) | **274 / 274** (48.2 B/px since 2026-08-30; was 340–349 / 58.9–62.2 B/px; 96/96 local sections since 2026-08-28 — was a global fallback with 0 local sections, `jxl_sectioned_mem_meta_2026-08-28.tsv`) | 275 / 275 (e9 t=12: 321) |

Consequences recorded in `heuristics.rs`:

- `estimate_encode_sectioned` (new, `pub(crate)`): `input + fixed(e) +
  floor(threads)·px + per_thread(e)·(t−1)` — fixed 8/32 MB (e7/e9),
  floor 80 B/px at t=1 / 68 B/px multi-threaded, per-thread 12/36 MiB
  (one 256² group's learn; `parallel-tree-learning` forks inside it).
  TYP covers every sectioned-engaged cell (< 2.5× at ≥ 2 MP), including
  palette/ChannelCompact/patches content since the 2026-08-28 meta-channel
  arm (stream 0 learns its own tree from the meta channels; the patches
  dictionary rides in LfGlobal as on the global path). MAX (1.8×) still
  covers the 2026-08-27 fallback peaks so a regression to the whole-image
  tree stays inside the admitted envelope.
- **Whole-image band RECALIBRATED (2026-08-28)**: `LOSSLESS_BPP_TREE = 540`
  was anchored on the 2026-08-01 12 MP cell (490 B/px) BEFORE the thirteen
  August reductions. The three-class grid
  (`benchmarks/jxl_lossless_band_2026-08-28.{tsv,meta}`: photo 64² → 12 MP,
  imac_dark, reddit; e5–e10 pre-shift labels; rgb + rgba) now pins base 92
  / e6 92 / e7–e8 128 / e9 160 / multi-seed 160 B/px with effort-dependent
  intercepts (16 / 24 / 64 / 160 MiB) and alpha +72 B/px. (2026-08-29
  ladder shift, issue #45: the multi-seed band's grid label "e10" is
  today's e11; `heuristics.rs` band boundaries moved with it.) `Auto`'s memory-pressure gate and
  `LosslessConfig::estimate_encode` are ~4× lower for e7–e9 (12 MP e7 TYP
  6.4 → 1.6 GB against 1.14 GB measured).
- **Sectioned t=1 excess — ATTRIBUTED AND REMOVED (2026-08-28)**: the
  extra size-growing single-worker phase (+114 MiB at 8.3 MP, +271 MiB at
  12 MP, identical at e7/e9) was the lossless patches detector's
  single-thread connected-component scan (`vardct/patches.rs`): its
  flat-index DFS stack grows through doubling reallocs on photo content
  (one foreground component) and sat at the encode peak; at t ≥ 2 the
  bounded union-by-min labeling + per-CC replay path ran instead. A/B via
  `MEM_PROBE_PATCHES=0` (`benchmarks/jxl_sectioned_mem_t1excess_2026-08-28.{tsv,meta}`)
  pinned it; the labeled path now runs at ≥ 1 MP on every thread count
  (bytes identical — hash-locks, the lossy/lossless patches fixtures and
  imac_dark/reddit/photo bytes unchanged). 12 MP e7/e9 sectioned t=1:
  855 → 584 MiB; 8.3 MP: 518 → 404 MiB. The sectioned floor is one value
  for every thread count now (`SECTIONED_BPP_THREADS1` = `_MULTI` = 68).
- **Patches-phase lifetime — MEASURED, ATTRIBUTED AND REMOVED (2026-08-30,
  the last #96 memory-residual item)**: on screen content the patches
  DETECTION working set sat AT the sectioned encode peak at every thread
  count — `MEM_PROBE_PATCHES` A/B: imac_dark +76 MiB, reddit.com
  +138.5 MiB (≈ +13.8 B/px); zero on photo at every size (the tree phases
  out-peak it there). Attributed with the new in-repo `mem_probe`
  alloc-sites mode (`JXL_ALLOC_SITES=1`, the zenjxl methodology ported):
  at the peak instant the detector held its u8→f32 conversion planes
  (12 B/px), the flood-fill planes and a **2× over-sized BFS seed queue**
  (127 MiB on imac — a leftover of the pre-2026-08-28 single-FIFO design)
  on top of the already-built whole-image i32 `ModularImage`. Fixed
  byte-identically (105-cell grid, incl. the content-adaptive e5/e6
  patches arms): detection now runs BEFORE the `ModularImage` build
  (`api.rs::encode_lossless_single`, layout-derived gate) and the seed
  queue is sized exactly (`vardct/patches.rs`). Screens now sit on the
  photo floor: imac 280985 KiB / reddit 523716 KiB ≈ 48.0–48.2 B/px
  (`benchmarks/jxl_sectioned_patches_lifetime_2026-08-30.{tsv,meta}`).
  Post-fix peak composition on imac (alloc-sites): 193 MiB = NINE
  whole-image channel clones from `select_best_rct` — the next
  sectioned-peak lever, tracked in the #96 residual follow-up issue.
