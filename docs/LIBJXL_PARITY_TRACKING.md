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
