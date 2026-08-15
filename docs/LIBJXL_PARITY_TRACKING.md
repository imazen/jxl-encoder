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

## Standing measured gaps (4K, t=1, default path)

| axis | cjxl 0.12 | ours 2026-08-14 | ours now (auto) | status |
|---|---|---|---|---|
| lossless e7 photo peak_live | ~290-306 MB RSS | 834 MB | **564 MB** | 1.8-1.9× over; next: gather-time bucketization (their pre-pass model) |
| lossless e9 photo peak_live | ~306 MB | 1130 MB | **849 MB** (screen 786) | raw props removed (exact bucketize); next: token columns + image floor |
| lossless e7 photo wall | 3.82 s | 11.3 s | **8.1 s** | 2.1× |
| lossless e9 photo wall | 23.1 s | 54.8 s | **46.4 s** | 2.0× (sectioned mode: 30.2 s = 1.31×) |
| lossless bytes | baseline | −6.2% (photo) −36..46% (screen) | ≈ same (auto ±0.1%) | we win |

Remaining structural items to parity: (1) gather-time property
bucketization via a threshold pre-pass (kills raw prop columns AND
narrows dedup keys — their design); (2) u16 saturating dedup counts;
(3) decode-speed-aware split preferences (RD/decode axis, not memory).
