# Per-group predictor pruning + WP-cache fusion + cbrt SIMD — 2026-08-14

Host: macOS/M4 Pro, threads as stated, zenjxl mem_probe_encode
(--features encode,decode,parallel), inputs ~/tmp/4kmem/inputs/*.bin
(the session's 4K photo/screen mosaics). Encoder commits: fusion
ecf83e55, cbrt a5d1dfe3, pruning d2444010. cjxl 0.12.0 reference from
libjxl_vs_zenjxl_4k_2026-08-13.tsv.

## 1. WP-cache fusion (ecf83e55, byte-identical)

(wp_pred, wp_max_error) depend only on (image, wp_params); gather and
collect walk the same pixels. Gather records, collect reads → ONE WP
state-machine walk per group instead of two (hybrid: global-collect
fills, local rewrite reads). Verified exact on sectioned photo e7
4,346,224 / hybrid photo e7 4,336,683 / hybrid screen e7 282,053.
Sectioned photo e7 wall 10.9 -> 9.95 s (−9%). peak_live unchanged.

## 2. Predictor pruning (d2444010, sectioned default K=8)

Root-cost score per predictor (dedup-weighted token histogram through
estimate_bits_u32 + ebits) over the group's gathered samples; keep the K
cheapest, Weighted always retained; drop pruned columns before the learn.

| cell (t=1) | off bytes | off wall | K=8 | K=6 |
|---|---|---|---|---|
| photo e7 sectioned | 4,346,224 | 10.2 s | −0.03% / 7.59 s (−25%) | +0.01% / 6.68 s (−34%) |
| photo e9 sectioned | 4,271,051 | 40.7 s | +0.04% / 30.2 s (−26%) | +0.06% / 26.4 s (−35%) |
| photo e7 hybrid | 4,336,683 | 19.5 s | — | +0.005% / 16.6 s (−15%) |
| screen e7 hybrid | 282,053 | 7.10 s | — | +0.16% / 7.03 s (−1%) |

Default: sectioned K=8 (byte-flat), hybrid full-14 (RD-max mode).
JXL_TREE_PRUNE_PREDICTORS=K overrides both; >= 14 disables. djxl decodes
the pruned default stream PIXEL-EXACT (4K photo); zenjxl-decoder
roundtrips at K=2 and K=14 (env_overrides tests). Caveat: two 4K
content classes measured; the wider-corpus audit rides #96's existing
default-flip item.

## 3. cbrt f64x4 vectorization (a5d1dfe3, byte-identical, lossy)

forward_xyb's per-lane scalar f64 Newton cbrt (2 divides/element = the
latency chain) now runs both iterations in f64x4 with the exact scalar
op order. 4K lossy e3 d1.25 t=1: xyb 60.7 -> 36.2 ms, encode_inner
217 -> 190 ms (−12%), bytes identical. Applies at every lossy effort.

## Sectioned mode, cumulative (fusion + K8 default)

| cell | 2026-08-13 | now | Δ | cjxl 0.12 | ratio |
|---|---|---|---|---|---|
| photo e7 t=1 | 10.9 s | **7.96 s** | −27% | 3.82 | 2.1× |
| photo e9 t=1 | 42.5 s | **30.2 s** | −29% | 23.1 | 1.31× |
| photo e7 t=8 | 1.84 s | **1.44 s** | −22% | 0.61 | 2.36× |
| photo e9 t=8 | 6.33 s | **4.58 s** | −28% | 3.51 | **1.30×** |

Bytes at each cell within ±0.04% of the pre-round sectioned stream
(still smaller than cjxl's 4,732,216 lossless e7 by ~8%); peak_live
unchanged at 469 MiB (the C-parity memory mode).

## e3 lossy attribution (for the remaining 1.5× vs cjxl falcon)

4K photo d1.25 t=1 phase split after the cbrt fix: xyb 36 / xform 45 /
entropy 85 (build_codes 46 + pass2_write 26 + ac_tok 15) / cfl1 16 ms.
build_codes mirrors libjxl's own approximate shift search (already
effort-gated Approximate <= e7); no missing falcon gate identified from
our side — remaining candidates are implementation-throughput work
(build_codes internals, BitWriter write path), not strategy.
