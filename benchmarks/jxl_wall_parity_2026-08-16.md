# Wall-parity round 1 — toward <= 1.3x cjxl — 2026-08-16

Host: macOS/M4 Pro, zenjxl mem_probe_encode (encode,decode,parallel —
which now implies jxl-encoder/parallel-tree-learning), 4K mosaics t=1
unless noted. cjxl 0.12.0 references (same box, 2026-08-13 meta): photo
lossless e7 3.82 s / e9 23.1 s (t=1), 0.61 / 3.51 (t=8).

## Landed (all byte-identical, suite 12/12)

1. zenjxl `parallel` now enables `parallel-tree-learning` (production
   fix — the lib default lacks it, so production t=8 lossless learned
   trees SEQUENTIALLY): photo e7 t=1 8.4 -> 7.86 s (the borrowed path
   is faster even at t=1), t=8 7.9 -> 3.04 s; e9 t=8 -> 21.6 s
   (t=8 peak_live rises to ~1.3 GB — parallel subtree learns).
2. Wire-stream token reuse: the global writer already collects the
   whole-image token stream for cost scoring + histogram building; the
   state now carries the FINAL post-LZ77 wire stream with per-section
   ranges, and per-group section writes emit their slice directly — no
   second per-group collect (a full WP walk) and no per-section LZ77
   re-apply. Cliff e7 t=1: 8.98 -> 7.71 s (-14%), the
   collect_residuals_per_group phase (12%) eliminated.
   TRAP (cost a failing e9 noise test): all_tokens is SHADOWED by the
   LZ77-transformed stream mid-writer while group_ranges described the
   raw layout — the state must store the (stream, ranges) PAIR from the
   same side of the transform. Guard: reuse only when
   num_passes == 1 && ranges.len() == num_groups.

## REFUTED: skip-dedup on high-unique content

Hypothesis: post-gather dedup (1.19 s at e7) costs more than its row
reduction saves on photo-like content (75 % unique). Byte-identical
skip implemented (TreeLearningParams::skip_dedup) and measured:
photo e7 t=1 7.55 -> 11.6 s (+54 %!). The packed-key SORT is a
cache-layout transform the split search depends on — key-sorted rows
give the accumulate loops bucket-coherent access; unsorted unmerged
rows are cache-hostile. At e9 the skip measured a small WIN
(46.3 -> 44.9 s, +150 MB peak) — unexplained asymmetry (e9 gathers
pre-bucketized columns; e7 raw). Do NOT re-add a dispatch without
explaining it. JXL_SKIP_DEDUP=1 stays as the A/B hatch. The
thread-aware variant also mis-read rayon's GLOBAL pool width at t=1
(effective_threads() sees the 12-wide default pool, not the encode's
1-thread config) — effective_threads is NOT a per-encode signal.

## Standing after round 1 (photo, bytes identical everywhere)

| cell | cjxl | ours | ratio | target 1.3x |
|---|---|---|---|---|
| e7 t=1 | 3.82 | **7.55** | 1.98x | 4.97 |
| e9 t=1 | 23.1 | **46.3** | 2.00x | 30.0 |
| e7 t=8 | 0.61 | **3.00** | 4.9x | 0.79 |
| e9 t=8 | 3.51 | **21.6** | 6.2x | 4.56 |

t=8 phase profile (e7, wall 3.07): the learn is 1.63 s with
find_best_split at ~1.5-thread effective parallelism — the subtree
fan-out cannot help the root levels. Next structural items, in order:
within-node parallel histogram accumulation for large nodes
(byte-identical: order-free integer adds; attacks BOTH t=8 tails),
learn-core width (predictor set K, corpus-gated), pre_quantize +
gather micro-costs, probe gather parallelization.

## Round 1b: within-node data-parallel inner loops (byte-identical)

fbs_accumulate now fans BUCKETS across the pool for nodes >= 256k rows
(disjoint count_increase slices — identical under any order), and the
stable-gather partition fans COLUMNS for nodes >= 1M rows (per-worker
scratch via for_each_init). Measured t=8 e7 3.06 / e9 21.6 — WITHIN
NOISE of before. Attribution: the t=8 critical path is spread across
the still-sequential per-prop passes — the per-prop bucket counting
sort (`sorted_by_bucket`, O(n) x props per node level), the swap
partition (taken on lopsided root splits where the gather variant's
cost model declines), derive_child_tensors, and the sweep. The
per-PROP outer parallelization (own workspace per prop, deterministic
candidate fold in prop order, hoisted capture totals) is the next
structural item — it covers all of these at once for the root levels.

## Cumulative standing (photo 4K, bytes identical, suite 12/12)

| cell | cjxl | turn start | now | target |
|---|---|---|---|---|
| e7 t=1 | 3.82 | 8.4-8.6 | **7.55** (1.98x) | 4.97 |
| e9 t=1 | 23.1 | ~48 | **46.3** (2.00x) | 30.0 |
| e7 t=8 (production zenjxl) | 0.61 | ~7.9 (seq learn) | **3.06** (5.0x) | 0.79 |
| e9 t=8 | 3.51 | ~52 | **21.6** (6.2x) | 4.56 |

Remaining ladder to 1.3x, in EV order: (1) per-prop parallel
find_best_split (t=8 root levels; also derive/partition/sorts);
(2) learn width — predictor set tightening under the corpus gate
(t=1: learn is 47-57% of wall, cost ~linear in K); (3) rct_select
(362 ms), patches gate at lossless (157 ms), prequant (581 ms);
(4) e9's exact-bucketize pre-walk overlap with the probe walk.

## Round 1c: coverage-gated keep cap (output-changing, corpus-gated)

Cap the auto keep-set to the top-5 statics WHEN they already carry
>= 90 % of the probe tree's static leaf mass (photos concentrate:
mosaic photo = 90.8 %; spread trees decline — a tree-SIZE gate was
REFUTED first: screen e9's 589-leaf spread tree got capped and cost
+11.4 % bytes; coverage is the content signal, size is not).

| cell (t=1) | before | after | bytes |
|---|---|---|---|
| photo e7 | 7.55 s | **7.10 s** (1.86x) | +0.10 % |
| photo e9 | 46.3 s | **42.3 s** (1.83x) | **−0.07 %** |
| photo e9 peak_live | 997 MB | **784 MB** | — |
| photo e9 t=8 | 21.6 s | **19.1 s** | — |
| screens e7/e9 | unchanged | unchanged | 0 (cap declines) |

18-image corpus: total +0.047 % vs prior default, 4/18 images move,
worst +0.18 % (9094 illustration); djxl decodes the new default
PIXEL-EXACT; hash-lock regenerated (1 fixture), suite 12/12.

## Round 2 (2026-08-17): per-prop parallel FBS + lossy setup fixes

**Lossless** (photo 4K, bytes identical, suite 12/12):
per-property parallel find_best_split (one evaluator body shared by the
sequential and parallel dispatches; ordered fold reproduces the strict-<
selection, block-capture for the tensor, first-in-order cap_totals;
bounded waves of 4 pooled workspaces at >= 1M-row nodes) + u32
bucket-sort indices (halves the largest workspace member — a cache win
even at t=1):

| cell | before | after | ratio |
|---|---|---|---|
| e7 t=1 | 7.10 | **6.51 s** | 1.70x |
| e9 t=1 | 42.3 | **38.2 s** | 1.65x |
| e7 t=8 | 2.92 | **2.74 s** | 4.5x |
| e9 t=8 | 19.1 | **18.2 s** | 5.2x |

**Lossy** (photo 4K PPM, d=1.25, best-of-3; bytes identical everywhere):
the low-effort wall gap was NOT the core (e3 t=8 inner 68 ms ~= cjxl's
whole run) but pre-inner setup: the content classifier ran a FULL-IMAGE
zenanalyze sweep (78 ms — more than the e3 core) on every encode while
its only consumer fires at e5-6; and the encoder then computed the
IDENTICAL sweep again for enc.zenanalyze_proxies. Banded the classifier
to its consumer's exact gates and shared ONE sweep between both
consumers; parallelized the sRGB u8->linear ingest LUT.

| cell | was | now | cjxl | ratio |
|---|---|---|---|---|
| e3 t=1 | 0.27 | **0.21 s** | 0.15 | 1.40x |
| e3 t=8 | 0.14 | **0.08 s** | 0.07 | **1.14x** |
| e5 t=1 | 1.55 | **1.49 s** | 1.47 | **1.01x** |
| e5 t=8 | 0.60 | **0.53 s** | 0.26 | 2.04x |
| e7 t=1 | 2.04 | **1.99 s** | 2.63 | **0.76x — win** |
| e7 t=8 | 0.66 | **0.60 s** | 0.52 | **1.15x** |

Remaining ladder: e5-t8 tail = patches_detect 162 ms (not scaling),
the shared proxy sweep 78 ms (f64 sum chains — exact parallelization
impossible, needs a corpus-gated deterministic-strip version),
quant_field 58 ms (flat across threads); e3-t1 last ~8 % = two-pass
entropy build; lossless t=8 root levels beyond the prop fan-out;
lossless e9 t=1 38.2 -> 30.0.
