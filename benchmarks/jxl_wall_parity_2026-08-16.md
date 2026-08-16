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
