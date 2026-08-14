# Per-site allocation attribution, 4K encode: jxl-encoder vs cjxl 0.12.0

Answers "which code lines hold the peak, and which are disproportionate to
their C equivalent." Measured 2026-08-13 on macOS aarch64 (M-series),
photo_3840x2160 (8.29 MP photo content), single-threaded.

Method — two instruments, one per side:

* **jxl-encoder**: zenjxl `examples/mem_probe_encode.rs` per-site profiler
  (`JXL_ALLOC_SITES=1`, zenjxl commit a7427d2c): counting global allocator +
  raw-stack capture per allocation ≥ 64 KiB, per-site live map snapshotted on
  every 8 MiB peak-live raise → per-line composition AT the peak instant.
  Overhead: +252 KB peak, +125 allocs, wall unchanged. Top table identical at
  min = 64 Ki / 8 Ki / 512 B.
* **cjxl 0.12.0** (homebrew, `{AppleClang 21}`): `MallocStackLogging=lite` +
  `malloc_history <pid> -allBySize` triggered at ≥ 95% of the cell's known
  peak RSS, classified by deepest interesting `jxl::` frame
  (`~/tmp/4kmem/heap_stacks.sh` + `analyze` in the .meta). cjxl numbers are
  live-bytes near peak, not exact-instant, so read them ±10%.

jxl-encoder at 19c2f996, `--features parallel`, threads=1. cjxl invoked as
`cjxl in.ppm /dev/null -d {0|1.25} -e {7|9} --num_threads=0 --quiet`.

## Lossless e7 (ours 1871 MB peak_live / cjxl 304 MB peak RSS, 211 MiB live classified)

| component | ours (live at peak) | attributed line | cjxl (live near peak) | factor |
|---|---:|---|---:|---:|
| tree-learn props columns (24 × i32 × 12.44 M) | 1139.1 MiB | tree_learn.rs:776 `reserve_exact_total` | — | — |
| tree-learn residual_tokens (14 × u8) | 166.1 MiB | tree_learn.rs:770 | — | — |
| tree-learn extra_bits (14 × u8) | 166.1 MiB | tree_learn.rs:773 | — | — |
| tree-learn gather in-flight | 40.9 MiB | `gather_samples_strided_with_budget_inner_backend` | — | — |
| **tree-learning total** | **1512.2 MiB** | | **< 0.1 MiB** | **> 10 000×** |
| modular image channels (i32, whole image ×2) | 189.8 MiB | budget.rs:435 via `extract_region` | 45.9 MiB (per-group `Channel::Create`) | 4.1× |
| RCT trial clone | 94.9 MiB | channel.rs:21 `clone` | (inside the 46) | — |
| tokens | 0 at this instant | | 43.3 MiB | — |
| output BitWriters | small | | 50.5 MiB | 0× (they buffer more) |
| input copies | 23.7 (bin, pre-enable) | | 71.2 MiB (ppm + 2 copies) | 0.33× |

## Lossless e9 (ours 1875 MB peak_live / cjxl 326 MB RSS, 170 MiB live classified)

At e9 the props columns are already freed at the peak (the free_props-before-dedup
fix); the peak instant is **inside dedup_samples_packed_sort**:

| component | ours (live at peak) | attributed line | cjxl | factor |
|---|---:|---|---:|---:|
| dedup packed-sort buffer (64 B/sample!) | 759.4 MiB | parallel.rs:23 ← `dedup_samples_packed_sort` closure | — | — |
| pre-quantized bucket columns (24 × u8) | 284.8 MiB | tree_learn.rs:1841 `bucketize_with_thresholds` | — | — |
| residual_tokens + extra_bits | 332.2 MiB | tree_learn.rs:770/:773 | — | — |
| dedup outputs (unique_indices 8 B + counts 4 B + perm 4 B) | 189.9 MiB | tree_learn.rs:3356/:3357/:3337 | — | — |
| **dedup + samples total** | **1566.3 MiB** | | **< 0.1 MiB** | **> 10 000×** |
| modular image channels | 189.8 MiB | budget.rs:435 | 46.4 MiB | 4.1× |
| RCT trial clone | 94.9 MiB | channel.rs:21 | — | — |
| tokens live at peak | 0 | | 0 (streamed per group) | — |

## Lossy e7 d1.25 (ours 517 MB peak_live / cjxl 405 MB RSS, 330 MiB live classified)

| component | ours | attributed line | cjxl | factor |
|---|---:|---|---:|---:|
| text-patch detection working set | 128.0 live / 258 churn | patches.rs:1202 `find_text_like_patches_with_min_peak` | ~0 visible | big |
| linear-RGB input (3 × f32 planes) | 94.9 | ingest.rs:62 | 71.2 (u8 copies ×3) | 1.3× |
| XYB planes | 94.9 | budget.rs:435 | streamed strips (17 MiB each) | ~5× |
| third f32 plane-set | 94.9 | jxl-encoder-simd/lib.rs:129 `vec_f32_dirty` | — | — |
| sub-512 B per-block objects | ~64 | (untracked tail) | dense arrays | — |
| output BitWriters | small | | 173.0 | 0× |
| tokens | 0 at peak | | 85.3 | 0× |

Lossy context: cjxl **e9** lossy (butteraugli loop) is 2649 MB RSS — ours is
517 MB at every lossy effort; we are 5× UNDER the C equivalent there. The
binding gap is lossless.

## The structural finding

cjxl 0.12 defaults to **streaming encode** (`EncodeFrameStreaming` in every
stack): strip-by-strip f32 planes (17 MiB each), per-group modular channels,
per-group token write-then-free — and its MA tree learning holds **under
0.1 MiB live**, because it learns per group cluster from a *sampled* subset of
pixels whose properties are **pre-quantized to u8** bucket indices at
AddSample time and **deduplicated on the fly** in a hash table. It never
materializes per-sample property columns at all.

Our gather materializes 12.44 M samples × (24 × i32 props + 14 u8 tokens +
14 u8 extra) = 1.47 GiB, then dedup materializes another 64 B/sample packed
key + 16 B/sample outputs. Every per-line fix so far (free-before-dedup,
waves, exact reserves, collector) trims transients around that design; the
design itself is the 6× vs C. No sequence of column-lifetime fixes reaches
306 MB while whole-image i32 columns exist.

Fix ladder implied by the numbers (exact-output-preserving first):

1. **i16 props columns** where the property range fits (measured
   JXL_PROP_RANGE_STATS: all 9 used props fit i16 at e7 8-bit, prop15 max
   1920): 1139 → 570 MiB at the e7 peak. Byte-identical.
2. **Shrink/wave the dedup pack key** (currently 64 B/sample; 24 u8 buckets +
   token + extra fit ~32 B): −380 MiB at the e9 peak. Byte-identical.
3. **u8-bucket-at-gather** (thresholds from a sampled/collector pre-pass,
   columns stored as u8): removes the i32 columns entirely, −854 MiB at e7 —
   output-changing unless thresholds reproduce today's exactly; the two-pass
   exact variant measured +23–35% wall (refuted at 5% budget).
4. **Group-streamed tree learning** (the libjxl design): the only path to
   ~300 MB parity; changes output; architectural.

## Addendum: post-fix state (same day, commits d1074adc + 8b9b6121)

Ladder items 1 and 2 shipped byte-identical with identical alloc counts:

* **d1074adc** — dedup keys packed at the rounded word width
  (`packed_sort_walk::<W>`, 40 B at e7 / 56 at e9, was fixed 64 B) +
  u32 `unique_indices`. e9 1920 → 1871 MB; e7 unchanged (gather-bound).
* **8b9b6121** — `PropColumn`: adaptive i16 property columns, promote to
  i32 on first out-of-range value; width-generic pre-quantize.
  e7 1871 → **1397 MB**, e9 1871 → **1774 MB**.

Post-fix composition at the e7 peak (now the DEDUP phase): pack 474.6 MiB +
image channels 189.8 + tokens/extras 332.2 + dedup outputs 142.5 + RCT clone
94.9 + buckets ~107. The e9 peak is the same phase with its 24-property
56 B pack (664 MiB). Worst-case cells over {photo, screen}, threads=1:
e7 peak_live 1366 MiB / RSS 2342 MiB; e9 1734 / 2529 (pinned in
`heuristics.rs::estimate_covers_measured_4k_cells_2026_08_13`).

What remains vs cjxl's ~306 MB is the architecture, not per-line waste:
whole-image gather + materialized dedup vs cjxl's per-group streamed
learning on sampled u8-quantized keys with on-the-fly dedup (ladder items
3-4). The exact-preserving per-line ladder is now mined out — every
component at the peak instant is either the image itself, the samples at
their minimal width, or the dedup working set at its minimal layout.

## Addendum 2 (same day, commits 5119668d + 21684778 + 3e237d11)

Three more profiler-guided exact chunks, each byte-identical on all four
cells with allocation counts flat:

* **5119668d** — dedup partitioned by the two lead key bytes (stable
  counting sort; per-partition pack/sort/walk with reused scratch). Only
  one partition's pack is ever live. photo e7 1397 → 1288, e9 1774 → 1324.
* **21684778** — the gather stores only the configured property columns
  (`TreeSamples::active_props`); at e7 that is 9 of 24. photo e7 → 1028,
  e9 unchanged (all 24 configured).
* **3e237d11** — both whole-image i32 copies freed before tree learning
  (`ImageSource` ownership + drop of the transformed copy after the
  per-group split). photo e7 → **834 MB**, e9 → **1130 MB**.

Worst-case cells over {photo, screen}: e7 peak_live 873 MiB / RSS 1433;
e9 1104 / 1785. Session cumulative from 3141 MB peak_live / 5426 RSS:
**e7 −73 %, e9 −64 %**, wall and allocs unchanged. Factor vs cjxl ~306 MB:
2.7× (e7) / 3.7× (e9). What remains at the peak: the per-predictor token
columns (332 MiB — inherent to evaluating 14 predictors over 12.4 M
samples), the samples' bucket/prop columns at minimal width, the per-group
image copies (the working image), and the dedup outputs. Below ~800 MB the
design itself must change (per-group streamed learning on sampled
quantized keys — the cjxl architecture).

## Addendum 3 (same day, commit 4b464975 — the lossy round)

The lossy peak (patches/AQ phase) got the same treatment: the text-patch
DFS stack packed to flat u32 indices in one reused buffer (was per-CC
(u32,u32) Vecs doubling to 128 MiB live on photo content), and the
interleaved linear-RGB input freed right after XYB via `LinearSource`
ownership whenever no perceptual loop can read it. All byte-identical.

4K peak_live, worst over {photo, screen}, loop-free build: lossy e7
528 → 443 MB (photo 517 → 375 — 1.13× cjxl's ~330 MiB live at the same
cell), e3 412 → 318, e9 517 → 434. Lossless untouched; its alloc count
drops another 1,755 (per-CC stack allocs gone). cjxl lossy e9 remains
2,649 MB (butteraugli loop) — we are 6× under there.

## Addendum 4 (same day, commit 95b2a3e3): the sectioned mode is BUILT

`JXL_LOSSLESS_LOCAL_TREES=1` (v1, env-gated, default off, imazen/jxl-encoder#96):
per-group local MA trees, spec-conformant single frame, djxl pixel-exact,
zenjxl-decoder roundtrip test. 4K photo threads=1, mode ON:

| | peak_live | RSS | bytes | wall | allocs |
|---|---|---|---|---|---|
| e7 | 834 → **469 MB** | 1433 → 649 | −2.0% | −5% | −17% |
| e9 | 1130 → **469 MB** | 1785 → 643 | +0.6% | −25% | −33% |

The whole-image tree-learning working set — the dominant lossless term all
session — is GONE from the peak. The mode's peak is the patches-detection
phase (~380 MiB of opsin/background planes + DFS) plus the modular image
built before it (95 MiB, reorderable). vs cjxl: live 469 vs ~211 classified
live (2.2×), RSS 649 vs 304 (2.1×) — remaining closure is ordinary
lifetime/strip work on the patches detector, tracked on #96.
