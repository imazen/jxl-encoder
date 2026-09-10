# Where lossy e5 spends its time at threads=8 — and which phases do NOT scale

`benchmarks/ladder_vs_cjxl_cbrt_2026-09-10.*` leaves **e5 at threads=8 as the
loosest band on the ladder** — 1.30x cjxl v0.12 at 2048² and 1.11x at 1024²,
against 0.85-0.89x for the same effort at threads=1. So the problem is thread
scaling, not per-pixel cost, and that is what this measures.

## Method

Harness: [`examples/effort_loop_profile.rs`](../jxl-encoder/examples/effort_loop_profile.rs)
with `__JXL_ENC_PHASE_TIMING=1`, built
`--features "parallel,__env_var_diagnostics"`. nature photo, 2048² centre crop,
d=1.0, e5. Each number is the steady-state per-encode figure from a 5-8 s loop.

**Two harness traps, both fixed in the same commit, both of which silently
produced a number that looked fine:**

1. The example ignored thread count entirely — it never called `with_threads`,
   and the encoder does not read `RAYON_NUM_THREADS`. A t=1 vs t=8 run reported
   400.0 ms/encode for BOTH arms. It now takes `threads` as an argument.
2. `parallel` is NOT a jxl-encoder default feature — only the CLI enables it —
   so `cargo run --example` builds encode single-threaded no matter what
   `with_threads` says. The example now REFUSES `threads > 1` when built without
   the feature rather than reporting two identical single-threaded runs.

A claim this file does NOT make: `content_class` is not serial. A first reading
put it at a flat 16 ms and called it Amdahl work; that reading came from the
non-parallel build. Measured properly it is 15.7 -> 2.5 ms (6.3x).

## Phase breakdown, 2048² e5 d1.0, nature

| phase | t=1 ms | t=8 ms | speedup | % of t=8 |
|---|--:|--:|--:|--:|
| **total (encode_inner)** | 411.8 | 106.0 | 3.88x | 100 % |
| acstrat | 239.4 | 46.6 | **5.14x** | 44.0 % |
| entropy | 51.0 | 22.3 | **2.29x** | 21.0 % |
| quant_field | 31.4 | 15.1 | **2.08x** | 14.2 % |
| xform | 56.5 | 10.5 | 5.38x | 9.9 % |
| gaborish | 13.0 | 4.2 | 3.10x | 4.0 % |
| xyb | 7.5 | 1.4 | 5.36x | 1.3 % |
| cfl1 | 8.6 | 1.4 | 6.14x | 1.3 % |

Plus `conv+setup` outside `encode_inner`: 18.3 -> 3.1 ms (5.9x).

**This corrects the standing note in CLAUDE.md that `acstrat` is "the lever" for
e5 t=8.** It is the largest phase by share at t=8 (44 %) but it is also the
BEST-scaling large phase (5.14x). The two phases holding the encode back are
`entropy` and `quant_field`, together 37.4 ms of 106 at t=8, both scaling at
about 2x. If they scaled like `acstrat` they would cost ~15 ms instead, i.e.
~21 % off the encode — which is roughly the whole gap to cjxl on this cell.

## Inside `entropy` (2.29x)

| sub-phase | t=1 ms | t=8 ms | speedup |
|---|--:|--:|--:|
| build_codes | 17.3 | 10.2 | **1.70x** |
| pass2_write | 16.4 | 5.6 | 2.93x |
| ac_tok | 9.5 | 4.8 | 1.98x |
| co (coeff orders) | 7.6 | 1.4 | 5.43x |
| bcm / lz77 / tok_dc | ~0.1 | ~0.1 | — |

`build_codes` parallelises only as `join(build_dc, build_ac)` with AC dominating,
which is exactly why it lands at 1.70x. Inside the AC call
(`build_entropy_code_ans_from_token_groups_with_strategy`, 64 groups, **6930
contexts**):

| | t=8 ms |
|---|--:|
| Phase A `accumulate_groups_parallel` | 3.2-3.6 |
| Phase B `build_entropy_code_from_accumulated_ans_with_strategy` | **7.16** |
| … of which `cluster_histograms` | **6.82** |

**`cluster_histograms` is 6.8 ms of a 106 ms encode (6.4 %) and is entirely
single-threaded**, clustering 6930 histograms down to 67. Phase A is already
parallel; Phase B is 95 % clustering.

Its two loops in `entropy_coding/cluster.rs::fast_cluster_histograms_with_prev`
are each ~464k histogram-distance evaluations (67 clusters x 6930 inputs):
- the **main clustering loop** is sequential ACROSS clusters (each new cluster
  changes the distances), but its inner walk over all inputs is a pure
  `min`-update plus an argmax — parallelisable if the argmax keeps
  lowest-index-wins on ties, which the sequential `>` comparison implies;
- the **assignment loop** needs its merge step checked before any claim: if it
  mutates `out` as it goes, later inputs see updated clusters and it is not
  independent.

## Inside `quant_field` (2.08x)

| sub-phase | t=1 ms | t=8 ms | speedup |
|---|--:|--:|--:|
| `mask1x1_for_pre_scale` | 7.32 | 2.00 | 3.66x |
| everything after it | 23.72 | 12.64 | **1.88x** |
| … `compute_quant_field_float` (of that) | 15.6 | 4.35 | 3.59x |

and inside `compute_quant_field_float`:

| step | t=1 ms | t=8 ms | speedup |
|---|--:|--:|--:|
| **`compute_pre_erosion`** | 2.44 | 2.44 | **1.00x — fully serial** |
| `fuzzy_erosion` | 7.98 | 1.13 | 7.06x |
| `per_block_modulations` + copy | 5.15 | 0.77 | 6.69x |
| masking | 0.01 | 0.01 | — |

`fuzzy_erosion` and `per_block_modulations` were strip-parallelised earlier on
2026-09-10; `compute_pre_erosion` was not, and it is now the only step in that
function with no thread scaling at all.

**It is structurally strip-parallel and nothing about it resists that**: each
output row consumes exactly 4 input rows plus a 1-row halo for its `y±1` reads,
and the per-row arithmetic (4 accumulations into `diff_buffer`, then one sum per
output block) is unchanged by splitting. What it needs is a row-ranged entry
point in `jxl-encoder-simd` — today `compute_pre_erosion` takes a tile rect and
internally expands it by 4 on every side, so calling it per strip does NOT
compose into the whole-image result.

~8.3 ms of the t=8 `quant_field` phase is still unattributed (between
`mask1x1_for_pre_scale` and `compute_quant_field_float`) — `median_mask1x1`,
`quantize_quant_field`, the pre-scale block and the pixel-loss dispatch live
there. Attributing it is the next measurement, not a guess.

## Target list, in measured order

1. `cluster_histograms` — 6.8 ms, 1.00x. Biggest single serial block.
2. the ~8.3 ms unattributed inside `quant_field` — measure before touching.
3. `compute_pre_erosion` — 2.44 ms, 1.00x. Smallest of the three but the
   clearest: the parallel shape is already proven twice in the same file.
4. `ac_tok` (4.8 ms, 1.98x) and `pass2_write` (5.6 ms, 2.93x).

Every one of these is a scaling fix, not an algorithm change, so each should be
byte-identical and provable as such by the hash locks.

## Scope

One image (nature photo), one size (2048²), one distance (d=1.0), one effort,
one host (aarch64, macOS 25.5.0), 8 threads. The phase SHARES will move with
content — a screenshot spends less in `acstrat` — but the SCALING factors are
properties of the code, not the content, and those are what this file is for.
