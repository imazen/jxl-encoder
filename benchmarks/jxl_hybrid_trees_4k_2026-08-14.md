# Hybrid per-group tree selection: measured results (4K, threads=1)

`SectionedTrees::Hybrid` (`JXL_LOSSLESS_LOCAL_TREES=hybrid`): learn the
global tree AND a per-group tree (per-group learns ride the gather waves on
clones of the per-group samples), write each group both ways, keep the
smaller section (per-group `use_global_tree` — plain spec bitstream).

| cell | global | sectioned | oracle (pre-build) | hybrid | vs global | vs cjxl 0.12 |
|---|---|---|---|---|---|---|
| photo e7 | 4,436,537 | 4,346,224 | 4,336,316 | **4,336,683** | **−2.25%** | −3.2% |
| photo e9 | 4,246,312 | 4,271,051 | 4,235,282 | **4,234,493** | **−0.28%** | −3.6% |
| screen e7 | 305,873 | (fallback) | — | **282,053** | **−7.79%** | −41.1% |
| screen e9 | 256,192 | (fallback) | — | **251,939** | **−1.66%** | −24.7% |

Local-winning groups: photo e7 97/135, e9 42/135 (from the oracle
instrumentation, `JXL_SECTION_SIZES=1`). Screens — where the independent-tile
study predicted +10-21% — WIN once the per-group choice exists: local trees
crush the global one on some regions and the min() keeps global elsewhere.
The tile numbers conflated "no global option" with "local trees".

djxl decodes hybrid output PIXEL-EXACT (photo + screen e7); zenjxl-decoder
round-trips; suite + knob test (hybrid <= min(global, sectioned)) green.

Peak_live: global-mode levels +~2.4 MB (836/1133 MB) — hybrid is the RD-max
mode; the memory mode remains Sectioned (469 MB). Wall at threads=1 is the
honest cost: photo e7 14.5 -> 27.3 s, e9 83 -> 148 s (per-group learns +
double section writes; embarrassingly parallel in production). Wall
reduction levers queued on #96: skip local attempts on tiny groups, cheaper
per-group learn params, prune-don't-relearn.

## Sample-cap sweep (context: capped global gathers are NOT free)

JXL_TREE_MAX_SAMPLES on the global path (bytes vs uncapped): photo e7
+2.1-3.5%, photo e9 +1.8-5.2%, screen e9 +4.6-16.9% across caps 4M..500k;
non-monotonic at photo 500k < 1M — the fixed-stride aliasing signature, so
RANDOMIZED capped sampling (JXL_TREE_SAMPLE_RANDOM) is the lever to
re-measure before any capped-global memory mode. Raw:
~/tmp/4kmem/sample_cap_sweep.txt (this box), reproduced by env on any build.
