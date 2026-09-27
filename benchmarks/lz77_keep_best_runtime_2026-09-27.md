# LZ77 keep-best candidate reuse — runtime, September 27, 2026

Follow-up to [the typed keep-best policy](lz77_keep_best_2026-09-27.md)
(#110). This change removes two classes of redundant work inside the
existing selection; the candidate set, the real serialized comparisons and
the strict tie policy are unchanged.

`apply_lz77_optimal` already ran `apply_lz77_backref` to build its cost
model and then discarded the result. The keep-best Greedy candidate then
parsed the same streams again with identical parameters.
`apply_lz77_optimal_keeping_greedy` now returns that by-product and
`Selection` replays it as the Greedy candidate instead of reparsing; the
replayed output is asserted token-for-token identical to a fresh
`apply_lz77(Greedy)` in `greedy_handover_is_verbatim`.

`Selection::build` is split into `parse` (token transform) and `finish`
(params + entropy-code construction). Candidates whose parse output is
untransformed (a no-op LZ77 rewrite) reuse the already-measured plain-unit
size; candidates whose per-stream ranges and tokens are coding-identical to
the incumbent (`ranges` equality plus `same_tokens`, which compares value,
context, length flag and length) reuse the incumbent's measured size.
The `measure` callback — the caller's real serialization — therefore runs
exactly once per distinct coded candidate, no more. Observation
(`observe`) is emitted for every candidate on every path including reuse,
so the measured ladder is complete; a unit oracle
(`ladder_matches_unreused_oracle`) measures every candidate independently
and requires identical sizes and identical selected bytes.

Ties remain strict `<`, so equal-size candidates retain the incumbent.
Error, cancellation, budget and hybrid-baseline semantics are unchanged.
No new public API; `Lz77Parse`, `GreedyReuse` and `same_tokens` are
`pub(crate)` or private. `JXL_LZ77_SKIP_GREEDY` retains its diagnostic
meaning; the replayed parse satisfies the same arm.

## Measurement

Baseline is `55e4769b` (parent) plus the new `lz77_keep_best_profile`
example copied in; candidate is `618b52d4`. Both built
`--release --features parallel` on the aarch64 host. `measure.py` runs
one warmup pair then four alternating repetitions per cell and reports the
minimum wall per binary; every cell hashes both outputs and requires them
equal. Data: `lz77_keep_best_runtime_2026-09-27.tsv`. Harness:
`~/tmp/kb-measure/measure.py` (macOS port of the session's wall.sh).

The grid: three source images (terminal.png screenshot, frymire-srgb
palette scan, nature photo) x {256, 512, 1024} center crops x {u8, f32} x
e8/e9 global, plus squeeze/local/hybrid coverage and four-thread cells at
1024-u8. Ratio is candidate/baseline wall; below 1.0 is faster.

| Group | n | median | range |
|---|---:|---:|---|
| u8, keep-best armed | 33 | 0.944 | 0.858 - 0.989 |
| f32 | 18 | 0.994 | 0.892 - 1.004 |
| squeeze/local/hybrid | 9 | 0.938 | 0.858 - 0.961 |
| four threads | 6 | 0.961 | 0.891 - 0.984 |
| e7 + arm-off controls | 6 | 1.000 | 0.987 - 1.003 |

By source: terminal median 0.969, frymire 0.963, nature 0.938. Aggregate
armed wall across the grid is 113.7 s -> 107.8 s (-5.2 %). All 57 cells
are byte-identical between the two binaries.

Effort-7 and arm-off cells are no-op controls: keep-best is ineligible at
e7, and arm-off exercises the same non-selection path in both binaries.
Both sit at ≈1.00, confirming the deltas above are attributable to the
reuse and not measurement drift.

## Identity evidence

Every measured cell is byte-identical between baseline and candidate
(sha256 match column), including all four modes and the four-thread cells.
Separately, `just lz77-keep-best-check recon-local` passes the complete
70-pair production gate: enabled output never grows, the frymire
256/e8/f32 bucket win stays 60,333 → 54,222, terminal retains 22,072, and
all strict/lossy controls are hash-identical.

## Cost accounting

On x86 the session's callgrind pair for terminal-512-u8-e9 showed the
keep-best arm at 4.779 GIr (on) vs 4.299 GIr (off) before the change —
≈+11 % selection overhead — and 4.426 GIr (on) vs 4.277 GIr (off) after;
i.e. the armed path lost ~353 M instructions and the residual selection
overhead fell to ≈+3.5 %. These instruction counts were captured on the
cloud x86 host before suspension; valgrind/callgrind does not support this
aarch64 host, so the local evidence is wall-only. A `sample` profile of
the optimized binary on nature-1024-u8-e9 attributes the remaining encode
time to the learned-ANS tree machinery (`bucketize_and_strip_props`,
`exact_bucketize_plan`, `estimate_bits_u32`); keep-best frames are below
the ≥5-sample leaf threshold on that input.

## Limits

The u8 medians above are where greedy replay fires (screenshot/palette
content where LZ77 transforms). Derived-f32 streams rarely transform, so
those cells are already dominated by the untransformed fast path and move
≈0 %. A +0.4 % cell (terminal-1024-f32-e8) is within run-to-run noise on
the shared host. No estimator, threshold, effort or candidate-set changes.
