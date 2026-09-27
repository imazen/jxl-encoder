# Typed LZ77 keep-best: encode cost, September 27, 2026

The strategy control is `EncoderImprovementsCustom::lossless_lz77_keep_best`.
Zenjxl and Aggressive enable it; LeanFaster and Libjxl disable it. The existing
method schedule supplies Greedy at lossless effort 8 and Optimal at 9+.
Selection requires effort >= 8, learned ANS and a non-RLE LZ77 method.
Forcing Greedy/Optimal below effort 8 does not enable this additional search. Explicit Huffman,
RLE, disabled LZ77, disabled learning and faster-decoding restrictions win.
Custom can disable selection without changing the other Zen policies. There
is no production read of the obsolete `JXL_LZ77_KEEP_BEST` variable.

This is an explicit size-versus-time policy, not a fitted content classifier.
No matcher threshold, chain length, candidate algorithm or cost arithmetic
changes in this follow-up. Effort 7 and the default lossless effort remain
outside selection. Lossy encoding does not consume this gate.

## Method

`examples/lz77_hash_ab.rs --keep-best-repeats N` alternates typed off/on order
inside every repetition and measures only encoding. Configuration construction,
hashing, source IO and artifact IO are outside the timer. Repetition 0 warms
both arms; every encoded output, including warmups, is retained by SHA256.
Every paired size must be non-growing; repeated output hashes must agree.
`scripts/lz77_keep_best_timing.py` verifies artifacts and complete pair grids,
records every command and summarizes median paired on/off ratios. Ratios of
separately reported median times can differ slightly from those paired medians.

One ARM64 Mac, normal runtime SIMD dispatch, `nice -n19`, one heavy process at
a time. Four real source images: frymire graphics, terminal screenshot, nature
photograph and brochure scan. Inputs are center crops, without resampling.
The u16 arm widens u8 with `v << 8`; f32 divides those integers by 65535. These
are derived float samples, not a native HDR corpus. All inputs are hash-bound.

The 64/256 grid uses five measured repetitions, efforts 7/8/9 and u8/u16/f32.
The larger/interaction grid uses three measured repetitions: 1024/2048 u8
photo/document global encoding; 259-pixel graphics/screenshot squeeze, local
and Hybrid encoding; and 1024-pixel photo/document global encoding with four
threads. The normal global cells use one thread. This selected interaction
grid does not qualify all content, bit depths, efforts or thread counts.
Efforts 10+ and x86 timing are not measured. No RSS result is claimed.

## Small-image measurements

Median across the eight source/size cells per depth (sixteen for integer
u8/u16 combined), with each cell using the paired median described above:

| Input | Effort | Median added time | Cell range |
|---|---:|---:|---:|
| u8/u16 | 8 | 13.2% | 9.3–26.2% |
| u8/u16 | 9 | 12.5% | 7.0–18.1% |
| derived f32 | 8 | 4.0% | 3.5–4.9% |
| derived f32 | 9 | 3.8% | 1.7–5.4% |

On the named frymire 256/e8/f32 regression, median encoding time is
553.906 → 575.556 ms (paired +4.0%); size is 60,333 → 54,222 bytes (-10.1%).
Across the 72 small-grid cells, seven improve size and 65 retain identical
hashes. All 24 effort-7 controls retain identical hashes; their paired timing
ratios span 0.991–1.022 with no selection work in either arm.

This timing harness does not decode images or measure perceptual quality.
The separate production regression gate performs exact lossless reconstruction
through jxl-rs and djxl, including the named win and rejected candidate.
The full-output gate, rather than timing or equal file lengths, establishes
those decoder results.

## Larger images and other paths

| u8 input/path | Effort | Threads | Median off → on | Paired added time |
|---|---:|---:|---:|---:|
| Nature 1024², global | 8 | 1 | 2729.6 → 2963.7 ms | 8.3% |
| Nature 1024², global | 9 | 1 | 5430.4 → 6163.5 ms | 13.5% |
| Nature 2048², global | 8 | 1 | 10208.0 → 10950.3 ms | 7.4% |
| Nature 2048², global | 9 | 1 | 18712.2 → 21255.5 ms | 13.6% |
| Brochure 1024², global | 8 | 1 | 359.8 → 400.7 ms | 11.4% |
| Brochure 1024², global | 9 | 1 | 886.1 → 948.7 ms | 7.1% |
| Brochure 2048², global | 8 | 1 | 1294.3 → 1439.2 ms | 11.1% |
| Brochure 2048², global | 9 | 1 | 2481.4 → 2702.2 ms | 8.6% |
| Nature 1024², global | 8 | 4 | 2073.7 → 2630.7 ms | 27.1% |
| Nature 1024², global | 9 | 4 | 4457.4 → 5556.0 ms | 24.7% |
| Brochure 1024², global | 8 | 4 | 265.1 → 307.4 ms | 15.9% |
| Brochure 1024², global | 9 | 4 | 742.4 → 800.0 ms | 7.6% |

The twelve 259-pixel squeeze/local/Hybrid cells add 9.0–22.4%. All 24 cells
in this larger/interaction grid retain identical hashes. Across both grids,
**7 of 96 pairs shrink and 89 remain identical**, with no size growth.
The policy buys extra search, not a general rate/time improvement. These
measurements support keeping LeanFaster outside the search; they do not
establish corpus-wide savings or justify a content-based gate.

[Small grid](lz77_keep_best_timing_small_2026-09-27.tsv),
[large/interaction grid](lz77_keep_best_timing_large_2026-09-27.tsv), and
[provenance and exact commands](lz77_keep_best_timing_2026-09-27.meta.json)
retain source/output hashes, sizes, times and configurations. All 1,056
encodes, raw repetition tables and full process logs are retained under
`~/tmp/jxl110-zen/{screen,large}/`; artifacts are content-addressed within
each directory. There is no cloud/NAS mirror claim.

[Descriptive size fits](lz77_keep_best_size_fit_2026-09-27.tsv) separate
`total = alpha + beta * megapixels` for CPU time and encoded bytes on the two
source-identical 64/256/1024/2048 ladders. Each row also reports the largest
observed residual. These are descriptive fits over four different crops,
not extrapolations or calibrated allocation/runtime bounds; changing content
and group structure means an intercept is not necessarily physical fixed
per-call overhead. No fitted coefficient enters source code.

Reproduction: build `lz77_hash_ab` in release with `parallel` against the
pinned sibling closure, then run `just lz77-keep-best-timing <binary>
<manifest.tsv> <new-output-directory> <repeats>`. The two manifests are
embedded in the provenance JSON. The binary measured explicit Custom off/on;
final strategy wiring additionally pins the effort-8 minimum and disables
LeanFaster. Neither adjustment changes a measured candidate or eligible cell.

## Validation

Against the CI-pinned sibling closure, all 75 hash-lock/strict-lock/divergence
checks pass without changing expectations, as do 1,625 default library tests
(29 existing ignores). The full workspace tests and doctests pass, including
1,645 feature-unified encoder library tests and 508 integration tests.
Workspace all-target Clippy with `-D warnings`, scoped formatting and both
real-image RD regressions pass. The separate 70-pair production gate passes
with the obsolete environment variable deliberately opposing the typed policy.
Its 140 output hashes match the original production-selector validation.

Full logs: `~/tmp/jxl-exact-cleanup/keep-best-zen-gated-2026-09-27/`,
`~/tmp/jxl110-keep-best-zen-typed/test.log`, and
`~/tmp/jxl110-zen/{workspace-after-cleanup,rd}.log`.
