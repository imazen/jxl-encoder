# Faster JPEG XL decoding with the current encoder and decoder

## Qualification goal

Ship opt-in JPEG XL profiles that decode photographs faster. Keep distinct
tiers when they offer different speed, rate or quality tradeoffs. Qualify
each tier on its photo workload at 1, 4 and 8 workers on i270 using the
direct paired harness and a small
[feature-specific imazen-26 cohort](faster_decode_imazen26.tsv).
Verify graphics conformance, report graphics speed and rate separately,
and allow graphics to keep their own settings or fall back to tier 0.

| Photo profile | Median throughput at 1 / 8 workers | Rate and quality |
|---|---|---|
| Balanced lossless, tier 1 with patches | ≥2.00× / 1.75× vs effort-7 tier 0 on photos ≥1 MP | Exact pixels. Growth ≤6% median / 12% p90 |
| Throughput lossless, fixed Gradient and prefix with size-selected distance-1 RLE | ≥1.25× / 1.10× vs `--fast-decode -g 0` on photos ≥1 MP, and ≥120 / 550 MP/s | Exact pixels. Bytes at or below comparator in every photo cell |
| Lossy with patches and independent EPF | ≥1.20× / 1.15× on photos ≥1 MP near d1 and d4 separately, at matched quality | ≤3% median / 10% p90 byte growth. Per SDR image, SSIMULACRA2 drop ≤0.25 and Butteraugli increase ≤max(0.01, 2%). Count fallbacks and require ≥75% fast choices |

Measure lossless group widths separately. For graphics-specific trials,
retain the balanced reference of ≤16% median and ≤25% p90 byte growth,
and the throughput reference of ≤50% of the stock bytes on the median
repeat-heavy screen.

Keep lossy options that shift quality in the sweep. Hold the reference
band fixed and adjust candidate distance to match rendered quality.
Record bytes and decode time at the measured candidate distance.
Keep same-distance measurements to show the size of the quality shift.
Decoder-only changes still require exact output on each frozen stream.

Ablate fused prefix, two-symbol reads and group interleaving. Test
singleton histograms, local histograms and palettes for lossless. Measure
lazy VarDCT tables (≥20% lower cold latency at ≤256 px), LF/DC and HF metadata
(≥15% less phase time and ≥5% higher throughput at 8 workers where hot),
small trees, scratch reuse and any measured scheduling bottleneck. General
decoder changes need ≥5% median speedup, ≤2% median control slowdown and
≤10% class p95 latency increase.

Use pinned same-stream decoder and end-to-end encoder comparisons, with
identical u8 RGB/RGBA endpoints, cold/warm and allocating/reused results,
three interleaved blocks of at least five samples, and A/A controls.
Cover tiny and group-boundary crops, ≥1 MP and native sizes, with a dense
lossy ladder. Keep imazen-26 test originals grouped across crops. Verify
lossless source pixels, decoder and worker-count parity, full rendering
with jxl-rs and djxl v0.12, and existing locks and format contracts for
strides, streaming, progressive, alpha, HDR and metadata. Retain per-image
rate, quality, timing, hashes and commands and verify landed changes on main.

The initial seven sources cover textured and smooth photos, repetitive
terminal content, palette and aliased lines, mixed web content, textured
document scans, and 16-bit HDR. The manifest pins PNG-v3 render bytes from
`0264d4d8a7e283c046ba2d1febb6e1fc547d511e` and split membership from
`21009633b62610415b7dcaccdec7806726d9e10c`. Class results describe this cohort.

## Related zqi experiments

The pinned zqi comparison builds a rate and decode-time curve for each
image, then reports the images whose curves cover a given metric target.
It compares in-memory decode into fresh RGB8 buffers. For this JPEG XL
cohort, measure the selected distance directly and check both
SSIMULACRA2 and Butteraugli on the rendered output.
[Zqi codec comparison](https://github.com/imazen/zqi/blob/c40aebae9f0b31e538e821e735a4ec1c663d78be/benchmarks/codec_cmp_zen4_2026-10-02.md#lossy-six-2048x2048-photos).

The zqi prefix microbenchmark combines symbol and code length in one
lookup, writes symbols directly to output, and uses a sentinel to track
refills. Six streams performed best among the reported sentinel
configurations on Zen 4. This gives JPEG XL decoder experiments three
measurable variables: fused lookups, refill frequency, and the number
of independent streams per worker. The i270 same-stream runs decide
which changes help JPEG XL.
[Zqi prefix microbenchmark](https://github.com/imazen/zqi/blob/c40aebae9f0b31e538e821e735a4ec1c663d78be/benchmarks/codec_cmp_zen4_2026-10-02.md).

The current zqi default branch lends workers disjoint mutable row bands
for full-width groups and uses scratch for column groups and partial
boundaries. Its direct-band measurement counts instructions on identical
pixels. Zen's VarDCT rendering already splits caller output into disjoint
tile fragments. Use the zqi result to prioritize profiling the paths that
still stage and copy output, then time those paths on i270.
[Zqi row-band study](https://github.com/imazen/zqi/blob/031154cd457a7b3119bba40e4a6a35cca0e738b6/docs/DECODE_SPEED_HISTORY.md#direct-row-bands-and-browser-scheduling)
and [Zen fragment renderer](https://github.com/imazen/zenjxl-decoder/blob/de5a5ef3fda613a6d858ef7e3a76c72057e07906/zenjxl-decoder/src/frame/render.rs#L1308).

The later zqi transform split holds color and entropy coding fixed while
comparing ordinary 8×8, larger merges and small splits at matched quality.
Larger merges improve rate on most curves in that study. The separately
linked fixed-transform decoder measures a narrower executable and exact
admitted output. The JPEG XL analogue is a same-stream dispatch test using
transform types already validated by HF metadata. Measure its native decode
phases directly before tying transform eligibility to a faster-decode tier.
[Zqi transform split and decoder cost study](https://github.com/imazen/zqi/blob/fd8271e7aa2c/docs/V4_V5_DECODER_COSTS.md#isolated-transform-splits).

## Initial pilot

The study tested existing encoder options at effort 7 with eight encode threads.
It used jxl-encoder `f574a317`, zenjxl-decoder `e58a4838`, and
[djxl v0.12.0](https://github.com/libjxl/libjxl/releases/tag/v0.12.0).
The inputs were CID22 validation 1025469 (512×512), a CLIC 2025 training photo
(2048×1358), gb82-sc terminal (1646×1062), and the Windows 95 palette image
(640×480). Both decoders reproduced every source pixel in all 36 lossless
streams, including four retained libjxl effort-1 controls. All 84 lossy streams
rendered in both decoders. Six lossy tier-1 controls produced identical djxl PFM
bytes to their baselines at distances 1 and 4.

## Lossless

Tier 1 with 128-pixel Modular groups is a balanced candidate. Tier 1 excludes
the weighted predictor and previous-channel properties at effort 7. The group
size override adds independently decodable groups and retains patch selection.
The CLI spelling is `--lossless --faster-decoding 1 -g 0`. For a library
profile that should make the same group-tree choice across encoder thread
counts, qualify explicit `SectionedTrees::On` separately: the measured
eight-thread control used the default group-tree policy.

| Input | Tier 1 / 128 px bytes | 1T decode | 8T decode |
|---|---:|---:|---:|
| CID22 photo | +4.57% | 2.34× | 4.55× |
| CLIC photo | +4.50% | 2.17× | 1.84× |
| Terminal screenshot | +15.70% | 2.51× | 2.47× |
| Windows palette | +12.67% | 2.04× | 2.83× |

Decode ratios use each image's ordinary effort-7 output at the same decode thread
count. The CID22 baseline has four 256-pixel groups, which limits its eight-thread
throughput. Tier 3 is a useful photo candidate: 6.85× at one thread on both
photos for +10.20% and +13.92% bytes. The tier-3 Modular gradient-property tree
has a dedicated lookup path in
[zenjxl-decoder](https://github.com/imazen/zenjxl-decoder/blob/e58a4838725704e37d90f2acd1d11dcccd7ab98a/zenjxl-decoder/src/frame/modular/decode/specialized_trees.rs).

Allowing patches paid off on terminal. Tier 1 / 128 px was 42,756 bytes and
36.4 MP/s at one thread. Tier 2, which also suppresses patches, was 159,932
bytes and 25.7 MP/s. Tier 3 grew to 176,701 bytes. The single-Gradient-leaf
`--fast-decode -g 0` output grew to 789,552 bytes and reached 105.9 MP/s.
An effort-1 prefix + distance-1 RLE control from libjxl v0.12 was 232,993 bytes
and reached 158.2 MP/s on the same screenshot. That is a useful wire-shape
control for a future writer candidate. Explicitly requesting the existing
RLE method with tier 1 / 128 px reproduced all four original streams
byte-for-byte.

A candidate for higher throughput can reuse fixed Gradient leaves and
distance-1 RLE over prefix-coded group streams. Try the prefix-only and RLE
forms with the same residuals, including the real histogram, group and TOC
costs. Prefer per-channel or per-group histograms when they recover enough
bytes. The existing decoder route requires a singleton distance symbol
of 1, split exponent zero, and a single unit-multiplier Gradient leaf after
MA-tree pruning. The prefix case reaches the
[distance-1 Gradient decoder](https://github.com/imazen/zenjxl-decoder/blob/e58a4838725704e37d90f2acd1d11dcccd7ab98a/zenjxl-decoder/src/frame/modular/decode/rle_gradient.rs).
Run it against the existing normal and fast-decode encodes at exact pixels
before setting a profile policy.

## Lossy

The fastest useful combination depends on how much EPF contributes to quality.
Tier 1 retains patches and selects the fixed Gradient DC tree for the tested
RGB frames. The normal EPF schedule with tier 1 was pixel-identical in all
six PFM comparisons, at +0.41–2.20% bytes and 1.04–1.16× eight-thread
decode throughput. DC with global extra channels follows a separate encoder
branch.

At distance 1, tier 1 with EPF off (`--faster-decoding 1 --epf 0`) was
1.25–1.32× at one thread and 1.17–1.33× at eight threads, with +0.41–1.44%
bytes. SSIMULACRA2 moved by −0.110 to +0.004 and Butteraugli moved by −0.0394
to +0.0038. The screenshot's tier-2 file was 130,158 bytes, versus 51,002
for tier 1 with EPF off. Both have the same requested distance and the
tier-1 combination retains patches.

At distance 4, EPF off made the two photos 1.99× and 2.06× faster at one
thread and raised their Butteraugli scores by 0.4313 and 0.2271. The
EPF-off streams at distance 3.5 reached or exceeded the distance-4
baseline on both metrics at +10.29% and +10.25% bytes, with 1.93× and 1.98×
one-thread throughput. These are measured rungs. On the CLIC photo,
tier 1 with one EPF pass at distance 4 satisfied both metrics at +1.66%
bytes, 1.70× one-thread and 1.56× eight-thread throughput. The 512-pixel
photo and the screenshot both lost Butteraugli quality with the same
selection. Keep the normal EPF schedule when a reduced-pass candidate
misses the required quality.

## Evidence and reproduction

The [full table](faster_decode_profile_2026-11-22.tsv) lists stream hashes,
both readback verdicts, quality scores and each timing round. The
[paired lossy table](faster_decode_profile_paired_2026-11-22.tsv) adds a
single interleaved comparison of the finalists against their distance-1/4
baselines. The [manifest](faster_decode_profile_2026-11-22.meta.json)
pins input hashes, build options, binaries and PFM controls. The retained
artifact directory records exact commands and holds all streams.

Encode with `cjxl-rs INPUT.png OUTPUT.jxl -e 7 --threads 8 -d DISTANCE`
and the manifest's profile flags. Decode in memory with
`zenjxl-decoder-cli STREAM.jxl --speedtest --data-type u8 --no-cms --num-reps 5 --warmup-reps 1`.
Set `--num-threads 0` for serial decode and `--num-threads 8` for the second
arm. The tested host was an Intel Core Ultra 7 270K Plus, with CPU 2 for serial
decode and CPUs 0–7 for the eight-thread arm. Ratios are medians of three
interleaved round ratios. The host was shared and logged a peak one-minute
load of 26.82, so timing is exploratory. The full follow-up wrapper took 66 s
and reported 0.70 GiB peak RSS. The study covers four sources and one ISA.
A shipping preset needs a wider quality ladder and held-out coverage for
photo, graphics, HDR, alpha and other runtime SIMD targets.
