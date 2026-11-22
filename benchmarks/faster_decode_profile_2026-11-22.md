# Faster JPEG XL decoding with the current encoder and decoder

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
