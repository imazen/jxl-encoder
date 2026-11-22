# Lossless fast-decode profiles on the imazen-26 feature cohort

The [pinned cohort](faster_decode_imazen26.tsv) supplied six SDR
originals and 15 distinct center crops. Effort-7 tier 0, tier 1 with
128 px groups, and `--fast-decode -g 0` produced 63 streams. Exact
RGB8 source samples were verified through zenjxl-decoder, jxl-rs
`f370be423933628eb7f783f1b61d4c765eca5f79` and djxl v0.12.0.
The three 16-bit encodes of HDR original 1064 also match the source
in all three decoders using a PNG16 readback. The initial PPM check
requested u16 from an output format that supports only u8. Tier 1
uses 37,914,116 bytes versus 34,707,356 for tier 0. The HDR control
has no timings in this table.

## Balanced lossless

| Original | Bytes vs tier 0 | 1-worker decode | 8-worker decode |
|---|---:|---:|---:|
| 1007 textured photo | +4.86% | 2.24× | 1.95× |
| 1019 smooth photo | +4.91% | 2.45× | 2.10× |
| 8457 terminal | +32.98% | 2.19× | 2.16× |
| 7026 aliased plot | +35.47% | 2.05× | 2.61× |
| 8106 mixed web page | +16.84% | 1.94× | 2.07× |
| 6824 scanned text | +10.97% | 2.27× | 2.08× |

The two held-out photos meet all balanced limits in this run. The held-out
terminal meets the speed target and exceeds the byte budget. At the native
size, the aliased plot also exceeds the byte budget and the mixed web
page misses the one-worker target. A single unconditional tier-1 policy
therefore fails the compact cohort.

### Palette caps and group size

The effort-7 grid varies the palette cap and channel compaction independently,
and compares 128, 256 and 512 px Modular groups. All 90 streams reproduce the
source samples through Zen at one, four and eight workers and through both
independent decoders. Each timed arm uses 25 individually recorded warm
decodes at each worker count. The encoder is `f574a317`, the timing decoder is
`88ce5198`, and A/A repeats the tier-0 stream.

| Native graphic | Tier-1 choice | Byte growth | Throughput at 1 / 8 workers |
|---|---|---:|---:|
| 8457 terminal | 128 px, palette cap 4,096 | +6.53% | 6.851× / 2.096× |
| 7026 plot | 128 px, stock palette cap | +35.47% | 2.024× / 1.774× |
| 7026 plot | 256 px, stock palette cap | +13.14% | 1.374× / 1.433× |
| 8106 mixed page | 128 px, stock palette cap | +16.84% | 1.945× / 2.134× |

The 4,096-color cap meets the rate and throughput thresholds on the
terminal at 128 px. Disabling palettes on the plot grows the 128 px file
from 8,719 to 18,250 bytes, and changing a positive cap does not move
its bytes. The mixed page has bit-identical native streams across all
tested palette caps. The three 128 px selections have a median byte
growth of 16.84% and p90 of 35.47%. Switching the plot to 256 px cuts
the median byte growth to 13.14% and the one-worker throughput median
to 1.945×. The complete graphics target remains open.

[Exact stream grid](faster_decode_imazen26_palette_grid_2026-11-22.tsv)
and [paired timing and p95](faster_decode_imazen26_palette_timing_2026-11-22.tsv).
The grid ran for 117 seconds at 0.19 GiB peak RSS. The raw records and
commands are in the `imazen26-lossy/palette-grid` run directory on i270.

## Throughput lossless

The dedicated two-symbol decoder compared against its parent
`e58a4838725704e37d90f2acd1d11dcccd7ab98a` on identical `--fast-decode -g 0`
files. The native photo median reaches 112.2 MP/s at one worker and
461.1 MP/s at eight. The 1024 px photo crops reach 113.8 and 583.6 MP/s.
Native graphics and document cells span 126.3–135.0 MP/s at one worker
and 539.3–709.7 MP/s at eight. The gain depends on the particular stream
and size. The native photo medians miss both absolute targets.

On terminal 8457, the expanded fixed-Gradient prefix options with
`--lz77 --lz77-method rle` produced the exact same stream hash as
`--fast-decode -g 0`. The shared-prefix Modular group path reads neither
the enable bit nor the method. This directly motivates comparing real
local prefix and distance-1 RLE streams.

## Method and evidence

Encoder `f574a3172b5009c9914e600903fb3602d74b60c6` was built with
Rust 1.100.0. The balanced comparison uses production decoder
`31b6fa590b4d367cdd7715a71eba81cdfd57a1cf`. The two-symbol binary is
the [dedicated prefix prototype](https://github.com/imazen/zenjxl-decoder/blob/0eba97ebea20c865ac154b7b397caf8a2198826c/benchmarks/fused_prefix_2026-11-22.md).
Every cell uses 1, 4 and 8 pinned P cores on i270, allocated in-memory u8
output, one warmup, and three interleaved blocks of five measured decodes
per arm. Ratios use the median of block-paired ratios. A/A repeats the
tier-0 stream on the production decoder. Block means do not measure
per-decode p95. Timing used 0.16 GiB peak RSS over 309 seconds.

[Per-size results](faster_decode_imazen26_lossless_2026-11-22.tsv).
The full logs, pinned binary and input hashes, streams, and reproduction
scripts are under `/mnt/v/output/jxl-encoder/fast-decode-2026-11-22/imazen26-lossless`
on the measured machine.

## Local prefix and distance-1 RLE trial

The [experiment branch](https://github.com/imazen/jxl-encoder/commit/fec813d3a3bec865f4a8f3f4e24332f1622ca6e8)
serializes shared prefix, local prefix, and local prefix with distance-1
RLE. Each group takes the smaller padded section and keeps its incumbent
on a tie. It separates the singleton distance code and stops matching
at channel boundaries. The opt-in uses the same existing fixed-Gradient
prefix flags as the byte-identical control above.

| Input | Candidate bytes / plain-prefix bytes | Candidate plus fused decoder / current plain prefix, 1 / 8 workers | Candidate plus fused decoder, MP/s at 1 / 8 workers |
|---|---:|---:|---:|
| 1007 native | 95.30% | 1.048× / 1.136× | 88.6 / 422.0 |
| 1007 1024 px | 97.81% | 1.048× / 0.986× | 86.0 / 477.4 |
| 1019 native | 74.85% | 1.168× / 1.040× | 114.0 / 453.2 |
| 1019 1024 px | 84.10% | 1.095× / 1.103× | 109.2 / 555.1 |
| 8457 terminal | 17.93% | 1.505× / 1.356× | 160.1 / 802.4 |
| 7026 plot | 12.30% | 1.550× / 1.311× | 168.4 / 813.9 |
| 8106 mixed web page | 37.18% | 1.422× / 1.305× | 145.9 / 732.3 |
| 6824 scanned text | 98.24% | 1.110× / 1.027× | 113.9 / 457.0 |

All eight files get smaller. Six readbacks per file match every original
sample, including production zenjxl-decoder at 1, 4 and 8 workers, the
fused prototype at 8, and independent jxl-rs and djxl. Earlier trials
exposed missing byte padding and a context-map field on a one-context
stream. The verified branch corrects both. Per-case timings compare
the same candidate streams on both decoder builds and both file shapes
on the same decoder. On the fused decoder alone the three graphics
files still improve at 1 and 8 workers, while both photos regress at
each size. The byte-first selector misses the photo throughput gates.

[Per-cell results and stream hashes](faster_decode_imazen26_local_prefix_2026-11-22.tsv).
The locked release build and all checks used 1.17 GiB peak RSS over
78 seconds. Reproduction and raw block timings are in
`imazen26-lossless/local-prefix-v3` under the run folder.

### What distance-1 runs add

The [prefix-only ablation](https://github.com/imazen/jxl-encoder/commit/bf72bf330f9f3edd1e5b34e2f83dc62401db12bd)
keeps both shared and local prefix choices and removes the RLE choice.
All eight ablation streams passed the same six exact readbacks. Relative
to shared prefix, local prefix alone reduced bytes by 0.11–1.08% on the
terminal, plot and web page. Against that local-prefix experiment, allowing
distance-1 runs reduced bytes by 82.05%, 87.57% and 62.42%, respectively.

The two stream sets were also interleaved directly on the same fused
decoder binary. RLE improved the three graphics cases by 1.27–1.38× at
one worker and 1.19–1.34× at eight. On the textured photo, it reduced
bytes by 0.77–2.32% and slowed one-worker decoding by 8.45–10.95%.
For the native smooth photo it reduced bytes by 22.59% and improved
one-worker decoding by 5.65%. This split explains why a byte-only
selector can help repeat-heavy content and still miss the photo gate.

[Direct ablation results](faster_decode_imazen26_rle_ablation_2026-11-22.tsv)
retain the A/A control, stream hashes and all worker counts. The
prefix-only build and validation used 1.16 GiB peak RSS. The direct
comparison used 0.13 GiB peak RSS and 33 seconds. Raw blocks and
reproduction commands are in `imazen26-lossless/rle-direct-ablation`.

### Separate channel histograms

The [encoder experiment](https://github.com/imazen/jxl-encoder/commit/7fb131664f54c8f1f15c759c5e1829310e4b0f5c)
adds per-channel prefix histograms to the existing section-size comparison.
The Modular tree splits only on channel number and selects Gradient in
every leaf. Both context layouts can use distance-1 runs. The distance
histogram retains its singleton symbol and each run stays within a channel.

| Native input | Byte change vs one local histogram | Current decoder at 1 / 8 workers | Fused prototype at 1 / 8 workers |
|---|---:|---:|---:|
| Textured photo 1007 | −8.56% | 0.998× / 0.963× | 0.955× / 0.963× |
| Smooth photo 1019 | −3.43% | 0.985× / 0.923× | 0.982× / 0.970× |
| Terminal 8457 | −3.24% | 0.993× / 1.024× | 0.990× / 1.091× |
| Plot 7026 | −5.16% | 0.982× / 1.004× | 0.989× / 1.157× |
| Mixed page 8106 | −1.83% | 0.991× / 0.970× | 0.990× / 0.961× |
| Scanned text 6824 | −1.88% | 0.996× / 1.039× | 0.921× / 0.942× |

The textured photo at 1024 px saves 9.97% bytes. On the native photos,
the older fused prototype reaches 84.84/111.71 MP/s at one worker and
379.35/482.07 MP/s at eight in the throughput-profile comparison
against shared prefix. Both native photo medians remain below the
throughput thresholds. The direct same-decoder comparisons above show
that the byte reduction comes with a modest photo decode cost.

All eight files pass six exact source readbacks. Odd 259×257 RGBA8
and GrayAlpha8 crops also reproduce source samples with both
independent decoders and Zen at one, four and eight workers. The
existing `--fast-decode -g 0` terminal stream is byte-identical.
The build and first comparison used 1.12 GiB peak RSS over 80 seconds.
The direct same-decoder comparisons used 0.12 GiB over 68 seconds.
[Direct paired results](faster_decode_imazen26_channel_prefix_2026-11-22.tsv).
Full commands, stream hashes and alpha controls are in
`imazen26-lossless/per-channel-prefix`.
