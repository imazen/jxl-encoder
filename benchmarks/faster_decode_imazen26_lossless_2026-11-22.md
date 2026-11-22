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

### Independent balanced night photo

The same existing tier-1 policy on native imazen-26 test photo 1069
delivers 2.540× and 2.136× paired throughput at one and eight workers.
Its 3,070,244-byte stream is 8.38% larger than tier 0. Across the three
native photo originals the median growth is 4.91% and the largest
increase is 8.38%. Median throughput is 2.45× and 2.10×, meeting the
photo gates independently of the graphics results.

Both night photo streams match the source RGB8 samples through
zenjxl-decoder at one, four and eight workers, jxl-rs and djxl v0.12.
Both timing arms use the row-span decoder `c266f4c7` at the same
warm allocating output endpoint and three interleaved blocks of
five samples per arm, including A/A. The serialized balanced-photo
and prefix-control run lasted 148 seconds at 0.55 GiB peak RSS.
[Results, p95 and stream hashes](faster_decode_imazen26_balanced_night_2026-11-22.tsv).
Raw records are retained in `imazen26-lossless/night-photo-balanced`.

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

### Whole-system retest with row-bounded RLE

The decoder's [row-span optimization](https://github.com/imazen/zenjxl-decoder/commit/c266f4c7f2b602b66d5d5f678917dffd645a119f)
ships on main. It consumes repeated residuals in rows and copies
zero-residual spans when the Gradient state allows it. The system retest
uses the frozen per-channel bitstreams from the preceding study and an
experimental decoder with fused literals and run lengths.

Across eight source and size pairs, each file is smaller than the
stock `--fast-decode -g 0` stream and all 24 cross-system readbacks
match. The terminal uses 17.35% of its stock bytes and reaches
3.027× / 1.995× paired throughput at one and eight workers. The
scanned-text source uses 96.40% of its stock bytes and reaches
1.182× / 1.184×. The baseline system uses the landed decoder.

The [whole-system measurements](https://github.com/imazen/zenjxl-decoder/blob/de5a5ef3fda613a6d858ef7e3a76c72057e07906/benchmarks/lossless_rle_throughput_2026-11-22.md)
also include the 16-group scheduling repeat. With that variant the
native photo medians reach 125.82 / 540.53 MP/s and 1.380× / 1.341× the
stock throughput at one and eight workers. That variant fell short of the native
eight-worker absolute target. Peak RSS in the direct system comparisons
was 0.11 GiB with standard batches and 0.12 GiB in the 16-group repeat.

### Shared and local prefix selection

The encoder experiment uses one table per Gradient channel and can share
those tables across all groups. Groups reset their entropy and predictor
state independently. Shared plain coding improved one-worker throughput
over the local-table candidate by 13.4% on the textured native photo and
15.2% on the scan. Both files grew by less than 5% versus local tables.
Shared RLE performed better than local tables on the terminal and mixed
page. Local tables kept an 8-worker advantage on the plot over shared RLE.

The experimental selector tries shared plain, then shared RLE. A shared
candidate must fit the stock single-table size and cost at most 5% more
than the local candidate, including padded sections and TOC. It uses
local tables when both shared candidates exceed the cap. This threshold
came from the pilot. The throughput confirmation uses the same source
images. The encoder reproduced all eight selected stream hashes.

| Warm allocating RGB8 | Median 1 worker | Median 8 workers | Paired throughput at 1 / 8 workers |
|---|---:|---:|---:|
| Four photo renditions | 129.49 MP/s | 564.41 MP/s | 1.441× / 1.386× |
| Two native photos | 132.83 MP/s | 564.41 MP/s | 1.464× / 1.443× |
| Four graphics and scan inputs | 313.19 MP/s | 936.16 MP/s | 3.035× / 1.759× |

The baseline is stock `--fast-decode -g 0` on the landed row-span decoder.
The selected streams use the experimental fused-token decoder with a
16-group batch. Every selected file shrank. The terminal used 17.32% of
stock bytes, the mixed page 38.02%, and the scan 99.24%. All 24 direct
stock/profile readbacks matched across decoder binaries and worker counts.
All eight integrated streams also passed six exact source checks each.
The forced shared modes each passed ten checks over RGBA8 and GrayAlpha8
odd crops. On native RGB16 HDR 1064 the integrated candidate occupies
89.82% of stock bytes. Both RGB16 files match the input sample-for-sample
in all three decoders. The RGB16 run did not time the profile.

Warm results use five interleaved blocks of five samples. The largest A/A
p95 ratio is 1.073×. Cold results use three blocks of five fresh processes.
The four photo medians reach 122.91 / 413.43 MP/s at one and eight workers.
Two cold eight-worker A/A ratios, 1.115× and 1.421×, repeat at 0.971×
and 1.054× with nine blocks. Cold and warm both allocate output.
The warm, cold and cold-tail repeat peak RSS values from `run-heavy`
are 0.12, 0.10 and 0.03 GiB. The integrated build, exact replay
and RGB16 run used 1.11, 0.58 and 0.61 GiB.

[Encoder selector](https://github.com/imazen/jxl-encoder/commit/15c4b1259672ca1eb3718701a9432101dbd2a658),
[decoder experiment](https://github.com/imazen/zenjxl-decoder/commit/01f35c77411422dbc8da0da2f6343f40dd07a15d),
and [per-cell timings and rate](faster_decode_imazen26_shared_prefix_2026-11-22.tsv).
The archive at `imazen26-lossless/shared-channel-prefix` contains the
commands, raw samples, source inputs, binary receipts and pixel checks.

On the landed decoder alone, the same selected streams reach
106.47 / 488.12 MP/s across the four photo renditions and
1.163× / 1.105× stock throughput. The scan reaches 1.012× / 0.932×.
The current profile therefore depends on the decoder companion
for the warm photo targets. All 24 stock/profile readbacks match.
Three interleaved blocks of five samples used 0.11 GiB peak RSS,
and the largest A/A p95 ratio was 1.086×.

### Fixed 128 px groups with Hybrid trees

The terminal keeps the 4,096-color palette cap. Every other encode
uses the same tier-1, effort-7 request with patches available and stock
palette controls. The experiment changes only the global, local or
Hybrid MA tree policy. All 12 streams match the source across five
readback endpoints, including jxl-rs and djxl v0.12.

| Native input | Hybrid bytes vs tier 0 | Paired throughput at 1 / 8 workers |
|---|---:|---:|
| Terminal 8457 | −8.35% | 3.960× / 1.515× |
| Plot 7026 | +14.70% | 0.881× / 0.967× |
| Mixed page 8106 | +10.71% | 1.284× / 1.372× |
| Scan 6824 | +8.22% | 2.097× / 1.925× |

Hybrid meets the graphics rate limits at 128 px. The three graphics
have median growth of 10.71% and p90 growth of 14.70%. Their throughput
medians of 1.284× and 1.372× miss both speed gates. The plot's
one-worker p95 increases by 13.2%, with A/A at 1.008×.

AVX2 cycle sampling on the plot and mixed page puts 86.54% and
79.50% of Hybrid user cycles in Modular NoWpTree channel decoding,
respectively. Their tier-0 streams instead spend 84.41% and 89.02%
in GeneralTree channel decoding. This identifies the decoder path
to measure when isolating the tree policy's speed cost.

The same row-span decoder ran both streams at each worker count,
with five interleaved blocks of five warm allocating decodes.
All 12 paired stock/Hybrid PNG16 checks match. Peak RSS was 0.75 GiB
for the 16-second exact-stream grid, 0.11 GiB for the 218-second timing
run and 0.02 GiB for the six-second cycle sampling.

[Layout rates and hashes](faster_decode_imazen26_balanced_tree_layout_2026-11-22.tsv)
and [paired Hybrid timings and p95](faster_decode_imazen26_balanced_hybrid_timing_2026-11-22.tsv).
The raw commands and profiles are in `imazen26-lossless/balanced-tree-layout`
on i270.

### Native photo group widths

The balanced photo setting is the existing `--lossless --faster-decoding 1
-g 0` at effort 7. The photo rate and throughput results above qualify
this choice separately from the graphics controls.

The separate throughput experiment uses fixed Gradient prediction and
selects among shared plain, shared RLE and local prefix coding. Only the
Modular group width changes between the two experiment arms. The
comparison is `--fast-decode -g 0` on the row-span decoder
`c266f4c7`, which uses 128 px groups. All four experiment files match
the source in zenjxl-decoder at 1/4/8 workers, jxl-rs and djxl v0.12.

| Experiment group width | Bytes vs stock, 1007 / 1019 | Photo median MP/s, 1 / 8 workers | Median paired speedup, 1 / 8 workers |
|---|---:|---:|---:|
| 256 px | −8.76% / −27.52% | 132.95 / 511.42 | 1.462× / 1.272× |
| 512 px | −8.81% / −23.44% | 124.81 / 382.19 | 1.371× / 0.939× |

The 256 px width meets the 120 MP/s one-worker target and both relative
throughput targets. It is 38.58 MP/s below the 550 MP/s eight-worker
target. At eight workers it improves on both originals compared with 512 px.

The experiment encoder is [15c4b125](https://github.com/imazen/jxl-encoder/commit/15c4b1259672ca1eb3718701a9432101dbd2a658).
The timing decoder uses frame-owned prefix lookups from
[the decoder experiment](https://github.com/imazen/zenjxl-decoder/commit/e8af08b565ef).
The timing binary was built from `2fd15d5a`. The experiment bookmark
adds test-only controls. Measurements use three interleaved blocks
of five warm allocating RGB8 decodes per arm plus A/A, on i270.
[Per-image results, p95 and stream hashes](faster_decode_imazen26_photo_group_width_2026-11-22.tsv).
Raw logs and readbacks are under `imazen26-lossless/photo-group-layout`.

### Independent throughput night photo

The 256 px profile was selected before timing on night photo 1069.
It reaches 177.54 MP/s at one worker and 673.36 MP/s at eight workers.
Its paired speedups are 1.721× and 1.338× versus the current
`--fast-decode -g 0` stream. The file shrinks from 5,999,352 to
3,929,829 bytes, a 34.50% reduction. Both streams reproduce all source
samples through Zen at 1/4/8 workers, jxl-rs and djxl v0.12.

The three native-photo medians are **151.12 and 562.19 MP/s**, with
paired speedups of **1.560× and 1.284×**. Every file is smaller than
stock. The textured original reaches 460.65 MP/s at eight workers.
This meets the current photo median targets on warm allocating decodes
with the experimental encoder and decoder. The balanced setting is
available through the existing CLI. The throughput variant still needs
the production encoder integration and decoder verification.

The two decoder revisions, encoder revision, pinning and warm
interleaved protocol are the same as in the group-width comparison.
[Paired results, p95 and stream hashes](faster_decode_imazen26_throughput_night_2026-11-22.tsv).
Raw commands and readbacks are in `imazen26-lossless/night-photo-throughput`.

### Production throughput profile

The explicit 256 px Gradient prefix and distance-1 RLE options activate
the full policy through the existing [CLI settings](../jxl-encoder-cli/README.md#faster-lossless-decoding-for-photographs).
Legacy strict settings and profiles keep their existing routes.
The encoder selects local or shared tables on eligible frames using
emitted section and TOC sizes. It retains the ordinary route for
frame layouts with auxiliary Modular images or local transforms.

The production encoder reproduces all three measured profile streams
byte-for-byte. A separate replay also compares six tier-0, balanced
and stock-fast stream hashes with the baseline encoder. Thirteen
encodes, including the plot, terminal, mixed page and an odd RGBA
control, match the source in zenjxl-decoder at one, four and eight
workers and in jxl-rs and djxl v0.12.

| Native photograph | Bytes vs stock | MP/s, 1 / 8 workers | Paired speedup, 1 / 8 workers |
|---|---:|---:|---:|
| 1007 textured | −8.76% | 114.60 / 451.60 | 1.365× / 1.230× |
| 1019 smooth | −27.52% | 151.39 / 573.60 | 1.561× / 1.376× |
| 1069 night | −34.50% | 177.73 / 666.32 | 1.720× / 1.437× |

The three-photo median is **151.39 / 573.60 MP/s** and **1.561× / 1.376×**
relative to stock `--fast-decode -g 0`. The integrated Zen decoder
runs the frame-owned lookup from
[6000956d](https://github.com/imazen/zenjxl-decoder/commit/6000956dc06e36b04e3b0c4ee9a527e9d6acd913).
The physical host and worker affinity are i270 with the same three
interleaved blocks of five warm allocating RGB8 decodes, plus A/A.
All nine timed endpoint pairs match in PNG16. The complete timing,
encoder byte-lock, divergence-drift and minimal-build run took 168
seconds at 1.34 GiB peak RSS. The independent stream replay used 5.06
GiB peak RSS across its 197-second wrapper, including decoder validation.

[Final rate, throughput and p95](faster_decode_imazen26_throughput_production_2026-11-22.tsv).
[Exact source and codestream hashes](faster_decode_imazen26_throughput_streams_2026-11-22.tsv).
Raw measurements and binary hashes are in `imazen26-lossless/photo-production`.

### Three additional native lossless photo controls

The three imazen-26 test originals previously used for lossy qualification
supply a separate lossless holdout. The aerial cityscape 1009 has repeating
fine edges, the macro photograph 1427 has detail against defocus, and
glacier 1447 has bright ice and narrow native geometry. The frozen SDR
renders span 7.472 to 12 MP. The four effort-7 settings are ordinary tier 0,
Balanced with 128 px groups, stock `--fast-decode -g 0`, and the production
256 px throughput profile. All encodes request eight workers.
[Source identities and hashes](faster_decode_photo_tiers_holdout_cohort_2026-11-22.tsv).

The encoder binary comes from `31c746be` and the Zen decoder from
`dda06cbe`. The encoder and decoder source and dependency closure
diffs against the inspected main revisions are empty. All 12 streams
reproduce the source RGB8 samples through both Zen decoder revisions
at 1, 4 and 8 workers, and through jxl-rs and djxl v0.12. All nine
historical tier-0, Balanced and stock-fast encodes from `f574a317`
match the current codestream hashes. The decoder baseline is `e58a4838`.

| Profile | Endpoint | Median throughput, 1 / 4 / 8 workers | Paired speedup, 1 / 4 / 8 workers |
|---|---|---:|---:|
| Balanced | Cold allocating | 26.76 / 87.77 / 154.56 MP/s | 2.377× / 2.175× / 2.145× |
| Balanced | Warm allocating | 26.99 / 89.71 / 158.99 MP/s | 2.387× / 2.192× / 2.041× |
| Balanced | Caller-owned | 27.30 / 91.91 / 163.52 MP/s | 2.391× / 2.225× / 1.998× |
| Throughput | Cold allocating | 126.25 / 353.35 / 436.98 MP/s | 1.177× / 1.208× / 1.035× |
| Throughput | Warm allocating | 130.14 / 383.78 / 514.82 MP/s | 1.171× / 1.146× / 1.045× |
| Throughput | Caller-owned | 139.51 / 450.09 / 607.73 MP/s | 1.166× / 1.217× / 1.202× |

Balanced meets both speed floors across the measured endpoints with
+5.28% median and +6.02% p90 bytes versus ordinary tier 0. Throughput
reduces bytes by 17.70% at the median. The largest remaining file
ratio is 89.14% of stock. The one-worker relative gain misses 1.25×
in all three endpoints. Eight-worker allocating output misses both
the relative and absolute floors. Caller-owned eight-worker output
clears both.

The aerial photo loses 7.44% one-worker warm throughput versus stock,
while using 10.86% fewer bytes. The largest selected normalized cohort
p95 ratio is 1.230× for cold eight-worker throughput, with A/A at
1.030×. Glacier 1447 supplies the only selected per-image p95 above
1.10×, at 1.333× in that endpoint. The caller-owned one-worker throughput
cohort p95 ratio is 1.110×. All cohort A/A p95 ratios are within +10%.
[Endpoint aggregates](faster_decode_lossless_native_holdout_aggregate_2026-11-22.tsv)
and [individual p95 outliers](faster_decode_lossless_native_holdout_outliers_2026-11-22.tsv).

On identical warm baseline files, current versus historical decoder
speedups for tier 0 are 1.190× / 1.163× at one and eight workers.
On stock fast they are 1.212× / 1.141×. The corresponding whole-system
profile ratios are 2.868× / 2.384× for Balanced versus historical tier
0 and 1.352× / 1.245× for throughput versus historical stock fast.
Each ratio is computed from direct paired observations.

The replay measures 2,970 decodes in three interleaved blocks of five
per arm on i270. Cold decodes are first in-memory decodes in fresh
processes. Warm allocating decodes have one warmup. The caller-owned
runner verifies exact RGB8 equivalence between streams and against
allocating decode. The run takes 1,040 seconds at 0.58 GiB peak RSS
under `run-heavy`. Full commands, verifiers, binary fingerprints and
frozen streams are under
`/mnt/v/output/jxl-encoder/fast-decode-2026-11-22/imazen26-lossless/native-holdout-1009-1427-1447`.
[Per-image timings](faster_decode_lossless_native_holdout_cells_2026-11-22.tsv)
and [stream and pixel hashes](faster_decode_lossless_native_holdout_streams_2026-11-22.tsv).
