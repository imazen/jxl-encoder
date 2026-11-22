# Faster-decode photo and graphics ladder replay on i270

The replay tests q5–q99 at 20 rungs, with 12 at q5–q60. It uses five
imazen-26 shapes from four originals. Photos 1007 and 1019 are 1024 px
crops with textured and smooth content. The graphics are the 64 px
aliased plot and the flat and visible-text 256 px crops from the same
terminal original. All 700 retained streams were encoded at e7 with
[jxl-encoder f574a317](https://github.com/imazen/jxl-encoder/commit/f574a3172b5009c9914e600903fb3602d74b60c6).

The replay checks all stream and reference-raster hashes and reproduces
all 700 pinned jxl-rs RGB8 readback hashes. It scores current
[zenjxl-decoder dda06cbe](https://github.com/imazen/zenjxl-decoder/commit/dda06cbe104184544a5e4cf9dcb13901fdb01342)
RGB8 pixels separately. Both renderers must score at least the
tier-0 SSIMULACRA2 score minus 0.25 points. Butteraugli may increase
by at most max(0.01, 2% of tier 0). Every selected
stream renders in djxl v0.12. The 216 distinct selected streams return the same RGB8
pixels on both Zen revisions and at each of 1, 4 and 8 workers, for
648 exact comparisons. The old decoder is
[e58a4838](https://github.com/imazen/zenjxl-decoder/commit/e58a4838725704e37d90f2acd1d11dcccd7ab98a).

The 3% and 10% file-size caps prefer fewer restoration filters among
eligible tier-1 streams, then fewer bytes. Normal-filter tier 1 is
eligible when no reduced-filter choice fits. Tier 4 has a separate
10% cap. Each comparison uses the same tier-0 file when its search
finds no eligible choice. The filter search may change distance
within the retained grid. This replay measures the broader
quality ladder. The named photo encode tool uses its narrower
d1 and d4 candidate sets.

## Warm with allocation

| Photo rungs | Choice | 1 worker | 4 workers | 8 workers | Median bytes | p90 bytes | Reduced-filter or tier-4 choices |
|---|---|---:|---:|---:|---:|---:|---:|
| All | 3% cap | 1.118× | 1.174× | 1.185× | +0.18% | +2.29% | 32/40 |
| All | 10% cap | 1.445× | 1.404× | 1.336× | +2.15% | +8.83% | 40/40 |
| All | Tier 4 | 1.219× | 1.233× | 1.257× | 0.00% | +7.50% | 25/40 |
| q5–q60 | 3% cap | 1.175× | 1.195× | 1.211× | +0.63% | +2.76% | 16/24 |
| q5–q60 | 10% cap | 1.716× | 1.553× | 1.485× | +4.93% | +9.39% | 24/24 |
| q65–q99 | 10% cap | 1.117× | 1.174× | 1.176× | +0.76% | +7.05% | 16/16 |

All timings use CMS-disabled in-memory decode and three interleaved
blocks of five processes per arm. The warm process runs one warmup
decode. Source/rung/worker speedup is the median of the paired sample
ratios. Class speedups are medians across cells. Class p95 is computed
after normalizing individual samples by each cell's baseline median.
Across all photo rungs at eight workers the 10% cap gives 1.336×
with the decoder held fixed and 1.322× against tier 0 on the old
decoder. The same-file decoder comparison is 0.988× and A/A is
0.997×. The whole-system ratio is measured from the paired samples.

On the visible-text terminal crop, tier 4 falls back at every rung.
The 10% tier-1 comparison selects reduced filters in 20/20 cells,
measures 1.238× at eight workers, and uses +2.51% median bytes. Each
full shape's pooled p95 is within 10% of baseline at every worker
count. The outlier table preserves all 23 individual warm profile cells above
1.10× alongside their A/A ratios.

[Class and quality-band results](faster_decode_lossy_ladder_replay_cohorts_2026-11-22.tsv)
and [warm tail outliers](faster_decode_lossy_ladder_replay_outliers_2026-11-22.tsv).

## Cold and caller-owned output

Both photo crops are replayed at q5, q55, q90 and q99. On these same
eight cells, the 10% cap has median eight-worker speedups of 1.154×
cold, 1.159× warm with allocation and 1.241× warm with reused output.
At one worker they are 1.106×, 1.102× and 1.109×. The reused output
harness measures preallocated RGB8 buffers and checks each stream
against its own allocating decode. The cold arm measures the first
decode of each process into a newly allocated buffer.

The separate 45-sample replay takes all seven cold or reused-output
profile rows above 1.10× p95 back below that limit. The largest
initial cold outlier was photo 1007 at q99 with tier 4 and four workers.
It moves from 1.746× to 0.937×, with A/A at 0.936× and 1.027× respectively.
The reported initial and repeated rows remain separate.

[Matched endpoint cells](faster_decode_lossy_ladder_replay_photo_endpoints_2026-11-22.tsv)
and [p95 replays](faster_decode_lossy_ladder_replay_tail_repeats_2026-11-22.tsv).

## Reproduction

The [driver instructions](../scripts/fast-decode/README.md#matched-quality-timing-across-the-ladder)
describe the frozen-stream inputs and commands. On i270 the run root is
`/mnt/v/output/jxl-encoder/fast-decode-2026-11-22/imazen26-lossy/ladder-dual-endpoint`.
It retains binary and input manifests, all 20,610 warm observations,
both metrics for each rendition, exact worker readbacks, the
per-rung photo table in `report-all/ladder-photo-rungs.tsv`, and
cold, reused and repeated tail data. The completed warm run took
359 s at 0.57 GiB peak RSS, the endpoint replay took 55 s at
0.06 GiB, and the tail replay took 32 s at 0.05 GiB under
`run-heavy`. The loader separately rejects a deliberately changed
frozen stream hash. Replaying selection from the finalized scores
reproduces every selected row exactly.
