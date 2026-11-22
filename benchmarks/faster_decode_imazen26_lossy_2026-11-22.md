# Quality-matched faster VarDCT decoding on i270

The profile trials retain patches by using faster-decoding tier 1 and
set EPF independently. The [cohort](faster_decode_imazen26.tsv) supplies
two held-out 1024 px photo crops and four graphics or document controls.
The encoder is `f574a3172b5009c9914e600903fb3602d74b60c6`, effort 7 with
eight pinned P cores. The production timing decoder is
`88ce5198e3649b490875eda30b6f7eda6d5e1271`, and the decoder-only comparison
uses the older `e58a4838725704e37d90f2acd1d11dcccd7ab98a` on the identical
tier-0 stream.

## Held-out photo crops

Both selected recipes use tier 1. Near d1 the same recipe turns off EPF
and Gaborish for both images, with separate measured distances. Near
d4 it turns off EPF and leaves Gaborish on. The byte and speed comparators
are the tier-0 streams at the named band distances.

| Band | Photo | Candidate distance | Byte change | Decode throughput, 1 / 8 workers | SSIMULACRA2 delta | Butteraugli delta |
|---|---|---:|---:|---:|---:|---:|
| d1 | 1007 | 0.98 | −1.82% | 1.189× / 1.175× | +0.4093 | −0.0948 |
| d1 | 1019 | 0.95 | +6.87% | 1.380× / 1.271× | −0.0516 | −0.0241 |
| d4 | 1007 | 3.9 | +1.69% | 1.332× / 1.303× | +0.2775 | −0.2143 |
| d4 | 1019 | 4.0 | +1.95% | 1.861× / 1.480× | +0.4193 | −0.1956 |

The crop medians clear the rate and speed targets in both bands.
At d1 the median throughput is 1.284× / 1.223× and byte growth is
2.53%. At d4 it is 1.596× / 1.392× and byte growth is 1.82%.
Both images use a fast candidate in both bands. The per-band p90 byte
growth on two cases is the larger observation, 6.87% and 1.95%.
These are measured per-image rungs. They do not establish a rule for
choosing the distances during encoding.

At d1, EPF-off alone reached a quality-passing rung at distance 0.98
on the smooth crop. It costs +2.23% bytes versus the coarser search's
d0.90 rung at +8.30%. The textured crop's EPF-off-only result reached
1.089× at one worker after rate matching, which leaves the median below
the throughput target for that pair. On the controls at d4, the terminal,
plot and scanned text have no quality-passing forced EPF candidate within
the 10% byte ceiling. The mixed page retains a candidate. Normal
filtering is the fallback for the remaining cells.

## Native 12 MP photos

The native-size comparison scores the timed Zen RGB8 endpoint as well
as independent jxl-rs output. Both must pass the same quality guards.
Tier 1 is used in each selected cell.

| Band | Photo | Filter choice | Distance | Byte change | Decode throughput, 1 / 8 workers | Zen SSIMULACRA2 delta | Zen Butteraugli delta |
|---|---|---|---:|---:|---:|---:|---:|
| d1 | 1007 | EPF off | 1.00 | +0.33% | 1.099× / 1.104× | −0.0082 | +0.0003 |
| d1 | 1019 | EPF and Gaborish off | 1.00 | +4.21% | 1.449× / 1.887× | −0.0375 | −0.0192 |
| d4 | 1007 | Two EPF passes | 4.00 | +0.89% | 1.223× / 1.184× | −0.0262 | −0.0937 |
| d4 | 1019 | EPF off | 4.08 | +3.39% | 2.078× / 1.654× | +0.3200 | −0.1499 |

The selected native medians are 1.274× / 1.495× and +2.27% bytes near d1,
and 1.650× / 1.419× and +2.14% bytes near d4. The p90 byte growth is
4.21% and 3.39%, respectively.

On 1007 at d1, turning off both filters passes on jxl-rs and increases
Butteraugli by 0.0502 on the timed endpoint, past its 0.0311 limit.
One EPF pass at d4 also passes on jxl-rs and drops 0.2539 SSIMULACRA2 on
Zen, past the 0.25 limit. The remaining selected faster options pass
both endpoints. The d1 results use two Gaborish settings. At d4, two EPF
passes can cover both originals after per-image distance selection.

[Native timings and jxl-rs scores](faster_decode_imazen26_lossy_native_2026-11-22.tsv)
and [Zen RGB8 scores](faster_decode_imazen26_lossy_native_target_2026-11-22.tsv)
record all candidates. The 56-stream quality grid, 1,395 timed decodes
and 69 exact revision/worker comparisons used 2.61 GiB peak RSS over
576 seconds. The timed finalists render in both independent decoders.
For the native A/A arms, 1019 near d4 at one worker exceeds the p95
limit at 1.111×. The data establish per-image choices and cohort
medians. Shipping still requires a policy that chooses and falls back
reliably.

## Quality shifts across the ladder

The ladder covers 20 quality settings from q5 to q99, with 12 rungs at
q5–q60 and eight above q60. It uses both photo crops, the flat and text
regions of the same terminal original, and the aliased plot. All 600
streams fully render through jxl-rs and djxl v0.12. At the same requested
distance, EPF off passes the quality guards on all 20 flat terminal rungs,
five with visible terminal text, and two on the aliased plot.

There are 500 nonbaseline comparisons. Of the 153 that miss either
quality guard at the same distance, 106 have a measured rung that meets
both guards with at most 10% more bytes. Of those, 34 fit within 3% and
16 use at most the baseline bytes. The matching uses the smallest
measured eligible rendition. For example, 1007 at d5 with EPF off
recovers a quality match at d4.5 and +6.79% bytes. Confirmed throughput
below is for the finer d1 and d4 timing comparisons.

[Same-distance counts](faster_decode_imazen26_lossy_ladder_2026-11-22.tsv)
and [rate at matched quality](faster_decode_imazen26_lossy_quality_match_2026-11-22.tsv).

### Tier 4

The tier-4 diagnostic adds 100 ladder streams across the same five
shapes, all fully rendered by both independent decoders. On the finer
d4 photo grid, tier 4 reaches the guards on Zen and jxl-rs at distance
4.10 on 1007 and 4.20 on 1019. The measured files grow by 0.56% and
1.42%. Their paired throughput ratios are 1.661× / 1.308× and
3.119× / 1.927× at one and eight workers. These comparisons use
1024 px crops.

Near d1, the 1007 finalist at distance 1.05 reduces bytes by 4.69%
and misses Zen's Butteraugli guard. On 1019, the sampled d1 grid has
no jxl-rs quality match within the 10% byte ceiling.

The fine-grid and verification run used 0.55 GiB peak RSS over
40 seconds. [Tier-4 timings and jxl-rs scores](faster_decode_imazen26_lossy_tier4_2026-11-22.tsv)
and [Zen scores](faster_decode_imazen26_lossy_tier4_target_2026-11-22.tsv).

At 12 MP near d4, candidate selection used both renderers before
timing. Tier 4 measured **2.594× / 2.660×** median throughput at
one and eight workers, for **+7.40%** median bytes. The textured photo
1007 passed at d3.9 with +9.07% bytes, and smooth 1019 passed at d4.4
with +5.74%. The median rate exceeds the 3% target. At d4, tier 4
on 1007 improves Zen SSIMULACRA2 by 3.29 and increases Butteraugli
by 0.1500, so that rung exceeds the 0.0882 Butteraugli allowance.

The concurrent tier-1 filter controls measured 1.632× / 1.373×
median throughput and +2.14% median bytes, including two EPF passes
on 1007 and EPF off on 1019. The worst A/A p95 ratio was 1.062×
across six source and worker cells. Nine timed streams render in
both independent decoders, and all 27 Zen revision and worker-count
comparisons are exact. The native run used 2.69 GiB peak RSS over
354 seconds.
[Native dual-endpoint quality grid](faster_decode_imazen26_lossy_tier4_native_grid_2026-11-22.tsv)
and [timings](faster_decode_imazen26_lossy_tier4_native_2026-11-22.tsv).

On smooth 1019, extending the search through d4.8 found a smaller
eligible file at d4.55. It grows by **2.48%** and decodes **3.571× /
3.404×** faster at one and eight workers. Both renderers pass the
quality guards. Zen's changes are +0.0557 SSIMULACRA2 and −0.1741
Butteraugli. The smaller d4.60 file misses the Butteraugli guard.

Using this tier-4 file for 1019 and the measured tier-1 EPF2 file for
1007 yields **+1.69%** median bytes and **2.395× / 2.269×** median
throughput at one and eight workers, against each photo's d4 tier-0
stream. Both tier-0 streams reproduce byte-for-byte with `--no-patches`.
These are two per-image choices. A production selector still needs
held-out evidence and patch-heavy controls. The two patch controls used
1.26 GiB peak RSS over three seconds.

Six overlapping renditions reproduced the bytes and both pairs of
quality scores exactly. All five timed streams render in both
independent decoders and pass 15 exact Zen revision and worker-count
comparisons. The refinement used 2.61 GiB peak RSS over 192 seconds.
[Fine quality grid](faster_decode_imazen26_lossy_tier4_native_refine_grid_2026-11-22.tsv)
and [timings](faster_decode_imazen26_lossy_tier4_native_refine_2026-11-22.tsv).

## Repeated crop timings

Nine interleaved blocks of five decodes give median paired ratios of
1.277× / 1.231× at d1 and 1.599× / 1.380× at d4. All four selected
files meet both quality guards on the timed Zen output. The d4
eight-worker A/A on 1019 still crosses the p95 limit at 1.104×, with
a median throughput ratio of 1.011×. The extended crop and ladder run
used 0.56 GiB peak RSS over 129 seconds.
[d1 results](faster_decode_imazen26_lossy_refine_confirm_2026-11-22.tsv)
and [d4 results](faster_decode_imazen26_lossy_match_confirm_2026-11-22.tsv).

## LF/DC and HF metadata profile

At eight workers on the native photos, existing decoder diagnostics
measure 2.61–6.33 ms in LF groups, including LF/DC reconstruction and
HF metadata. That phase accounts for 9.0–16.6% of median same-run
decode latency across the five profiles. The parallel LF and HF join
uses 2.37–6.13 ms, with a 0.18–0.22 ms copy-back. The combined AC decode
and render phase takes 22.5–39.7 ms. Independent `perf` sampling on the
same files puts AC group entropy and EPF among the largest CPU costs.
The fastest tested file still spends 15.93% of sampled CPU time handling
user address faults, which is a useful place to test allocation reuse.
This phase capture measures the current implementation. The optimization
targets need a same-stream before/after comparison.

The oracle and profile run took 34 seconds and used 0.24 GiB peak RSS.
Raw sampling stacks and existing phase logs are under `lf-profile`
in the run directory.

## Evidence

The [match results](faster_decode_imazen26_lossy_match_2026-11-22.tsv)
include all timed finalists, both decoder revisions, and A/A.
The [d1 refinement](faster_decode_imazen26_lossy_refine_2026-11-22.tsv)
evaluates disabling Gaborish with automatic and forced EPF. Twenty overlapping rows
reproduced the metrics, bytes, stream hashes and RGB8 hashes exactly.
The timed streams fully render with independent jxl-rs
`f370be423933628eb7f783f1b61d4c765eca5f79` and djxl v0.12.0
(`a7a9c78`). Across both phases, 195 comparisons match RGB8 exactly
between the two Zen builds and all worker counts.

The quality metrics use pinned jxl-rs's RGB8 reconstruction against
the RGB8 source, with SSIMULACRA2 0.7.1 and Butteraugli 0.9.4
(`d2466a4e`). Each reported decode timing is from a fresh process that
warms once, then times one allocating in-memory RGB8 decode. Use
three interleaved blocks of five samples at 1, 4 and 8 workers.
The matched phase used 0.61 GiB peak RSS over 141 seconds and the
refinement and conformance checks used 0.52 GiB over 65 seconds.

Controls with a p95 ratio over 1.10 are visible in the summaries. For
example, the baseline A/A p95 at one worker on 1007 near d4 reached
1.562× while its median throughput ratio was 0.993×. The first
15-sample p95 is the worst sample per cell, so the control gate needs
longer confirmation. Raw inputs, streams, hashes, commands, and timing
samples are under `/mnt/v/output/jxl-encoder/fast-decode-2026-11-22/imazen26-lossy`
on i270.

## Rate-first filter replay

The replay chooses the smallest measured tier-1 file that passes both
quality guards on both renderers and stays within 10% of tier-0 bytes.
It considers EPF-off settings near d1 and zero, one or two EPF passes
near d4. The d1 candidates can vary Gaborish. All tier-1 choices keep
patch selection available. When no scored fast rendition qualifies,
the replay uses the tier-0 file. It uses the frozen distance grid and
selects without timing data.

| Photo size | Quality band | Median byte change | Median paired throughput, 1 / 8 workers |
|---|---|---:|---:|
| Native | d1 | +2.27% | 1.281× / 1.514× |
| Native | d4 | +2.14% | 1.667× / 1.421× |
| 1024 px crop | d1 | +0.21% | 1.212× / 1.213× |
| 1024 px crop | d4 | −0.61% | 1.445× / 1.295× |

Both photo originals use fast settings in every size and band. The
largest byte increase on a photo is 4.21%. The scan crop retains EPF
off near d1 and falls back to tier 0 near d4. This replay reuses the two
originals that informed the filter search, so independent source
qualification remains open. It chooses from captured renditions and
scores. The encoder still needs a runtime search and its encode-cost
measurement.

The fresh timing uses the row-span decoder from
[zenjxl-decoder main](https://github.com/imazen/zenjxl-decoder/commit/c266f4c7f2b602b66d5d5f678917dffd645a119f)
on both sides. It alternates three blocks of five warm allocating RGB8
decodes at one, four and eight workers. A/A tails above 1.10 remain in
the timing table. The run used 0.21 GiB peak RSS over 118 seconds.

[Selection script](../scripts/fast-decode/select_scored_filters.py),
[selected hashes and quality deltas](faster_decode_imazen26_lossy_rate_first_2026-11-22.tsv)
and [paired rate and timing](faster_decode_imazen26_lossy_rate_first_timing_2026-11-22.tsv).
Raw timing and command records are in `imazen26-lossy/gated-filter-replay`
on i270.
