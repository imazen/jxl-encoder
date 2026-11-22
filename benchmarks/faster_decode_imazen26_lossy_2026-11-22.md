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
