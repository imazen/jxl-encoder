# Photo fast-decode replay with libjxl v0.13.0

The existing lossy photo tiers clear the warm allocating throughput and
rate targets on i270 at both quality bands. All 60 settings shared with the earlier native sweep
reproduce their stream hashes and both renderers' Butteraugli and
SSIMULACRA2 scores. The full search scores 88 candidates across four sRGB
originals. The Display P3 original retains its tier-0 result and is counted
in the photo medians and fast-selection rates.

| Band | Tier | Paired median speedup at 1 / 4 / 8 workers | File growth median / p90 | Fast selections |
|---|---|---:|---:|---:|
| d1 | Balanced | 1.219× / 1.564× / 1.515× | 0.74% / 2.66% | 4/5 |
| d1 | Throughput | 1.220× / 1.659× / 1.653× | 2.73% / 9.08% | 4/5 |
| d4 | Balanced | 1.357× / 1.196× / 1.241× | 1.25% / 2.51% | 4/5 |
| d4 | Throughput | 1.362× / 1.192× / 1.230× | 1.67% / 2.76% | 4/5 |

Each speedup uses the same Zen decoder at the same worker count on the
candidate and its tier-0 control. The sRGB candidates meet the existing
Butteraugli and SSIMULACRA2 guards in both Zen and jxl-rs. These timings use
warm allocating RGB8 with pinned cores, three interleaved blocks of five
samples per arm, and an A/A arm. This replay supports retaining the current
photo tier settings across the reference change. The readback stage passes
69 exact decoder and worker-count comparisons. The combined encoder at
`82288f62` reproduces the 23 distinct selected stream hashes, including
the P3 fallback. The full ICC-and-photo command takes 1,208 seconds at
2.61 GiB peak RSS, including the encoder builds.
[Photo tier definitions and gates](faster_decode_profile_2026-11-22.md).

The same selected streams also run in fresh processes with zero warmup and
with warm caller-owned output. Each candidate and control retains its own
quality-matched pixels. The caller-owned runner checks every stream
against its allocating readback.

| Band | Tier | Cold allocating speedup at 1 / 4 / 8 workers | Warm caller-owned speedup at 1 / 4 / 8 workers |
|---|---|---:|---:|
| d1 | Balanced | 1.219× / 1.464× / 1.420× | 1.230× / 1.274× / 1.296× |
| d1 | Throughput | 1.219× / 1.518× / 1.484× | 1.187× / 1.472× / 1.499× |
| d4 | Balanced | 1.362× / 1.240× / 1.211× | 1.393× / 1.321× / 1.350× |
| d4 | Throughput | 1.366× / 1.226× / 1.217× | 1.388× / 1.322× / 1.330× |

Balanced clears both lossy speed targets in all three measured modes.
Throughput falls below the 1.20× target at d1 with one worker and
caller-owned output. Balanced provides the qualified choice for that
endpoint. Both tiers clear all measured eight-worker targets. The cold
and caller-owned passes take 480 seconds at 0.38 GiB peak RSS.

## Native lossless photos

The combined encoder reproduces all twelve pre-migration stream hashes on
the foliage, sky/architecture and low-light originals. All fifteen streams,
including the three libjxl v0.13.0 effort-7 controls, reproduce every RGB8
source sample in Zen at 1, 4 and 8 workers, jxl-rs main, and djxl.

| Setting | Comparison | Paired median speedup at 1 / 4 / 8 workers | Median MP/s at 1 / 4 / 8 workers | Median / p90 file growth |
|---|---|---:|---:|---:|
| Fixed Gradient, prefix and selected RLE, 256 px groups | Stock `--fast-decode -g 0` | 1.356× / 1.361× / 1.315× | 165.49 / 456.77 / 683.60 | −27.52% / −8.76% |
| Tier 1, 128 px groups with patches | Effort-7 tier 0 | 2.720× / 2.405× / 2.332× | 29.28 / 98.68 / 176.77 | +4.90% / +8.38% |

The warm allocating medians clear both lossless speed and rate gates on
this feature panel. The throughput setting saves 8.76%, 27.52% and 34.50%
on the three files. Its one-worker ratios are 1.038×, 1.356× and 1.856×.
The cold and caller-owned replay uses the same source-exact streams.
Both lossless settings clear their targets at these endpoints on the
three-photo panel.

| Setting | Endpoint | Paired median speedup at 1 / 4 / 8 workers | Median MP/s at 1 / 4 / 8 workers |
|---|---|---:|---:|
| Fixed Gradient, prefix and selected RLE | Cold allocating | 1.358× / 1.429× / 1.351× | 159.77 / 422.42 / 616.88 |
| Fixed Gradient, prefix and selected RLE | Warm caller-owned | 1.391× / 1.508× / 1.449× | 184.86 / 559.83 / 822.94 |
| Tier 1, 128 px groups with patches | Cold allocating | 2.692× / 2.381× / 2.258× | 29.14 / 96.76 / 176.57 |
| Tier 1, 128 px groups with patches | Warm caller-owned | 2.744× / 2.448× / 2.295× | 29.87 / 100.74 / 181.25 |

The cold pass verifies each profile against its baseline at 1, 4 and 8
workers before timing. The caller-owned pass verifies reuse against the
allocating output and checks exact samples across streams. The two endpoint
replays take 402 seconds at 0.27 GiB peak RSS.
[Earlier endpoint and holdout results](faster_decode_imazen26_lossless_2026-11-22.md).

The lossless command takes 458 seconds at 0.58 GiB peak RSS. It encodes at
effort 7 with eight threads and uses the same three interleaved five-sample
blocks as the lossy comparison. The fixed-Gradient setting uses these CLI
options.

```sh
cjxl-rs "$PHOTO" "$OUT" -e 7 --lossless --threads 8 \
  --no-ans --no-tree-learning -P 5 --lz77 --lz77-method rle \
  --no-patches --no-squeeze -g 1
```

## Reproduction

The encoder includes the forced-prefix ICC change in `8fe03b13`.
Zen uses `05042e1b`, the historical Zen control uses `e58a4838`, and
the released libjxl tools report `v0.13.0 5a7da83`. The independent
decoder uses jxl-rs main at `f370be42`. The source and executable
SHA-256s are in the run manifest. The cohort consists of the five originals
from the existing photo filter manifest. [Inputs](faster_decode_imazen26_photo_filters.tsv).

```sh
python3 scripts/fast-decode/measure_lossy.py \
  "$OUT" benchmarks/faster_decode_imazen26_photo_filters.tsv "$CORPUS" \
  "$ENCODER" "$ZEN" "$ZEN_BASELINE" "$DJXL_V013" "$JXL_RS_MAIN" "$SCORER" \
  --stage photo-tiers-native
python3 scripts/fast-decode/summarize_lossy.py "$OUT" "$SUMMARY"
```

Full commands, raw timings, quality scores, decoder readbacks and streams are
under `/mnt/v/output/jxl-encoder/libjxl-v013-migration-2026-11-25/` on i270.
