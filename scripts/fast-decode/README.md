# Fast-decode profile measurements

The [cohort manifest](../../benchmarks/faster_decode_imazen26.tsv) pins
the PNG-v3 originals. The driver checks each SHA-256 before extracting
the recorded crop. The scorer takes packed RGB8 sRGB input.
It uses SSIMULACRA2 0.7.1 and the Butteraugli revision pinned by the
encoder workspace. It runs outside the encoder's dependency graph.

Build the scorer with the locked standalone manifest.

```sh
cargo build --locked --release --manifest-path scripts/fast-decode/score-rgb8/Cargo.toml
```

Run the driver on i270 through the local resource wrapper. Supply the
release binaries and paths explicitly. The output directory must be new.

```sh
python3 scripts/fast-decode/measure_lossy.py \
  "$OUT" benchmarks/faster_decode_imazen26.tsv "$CORPUS" \
  "$ENCODER" "$ZEN_CURRENT" "$ZEN_BASELINE" "$DJXL_V012" "$JXL_RS" "$SCORER" \
  --stage match
python3 scripts/fast-decode/summarize_lossy.py "$OUT" "$SUMMARY"
```

- `match` scores the two 1024 px photo crops and four SDR controls.
- `refine` expands the near-d1 photo ladder and independently disables
  Gaborish.
- `native` probes both distance bands on the two 3000×4000 originals.
- `ladder` records quality across the encoder's quality scale.
  `ladder-gaboff` extends it to EPF and Gaborish off. `ladder-text`
  uses a pinned terminal crop with visible text. `ladder-tier4`
  provides the tier-4 diagnostic on all five shapes.
- `tier4-match` refines the tier-4 distance around d1 and d4 on the
  two photos, then times the measured quality matches within the byte limit.
- `tier4-native` refines d4 at 12 MP. Candidate selection requires both the
  independent jxl-rs and the measured Zen RGB8 outputs to clear the guards.
  The results also include the best measured tier-1 filter choices.
- `tier4-native-refine` extends the smooth photo's d4 search to measure
  how far tier 4 can reduce rate while holding quality.
- `d1-independent-filter` tests native photos 1007, 1019 and 1069 with
  EPF and Gaborish selected independently of the DC coding tier.
- `photo-filter-holdout` applies the frozen d1 and d4 options to
  backlit interior 1237. Both stages use the
  [photo filter manifest](../../benchmarks/faster_decode_imazen26_photo_filters.tsv)
  and match the two rendered RGB8 endpoints.
- `palette-grid` compares exact lossless graphics streams across Modular
  group sizes, palette caps and channel compaction. It checks the source
  pixels in three decoders before timing distinct streams.
- `verify --measured MATCH_DIR` checks exact RGB8 for both timed decoder
  binaries across measured worker counts and scores the target decoder.
- `confirm --measured MATCH_DIR --blocks 9` repeats the photo timing
  using the retained tier-0 and EPF-off files, including the
  EPF-off/Gaborish-off arm when available.

Use `match_ladder.py OUTPUT LADDER LADDER_GABOFF LADDER_TEXT` for the
smallest measured candidate matching each tier-0 quality rung. The table
preserves the same-distance verdict, the rate after matching, and missing
matches at quality extremes.

The driver selects the smallest candidate per profile that passes both
quality guards and the per-file rate ceiling. Timed normal-filter and
tier-0 arms support fallback analysis. The same stream runs on current
and baseline decoders, with A/A on the current baseline. Each timing block
has five interleaved processes per arm. Each process warms once, then
times an allocating in-memory RGB8 decode. The recorded pixel count and
CLI throughput recover individual decode latency.

Replay measured d1 and d4 selections with both renderers' quality guards
and only measured candidate bytes as the primary ranking criterion.

```sh
python3 scripts/fast-decode/select_scored_filters.py LOSSY_RUN_DIR SELECTED.tsv
```

The `photo-filter-backlit-bracket` stage tests lower distances on image
1237 with EPF and Gaborish off. It times the ranked selection and the
smallest distinct filter-free stream within the 10% rate cap that passes
both rendered-quality guards. Keep this tuning bracket separate from
the frozen `photo-filter-holdout` result.
