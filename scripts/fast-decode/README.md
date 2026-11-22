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
  uses a pinned terminal crop with visible text.
- `verify --measured MATCH_DIR` checks exact RGB8 for both timed decoder
  binaries across measured worker counts and scores the target decoder.
- `confirm --measured MATCH_DIR --blocks 9` repeats the photo timing
  using the retained tier-0 and EPF-off files, including the
  EPF-off/Gaborish-off arm when available.

The driver selects the smallest candidate per profile that passes both
quality guards and the per-file rate ceiling. Timed normal-filter and
tier-0 arms support fallback analysis. The same stream runs on current
and baseline decoders, with A/A on the current baseline. Each timing block
has five interleaved processes per arm. Each process warms once, then
times an allocating in-memory RGB8 decode. The recorded pixel count and
CLI throughput recover individual decode latency.
