# Native JXL targeting with complete candidate maps — September 8, 2026

The shared Rust controller now drives JXL's native allocation loop and is
calibrated only on canonical imazen-26 training families. This is a reproducible
development baseline, **not a shippable model or qualified spatial RD win**.
D's active H3 maps do not establish a general independent-judge benefit.

## Owner and configuration

`zensim-target::SeedCurve` now owns the existing median/monotone-envelope fitter
and inverse; `demo_matrix` delegates to it with its JSON layout preserved.
`target_search_with_backend_and_bake` accepts a codec-native backend, preserving
the existing search and complete `BakeScorer` surface. No built-in codec feature
is required for that adapter. The JXL `zensim_diffmap_rd` example owns the native
adapter and byte/decode/work records; zensim's existing RD analyzer validates
and summarizes them. No second scorer, calibration algorithm or controller was
introduced. The old target mode remains historical.

Three arms use distance 0.01..25, effort 8, Zenjxl strategy, no automatic
resampling, opaque sRGB8, formula revision 1 and eight Rayon threads:

- `scalar`: zero internal zensim updates, reconstructions or maps.
- `neutral`: two native updates, three current complete-model maps, H3 gain 0.
- `active`: the same native work with H3 gain 10 and bin 8.

No native target controller or legacy secant acts inside those encodes. The
shared outer controller receives a target and optionally a frozen training seed
and slope. One shot means one complete encode and independent jxl-rs decode;
internal reconstructions are additional work, explicitly counted.

## Data, bounds and targeting

The [preceding binding record](zensim_candidate_binding_2026-09-08.md) owns the
source selection and chronology: imazen-26 `187fbf338ce08e8e6654db7f04ddae58d5263da2`,
August 27 family split, August 30 measured variant index, 12 train and 8 validation
families (four content classes), existing longest-side-256 PNG bytes. This
calibrates seeds on clean splits; it does not repair the historical model's
training provenance or qualify human rank. No terminal holdout was consumed.

Each arm gets its own train-only calibration from 21 log-spaced distances:
756 train encodes. All 504 validation bound encodes finish before the first
steering run. Neither bounds nor witness knobs enter runtime search. Requests
are `[-10,30,70,90,99]` plus five attained neutral-ladder score quantiles. Only
requests with a measured ±1 witness in **all three arms** enter paired steering.
The band is an experiment setting, not an established perceptual tolerance.

There are **45/80 jointly witnessed targets**, including two negative targets.
Forty are near 90–100. The sparse grid leaves many interior requests unwitnessed;
those are not declared impossible or counted as controller failures. Neutral
floors vary from −40.69 to +17.45 across images. All three arms hit ±1 within
three complete encodes on the admitted subset. Midpoint seeding hits none at
budgets 1–3; its comparison measures the value of calibration, not spatial maps.

| arm | budget | median abs error | p95 abs error | hits ±1 | mean encodes | median total ms |
|---|---:|---:|---:|---:|---:|---:|
| active | 1 | 0.281 | 2.383 | 36/45 | 1.00 | 19.37 |
| active | 2 | 0.188 | 1.000 | 43/45 | 1.20 | 24.13 |
| active | 3 | 0.188 | 0.911 | 45/45 | 1.24 | 24.51 |
| neutral | 1 | 0.178 | 1.943 | 40/45 | 1.00 | 19.73 |
| neutral | 2 | 0.156 | 1.000 | 43/45 | 1.11 | 21.15 |
| neutral | 3 | 0.156 | 0.944 | 45/45 | 1.16 | 20.41 |
| scalar | 1 | 0.369 | 2.345 | 38/45 | 1.00 | 12.17 |
| scalar | 2 | 0.322 | 1.255 | 42/45 | 1.16 | 15.63 |
| scalar | 3 | 0.261 | 0.883 | 45/45 | 1.22 | 17.17 |

P95 uses the existing analyzer's nearest-rank convention. At budget 1, active
maps miss nine targets versus five for neutral maps and seven for scalar-only.
At budget 3 there are no undershoots beyond one point in any calibrated arm.
The shared controller stops on a hit, so the budget is a maximum, not an
instruction to spend all three encodes. Full distributions, classes, per-family
paired deltas and attained extrema are in the [JSON](zensim_native_targeting_2026-09-08.json).

## Independent quality and complete cost

SSIMULACRA2 and Butteraugli independently judge all 1,314 validation ladder and
selected-output pairs. The existing per-image log-byte interpolation owner
compares active outputs with both scalar-only and neutral frontiers without
extrapolation. It deduplicates identical target outputs within arm/budget.
These sparse curves require direct matched-quality confirmation; nominal-target
byte differences are not RD gains.

For the active ladder versus scalar-only, photo mean savings are **−0.32% on
Butteraugli and −0.28% on SSIMULACRA2**. Versus neutral, photo means are **−0.28%
and −0.31%**. Other content classes are mixed. Three-shot active output also
fails to establish a broad photo benefit on both judges. Consequently the
spatial RD gate remains unmet; strong D finite-block coherence alone did not
make this fixed H3 strategy useful across content.

The 810 steering cases use **1,264 full encodes**, 1,264 external search pixel
comparisons, **2,526 internal reconstructions**, 2,526 native pixel comparisons
and 2,526 map evaluations. Terminal verification independently decodes and
scores each selected bitstream again: 810 decodes/comparisons, **no re-encode**.
Finite feature probes used to construct a map are not additional pixel passes.
The validation bound oracle separately costs 504 full encodes, 504 external
scores and 1,008 reconstructions/native comparisons/maps. Training cost is
separate again; none is hidden in a one-shot claim.

Timing above comes from `validate-quiet` after build/test jobs finished; the
process list and capped execution log are preserved. The 29-second full command
includes the bound oracle and output persistence. Table timing includes search
and terminal decode/score, excludes oracle/calibration and PNG persistence, and
still contains research trace/hash overhead. Geometry, arm order and small
sample limit latency generalization. `validate-verified` overlapped a build and
is excluded from performance claims. Process peak RSS is cumulative and includes
harness state; it is not isolated model memory.

## Reproduction and checks

- Local checked driver SHA256: `738d3c90151b4321dbc0d897ab6b29576228e54953f766493f283019462f3231`.
- Model D SHA256: `cd1098b450ef6941b6925b24bcbd129715b6f07c4fe84838a92e13ab364ddea6`.
  It remains the unqualified baseline, not a newly trained or selected winner.
- Pinned CI-closure driver SHA256: `c54283ba443374f7a1a5cc4af1e92c3e56a15f0f5e4a926fae94c1ecc0a2e43b`.
  The existing clean-source exporter fetched all declared sibling commits; the
  zensim pin advances to `f99b91eb54f00d976417ba18837870530187befb`. Locked
  workspace/all-target and exact-example Clippy pass. Refitting and evaluating
  this dependency set reproduce all 756 train bounds, 504 validation bounds
  and 810 steering cases exactly on every nontiming field. `BUILD_PINNED.json`
  and `REPRO_PINNED.json` bind that check; its overlapping build timings are
  not used in the latency table.
- Rust 1.98.1; exact lockfile, complete source patches including new files,
  parent revisions, source hashes, commands, calibration and binary are retained
  under `/mnt/v/output/zensim/jxl-native-target-2026-09-08`.
- `train-final`/`validate-final` are prior checked-development outputs.
  `train-verified`/`validate-verified` reproduce every nontiming score, knob,
  probe, bitstream and reconstruction hash. `validate-quiet` reproduces all of
  those again. Its judge values are reused only after comparing every PNG's
  bytes exactly with the independently judged run; the reuse record is explicit.
- Six CLI controls refuse wrong model hashes, train/validation overlap, flat,
  nonmonotone or missing curves, and validation-labelled fitting data before
  creating output. Five analyzer mutations refuse incomplete/duplicate cells,
  unconsumed maps and wrong selected scores/bytes. Existing candidate engagement
  and stale-model-cache controls are in the preceding binding record.
- Shared library/example tests: 9 passed; scoped Clippy passed. JXL default
  suite: **2,320 passed, 228 ignored**, selecting the required libjxl 0.12 via
  `DJXL_PATH`. The first test run failed because an older default decoder could
  not load `libIlmImf-2_5.so.25`; that environment failure is retained.
- JXL workspace/all-target Clippy and the RD example's exact feature combination
  (`__expert,zensim-loop,ssim2-loop,parallel`) pass. Checking only `zensim-loop`
  misses that gated example. Formatting/diff checks, zensim CI-exact Clippy and
  the 605-script lint pass. No production encoder change in this increment;
  the preceding binding's RD regression and default-byte checks remain separate.

Reproduce from the JXL repository after building that exact example feature set:

```sh
B=target/release/examples/zensim_diffmap_rd
# SOURCES contains the pinned train_source_manifest.json/source_manifest.json.
"$B" --bake "$BAKE" --native-fit "$SOURCES" --out-dir "$TRAIN"
"$B" --bake "$BAKE" --native-eval "$SOURCES" \
  --native-calibration "$TRAIN/calibration.json" --out-dir "$VALIDATION"
python3 ../zensim/scripts/v_next/rd_probe_analyze_2026-07-18.py \
  --target-loop "$VALIDATION"
```

Use fresh directories and the same driver bytes for fit/eval. The archived
command records name all concrete paths. Calibration is bound to model, driver,
configuration, corpus and split. AVIF/JPEG/WebP native qualification, broader
coverage/geometry, candidate selection and full SDR/HDR/color/alpha qualification
remain required work for the complete product.
