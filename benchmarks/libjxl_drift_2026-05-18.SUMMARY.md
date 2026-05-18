# libjxl HEAD refresh + drift bench — 2026-05-18 (W19-2)

## Pinned-baseline drift check

| | OLD baseline | NEW HEAD |
|---|---|---|
| commit | `d2c7032` | `4279d48` |
| date | 2026-02-22 | 2026-05-12 |
| version banner | `cjxl v0.12.0 d2c7032` | `cjxl v0.12.0 4279d48` |
| commits in delta | — | 81 |
| files touched | — | 274 |
| insertions / deletions | — | 1,529 / 49,928 |

The very large deletion count is the `tools/jpegli*` and `tools/jni/*` tree being
removed — that code never participated in the JPEG XL encoder hot path.

`lib/jxl/quant_weights.cc`, `lib/jxl/enc_quant_weights.cc`,
`lib/jxl/enc_adaptive_quantization.cc`, `lib/jxl/enc_ac_strategy.cc`,
`lib/jxl/enc_chroma_from_luma.cc` are all unchanged. The encoder-touching diff
is overwhelmingly safety hardening (overflow / NaN / null / buffer-size guards),
streaming-encode plumbing (`acc28c0`, `032d39a`, `1389871`, `b3510d1`,
`6553831`), and one CI / GitHub-Actions noise floor.

## Results: 39/39 cells byte-identical

| Bench | Cells | Drift |
|---|---|---|
| RD parity (5 CLIC × d∈{0.5,1.0,2.0,5.0}, e7) | 20 | **0** |
| Lossless photos (5 CLIC, e7 -d 0) | 5 | **0** |
| Lossless screenshots (5 gb82-sc, e7 -d 0) | 5 | **0** |
| HDR W4 (PQ/HLG/BT709 × d∈{1,2,5}, e7) | 9 | **0** |
| **TOTAL** | **39** | **0** |

The lossy and lossless bytes from `cjxl_old` and `cjxl_new` are bit-identical on
every cell, and the Rust butteraugli scores against `djxl`-decoded output match
to all six printed decimal places. The 81-commit delta does not change libjxl
encoder output at default `effort=7` on any axis we currently bench.

## Drift cells (NEW cjxl beats OLD by >2%)

**None.** There are no cells where NEW cjxl improved or regressed vs OLD by any
measurable amount on this corpus.

## Implication for W19-2 (Diff-review investigation priority)

The drift-detection bench produced an empty diff-review set. We do not need to
prioritize any specific libjxl commit for cherry-picking into our cost model.
The next time this bench detects drift > 2% on a real cell, that cell becomes a
priority signal — until then, our current libjxl-r0 baseline (`d2c7032`) is
indistinguishable from HEAD on the things we measure.

The cjxl binary has been moved forward to `4279d48` (saved at
`/tmp/cjxl_new_4279d48`); the OLD binary is preserved at `/tmp/cjxl_old_d2c7032`
for future side-by-sides. The libjxl checkout itself is now on `main` at HEAD;
the user's prior `docs-encoder-internals` branch is intact and the
disabled-debug-prints diff is stashed (`git stash list`).

## Bench artifacts

```
benchmarks/libjxl_drift_rd_2026-05-18.{tsv,meta}             # 60 rows  (5 img × 4 d × 3 enc)
benchmarks/libjxl_drift_lossless_2026-05-18.{tsv,meta}       # 15 rows  (5 img × 3 enc)
benchmarks/libjxl_drift_screenshots_2026-05-18.{tsv,meta}    # 10 rows  (5 img × 2 enc)
benchmarks/hdr_drift_2026-05-18/{hdr_old,hdr_new,hdr_drift}.tsv  +  .meta
scripts/libjxl_drift_{bench,lossless,screenshots,hdr}.sh     # reproducer
```
