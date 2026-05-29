# zenjxl lossy-JPEG closed loop — LANDED (path-patched), publish-migration checklist

The lossy JPEG → JXL recompression closed loop **shipped in `zenjxl`** (commit
`ac6826f9`, 2026-05-28) as the opt-in `jpeg-lossy` feature + `zenjxl::jpeg_lossy`
module. The live code lives in `zenjxl` (not here) — this dir is now just the
publish-migration checklist.

## What landed in zenjxl

- `jpeg-lossy` feature =
  `["encode", "decode", "jxl-encoder/jpeg-reencoding", "dep:zenjpeg"]`.
- Paths + router: `JpegRecompressMethod {Coarsen, Reencode, Auto}` +
  `recompress_jpeg_lossy(jpeg, method, target, higher_is_better, scorer, effort)`.
  Coarsen = PreserveJxl coeff-domain; Reencode = VarDCT pixel re-encode (reuses
  the lossless-transcode pixels as input); Auto = run both, keep the smaller.
- Relative + inferred targets: `QualityTarget {Relative, Inferred}` +
  `recompress_jpeg_lossy_target`. Inferred = achievability clamp (unreachable
  absolute target → lossless floor). Preliminary `predict_inferred_floor`
  (`zenjpeg::detect` → N=5 floor table per `InferredMetric`) +
  `QualityTarget::inferred_preliminary`.
- Convenience: `recompress_jpeg_lossy_relative`, `recompress_jpeg_coarsen`.
  Metric-agnostic scorer callback over decoded RGB8.
- `tests/jpeg_lossy.rs` (8/8) + `tests/fixtures/tiny.jpg`.

## Dev-coupling (current state)

zenjxl deps `jxl-encoder = "0.3.2"` with committed `[patch.crates-io]`
redirecting `jxl-encoder` + `zenjpeg` to the local siblings (0.3.2 / 0.8.7 are
unpublished). This matches jxl-encoder's own committed `zenjpeg` path-patch.
**zenjxl now requires the sibling checkouts to build** (it no longer builds
standalone against published deps) — the accepted tradeoff for landing without a
release.

## Publish-migration checklist (when the chain is published)

1. Publish `zenjpeg 0.8.7` (crates.io has 0.8.3 with a `magetypes` API
   mismatch) — CI green + GitHub release + go-ahead.
2. Publish `jxl-encoder 0.3.2` (lossy-JPEG API) — CI green + GitHub release +
   go-ahead.
3. In `zenjxl/Cargo.toml`: remove the `[patch.crates-io]` block (zenjxl then
   builds standalone against the published 0.3.2). Keep the `jpeg-lossy` feature
   and `jxl-encoder = "0.3.2"` dep as-is.
4. `cargo test -p zenjxl --features jpeg-lossy --test jpeg_lossy` (expect 3/3).

See `../JPEG_LOSSY_RECOMPRESSION.md` for the RD strategy + the `QualityTarget` /
`JpegRecompressMethod` API naming this loop grows into.
