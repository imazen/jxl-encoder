# zenjxl lossy-JPEG closed loop — validated, staged, gated on publish

This directory holds the **validated** productization of the lossy JPEG → JXL
recompression closed loop, ready to drop into the `zenjxl` crate. It lives here
(in jxl-encoder's repo) because it cannot be committed to `zenjxl` yet — see
"Publish gate" below.

## Why it lives in zenjxl (not jxl-encoder, not the app layer)

`zenjxl` already deps **both** `jxl-encoder` (encode, incl. the PreserveJxl
coefficient-domain coarsener) **and** `zenjxl-decoder` (decode), plus zencodec
traits. So it can run the full **encode → decode → score → bisect** loop
in-process. `jxl-encoder` stays a lean building block (no decoder/metric dep);
the app layer doesn't need to own the loop. The loop is metric-agnostic: the
caller supplies a scorer callback over decoded RGB8, so it drives a
zensim-A / cvvdp / butteraugli (or any) target.

## Validation (2026-05-28)

Built and tested against local path-patched `jxl-encoder 0.3.2` + `zenjpeg 0.8.7`:

```
running 3 tests
test coarsen_is_monotone_and_decodes ... ok
test unreachable_target_returns_lossless_floor ... ok
test relative_loop_looser_target_is_smaller ... ok
test result: ok. 3 passed; 0 failed
```

The loop coarsens (PreserveJxl), decodes (zenjxl-decoder), scores (MSE callback),
bisects to the target, and falls back to the lossless floor when the target is
unreachable — all in-process.

## Publish gate (why it isn't committed to zenjxl)

`zenjxl` deps the **published** `jxl-encoder 0.3.1` from crates.io. The loop
calls `jxl_encoder::jpeg::encode_jpeg_recompress_auto_codestream`, which is new
in the unpublished `0.3.2`. The publish chain is:

1. `zenjpeg 0.8.7` (jxl-encoder path-patches it today; crates.io has 0.8.3 with a
   `magetypes` API mismatch) → publish.
2. `jxl-encoder 0.3.2` (with the lossy-JPEG API: `encode_jpeg_recompress_auto_codestream`,
   `encode_jpeg_recompress_planar_codestream`, `coarsen_policy`, …) → publish.
3. `zenjxl`: bump dep to `jxl-encoder = "0.3.2"`, add the `jpeg-lossy` feature,
   drop in these files. → publish.

Each publish needs the standard release gate (CI green on all platforms, README
review, GitHub release, user sign-off) per CLAUDE.md. This is a release-
engineering decision, not a code task.

## How to land (after the publish chain)

1. `zenjxl/Cargo.toml` [features]: add
   `jpeg-lossy = ["encode", "decode", "jxl-encoder/jpeg-reencoding"]`
   and bump `jxl-encoder` dep to `"0.3.2"`.
2. `zenjxl/src/lib.rs`: add
   `#[cfg(feature = "jpeg-lossy")] pub mod jpeg_lossy;`
3. Copy `jpeg_lossy.rs` → `zenjxl/src/jpeg_lossy.rs`.
4. Copy `jpeg_lossy_test.rs` → `zenjxl/tests/jpeg_lossy.rs`,
   `tiny.jpg` → `zenjxl/tests/fixtures/tiny.jpg`.
5. `cargo test -p zenjxl --features jpeg-lossy --test jpeg_lossy` (expect 3/3).

## Alternative (dev-coupled, no publish)

`zenjxl` could commit `[patch.crates-io]` entries pointing `jxl-encoder` +
`zenjpeg` at the local siblings — the same pattern jxl-encoder itself uses for
its unpublished `zenjpeg` dep. This makes `jpeg-lossy` build in the sibling-
checkout dev/CI environment WITHOUT publishing, but couples every `zenjxl` build
to the sibling checkouts (it can no longer build standalone against published
deps). Tradeoff is the user's call.

See `../JPEG_LOSSY_RECOMPRESSION.md` for the full RD strategy and `QualityTarget`
/ `JpegRecompressMethod` API naming this loop will grow into.
