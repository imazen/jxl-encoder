# JPEG transcode: JPEG XL render vs the JPEG's own decode (2026-10-07)

**Question.** Lossless JPEG transcodes (`LosslessConfig::encode_jpeg_transcode`)
of zenjpeg JPEGs (three quant tables, jpegli AQ, auto_optimize, progressive,
4:4:4 and 4:2:0, 1024-pixel photos, q30/q90) render up to 45 levels away
from the libjpeg-turbo / zenjpeg decode of the same JPEG. Is the transcoder
wrong?

**Answer: no.** The difference is the JPEG XL decode semantics for a
recompressed-JPEG frame, and libjxl v0.12 produces the same pixels.

- On all 8 inputs at e7 and e9 (16 cells, source `108a4875`), djxl renders
  our transcode **byte-identically** to its render of `cjxl
  --lossless_jpeg=1` at the same effort, and `djxl` reconstructs the
  original JPEG byte-exactly from ours.
- A model of the spec decode reproduces every render within ±1
  (`model=jxl-cfl` vs `jxl-render`, `jxl-nocfl` vs `jxl-render-nocfl`).
  Its terms, all fixed by the decoder rather than chosen by an encoder:
  1. **No per-sample clamp.** libjpeg-style decoders round and clamp each
     Y/Cb/Cr sample to [0,255] before YCbCr->RGB; a JPEG XL decoder converts
     unclamped floats. Exact dequant + float IDCT without the clamp
     (`jpeg-float`) is already 9-39 away from the JPEG decode; with the
     clamp (`jpeg-clamp`) it is within 3 on every input.
  2. **Default AC quant bias.** `AdjustQuantBias` with `kDefaultQuantBias`
     applies to every AC coefficient. `opsin_inverse_matrix` (which carries
     `quant_biases`) is only signalled when `xyb_encoded`, and a JPEG
     recompression frame cannot be XYB (libjxl `image_metadata.cc:84`,
     `dec_group.cc:234`), so no encoder can change it.
  3. **Floating-point chroma-from-luma (4:4:4 only).** The decoder adds
     `ytox/84 * Y_dequant` in float, while JBRD reconstruction (and the
     original JPEG) uses the integer-rounded prediction; per-coefficient error
     up to `0.5 * Q_chroma`, coherent with luma edges. This is the only term
     an encoder controls: libjxl's `--jpeg_reconstruction_cfl=0` lowers the
     4:4:4 max from 39 to 8 (1407 q90) and 38 to 17 (8370 q90), but terms
     1-2 remain (8370 q30 4:4:4 stays at 36).

Max |d| vs the zenjpeg decode (RGB8), from the TSV:

| input | spec render (= ours = libjxl) | libjxl, CfL off | exact dequant, no clamp | exact dequant + clamp |
|---|---|---|---|---|
| 1407 q30 4:4:4 | 40 | 25 | 25 | 3 |
| 1407 q90 4:4:4 | 39 | 8 | 9 | 3 |
| 8370 q30 4:4:4 | 45 | 36 | 37 | 3 |
| 8370 q90 4:4:4 | 38 | 17 | 16 | 3 |
| 1407 q30 4:2:0 | 27 | 27 | 30 | 3 |
| 1407 q90 4:2:0 | 10 | 10 | 11 | 3 |
| 8370 q30 4:2:0 | 37 | 37 | 39 | 3 |
| 8370 q90 4:2:0 | 16 | 16 | 15 | 3 |

JPEG-identical pixels therefore come only from JBRD reconstruction followed
by a JPEG decode (or a decoder mode that renders recompressed frames with
JPEG semantics); a pixel-domain JPEG XL decode is an approximation by spec.

**4:2:0 reconstruction panic in jxl-oxide.** jxl-oxide (imazen fork
`08395e61`) panics reconstructing every 4:2:0 progressive input
(`jxl-grid` "coordinate out of range: (64, 0) not in 64x64"), for libjxl's
transcodes as well as ours, while djxl reconstructs both byte-exactly.
`jxl-jbr/src/reconstruct.rs` computes `max_hsample`/`max_vsample` for a
single-component scan from that component alone, so a non-interleaved chroma
scan walks the full luma block width over the half-width chroma grid.

## Provenance

- Source: jxl-encoder `108a4875` (main); `src/jpeg/` unchanged since
  `2b13266`, which produced the original report's files.
- Reference: cjxl/djxl v0.12.0 `a7a9c78` (`--lossless_jpeg=1 -e 7|9`, and
  `--jpeg_reconstruction_cfl=0 -e 7` for the CfL-off column).
- JPEG decode reference: zenjpeg (bit-identical to libjpeg-turbo via
  ImageMagick 7.1.2-18 on these inputs).
- Inputs (sha256): 1407 q30 4:2:0 `c94d0414`, q30 4:4:4 `ffa94b72`,
  q90 4:2:0 `f11f37cb`, q90 4:4:4 `6b48ca7f`; 8370 q30 4:2:0 `89a43ff6`,
  q30 4:4:4 `d3aa6927`, q90 4:2:0 `16197c2b`, q90 4:4:4 `a8dde8fb`.
  1407 is 1024x1024, 8370 is 768x1024; all progressive, three DQT tables.
- Model: `scripts/jpeg_transcode_render_model.py` on dumps from
  `examples/jpeg_coeff_dump.rs`; one-input reproduction:
  `just jpeg-render-attribution <label> <jpeg> [effort]`.
- Host: r5900xt (Ryzen 9 5900XT), 2026-10-07.

Regression coverage: `tests/it/jpeg_cfl_reference.rs`
`jpeg_cfl_reference_zenjpeg_styles_match_libjxl_render_and_reconstruct`.
