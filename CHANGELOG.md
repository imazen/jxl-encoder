# Changelog

## [0.2.0] — 2026-03-06

### Quality — At parity with cjxl e7

Size parity (grand average -0.0% vs cjxl e7) across 41 CID22 images × 9 distances.
Butteraugli and SSIM2 metrics within ±1% at most distances.

**Key quality fixes:**
- Compute adaptive quant on pre-gaborish XYB (was post-gaborish, inflating masking)
- Match libjxl ties-to-even rounding (`round_ties_even()` vs `round()`)
- Fix merge sub-cost entropy_mul adjustments (kFavor2X2 discount was missed)
- Fix EPF sharpness integer division to match libjxl exactly
- Fix global_scale formula to use effort-matched fixed q values
- Remove AC strategy distance gates (match libjxl effort-level gating)
- Correct AdjustQuantBlockAC effort gating (effort >= 5, not <= 5)

### New features

- **Zensim quantization loop** (`--zensim-iters N`, `--features zensim-loop`):
  Alternative to butteraugli loop using zensim psychovisual metric.
  ~2x faster than butteraugli loop with comparable quality improvement.
- **SSIM2 quantization loop** (`--ssim2-iters N`, `--features ssim2-loop`):
  Alternative loop using SSIMULACRA2 for per-block quality refinement.
- **HDR/non-sRGB color encoding** (`with_color_encoding()`):
  Signal custom transfer function, primaries, and white point.
- **LfFrame** (`--lf-frame`): Separate DC frame for progressive display.
- **Progressive encoding** (`--progressive`, `--qprogressive`):
  2-pass or 3-pass coefficient splitting for incremental decode.
- **Splines** (API: `LossyConfig::with_splines()`):
  Gaussian-blurred parametric curves for thin features.
- **Patches/dictionary** (default-on, `--no-patches` to disable):
  Auto-detect repeated patterns in screenshots/UI. 33-47% savings on screenshots.
- **Lossy delta palette** (`--lossy-palette`):
  Near-lossless with error diffusion for palette-like images.
- **Grayscale lossy** encoding.
- **16-bit and float pixel input** (Rgb16, Rgba16, Gray16, GrayAlpha16,
  RgbLinearF32, RgbaLinearF32, GrayLinearF32, GrayAlphaLinearF32).

### Performance

- **2.5x overall speedup** on 1024×1024 photos at effort 7 (release build).
- SIMD (AVX2 + NEON + WASM SIMD128) for 14 hot kernels: DCT/IDCT, XYB,
  quantize, dequant, entropy, gaborish, mask1x1, pixel_loss, block_l2, EPF.
- Parallel transform+quantize, AC tokenization, CfL, AC strategy search.
- 86x faster tree learning (incremental entropy, count_increase buckets, nlog2n LUT).
- Token struct compacted from 12 to 8 bytes. Two-phase re-tokenization eliminates
  AC token storage.
- Fast powf (libjxl fast_math port) replaces libm powf throughout.
- Pre-sized allocations, buffer pooling, early memory release.

### Lossless

- **Beats cjxl e7** on CLIC photos. Average: -0.7% (7 of 8 images smaller).
- Tree learning with 14 predictors, 50% pixel sampling, 256 quantization buckets.
- RCT selection (best of 7 candidates) for multi-group images.
- Per-histogram HybridUint config optimization.
- LZ77: RLE (e7), greedy (e8), optimal Viterbi DP (e9+).
- Squeeze transform (Haar wavelet) opt-in via `.with_squeeze(true)`.
- Lossless patches: 37% savings on screenshots, zero overhead on photos.
- Palette transform with auto-detect.

### Entropy coding

- ANS: 28-config HybridUint optimization, RLE logcount encoding,
  flat distribution cost baseline, precise population cost for shift selection.
- LZ77 for ICC profiles.
- Non-simple context map encoding for >8 histograms.
- Max histogram clusters increased from 64 to 128.
- Content-adaptive block context map (QF-based splitting).

### Bug fixes

- U64 varint encoding for values >= 273.
- Container box headers for >4GB payloads.
- F16 Inf/NaN/overflow rejection.
- ZeroIfNegative clamp in XYB conversion.
- Intensity target scaling in XYB.
- Custom coefficient orders limited to buckets ≤ 6.
- LZ77 distance cost table extended to 139 entries.
- Palette transform bit widths corrected (u2S selectors).
- ANS alias table log_alpha_size consistency across distributions.
- Predictor formulas 10-13 corrected (AverageWest/NorthWest, AverageAll, etc.).

### Dependencies

- archmage 0.9, magetypes 0.9
- butteraugli 0.8
- zensim 0.1 (optional, for zensim-loop feature)
- fast-ssim2 0.7 (optional, for ssim2-loop feature)

## [0.1.3] — 2026-02-14

Initial public release on crates.io. VarDCT lossy + Modular lossless encoder
with ANS entropy coding, 19/27 AC strategies, adaptive quantization,
chroma-from-luma, gaborish, noise synthesis, and butteraugli quantization loop.
