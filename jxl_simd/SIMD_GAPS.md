# SIMD Gaps: libjxl Highway → jxl-encoder-rs

## Priority Gaps (ranked by wall-clock impact)

| Priority | Gap | Why | Est. Speedup |
|----------|-----|-----|--------------|
| ~~**P0**~~ | ~~DCT/IDCT 32×32~~ | ✅ DONE — AVX2 in `dct32.rs`/`idct32.rs`, wired into encoder | 3-5× on DCT32 blocks |
| ~~**P1**~~ | ~~DCT/IDCT 32×16, 16×32~~ | ✅ DONE — same modules, all 6 functions | 3-5× on DCT32x16 blocks |
| ~~**P2**~~ | ~~sRGB→linear conversion~~ | ✅ DONE — 256-entry const LUT for u8 (eliminates 24M powf calls) | 4-8× on input conversion |
| **P3** | Quantize AC (DCT16+) | Only DCT8 quantize is SIMD. DCT16/32/64 blocks quantize via scalar loops over 256-4096 coefficients. | 2-4× on large block quant |
| ~~**P4**~~ | ~~DCT/IDCT 64×64, 64×32, 32×64~~ | ✅ DONE — AVX2 in `dct64.rs`/`idct64.rs`, wired into encoder | 3-5× on DCT64 blocks |
| **P5** | Large block transpose (32×32, 64×64) | Supporting operation for large DCTs. Scalar element-by-element copy. | 2-3× (part of DCT chain) |
| **P6** | ANS token cost (Shannon) | `log2()` per histogram bin during strategy evaluation. Moderate frequency. | 2-4× on entropy estimation |
| **P7** | FastPow2f / FastPowf general | Used by adaptive quant modulations, but currently embedded. General SIMD fast math library would benefit multiple callers. | Enables other optimizations |
| **P8** | AFV 4×4 basis matrix | 256 scalar FMAs. Low frequency (corner blocks only). | 4× but rare |
| **P9** | Noise SAD estimation | Nested 4-level loop, but runs once per encode. | Negligible overall |

## Full Comparison Table

| # | Operation | libjxl File | Our File | Status | Impact |
|---|-----------|------------|----------|--------|--------|
| 1 | DCT 8×8 | `dct-inl.h` | `jxl_simd/src/dct8.rs` | ✅ | — |
| 2 | IDCT 8×8 | `dct-inl.h` | `jxl_simd/src/dct8.rs` | ✅ | — |
| 3 | DCT 16×8, 8×16, 16×16 | `dct-inl.h` | `jxl_simd/src/dct16.rs` | ✅ | — |
| 4 | IDCT 16×8, 8×16, 16×16 | `dct-inl.h` | `jxl_simd/src/idct16.rs` | ✅ | — |
| 5 | DCT 32×32 | `dct-inl.h` | `jxl_simd/src/dct32.rs` | ✅ | ~~P0~~ |
| 6 | DCT 32×16, 16×32 | `dct-inl.h` | `jxl_simd/src/dct32.rs` | ✅ | ~~P1~~ |
| 7 | DCT 64×64 | `dct-inl.h` | `jxl_simd/src/dct64.rs` | ✅ | ~~P4~~ |
| 8 | DCT 64×32, 32×64 | `dct-inl.h` | `jxl_simd/src/dct64.rs` | ✅ | ~~P4~~ |
| 9 | IDCT 32×32 | `dct-inl.h` | `jxl_simd/src/idct32.rs` | ✅ | ~~P0~~ |
| 10 | IDCT 32×16, 16×32 | `dct-inl.h` | `jxl_simd/src/idct32.rs` | ✅ | ~~P1~~ |
| 11 | IDCT 64×64, 64×32, 32×64 | `dct-inl.h` | `jxl_simd/src/idct64.rs` | ✅ | ~~P4~~ |
| 12 | 8×8 transpose | `transpose-inl.h` | `jxl_simd/src/transpose.rs` | ✅ | — |
| 13 | NxM transpose (N>8) | `transpose-inl.h` | `vardct/dct/forward_large.rs` | ❌ | P5 |
| 14 | RGB→XYB | `enc_xyb.cc` | `jxl_simd/src/xyb.rs` | ✅ | — |
| 15 | XYB→RGB | `dec_xyb-inl.h` | `jxl_simd/src/xyb.rs` | ✅ | — |
| 16 | sRGB→linear | `cms/transfer_functions-inl.h` | `api.rs` (const LUT) | ✅ | ~~P2~~ |
| 17 | AdjustQuantBias | `quantizer-inl.h` | `jxl_simd/src/dequant.rs` | ✅ | — |
| 18 | Quantize AC (DCT8) | encoder | `jxl_simd/src/quantize.rs` | ✅ | — |
| 19 | Quantize AC (DCT16+) | encoder | `vardct/quantize.rs` | ❌ | **P3** |
| 20 | Gaborish 5×5 | `convolve-inl.h` | `jxl_simd/src/gaborish5x5.rs` | ✅ | — |
| 21 | Gaborish 3×3 | `convolve-inl.h` | `jxl_simd/src/gab.rs` | ✅ | — |
| 22 | EPF step1/step2 | EPF code | `jxl_simd/src/epf.rs` | ✅ | — |
| 23 | Mask1x1 blur | `convolve-inl.h` | `jxl_simd/src/mask1x1.rs` | ✅ | — |
| 24 | Pre-erosion + masking | encoder | `jxl_simd/src/adaptive_quant.rs` | ✅ | — |
| 25 | Per-block modulations | encoder | `jxl_simd/src/adaptive_quant.rs` | ✅ | — |
| 26 | Entropy coeff est. | encoder | `jxl_simd/src/entropy.rs` | ✅ | — |
| 27 | Pixel-domain loss | encoder | `jxl_simd/src/pixel_loss.rs` | ✅ | — |
| 28 | Block L2 error | encoder | `jxl_simd/src/block_l2.rs` | ✅ | — |
| 29 | CfL LS + Newton | encoder | `jxl_simd/src/cfl.rs` | ✅ | — |
| 30 | ANS token cost | `enc_ans_simd.cc` | `entropy_coding/histogram.rs` | ❌ | P6 |
| 31 | Modular cost est. | `enc_modular_simd.cc` | `modular/tree.rs` | ❌ | LOW |
| 32 | FastLog2f | `fast_math-inl.h` | `mask1x1.rs` embedded | ⚠️ | P7 |
| 33 | FastPow2f/FastPowf | `fast_math-inl.h` | None | ❌ | P7 |
| 34 | AFV 4×4 basis | `enc_transforms-inl.h` | `vardct/afv.rs` | ❌ | P8 |
| 35 | Wiener denoising | encoder | `jxl_simd/src/noise.rs` | ✅ | — |
