# Context Handoff — Feb 2, 2026

## Mission: Achieve libjxl Parity and Unify Encoders

The goal is to close the quality gap with full libjxl (cjxl) and merge the "tiny" encoder
into the main encoder path. The tiny encoder started as a port of libjxl-tiny but has
evolved beyond it. The original VarDCT encoder (`jxl_enc/src/vardct/`) failed completely
and should be abandoned.

## Current RD Position vs cjxl e7

Single CLIC 2025 image (1024x1024), tested Feb 2, 2026:

| Distance | Our Size | Our SSIM2 | cjxl Size | cjxl SSIM2 | Gap |
|----------|----------|-----------|-----------|------------|-----|
| d=1.0    | ~514KB   | ~80.9     | ~520KB    | ~80.7      | **+0.2** |
| d=2.0    | 143KB    | 76.1      | 162KB     | 81.2       | **-5.1** |
| d=4.0    | 88KB     | 63.4      | 100KB     | 70.3       | **-6.9** |

**We match/beat cjxl at d≤1.0 but lose significantly at higher distances.**

## Investigation Summary (Feb 2, 2026)

### What Was Tested

1. **K_AC_QUANT constant change** (0.8294 → 0.765 to match libjxl)
   - Result: WORSE quality (74.96 vs 76.11 SSIM2)
   - Conclusion: Constants are NOT the issue

2. **AC_QUANT for global_scale** (0.8 → 0.39 to match libjxl Hare mode)
   - Result: AC_QUANT cancels out with inv_scale in quantization formula
   - Conclusion: Changing AC_QUANT alone has no effect

3. **adjust_quant_field mean/max blending** (ported from libjxl)
   - Result: No significant improvement
   - Conclusion: Correctly ported but not the root cause

### What We Know

- **libjxl uses FIXED global_scale** in the main encoding path (`q = 0.39/distance`),
  NOT content-adaptive median/MAD calculation
- **The K_AC_QUANT/AC_QUANT ratio** affects quantization, but matching libjxl's
  ratio (0.765/0.39 ≈ 1.96) produces WORSE results than our current ratio
- **The quality gap grows with distance** — suggests the issue is in how we handle
  coarser quantization, not the base algorithm

### Remaining Hypotheses

1. **Adaptive quant masking algorithm differences**
   - Pre-erosion computation
   - Fuzzy erosion weights
   - Per-block modulations (ComputeMask, HfModulation, etc.)

2. **Coefficient thresholding in QuantizeBlockAC**
   - Our `quantize_coeff_ac` uses quadrant-based thresholds
   - libjxl may use different thresholding

3. **Missing features that help at high distances**
   - DCT4x8/DCT8x4/DCT4x4 (small blocks for edges)
   - Error diffusion (spreads quantization error)

## Encoder Architecture

### Production Encoder: `jxl_enc/src/tiny/`

This is the working encoder. Key files:

- `encoder.rs` — Main encode pipeline (~2500 lines)
- `adaptive_quant.rs` — Per-block quantization field computation
- `ac_strategy.rs` — DCT transform selection (8x8, 16x8, 8x16, 16x16)
- `frame.rs` — Frame header, DistanceParams, global_scale
- `quant.rs` — Quantization weights and coefficient quantization
- `entropy_code.rs` — Huffman and ANS entropy coding
- `dc_coding.rs` — DC coefficient encoding
- `ac_group.rs` — AC coefficient tokenization

### Dead Code: `jxl_enc/src/vardct/`

**DO NOT USE.** This was an earlier VarDCT implementation that never worked correctly.
It produces hard-to-diagnose errors and should be removed entirely during the merge.

### Reference Implementations

- **libjxl (C++)**: `~/work/jxl-efforts/libjxl` — PRIMARY reference
- **libjxl-tiny (C++)**: `~/work/libjxl-tiny` — Historical, DO NOT use for quality reference
- **jxl-rs (Rust decoder)**: `~/work/jxl-rs` — For roundtrip testing

## Key Constants Comparison

| Constant | Our Value | libjxl-tiny | libjxl (Hare) | Notes |
|----------|-----------|-------------|---------------|-------|
| K_AC_QUANT (adaptive quant) | 0.8294 | 0.8294 | 0.765 | Changing hurts quality |
| AC_QUANT (global_scale) | 0.8 | 0.8 | 0.39 | Cancels out |
| QUANT_FIELD_TARGET | 5.0 | 5.0 | 5.0 | Same |
| match_gamma_offset | 0.019 | 0.019 | 0.019 | Same |
| kXMul (pre-erosion) | 23.427 | 23.427 | 23.427 | Same |

## What Works in Tiny Encoder

- [x] XYB color conversion
- [x] Adaptive quantization (per-block masking)
- [x] Chroma-from-luma
- [x] AC strategy selection (DCT8/DCT16x8/DCT8x16/DCT16x16)
- [x] DC coding with gradient predictor
- [x] AC coding with channel interleaving
- [x] Multi-group encoding (>256x256)
- [x] Dynamic Huffman codes
- [x] ANS entropy coding (`--ans` flag)
- [x] Custom coefficient ordering
- [x] Noise synthesis (`--noise` flag)
- [x] Gaborish inverse (default on)
- [x] Modular encoder (lossless path)
- [x] RGBA support

## What's Missing vs Full libjxl

Priority ranked by expected impact on high-distance quality gap:

1. **DCT4x8/DCT8x4/DCT4x4** — Small transforms for edges/detail
2. **Error diffusion** — Spreads quantization error to neighbors
3. **DCT32x32 fix** — Blocked by DC extraction bug (see CLAUDE.md)
4. **Better AC strategy cost model** — Our entropy estimation is simplified
5. **Progressive encoding** — No RD impact, UX feature

## Test Commands

```bash
# Build
cargo build --release -p jxl_enc_cli

# Run RD regression test
just rd-regression

# Encode single image
./target/release/cjxl-rs -d 2.0 input.png output.jxl

# Compare with cjxl
cjxl -d 2.0 -e 7 input.png cjxl_output.jxl
djxl output.jxl decoded.png
ssimulacra2 input.png decoded.png
```

## Files to Read First

1. `CLAUDE.md` — Full project docs, known bugs, resolved bugs
2. `INVESTIGATION.md` — Detailed investigation notes
3. `jxl_enc/src/tiny/encoder.rs` — Main encode function
4. `jxl_enc/src/tiny/adaptive_quant.rs` — Adaptive quant pipeline

## Suggested Next Steps

1. **Deep dive into adaptive quant algorithm**
   - Add debug output to compare intermediate values with libjxl
   - Check pre-erosion, fuzzy erosion, per-block modulations
   - Build libjxl with debug output to compare

2. **Check coefficient thresholding**
   - Compare `quantize_coeff_ac` with libjxl's QuantizeBlockAC
   - The quadrant-based thresholds may differ

3. **Implement DCT4x8/DCT8x4**
   - Forward transforms exist in `jxl_enc_transforms`
   - Need: quant weights, strategy selection, LLF extraction

4. **Remove dead vardct code**
   - Delete `jxl_enc/src/vardct/` directory
   - Update module structure to make tiny encoder the main path

## Working Tree State

Clean. All tests pass. Commit `1d1c092`.

## Key Insight from Investigation

The quality gap is NOT from simple constant differences. Our K_AC_QUANT=0.8294 with
AC_QUANT=0.8 actually produces BETTER results than matching libjxl's 0.765/0.39 ratio.

The issue is somewhere in the adaptive quant algorithm implementation or coefficient
thresholding, not the top-level parameters. A detailed comparison of intermediate
values between our code and libjxl is needed.
