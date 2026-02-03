# Context Handoff: Color/Brightness Bug Discovery

**Date**: Feb 3, 2026
**Issue**: All encoder output is too dark compared to reference cjxl

## Summary

While investigating DCT32x32 single-group quality issues, we discovered a **more fundamental bug**: all encoder output (not just DCT32x32) is noticeably darker than the original and reference cjxl output.

## Evidence

Visual comparison at `/mnt/v/output/jxl-encoder-rs/dct32x32-bug/`:

| Image | Brightness |
|-------|------------|
| `original_256.png` | Correct (bright, vibrant) |
| `ref_256_decoded.png` (cjxl d=3.0) | Correct (matches original) |
| `single_normal_decoded.png` (ours d=3.0) | **Too dark** |
| `single_dct32_decoded.png` (ours d=3.0) | **Too dark** |
| `ours_d1_decoded.png` (ours d=1.0) | **Still too dark** |

Key finding: Even at d=1.0 (high quality), our output is darker. This proves it's NOT a quantization issue.

## File Size Comparison (d=3.0)

```
cjxl reference 256x256: 22,960 bytes
Our normal 256x256:     16,944 bytes (26% smaller)
Our DCT32 256x256:       8,367 bytes (64% smaller!)
```

Our files are much smaller, which may indicate over-quantization, but the darkness persists even at d=1.0.

## Root Cause Hypothesis

Color space conversion bug, likely in one of:
1. `sRGB → linear` conversion on input (in `TinyEncoder::encode`)
2. `linear → XYB` conversion
3. `XYB → linear` in the decoder's interpretation of our coefficients
4. DC/quantization scaling that affects overall brightness

## Files to Investigate

1. **`jxl_enc/src/tiny/encoder.rs`** - `convert_to_xyb_padded()` function
2. **`jxl_enc/src/tiny/xyb.rs`** (if exists) - XYB conversion
3. **Input handling** - Check if we're treating sRGB input as linear or vice versa

## Quick Verification Test

```bash
# Encode same image with cjxl and our encoder, decode both with djxl, compare
OUT=/mnt/v/output/jxl-encoder-rs/dct32x32-bug
cd ~/work/jxl-encoder-rs

# Our encode
./target/release/cjxl-rs "$OUT/original_256.png" "$OUT/test_ours.jxl" -d 1.0

# Reference encode
~/work/jxl-efforts/libjxl/build/tools/cjxl "$OUT/original_256.png" "$OUT/test_ref.jxl" -d 1.0

# Decode both with same decoder
~/work/jxl-efforts/libjxl/build/tools/djxl "$OUT/test_ours.jxl" "$OUT/test_ours_dec.png"
~/work/jxl-efforts/libjxl/build/tools/djxl "$OUT/test_ref.jxl" "$OUT/test_ref_dec.png"

# Compare visually
display "$OUT/test_ours_dec.png" "$OUT/test_ref_dec.png"
```

## Relationship to DCT32x32 Bug

The DCT32x32 single-group SSIM2=-48 issue is likely:
1. **Partly caused by** this overall darkness bug
2. **Plus** a separate spatial averaging issue where AC coefficients aren't being decoded correctly

The darkness bug affects ALL encodes. The DCT32x32 spatial issue is on top of that.

## What Was Fixed This Session

1. **DCT32x32 quant weights** - Were inverted (dequant instead of quant). Fixed in `quant.rs`.
2. **Rectangular transform storage** - DCT16x8/DCT8x16 had wrong coefficient→block mapping. Fixed in `encoder.rs`.

These fixes are committed but the color bug predates them.

## Next Steps

1. Compare XYB coefficients between our encoder and cjxl for the same input
2. Check if our sRGB→linear conversion matches the PNG's gamma
3. Verify DC quantization scaling
4. Check if the bug is in encoding or if djxl is misinterpreting our bitstream

## Commits This Session

```
bb054bf docs: update known bugs and add rectangular transform fix to resolved
d900650 test: add DCT32x32 investigation tests and production vs reference comparison
f8e4085 fix: correct rectangular transform coefficient storage mapping
b0278a7 fix: invert DCT32x32 quant weights (dequant→quant)
```
