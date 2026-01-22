# VarDCT Encoder Status

This document tracks the compatibility status of the VarDCT (lossy) encoder.

**Test Date:** January 22, 2026
**Decoder:** jxl-oxide
**Encoding Mode:** VarDCT (lossy), distance=1.0

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | 280 |
| Encode Success | 100% |
| Decode Success | 100% |

**All 280 test cases pass!** The VarDCT encoder is fully working for all tested dimensions and patterns.

## Tested Dimensions

### Single-group (≤256px)
- 8×8, 9×9, 15×15, 16×16, 17×17
- 24×24, 31×31, 32×32, 33×33
- 48×48, 64×64, 127×127, 128×128
- 255×255, 256×256
- Asymmetric: 8×16, 16×8, 16×32, 32×16, 256×128, 128×256

### Multi-group (>256px)
- 257×257, 300×300, 512×512
- Asymmetric: 257×128, 128×257, 300×200, 200×300

## Test Patterns

All patterns decode successfully at all dimensions:

| Pattern | Description |
|---------|-------------|
| solid_gray | Uniform gray (128,128,128) |
| h_gradient | Horizontal gradient (black to white) |
| v_gradient | Vertical gradient (black to white) |
| d_gradient | Diagonal gradient |
| radial | Radial gradient from center |
| checker_1 | 1px checkerboard |
| checker_4 | 4px block checkerboard |
| stripes | 2px horizontal stripes |
| color_bars | 7-bar color test pattern |
| noise | Deterministic pseudo-random noise |

## Running the Compatibility Test

```bash
# Run the full compatibility matrix (~5 minutes)
cargo test --package jxl_enc test_vardct_compatibility_matrix -- --nocapture --ignored
```

## Features

The VarDCT encoder includes:
- DCT8/DCT16/DCT32 transform support
- Variance-based AC strategy selection
- Chroma-from-Luma (CfL) correlation
- Adaptive quantization
- Histogram clustering (up to 8 clusters)
- Multi-group encoding for images >256px
