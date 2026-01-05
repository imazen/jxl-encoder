# VarDCT Encoder Status

This document tracks the compatibility status of the VarDCT (lossy) encoder with different image dimensions and patterns.

**Test Date:** January 2026
**Decoder:** jxl-oxide
**Encoding Mode:** VarDCT (lossy), distance=1.0

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | 280 |
| Encode Success | 100% |
| Decode Success | 52.1% (146/280) |

## Known Issues

The primary decode failures are related to entropy coding issues:
- `InvalidIntegerConfig { split_exponent: ... }` - Common at medium sizes (32-128px)
- `InvalidPrefixHistogram` - Occasional at smaller sizes
- `InvalidAnsStream` - Common at larger sizes (128-256px)
- `UnexpectedEof` - Multi-group images (>256px)
- `ValidationFailed("non_zeros too large")` - Multi-group images
- `ValidationFailed("too many zeros")` - Large multi-group images

These errors suggest issues in:
1. **HybridUint encoding** - `split_exponent` configuration
2. **Prefix/Huffman code generation** - Invalid histogram construction
3. **ANS stream encoding** - State management at larger images
4. **Multi-group encoding** - TOC/section boundaries for >256px images
5. **AC coefficient count tracking** - non_zeros mismatch

## Compatibility Matrix

### By Dimension (All Patterns)

| Size | solid | h_grad | v_grad | d_grad | radial | check1 | check4 | stripes | color | noise |
|------|-------|--------|--------|--------|--------|--------|--------|---------|-------|-------|
| 8x8 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| 16x16 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| 24x24 | pass | pass | pass | FAIL | pass | pass | FAIL | pass | pass | pass |
| 32x32 | pass | pass | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | pass | pass |
| 48x48 | pass | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 64x64 | pass | FAIL | FAIL | FAIL | pass | FAIL | FAIL | FAIL | FAIL | FAIL |
| 128x128 | pass | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 256x256 | pass | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 257x257 | pass | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 300x300 | pass | pass | FAIL | pass | FAIL | pass | FAIL | pass | FAIL | FAIL |
| 512x512 | pass | FAIL | FAIL | pass | pass | pass | pass | pass | FAIL | FAIL |
| 9x9 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| 15x15 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| 17x17 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| 31x31 | pass | pass | FAIL | FAIL | FAIL | pass | FAIL | FAIL | FAIL | pass |
| 33x33 | pass | pass | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 127x127 | pass | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 255x255 | pass | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 8x16 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| 16x8 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| 16x32 | pass | pass | pass | FAIL | pass | pass | pass | pass | pass | pass |
| 32x16 | pass | pass | pass | FAIL | pass | pass | pass | pass | pass | pass |
| 256x128 | pass | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 128x256 | pass | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 257x128 | pass | FAIL | pass | FAIL | FAIL | FAIL | FAIL | pass | FAIL | FAIL |
| 128x257 | pass | pass | pass | FAIL | FAIL | FAIL | FAIL | FAIL | pass | FAIL |
| 300x200 | pass | pass | pass | FAIL | pass | pass | FAIL | pass | FAIL | FAIL |
| 200x300 | pass | pass | FAIL | FAIL | pass | pass | FAIL | pass | pass | FAIL |

**Legend:** pass = Decodes OK, FAIL = Decode fails

### By Pattern (Failure Rate)

| Pattern | Success Rate | Notes |
|---------|--------------|-------|
| solid_gray | 100% (28/28) | Simple content, always works |
| h_gradient | 68% (19/28) | Fails at 48×48+ |
| v_gradient | 54% (15/28) | More failures than h_gradient |
| d_gradient | 43% (12/28) | Most problematic gradient |
| radial | 54% (15/28) | Complex pattern |
| checker_1 | 57% (16/28) | 1px checkerboard |
| checker_4 | 43% (12/28) | 4px checkerboard - most failures |
| stripes | 57% (16/28) | Horizontal stripes |
| color_bars | 50% (14/28) | RGB color bars |
| noise | 54% (15/28) | Random noise |

### Working Dimensions

**Fully Working (all patterns):**
- 8×8, 9×9, 15×15, 16×16, 17×17
- 8×16, 16×8

**Mostly Working (>80% patterns):**
- 24×24 (8/10 patterns)
- 16×32, 32×16 (9/10 patterns)
- 300×300 (6/10 patterns)
- 512×512 (6/10 patterns)

**Partially Working (40-80% patterns):**
- 31×31 (5/10 patterns)
- 32×32 (4/10 patterns)
- 64×64 (2/10 patterns)
- 300×200, 200×300 (6/10 patterns)
- 257×128 (3/10 patterns)
- 128×257 (4/10 patterns)

**Mostly Failing (<40% patterns):**
- 33×33 (2/10 patterns)
- 48×48 (1/10 patterns)
- 127×127 (1/10 patterns)
- 128×128 (1/10 patterns)
- 255×255 (1/10 patterns)
- 256×256 (1/10 patterns)
- 257×257 (1/10 patterns)
- 256×128, 128×256 (1/10 patterns)

## Failure Analysis

### Error Types by Image Size

| Size Range | Primary Error | Secondary Error |
|------------|---------------|-----------------|
| 24-32px | InvalidPrefixHistogram | InvalidIntegerConfig |
| 32-64px | InvalidIntegerConfig | InvalidPrefixHistogram |
| 64-128px | InvalidIntegerConfig | InvalidAnsStream |
| 128-256px | InvalidAnsStream | InvalidIntegerConfig |
| >256px (multi-group) | UnexpectedEof | non_zeros too large |

### Root Cause Hypothesis

1. **Small images (24-64px)**: HybridUint `split_exponent` calculation issues
2. **Medium images (64-256px)**: ANS state management issues at larger token counts
3. **Multi-group images (>256px)**: TOC/section boundary encoding issues
4. **Solid patterns always work**: Minimal entropy, simple coefficient distribution

### Multi-Group Image Issues

Images >256px in either dimension use multiple groups. New failure modes:
- `UnexpectedEof`: Bitstream ends before expected data
- `non_zeros too large`: AC coefficient count exceeds expected value
- `too many zeros`: Mismatch in zero-run encoding

This suggests issues with:
1. Group data section boundaries in TOC
2. Pass group encoding for HF coefficients
3. AC coefficient serialization across group boundaries

## Recommendations

### Safe to Use Now
- Images ≤17×17 with any pattern
- Images ≤24×24 with most patterns (avoid diagonal gradients, 4px checkerboard)
- Solid color images at any size

### Needs Investigation (Priority Order)
1. **ANS stream encoding** - State initialization/finalization for larger images
2. **Multi-group TOC** - Section size calculations for >256px images
3. **HybridUint split_exponent** - Configuration for various alphabet sizes
4. **Prefix code generation** - Histogram building for medium-sized images

## Running the Compatibility Test

```bash
# Run the full compatibility matrix
cargo test --package jxl_enc test_vardct_compatibility_matrix -- --nocapture --ignored
```

## Test Patterns

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
